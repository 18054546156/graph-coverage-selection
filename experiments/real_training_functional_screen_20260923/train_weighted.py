"""Real ResNet-18 training with per-sample loss weights, on the official test split.

Deliberately a standalone runner rather than a patch to graphcov: this project has
already broken another user's in-flight jobs by editing shared code, and
`graphcov.run.evaluation.train_and_evaluate` hardcodes `nn.CrossEntropyLoss()`
with a 2-tuple loader. Only model/data/selection are borrowed from graphcov.

Three things this fixes relative to the shared path:

1. **Determinism is actually complete.** `graphcov.run.evaluation.set_seed`
   sets cudnn flags but never calls `torch.use_deterministic_algorithms`, and
   its in-process `PYTHONHASHSEED` assignment is a no-op. The DataLoader there
   has no `generator`/`worker_init_fn`. All four are handled here, and
   `CUBLAS_WORKSPACE_CONFIG` is set before torch touches cuBLAS.
2. **No test-based model selection.** The shared path can reload a "best"
   checkpoint chosen on an eval loader that falls back to the *test* loader when
   no validation set is passed. Here the reported model is the final epoch, full
   stop, and the official test split is touched exactly once per run.
3. **Per-sample weights.** `CrossEntropyLoss(reduction='none')` combined with a
   weighted mean, so the objective is `sum_i w_i * l_i / sum_i w_i` over each
   batch. Weights are normalised to mean 1 over the whole subset so that the
   effective learning rate matches the equal-weight arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Must precede any cuBLAS initialisation, hence before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class WeightedSubset(Dataset):
    """Subset that also yields a per-sample loss weight."""

    def __init__(self, base: Dataset, weights: np.ndarray):
        if len(base) != len(weights):
            raise ValueError(f"subset length {len(base)} != weights {len(weights)}")
        self.base = base
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return img, label, self.weights[i]


def make_deterministic(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The piece graphcov's set_seed never calls. warn_only=False so a
    # nondeterministic kernel is a hard error rather than silent variance.
    torch.use_deterministic_algorithms(True, warn_only=False)


def evaluate(model, loader, device, n_classes: int) -> dict:
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            labels = labels.to(dtype=torch.long).reshape(-1)
            logits = model(imgs.to(device, non_blocking=True))
            probs_all.append(torch.softmax(logits.float(), dim=1).cpu())
            labels_all.append(labels)
    probs = torch.cat(probs_all).numpy()
    y = torch.cat(labels_all).numpy()
    pred = probs.argmax(axis=1)

    recalls = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan
                        for c in range(n_classes)])
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    one_hot = np.eye(n_classes)[y]
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "worst_class_recall": float(np.nanmin(recalls)),
        "ece15": float(ece),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)))),
        "per_class_recall": [float(v) for v in recalls],
    }


def train_one_cell(cell, datasets, args, device) -> dict:
    from graphcov.run.data import wrap_with_augmentation
    from graphcov.run.embeddings import ResNet18WithFeatures

    train_ds, test_ds, info, n_classes = datasets
    seed = cell["train_seed"]
    make_deterministic(seed)

    idx = np.asarray(cell["selected_indices"], dtype=np.int64)
    w = np.asarray(cell["weights"], dtype=np.float64)
    if len(w) != len(idx):
        raise ValueError("weights and indices misaligned")
    w = w / w.mean()  # mean 1: keeps the effective LR comparable to equal weighting

    subset = Subset(train_ds, idx.tolist())
    if args.augment:
        subset = wrap_with_augmentation(subset, in_channels=info["n_channels"], size=args.size)
    subset = WeightedSubset(subset, w)

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        s = seed + worker_id
        np.random.seed(s % (2 ** 32))
        torch.manual_seed(s)

    train_loader = DataLoader(
        subset, batch_size=min(args.batch_size, len(subset)), shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        generator=generator, worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    model = ResNet18WithFeatures(n_classes, info["n_channels"], pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        for imgs, labels, weights in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True).reshape(-1)
            weights = weights.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            per_sample = criterion(model(imgs), labels)
            # Weighted mean, not weighted sum: a batch whose weights happen to be
            # large must not also get a larger effective step size.
            (per_sample * weights).sum().div(weights.sum()).backward()
            optimizer.step()
        scheduler.step()
    train_seconds = time.time() - t0

    # Official test split, final-epoch model, touched exactly once.
    result = evaluate(model, test_loader, device, n_classes)
    return {
        **{k: v for k, v in cell.items() if k not in ("selected_indices", "weights")},
        **result,
        "train_seconds": train_seconds,
        "n_train": int(len(idx)),
        "weight_mean": float(w.mean()),
        "weight_cv": float(w.std() / w.mean()),
        "selection_sha256": hashlib.sha256(np.sort(idx).tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(np.round(w, 6).tobytes()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=Path, required=True,
                   help="jsonl of cells, each with dataset/arm/weighting/replicate/"
                        "train_seed/selected_indices/weights")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--graphcov-root", type=Path, required=True)
    args = p.parse_args()

    import sys
    # The cluster's FAISS probe can abort at import before graphcov's own
    # ImportError fallback; nothing here needs it.
    sys.modules.setdefault("faiss", None)
    sys.path.insert(0, str(args.graphcov_root))
    from graphcov.run.data import load_dataset

    cells = [json.loads(l) for l in args.cells.open(encoding="utf-8")]
    mine = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard]
    # Group by dataset so each split is loaded once per shard.
    mine.sort(key=lambda c: (c["dataset"], c["arm"], c["replicate"], c["train_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} cells on {device}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loaded, rows = None, []
    with args.output.open("w", encoding="utf-8") as out:
        for n, cell in enumerate(mine, 1):
            if loaded is None or loaded[0] != cell["dataset"]:
                train_ds, info = load_dataset(cell["dataset"], "train", size=args.size)
                test_ds, _ = load_dataset(cell["dataset"], "test", size=args.size)
                loaded = (cell["dataset"], (train_ds, test_ds, info, len(info["label"])))
                print(f"loaded {cell['dataset']}: train={len(train_ds)} test={len(test_ds)} "
                      f"classes={len(info['label'])}", flush=True)
            row = train_one_cell(cell, loaded[1], args, device)
            rows.append(row)
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            print(f"[{n}/{len(mine)}] {cell['dataset']} {cell['arm']} {cell['weighting']} "
                  f"rep={cell['replicate']} seed={cell['train_seed']} "
                  f"BA={row['balanced_accuracy']:.4f} ({row['train_seconds']:.0f}s)", flush=True)
    print(json.dumps({"shard": args.shard, "rows": len(rows)}))


if __name__ == "__main__":
    main()
