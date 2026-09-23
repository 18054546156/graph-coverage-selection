"""Measure selection-set functionals against real ResNet training endpoints.

This driver reuses the existing functional-screen library, but replaces the
linear-probe endpoint with one deterministic ResNet-18 training run per
selection. All training weights are equal so the result isolates which
properties of the selected set predict downstream test balanced accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import functional_screen as F
import run_linear_probe as R
import train_weighted as T


def load_raw_dataset(graphcov_root: Path, dataset: str, size: int, medmnist_root: Path):
    sys.modules.setdefault("faiss", None)
    sys.path.insert(0, str(graphcov_root))
    from graphcov.run.data import get_transform
    from medmnist import INFO
    import medmnist

    info = INFO[dataset]
    data_class = getattr(medmnist, info["python_class"])
    transform = get_transform(info["n_channels"], size)
    train_ds = data_class(
        split="train", transform=transform, download=False,
        root=str(medmnist_root), size=size
    )
    test_ds = data_class(
        split="test", transform=transform, download=False,
        root=str(medmnist_root), size=size
    )
    return train_ds, test_ds, info, len(info["label"])


def run_dataset(args, dataset: str, device: torch.device) -> int:
    emb_path = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
    x = np.asarray(np.load(emb_path, allow_pickle=False)["embeddings"], dtype=np.float32)
    y_all = R.read_train_labels(emb_path, args.data_root / f"{dataset}_{args.size}.npz")
    di = F.DATASETS.index(dataset)
    align = R.assert_labels_aligned(x, y_all, dataset, args.seed + 500009 * di, 0.30)
    z_all = R.normalize(x)
    del x

    dyn_all = None
    if args.dynamics_root is not None:
        dyn_all = F.load_dynamics(args.dynamics_root, dataset, len(y_all))

    train_ds, test_ds, info, n_classes = load_raw_dataset(
        args.graphcov_root, dataset, args.size, args.medmnist_root
    )
    datasets = (train_ds, test_ds, info, n_classes)

    out_path = args.output / f"{dataset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        print(f"[{dataset}] n={len(y_all)} align_ba={align:.4f} device={device}", flush=True)
        for block in range(args.blocks):
            bseed = args.seed + 100003 * di + 1009 * block
            pool_ids, _audit_ids = R.split_block(
                np.arange(len(y_all)), y_all, 1.0 - F.POOL_FRACTION, bseed
            )
            zp = torch.as_tensor(z_all[pool_ids], device=device)
            yp = y_all[pool_ids]
            dyn = ({k: v[pool_ids] for k, v in dyn_all.items()}
                   if dyn_all is not None else None)
            pre = F.pool_precompute(zp, yp)
            library = F.build_library(
                zp, yp, pre, args.budget, bseed,
                args.n_random, args.n_perturb_seeds, dyn
            )
            print(f"  block={block} pool={len(pool_ids)} library={len(library)}", flush=True)

            for index, (name, family, m, sel) in enumerate(library):
                features, _weights = F.functionals(zp, yp, sel, pre)
                if dyn is not None:
                    features.update(F.dyn_functionals(dyn, sel, yp))

                # Use raw train indices and equal weights. The train endpoint is
                # deliberately independent of the probe audit set and Voronoi weights.
                selected_ids = pool_ids[sel].astype(np.int64)
                train_seed = args.train_seed_offset + 100000 * di + 1000 * block
                cell = {
                    "dataset": dataset,
                    "block": block,
                    "budget": args.budget,
                    "name": name,
                    "family": family,
                    "m": m,
                    "train_seed": train_seed,
                    "selected_indices": selected_ids.tolist(),
                    "weights": np.ones(len(selected_ids), dtype=np.float64).tolist(),
                }
                started = time.time()
                result = T.train_one_cell(cell, datasets, args, device)
                row = {
                    **{k: v for k, v in result.items()
                       if k not in ("selected_indices", "weights")},
                    **{(k if k.startswith("F_") else f"F_{k}"): v
                       for k, v in features.items()},
                    "ba_real": result["balanced_accuracy"],
                    "endpoint": "resnet18_official_test_equal_weight_final_epoch",
                    "screen_index": index,
                    "endpoint_seconds": time.time() - started,
                    "alignment_probe_ba": align,
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                rows += 1
                if rows % args.progress_every == 0:
                    print(f"  {dataset} block={block} {rows}/{len(library)} "
                          f"BA={row['ba_real']:.4f} elapsed={time.time()-started:.0f}s",
                          flush=True)
    del z_all
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dynamics-root", type=Path, default=None)
    p.add_argument("--graphcov-root", type=Path, required=True)
    p.add_argument("--medmnist-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--n-random", type=int, default=60)
    p.add_argument("--n-perturb-seeds", type=int, default=2)
    p.add_argument("--train-seed-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=1)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    for dataset in args.datasets:
        total += run_dataset(args, dataset, device)
    print(json.dumps({"rows": total, "output": str(args.output)}))


if __name__ == "__main__":
    main()
