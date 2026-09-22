"""Run one fixed-selection PathMNIST training-mode probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphcov.run.data import load_dataset
from graphcov.run.evaluation import evaluate_selection


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return bytes_sha256(np.ascontiguousarray(value).tobytes())


def state_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mode", choices=["iteration", "epoch"], required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    selected = np.asarray(np.load(args.selection, allow_pickle=False), dtype=np.int64)
    train_ds, info = load_dataset(args.dataset, split="train", size=224, verbose=False)
    test_ds, _ = load_dataset(args.dataset, split="test", size=224, verbose=False)
    if not len(selected) or np.any(selected < 0) or np.any(selected >= len(train_ds)):
        raise RuntimeError("selection indices are empty or outside the train split")

    result = evaluate_selection(
        train_dataset=train_ds,
        test_dataset=test_ds,
        selected_indices=selected.tolist(),
        num_classes=len(info["label"]),
        in_channels=info["n_channels"],
        training_paradigm=args.mode,
        epochs=args.epochs,
        iterations=args.iterations,
        test_interval=args.iterations,
        test_every_n_epochs=10,
        batch_size=args.batch_size,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        augment=False,
        size=224,
        seed=args.seed,
        return_history=True,
        return_model=True,
        verbose=False,
        deterministic=True,
        num_workers=args.num_workers,
    )
    final_acc, final_ba, history, best_metrics, per_class, model = result

    checkpoint = args.output / "final.pt"
    torch.save({"state_dict": model.state_dict(), "seed": args.seed}, checkpoint)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(next(model.parameters()).device)).cpu().numpy())
            labels.append(np.asarray(batch_labels).reshape(-1))
    logits_array = np.concatenate(logits)
    labels_array = np.concatenate(labels).astype(np.int64)
    prediction_path = args.output / "predictions_clean.npz"
    np.savez_compressed(prediction_path, y_true=labels_array, logits=logits_array)

    updates = args.iterations if args.mode == "iteration" else args.epochs * int(np.ceil(len(selected) / args.batch_size))
    trace = {
        "dataset": args.dataset,
        "method": args.method,
        "ratio": args.ratio,
        "mode": args.mode,
        "selection_seed": 42,
        "training_seed": args.seed,
        "selected_count": int(len(selected)),
        "selected_indices_sha256": array_sha256(selected),
        "selected_indices_file_sha256": file_sha256(args.selection),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_state_dict_sha256": state_sha256(model.state_dict()),
        "prediction_sha256": file_sha256(prediction_path),
        "final_acc": float(final_acc),
        "final_ba": float(final_ba),
        "best_metrics": best_metrics,
        "per_class": per_class,
        "epochs": args.epochs,
        "iterations": args.iterations,
        "optimizer_updates": int(updates),
        "batch_size": args.batch_size,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "author_commit": git_head(args.repo),
        "deterministic": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    (args.output / "diagnostic_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps({
        "dataset": args.dataset, "method": args.method, "ratio": args.ratio,
        "mode": args.mode, "selection_seed": 42, "training_seed": args.seed,
        "acc": float(final_acc), "ba": float(final_ba), "best": best_metrics,
        "per_class": per_class,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output / "run_complete.json").write_text(json.dumps({
        "status": "complete", "mode": args.mode,
        "selection_sha256": trace["selected_indices_sha256"],
        "checkpoint_sha256": trace["checkpoint_sha256"],
        "prediction_sha256": trace["prediction_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(trace, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
