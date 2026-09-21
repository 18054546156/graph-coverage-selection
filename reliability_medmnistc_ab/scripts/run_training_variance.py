"""Step 2: selection-seed x training-seed reliability experiment.

``plan`` only writes a plan. ``run`` executes exactly one verified cell.
``aggregate`` summarizes completed cells and separates selection, training,
and selection-by-training variance for complete crossed designs.

This script consumes immutable clean-train selection artifacts. It never
reruns a selector or uses test data for selection/checkpoint choice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

DATASETS = ["organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist"]
METHODS = ["random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding", "graph_a2"]
METRICS = ["accuracy", "balanced_accuracy", "worst_class_recall", "nll", "brier", "ece", "aurc", "ba_drop"]


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def ratio_token(ratio: float) -> str:
    return f"{ratio:.3f}".rstrip("0").rstrip(".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_save_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def selection_directory(root: Path, dataset: str, ratio: float, method: str, seed: int) -> Path:
    return root / method / dataset / f"ratio_{ratio:g}" / f"selection_seed_{seed}"


def inspect_selection(root: Path, dataset: str, ratio: float, method: str, seed: int, labels=None) -> dict | None:
    """Load one exact selection artifact and validate metadata and hashes."""
    directory = selection_directory(root, dataset, ratio, method, seed)
    indices_path = directory / "selected_indices.npy"
    if not indices_path.is_file():
        return None
    required = ["selection_config.json", "class_counts.json", "index_order_sha256.txt"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"selection artifact missing {missing}: {directory}")

    config = json.loads((directory / "selection_config.json").read_text(encoding="utf-8"))
    for key, expected in {"dataset": dataset, "method": method, "selection_seed": seed}.items():
        if config.get(key) != expected:
            raise RuntimeError(f"selection {key} mismatch at {directory}: {config.get(key)!r} != {expected!r}")
    if not math.isclose(float(config.get("ratio", float("nan"))), ratio, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(f"selection ratio mismatch at {directory}: {config.get('ratio')} != {ratio}")

    selected_raw = np.load(indices_path, allow_pickle=False)
    if selected_raw.ndim != 1 or not np.issubdtype(selected_raw.dtype, np.integer):
        raise RuntimeError(f"selected_indices.npy must be a one-dimensional integer array: {directory}")
    selected = selected_raw.astype(np.int64, copy=False)
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError(f"duplicate selected indices: {directory}")
    selected_hash = sha256_array(selected)
    if config.get("selected_indices_sha256") != selected_hash:
        raise RuntimeError(f"selection_config hash mismatch: {directory}")
    if (directory / "index_order_sha256.txt").read_text(encoding="utf-8").strip() != selected_hash:
        raise RuntimeError(f"index_order_sha256 mismatch: {directory}")
    if int(config.get("n_selected", -1)) != len(selected):
        raise RuntimeError(f"selection count mismatch: {directory}")

    counts_raw = json.loads((directory / "class_counts.json").read_text(encoding="utf-8"))
    recorded_counts = {int(key): int(value) for key, value in counts_raw.items()}
    budget = int(config.get("budget_per_class", -1))
    if sum(recorded_counts.values()) != len(selected) or budget < 1:
        raise RuntimeError(f"invalid recorded selection counts: {directory}")
    if any(value != budget for value in recorded_counts.values()):
        raise RuntimeError(f"recorded class quota mismatch at {directory}: {recorded_counts}")

    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= len(labels)):
            raise RuntimeError(f"selection outside canonical train range [0,{len(labels)}): {directory}")
        n_classes = int(labels.max()) + 1 if len(labels) else 0
        actual = np.bincount(labels[selected], minlength=n_classes)
        recorded = np.asarray([recorded_counts.get(index, 0) for index in range(n_classes)])
        if set(recorded_counts) != set(range(n_classes)):
            raise RuntimeError(
                f"class_counts.json must contain exactly classes 0..{n_classes - 1}: {directory}"
            )
        if not np.array_equal(actual, recorded):
            raise RuntimeError(f"selected indices disagree with class counts at {directory}: {actual.tolist()} != {recorded.tolist()}")
        if not np.all(actual == budget):
            raise RuntimeError(f"actual class quota mismatch at {directory}: expected {budget}, got {actual.tolist()}")
        source_hash = config.get("source_train_indices_sha256")
        canonical_hash = sha256_array(np.arange(len(labels), dtype=np.int64))
        if not source_hash:
            raise RuntimeError(f"selection config lacks source_train_indices_sha256: {directory}")
        if source_hash != canonical_hash:
            raise RuntimeError(f"selection is not based on the full canonical train order: {directory}")

    return {"path": indices_path, "indices": selected, "sha256": selected_hash, "config": config}


def run_paths(args, row: dict) -> dict[str, Path]:
    ratio = ratio_token(float(row["ratio"]))
    key = f"{row['dataset']}__{row['method']}__ratio_{ratio}__sel{row['selection_seed']}__train{row['training_seed']}"
    suffix = Path(row["method"]) / row["dataset"] / f"ratio_{ratio}" / f"selection_seed_{row['selection_seed']}" / f"train_seed_{row['training_seed']}"
    return {
        "key": Path(key), "checkpoint": args.checkpoint_root / suffix,
        "metrics": args.metrics_root / suffix, "predictions": args.prediction_root / suffix,
    }


def build_plan(args) -> list[dict]:
    rows = []
    factors = itertools.product(args.datasets, args.methods, args.ratios, args.selection_seeds, args.training_seeds)
    for dataset, method, ratio, selection_seed, training_seed in factors:
        error = ""
        try:
            artifact = inspect_selection(args.selection_root, dataset, ratio, method, selection_seed)
            status = "ready" if artifact else "missing_selection"
        except Exception as exc:
            artifact, status, error = None, "invalid_selection", str(exc)
        row = {
            "dataset": dataset, "method": method, "ratio": ratio,
            "selection_seed": selection_seed, "training_seed": training_seed,
            "selection_path": str(artifact["path"]) if artifact else "",
            "selection_sha256": artifact["sha256"] if artifact else "",
            "status": status, "error": error,
        }
        paths = run_paths(args, row)
        row.update(
            run_key=str(paths["key"]), checkpoint_dir=str(paths["checkpoint"]),
            metrics_dir=str(paths["metrics"]), prediction_dir=str(paths["predictions"]),
        )
        rows.append(row)
    return rows


def write_plan(args, rows: list[dict]) -> None:
    payload = {"kind": "selection_seed_x_training_seed_plan", "created_unix": time.time(), "config": vars(args), "rows": rows}
    atomic_write_json(args.plan, payload)
    csv_path = args.plan.with_suffix(".csv")
    temporary = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    statuses = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    print(json.dumps({"total": len(rows), "statuses": statuses, "json": str(args.plan), "csv": str(csv_path)}, indent=2))


def make_dataset(name: str, split: str, root: Path, size: int):
    import medmnist
    from medmnist import INFO
    from torchvision import transforms

    info = INFO[name]
    dataset_class = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * info["n_channels"], [0.5] * info["n_channels"])])
    return dataset_class(split=split, size=size, root=str(root), transform=transform, download=False), info


def metric_row(y_true: np.ndarray, logits: np.ndarray, n_classes: int, allow_missing: bool) -> tuple[dict, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64)
    if len(y_true) == 0:
        raise RuntimeError("cannot evaluate an empty dataset")
    if logits.ndim != 2 or logits.shape != (len(y_true), n_classes) or not np.isfinite(logits).all():
        raise RuntimeError(f"invalid logits: labels={len(y_true)}, logits={logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = probs.argmax(1)
    recalls, missing = [], []
    for class_id in range(n_classes):
        mask = y_true == class_id
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == class_id)))
        else:
            recalls.append(float("nan"))
            missing.append(class_id)
    if missing and not allow_missing:
        raise RuntimeError(f"test split is missing required classes: {missing}")
    observed = np.asarray([value for value in recalls if np.isfinite(value)])
    true_probs = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    confidence = probs.max(1)
    correct = (pred == y_true).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    accepted_n = max(1, int(round(0.8 * len(errors))))
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((confidence >= lo) if index == 0 else (confidence > lo)) & (confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": float(correct.mean()), "balanced_accuracy": float(observed.mean()),
        "worst_class_recall": float(observed.min()), "per_class_recall": json.dumps(recalls),
        "n_classes_observed": int(len(observed)), "nll": float(-np.log(true_probs).mean()),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))), "ece": float(ece),
        "aurc": float(risks.mean()), "risk_at_80": float(risks[accepted_n - 1]),
        "high_confidence_errors": int(np.sum((confidence >= 0.8) & (correct == 0))), "n": int(len(y_true)),
    }, probs.astype(np.float32)


def predict(model, dataset, batch_size: int, workers: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    labels, logits = [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits.append(model(images.to(device, non_blocking=True)).cpu().numpy())
            labels.append(np.asarray(batch_labels).reshape(-1))
    return np.concatenate(labels).astype(np.int64), np.concatenate(logits).astype(np.float32)


def validate_prediction(path: Path, clean_ids: np.ndarray, severities: np.ndarray, labels: np.ndarray, n_classes: int) -> None:
    """Validate a resumable prediction artifact before trusting/skipping it."""
    with np.load(path, allow_pickle=False) as package:
        required = {"clean_id", "severity", "labels", "logits", "probabilities"}
        if not required.issubset(package.files):
            raise RuntimeError(f"prediction artifact missing keys at {path}: {package.files}")
        stored_ids = np.asarray(package["clean_id"], dtype=np.int64)
        stored_severity = np.asarray(package["severity"], dtype=np.int64)
        stored_labels = np.asarray(package["labels"], dtype=np.int64)
        logits = np.asarray(package["logits"])
        probabilities = np.asarray(package["probabilities"])
    if not np.array_equal(stored_ids, np.asarray(clean_ids, dtype=np.int64)):
        raise RuntimeError(f"clean IDs do not align in {path}")
    if not np.array_equal(stored_severity, np.asarray(severities, dtype=np.int64)):
        raise RuntimeError(f"severity IDs do not align in {path}")
    if not np.array_equal(stored_labels, np.asarray(labels, dtype=np.int64)):
        raise RuntimeError(f"labels do not align in {path}")
    expected_shape = (len(stored_labels), n_classes)
    if logits.shape != expected_shape or probabilities.shape != expected_shape:
        raise RuntimeError(f"prediction shape mismatch in {path}: {logits.shape}, {probabilities.shape}, expected {expected_shape}")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError(f"non-finite predictions in {path}")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
        raise RuntimeError(f"probabilities do not sum to one in {path}")


def new_model(n_classes: int, in_channels: int, device):
    import torch
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def train_model(args, seed: int, dataset, selected: np.ndarray, n_classes: int, in_channels: int, device, history_path: Path):
    import torch
    from torch.utils.data import DataLoader, Subset

    seed_everything(seed, args.deterministic)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, selected.tolist()), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    model = new_model(n_classes, in_channels, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = correct = seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.reshape(-1).to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            correct += int((output.argmax(1) == labels).sum())
            seen += count
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / seen, "train_accuracy": correct / seen, "lr": scheduler.get_last_lr()[0]})
        atomic_write_json(history_path, history)
        if (epoch + 1) % max(1, args.log_every) == 0:
            print(f"epoch={epoch + 1}/{args.epochs} loss={history[-1]['train_loss']:.6f}", flush=True)
    return model, history


def load_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                rows[(str(row["corruption"]), int(row["severity"]))] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def save_rows(path: Path, rows: dict[tuple[str, int], dict]) -> None:
    ordered = [rows[key] for key in sorted(rows, key=lambda key: (key != ("clean", 0), key))]
    atomic_write_text(path, "".join(json.dumps(row, default=json_default) + "\n" for row in ordered))


def save_csv(path: Path, rows: dict[tuple[str, int], dict]) -> None:
    records = [rows[key] for key in sorted(rows, key=lambda key: (key != ("clean", 0), key))]
    fields = list(dict.fromkeys(key for record in records for key in record))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def atomic_dataframe_to_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_checkpoint(path: Path, n_classes: int, in_channels: int, device, config_hash: str):
    import torch

    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("config_hash") != config_hash:
        raise RuntimeError(f"checkpoint configuration mismatch: {path}")
    model = new_model(n_classes, in_channels, device)
    model.load_state_dict(checkpoint["model"])
    return model


def run_one(args, row: dict) -> None:
    import torch

    train_dataset, info = make_dataset(row["dataset"], "train", args.data_root, args.image_size)
    test_dataset, _ = make_dataset(row["dataset"], "test", args.data_root, args.image_size)
    train_labels = np.asarray(train_dataset.labels).reshape(-1).astype(np.int64)
    test_labels = np.asarray(test_dataset.labels).reshape(-1).astype(np.int64)
    n_classes = len(info["label"])
    artifact = inspect_selection(args.selection_root, row["dataset"], float(row["ratio"]), row["method"], int(row["selection_seed"]), train_labels)
    if artifact is None or artifact["sha256"] != row["selection_sha256"]:
        raise RuntimeError("selection missing or changed between planning and execution")
    selected = artifact["indices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("formal run requires CUDA; --allow-cpu is smoke-only")

    paths = run_paths(args, row)
    for key in ("checkpoint", "metrics", "predictions"):
        paths[key].mkdir(parents=True, exist_ok=True)
    try:
        from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB
        from medmnistc.dataset import CorruptedMedMNIST
    except ImportError as exc:
        raise RuntimeError("pinned medmnistc-api must be on PYTHONPATH") from exc
    corruptions = list(CORRUPTIONS_DS[row["dataset"]])

    stable_config = {
        "schema_version": 2, "dataset": row["dataset"], "method": row["method"], "ratio": float(row["ratio"]),
        "selection_seed": int(row["selection_seed"]), "training_seed": int(row["training_seed"]),
        "selection_path": str(artifact["path"]), "selection_sha256": artifact["sha256"], "selection_config": artifact["config"],
        "selected_n": len(selected), "train_labels_sha256": sha256_array(train_labels), "test_labels_sha256": sha256_array(test_labels),
        "image_size": args.image_size, "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "optimizer": {"name": "SGD", "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay},
        "scheduler": "CosineAnnealingLR_per_epoch", "deterministic": args.deterministic, "checkpoint_rule": "final_epoch",
        "corruptions": corruptions, "severities": [1, 2, 3, 4, 5], "script_sha256": sha256_file(Path(__file__)),
    }
    config_hash = hashlib.sha256(json.dumps(stable_config, sort_keys=True, default=json_default).encode()).hexdigest()
    stable_config["config_hash"] = config_hash
    config_path = paths["metrics"] / "run_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")).get("config_hash") != config_hash:
            raise RuntimeError(f"existing output has another configuration: {paths['metrics']}")
    else:
        atomic_write_json(config_path, stable_config)

    checkpoint_path = paths["checkpoint"] / "final.pt"
    history_path = paths["metrics"] / "training_history.json"
    if checkpoint_path.is_file():
        if not history_path.is_file():
            raise RuntimeError(f"checkpoint exists without training history: {checkpoint_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != args.epochs or int(history[-1].get("epoch", -1)) != args.epochs:
            raise RuntimeError(f"training history is incomplete for {checkpoint_path}")
        model = load_checkpoint(checkpoint_path, n_classes, info["n_channels"], device, config_hash)
        print(f"resume evaluation from {checkpoint_path}", flush=True)
    else:
        model, _ = train_model(
            args, int(row["training_seed"]), train_dataset, selected,
            n_classes, info["n_channels"], device,
            history_path,
        )
        atomic_torch_save(checkpoint_path, {"model": model.state_dict(), "epoch": args.epochs, "config_hash": config_hash, "checkpoint_rule": "final_epoch"})

    rows = load_rows(paths["metrics"] / "metrics.jsonl")
    meta = {
        "dataset": row["dataset"], "method": row["method"], "ratio": float(row["ratio"]),
        "selection_seed": int(row["selection_seed"]), "training_seed": int(row["training_seed"]),
        "selection_sha256": artifact["sha256"], "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    clean_prediction = paths["predictions"] / "clean.npz"
    clean_ids = np.arange(len(test_dataset), dtype=np.int64)
    clean_severities = np.zeros(len(test_dataset), dtype=np.int64)
    if clean_prediction.is_file():
        validate_prediction(clean_prediction, clean_ids, clean_severities, test_labels, n_classes)
    if ("clean", 0) not in rows or not clean_prediction.is_file():
        y_clean, logits_clean = predict(model, test_dataset, args.batch_size, args.workers, device)
        if not np.array_equal(y_clean, test_labels):
            raise RuntimeError("clean test label order mismatch")
        clean_metrics, clean_probs = metric_row(y_clean, logits_clean, n_classes, args.allow_missing_classes)
        atomic_save_npz(clean_prediction, clean_id=clean_ids, severity=clean_severities, labels=y_clean, logits=logits_clean, probabilities=clean_probs)
        rows[("clean", 0)] = dict(meta, split="clean", corruption="clean", severity=0, ba_drop=0.0, **clean_metrics)
        save_rows(paths["metrics"] / "metrics.jsonl", rows)

    clean_ba = float(rows[("clean", 0)]["balanced_accuracy"])
    for corruption in corruptions:
        prediction_path = paths["predictions"] / f"{corruption}.npz"
        required = {(corruption, severity) for severity in range(1, 6)}
        expected_labels = np.tile(test_labels, 5)
        clean_ids = np.tile(np.arange(len(test_dataset), dtype=np.int64), 5)
        severity_ids = np.repeat(np.arange(1, 6, dtype=np.int64), len(test_dataset))
        if prediction_path.is_file():
            validate_prediction(prediction_path, clean_ids, severity_ids, expected_labels, n_classes)
        if required.issubset(rows) and prediction_path.is_file():
            continue
        dataset = CorruptedMedMNIST(row["dataset"], corruption, root=str(args.corrupted_root), as_rgb=DATASET_RGB[row["dataset"]], norm_mean=[0.5] * info["n_channels"], norm_std=[0.5] * info["n_channels"], mmap_mode="r")
        if len(dataset) != 5 * len(test_dataset):
            raise RuntimeError(f"{corruption}: wrong corruption length")
        y_corrupt, logits_corrupt = predict(model, dataset, args.batch_size, args.workers, device)
        if not np.array_equal(y_corrupt, expected_labels):
            raise RuntimeError(f"{corruption}: label/clean-ID alignment failure")
        _, probabilities = metric_row(y_corrupt, logits_corrupt, n_classes, args.allow_missing_classes)
        atomic_save_npz(
            prediction_path, clean_id=clean_ids, severity=severity_ids, labels=y_corrupt,
            logits=logits_corrupt, probabilities=probabilities,
        )
        for severity in range(1, 6):
            start, end = (severity - 1) * len(test_dataset), severity * len(test_dataset)
            metrics, _ = metric_row(y_corrupt[start:end], logits_corrupt[start:end], n_classes, args.allow_missing_classes)
            rows[(corruption, severity)] = dict(meta, split="corrupted", corruption=corruption, severity=severity, ba_drop=clean_ba - metrics["balanced_accuracy"], **metrics)
        save_rows(paths["metrics"] / "metrics.jsonl", rows)

    required = {("clean", 0)} | {(name, severity) for name in corruptions for severity in range(1, 6)}
    missing_metrics = sorted(required - set(rows))
    prediction_paths = {name: paths["predictions"] / f"{name}.npz" for name in ["clean", *corruptions]}
    missing_predictions = [name for name, path in prediction_paths.items() if not path.is_file()]
    if missing_metrics or missing_predictions:
        raise RuntimeError(f"incomplete run: metrics={missing_metrics}, predictions={missing_predictions}")
    save_csv(paths["metrics"] / "metrics.csv", rows)
    atomic_write_json(paths["metrics"] / "run_complete.json", {
        "status": "complete", "completed_unix": time.time(), "config_hash": config_hash,
        "conditions": len(rows), "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_sha256": sha256_file(paths["metrics"] / "metrics.csv"),
        "prediction_sha256": {name: sha256_file(path) for name, path in prediction_paths.items()},
    })
    print(f"completed {paths['key']}", flush=True)


def add_corruption_mean(frame):
    import pandas as pd

    corrupted = frame[frame["split"] == "corrupted"]
    if corrupted.empty:
        return frame
    keys = ["dataset", "method", "ratio", "selection_seed", "training_seed", "selection_sha256"]
    metrics = [column for column in METRICS if column in corrupted]
    overall = corrupted.groupby(keys, dropna=False)[metrics].mean().reset_index()
    overall["split"], overall["corruption"], overall["severity"] = "corruption_mean", "all", 0
    return pd.concat([frame, overall], ignore_index=True, sort=False)


def compute_variance_components(frame):
    import pandas as pd

    keys = ["dataset", "method", "ratio", "split", "corruption", "severity"]
    output = []
    for values, group in frame.groupby(keys, dropna=False):
        base = dict(zip(keys, values))
        for metric in [column for column in METRICS if column in group]:
            cells = group.pivot_table(index="selection_seed", columns="training_seed", values=metric, aggfunc="first")
            n_selection, n_training = cells.shape
            effective = group[["selection_seed", "selection_sha256"]].drop_duplicates()["selection_sha256"].nunique()
            row = dict(base, metric=metric, selection_levels=n_selection, effective_selection_sets=effective, training_levels=n_training)
            if n_selection < 2 or n_training < 2 or cells.isna().any().any():
                row.update(status="insufficient_or_incomplete", variance_selection=np.nan, variance_training=np.nan, variance_selection_x_training=np.nan)
            else:
                matrix = cells.to_numpy(dtype=float)
                grand, row_mean, col_mean = matrix.mean(), matrix.mean(1), matrix.mean(0)
                residual = matrix - row_mean[:, None] - col_mean[None, :] + grand
                ms_selection = n_training * np.sum((row_mean - grand) ** 2) / (n_selection - 1)
                ms_training = n_selection * np.sum((col_mean - grand) ** 2) / (n_training - 1)
                ms_interaction = np.sum(residual ** 2) / ((n_selection - 1) * (n_training - 1))
                row.update(
                    status="complete" if effective > 1 else "duplicate_selection_sets",
                    variance_selection=float(max((ms_selection - ms_interaction) / n_training, 0)),
                    variance_training=float(max((ms_training - ms_interaction) / n_selection, 0)),
                    variance_selection_x_training=float(ms_interaction),
                )
            output.append(row)
    return pd.DataFrame(output)


def validate_completed_run(args, row: dict):
    """Return verified metrics records for one planned cell."""
    import pandas as pd

    paths = run_paths(args, row)
    metrics_path = paths["metrics"] / "metrics.csv"
    marker_path = paths["metrics"] / "run_complete.json"
    config_path = paths["metrics"] / "run_config.json"
    checkpoint_path = paths["checkpoint"] / "final.pt"
    required_files = [metrics_path, marker_path, config_path, checkpoint_path]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing completed-run files: {missing}")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("config_hash") != config.get("config_hash"):
        raise RuntimeError("run_complete.json and run_config.json disagree")
    config_payload = dict(config)
    recorded_config_hash = config_payload.pop("config_hash", None)
    actual_config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, default=json_default).encode()
    ).hexdigest()
    if recorded_config_hash != actual_config_hash:
        raise RuntimeError("run_config.json hash is invalid")

    for key in ("dataset", "method", "selection_seed", "training_seed"):
        if config.get(key) != row[key]:
            raise RuntimeError(f"run config {key} mismatch: {config.get(key)!r} != {row[key]!r}")
    if not math.isclose(float(config.get("ratio", float("nan"))), float(row["ratio"]), rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError("run config ratio mismatch")
    if config.get("selection_sha256") != row["selection_sha256"]:
        raise RuntimeError("run config selection SHA256 mismatch")
    if marker.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("checkpoint SHA256 mismatch")
    if marker.get("metrics_sha256") != sha256_file(metrics_path):
        raise RuntimeError("metrics SHA256 mismatch")

    corruptions = list(config.get("corruptions", []))
    prediction_paths = {name: paths["predictions"] / f"{name}.npz" for name in ["clean", *corruptions]}
    recorded_prediction_hashes = marker.get("prediction_sha256", {})
    if set(recorded_prediction_hashes) != set(prediction_paths):
        raise RuntimeError("prediction hash manifest has wrong corruption set")
    for name, path in prediction_paths.items():
        if not path.is_file() or recorded_prediction_hashes[name] != sha256_file(path):
            raise RuntimeError(f"prediction SHA256 mismatch: {name}")

    frame = pd.read_csv(metrics_path)
    expected_conditions = {("clean", 0)} | {
        (name, severity) for name in corruptions for severity in range(1, 6)
    }
    actual_conditions = {
        (str(record["corruption"]), int(record["severity"])) for record in frame.to_dict("records")
    }
    if len(frame) != len(expected_conditions) or actual_conditions != expected_conditions:
        raise RuntimeError("metrics.csv does not contain exactly the expected conditions")
    if int(marker.get("conditions", -1)) != len(expected_conditions):
        raise RuntimeError("run_complete.json condition count mismatch")
    return frame.to_dict("records")


def aggregate(args, plan_rows: list[dict]) -> None:
    import pandas as pd

    records = []
    rejected = []
    for row in plan_rows:
        if row["status"] != "ready":
            rejected.append({"run_key": row["run_key"], "reason": row["status"], "error": row.get("error", "")})
            continue
        try:
            records.extend(validate_completed_run(args, row))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
            rejected.append({"run_key": row["run_key"], "reason": "not_verified_complete", "error": str(exc)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "aggregate_rejections.json", rejected)
    if not records:
        print(f"No verified completed metrics.csv found; rejected={len(rejected)}")
        return
    frame = pd.DataFrame(records)
    for column in ["ratio", "selection_seed", "training_seed", "severity", *METRICS]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = add_corruption_mean(frame)
    atomic_dataframe_to_csv(frame, args.output_root / "raw_metrics_all.csv")
    keys = ["dataset", "method", "ratio", "split", "corruption", "severity"]
    metrics = [column for column in METRICS if column in frame]
    summary = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["__".join(value).strip("_") if isinstance(value, tuple) else value for value in summary.columns]
    atomic_dataframe_to_csv(summary, args.output_root / "training_variance_summary.csv")
    atomic_dataframe_to_csv(compute_variance_components(frame), args.output_root / "variance_components.csv")
    pivot_keys = ["dataset", "ratio", "selection_seed", "training_seed", "split", "corruption", "severity"]
    pivot = frame.pivot_table(index=pivot_keys, columns="method", values="balanced_accuracy", aggfunc="first").reset_index()
    if "random" in pivot and "graph_a2" in pivot:
        pivot["graph_a2_minus_random_ba"] = pivot["graph_a2"] - pivot["random"]
    atomic_dataframe_to_csv(pivot, args.output_root / "paired_method_differences.csv")
    print(f"aggregated {len(frame)} rows into {args.output_root}; rejected={len(rejected)}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "run", "aggregate"], default="plan")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corrupted-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--ratios", default="0.02,0.05")
    parser.add_argument("--selection-seeds", default="1,42,2026")
    parser.add_argument("--training-seeds", default="42,43,44,45,46")
    parser.add_argument("--dataset")
    parser.add_argument("--method")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-missing-classes", action="store_true")
    return parser


def main() -> int:
    args = parser().parse_args()
    args.datasets, args.methods = parse_csv(args.datasets), parse_csv(args.methods)
    args.ratios = parse_csv(args.ratios, float)
    args.selection_seeds = parse_csv(args.selection_seeds, int)
    args.training_seeds = parse_csv(args.training_seeds, int)
    if args.dataset is not None:
        args.datasets = [args.dataset]
    if args.method is not None:
        args.methods = [args.method]
    if args.ratio is not None:
        args.ratios = [args.ratio]
    if args.selection_seed is not None:
        args.selection_seeds = [args.selection_seed]
    if args.training_seed is not None:
        args.training_seeds = [args.training_seed]
    rows = build_plan(args)
    if args.mode == "plan":
        write_plan(args, rows)
        return 0
    if args.mode == "aggregate":
        aggregate(args, rows)
        return 0
    ready = [row for row in rows if row["status"] == "ready"]
    if len(ready) != 1:
        raise SystemExit(f"run requires exactly one verified cell; statuses={[(row['run_key'], row['status']) for row in rows]}")
    run_one(args, ready[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
