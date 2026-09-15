"""Auditable selection, training, evaluation, and summary pipeline.

The upstream selector, trainer, and MedMNIST-C implementation are imported
without modification. The command wrappers select a phase through ``PHASE``.
"""

from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset


# ---------- paths and frozen protocol ----------
PREP = Path(os.environ.get("PREP_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = Path(os.environ.get("GRAPH_ROOT", str(PREP))).resolve()
MEDC_ROOT = Path(os.environ.get("MEDC_ROOT", str(PREP / "third_party" / "medmnistc"))).resolve()
DATA_ROOT = Path(os.environ.get("MEDMNIST_ROOT", str(PREP / "medmnist_data"))).resolve()
CORR_ROOT = Path(os.environ.get("MEDMNISTC_ROOT", str(PREP / "medmnistc_data"))).resolve()
OUT = Path(os.environ.get("RELIABILITY_OUT", str(PREP / "full_run_outputs"))).resolve()
CACHE_ROOT = Path(os.environ.get("GRAPH_CACHE", str(PREP / "graph_cache"))).resolve()
SELECTION_OUT = Path(os.environ.get("SELECTION_OUT", str(OUT / "selections"))).resolve()
DATASETS = [x.strip().lower() for x in os.environ.get(
    "DATASETS", "organsmnist,organamnist,pathmnist,tissuemnist,bloodmnist"
).split(",") if x.strip()]
method_text = os.environ.get(
    "METHODS", "random,el2n_top,forgetting,eva,facility,fps,herding,graph_a2"
)
METHODS = [x.strip().lower() for x in method_text.split(",") if x.strip()]
RATIOS = [float(x) for x in os.environ.get("RATIOS", "0.02,0.05").split(",")]
SELECTION_SEED = int(os.environ.get("SELECTION_SEED", "42"))
training_seed_text = os.environ.get("TRAINING_SEEDS", os.environ.get("SEEDS", "42,43,44,45,46"))
TRAINING_SEEDS = [int(x) for x in training_seed_text.split(",") if x.strip()]
EPOCHS = int(os.environ.get("EPOCHS", "1000"))
DYNAMICS_EPOCHS = int(os.environ.get("DYNAMICS_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
EMBEDDING_SOURCE = os.environ.get("EMBEDDING_SOURCE", "uni")
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
SMOKE_N = int(os.environ.get("SMOKE_N", "0"))
RUN_FULL_TRAIN = os.environ.get("RUN_FULL_TRAIN", "1") == "1"
CORR_HASH = os.environ.get("CORR_HASH", "0") == "1"
PHASE = os.environ.get("PHASE", "full").strip().lower()
if PHASE not in {"select", "train", "evaluate", "full", "summarize"}:
    raise ValueError("PHASE must be select, train, evaluate, full, or summarize")
FACILITY_GLOBAL = os.environ.get("FACILITY_GLOBAL_SELECTION", "0") == "1"
GRAPH_GLOBAL = os.environ.get("GRAPH_GLOBAL_SELECTION", "1") == "1"
GRAPH_K = int(os.environ.get("GRAPH_K_NEIGHBORS", "50"))
GRAPH_HOPS = int(os.environ.get("GRAPH_K_HOPS", "2"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep cache namespaces separate across smoke/full runs and protocol changes.
CACHE = CACHE_ROOT / (
    f"img{SIZE}_dyn{DYNAMICS_EPOCHS}_sel{SELECTION_SEED}_smoke{SMOKE_N or 'full'}"
)

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
CORR_ROOT.mkdir(parents=True, exist_ok=True)
SELECTION_OUT.mkdir(parents=True, exist_ok=True)


def parse_corruptions(name):
    available = list(CORRUPTIONS_DS[name].keys())
    requested = os.environ.get("CORRUPTIONS", "").strip()
    if not requested:
        return available
    requested = [x.strip() for x in requested.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise ValueError(f"Unsupported corruption for {name}: {unknown}; available={available}")
    return requested


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, value)
    os.replace(tmp, path)


def atomic_save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def array_sha256(value):
    arr = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path):
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


print({
    "PREP": str(PREP), "DATA_ROOT": str(DATA_ROOT), "CORR_ROOT": str(CORR_ROOT),
    "OUT": str(OUT), "SELECTION_OUT": str(SELECTION_OUT), "device": str(DEVICE),
    "datasets": DATASETS, "methods": METHODS, "ratios": RATIOS,
    "selection_seed": SELECTION_SEED, "training_seeds": TRAINING_SEEDS,
    "epochs": EPOCHS, "augment": AUGMENT, "smoke_n": SMOKE_N,
})


# ---------- import pinned upstream code ----------
sys.path.insert(0, str(GRAPH_ROOT))
sys.path.insert(0, str(MEDC_ROOT))
from medmnist import INFO
from graphcov.run.data import get_transform, get_train_transform, AugmentedDataset
from graphcov.run.embeddings import load_or_compute_embeddings, load_or_compute_raw_dynamics
from graphcov.run.eva import get_optimal_windows, derive_eva_scores
from graphcov.run.selection import select, get_available_methods
from graphcov.run.embeddings import ResNet18WithFeatures
from graphcov.run.evaluation import train_one_epoch, set_seed
from medmnistc.dataset_manager import DatasetManager
from medmnistc.dataset import CorruptedMedMNIST
from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB

assert [m for m in METHODS if m not in get_available_methods()] == [], get_available_methods()
assert DATASETS and all(name in INFO for name in DATASETS), DATASETS
assert TRAINING_SEEDS, "TRAINING_SEEDS must not be empty"
print("Pinned GraphCov methods:", METHODS)
print("GraphCov commit:", git_commit(GRAPH_ROOT))
print("MedMNIST-C commit:", git_commit(MEDC_ROOT))


# ---------- data and corruption integrity ----------
def load_med(name, split, size=SIZE):
    info = INFO[name]
    dataset_class = getattr(__import__("medmnist", fromlist=[info["python_class"]]), info["python_class"])
    download = os.environ.get("DOWNLOAD_MEDMNIST", "0") == "1"
    dataset = dataset_class(
        split=split, transform=get_transform(info["n_channels"], size),
        download=download, size=size, root=str(DATA_ROOT),
    )
    return dataset, info


def flat_labels(dataset):
    return np.asarray(dataset.labels).reshape(-1).astype(np.int64)


def limit_dataset(dataset, indices):
    if len(indices) == len(dataset):
        return dataset
    return Subset(dataset, indices.tolist())


def validate_corruption_file(path, clean_labels, dataset_name, corruption):
    path = Path(path)
    with np.load(path, allow_pickle=False) as package:
        required = {"test_images", "test_labels"}
        if not required.issubset(package.files):
            raise RuntimeError(f"{path} missing keys; found {package.files}")
        images = package["test_images"]
        labels = np.asarray(package["test_labels"]).reshape(-1).astype(np.int64)
        expected_n = 5 * len(clean_labels)
        if len(images) != expected_n or len(labels) != expected_n:
            raise RuntimeError(f"{path}: expected {expected_n} rows, got images={len(images)}, labels={len(labels)}")
        if images.dtype != np.uint8:
            raise RuntimeError(f"{path}: expected uint8 images, got {images.dtype}")
        for severity in range(5):
            block = labels[severity * len(clean_labels):(severity + 1) * len(clean_labels)]
            if not np.array_equal(block, clean_labels):
                raise RuntimeError(f"{path}: labels do not align at severity {severity + 1}")
        record = {
            "dataset": dataset_name, "corruption": corruption,
            "path": str(path), "bytes": path.stat().st_size,
            "shape": list(images.shape), "dtype": str(images.dtype),
            "n_test": len(clean_labels), "n_severity": 5,
            "labels_sha256": array_sha256(labels),
        }
    if CORR_HASH:
        record["file_sha256"] = file_sha256(path)
    return record


def ensure_corruptions(name, clean_labels):
    wanted = parse_corruptions(name)
    dataset_dir = CORR_ROOT / name
    missing = [c for c in wanted if not (dataset_dir / f"{c}.npz").exists()]
    if missing:
        print("Generating missing MedMNIST-C files through upstream API:", name, missing)
        DatasetManager(
            medmnist_path=str(DATA_ROOT), output_path=str(CORR_ROOT), random_seed=0,
        ).create_dataset(name)
    records = []
    for corruption in wanted:
        records.append(validate_corruption_file(
            dataset_dir / f"{corruption}.npz", clean_labels, name, corruption,
        ))
    atomic_write_json(OUT / "_data_manifests" / f"{name}.json", {
        "dataset": name, "clean_test_labels_sha256": array_sha256(clean_labels),
        "corruptions": records, "medmnistc_commit": git_commit(MEDC_ROOT),
    })
    return wanted


def make_corruption_view(dataset, clean_indices, full_test_n):
    clean_indices = np.asarray(clean_indices, dtype=np.int64)
    source_indices = np.concatenate([
        severity * full_test_n + clean_indices for severity in range(5)
    ])
    view = Subset(dataset, source_indices.tolist())
    clean_ids = np.tile(clean_indices, 5)
    severities = np.repeat(np.arange(1, 6, dtype=np.int64), len(clean_indices))
    return view, clean_ids, severities


# ---------- metrics ----------
def log_softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise RuntimeError("logits must be a finite 2-D array")
    max_logits = logits.max(axis=1, keepdims=True)
    log_norm = max_logits + np.log(np.exp(logits - max_logits).sum(axis=1, keepdims=True))
    return logits - log_norm


def probs_from_logits(logits):
    probs = np.exp(log_softmax_np(logits)).astype(np.float32)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise RuntimeError("invalid probabilities derived from logits")
    return probs


def ece_top(y, probs, bins=15):
    pred = probs.argmax(1)
    confidence = probs.max(1)
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        mask = ((confidence >= lo) if index == 0 else (confidence > lo)) & (confidence <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float((pred[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return float(total)


def risk_coverage(y, probs, accept_fraction=0.80):
    if len(y) == 0:
        raise RuntimeError("cannot evaluate an empty test set")
    pred = probs.argmax(1)
    order = np.argsort(-probs.max(1), kind="stable")
    accepted_n = max(1, int(round(len(y) * accept_fraction)))
    accepted = order[:accepted_n]
    risk = float(np.mean(pred[accepted] != y[accepted]))
    errors = (pred[order] != y[order]).astype(np.float64)
    aurc = float(np.mean(np.cumsum(errors) / np.arange(1, len(y) + 1)))
    return risk, aurc, accepted_n


def metric_row(y, logits, num_classes, prefix=""):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64)
    if len(y) == 0 or logits.shape[0] != len(y):
        raise RuntimeError("labels and logits have incompatible lengths")
    log_probs = log_softmax_np(logits)
    probs = np.exp(log_probs).astype(np.float32)
    pred = probs.argmax(1)
    recalls = {}
    for class_id in range(num_classes):
        mask = y == class_id
        recalls[class_id] = float(np.mean(pred[mask] == class_id)) if mask.any() else float("nan")
    observed = [value for value in recalls.values() if not math.isnan(value)]
    if not observed:
        raise RuntimeError("no observed classes")
    risk80, aurc, accepted_n = risk_coverage(y, probs)
    true_log_prob = log_probs[np.arange(len(y)), y]
    onehot = np.eye(num_classes, dtype=np.float64)[y]
    high_conf_errors = int(np.sum((probs.max(1) >= 0.80) & (pred != y)))
    row = {
        prefix + "n_samples": int(len(y)),
        prefix + "n_classes_observed": int(len(observed)),
        prefix + "acc": float(np.mean(pred == y)),
        prefix + "ba": float(np.mean(observed)),
        prefix + "worst_recall": float(np.min(observed)),
        prefix + "nll": float(-np.mean(true_log_prob)),
        prefix + "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        prefix + "ece15": ece_top(y, probs),
        prefix + "risk_at_80": risk80,
        prefix + "risk_at_80_n": int(accepted_n),
        prefix + "aurc": aurc,
        prefix + "high_conf_errors": high_conf_errors,
    }
    for class_id, value in recalls.items():
        row[f"{prefix}recall_{class_id}"] = value
    return row, probs, pred


def save_predictions(path, clean_id, y, logits, probs, severity):
    atomic_save_npz(
        path, sample_id=np.asarray(clean_id, dtype=np.int64),
        clean_id=np.asarray(clean_id, dtype=np.int64),
        severity=np.asarray(severity, dtype=np.int64),
        y_true=np.asarray(y, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        probs=np.asarray(probs, dtype=np.float32),
    )


# ---------- selection ----------
def get_selection_data(name, train_224, train_28, info):
    needs_embedding = any(m in {"facility", "fps", "herding", "graph_a2"} for m in METHODS)
    embeddings = None
    if needs_embedding:
        embeddings = load_or_compute_embeddings(
            name, "train", EMBEDDING_SOURCE, train_224, len(info["label"]),
            info["n_channels"], size=SIZE, seed=SELECTION_SEED,
            cache_dir=CACHE, verbose=True,
        )["embeddings"]
    dynamics = None
    if any(m in {"el2n_top", "forgetting", "eva"} for m in METHODS):
        dynamics = load_or_compute_raw_dynamics(
            name, "train", train_28, len(info["label"]), info["n_channels"],
            size=28, seed=SELECTION_SEED, eva_epochs=DYNAMICS_EPOCHS,
            window_size=10, cache_dir=CACHE, verbose=True,
        )
    return embeddings, dynamics


def choose(name, labels, info, embeddings, dynamics, ratio):
    n_classes = len(info["label"])
    budget = int(len(labels) * ratio) // n_classes
    if budget < 1:
        raise ValueError(f"ratio={ratio} gives zero samples per class for {name}")
    selected = {}
    base = dict(labels=labels, budget_per_class=budget, seed=SELECTION_SEED, verbose=False)
    for method in METHODS:
        args = dict(base, method=method, embeddings=embeddings)
        if dynamics is not None:
            all_l2 = dynamics["all_l2_scores"]
            args["el2n_scores"] = all_l2[:min(20, len(all_l2))].mean(0)
            args["forgetting_scores"] = dynamics["forgetting_scores"]
            if method == "eva":
                early, late = get_optimal_windows(name, ratio, eva_epochs=DYNAMICS_EPOCHS, verbose=False)
                args["eva_scores"] = derive_eva_scores(
                    all_l2, window_size=10, early_window_start=early,
                    late_window_start=late, verbose=False,
                )[0]
        if method in {"facility", "fps", "herding", "graph_a2"}:
            args["importance"] = np.ones(len(labels), dtype=np.float32)
        if method == "facility":
            args["global_selection"] = FACILITY_GLOBAL
        if method == "graph_a2":
            args.update(
                global_selection=GRAPH_GLOBAL,
                k_neighbors=GRAPH_K,
                k_hops=GRAPH_HOPS,
            )
        local_indices = np.asarray(select(**args), dtype=np.int64)
        if len(np.unique(local_indices)) != len(local_indices):
            raise RuntimeError(f"duplicate selected index: {method}")
        if np.any(local_indices < 0) or np.any(local_indices >= len(labels)):
            raise RuntimeError(f"out-of-range selected index: {method}")
        counts = np.bincount(labels[local_indices], minlength=n_classes)
        if not np.all(counts == budget):
            raise RuntimeError(f"quota mismatch for {method}: expected {budget}, got {counts.tolist()}")
        selected[method] = local_indices
    return selected, budget


def save_selection_artifact(name, method, ratio, local_indices, source_train_indices, labels, budget, info):
    ratio_label = f"{ratio:g}"
    directory = SELECTION_OUT / method / name / f"ratio_{ratio_label}" / f"selection_seed_{SELECTION_SEED}"
    directory.mkdir(parents=True, exist_ok=True)
    original_indices = np.asarray(source_train_indices[local_indices], dtype=np.int64)
    atomic_save_npy(directory / "selected_indices.npy", original_indices)
    atomic_save_npy(directory / "selected_local_indices.npy", local_indices)
    atomic_write_text(directory / "selection_seed.txt", f"{SELECTION_SEED}\n")
    counts = np.bincount(labels[local_indices], minlength=len(info["label"]))
    atomic_write_json(directory / "class_counts.json", {str(i): int(v) for i, v in enumerate(counts)})
    atomic_write_text(directory / "index_order_sha256.txt", array_sha256(original_indices) + "\n")
    atomic_write_json(directory / "selection_config.json", {
        "dataset": name, "method": method, "ratio": ratio,
        "selection_seed": SELECTION_SEED, "budget_per_class": int(budget),
        "n_selected": int(len(local_indices)), "source_train_indices_sha256": array_sha256(source_train_indices),
        "selected_indices_sha256": array_sha256(original_indices),
        "graphcov_commit": git_commit(GRAPH_ROOT), "embedding_source": EMBEDDING_SOURCE,
        "image_size": SIZE, "dynamic_image_size": 28, "dynamics_epochs": DYNAMICS_EPOCHS,
        "facility_global": FACILITY_GLOBAL,
        "graph_global": GRAPH_GLOBAL if method == "graph_a2" else None,
        "graph_k": GRAPH_K if method == "graph_a2" else None,
        "graph_hops": GRAPH_HOPS if method == "graph_a2" else None,
    })
    return original_indices


# ---------- resumable run state ----------
def run_directory(name, method, ratio_label, train_seed):
    return OUT / name / method / f"ratio_{ratio_label}" / f"selection_seed_{SELECTION_SEED}" / f"train_seed_{train_seed}" / f"aug_{int(AUGMENT)}"


def row_key(row):
    return (str(row["corruption"]), int(row.get("severity", 0)))


def load_run_rows(path):
    rows = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                rows[row_key(row)] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def save_run_rows(path, rows):
    ordered = sorted(rows.values(), key=lambda row: (row_key(row)[0] != "clean", row_key(row)))
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=True, default=json_default) + "\n" for row in ordered))


def run_config(name, method, ratio_label, ratio_value, train_seed, n_selected, budget, corruptions):
    return {
        "schema_version": 2, "dataset": name, "method": method,
        "ratio": ratio_label if ratio_value is None else float(ratio_value),
        "selection_seed": SELECTION_SEED, "training_seed": int(train_seed),
        "augment": int(AUGMENT), "n_selected": int(n_selected),
        "budget_per_class": None if budget is None else int(budget),
        "corruptions": list(corruptions), "severity": [1, 2, 3, 4, 5],
        "image_size": SIZE, "epochs": EPOCHS, "dynamics_image_size": 28,
        "dynamics_epochs": DYNAMICS_EPOCHS, "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS, "embedding_source": EMBEDDING_SOURCE,
        "optimizer": {"name": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005},
        "scheduler": "CosineAnnealingLR per epoch", "checkpoint_rule": "final_epoch",
        "cache_namespace": str(CACHE),
        "graphcov_commit": git_commit(GRAPH_ROOT), "medmnistc_commit": git_commit(MEDC_ROOT),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "hostname": socket.gethostname(), "platform": platform.platform(),
    }


def model_from_checkpoint(path, num_classes, in_channels, config_hash):
    checkpoint = torch.load(path, map_location=DEVICE)
    if checkpoint.get("config_hash") != config_hash:
        raise RuntimeError(f"checkpoint config mismatch: {path}")
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_model(train_ds, selected, num_classes, in_channels, seed):
    set_seed(seed, deterministic=False)
    dataset = train_ds
    if AUGMENT:
        dataset = AugmentedDataset(dataset, get_train_transform(in_channels, SIZE))
    loader = DataLoader(
        Subset(dataset, selected.tolist()), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % max(1, EPOCHS // 10) == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}")
    return model


def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(DEVICE)).cpu().numpy())
            labels.append(np.asarray(batch_labels).reshape(-1))
    return np.concatenate(labels), np.concatenate(logits)


def is_complete(run_dir, expected_corruptions):
    marker = run_dir / "run_complete.json"
    if not marker.exists() or not (run_dir / "final.pt").exists():
        return False
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("status") != "complete":
            return False
    except json.JSONDecodeError:
        return False
    rows = load_run_rows(run_dir / "metrics.jsonl")
    required = {("clean", 0)} | {(c, severity) for c in expected_corruptions for severity in range(1, 6)}
    if not required.issubset(rows):
        return False
    if not (run_dir / "predictions_clean.npz").exists():
        return False
    return all((run_dir / f"predictions_{c}.npz").exists() for c in expected_corruptions)


def run_one(name, method, ratio_label, ratio_value, train_seed, train_ds, test_ds, y_test, test_indices,
            selected_local, selected_original, info, budget, corruptions):
    run_dir = run_directory(name, method, ratio_label, train_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(
        name, method, ratio_label, ratio_value, train_seed,
        len(selected_local), budget, corruptions,
    )
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=json_default).encode()).hexdigest()
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if old_config.get("config_hash") != config_hash:
            raise RuntimeError(f"existing run has a different configuration: {run_dir}")
    else:
        config["config_hash"] = config_hash
        atomic_write_json(config_path, config)
    if is_complete(run_dir, corruptions):
        print("skip complete", run_dir)
        return

    atomic_save_npy(run_dir / "selected_indices.npy", selected_original)
    atomic_save_npy(run_dir / "selected_local_indices.npy", selected_local)
    atomic_write_text(run_dir / "selection_index_order_sha256.txt", array_sha256(selected_original) + "\n")
    atomic_write_json(run_dir / "run_status.json", {"status": "started", "updated_at": time.time()})

    model = None
    checkpoint_path = run_dir / "final.pt"
    if checkpoint_path.exists():
        try:
            model = model_from_checkpoint(checkpoint_path, len(info["label"]), info["n_channels"], config_hash)
            print("resume from final checkpoint", checkpoint_path)
        except Exception as error:
            print("checkpoint cannot be resumed; retraining:", error)
    if model is None:
        if PHASE == "evaluate":
            raise FileNotFoundError(
                f"evaluate phase requires an existing final.pt: {checkpoint_path}"
            )
        print("training", run_dir, "n=", len(selected_local), "budget_per_class=", budget)
        model = train_model(train_ds, selected_local, len(info["label"]), info["n_channels"], train_seed)
        checkpoint = {
            "state_dict": model.state_dict(), "config_hash": config_hash,
            "dataset": name, "method": method, "ratio": ratio_label,
            "selection_seed": SELECTION_SEED, "training_seed": train_seed,
            "augment": AUGMENT, "size": SIZE, "checkpoint_rule": "final_epoch",
        }
        tmp_checkpoint = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        torch.save(checkpoint, tmp_checkpoint)
        os.replace(tmp_checkpoint, checkpoint_path)

    if PHASE == "train":
        atomic_write_json(run_dir / "run_status.json", {
            "status": "trained", "updated_at": time.time(),
            "checkpoint": str(checkpoint_path),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    rows = load_run_rows(run_dir / "metrics.jsonl")
    meta = {
        "dataset": name, "method": method, "ratio": ratio_label if ratio_value is None else float(ratio_value),
        "selection_seed": SELECTION_SEED, "training_seed": train_seed,
        "augment": int(AUGMENT), "n_selected": int(len(selected_local)),
        "budget_per_class": None if budget is None else int(budget),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if ("clean", 0) not in rows or not (run_dir / "predictions_clean.npz").exists():
        y, logits = predict(model, test_ds)
        if not np.array_equal(y, y_test[test_indices]):
            raise RuntimeError(f"clean label order mismatch: {run_dir}")
        row, probs, _ = metric_row(y, logits, len(info["label"]))
        save_predictions(run_dir / "predictions_clean.npz", test_indices, y, logits, probs, np.zeros(len(y), dtype=np.int64))
        clean_row = dict(meta, corruption="clean", severity=0)
        clean_row.update(row)
        rows[("clean", 0)] = clean_row
        save_run_rows(run_dir / "metrics.jsonl", rows)
        atomic_write_json(run_dir / "run_status.json", {"status": "clean_complete", "updated_at": time.time()})

    for corruption in corruptions:
        needed = {(corruption, severity) for severity in range(1, 6)}
        if needed.issubset(rows) and (run_dir / f"predictions_{corruption}.npz").exists():
            continue
        corruption_dataset = CorruptedMedMNIST(
            name, corruption, norm_mean=[0.5] * info["n_channels"],
            norm_std=[0.5] * info["n_channels"], root=str(CORR_ROOT),
            as_rgb=DATASET_RGB[name], mmap_mode="r",
        )
        view, clean_ids, severities = make_corruption_view(corruption_dataset, test_indices, len(y_test))
        cy, clogits = predict(model, view)
        if not np.array_equal(cy, np.tile(y_test[test_indices], 5)):
            raise RuntimeError(f"corruption label order mismatch: {name}/{corruption}")
        cprobs = probs_from_logits(clogits)
        save_predictions(run_dir / f"predictions_{corruption}.npz", clean_ids, cy, clogits, cprobs, severities)
        n_eval = len(test_indices)
        for severity in range(1, 6):
            start, end = (severity - 1) * n_eval, severity * n_eval
            crow, _, _ = metric_row(cy[start:end], clogits[start:end], len(info["label"]))
            cbase = dict(meta, corruption=corruption, severity=severity)
            cbase.update(crow)
            cbase["clean_ba_drop"] = float(rows[("clean", 0)]["ba"] - crow["ba"])
            cbase["clean_worst_recall_drop"] = float(rows[("clean", 0)]["worst_recall"] - crow["worst_recall"])
            rows[(corruption, severity)] = cbase
        save_run_rows(run_dir / "metrics.jsonl", rows)
        atomic_write_json(run_dir / "run_status.json", {"status": f"{corruption}_complete", "updated_at": time.time()})
        del corruption_dataset, view
    atomic_write_json(run_dir / "run_complete.json", {
        "status": "complete", "completed_at": time.time(),
        "conditions": len(rows), "config_hash": config_hash,
    })
    atomic_write_json(run_dir / "run_status.json", {"status": "complete", "updated_at": time.time()})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def consolidate():
    rows = {}
    for path in OUT.rglob("metrics.jsonl"):
        if path == OUT / "metrics.jsonl":
            continue
        marker = path.parent / "run_complete.json"
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("status") != "complete":
                continue
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                identity = (
                    row["dataset"], row["method"], str(row["ratio"]),
                    int(row["selection_seed"]), int(row["training_seed"]),
                    int(row["augment"]), row["corruption"], int(row.get("severity", 0)),
                )
                rows[identity] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    ordered = [rows[key] for key in sorted(rows)]
    atomic_write_text(OUT / "metrics.jsonl", "".join(json.dumps(row, ensure_ascii=True, default=json_default) + "\n" for row in ordered))
    frame = pd.DataFrame(ordered)
    frame.to_csv(OUT / "metrics.csv", index=False)
    if not frame.empty:
        frame[frame.corruption == "clean"].to_csv(OUT / "clean_metrics.csv", index=False)
        frame[frame.corruption != "clean"].to_csv(OUT / "corruption_metrics_by_severity.csv", index=False)
    print("consolidated rows:", len(frame), "output:", OUT)


# ---------- main execution ----------
if PHASE == "summarize":
    consolidate()
    raise SystemExit(0)

for dataset_name in DATASETS:
    print("==========", dataset_name, "==========")
    train_full, info = load_med(dataset_name, "train", SIZE)
    test_full, _ = load_med(dataset_name, "test", SIZE)
    train_28_full, _ = load_med(dataset_name, "train", 28)
    y_train_full = flat_labels(train_full)
    y_test_full = flat_labels(test_full)
    y_train_28 = flat_labels(train_28_full)
    if not np.array_equal(y_train_full, y_train_28):
        raise RuntimeError(f"224/28 train labels differ for {dataset_name}")
    train_source_indices = np.arange(len(train_full), dtype=np.int64)
    test_source_indices = np.arange(len(test_full), dtype=np.int64)
    if SMOKE_N > 0:
        train_source_indices = train_source_indices[:min(SMOKE_N, len(train_source_indices))]
        test_source_indices = test_source_indices[:min(SMOKE_N, len(test_source_indices))]
    train_ds = limit_dataset(train_full, train_source_indices)
    test_ds = limit_dataset(test_full, test_source_indices)
    train_28_ds = limit_dataset(train_28_full, train_source_indices)
    y_train = y_train_full[train_source_indices]
    y_test = y_test_full[test_source_indices]
    corruptions = (
        [] if PHASE in {"select", "train"}
        else ensure_corruptions(dataset_name, y_test_full)
    )
    embeddings, dynamics = get_selection_data(dataset_name, train_ds, train_28_ds, info)

    for ratio in RATIOS:
        chosen, budget = choose(dataset_name, y_train, info, embeddings, dynamics, ratio)
        for method, local_indices in chosen.items():
            original_indices = save_selection_artifact(
                dataset_name, method, ratio, local_indices, train_source_indices,
                y_train, budget, info,
            )
            if PHASE == "select":
                continue
            for train_seed in TRAINING_SEEDS:
                run_one(
                    dataset_name, method, f"{ratio:g}", ratio, train_seed, train_ds, test_ds,
                    y_test_full, test_source_indices, local_indices, original_indices,
                    info, budget, corruptions,
                )

    if RUN_FULL_TRAIN:
        full_local = np.arange(len(train_ds), dtype=np.int64)
        full_original = train_source_indices.copy()
        for train_seed in TRAINING_SEEDS:
            run_one(
                dataset_name, "full_train", "full", None, train_seed, train_ds, test_ds,
                y_test_full, test_source_indices, full_local, full_original,
                info, None, corruptions,
            )

consolidate()
