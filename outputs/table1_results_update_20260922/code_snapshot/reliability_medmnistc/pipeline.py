"""Auditable selection, training, evaluation, and summary pipeline.

The upstream selector, trainer, and MedMNIST-C implementation are imported
without modification. The command wrappers select a phase through ``PHASE``.
"""

from pathlib import Path
import csv
import fcntl
import hashlib
import json
import math
import os
import random
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
TRAIN_SOURCE = os.environ.get("TRAIN_SOURCE", "clean").strip().lower()
if TRAIN_SOURCE not in {"clean", "corrupted"}:
    raise ValueError("TRAIN_SOURCE must be clean or corrupted")
TRAIN_CORRUPTION = os.environ.get("TRAIN_CORRUPTION", "pixelate").strip()
TRAIN_SEVERITY = int(os.environ.get("TRAIN_SEVERITY", "3"))
if not 1 <= TRAIN_SEVERITY <= 5:
    raise ValueError("TRAIN_SEVERITY must be in [1, 5]")
TRAIN_ON_CORRUPTED = os.environ.get("TRAIN_ON_CORRUPTED", "0") == "1"
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
# Training determinism. train_model() previously hardcoded
# set_seed(seed, deterministic=False), which leaves cudnn.benchmark on and
# permits nondeterministic kernels. Measured consequence: re-running an
# identical (selection sha256, training seed, config) cell across trees
# produced clean-BA spreads of 5.8pp (random) to 8.7pp (forgetting) -- replicate
# noise larger than the effects being measured.
#
# The opt-in knob is deliberately NOT named DETERMINISTIC: several existing
# launchers already `export DETERMINISTIC=1` as dead code (nothing read it), so
# binding to that name would silently flip protocol the next time any of those
# scripts is resubmitted, including other users' in-flight jobs. A new name
# makes every opt-in explicit and greppable. Costs ~10-50% wall clock.
DETERMINISTIC = os.environ.get("DETERMINISTIC_TRAINING", "0") == "1"
if not DETERMINISTIC and os.environ.get("DETERMINISTIC", "0") == "1":
    print(
        "WARNING: DETERMINISTIC=1 is set but has no effect -- it was never read "
        "by this pipeline. Training remains nondeterministic (measured replicate "
        "spread 5.8-8.7pp). Set DETERMINISTIC_TRAINING=1 to actually enable it.",
        file=sys.stderr, flush=True,
    )
SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
SMOKE_N = int(os.environ.get("SMOKE_N", "0"))
RUN_FULL_TRAIN = os.environ.get("RUN_FULL_TRAIN", "1") == "1"
CORR_HASH = os.environ.get("CORR_HASH", "0") == "1"
CLEAN_ONLY = os.environ.get("CLEAN_ONLY", "0") == "1"
AUTO_CONSOLIDATE = os.environ.get("AUTO_CONSOLIDATE", "1") == "1"
ALLOW_MISSING_CLASSES = os.environ.get("ALLOW_MISSING_CLASSES", "0") == "1"
EXPECTED_GRAPHCOV_COMMIT = os.environ.get(
    "EXPECTED_GRAPHCOV_COMMIT", "8cf757adc4c333dc1427d511f0de2f246d15ebac"
)
PHASE = os.environ.get("PHASE", "full").strip().lower()
if PHASE not in {"select", "validate", "train", "evaluate", "full", "summarize"}:
    raise ValueError("PHASE must be select, validate, train, evaluate, full, or summarize")
FACILITY_GLOBAL = os.environ.get("FACILITY_GLOBAL_SELECTION", "0") == "1"
FACILITY_EXECUTION_DEVICE = os.environ.get("FACILITY_EXECUTION_DEVICE", "auto").strip().lower()
if FACILITY_EXECUTION_DEVICE not in {"auto", "cpu", "cuda"}:
    raise ValueError("FACILITY_EXECUTION_DEVICE must be auto, cpu, or cuda")
FACILITY_CPU_MIN_CLASS_SIZE = int(os.environ.get("FACILITY_CPU_MIN_CLASS_SIZE", "40000"))
if FACILITY_CPU_MIN_CLASS_SIZE < 1:
    raise ValueError("FACILITY_CPU_MIN_CLASS_SIZE must be positive")
GRAPH_GLOBAL = os.environ.get("GRAPH_GLOBAL_SELECTION", "1") == "1"
GRAPH_K = int(os.environ.get("GRAPH_K_NEIGHBORS", "50"))
GRAPH_HOPS = int(os.environ.get("GRAPH_K_HOPS", "2"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep cache namespaces separate across smoke/full runs and protocol changes.
CACHE = CACHE_ROOT / (
    f"img{SIZE}_dyn{DYNAMICS_EPOCHS}_sel{SELECTION_SEED}_smoke{SMOKE_N or 'full'}"
)
# Pretrained embeddings do not depend on the selection seed. Keep them in a
# shared namespace so paired-seed jobs reuse one exact extraction. Dynamics
# remain in CACHE because they are seed-dependent.
EMBEDDING_CACHE = CACHE_ROOT / (
    f"embeddings_img{SIZE}_smoke{SMOKE_N or 'full'}"
)

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
CORR_ROOT.mkdir(parents=True, exist_ok=True)
SELECTION_OUT.mkdir(parents=True, exist_ok=True)
VERIFIED_SOURCE_FILES = {}


def parse_corruptions(name):
    if CLEAN_ONLY:
        return []
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
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_save_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(tmp, value)
    os.replace(tmp, path)


def atomic_save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
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


def verify_source_file(path, expected_sha256):
    path = Path(path)
    cache_key = (str(path.resolve()), expected_sha256)
    if cache_key in VERIFIED_SOURCE_FILES:
        return
    if not path.is_file():
        raise RuntimeError(f"selection source file is missing: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"selection source SHA256 mismatch: {path}: {actual} != {expected_sha256}")
    VERIFIED_SOURCE_FILES[cache_key] = True


def atomic_write_dataframe_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


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
        "deterministic": DETERMINISTIC,
        "clean_only": CLEAN_ONLY,
    "train_source": TRAIN_SOURCE, "train_corruption": TRAIN_CORRUPTION,
    "train_severity": TRAIN_SEVERITY, "train_on_corrupted": TRAIN_ON_CORRUPTED,
})


# ---------- import pinned upstream code ----------
sys.path.insert(0, str(GRAPH_ROOT))
sys.path.insert(0, str(MEDC_ROOT))
from medmnist import INFO
from graphcov.run.data import get_transform, get_train_transform, AugmentedDataset
from graphcov.run.embeddings import (
    get_cache_path,
    get_dynamics_cache_path,
    load_or_compute_embeddings,
    load_or_compute_raw_dynamics,
)
from graphcov.run.eva import get_optimal_windows, derive_eva_scores
from graphcov.run import selection as graphcov_selection
from graphcov.run.selection import get_available_methods
from graphcov.run.embeddings import ResNet18WithFeatures
from graphcov.run.evaluation import train_one_epoch, set_seed
from medmnistc.dataset import CorruptedMedMNIST
from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB

assert [m for m in METHODS if m not in get_available_methods()] == [], get_available_methods()
assert DATASETS and all(name in INFO for name in DATASETS), DATASETS
assert TRAINING_SEEDS, "TRAINING_SEEDS must not be empty"


def assert_graphcov_source_unchanged():
    import subprocess

    try:
        subprocess.run(
            ["git", "-C", str(GRAPH_ROOT), "merge-base", "--is-ancestor", EXPECTED_GRAPHCOV_COMMIT, "HEAD"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(GRAPH_ROOT), "diff", "--quiet", EXPECTED_GRAPHCOV_COMMIT, "--", "graphcov"],
            check=True, capture_output=True, text=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"graphcov/ must remain identical to official {EXPECTED_GRAPHCOV_COMMIT}"
        ) from exc


assert_graphcov_source_unchanged()
print("Pinned GraphCov methods:", METHODS)
print("GraphCov official base commit:", EXPECTED_GRAPHCOV_COMMIT)
print("MedMNIST-C commit:", git_commit(MEDC_ROOT))


# ---------- data and corruption integrity ----------
class CorruptedViewDataset(torch.utils.data.Dataset):
    """Deterministic on-the-fly train view; keeps clean and corrupted IDs aligned."""

    def __init__(self, base, transform, dataset_name, corruption, severity):
        self.base = base
        self.imgs = base.imgs
        self.labels = base.labels
        self.transform = transform
        self.dataset_name = dataset_name
        self.corruptor = CORRUPTIONS_DS[dataset_name][corruption]
        self.severity = int(severity) - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(np.asarray(self.imgs[index])).convert("RGB")
        image = self.corruptor.apply(image, self.severity)
        if not DATASET_RGB[self.dataset_name]:
            image = Image.fromarray(image).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        # Match MedMNIST's label shape so GraphCov embedding workers can use
        # both scalar and array-label code paths consistently.
        return image, np.asarray(self.labels[index]).reshape(-1)


def load_med(name, split, size=SIZE, source_override=None):
    info = INFO[name]
    dataset_class = getattr(__import__("medmnist", fromlist=[info["python_class"]]), info["python_class"])
    download = os.environ.get("DOWNLOAD_MEDMNIST", "0") == "1"
    source = TRAIN_SOURCE if source_override is None and split == "train" else (source_override or "clean")
    if split == "train" and source == "corrupted":
        base = dataset_class(
            split=split, transform=None, download=download, size=size,
            root=str(DATA_ROOT), as_rgb=True,
        )
        if TRAIN_CORRUPTION not in CORRUPTIONS_DS[name]:
            raise RuntimeError(
                f"{TRAIN_CORRUPTION!r} is not available for {name}; "
                f"available={sorted(CORRUPTIONS_DS[name])}"
            )
        return CorruptedViewDataset(
            base, get_transform(info["n_channels"], size), name,
            TRAIN_CORRUPTION, TRAIN_SEVERITY,
        ), info
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
        raise RuntimeError(
            f"missing MedMNIST-C files for {name}: {missing}; run "
            "reliability_medmnistc/scripts/generate_medmnistc.py before evaluation"
        )
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
    missing_classes = [class_id for class_id, value in recalls.items() if math.isnan(value)]
    if missing_classes and not ALLOW_MISSING_CLASSES:
        raise RuntimeError(f"evaluation split is missing required classes: {missing_classes}")
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
def resolve_facility_device(labels):
    """Choose a deterministic execution device without changing Facility's objective."""
    if FACILITY_EXECUTION_DEVICE != "auto":
        resolved = FACILITY_EXECUTION_DEVICE
    else:
        _, counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
        largest_class = int(counts.max()) if counts.size else 0
        resolved = "cpu" if largest_class >= FACILITY_CPU_MIN_CLASS_SIZE else "cuda"
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Facility requires CUDA under the configured device policy")
    return resolved


def get_selection_data(name, train_224, train_28, info):
    needs_embedding = any(m in {"facility", "fps", "herding", "graph_a2"} for m in METHODS)
    embeddings = None
    sources = {}
    if needs_embedding:
        EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)
        lock_path = EMBEDDING_CACHE / f".{name}_{EMBEDDING_SOURCE}.lock"
        # Two paired-seed jobs can start together. Serialize only the
        # seed-independent extraction for this dataset; different datasets
        # still run concurrently.
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            embedding_data = load_or_compute_embeddings(
                name, "train", EMBEDDING_SOURCE, train_224, len(info["label"]),
                info["n_channels"], size=SIZE, seed=SELECTION_SEED,
                cache_dir=EMBEDDING_CACHE, verbose=True,
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        embeddings = np.asarray(embedding_data["embeddings"])
        embedding_cache = get_cache_path(
            name, "train", EMBEDDING_SOURCE, SIZE, SELECTION_SEED,
            cache_dir=EMBEDDING_CACHE,
        )
        sources["embedding"] = {
            "source": EMBEDDING_SOURCE,
            "path": str(embedding_cache),
            "file_sha256": file_sha256(embedding_cache),
            "array_sha256": array_sha256(embeddings),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }
    dynamics = None
    if any(m in {"el2n_top", "forgetting", "eva"} for m in METHODS):
        dynamics = load_or_compute_raw_dynamics(
            name, "train", train_28, len(info["label"]), info["n_channels"],
            size=28, seed=SELECTION_SEED, eva_epochs=DYNAMICS_EPOCHS,
            window_size=10, cache_dir=CACHE, verbose=True,
        )
        dynamics_cache = get_dynamics_cache_path(
            name, "train", 28, SELECTION_SEED, DYNAMICS_EPOCHS, CACHE,
        )
        sources["dynamics"] = {
            "path": str(dynamics_cache),
            "file_sha256": file_sha256(dynamics_cache),
            "all_l2_sha256": array_sha256(dynamics["all_l2_scores"]),
            "forgetting_sha256": array_sha256(dynamics["forgetting_scores"]),
            "all_l2_shape": list(np.asarray(dynamics["all_l2_scores"]).shape),
        }
    return embeddings, dynamics, sources


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
        previous_device = None
        if method == "facility":
            resolved_device = resolve_facility_device(labels)
            previous_device = graphcov_selection.device
            graphcov_selection.device = torch.device(resolved_device)
            print(
                f"  [Facility] execution_device={resolved_device}, "
                f"cpu_min_class_size={FACILITY_CPU_MIN_CLASS_SIZE}"
            )
        try:
            local_indices = np.asarray(graphcov_selection.select(**args), dtype=np.int64)
        finally:
            if previous_device is not None:
                graphcov_selection.device = previous_device
        if len(np.unique(local_indices)) != len(local_indices):
            raise RuntimeError(f"duplicate selected index: {method}")
        if np.any(local_indices < 0) or np.any(local_indices >= len(labels)):
            raise RuntimeError(f"out-of-range selected index: {method}")
        counts = np.bincount(labels[local_indices], minlength=n_classes)
        if not np.all(counts == budget):
            raise RuntimeError(f"quota mismatch for {method}: expected {budget}, got {counts.tolist()}")
        selected[method] = local_indices
    return selected, budget


def save_selection_artifact(name, method, ratio, local_indices, source_train_indices, labels, budget, info, sources):
    ratio_label = f"{ratio:g}"
    directory = SELECTION_OUT / method / name / f"ratio_{ratio_label}" / f"selection_seed_{SELECTION_SEED}"
    if directory.exists():
        existing_local, existing_original, existing_budget, _ = load_selection_artifact(
            name, method, ratio, source_train_indices, labels, info,
        )
        expected_original = np.asarray(source_train_indices[local_indices], dtype=np.int64)
        if (
            existing_budget != int(budget)
            or not np.array_equal(existing_local, np.asarray(local_indices, dtype=np.int64))
            or not np.array_equal(existing_original, expected_original)
        ):
            raise RuntimeError(f"refusing to overwrite a different selection artifact: {directory}")
        print("reuse verified selection artifact", directory)
        return existing_original
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + f".tmp.{os.getpid()}")
    staging.mkdir(parents=False, exist_ok=False)
    original_indices = np.asarray(source_train_indices[local_indices], dtype=np.int64)
    atomic_save_npy(staging / "selected_indices.npy", original_indices)
    atomic_save_npy(staging / "selected_local_indices.npy", local_indices)
    atomic_write_text(staging / "selection_seed.txt", f"{SELECTION_SEED}\n")
    counts = np.bincount(labels[local_indices], minlength=len(info["label"]))
    atomic_write_json(staging / "class_counts.json", {str(i): int(v) for i, v in enumerate(counts)})
    atomic_write_text(staging / "index_order_sha256.txt", array_sha256(original_indices) + "\n")
    method_sources = {}
    if method in {"facility", "fps", "herding", "graph_a2"}:
        method_sources["embedding"] = sources["embedding"]
    if method in {"el2n_top", "forgetting", "eva"}:
        method_sources["dynamics"] = sources["dynamics"]
    atomic_write_json(staging / "selection_config.json", {
        "dataset": name, "method": method, "ratio": ratio,
        "selection_seed": SELECTION_SEED, "budget_per_class": int(budget),
        "n_selected": int(len(local_indices)), "source_train_indices_sha256": array_sha256(source_train_indices),
        "selected_indices_sha256": array_sha256(original_indices),
        "graphcov_commit": EXPECTED_GRAPHCOV_COMMIT, "embedding_source": EMBEDDING_SOURCE,
        "image_size": SIZE, "dynamic_image_size": 28, "dynamics_epochs": DYNAMICS_EPOCHS,
        "source_train_labels_sha256": array_sha256(labels),
        "selection_sources": method_sources,
        "facility_global": FACILITY_GLOBAL,
        "facility_execution_device": resolve_facility_device(labels) if method == "facility" else None,
        "facility_cpu_min_class_size": FACILITY_CPU_MIN_CLASS_SIZE if method == "facility" else None,
        "graph_global": GRAPH_GLOBAL if method == "graph_a2" else None,
        "graph_k": GRAPH_K if method == "graph_a2" else None,
        "graph_hops": GRAPH_HOPS if method == "graph_a2" else None,
        "selection_train_source": TRAIN_SOURCE,
        "selection_train_corruption": TRAIN_CORRUPTION if TRAIN_SOURCE == "corrupted" else None,
        "selection_train_severity": TRAIN_SEVERITY if TRAIN_SOURCE == "corrupted" else None,
    })
    os.replace(staging, directory)
    return original_indices


def load_selection_artifact(name, method, ratio, source_train_indices, labels, info):
    """Load one exact selected-index artifact without rerunning selection."""
    ratio_label = f"{ratio:g}"
    directory = SELECTION_OUT / method / name / f"ratio_{ratio_label}" / f"selection_seed_{SELECTION_SEED}"
    required = [
        "selected_indices.npy", "selected_local_indices.npy", "selection_seed.txt",
        "class_counts.json", "index_order_sha256.txt", "selection_config.json",
    ]
    missing = [filename for filename in required if not (directory / filename).is_file()]
    if missing:
        raise RuntimeError(f"selection artifact missing {missing}: {directory}")

    config = json.loads((directory / "selection_config.json").read_text(encoding="utf-8"))
    expected_scalars = {
        "dataset": name,
        "method": method,
        "selection_seed": SELECTION_SEED,
        "graphcov_commit": EXPECTED_GRAPHCOV_COMMIT,
        "image_size": SIZE,
        "dynamic_image_size": 28,
        "dynamics_epochs": DYNAMICS_EPOCHS,
        "embedding_source": EMBEDDING_SOURCE,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise RuntimeError(f"selection config {key} mismatch at {directory}: {config.get(key)!r} != {expected!r}")
    expected_source = TRAIN_SOURCE
    if config.get("selection_train_source", "clean") != expected_source:
        raise RuntimeError(
            f"selection train source mismatch at {directory}: "
            f"{config.get('selection_train_source')!r} != {expected_source!r}"
        )
    if expected_source == "corrupted":
        if config.get("selection_train_corruption") != TRAIN_CORRUPTION:
            raise RuntimeError(f"selection corruption mismatch at {directory}")
        if int(config.get("selection_train_severity", -1)) != TRAIN_SEVERITY:
            raise RuntimeError(f"selection severity mismatch at {directory}")
    if not math.isclose(float(config.get("ratio", float("nan"))), ratio, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"selection ratio mismatch at {directory}")
    expected_source_keys = set()
    if method in {"facility", "fps", "herding", "graph_a2"}:
        expected_source_keys.add("embedding")
    if method in {"el2n_top", "forgetting", "eva"}:
        expected_source_keys.add("dynamics")
    if set(config.get("selection_sources", {})) != expected_source_keys:
        raise RuntimeError(f"selection source manifest mismatch at {directory}")
    for source_name in expected_source_keys:
        source = config["selection_sources"][source_name]
        expected_hash = source.get("file_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f"invalid {source_name} source hash at {directory}")
        verify_source_file(source.get("path", ""), expected_hash)
    if "embedding" in expected_source_keys:
        shape = config["selection_sources"]["embedding"].get("shape")
        if not isinstance(shape, list) or not shape or int(shape[0]) != len(labels):
            raise RuntimeError(f"embedding row count mismatch at {directory}")
    if "dynamics" in expected_source_keys:
        shape = config["selection_sources"]["dynamics"].get("all_l2_shape")
        if shape != [DYNAMICS_EPOCHS, len(labels)]:
            raise RuntimeError(f"dynamics shape mismatch at {directory}: {shape}")
    if method == "facility" and bool(config.get("facility_global")) != FACILITY_GLOBAL:
        raise RuntimeError(f"facility mode mismatch at {directory}")
    if method == "facility" and config.get("facility_execution_device") is not None:
        expected_device = resolve_facility_device(labels)
        if config.get("facility_execution_device") != expected_device:
            raise RuntimeError(f"facility execution device mismatch at {directory}")
        if int(config.get("facility_cpu_min_class_size", -1)) != FACILITY_CPU_MIN_CLASS_SIZE:
            raise RuntimeError(f"facility CPU threshold mismatch at {directory}")
    if method == "graph_a2":
        graph_expected = {
            "graph_global": GRAPH_GLOBAL,
            "graph_k": GRAPH_K,
            "graph_hops": GRAPH_HOPS,
        }
        for key, expected in graph_expected.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Graph-A2 {key} mismatch at {directory}")

    original_raw = np.load(directory / "selected_indices.npy", allow_pickle=False)
    local_raw = np.load(directory / "selected_local_indices.npy", allow_pickle=False)
    if original_raw.ndim != 1 or local_raw.ndim != 1:
        raise RuntimeError(f"selection indices must be one-dimensional: {directory}")
    if not np.issubdtype(original_raw.dtype, np.integer) or not np.issubdtype(local_raw.dtype, np.integer):
        raise RuntimeError(f"selection indices must use integer dtype: {directory}")
    original = original_raw.astype(np.int64, copy=False)
    local = local_raw.astype(np.int64, copy=False)
    if len(original) != len(local) or len(np.unique(original)) != len(original) or len(np.unique(local)) != len(local):
        raise RuntimeError(f"selection count/uniqueness mismatch: {directory}")
    if local.size and (int(local.min()) < 0 or int(local.max()) >= len(source_train_indices)):
        raise RuntimeError(f"local selection index out of range: {directory}")
    if not np.array_equal(np.asarray(source_train_indices, dtype=np.int64)[local], original):
        raise RuntimeError(f"local/original selection mapping mismatch: {directory}")
    if config.get("source_train_indices_sha256") != array_sha256(source_train_indices):
        raise RuntimeError(f"source train order hash mismatch: {directory}")
    if config.get("source_train_labels_sha256") != array_sha256(labels):
        raise RuntimeError(f"source train label hash mismatch: {directory}")
    selected_hash = array_sha256(original)
    if config.get("selected_indices_sha256") != selected_hash:
        raise RuntimeError(f"selected-index config hash mismatch: {directory}")
    if (directory / "index_order_sha256.txt").read_text(encoding="utf-8").strip() != selected_hash:
        raise RuntimeError(f"selected-index sidecar hash mismatch: {directory}")
    if int((directory / "selection_seed.txt").read_text(encoding="utf-8").strip()) != SELECTION_SEED:
        raise RuntimeError(f"selection seed sidecar mismatch: {directory}")
    if int(config.get("n_selected", -1)) != len(original):
        raise RuntimeError(f"selection size mismatch: {directory}")

    n_classes = len(info["label"])
    budget = int(config.get("budget_per_class", -1))
    actual_counts = np.bincount(np.asarray(labels, dtype=np.int64)[local], minlength=n_classes)
    recorded_raw = json.loads((directory / "class_counts.json").read_text(encoding="utf-8"))
    if set(recorded_raw) != {str(index) for index in range(n_classes)}:
        raise RuntimeError(f"class-count keys mismatch: {directory}")
    recorded_counts = np.asarray([int(recorded_raw[str(index)]) for index in range(n_classes)])
    if budget < 1 or not np.all(actual_counts == budget) or not np.array_equal(actual_counts, recorded_counts):
        raise RuntimeError(f"class quota mismatch at {directory}: {actual_counts.tolist()}")
    return local, original, budget, config


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


def run_config(name, method, ratio_label, ratio_value, train_seed, n_selected, budget, corruptions, selection_sha256):
    # NOTE on "deterministic": this dict is hashed into config_hash, which
    # gates checkpoint reuse. The key is therefore emitted ONLY when
    # DETERMINISTIC=1. Emitting it unconditionally would change config_hash
    # for every legacy run and invalidate every existing checkpoint on resume.
    # Conditional emission gives the behaviour we actually want: legacy runs
    # keep their hash, while a deterministic run hashes differently because it
    # genuinely is a different training protocol and must not be pooled with
    # nondeterministic replicates.
    deterministic_fields = {"deterministic": True} if DETERMINISTIC else {}
    return {
        **deterministic_fields,
        "schema_version": 2, "dataset": name, "method": method,
        "ratio": ratio_label if ratio_value is None else float(ratio_value),
        "selection_seed": SELECTION_SEED, "training_seed": int(train_seed),
        "augment": int(AUGMENT), "n_selected": int(n_selected),
        "budget_per_class": None if budget is None else int(budget),
        "selection_sha256": selection_sha256,
        "corruptions": list(corruptions), "severity": [1, 2, 3, 4, 5],
        "clean_only": CLEAN_ONLY,
        "image_size": SIZE, "epochs": EPOCHS, "dynamics_image_size": 28,
        "dynamics_epochs": DYNAMICS_EPOCHS, "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS, "embedding_source": EMBEDDING_SOURCE,
        "optimizer": {"name": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005},
        "scheduler": "CosineAnnealingLR per epoch", "checkpoint_rule": "final_epoch",
        "cache_namespace": str(CACHE),
        "graphcov_commit": EXPECTED_GRAPHCOV_COMMIT, "medmnistc_commit": git_commit(MEDC_ROOT),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "allow_missing_classes": ALLOW_MISSING_CLASSES,
        "selection_train_source": TRAIN_SOURCE,
        "selection_train_corruption": TRAIN_CORRUPTION if TRAIN_SOURCE == "corrupted" else None,
        "selection_train_severity": TRAIN_SEVERITY if TRAIN_SOURCE == "corrupted" else None,
        "training_source": "corrupted" if TRAIN_ON_CORRUPTED else "clean",
    }


def model_from_checkpoint(path, num_classes, in_channels, config_hash):
    checkpoint = torch.load(path, map_location=DEVICE)
    if checkpoint.get("config_hash") != config_hash:
        raise RuntimeError(f"checkpoint config mismatch: {path}")
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_model(train_ds, selected, num_classes, in_channels, seed, history_path):
    set_seed(seed, deterministic=DETERMINISTIC)
    dataset = train_ds
    if AUGMENT:
        dataset = AugmentedDataset(dataset, get_train_transform(in_channels, SIZE))
    loader_kwargs = {}
    if DETERMINISTIC:
        # Pin the shuffle stream to its own generator so the batch order no
        # longer depends on how much global RNG the surrounding code consumed.
        # Only under DETERMINISTIC=1 -- attaching a generator unconditionally
        # would change the sampling stream, and therefore every result, even
        # in the legacy path.
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = lambda worker_id: set_seed(
            seed + worker_id, deterministic=False
        )
    loader = DataLoader(
        Subset(dataset, selected.tolist()), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False, **loader_kwargs,
    )
    model = ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(EPOCHS):
        loss, accuracy = train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(loss),
            "train_accuracy": float(accuracy),
            "lr": float(scheduler.get_last_lr()[0]),
        })
        atomic_write_json(history_path, history)
        if (epoch + 1) % max(1, EPOCHS // 10) == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}")
    return model, history


def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(DEVICE)).cpu().numpy())
            labels.append(np.asarray(batch_labels).reshape(-1))
    return np.concatenate(labels), np.concatenate(logits)


def validate_prediction(path, clean_ids, severities, labels, num_classes):
    with np.load(path, allow_pickle=False) as package:
        required = {"clean_id", "severity", "y_true", "logits", "probs"}
        if not required.issubset(package.files):
            raise RuntimeError(f"prediction file missing keys at {path}: {package.files}")
        stored_ids = np.asarray(package["clean_id"], dtype=np.int64)
        stored_severities = np.asarray(package["severity"], dtype=np.int64)
        stored_labels = np.asarray(package["y_true"], dtype=np.int64)
        logits = np.asarray(package["logits"])
        probs = np.asarray(package["probs"])
    if not np.array_equal(stored_ids, np.asarray(clean_ids, dtype=np.int64)):
        raise RuntimeError(f"clean IDs mismatch: {path}")
    if not np.array_equal(stored_severities, np.asarray(severities, dtype=np.int64)):
        raise RuntimeError(f"severity IDs mismatch: {path}")
    if not np.array_equal(stored_labels, np.asarray(labels, dtype=np.int64)):
        raise RuntimeError(f"labels mismatch: {path}")
    expected_shape = (len(stored_labels), num_classes)
    if logits.shape != expected_shape or probs.shape != expected_shape:
        raise RuntimeError(f"prediction shape mismatch: {path}")
    if not np.isfinite(logits).all() or not np.isfinite(probs).all():
        raise RuntimeError(f"non-finite prediction values: {path}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"probabilities do not sum to one: {path}")


def validate_metric_rows(rows, required, meta, num_classes):
    if set(rows) != required:
        raise RuntimeError(
            f"metrics condition set mismatch: missing={sorted(required - set(rows))}, "
            f"extra={sorted(set(rows) - required)}"
        )
    metric_names = ["acc", "ba", "worst_recall", "nll", "brier", "ece15", "aurc"]
    for condition, row in rows.items():
        for key in (
            "dataset", "method", "ratio", "selection_seed", "training_seed", "augment",
            "n_selected", "budget_per_class", "selection_sha256", "checkpoint_sha256",
        ):
            if row.get(key) != meta[key]:
                raise RuntimeError(f"metric row {condition} has mismatched {key}")
        if int(row.get("n_classes_observed", -1)) != num_classes and not ALLOW_MISSING_CLASSES:
            raise RuntimeError(f"metric row {condition} does not contain all {num_classes} classes")
        for key in metric_names:
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(f"metric row {condition} has invalid {key}")


def validate_training_history(path, expected_epochs):
    try:
        history = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid training history: {path}") from exc
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise RuntimeError(f"training history length mismatch: {path}")
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict) or int(row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError(f"training history epoch mismatch at {path}: {expected_epoch}")
        for key in ("train_loss", "train_accuracy", "lr"):
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise RuntimeError(f"training history has invalid {key} at {path}: {expected_epoch}")
    return history


def validate_completion_marker(run_dir, completion):
    if completion.get("status") != "complete":
        raise RuntimeError(f"run completion status is not complete: {run_dir}")
    required_paths = {
        "config": run_dir / "run_config.json",
        "checkpoint": run_dir / "final.pt",
        "history": run_dir / "training_history.json",
        "metrics": run_dir / "metrics.jsonl",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"completed run is missing files: {missing}")
    config = json.loads(required_paths["config"].read_text(encoding="utf-8"))
    recorded_config_hash = config.pop("config_hash", None)
    actual_config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=json_default).encode()
    ).hexdigest()
    if recorded_config_hash != actual_config_hash or completion.get("config_hash") != actual_config_hash:
        raise RuntimeError(f"run config hash mismatch: {run_dir}")
    selected_path = run_dir / "selected_indices.npy"
    local_path = run_dir / "selected_local_indices.npy"
    if not selected_path.is_file() or not local_path.is_file():
        raise RuntimeError(f"completed run is missing selected indices: {run_dir}")
    selected = np.load(selected_path, allow_pickle=False)
    local = np.load(local_path, allow_pickle=False)
    if (
        selected.ndim != 1 or local.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or not np.issubdtype(local.dtype, np.integer)
        or len(selected) != int(config.get("n_selected", -1))
        or len(local) != len(selected)
        or len(np.unique(selected)) != len(selected)
        or len(np.unique(local)) != len(local)
    ):
        raise RuntimeError(f"completed run has invalid selected indices: {run_dir}")
    selection_hash = array_sha256(selected.astype(np.int64, copy=False))
    if completion.get("selection_sha256") != selection_hash or config.get("selection_sha256") != selection_hash:
        raise RuntimeError(f"completed run selection hash mismatch: {run_dir}")
    if completion.get("selection_config_sha256") != config.get("selection_config_sha256"):
        raise RuntimeError(f"completed run selection config hash mismatch: {run_dir}")
    expected_epochs = int(config.get("epochs", -1))
    if expected_epochs < 1:
        raise RuntimeError(f"invalid epoch count in run config: {run_dir}")
    validate_training_history(required_paths["history"], expected_epochs)
    expected_hashes = {
        "checkpoint_sha256": file_sha256(required_paths["checkpoint"]),
        "training_history_sha256": file_sha256(required_paths["history"]),
        "metrics_sha256": file_sha256(required_paths["metrics"]),
    }
    for key, actual in expected_hashes.items():
        if completion.get(key) != actual:
            raise RuntimeError(f"{key} mismatch: {run_dir}")
    prediction_hashes = completion.get("prediction_sha256", {})
    corruptions = config.get("corruptions")
    if not isinstance(corruptions, list) or not all(isinstance(name, str) and name for name in corruptions):
        raise RuntimeError(f"invalid corruption list in run config: {run_dir}")
    expected_prediction_names = {"clean", *corruptions}
    if set(prediction_hashes) != expected_prediction_names:
        raise RuntimeError(f"prediction hash manifest condition mismatch: {run_dir}")
    for name, expected in prediction_hashes.items():
        path = run_dir / ("predictions_clean.npz" if name == "clean" else f"predictions_{name}.npz")
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"prediction hash mismatch for {name}: {run_dir}")
    num_classes = int(config.get("num_classes", -1))
    if num_classes < 2:
        raise RuntimeError(f"invalid class count in run config: {run_dir}")
    rows = load_run_rows(required_paths["metrics"])
    required_conditions = {("clean", 0)} | {
        (corruption, severity) for corruption in corruptions for severity in range(1, 6)
    }
    meta = {
        key: config[key]
        for key in (
            "dataset", "method", "ratio", "selection_seed", "training_seed", "augment",
            "n_selected", "budget_per_class", "selection_sha256",
        )
    }
    meta["checkpoint_sha256"] = completion.get("checkpoint_sha256")
    validate_metric_rows(rows, required_conditions, meta, num_classes)
    if int(completion.get("conditions", -1)) != len(required_conditions):
        raise RuntimeError(f"completion condition count mismatch: {run_dir}")
    return required_paths["metrics"]


def run_one(name, method, ratio_label, ratio_value, train_seed, train_ds, test_ds, y_test, test_indices,
            selected_local, selected_original, info, budget, corruptions, selection_config):
    run_dir = run_directory(name, method, ratio_label, train_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(
        name, method, ratio_label, ratio_value, train_seed,
        len(selected_local), budget, corruptions, array_sha256(selected_original),
    )
    config["num_classes"] = len(info["label"])
    selection_config_sha256 = hashlib.sha256(
        json.dumps(selection_config, sort_keys=True, default=json_default).encode()
    ).hexdigest()
    config["selection_config_sha256"] = selection_config_sha256
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=json_default).encode()).hexdigest()
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        old_recorded_hash = old_config.pop("config_hash", None)
        old_actual_hash = hashlib.sha256(
            json.dumps(old_config, sort_keys=True, default=json_default).encode()
        ).hexdigest()
        if old_recorded_hash != config_hash or old_actual_hash != config_hash:
            raise RuntimeError(f"existing run has a different configuration: {run_dir}")
    else:
        config["config_hash"] = config_hash
        atomic_write_json(config_path, config)
    completion_path = run_dir / "run_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        validate_completion_marker(run_dir, completion)
        print("skip verified complete", run_dir)
        return
    atomic_save_npy(run_dir / "selected_indices.npy", selected_original)
    atomic_save_npy(run_dir / "selected_local_indices.npy", selected_local)
    atomic_write_text(run_dir / "selection_index_order_sha256.txt", array_sha256(selected_original) + "\n")
    atomic_write_json(run_dir / "run_status.json", {"status": "started", "updated_at": time.time()})

    model = None
    checkpoint_path = run_dir / "final.pt"
    history_path = run_dir / "training_history.json"
    if checkpoint_path.exists():
        if not history_path.is_file():
            raise RuntimeError(f"checkpoint exists without training history: {checkpoint_path}")
        validate_training_history(history_path, EPOCHS)
        model = model_from_checkpoint(checkpoint_path, len(info["label"]), info["n_channels"], config_hash)
        print("resume from final checkpoint", checkpoint_path)
    if model is None:
        if PHASE == "evaluate":
            raise FileNotFoundError(
                f"evaluate phase requires an existing final.pt: {checkpoint_path}"
            )
        print("training", run_dir, "n=", len(selected_local), "budget_per_class=", budget)
        model, _ = train_model(
            train_ds, selected_local, len(info["label"]), info["n_channels"],
            train_seed, history_path,
        )
        checkpoint = {
            "state_dict": model.state_dict(), "config_hash": config_hash,
            "dataset": name, "method": method, "ratio": ratio_label,
            "selection_seed": SELECTION_SEED, "training_seed": train_seed,
            "augment": AUGMENT, "size": SIZE, "checkpoint_rule": "final_epoch",
        }
        tmp_checkpoint = checkpoint_path.with_name(checkpoint_path.name + f".tmp.{os.getpid()}")
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
        "deterministic": int(DETERMINISTIC),
        "budget_per_class": None if budget is None else int(budget),
        "selection_sha256": array_sha256(selected_original),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    clean_prediction = run_dir / "predictions_clean.npz"
    expected_clean_labels = y_test[test_indices]
    clean_severity = np.zeros(len(test_indices), dtype=np.int64)
    if clean_prediction.is_file():
        validate_prediction(clean_prediction, test_indices, clean_severity, expected_clean_labels, len(info["label"]))
    if ("clean", 0) not in rows or not clean_prediction.exists():
        y, logits = predict(model, test_ds)
        if not np.array_equal(y, y_test[test_indices]):
            raise RuntimeError(f"clean label order mismatch: {run_dir}")
        row, probs, _ = metric_row(y, logits, len(info["label"]))
        save_predictions(clean_prediction, test_indices, y, logits, probs, clean_severity)
        clean_row = dict(meta, corruption="clean", severity=0)
        clean_row.update(row)
        rows[("clean", 0)] = clean_row
        save_run_rows(run_dir / "metrics.jsonl", rows)
        atomic_write_json(run_dir / "run_status.json", {"status": "clean_complete", "updated_at": time.time()})

    for corruption in corruptions:
        needed = {(corruption, severity) for severity in range(1, 6)}
        prediction_path = run_dir / f"predictions_{corruption}.npz"
        expected_corrupt_labels = np.tile(y_test[test_indices], 5)
        expected_clean_ids = np.tile(test_indices, 5)
        expected_severities = np.repeat(np.arange(1, 6, dtype=np.int64), len(test_indices))
        if prediction_path.is_file():
            validate_prediction(
                prediction_path, expected_clean_ids, expected_severities,
                expected_corrupt_labels, len(info["label"]),
            )
        if needed.issubset(rows) and prediction_path.exists():
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
        save_predictions(prediction_path, clean_ids, cy, clogits, cprobs, severities)
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
    required = {("clean", 0)} | {(corruption, severity) for corruption in corruptions for severity in range(1, 6)}
    validate_metric_rows(rows, required, meta, len(info["label"]))
    metrics_path = run_dir / "metrics.jsonl"
    prediction_paths = {
        "clean": clean_prediction,
        **{corruption: run_dir / f"predictions_{corruption}.npz" for corruption in corruptions},
    }
    for corruption, path in prediction_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing prediction artifact: {corruption} {path}")
    atomic_write_json(run_dir / "run_complete.json", {
        "status": "complete", "completed_at": time.time(),
        "conditions": len(rows), "config_hash": config_hash,
        "selection_sha256": array_sha256(selected_original),
        "selection_config_sha256": selection_config_sha256,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_history_sha256": file_sha256(history_path),
        "metrics_sha256": file_sha256(metrics_path),
        "prediction_sha256": {key: file_sha256(path) for key, path in prediction_paths.items()},
    })
    atomic_write_json(run_dir / "run_status.json", {"status": "complete", "updated_at": time.time()})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def consolidate():
    rows = {}
    rejected = []
    for path in OUT.rglob("metrics.jsonl"):
        if path == OUT / "metrics.jsonl":
            continue
        marker = path.parent / "run_complete.json"
        try:
            completion = json.loads(marker.read_text(encoding="utf-8"))
            if completion.get("status") != "complete":
                continue
            validate_completion_marker(path.parent, completion)
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            rejected.append({"run_dir": str(path.parent), "error": str(exc)})
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
    atomic_write_dataframe_csv(OUT / "metrics.csv", frame)
    if not frame.empty:
        atomic_write_dataframe_csv(OUT / "clean_metrics.csv", frame[frame.corruption == "clean"])
        atomic_write_dataframe_csv(OUT / "corruption_metrics_by_severity.csv", frame[frame.corruption != "clean"])
    atomic_write_json(OUT / "summary_rejections.json", rejected)
    print("consolidated rows:", len(frame), "rejected:", len(rejected), "output:", OUT)


# ---------- main execution ----------
if PHASE == "summarize":
    consolidate()
    raise SystemExit(0)

for dataset_name in DATASETS:
    print("==========", dataset_name, "==========")
    train_full, info = load_med(dataset_name, "train", SIZE)
    # Selection and downstream training can intentionally use different views,
    # while original IDs and labels remain shared.
    if TRAIN_SOURCE == "corrupted" and not TRAIN_ON_CORRUPTED:
        training_full, _ = load_med(dataset_name, "train", SIZE, source_override="clean")
    else:
        training_full = train_full
    test_full, _ = load_med(dataset_name, "test", SIZE)
    y_train_full = flat_labels(train_full)
    y_test_full = flat_labels(test_full)
    train_source_indices = np.arange(len(train_full), dtype=np.int64)
    test_source_indices = np.arange(len(test_full), dtype=np.int64)
    if SMOKE_N > 0:
        train_source_indices = train_source_indices[:min(SMOKE_N, len(train_source_indices))]
        test_source_indices = test_source_indices[:min(SMOKE_N, len(test_source_indices))]
    train_ds = limit_dataset(train_full, train_source_indices)
    training_ds = limit_dataset(training_full, train_source_indices)
    test_ds = limit_dataset(test_full, test_source_indices)
    y_train = y_train_full[train_source_indices]
    corruptions = parse_corruptions(dataset_name)
    if PHASE in {"evaluate", "full"} and not CLEAN_ONLY:
        ensure_corruptions(dataset_name, y_test_full)

    selecting = PHASE in {"select", "full"}
    embeddings = dynamics = None
    selection_sources = None
    if selecting:
        train_28_full, _ = load_med(dataset_name, "train", 28)
        y_train_28 = flat_labels(train_28_full)
        if not np.array_equal(y_train_full, y_train_28):
            raise RuntimeError(f"224/28 train labels differ for {dataset_name}")
        train_28_ds = limit_dataset(train_28_full, train_source_indices)
        embeddings, dynamics, selection_sources = get_selection_data(
            dataset_name, train_ds, train_28_ds, info,
        )
        if TRAIN_SOURCE == "corrupted":
            atomic_write_json(OUT / "_data_manifests" / f"{dataset_name}_train_view.json", {
                "dataset": dataset_name,
                "source": "medmnist train split, transformed on the fly",
                "corruption": TRAIN_CORRUPTION,
                "severity": TRAIN_SEVERITY,
                "selection_train_source": TRAIN_SOURCE,
                "training_source": "corrupted" if TRAIN_ON_CORRUPTED else "clean",
                "clean_train_labels_sha256": array_sha256(y_train_full),
                "n_train": int(len(y_train_full)),
            })

    for ratio in RATIOS:
        chosen = None
        if selecting:
            chosen, budget = choose(dataset_name, y_train, info, embeddings, dynamics, ratio)
        for method in METHODS:
            if selecting:
                local_indices = chosen[method]
                original_indices = save_selection_artifact(
                    dataset_name, method, ratio, local_indices, train_source_indices,
                    y_train, budget, info, selection_sources,
                )
                selection_config = json.loads(
                    (SELECTION_OUT / method / dataset_name / f"ratio_{ratio:g}" /
                     f"selection_seed_{SELECTION_SEED}" / "selection_config.json").read_text(encoding="utf-8")
                )
            else:
                local_indices, original_indices, budget, selection_config = load_selection_artifact(
                    dataset_name, method, ratio, train_source_indices, y_train, info,
                )
            if PHASE in {"select", "validate"}:
                continue
            for train_seed in TRAINING_SEEDS:
                run_one(
                    dataset_name, method, f"{ratio:g}", ratio, train_seed, training_ds, test_ds,
                    y_test_full, test_source_indices, local_indices, original_indices,
                    info, budget, corruptions, selection_config,
                )

    if RUN_FULL_TRAIN:
        full_local = np.arange(len(train_ds), dtype=np.int64)
        full_original = train_source_indices.copy()
        for train_seed in TRAINING_SEEDS:
            run_one(
                dataset_name, "full_train", "full", None, train_seed, train_ds, test_ds,
                y_test_full, test_source_indices, full_local, full_original,
                info, None, corruptions, {
                    "kind": "full_train", "source_train_indices_sha256": array_sha256(train_source_indices),
                    "source_train_labels_sha256": array_sha256(y_train),
                },
            )

if PHASE == "validate":
    print("all requested selection artifacts validated")
elif AUTO_CONSOLIDATE:
    consolidate()
