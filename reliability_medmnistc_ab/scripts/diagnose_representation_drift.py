"""Step 1: diagnose representation drift and selector difficulty spectra.

This is a diagnostic, not a downstream robustness result.  It uses clean
training images and training-side views only; official MedMNIST-C test files
are never used for selection or tuning.  The view corruptors are imported
from the pinned medmnistc-api registry when available, so their names and
severity parameters match the upstream implementation.

Typical HPC usage:
  python diagnose_representation_drift.py --datasets pathmnist --pilot-n 1000
  python diagnose_representation_drift.py --datasets pathmnist --pilot-n 0 --save-knn

The full run is intentionally opt-in (--pilot-n 0).  For PathMNIST, 11 views
times 5 severities over 90k train images is a substantial UNI workload.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ["organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist"]
DEFAULT_METHODS = ["random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding", "graph_a2"]
DEFAULT_RATIOS = [0.02, 0.05]
DYNAMICS_METHODS = {"el2n_top", "forgetting", "eva"}
KNOWN_VIEW_NAMES = {
    "pathmnist": ["pixelate", "jpeg_compression", "defocus_blur", "motion_blur", "brightness_up", "brightness_down", "contrast_up", "contrast_down", "saturate", "stain_deposit", "bubble"],
    "bloodmnist": ["pixelate", "jpeg_compression", "defocus_blur", "motion_blur", "brightness_up", "brightness_down", "contrast_up", "contrast_down", "saturate", "stain_deposit", "bubble"],
    "tissuemnist": ["pixelate", "jpeg_compression", "impulse_noise", "gaussian_blur", "brightness_up", "brightness_down", "contrast_up", "contrast_down"],
    "organsmnist": ["pixelate", "jpeg_compression", "gaussian_noise", "speckle_noise", "impulse_noise", "shot_noise", "gaussian_blur", "brightness_up", "brightness_down", "contrast_up", "contrast_down", "gamma_corr_up", "gamma_corr_down"],
    "organamnist": ["pixelate", "jpeg_compression", "gaussian_noise", "speckle_noise", "impulse_noise", "shot_noise", "gaussian_blur", "brightness_up", "brightness_down", "contrast_up", "contrast_down", "gamma_corr_up", "gamma_corr_down"],
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def add_import_paths(graphcov_root: Path) -> None:
    root = str(graphcov_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_medmnist(dataset_name: str, data_root: Path, size: int):
    import medmnist
    from medmnist import INFO

    info = INFO[dataset_name]
    cls = getattr(medmnist, info["python_class"])
    dataset = cls(split="train", size=size, root=str(data_root), transform=None, download=False)
    labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    return dataset, labels, info


class RawViewDataset:
    """Minimal Dataset wrapper for UNI input from raw MedMNIST arrays."""

    def __init__(self, base, indices: np.ndarray, corruptor=None, severity: int = 0, seed: int = 0, stream: str = ""):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.corruptor = corruptor
        self.severity = severity
        self.seed = seed
        self.stream = stream

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index: int):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        global_index = int(self.indices[local_index])
        image = Image.fromarray(np.asarray(self.base.imgs[global_index])).convert("RGB")
        if self.corruptor is not None:
            # Most medmnistc-api corruptions (GaussianNoise, SpeckleNoise, ShotNoise,
            # MotionBlur angle, StainDeposit, Bubble, Characters, BlackCorner) draw
            # from the *global* numpy/python random state inside apply(); only
            # ImpulseNoise consumes an explicit `self.rng` Generator, and that
            # attribute does not exist on a freshly constructed corruptor (it is
            # only set by medmnistc's official DatasetManager pipeline, which this
            # diagnostic does not run) -- so `apply()` would raise AttributeError
            # without it. Reseed both global streams and set `.rng` unconditionally,
            # keyed only on (seed, global_index), so every corruption is
            # reproducible per image regardless of DataLoader worker count/order.
            seed_material = f"{self.seed}|{self.stream}|{self.severity}|{global_index}".encode("utf-8")
            item_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "little")
            np.random.seed(item_seed)
            random.seed(item_seed)
            self.corruptor.rng = np.random.default_rng(item_seed)
            image = self.corruptor.apply(image, self.severity)
        tensor = to_tensor(image)
        return tensor, global_index


def normalize_for_uni(images, channels: int):
    import torch

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_uni(model, dataset, batch_size: int, workers: int, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    chunks, indices = [], []
    model.eval()
    with torch.inference_mode():
        for images, global_indices in loader:
            images = normalize_for_uni(images.to(device, non_blocking=True), images.shape[1])
            features = model(images)
            chunks.append(features.detach().float().cpu().numpy())
            indices.append(global_indices.numpy())
    result = np.empty((len(dataset.base.imgs) if hasattr(dataset, "base") else len(dataset), chunks[0].shape[1]), dtype=np.float32)
    flat_indices = np.concatenate(indices)
    result[flat_indices] = np.concatenate(chunks).astype(np.float32)
    return result


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(np.sum(cross * cross) / denom) if denom > 0 else float("nan")


def reciprocal_knn_rate(neighbor_idx: np.ndarray) -> np.ndarray:
    """Per-point fraction of directed kNN edges reciprocated within one graph.

    This is distinct from cross-space overlap of corresponding samples'
    directed kNN sets.
    """
    neighbor_sets = [set(int(j) for j in row) for row in neighbor_idx]
    rates = np.empty(len(neighbor_idx), dtype=np.float64)
    for i, row in enumerate(neighbor_idx):
        if len(row) == 0:
            rates[i] = float("nan")
            continue
        reciprocal = sum(1 for j in row if i in neighbor_sets[int(j)])
        rates[i] = reciprocal / len(row)
    return rates


def knn_metrics(clean: np.ndarray, view: np.ndarray, k: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.stats import spearmanr
    from graphcov.run.graph import build_knn_graph

    clean_idx, clean_dist = build_knn_graph(clean, k, verbose=False)
    view_idx, view_dist = build_knn_graph(view, k, verbose=False)
    # For corresponding samples, measure overlap of their directed kNN sets in
    # the two representation spaces. Do not call this reciprocal/mutual kNN.
    overlap_values = np.asarray([len(set(a.tolist()) & set(b.tolist())) / max(1, k) for a, b in zip(clean_idx, view_idx)], dtype=np.float64)
    reciprocal_clean = reciprocal_knn_rate(clean_idx)
    reciprocal_view = reciprocal_knn_rate(view_idx)

    # Spearman uses rank k+1 for a neighbor absent from the other list.
    rank_values = []
    common_weight_deltas = []
    common_weight_delta_values = []
    union_weight_delta_values = []
    for i in range(len(clean)):
        c_rank = {int(j): r + 1 for r, j in enumerate(clean_idx[i])}
        v_rank = {int(j): r + 1 for r, j in enumerate(view_idx[i])}
        union = sorted(set(c_rank) | set(v_rank))
        a = [c_rank.get(j, k + 1) for j in union]
        b = [v_rank.get(j, k + 1) for j in union]
        rank_values.append(float(spearmanr(a, b).statistic) if len(union) > 1 else 1.0)

        view_weights = {int(j): 1.0 - float(d) / 2.0 for j, d in zip(view_idx[i], view_dist[i])}
        clean_weights = {int(j): 1.0 - float(d) / 2.0 for j, d in zip(clean_idx[i], clean_dist[i])}
        common = set(clean_weights) & set(view_weights)
        edge_union = set(clean_weights) | set(view_weights)
        if common:
            delta = float(np.mean([abs(clean_weights[j] - view_weights[j]) for j in common]))
            common_weight_deltas.append(delta)
            common_weight_delta_values.append(delta)
        else:
            common_weight_delta_values.append(float("nan"))
        union_weight_delta_values.append(float(np.mean([
            abs(clean_weights.get(j, 0.0) - view_weights.get(j, 0.0)) for j in edge_union
        ])))

    metrics = {
        "cross_space_knn_overlap_mean": float(np.mean(overlap_values)),
        "cross_space_knn_overlap_median": float(np.median(overlap_values)),
        "neighbor_rank_spearman_mean": float(np.nanmean(rank_values)),
        "common_edge_weight_abs_change_mean": float(np.mean(common_weight_deltas)) if common_weight_deltas else float("nan"),
        "union_edge_weight_abs_change_mean": float(np.mean(union_weight_delta_values)),
        "reciprocal_knn_rate_clean_mean": float(np.nanmean(reciprocal_clean)),
        "reciprocal_knn_rate_view_mean": float(np.nanmean(reciprocal_view)),
        "reciprocal_knn_rate_drop_mean": float(np.nanmean(reciprocal_clean) - np.nanmean(reciprocal_view)),
    }
    return (
        metrics,
        clean_idx,
        view_idx,
        overlap_values,
        np.asarray(common_weight_delta_values, dtype=np.float64),
        np.asarray(union_weight_delta_values, dtype=np.float64),
    )


def per_class(values: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for c in np.unique(labels):
        v = np.asarray(values)[labels == c]
        v = v[np.isfinite(v)]
        rows.append({"class": int(c), "n": int(len(v)), "mean": float(np.mean(v)) if len(v) else float("nan"), "median": float(np.median(v)) if len(v) else float("nan"), "p10": float(np.percentile(v, 10)) if len(v) else float("nan"), "p90": float(np.percentile(v, 90)) if len(v) else float("nan")})
    return rows


def coverage_score(all_features: np.ndarray, query_indices: np.ndarray, selected_indices: np.ndarray) -> float:
    if len(selected_indices) == 0:
        return float("nan")
    q = all_features[query_indices].astype(np.float32)
    s = all_features[selected_indices].astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    # Query chunks keep memory bounded for large selected sets.
    best = np.full(len(q), -1.0, dtype=np.float32)
    for start in range(0, len(q), 512):
        best[start:start + 512] = (q[start:start + 512] @ s.T).max(axis=1)
    return float(np.mean(best))


def load_verified_selection(selection_root: Path, dataset: str, ratio: float, method: str, seed: int, labels: np.ndarray) -> np.ndarray | None:
    """Load a selection artifact from its exact expected path and verify it
    against the manifest files written by
    reliability_medmnistc/reliability/pipeline.py's save_selection_artifact
    (selection_config.json, class_counts.json, index_order_sha256.txt).

    Returns None only when the artifact directory is genuinely absent. Any
    structural mismatch (dataset/method/seed/ratio, SHA256, class quota,
    duplicate or out-of-range index) raises instead of silently trusting a
    fuzzy path match, so a stale or mismatched artifact can never feed the
    diagnostic silently.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_train = len(labels)
    n_classes = int(labels.max()) + 1 if len(labels) else 0
    directory = selection_root / method / dataset / f"ratio_{ratio:g}" / f"selection_seed_{seed}"
    indices_path = directory / "selected_indices.npy"
    if not indices_path.exists():
        return None

    config_path = directory / "selection_config.json"
    if not config_path.exists():
        raise RuntimeError(f"selection artifact missing selection_config.json: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("dataset") != dataset or config.get("method") != method:
        raise RuntimeError(f"selection_config.json dataset/method mismatch at {directory}: {config}")
    if int(config.get("selection_seed", -1)) != seed:
        raise RuntimeError(f"selection_config.json seed mismatch at {directory}: expected {seed}, got {config.get('selection_seed')}")
    config_ratio = float(config.get("ratio", float("nan")))
    if not math.isclose(config_ratio, ratio, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(f"selection_config.json ratio mismatch at {directory}: expected {ratio}, got {config_ratio}")

    selected_raw = np.load(indices_path, allow_pickle=False)
    if selected_raw.ndim != 1 or not np.issubdtype(selected_raw.dtype, np.integer):
        raise RuntimeError(f"selected_indices.npy must be a one-dimensional integer array: {directory}")
    selected = selected_raw.astype(np.int64, copy=False)
    actual_sha = sha256_array(selected)
    expected_sha = config.get("selected_indices_sha256")
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError(f"selected_indices.npy sha256 mismatch at {directory}: config says {expected_sha}, file hashes to {actual_sha}")
    sha_path = directory / "index_order_sha256.txt"
    if sha_path.exists():
        recorded = sha_path.read_text(encoding="utf-8").strip()
        if recorded != actual_sha:
            raise RuntimeError(f"index_order_sha256.txt mismatch at {directory}: {recorded} != {actual_sha}")

    if len(np.unique(selected)) != len(selected):
        raise RuntimeError(f"duplicate indices in selection artifact: {directory}")
    if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= n_train):
        raise RuntimeError(f"selection indices out of range [0, {n_train}) at {directory}")
    expected_n = config.get("n_selected")
    if expected_n is not None and int(expected_n) != len(selected):
        raise RuntimeError(f"n_selected mismatch at {directory}: config says {expected_n}, file has {len(selected)}")

    source_hash = config.get("source_train_indices_sha256")
    expected_source_hash = sha256_array(np.arange(n_train, dtype=np.int64))
    if source_hash and source_hash != expected_source_hash:
        raise RuntimeError(
            f"selection was not made against the full canonical train index order at {directory}: "
            f"{source_hash} != {expected_source_hash}"
        )

    actual_counts = np.bincount(labels[selected], minlength=n_classes)
    counts_path = directory / "class_counts.json"
    if counts_path.exists():
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        recorded_counts = np.asarray([int(counts.get(str(c), 0)) for c in range(n_classes)], dtype=np.int64)
        total = int(recorded_counts.sum())
        if total != len(selected):
            raise RuntimeError(f"class_counts.json sums to {total}, expected {len(selected)} at {directory}")
        if not np.array_equal(recorded_counts, actual_counts):
            raise RuntimeError(
                f"class_counts.json does not match labels at selected indices at {directory}: "
                f"recorded={recorded_counts.tolist()}, actual={actual_counts.tolist()}"
            )
        budget = config.get("budget_per_class")
        if budget is not None and not np.all(actual_counts == int(budget)):
            raise RuntimeError(f"class quota mismatch at {directory}: expected {budget} per class, got {actual_counts.tolist()}")
    else:
        raise RuntimeError(f"selection artifact missing class_counts.json: {directory}")

    return selected


def load_difficulty(
    cache_root: Path,
    dataset: str,
    n_train: int,
    *,
    size: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one exact official GraphCov raw-dynamics cache.

    GraphCov 8cf757a stores no metadata inside this NPZ. Its canonical filename
    is therefore part of the cache identity. Never scan recursively or accept a
    similarly named cache: doing so can silently mix selection seeds, image
    sizes, or dynamics epoch counts.
    """
    path = cache_root / f"{dataset}_train_dynamics_{size}_e{epochs}_s{seed}.npz"
    source = {
        "status": "missing",
        "path": str(path),
        "dataset": dataset,
        "split": "train",
        "image_size": int(size),
        "epochs": int(epochs),
        "seed": int(seed),
    }
    if not path.exists():
        return {}, source

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = {"all_l2_scores", "forgetting_scores"} - set(data.files)
            if missing:
                raise RuntimeError(f"raw-dynamics cache missing keys {sorted(missing)}: {path}")
            all_l2 = np.asarray(data["all_l2_scores"])
            forgetting = np.asarray(data["forgetting_scores"])
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"cannot read raw-dynamics cache {path}: {exc}") from exc

    if all_l2.shape != (epochs, n_train):
        raise RuntimeError(
            f"all_l2_scores shape mismatch in {path}: expected {(epochs, n_train)}, got {all_l2.shape}"
        )
    if forgetting.shape != (n_train,):
        raise RuntimeError(
            f"forgetting_scores shape mismatch in {path}: expected {(n_train,)}, got {forgetting.shape}"
        )
    if not np.isfinite(all_l2).all() or not np.isfinite(forgetting).all():
        raise RuntimeError(f"non-finite difficulty score in raw-dynamics cache: {path}")

    el2n_epochs = min(20, epochs)
    scores = {
        "el2n": all_l2[:el2n_epochs].mean(axis=0),
        "forgetting": forgetting,
    }
    source.update(
        status="verified",
        sha256=sha256_file(path),
        all_l2_shape=list(all_l2.shape),
        forgetting_shape=list(forgetting.shape),
        el2n_epochs=int(el2n_epochs),
    )
    return scores, source


def validate_selection_dynamics_config(
    selection_root: Path,
    dataset: str,
    ratio: float,
    method: str,
    seed: int,
    *,
    size: int,
    epochs: int,
) -> None:
    """Ensure a dynamics-based selection used the same score protocol."""
    if method not in DYNAMICS_METHODS:
        return
    directory = selection_root / method / dataset / f"ratio_{ratio:g}" / f"selection_seed_{seed}"
    config_path = directory / "selection_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"dynamic_image_size": int(size), "dynamics_epochs": int(epochs)}
    for key, value in expected.items():
        if key not in config:
            raise RuntimeError(f"selection config lacks required {key} at {config_path}")
        if int(config[key]) != value:
            raise RuntimeError(
                f"selection dynamics mismatch at {config_path}: expected {key}={value}, got {config[key]}"
            )


def load_corruptors(dataset: str) -> dict[str, Any]:
    try:
        from medmnistc.corruptions.registry import CORRUPTIONS_DS
    except ImportError as exc:
        raise RuntimeError("medmnistc-api must be on PYTHONPATH for exact training-side views") from exc
    return CORRUPTIONS_DS[dataset]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphcov-root", type=Path, default=Path(os.environ.get("GRAPHCOV_ROOT", ".")))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--selection-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True, help="Existing GraphCov embedding/dynamics cache")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--ratios", default="0.02,0.05")
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--dynamics-seed", type=int, default=None, help="Raw-dynamics seed; defaults to --selection-seed")
    p.add_argument("--dynamics-size", type=int, default=28)
    p.add_argument("--dynamics-epochs", type=int, default=200)
    p.add_argument("--pilot-n", type=int, default=1000, help="Images per dataset; 0 means all")
    p.add_argument("--views", default="all", help="all or comma-separated corruption names")
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-knn", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    ratios = parse_csv(args.ratios, float)
    severities = [int(x) - 1 for x in parse_csv(args.severities, int)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "training_side_representation_drift_diagnostic",
        "created_unix": time.time(),
        "official_test_used": False,
        "medmnistc_test_used": False,
        "view_source": "medmnistc-api registry applied to clean train images",
        "config": vars(args) | {"datasets": datasets, "methods": methods, "ratios": ratios, "severities_zero_based": severities},
        "results": [],
        "missing_selections": [],
        "difficulty_sources": {},
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    if args.dry_run:
        for dataset in datasets:
            try:
                names = list(load_corruptors(dataset)) if args.views == "all" else parse_csv(args.views)
            except RuntimeError:
                names = KNOWN_VIEW_NAMES.get(dataset, []) if args.views == "all" else parse_csv(args.views)
            print(json.dumps({"dataset": dataset, "n_views": len(names), "corruptions": names, "severities": [x + 1 for x in severities]}))
        print(f"Dry run manifest: {args.output_root / 'run_manifest.json'}")
        return 0

    add_import_paths(args.graphcov_root)
    import torch
    from graphcov.run.embeddings import get_uni_model, get_cache_path, load_cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    all_rows = []
    all_class_rows = []
    for dataset_name in datasets:
        base, labels, info = load_medmnist(dataset_name, args.data_root, 224)
        n = len(labels)
        n_classes = len(info["label"])
        sample_n = n if args.pilot_n <= 0 else min(n, args.pilot_n)
        sample_rng = np.random.default_rng(args.seed)
        sample_indices = np.sort(sample_rng.choice(n, size=sample_n, replace=False)) if sample_n < n else np.arange(n)
        dynamics_seed = args.selection_seed if args.dynamics_seed is None else args.dynamics_seed
        difficulty, difficulty_source = load_difficulty(
            args.cache_root,
            dataset_name,
            n,
            size=args.dynamics_size,
            epochs=args.dynamics_epochs,
            seed=dynamics_seed,
        )
        manifest["difficulty_sources"][dataset_name] = difficulty_source

        cache_path = get_cache_path(dataset_name, "train", "uni", 224, cache_dir=args.cache_root)
        cached = load_cached(cache_path)
        if cached is None or cached["embeddings"].shape[0] != n:
            raise RuntimeError(f"Missing/full UNI cache for {dataset_name}: {cache_path}; this diagnostic will not silently recompute it")
        clean_full = np.asarray(cached["embeddings"], dtype=np.float32)
        model = get_uni_model()
        clean = clean_full[sample_indices]
        labels_sample = labels[sample_indices]
        corruptions = load_corruptors(dataset_name)
        names = list(corruptions) if args.views == "all" else parse_csv(args.views)
        rows = []
        class_rows = []

        # Selection features are extracted together with the pilot query so a
        # pilot still measures coverage against the complete selected set.
        selected_map = {}
        for method, ratio in itertools.product(methods, ratios):
            selected = load_verified_selection(args.selection_root, dataset_name, ratio, method, args.selection_seed, labels)
            if selected is None:
                manifest["missing_selections"].append({"dataset": dataset_name, "method": method, "ratio": ratio, "seed": args.selection_seed})
                continue
            validate_selection_dynamics_config(
                args.selection_root,
                dataset_name,
                ratio,
                method,
                args.selection_seed,
                size=args.dynamics_size,
                epochs=args.dynamics_epochs,
            )
            selected_map[(method, ratio)] = selected

        all_selected = np.unique(np.concatenate(list(selected_map.values()))) if selected_map else np.empty(0, dtype=np.int64)
        work_indices = np.unique(np.concatenate([sample_indices, all_selected]))
        # Baseline clean-space coverage per selection so every corrupted row can
        # report how much coverage the corruption erodes relative to clean.
        clean_coverage_map = {key: coverage_score(clean_full, sample_indices, selected) for key, selected in selected_map.items()}

        for corruption_name in names:
            if corruption_name not in corruptions:
                raise ValueError(f"Unknown {dataset_name} corruption {corruption_name}; available={list(corruptions)}")
            for severity in severities:
                view_ds = RawViewDataset(
                    base, work_indices, corruptions[corruption_name], severity,
                    args.seed, stream=f"{dataset_name}/{corruption_name}",
                )
                view_work = extract_uni(model, view_ds, args.batch_size, args.workers, device)
                view_sample = view_work[sample_indices]
                drift, clean_knn, view_knn, overlap_values, common_weight_delta_values, union_weight_delta_values = knn_metrics(clean, view_sample, args.knn_k)
                row = {"dataset": dataset_name, "view": corruption_name, "severity": severity + 1, "n": sample_n, "cache_path": str(cache_path), "cka": linear_cka(clean, view_sample), **drift}
                for (method, ratio), selected in selected_map.items():
                    # view_work is indexed by the original MedMNIST train ID.
                    clean_coverage = clean_coverage_map[(method, ratio)]
                    view_coverage = coverage_score(view_work, sample_indices, selected)
                    row[f"coverage_clean__{method}__ratio_{ratio:g}"] = clean_coverage
                    row[f"coverage_corrupted__{method}__ratio_{ratio:g}"] = view_coverage
                    row[f"coverage_drop__{method}__ratio_{ratio:g}"] = float(clean_coverage - view_coverage)
                rows.append(row)
                # Per-class neighborhood overlap is computed from the sample's
                # clean/view neighbor sets, not from class-restricted graphs.
                overlap_class_rows = per_class(overlap_values, labels_sample)
                common_weight_class_rows = per_class(common_weight_delta_values, labels_sample)
                union_weight_class_rows = per_class(union_weight_delta_values, labels_sample)
                for overlap_row, common_weight_row, union_weight_row in zip(
                    overlap_class_rows, common_weight_class_rows, union_weight_class_rows
                ):
                    class_rows.append({"dataset": dataset_name, "view": corruption_name, "severity": severity + 1, "class": overlap_row["class"], "n": overlap_row["n"], "cross_space_knn_overlap_mean": overlap_row["mean"], "cross_space_knn_overlap_median": overlap_row["median"], "cross_space_knn_overlap_p10": overlap_row["p10"], "cross_space_knn_overlap_p90": overlap_row["p90"], "common_edge_weight_abs_change_mean": common_weight_row["mean"], "common_edge_weight_abs_change_median": common_weight_row["median"], "common_edge_weight_abs_change_p10": common_weight_row["p10"], "common_edge_weight_abs_change_p90": common_weight_row["p90"], "union_edge_weight_abs_change_mean": union_weight_row["mean"], "union_edge_weight_abs_change_median": union_weight_row["median"], "union_edge_weight_abs_change_p10": union_weight_row["p10"], "union_edge_weight_abs_change_p90": union_weight_row["p90"]})
                if args.save_knn:
                    np.savez_compressed(args.output_root / f"{dataset_name}__{corruption_name}__s{severity + 1}__knn.npz", sample_indices=sample_indices, clean_knn=clean_knn, view_knn=view_knn)
                print(f"{dataset_name} {corruption_name} s{severity + 1}: cka={row['cka']:.4f} knn_overlap={row['cross_space_knn_overlap_mean']:.4f} reciprocal_drop={row['reciprocal_knn_rate_drop_mean']:.4f}", flush=True)

        all_rows.extend(rows)
        all_class_rows.extend(class_rows)
        for (method, ratio), selected in selected_map.items():
            for score_name, score in difficulty.items():
                q = np.quantile(score, [0, .25, .5, .75, 1])
                selected_score = score[selected]
                record = {"dataset": dataset_name, "method": method, "ratio": ratio, "selection_seed": args.selection_seed, "score": score_name, "full_mean": float(np.mean(score)), "selected_mean": float(np.mean(selected_score)), "full_median": float(np.median(score)), "selected_median": float(np.median(selected_score)), "selected_q25": float(np.quantile(selected_score, .25)), "selected_q75": float(np.quantile(selected_score, .75)), "full_q0": float(q[0]), "full_q25": float(q[1]), "full_q50": float(q[2]), "full_q75": float(q[3]), "full_q100": float(q[4]), "selected_n": int(len(selected))}
                manifest["results"].append(record)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import csv

    def write_csv(path: Path, records: list[dict]) -> None:
        if not records:
            return
        # Per-dataset rows can carry different method/ratio coverage columns
        # (e.g. one dataset is missing a selection another dataset has), so the
        # header must be the union of keys across all records, not just the
        # first record's keys, or DictWriter raises/drops fields.
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    # Written once across all datasets (not per-dataset) so a multi-dataset run
    # cannot overwrite an earlier dataset's rows with a later dataset's rows.
    write_csv(args.output_root / "representation_drift.csv", all_rows)
    write_csv(args.output_root / "neighbor_overlap_by_class.csv", all_class_rows)
    write_csv(args.output_root / "difficulty_spectrum.csv", manifest["results"])
    manifest["completed_unix"] = time.time()
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
