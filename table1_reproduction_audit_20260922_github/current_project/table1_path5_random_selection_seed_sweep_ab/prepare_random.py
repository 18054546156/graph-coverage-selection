#!/usr/bin/env python3
"""Generate and freeze the five public-code Random selection-seed subsets."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from official_runtime import configure_official_runtime
from paired_core import canonicalize_indices, file_sha256, index_sha256, validate_equal_quota


def load_npz_member(npz_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(npz_path) as archive:
        name = member if member in archive.namelist() else f"{member}.npy"
        with archive.open(name) as handle:
            return np.load(BytesIO(handle.read()), allow_pickle=False)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(temporary, path)


def public_random_select(labels: np.ndarray, budget_per_class: int, seed: int) -> list[int]:
    """Exact public GraphCov ``_select_random`` logic without importing FAISS."""
    np.random.seed(seed)
    selected: list[int] = []
    for class_value in np.unique(labels):
        class_indices = np.where(labels == class_value)[0]
        k = min(budget_per_class, len(class_indices))
        selected.extend(np.random.choice(class_indices, k, replace=False).tolist())
    return selected


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    configure_official_runtime()

    dataset = protocol["dataset"]
    npz_path = Path(dataset["hpc_npz"]).expanduser()
    labels = np.asarray(load_npz_member(npz_path, "train_labels")).reshape(-1)
    if len(labels) != int(dataset["n_train"]):
        raise ValueError(f"training label length mismatch: {len(labels)}")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "graphcov-table1/path5-random-selection-seed-sweep-manifest-v1",
        "dataset": dataset["name"],
        "ratio": dataset["ratio"],
        "selection": protocol["selection"],
        "test_read": False,
        "arms": {},
    }
    canonical_by_seed: dict[int, np.ndarray] = {}
    for seed in protocol["selection"]["selection_seeds"]:
        raw = np.asarray(
            public_random_select(
                labels,
                budget_per_class=int(dataset["budget_per_class"]),
                seed=int(seed),
            ),
            dtype=np.int64,
        )
        canonical = canonicalize_indices(
            raw,
            n_train=int(dataset["n_train"]),
            n_selected=int(dataset["n_selected"]),
        )
        counts = validate_equal_quota(
            labels,
            canonical,
            n_classes=int(dataset["n_classes"]),
            budget_per_class=int(dataset["budget_per_class"]),
        )
        arm = f"random_seed{int(seed)}"
        arm_root = args.artifacts / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        index_path = arm_root / "selected_indices.npy"
        np.save(index_path, canonical, allow_pickle=False)
        canonical_by_seed[int(seed)] = canonical
        manifest["arms"][arm] = {
            "label": f"Random (selection seed {int(seed)} frozen)",
            "selection_seed": int(seed),
            "canonical_path": str(index_path.resolve()),
            "canonical_file_sha256": file_sha256(index_path),
            "canonical_index_sha256": index_sha256(canonical),
            "canonical_order": "strictly increasing",
            "n_selected": int(len(canonical)),
            "class_counts": counts,
        }
    overlap = {}
    for left_seed, left in canonical_by_seed.items():
        overlap[str(left_seed)] = {}
        left_set = set(left.tolist())
        for right_seed, right in canonical_by_seed.items():
            right_set = set(right.tolist())
            overlap[str(left_seed)][str(right_seed)] = len(left_set & right_set) / len(left_set | right_set)
    manifest["pairwise_jaccard"] = overlap
    atomic_json(args.artifacts / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
