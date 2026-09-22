#!/usr/bin/env python3
"""Freeze and validate the two source subsets before any downstream training."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from paired_core import canonicalize_indices, file_sha256, index_sha256, validate_equal_quota


def load_npz_member(npz_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(npz_path) as archive:
        member_name = member if member in archive.namelist() else f"{member}.npy"
        if member_name not in archive.namelist():
            raise KeyError(f"{member} not found in {npz_path}")
        with archive.open(member_name) as handle:
            return np.load(BytesIO(handle.read()), allow_pickle=False)


def resolve_source(paths: list[str], base_dir: Path) -> Path:
    candidates = [Path(path) if Path(path).is_absolute() else base_dir / path for path in paths]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"expected exactly one existing source, found {existing}")
    return existing[0]


def freeze(protocol_path: Path, output_root: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    labels = load_npz_member(Path(dataset["hpc_npz"]).expanduser(), "train_labels")
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != int(dataset["n_train"]):
        raise ValueError(f"train-label length mismatch: {len(labels)}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "graphcov-table1/path5-fixed-selection-manifest-v1",
        "protocol": str(protocol_path.resolve()),
        "dataset": dataset["name"],
        "ratio": dataset["ratio"],
        "canonical_index_order": protocol["canonical_index_order"],
        "test_read": False,
        "arms": {},
    }
    for arm_name, arm in protocol["arms"].items():
        source = resolve_source(arm["source_paths"], protocol_path.resolve().parent)
        source_file_sha = file_sha256(source)
        expected_source_file_sha = arm.get("expected_source_file_sha256")
        if expected_source_file_sha and source_file_sha != expected_source_file_sha:
            raise ValueError(
                f"{arm_name} source-file SHA mismatch: {source_file_sha} != {expected_source_file_sha}"
            )
        source_values = np.load(source, allow_pickle=False)
        source_order_sha = index_sha256(source_values)
        expected_source_sha = arm.get("expected_source_order_index_sha256")
        if expected_source_sha and source_order_sha != expected_source_sha:
            raise ValueError(
                f"{arm_name} source-order SHA mismatch: {source_order_sha} != {expected_source_sha}"
            )
        canonical = canonicalize_indices(
            source_values,
            n_train=int(dataset["n_train"]),
            n_selected=int(dataset["n_selected"]),
        )
        canonical_sha = index_sha256(canonical)
        expected_canonical_sha = arm.get("expected_canonical_index_sha256")
        if expected_canonical_sha and canonical_sha != expected_canonical_sha:
            raise ValueError(
                f"{arm_name} canonical SHA mismatch: {canonical_sha} != {expected_canonical_sha}"
            )
        class_counts = validate_equal_quota(
            labels,
            canonical,
            n_classes=int(dataset["n_classes"]),
            budget_per_class=int(dataset["budget_per_class"]),
        )
        arm_root = output_root / arm_name
        arm_root.mkdir(parents=True, exist_ok=True)
        destination = arm_root / "selected_indices.npy"
        np.save(destination, canonical, allow_pickle=False)
        manifest["arms"][arm_name] = {
            "label": arm["label"],
            "selection_seed": arm["selection_seed"],
            "source_path": str(source),
            "origin_hpc_path": arm.get("origin_hpc_path"),
            "source_file_sha256": source_file_sha,
            "source_order_index_sha256": source_order_sha,
            "canonical_path": str(destination.resolve()),
            "canonical_file_sha256": file_sha256(destination),
            "canonical_index_sha256": canonical_sha,
            "strictly_increasing": bool(np.all(canonical[1:] > canonical[:-1])),
            "n_selected": int(len(canonical)),
            "class_counts": class_counts,
        }

    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(temporary, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.protocol, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
