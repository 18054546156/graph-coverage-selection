"""Run and provenance-log a source-train-only D8 selection smoke on HPC."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from d8_ot import select_equal_mass_ot


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_d8_hpc_smoke.py EMBEDDINGS_NPZ DATA_NPZ OUTPUT_DIR")

    embedding_path, dataset_path, output_dir = map(Path, sys.argv[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    result_path = output_dir / "selected_indices.npz"
    started = time.time()
    run = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "selection-only HPC smoke; no classifier training or test-set access",
        "method": "d8_equal_mass_class_conditional_entropic_ot_heuristic",
        "dataset": "organsmnist",
        "split": "train",
        "ratio": 0.02,
        "budget_per_class": 25,
        "seed": 0,
        "epsilon": 0.05,
        "outer_iterations": 8,
        "sinkhorn_iterations": 60,
        "embedding_path": str(embedding_path),
        "dataset_path": str(dataset_path),
        "output_path": str(output_dir),
        "slurm": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE", "SLURM_NODELIST",
        )},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    metadata_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    try:
        run["input_sha256"] = {
            "embeddings_npz": sha256(embedding_path),
            "dataset_npz": sha256(dataset_path),
            "selector_py": sha256(Path(__file__).with_name("d8_ot.py")),
        }
        with np.load(embedding_path, allow_pickle=False) as archive:
            embeddings = archive["embeddings"]
        with np.load(dataset_path, allow_pickle=False) as archive:
            labels = archive["train_labels"].reshape(-1)
        if len(embeddings) != len(labels):
            raise ValueError(f"embedding/label row mismatch: {len(embeddings)} vs {len(labels)}")

        classes, counts = np.unique(labels, return_counts=True)
        run["input_shape"] = list(embeddings.shape)
        run["class_counts"] = {str(int(label)): int(count) for label, count in zip(classes, counts)}
        run["selection_started_utc"] = datetime.now(timezone.utc).isoformat()
        selected, diagnostics = select_equal_mass_ot(
            embeddings,
            labels,
            budget_per_class=25,
            seed=0,
            epsilon=0.05,
            outer_iterations=8,
            sinkhorn_iterations=60,
        )
        run["selection_finished_utc"] = datetime.now(timezone.utc).isoformat()

        selected_labels, selected_counts = np.unique(labels[selected], return_counts=True)
        expected_count = 25 * len(classes)
        if len(selected) != expected_count or len(np.unique(selected)) != expected_count:
            raise RuntimeError("selected count/uniqueness validation failed")
        if not np.array_equal(selected_labels, classes) or not np.all(selected_counts == 25):
            raise RuntimeError("per-class quota validation failed")

        np.savez_compressed(result_path, selected_indices=selected)
        run["selected_count"] = int(len(selected))
        run["selected_indices_sha256"] = hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()
        run["class_diagnostics"] = diagnostics
        run["validation"] = {"unique_indices": True, "exact_per_class_quota": True}
        run["status"] = "complete"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = round(time.time() - started, 3)
        run["max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    print(json.dumps({key: run[key] for key in (
        "status", "selected_count", "selected_indices_sha256", "elapsed_seconds", "max_rss_kib"
    )}, indent=2))


if __name__ == "__main__":
    main()
