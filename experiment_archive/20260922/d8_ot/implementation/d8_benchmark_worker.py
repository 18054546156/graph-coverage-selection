"""Run four deterministic D8 clean-benchmark cells in one Slurm array worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist")
SEEDS = (0, 1, 2, 3, 4, 5, 6, 2026)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_trace(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    root = Path(os.environ["D8_ROOT"])
    source = Path(os.environ["SOURCE_ROOT"])
    python = Path(os.environ["PYTHON_BIN"])
    smoke = sys.argv[1:] == ["--smoke"]
    task_id = 0 if smoke else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not smoke and not 0 <= task_id < 10:
        raise ValueError(f"expected worker index 0..9, got {task_id}")

    runner = source / "reliability_medmnistc" / "scripts" / "run_pipeline.py"
    trace_path = root / "logs" / ("integration-smoke.jsonl" if smoke else f"worker-{task_id}.jsonl")
    script_hashes = {
        "worker": file_hash(Path(__file__)),
        "selector": file_hash(root / "source_snapshot" / "graphcov" / "run" / "d8_ot.py"),
        "adapter": file_hash(root / "source_snapshot" / "graphcov" / "run" / "d8_registry.py"),
        "pipeline": file_hash(source / "reliability_medmnistc" / "reliability" / "pipeline.py"),
        "registry": file_hash(source / "graphcov" / "run" / "selection.py"),
    }

    cell_indices = (0,) if smoke else (task_id, task_id + 10, task_id + 20, task_id + 30)
    for cell_index in cell_indices:
        dataset = DATASETS[cell_index // len(SEEDS)]
        seed = SEEDS[cell_index % len(SEEDS)]
        cell_root = (
            root / "integration_smoke" / dataset / f"seed_{seed}"
            if smoke else root / "training" / dataset / f"seed_{seed}"
        )
        results_root = cell_root / "results"
        selection_root = cell_root / "selections"
        config_path = cell_root / "config.yaml"
        cell_root.mkdir(parents=True, exist_ok=True)

        config_path.write_text(
            "\n".join((
                f"name: d8_ot_{dataset}_seed_{seed}_{'integration_smoke' if smoke else 'formal'}",
                f"datasets: [{dataset}]",
                "methods: [d8_ot]",
                "ratios: [0.02]",
                f"selection_seed: {seed}",
                f"training_seeds: [{seed}]",
                f"epochs: {1 if smoke else 1000}",
                "dynamics_epochs: 200",
                "image_size: 224",
                "batch_size: 256",
                "num_workers: 4",
                "embedding_source: uni",
                "augment: false",
                "download_medmnist: false",
                "corruption_hash: false",
                "method_config:",
                "  d8_ot:",
                "    epsilon: 0.05",
                "    outer_iterations: 8",
                "    sinkhorn_iterations: 60",
                "checkpoint_rule: final_epoch",
                "",
            )),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.update({
            "METHODS": "d8_ot",
            "DATASETS": dataset,
            "RATIOS": "0.02",
            "SELECTION_SEED": str(seed),
            "TRAINING_SEEDS": str(seed),
            "EPOCHS": "1" if smoke else "1000",
            "DYNAMICS_EPOCHS": "200",
            "IMAGE_SIZE": "224",
            "BATCH_SIZE": "256",
            "NUM_WORKERS": "4",
            "EMBEDDING_SOURCE": "uni",
            "AUGMENT": "0",
            "DOWNLOAD_MEDMNIST": "0",
            "CORR_HASH": "0",
            "CLEAN_ONLY": "1",
            "DETERMINISTIC_TRAINING": "1",
            "TRAIN_SOURCE": "clean",
            "TRAIN_ON_CORRUPTED": "0",
            "RUN_FULL_TRAIN": "0",
            "AUTO_CONSOLIDATE": "1",
            "SMOKE_N": "0",
            "RELIABILITY_OUT": str(results_root),
            "SELECTION_OUT": str(selection_root),
            "GRAPH_CACHE": str(root / "cache"),
        })

        started = time.time()
        append_trace(trace_path, {
            "event": "cell_started",
            "utc": datetime.now(timezone.utc).isoformat(),
            "cell_index": cell_index,
            "dataset": dataset,
            "selection_seed": seed,
            "training_seed": seed,
            "config": str(config_path),
            "source_hashes": script_hashes,
        })
        subprocess.run(
            [str(python), str(runner), "--config", str(config_path), "--phase", "full"],
            cwd=source,
            env=env,
            check=True,
        )

        completion = list(results_root.rglob("run_complete.json"))
        if len(completion) != 1:
            raise RuntimeError(f"expected one completion marker, found {len(completion)} under {results_root}")
        marker = json.loads(completion[0].read_text(encoding="utf-8"))
        if marker.get("status") not in (None, "complete", "completed"):
            raise RuntimeError(f"unexpected completion marker status: {marker}")
        append_trace(trace_path, {
            "event": "cell_completed",
            "utc": datetime.now(timezone.utc).isoformat(),
            "cell_index": cell_index,
            "dataset": dataset,
            "selection_seed": seed,
            "training_seed": seed,
            "completion_marker": str(completion[0]),
            "elapsed_seconds": round(time.time() - started, 3),
        })


if __name__ == "__main__":
    main()
