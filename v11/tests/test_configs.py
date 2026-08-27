from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v11.experiments.job1_select import load_frozen_reference, validate_job
from v11.methods.selection import MarginResult


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "v11" / "configs"


def load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_all_job1_dependencies_are_ordered():
    for name in ("job1_derma_atomic.json", "job1_table1_calibration.json"):
        config = load(name)
        assert config["schema"] == "graphcov-v11/job1-config-v1"
        for job in config["jobs"]:
            validate_job(job)


def test_job2_counts_and_protocol():
    from v11.experiments.job2_downstream import expand_jobs, resolve_evaluation_split

    derma = load("job2_derma_atomic_seed42.json")
    table = load("job2_table1_validation_3seeds.json")
    assert len(expand_jobs(derma)) == 12
    assert len(expand_jobs(table)) == 75
    for config in (derma, table):
        assert config["evaluation_split"] == "val"
        assert config["training"]["epochs"] == 1000
        assert config["training"]["size"] == 224
        assert config["training"]["augmentation"] is False
    assert resolve_evaluation_split(table, None) == "val"


def test_job2_requires_explicit_test_authorization():
    import pytest

    from v11.experiments.job2_downstream import resolve_evaluation_split

    test_config = {"evaluation_split": "test"}
    assert resolve_evaluation_split(test_config, "test") == "test"
    with pytest.raises(ValueError, match="--evaluation-split test"):
        resolve_evaluation_split(test_config, None)


def test_table1_has_dataset_specific_targets():
    config = load("job1_table1_calibration.json")
    targets = {job["dataset"]: job["calibration_target"] for job in config["jobs"]}
    assert targets["organamnist"] == {"k": 20, "hops": 4, "max_degree": 70}
    assert targets["pathmnist"] == {"k": 20, "hops": 3, "max_degree": 70}
    assert targets["tissuemnist"] == {"k": 15, "hops": 3, "max_degree": 50}


def test_table1_all_reference_indices_are_required_and_isolated():
    config = load("job1_table1_calibration.json")
    assert len(config["jobs"]) == 5
    for job in config["jobs"]:
        assert job["reference_mode"] == "frozen"
        frozen = job["frozen_reference_indices"]
        assert set(frozen) == {"r0p02", "r0p05"}
        for ratio_key, spec in frozen.items():
            assert spec["required"] is True
            assert spec["path"].startswith(
                "{project_root}/table1_reproduction/outputs/job1_selection/"
            )
            assert f"/{job['dataset']}/{ratio_key}/graph_a2/seed42/" in spec["path"]


def test_frozen_reference_is_reused_exactly(tmp_path):
    run_dir = tmp_path / "table1" / "r0p02" / "graph_a2" / "seed42"
    run_dir.mkdir(parents=True)
    selected = np.asarray([3, 0, 2, 1], dtype=np.int64)
    order = np.asarray([0, 3, 1, 2], dtype=np.int64)
    np.save(run_dir / "selected_indices.npy", selected)
    np.save(run_dir / "selection_order.npy", order)
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int64)
    margins = MarginResult(
        same_distance=np.ones(6, dtype=np.float32),
        different_distance=np.ones(6, dtype=np.float32),
        ratio=np.ones(6, dtype=np.float32),
        gap=np.zeros(6, dtype=np.float32),
        unsafe=np.asarray([False, True, False, False, True, False]),
    )

    result, audit = load_frozen_reference(
        {"path": str(run_dir / "selected_indices.npy"), "required": True},
        labels=labels,
        margins=margins,
        budget=2,
        project_root=tmp_path,
        config_dir=tmp_path,
    )

    assert np.array_equal(result.indices, selected)
    assert np.array_equal(result.order, order)
    assert audit["status"] == "reused_frozen"
    assert audit["direct_reuse"] is True
    assert audit["exact_array_match"] is True
