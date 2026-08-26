from __future__ import annotations

import json
from pathlib import Path

from v11.experiments.job1_select import validate_job
from v11.experiments.job2_downstream import expand_jobs


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
    derma = load("job2_derma_atomic_seed42.json")
    table = load("job2_table1_validation_3seeds.json")
    assert len(expand_jobs(derma)) == 12
    assert len(expand_jobs(table)) == 75
    for config in (derma, table):
        assert config["evaluation_split"] == "val"
        assert config["training"]["epochs"] == 1000
        assert config["training"]["size"] == 224
        assert config["training"]["augmentation"] is False


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
        frozen = job["frozen_reference_indices"]
        assert set(frozen) == {"r0p02", "r0p05"}
        for ratio_key, spec in frozen.items():
            assert spec["required"] is True
            assert spec["path"].startswith(
                "{project_root}/table1_reproduction/outputs/job1_selection/"
            )
            assert f"/{job['dataset']}/{ratio_key}/graph_a2/seed42/" in spec["path"]
