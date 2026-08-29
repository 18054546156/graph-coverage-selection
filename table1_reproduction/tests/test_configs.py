from __future__ import annotations

import json
from pathlib import Path

from table1_reproduction.experiments.job1_select import selection_seeds
from table1_reproduction.experiments.job2_downstream import expand_jobs, selection_seed_for
from table1_reproduction.official_runtime import verify_official_vendor


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_official_vendor_snapshot_is_intact():
    verify_official_vendor()


def test_table1_methods_and_protocol_are_explicit():
    config = load("job1_table1.json")
    assert config["ratios"] == [0.02, 0.05]
    assert config["training_seeds"] == [42, 43, 44, 45, 46]
    assert list(config["methods"]) == [
        "random", "el2n", "forgetting", "eva",
        "facility", "fps", "herding", "graph_a2",
    ]
    assert config["selection"]["k_neighbors"] == 50
    assert config["selection"]["k_hops"] == 2
    assert config["selection"]["global_selection"] is False
    assert config["methods"]["graph_a2"]["global_selection"] is True
    assert config["methods"]["graph_a2"]["k_neighbors"] == 50
    assert config["methods"]["facility"]["global_selection"] is False
    assert config["dynamics"] == {"size": 28, "epochs": 200, "window_size": 10, "seed": 42}


def test_selection_seed_policy_matches_official_runner():
    config = load("job1_table1.json")
    assert selection_seeds(config["methods"]["random"], config) == [42, 43, 44, 45, 46]
    assert selection_seeds(config["methods"]["fps"], config) == [42, 43, 44, 45, 46]
    assert selection_seeds(config["methods"]["graph_a2"], config) == [42]

    job2 = load("job2_table1.json")
    assert selection_seed_for("random", 45, job2) == 45
    assert selection_seed_for("fps", 44, job2) == 44
    assert selection_seed_for("graph_a2", 44, job2) == 42


def test_job2_has_exact_table1_workload():
    config = load("job2_table1.json")
    jobs = expand_jobs(config)
    assert len(jobs) == 5 * 2 * 8 * 5
    assert len({(job["dataset"], job["ratio"], job["method"]) for job in jobs}) == 80
    assert config["validation_split"] == "val"
    assert config["evaluation_split"] == "test"
    assert config["selection_root"].endswith("job1_selection_k50_global")
    assert config["output_root"].endswith("job2_table1_k50_global_valckpt")
    assert config["training"]["evaluate_every"] == 10


def test_graph_a2_three_seed_reproduction_workload():
    config = load("job2_graph_a2_3seeds.json")
    jobs = expand_jobs(config)
    assert config["evaluation_split"] == "test"
    assert config["methods"] == ["graph_a2"]
    assert config["training_seeds"] == [42, 43, 44]
    assert len(jobs) == 5 * 2 * 1 * 3
    assert all(selection_seed_for("graph_a2", seed, config) == 42 for seed in (42, 43, 44))


def test_job1_expected_selection_workload():
    config = load("job1_table1.json")
    count = 0
    for method_cfg in config["methods"].values():
        count += len(config["datasets"]) * len(config["ratios"]) * len(selection_seeds(method_cfg, config))
    assert count == 160


def test_extra_methods_are_not_main_table_methods():
    main = set(load("job1_table1.json")["methods"])
    extra = set(load("job1_extra_diagnostics.json")["methods"])
    assert extra == {"heat_kernel", "graph_coverage"}
    assert main.isdisjoint(extra)
