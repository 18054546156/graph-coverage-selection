from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_configs_have_frozen_protocol_fields():
    for path in (ROOT / "configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["selection_seed"] == 42
        assert config["image_size"] == 224
        assert config["dynamics_epochs"] == 200
        assert config["checkpoint_rule"] == "final_epoch"
        assert 0.01 in config["ratios"] or 0.02 in config["ratios"]


def test_graph_a2_parameters_are_explicit():
    config = yaml.safe_load(
        (ROOT / "configs" / "full_5datasets_8methods.yaml").read_text(encoding="utf-8")
    )
    graph = config["method_config"]["graph_a2"]
    assert graph == {
        "global_selection": True,
        "k_neighbors": 50,
        "k_hops": 2,
        "kernel": "A_sym_plus_A_sym_squared",
    }
