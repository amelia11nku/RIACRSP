from __future__ import annotations

from scripts.run_paper_gurobi_validation import (
    CONFIG_PATH,
    ROOT,
    instance_characteristics,
    load_json,
    verify_protocol,
)
from rcias_clgri.data.loader import load_instance


def test_paper_core_small_protocol_is_complete_and_frozen():
    config = load_json(CONFIG_PATH)
    verify_protocol(config)
    assert len(config["runs"]) == 15
    assert {case["cf_level"] for case in config["runs"]} == {"CF1", "CF2", "CF3"}
    assert {case["replicate"] for case in config["runs"]} == {
        "R01", "R02", "R03", "R04", "R05"
    }
    assert config["gurobi"] == {
        "model": "rcias_clgri/exact/general_gurobi.py",
        "objective": "makespan",
        "time_limit_seconds": 3600.0,
        "seed": 671501,
        "threads": 1,
        "mip_gap": 0.0,
        "use_h1_mip_start": True,
    }


def test_paper_core_small_inventory_characteristics_are_valid():
    config = load_json(CONFIG_PATH)
    for case in config["runs"]:
        metrics = instance_characteristics(load_instance(ROOT / case["relative_path"]))
        assert metrics["n_products"] > 0
        assert metrics["n_operations"] > 0
        assert metrics["n_islands"] > 0
        assert 0.0 <= metrics["precedence_density"] <= 1.0
        assert 0.0 < metrics["eligibility_fraction_mean"] <= 1.0
