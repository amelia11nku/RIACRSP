import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit"


def load_json(name: str) -> dict:
    return json.loads((AUDIT / name).read_text())


def test_phase6p_p0_integration_gate_passed() -> None:
    result = load_json("live_search_integration.json")
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["repository"]["expected_start_commit"] == (
        "936836445b8e7646574076645750f836a53b6d8d"
    )
    assert result["alns"]["weight_update_from_initial_one"] == {
        "accepted": 1.0,
        "best": 1.8,
        "rejected": 0.82,
    }
    assert result["phase6n"]["production_live_entry_point"] is None
    assert len(result["phase6n"]["checkpoint_grid"]) == 9
    assert result["constraints"]["r13_accessed"] is False
    assert result["constraints"]["r14_accessed"] is False


def test_phase6p_full_bank_and_multi_origin_audit() -> None:
    result = load_json("candidate_bank_semantics.json")
    assert result["status"] == "PASS"
    assert result["states"] == 864
    assert result["instances"] == 18
    assert result["requested_proposals"] == 864 * 24
    assert result["unique_targets"] == 20_441
    assert result["unique_targets_per_state_distribution"] == {
        "21": 8,
        "22": 38,
        "23": 195,
        "24": 623,
    }
    assert result["cross_destroy_duplicate_targets"] == 2
    assert result["cross_destroy_duplicate_states"] == 2
    assert result["cross_destroy_resolution"] == (
        "MEAN_CURRENT_ALNS_DESTROY_WEIGHT_ACROSS_UNIQUE_ORIGINS_REQUIRED"
    )

    with (AUDIT / "multi_origin_target_audit.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cross = [row for row in rows if row["cross_destroy_duplicate"] == "True"]
    assert len(rows) == 20_441
    assert {
        (row["state_id"], row["target_set_id"], row["all_origin_destroy_operators"])
        for row in cross
    } == {
        (
            "CB1_CAUR_S_CF1_RI2_TI2_R12_C02__seed721202__it0000742",
            "ts_1b621003d3db339813ea",
            '["overloaded_island","related"]',
        ),
        (
            "CB1_CAUR_S_CF1_RI2_TI2_R12__seed721203__it0000154",
            "ts_04c5ade60d2ec3452a03",
            '["f_bottleneck","related"]',
        ),
    }
