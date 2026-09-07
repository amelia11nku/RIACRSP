import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "outputs/phase6m_selective_confidence_v1/quality"


def test_phase6m_m4_quality_is_terminal_and_complete():
    result = json.loads((QUALITY / "development_quality.json").read_text())
    assert result["status"] == "M4_COMPLETE"
    assert result["decision"] == "MODEL_REVISION_QUALITY"
    assert result["eligible_gate_count"] == 0
    assert result["selected_gate"] is None
    assert all(result["raw_quality_checks"].values())
    assert not all(result["formal_intervention_readiness_checks"].values())
    assert result["runtime_stage"] == "NOT_RUN_STOPPED_ON_M4_QUALITY"
    assert result["solver_gate"] == "NOT_RUN_QUALITY_GATE_FAILED"
    assert result["r13_accessed"] is False
    assert result["r14_accessed"] is False


def test_phase6m_gate_grid_and_support_are_complete():
    result = json.loads((QUALITY / "development_quality.json").read_text())
    grid = pd.read_csv(QUALITY / "gate_grid.csv")
    support = pd.read_csv(QUALITY / "support_by_regime.csv")
    risk = pd.read_csv(QUALITY / "selective_risk_curve.csv")
    assert len(grid) == 18
    assert int(grid["retained"].sum()) == 0
    assert int(grid["interventions"].max()) == 0
    assert int(grid["lower_bound_pass"].max()) == 0
    assert result["support"]["candidate_support_rate"] > 0.98
    assert result["support"]["selected_winner_support_rate"] == 1.0
    assert result["support"]["numeric_boundary_failures"] == 55
    assert {"overall", "scale", "CF_level", "search_stage"}.issubset(set(support.dimension))
    assert {"reliability", "selective_coverage", "probability_threshold"} == set(risk.curve_type)


def test_phase6m_raw_ranker_exactly_matches_phase6l():
    result = json.loads((QUALITY / "development_quality.json").read_text())
    assert result["raw_ranking"]["exactly_reproduces_frozen_phase6l"] is True
    assert result["raw_ranking"]["state_count"] == 288
    assert result["raw_ranking"]["action_count"] == 6809
    # Predictions/state metrics are identical. Each phase keeps its own
    # preregistered bootstrap seed, so interval endpoints are audited separately.
    for metric in (
        "overall_spearman", "pairwise_accuracy", "ndcg_at_1",
        "selected_lift", "selection_regret",
    ):
        value = result["comparison"]["phase6l_score_free"]["raw_metrics"][metric]
        assert result["raw_ranking"][metric] == value
    assert result["raw_ranking"]["selected_lift_lcb"] > 0.0
    assert result["comparison"]["phase6l_score_free"]["raw_metrics"]["selected_lift_lcb"] > 0.0
