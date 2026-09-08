import json
from pathlib import Path

import pandas as pd

from scripts.audit_phase6o_top_selection import rank_indices, route_decision, topk_metrics


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/phase6o_neural_shortlist_v1/audit"


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for state, source, scale in (
        ("original", "ORIGINAL_PHASE6J_CAUR", "S"),
        ("new", "NEW_ALNS_EXPANSION", "L"),
    ):
        for index, (truth, prediction) in enumerate(((0.3, 0.3), (0.2, 0.2), (0.0, 0.0))):
            rows.append({
                "state_id": state,
                "target_set_id": f"target_{index}",
                "continuation_advantage_mean": truth,
                "ensemble_advantage_mean": prediction,
                "origin_family": f"origin_{index}",
                "phase6n_data_origin": source,
                "scale": scale,
                "CF_level": "CF1",
                "search_stage": "EARLY",
            })
    return pd.DataFrame(rows)


def test_phase6o_rank_tie_break_is_lexical():
    frame = pd.DataFrame({
        "target_set_id": ["target_b", "target_a"],
        "score": [1.0, 1.0],
    })
    assert list(rank_indices(frame, "score")) == [1, 0]


def test_phase6o_topk_uses_full_bank_then_shortlists():
    phase6n = _candidate_rows()
    phase6l = phase6n[phase6n.state_id.eq("original")].copy()
    result = topk_metrics(phase6n, phase6l)
    expanded = result[
        result.slice_dimension.eq("scope")
        & result.slice_value.eq("expanded_864")
        & result.k.eq(1)
    ].iloc[0]
    assert expanded.states == 2
    assert expanded.exact_best_recall == 1.0
    assert expanded.oracle_shortlist_selected_lift == 0.3
    assert expanded.mean_topk_origin_family_diversity == 1.0


def test_phase6o_route_fails_closed_when_shortlist_recall_is_weak():
    rows = []
    for scope in ("common_original_288", "expanded_864"):
        for k in (1, 2, 3, 4, 6):
            rows.append({
                "slice_dimension": "scope",
                "slice_value": scope,
                "k": k,
                "exact_best_recall": 0.4 if k == 6 else 0.1,
                "near_best_recall_epsilon_0_005": 0.5 if k == 6 else 0.2,
                "oracle_shortlist_selected_lift": 0.03 if k > 1 else 0.006,
                "fallback_beating_candidate_present": 0.95,
            })
    for scale in ("S", "M", "L"):
        for k in (1, 2, 3, 4, 6):
            rows.append({
                "slice_dimension": "scale",
                "slice_value": scale,
                "k": k,
                "exact_best_recall": 0.4,
                "near_best_recall_epsilon_0_005": 0.5,
                "oracle_shortlist_selected_lift": 0.03,
                "fallback_beating_candidate_present": 0.95,
            })
    decision = route_decision(pd.DataFrame(rows))
    assert decision["decision"] == "PROCEED_TOP_UTILITY_RETRAIN"
    assert decision["selected_shortlist_k"] is None


def test_phase6o_o0_route_is_frozen_from_complete_topk_evidence():
    result = json.loads((AUDIT / "loss_alignment_audit.json").read_text())
    topk = pd.read_csv(AUDIT / "topk_metrics.csv")
    assert result["status"] == "PASS"
    assert result["route"]["decision"] == "PROCEED_TOP_UTILITY_RETRAIN"
    assert result["route"]["selected_shortlist_k"] is None
    assert set(topk.k) == {1, 2, 3, 4, 6, 8}
    assert {"common_original_288", "expanded_864"}.issubset(set(topk.slice_value))
    assert {"S", "M", "L"}.issubset(set(topk.slice_value))
    assert {"CF1", "CF2", "CF3"}.issubset(set(topk.slice_value))
    assert result["crn_summary"]["winner_disagreement_states"] == 501
    assert result["crn_summary"]["targeted_relabeling_recommended_for_o1"] is True
    assert result["optimizer_steps_started"] is False
    assert result["live_solver_runs_started"] is False
    assert result["r13_accessed"] is False
    assert result["r14_accessed"] is False


def test_phase6o_o0_error_shift_and_protection_evidence_are_complete():
    result = json.loads((AUDIT / "loss_alignment_audit.json").read_text())
    errors = pd.read_csv(AUDIT / "top_selection_error.csv")
    shift = pd.read_csv(AUDIT / "source_shift_metrics.csv")
    protected = json.loads((AUDIT / "protected_phase6n_evidence.json").read_text())
    assert len(errors) == 864
    assert errors.state_id.nunique() == 864
    assert len(protected) == result["starting_boundary"]["protected_phase6n_files"]
    assert result["relation_block_drift_summary"]["runs_complete"] == 9
    assert result["relation_block_drift_summary"]["maximum_checkpoint_replay_error"] == 0.0
    assert set(shift.source) == {"ORIGINAL_PHASE6J_CAUR", "NEW_ALNS_EXPANSION"}
    assert {
        "candidate_origin_distribution",
        "continuation_advantage",
        "crn_disagreement",
        "fine_tuned_relation_block_prediction_drift",
        "model_error",
        "true_top_candidate_origin",
    }.issubset(set(shift.category))
