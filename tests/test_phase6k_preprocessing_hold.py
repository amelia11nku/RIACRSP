import copy

import pytest

from scripts.finalize_phase6k_preprocessing_hold import audit_preprocessing_report


def report():
    keys = ("bank_identity_and_order_exact", "score_values_exact", "features_exact", "fallback_exact")
    rows = [{"state_id": str(i), "candidate_rows": 24 if i < 185 else 23,
             **{k: True for k in keys}, "feature_checks": {"normalized_frozen_score_rank": True}}
            for i in range(288)]
    rows[0]["score_values_exact"] = False
    return {"rows": rows, "checks": {k: all(r[k] for r in rows) for k in keys},
            "mismatch_states": {k: sum(not r[k] for r in rows) for k in keys},
            "r13_accessed": False, "r14_accessed": False}


def test_single_score_mismatch_is_not_hidden_by_equal_actions_or_features():
    checks, counts = audit_preprocessing_report(report())
    assert checks["features_exact"] and not checks["score_values_exact"]
    assert counts["score_values_exact"] == 1


@pytest.mark.parametrize("corruption", ["duplicate", "candidate", "summary", "feature", "access"])
def test_incomplete_or_inconsistent_equivalence_cannot_be_finalized(corruption):
    value = copy.deepcopy(report())
    if corruption == "duplicate":
        value["rows"][1]["state_id"] = "0"
    elif corruption == "candidate":
        value["rows"][0]["candidate_rows"] -= 1
    elif corruption == "summary":
        value["checks"]["score_values_exact"] = True
    elif corruption == "feature":
        value["rows"][0]["feature_checks"]["normalized_frozen_score_rank"] = False
    else:
        value["r13_accessed"] = True
    with pytest.raises(ValueError):
        audit_preprocessing_report(value)
