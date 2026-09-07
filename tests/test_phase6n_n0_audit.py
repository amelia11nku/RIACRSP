import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.audit_phase6n_architecture_data import best_so_far_behavior_audit


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/audit"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_phase6n_n0_audit_has_frozen_scope_and_no_holdout_access():
    value = json.loads((AUDIT / "architecture_data_audit.json").read_text())
    assert value["status"] == "N0_COMPLETE"
    assert value["decision"] == "PROCEED_TO_PREREGISTRATION"
    assert value["data_integrity"]["candidate_rows"] == 6809
    assert value["data_integrity"]["states"] == 288
    assert value["data_integrity"]["instances"] == 18
    assert value["data_integrity"]["candidate_identity_order_preserved"] is True
    assert value["data_integrity"]["canonical_fallback_exactly_one_per_state"] is True
    assert value["historical_score_online_forward_calls"] == 0
    assert value["new_continuation_rollouts"] is False
    assert value["optimizer_steps_started"] is False
    assert value["r13_accessed"] is False
    assert value["r14_accessed"] is False
    assert not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists()
    assert not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists()


def test_phase6n_n0_tables_are_complete_and_not_selected_examples_only():
    immediate = pd.read_csv(AUDIT / "immediate_vs_continuation.csv")
    collision = pd.read_csv(AUDIT / "candidate_representation_collision.csv")
    assert len(immediate) == 6809
    assert immediate[["state_id", "target_set_id"]].duplicated().sum() == 0
    assert len(collision) > 0
    assert collision.anchor_target_set_id.ne(collision.neighbor_target_set_id).all()
    assert collision.cheap_feature_rms_distance.ge(0).all()
    assert collision.target_operation_jaccard.between(0, 1).all()


def test_phase6n_predecessor_manifest_still_matches_phase6l_phase6m_files():
    manifest = json.loads(
        (AUDIT / "protected_phase6l_phase6m_evidence.json").read_text()
    )
    assert len(manifest) >= 180
    for relative, expected in manifest.items():
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size == expected["bytes"]
        assert _digest(path) == expected["sha256"]


def test_phase6n_best_so_far_is_monotone_even_when_worse_current_is_accepted():
    result = best_so_far_behavior_audit()
    assert result["status"] == "PASS"
    assert result["accepted_worse_current_events"] > 0
    assert all(result["checks"].values())
