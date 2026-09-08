from pathlib import Path
import inspect

import pandas as pd
import pytest

from scripts import run_phase6o_targeted_relabeling as relabel


ROOT = Path(__file__).resolve().parents[1]


def test_phase6o_relabel_boundary_is_frozen_and_complete():
    config, phase6j_config, plan, union, preregistration_sha256 = relabel.validate_boundary()
    assert config["route"]["decision"] == "PROCEED_TOP_UTILITY_RETRAIN"
    assert plan["status"] == "FROZEN_BEFORE_ADDITIONAL_CONTINUATION"
    assert len(union) == 4786
    assert union.state_id.nunique() == 864
    assert plan["actual_additional_continuation_rows"] == 14358
    assert phase6j_config["rng"]["proposal_namespace"] == 692000000
    assert plan["repair_seed_namespace"] == phase6j_config["rng"]["repair_namespace"]
    assert plan["continuation_seed_namespace"] == phase6j_config["rng"]["continuation_namespace"]
    assert len(preregistration_sha256) == 64


def test_phase6o_relabel_progress_accounts_for_every_frozen_state():
    statuses = [
        {"elapsed_seconds": 4.0, "candidate_count": 5, "additional_rows": 15},
        {"elapsed_seconds": 6.0, "candidate_count": 7, "additional_rows": 21},
    ]
    progress = relabel.progress_payload(statuses, 0.0, "RUNNING")
    assert progress["states_complete"] == 2
    assert progress["states_expected"] == 864
    assert progress["state_candidate_rows_complete"] == 12
    assert progress["state_candidate_rows_expected"] == 4786
    assert progress["additional_rows_complete"] == 36
    assert progress["additional_rows_expected"] == 14358
    assert progress["measured_seconds_per_state"] == 5.0
    assert progress["historical_score_calls"] == 0


def test_phase6o_relabel_worker_has_no_historical_scorer_dependency():
    source = Path(relabel.__file__).read_text()
    assert "score_frozen_candidate_bank" not in source
    assert "FrozenLiveInference" not in source
    assert "load_policy" not in source
    assert "phase6j_config" in inspect.signature(relabel.collect_state).parameters


def test_phase6o_union_keeps_one_fallback_and_never_replaces_full_bank():
    union = pd.read_csv(relabel.UNION)
    source = pd.read_parquet(relabel.PHASE6N_GROUPED)
    assert union.groupby("state_id").is_fallback.sum().eq(1).all()
    assert source.groupby("state_id").requested_bank_count.first().eq(24).all()
    assert source.groupby("state_id").size().between(21, 24).all()
    assert union.groupby("state_id").size().between(4, 9).all()
    assert (union.groupby("state_id").size() < source.groupby("state_id").size()).all()


def test_phase6o_fallback_reference_accepts_only_registered_reanchoring():
    assert relabel.validate_fallback_reference(
        source="ORIGINAL_PHASE6J_CAUR",
        state_id="known",
        canonical_fallback_id="canonical",
        frozen_union_fallback_ids=["canonical"],
        historical_replay_fallback_id="historical",
        known_reanchoring={"known": ("historical", "canonical")},
    )
    assert not relabel.validate_fallback_reference(
        source="NEW_ALNS_EXPANSION",
        state_id="same",
        canonical_fallback_id="canonical",
        frozen_union_fallback_ids=["canonical"],
        historical_replay_fallback_id="canonical",
        known_reanchoring={},
    )
    with pytest.raises(RuntimeError, match="unregistered"):
        relabel.validate_fallback_reference(
            source="ORIGINAL_PHASE6J_CAUR",
            state_id="unknown",
            canonical_fallback_id="canonical",
            frozen_union_fallback_ids=["canonical"],
            historical_replay_fallback_id="historical",
            known_reanchoring={},
        )
    with pytest.raises(RuntimeError, match="new-state"):
        relabel.validate_fallback_reference(
            source="NEW_ALNS_EXPANSION",
            state_id="new",
            canonical_fallback_id="canonical",
            frozen_union_fallback_ids=["canonical"],
            historical_replay_fallback_id="historical",
            known_reanchoring={"new": ("historical", "canonical")},
        )


def test_phase6o_recovery_diagnosis_matches_frozen_phase6l_reanchoring():
    diagnosis = relabel.load_json(relabel.RECOVERY_DIAGNOSIS)
    assert diagnosis["completed_states"] == 288
    assert diagnosis["failed_state_raw_or_status_exists"] is False
    assert len(diagnosis["known_mismatches"]) == 10
    assert {row["scale"] for row in diagnosis["known_mismatches"]} == {"M", "S"}
    assert diagnosis["completed_states_with_historical_vs_canonical_fallback_mismatch"] == 0
