import json
from pathlib import Path

from scripts import train_phase6o_top_utility as training


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/phase6o_neural_shortlist_v1/training/training_protocol.json"


def protocol():
    return json.loads(PROTOCOL.read_text())


def test_phase6o_training_protocol_is_frozen_before_optimizer():
    value = training.validate_protocol()
    assert value["status"] == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP"
    assert value["optimizer_steps_started"] is False
    assert value["outer_oof_generated"] is False
    assert value["model"]["family"] == "O1_TOP_UTILITY_FROZEN_ENCODER"
    assert value["model"]["trainable_parameters"] == 667838
    assert value["model"]["frozen_encoder_parameters"] == 5098624


def test_phase6o_training_protocol_uses_symmetric_isolated_inner_folds():
    value = protocol()
    assert value["symmetric_two_way_inner_validation"] is True
    for outer in value["fold_audit"]:
        assert outer["held_training_instance_overlap"] == []
        directions = outer["symmetric_inner_directions"]
        assert len(directions) == 2
        assert directions[0]["training_fold"] == directions[1]["validation_fold"]
        assert directions[0]["validation_fold"] == directions[1]["training_fold"]
        assert all(direction["instance_overlap"] == [] for direction in directions)


def test_phase6o_training_protocol_balances_sources_and_uses_top_loss():
    value = protocol()
    for outer in value["fold_audit"]:
        for contract in [outer["outer_contract"], *[
            direction["contract"] for direction in outer["symmetric_inner_directions"]
        ]]:
            assert abs(contract["source_weight_totals"]["ORIGINAL_PHASE6J_CAUR"] - 0.5) < 1e-12
            assert abs(contract["source_weight_totals"]["NEW_ALNS_EXPANSION"] - 0.5) < 1e-12
            assert contract["retained_pair_count"] > 0
    objective = value["training"]["objective"]
    assert objective["broad_all_pairs_phase6n_loss_used"] is False
    assert objective["phase6n_standard_z_listnet_used"] is False
    assert value["training"]["objective_weights"] == {
        "top_set_cross_entropy": 1.0,
        "near_best_vs_rest_regret_logistic": 1.0,
        "soft_expected_utility": 1.0,
        "advantage_huber": 0.1,
        "beats_fallback_bce": 0.1,
    }


def test_phase6o_training_protocol_preserves_holdout_and_score_locks():
    value = protocol()
    assert value["historical_score_calls"] == 0
    assert value["gurobi_run"] is False
    assert value["r13_accessed"] is False
    assert value["r14_accessed"] is False
