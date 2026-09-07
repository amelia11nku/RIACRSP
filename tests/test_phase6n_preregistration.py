import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
PREREG = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration"


def _config():
    return json.loads(CONFIG.read_text())


def test_phase6n_has_exactly_one_promotable_primary_and_locked_boundaries():
    value = _config()
    assert value["primary_family"] == "N1_CANDIDATE_CONDITIONED_CSG"
    assert value["promotable_families"] == [value["primary_family"]]
    boundary = value["boundaries"]
    assert boundary["candidate_generator_rules"] == 24
    assert boundary["historical_score_primary_input_calls"] == 0
    assert boundary["historical_score_deployable_runtime_calls"] == 0
    assert boundary["heteroscedastic_scale_head"] is False
    assert boundary["immediate_utility_hard_gate"] is False
    assert boundary["r13"].startswith("LOCKED")
    assert boundary["r14"].startswith("LOCKED")


def test_phase6n_data_expansion_is_equal_and_outcome_independent():
    data = _config()["data_generation"]
    assert data["instances"] == 18
    assert data["new_source_trajectories"] == 72
    assert data["new_states"] == 576
    assert data["total_states_per_instance"] == 48
    assert data["total_states"] == 864
    assert len(data["new_trajectory_seeds"]) == 4
    assert len(set(data["new_trajectory_seeds"])) == 4
    assert len(data["progress_anchors"]) == 8
    assert data["continuation_horizon"] == 4
    assert len(data["continuation_crn_seeds"]) == 2
    assert "quality cannot change" in data["outcome_independent_stop_rule"]


def test_phase6n_model_is_one_pass_candidate_conditioned_and_scale_free():
    model = _config()["primary_model"]
    assert model["graph_encoder"]["execution"] == "once per state"
    assert model["graph_encoder"]["trainable"] == ["relation block 1"]
    assert len(model["relation_boundary"]["families"]) == 6
    assert model["relation_boundary"]["direction_channels"] == [
        "outgoing_from_target",
        "incoming_to_target",
    ]
    assert len(model["critical_conditioning"]["sets"]) == 3
    assert model["excluded_heads"] == [
        "immediate utility",
        "heteroscedastic scale",
        "winner-only selector",
    ]


def test_phase6n_whole_instance_and_empirical_calibration_are_frozen():
    value = _config()
    folds = value["structural_folds"]
    assert folds["candidate_row_split_forbidden"] is True
    assert folds["outer_held_instances"] == 6
    assert folds["outer_training_instances"] == 12
    assert folds["outer_held_states_after_expansion"] == 288
    calibration = value["residual_calibration"]
    assert calibration["one_sided_quantile"] == 0.9
    assert calibration["fit_scope"].startswith("inner-validation")
    assert "frozen before outer held" in calibration["outer_fold_isolation"]


def test_phase6n_direct_gate_excludes_phase6m_scale_and_immediate_gate():
    gate = _config()["direct_decision"]
    assert gate["probability_threshold"] == 0.55
    assert gate["phase6m_selector"] is False
    assert gate["immediate_utility_gate"] is False
    assert gate["distance_support_gate"] is False
    assert gate["fallback_on_failure"] == "canonical operator_related"


def test_phase6n_preregistration_was_frozen_before_execution():
    value = json.loads((PREREG / "preregistration.json").read_text())
    plan = json.loads((PREREG / "data_generation_plan.json").read_text())
    assert value["status"] == "FROZEN_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP"
    assert value["new_rollouts_started"] is False
    assert value["optimizer_steps_started"] is False
    assert value["outer_oof_generated"] is False
    assert value["r13_accessed"] is False
    assert value["r14_accessed"] is False
    assert plan["status"] == "FROZEN_BEFORE_OUTCOME_GENERATION"
    assert plan["new_states"] == 576
    assert plan["combined_states"] == 864
