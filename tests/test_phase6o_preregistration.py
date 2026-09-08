import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
PREREG = ROOT / "outputs/phase6o_neural_shortlist_v1/preregistration"


def test_phase6o_route_b_config_freezes_top_utility_boundaries():
    config = json.loads(CONFIG.read_text())
    assert config["route"]["decision"] == "PROCEED_TOP_UTILITY_RETRAIN"
    assert config["route"]["route_a_started"] is False
    assert config["model"]["final_relation_block_trainable_primary"] is False
    assert config["training"]["source_balancing"]["ORIGINAL_PHASE6J_CAUR_aggregate_weight"] == 0.5
    assert config["training"]["source_balancing"]["NEW_ALNS_EXPANSION_aggregate_weight"] == 0.5
    assert config["training"]["objective"]["broad_all_pairs_phase6n_loss_used"] is False
    assert config["training"]["objective"]["phase6n_standard_z_listnet_used"] is False
    assert config["training"]["inner_selection"]["single_validation_fold_selection_forbidden"] is True
    assert config["locks"]["r13_accessed"] is False
    assert config["locks"]["r14_accessed"] is False


def test_phase6o_targeted_relabel_union_is_fixed_and_inside_full_bank():
    plan = json.loads((PREREG / "targeted_relabeling_plan.json").read_text())
    union = pd.read_csv(PREREG / "targeted_relabel_candidate_union.csv")
    source = pd.read_parquet(
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
    )
    assert plan["status"] == "FROZEN_BEFORE_ADDITIONAL_CONTINUATION"
    assert len(union) == 4786
    assert union.state_id.nunique() == 864
    assert len(union) * 3 == plan["actual_additional_continuation_rows"] == 14358
    assert union.groupby("state_id").is_fallback.sum().eq(1).all()
    assert set(zip(union.state_id, union.target_set_id)).issubset(
        set(zip(source.state_id, source.target_set_id))
    )


def test_phase6o_preregistration_is_fail_closed_before_new_outcomes():
    prereg = json.loads((PREREG / "preregistration.json").read_text())
    assert prereg["status"] == "FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP"
    assert prereg["route"] == "PROCEED_TOP_UTILITY_RETRAIN"
    assert prereg["additional_outcomes_started"] is False
    assert prereg["optimizer_steps_started"] is False
    assert prereg["live_solver_runs_started"] is False
    assert prereg["historical_score_calls"] == 0
    assert prereg["gurobi_run"] is False
    assert prereg["r13_accessed"] is False
    assert prereg["r14_accessed"] is False
