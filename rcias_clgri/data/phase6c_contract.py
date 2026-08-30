"""Frozen Phase 6C NI dataset field and task contract."""

from __future__ import annotations

MODEL_INPUT_ALLOWED = "MODEL_INPUT_ALLOWED"
LABEL_ONLY = "LABEL_ONLY"
IDENTIFIER_ONLY = "IDENTIFIER_ONLY"
ANALYSIS_ONLY = "ANALYSIS_ONLY"
FORBIDDEN_FUTURE_INFORMATION = "FORBIDDEN_FUTURE_INFORMATION"

STATE_FIELDS = {
    "instance_id": IDENTIFIER_ONLY,
    "instance_relative_path": IDENTIFIER_ONLY,
    "training_split": ANALYSIS_ONLY,
    "scale": MODEL_INPUT_ALLOWED,
    "CF_level": MODEL_INPUT_ALLOWED,
    "RI_level": MODEL_INPUT_ALLOWED,
    "TI_level": MODEL_INPUT_ALLOWED,
    "replicate": ANALYSIS_ONLY,
    "trajectory_run": IDENTIFIER_ONLY,
    "trajectory_seed": IDENTIFIER_ONLY,
    "state_sampling_seed": IDENTIFIER_ONLY,
    "state_id": IDENTIFIER_ONLY,
    "search_stage": MODEL_INPUT_ALLOWED,
    "search_progress": MODEL_INPUT_ALLOWED,
    "elapsed_time": ANALYSIS_ONLY,
    "iteration": ANALYSIS_ONLY,
    "current_makespan": MODEL_INPUT_ALLOWED,
    "current_candidate": IDENTIFIER_ONLY,
    "candidate_sha256": IDENTIFIER_ONLY,
    "bottleneck_proxy": MODEL_INPUT_ALLOWED,
    "trajectory_destroy_operator": ANALYSIS_ONLY,
    "trajectory_repair_operator": ANALYSIS_ONLY,
    "requested_arm_count": ANALYSIS_ONLY,
    "unique_arm_count": ANALYSIS_ONLY,
    "duplicate_arm_count": ANALYSIS_ONLY,
    "run_id": IDENTIFIER_ONLY,
    "suite": ANALYSIS_ONLY,
    "seed": IDENTIFIER_ONLY,
    "historical_best_makespan": ANALYSIS_ONLY,
    "temperature_before": ANALYSIS_ONLY,
    "operator_weights_before": ANALYSIS_ONLY,
    "historical_best_candidate": IDENTIFIER_ONLY,
}

RAW_OUTCOME_FIELDS = {
    "instance_id": IDENTIFIER_ONLY, "training_split": ANALYSIS_ONLY, "state_id": IDENTIFIER_ONLY,
    "target_set_id": IDENTIFIER_ONLY, "repair_seed_group": ANALYSIS_ONLY, "repair_seed": IDENTIFIER_ONLY,
    "repair_operator": ANALYSIS_ONLY, "candidate_trials": ANALYSIS_ONLY, "destroy_count": ANALYSIS_ONLY,
    "counterfactual_makespan": LABEL_ONLY, "absolute_improvement": LABEL_ONLY,
    "relative_improvement": LABEL_ONLY, "improved": LABEL_ONLY, "decoder_evaluations": ANALYSIS_ONLY,
}

AGGREGATE_FIELDS = {
    "instance_id": IDENTIFIER_ONLY, "training_split": ANALYSIS_ONLY, "state_id": IDENTIFIER_ONLY,
    "scale": MODEL_INPUT_ALLOWED, "CF_level": MODEL_INPUT_ALLOWED, "RI_level": MODEL_INPUT_ALLOWED,
    "TI_level": MODEL_INPUT_ALLOWED, "search_stage": MODEL_INPUT_ALLOWED,
    "bottleneck_proxy": MODEL_INPUT_ALLOWED, "current_makespan": MODEL_INPUT_ALLOWED,
    "target_set_id": IDENTIFIER_ONLY, "arm_family": MODEL_INPUT_ALLOWED,
    "origin_destroy_operator": MODEL_INPUT_ALLOWED, "origin_rules": MODEL_INPUT_ALLOWED,
    "origin_families": MODEL_INPUT_ALLOWED, "destroy_count": ANALYSIS_ONLY,
    "destroyed_operation_ids": IDENTIFIER_ONLY, "repair_operator": ANALYSIS_ONLY,
    "repair_seed_count": ANALYSIS_ONLY, "mean_counterfactual_makespan": LABEL_ONLY,
    "median_counterfactual_makespan": LABEL_ONLY, "mean_absolute_improvement": LABEL_ONLY,
    "mean_relative_improvement": LABEL_ONLY, "median_relative_improvement": LABEL_ONLY,
    "std_relative_improvement": LABEL_ONLY, "improvement_probability": LABEL_ONLY,
    "positive_seed_count": LABEL_ONLY, "positive_under_1_of_3": LABEL_ONLY,
    "positive_under_2_of_3": LABEL_ONLY, "positive_under_3_of_3": LABEL_ONLY,
    "rank_within_state": LABEL_ONLY, "rank_percentile": LABEL_ONLY, "top1": LABEL_ONLY,
    "top3": LABEL_ONLY, "top5": LABEL_ONLY, "regret_to_best": LABEL_ONLY,
    "robust_regret_to_best": LABEL_ONLY,
}

OPERATION_FEATURES = {
    "product_id", "assigned_island", "required_configuration", "processing_time",
    "operation_start", "operation_end", "operation_slack", "is_on_processing_critical_path",
    "is_on_resource_critical_chain", "criticality_score", "number_of_predecessors",
    "number_of_successors", "eligible_island_count", "island_load_before", "island_relative_load",
    "preceding_reconfiguration_time", "following_reconfiguration_time",
    "local_reconfiguration_contribution", "W_waiting_or_delay_contribution",
    "F_waiting_or_delay_contribution", "synchronization_wait_contribution", "position_in_product",
    "position_in_island_chain", "position_in_W_chain", "position_in_F_chain",
}
MEMBERSHIP_FIELDS = {
    "instance_id": IDENTIFIER_ONLY, "training_split": ANALYSIS_ONLY, "state_id": IDENTIFIER_ONLY,
    "target_set_id": IDENTIFIER_ONLY, "operation_id": IDENTIFIER_ONLY, "is_targeted": MODEL_INPUT_ALLOWED,
    "target_set_mean_relative_improvement": LABEL_ONLY,
    "target_set_improvement_probability": LABEL_ONLY, "target_set_rank": LABEL_ONLY,
    **{field: MODEL_INPUT_ALLOWED for field in OPERATION_FEATURES},
}

PAIR_FIELDS = {
    "instance_id": IDENTIFIER_ONLY, "training_split": ANALYSIS_ONLY, "state_id": IDENTIFIER_ONLY,
    "pair_rule": MODEL_INPUT_ALLOWED, "reference_target_set_id": IDENTIFIER_ONLY,
    "perturbed_target_set_id": IDENTIFIER_ONLY, "removed_operations": IDENTIFIER_ONLY,
    "added_operations": IDENTIFIER_ONLY, "destroy_count": ANALYSIS_ONLY,
    "repair_operator": ANALYSIS_ONLY, "reference_mean_relative_improvement": LABEL_ONLY,
    "perturbed_mean_relative_improvement": LABEL_ONLY, "mean_gain_difference": LABEL_ONLY,
    "pairwise_preference": LABEL_ONLY,
}

TABLE_FIELDS = {
    "states": STATE_FIELDS,
    "repair_seed_outcomes": RAW_OUTCOME_FIELDS,
    "target_set_aggregates": AGGREGATE_FIELDS,
    "target_membership": MEMBERSHIP_FIELDS,
    "operation_pairs": PAIR_FIELDS,
}

FORBIDDEN_FIELDS = (
    "future_best_makespan", "future_window_improvement", "trajectory_final_makespan",
    "accepted_after_action", "rank_from_future_trajectory", "after_state_candidate",
)

SUPPORTED_TASKS = (
    "set_level_improvement_classification",
    "within_state_target_set_ranking",
    "top_k_target_set_retrieval",
    "secondary_operator_selection",
    "conditional_operation_pair_preference",
)
