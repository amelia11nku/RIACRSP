"""Feature catalogues used by the pure-Python graph builder."""

OPERATION_FEATURES = (
    "min_processing_time",
    "mean_processing_time",
    "max_processing_time",
    "num_eligible_islands",
    "dag_in_degree",
    "dag_out_degree",
    "transitive_pred_ratio",
    "transitive_succ_ratio",
    "is_scheduled",
    "is_topological_ready",
    "product_progress_ratio",
    "scheduled_start",
    "scheduled_end",
)

PRODUCT_FEATURES = (
    "num_operations",
    "scheduled_ratio",
    "remaining_workload_estimate",
    "last_actual_completion",
)

ISLAND_FEATURES = (
    "total_processing_load",
    "total_reconfiguration_time",
    "tail_completion_time",
    "num_scheduled_operations",
    "num_supported_configurations",
    "num_free_gaps",
)

W_FEATURES = (
    "num_tasks",
    "total_loaded_time",
    "total_empty_time",
    "tail_completion_time",
    "largest_free_gap",
)

F_FEATURES = (
    "num_tasks",
    "total_busy_time",
    "tail_return_time",
    "largest_free_gap",
)

OPERATION_ISLAND_EDGE_FEATURES = (
    "processing_time",
    "earliest_machine_insertion_start",
    "predicted_operation_completion",
    "setup_before_at_best_gap",
    "setup_after_at_best_gap",
    "incremental_reconfiguration_cost",
    "workpiece_loaded_distance",
    "best_w_arrival_lower_bound",
    "best_f_arrival_lower_bound",
    "predicted_sync_wait",
)
