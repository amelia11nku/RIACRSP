"""Performance-independent structural metrics for RCIAS benchmark instances."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev

from rcias_clgri.data.instance import Instance


def benchmark_family(instance_id: str) -> str:
    if instance_id.startswith("BR_"):
        return "Brandimarte"
    if instance_id.startswith("HU_E_"):
        return "Hurink E"
    if instance_id.startswith("HU_R_"):
        return "Hurink R"
    if instance_id.startswith("HU_V_"):
        return "Hurink V"
    return "Unknown"


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probabilities = [count / total for count in counts if count]
    return -sum(p * math.log(p) for p in probabilities) / math.log(len(counts))


def _ready_set_proxy(instance: Instance) -> tuple[float, int]:
    completed: set[str] = set()
    sizes = []
    while len(completed) < instance.num_operations:
        ready = sorted(
            operation
            for operation in instance.operations
            if operation not in completed
            and instance.predecessors[operation].issubset(completed)
        )
        if not ready:
            raise RuntimeError("precedence-only simulation found no ready operation")
        sizes.append(len(ready))
        completed.add(ready[0])
    return mean(sizes), max(sizes)


def benchmark_metrics(instance: Instance) -> dict[str, object]:
    islands = max(1, len(instance.islands))
    configs = max(1, len(instance.configurations))
    route = [
        len(instance.operation_data[operation].eligible_islands) / islands
        for operation in instance.operations
    ]
    capability = [
        len(instance.island_data[island].supported_configs) / configs
        for island in instance.islands
    ]
    selectivity = [
        sum(
            configuration in instance.island_data[island].supported_configs
            for island in instance.islands
        ) / islands
        for configuration in instance.configurations
    ]
    processing_cv = []
    min_max_ratios = []
    processing_values = []
    for operation in instance.operations:
        values = [
            float(instance.processing_time[(operation, island)])
            for island in instance.operation_data[operation].eligible_islands
        ]
        processing_values.extend(values)
        if len(values) > 1:
            processing_cv.append(pstdev(values) / mean(values))
            min_max_ratios.append(min(values) / max(values))
    edge_count = sum(len(product.precedence) for product in instance.product_data.values())
    potential_edges = sum(
        len(product.operations) * max(0, len(product.operations) - 1) / 2
        for product in instance.product_data.values()
    )
    ready_mean, ready_max = _ready_set_proxy(instance)
    reconfiguration = [float(value) for value in instance.reconfiguration_time.values()]
    reconfiguration_nonzero = [value for value in reconfiguration if value > 0]
    off_diagonal_keys = [
        key for key in instance.reconfiguration_time if key[1] != key[2]
    ]
    config_counts = [
        sum(
            instance.operation_data[operation].required_config == configuration
            for operation in instance.operations
        )
        for configuration in instance.configurations
    ]
    processing_mean = mean(processing_values) if processing_values else 0.0
    w_times = [float(value) for value in instance.w_loaded_time.values()]
    f_times = [
        float(instance.f_outbound_time[key] + instance.f_return_time[key])
        for key in instance.f_outbound_time
    ]
    return {
        "family": benchmark_family(instance.instance_id),
        "instance_id": instance.instance_id,
        "number_of_products": len(instance.products),
        "number_of_operations": instance.num_operations,
        "number_of_islands": len(instance.islands),
        "number_of_W_AGVs": len(instance.agvs_w),
        "number_of_F_AGVs": len(instance.agvs_f),
        "number_of_configurations": len(instance.configurations),
        "number_of_precedence_edges": edge_count,
        "F_route_mean": mean(route),
        "F_route_median": median(route),
        "R_full_op": mean(float(value == 1.0) for value in route),
        "R_high_op": mean(float(value >= 0.8) for value in route),
        "F_route_min": min(route),
        "F_route_max": max(route),
        "F_cap_mean": mean(capability),
        "R_full_island": mean(float(value == 1.0) for value in capability),
        "capability_heterogeneity": pstdev(capability),
        "capability_selectivity_mean": mean(selectivity),
        "capability_selectivity_median": median(selectivity),
        "capability_selectivity_std": pstdev(selectivity),
        "capability_selectivity_min": min(selectivity),
        "capability_selectivity_max": max(selectivity),
        "processing_CV_mean": mean(processing_cv) if processing_cv else 0.0,
        "processing_CV_median": median(processing_cv) if processing_cv else 0.0,
        "processing_CV_p90": _percentile(processing_cv, 0.9),
        "processing_min_max_ratio_mean": mean(min_max_ratios) if min_max_ratios else 1.0,
        "mean_processing_time": processing_mean,
        "precedence_edge_density": edge_count / max(1.0, potential_edges),
        "mean_predecessor_count": mean(len(instance.predecessors[op]) for op in instance.operations),
        "mean_successor_count": mean(len(instance.successors[op]) for op in instance.operations),
        "number_of_source_operations": sum(not instance.predecessors[op] for op in instance.operations),
        "number_of_sink_operations": sum(not instance.successors[op] for op in instance.operations),
        "mean_ready_set_size": ready_mean,
        "max_ready_set_size": ready_max,
        "mean_reconfiguration_time": mean(reconfiguration) if reconfiguration else 0.0,
        "median_reconfiguration_time": median(reconfiguration) if reconfiguration else 0.0,
        "maximum_reconfiguration_time": max(reconfiguration, default=0.0),
        "nonzero_reconfiguration_std": pstdev(reconfiguration_nonzero) if reconfiguration_nonzero else 0.0,
        "nonzero_reconfiguration_CV": (
            pstdev(reconfiguration_nonzero) / mean(reconfiguration_nonzero)
            if reconfiguration_nonzero else 0.0
        ),
        "RI": mean(reconfiguration_nonzero) / max(processing_mean, 1e-12) if reconfiguration_nonzero else 0.0,
        "configuration_diversity_entropy": _normalized_entropy(config_counts),
        "nonzero_offdiagonal_reconfiguration_ratio": (
            mean(float(instance.reconfiguration_time[key] > 0) for key in off_diagonal_keys)
            if off_diagonal_keys else 0.0
        ),
        "W_transport_intensity": mean(w_times) / max(processing_mean, 1e-12) if w_times else 0.0,
        "F_transport_intensity": mean(f_times) / max(processing_mean, 1e-12) if f_times else 0.0,
        "operations_per_W_AGV": instance.num_operations / max(1, len(instance.agvs_w)),
        "operations_per_F_AGV": instance.num_operations / max(1, len(instance.agvs_f)),
        "islands_per_W_AGV": len(instance.islands) / max(1, len(instance.agvs_w)),
        "islands_per_F_AGV": len(instance.islands) / max(1, len(instance.agvs_f)),
    }
