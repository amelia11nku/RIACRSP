"""Pure-Python dynamic Capability-Logistics Coupled Graph builder."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.insertion_decoder import Action, InsertionDecoder
from rcias_clgri.env.schedule import Schedule


@dataclass(frozen=True)
class EdgeRecord:
    source_type: str
    relation: str
    target_type: str
    source_id: str
    target_id: str
    features: Mapping[str, float | str]


@dataclass(frozen=True)
class GraphState:
    node_features: Mapping[str, Mapping[str, Mapping[str, float | str]]]
    edges: tuple[EdgeRecord, ...]
    ready_operations: tuple[str, ...]
    operation_mask: Mapping[str, bool]
    island_masks: Mapping[str, Mapping[str, bool]]
    w_masks: Mapping[tuple[str, str], tuple[str | None, ...]]
    f_masks: Mapping[tuple[str, str], tuple[str, ...]]


def _largest_gap(intervals: list[tuple[float, float]], horizon: float) -> float:
    if not intervals:
        return horizon
    ordered = sorted(intervals)
    largest = ordered[0][0]
    for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]):
        largest = max(largest, right_start - left_end)
    return max(largest, horizon - ordered[-1][1])


def _ready(instance: Instance, schedule: Schedule) -> tuple[str, ...]:
    selected = schedule.scheduled_operations
    return tuple(
        op_id for op_id in instance.operations
        if op_id not in selected and instance.predecessors[op_id] <= selected
    )


def _candidate_edge_features(
    instance: Instance,
    schedule: Schedule,
    decoder: InsertionDecoder,
    op_id: str,
    island_id: str,
) -> dict[str, float]:
    product_id = instance.product_of[op_id]
    sequence = schedule.product_sequences[product_id]
    pickup = "WH" if not sequence else schedule.operation_schedules[sequence[-1]].island_id
    w_choices: tuple[str | None, ...] = (None,) if pickup == island_id else instance.agvs_w
    probes = [
        decoder.probe_action(schedule, Action(op_id, island_id, w_id, f_id))
        for w_id in w_choices
        for f_id in instance.agvs_f
    ]
    best = min(probes, key=lambda probe: probe.predicted_completion)
    machine_ready = best.machine_probe.reconfiguration_end
    sync_ready = max(best.product_ready, best.w_ready, best.f_probe.task.arrival_island)
    return {
        "processing_time": float(instance.processing_time[(op_id, island_id)]),
        "earliest_machine_insertion_start": best.machine_probe.processing_start,
        "predicted_operation_completion": best.predicted_completion,
        "setup_before_at_best_gap": best.machine_probe.setup_before,
        "setup_after_at_best_gap": best.machine_probe.setup_after,
        "incremental_reconfiguration_cost": best.machine_probe.incremental_reconfiguration_cost,
        "workpiece_loaded_distance": 0.0 if pickup == island_id else instance.distance[(pickup, island_id)],
        "best_w_arrival_lower_bound": best.w_ready,
        "best_f_arrival_lower_bound": best.f_probe.task.arrival_island,
        "predicted_sync_wait": max(0.0, sync_ready - machine_ready),
    }


def build_graph_state(instance: Instance, schedule: Schedule) -> GraphState:
    """Build five node types, typed relations, features, and hard action masks."""

    horizon = instance.horizon
    ready = _ready(instance, schedule)
    ready_set = set(ready)
    node_features: dict[str, dict[str, dict[str, float | str]]] = {
        "O": {}, "J": {}, "M": {}, "W": {}, "F": {},
    }
    for op_id in instance.operations:
        operation = instance.operation_data[op_id]
        times = list(operation.processing_time.values())
        record = schedule.operation_schedules.get(op_id)
        product_sequence = schedule.product_sequences[operation.product_id]
        product_size = len(instance.product_data[operation.product_id].operations)
        node_features["O"][op_id] = {
            "min_processing_time": min(times) / horizon,
            "mean_processing_time": mean(times) / horizon,
            "max_processing_time": max(times) / horizon,
            "num_eligible_islands": float(len(operation.eligible_islands)),
            "dag_in_degree": float(len(instance.predecessors[op_id])),
            "dag_out_degree": float(len(instance.successors[op_id])),
            "transitive_pred_ratio": len(instance.transitive_predecessors[op_id]) / max(1, product_size - 1),
            "transitive_succ_ratio": len(instance.transitive_successors[op_id]) / max(1, product_size - 1),
            "required_configuration": operation.required_config,
            "is_scheduled": float(record is not None),
            "is_topological_ready": float(op_id in ready_set),
            "product_progress_ratio": len(product_sequence) / product_size,
            "scheduled_start": 0.0 if record is None else record.start_time / horizon,
            "scheduled_end": 0.0 if record is None else record.completion_time / horizon,
        }
    for product_id in instance.products:
        sequence = schedule.product_sequences[product_id]
        operations = instance.product_data[product_id].operations
        remaining = [op_id for op_id in operations if op_id not in schedule.operation_schedules]
        remaining_work = sum(
            min(instance.operation_data[op_id].processing_time.values()) for op_id in remaining
        )
        last_completion = 0.0 if not sequence else schedule.operation_schedules[sequence[-1]].completion_time
        node_features["J"][product_id] = {
            "num_operations": float(len(operations)),
            "scheduled_ratio": len(sequence) / len(operations),
            "remaining_workload_estimate": remaining_work / horizon,
            "last_actual_completion": last_completion / horizon,
            "current_workpiece_location": "WH" if not sequence else schedule.operation_schedules[sequence[-1]].island_id,
        }
    for island_id in instance.islands:
        sequence = schedule.island_timelines[island_id]
        records = [schedule.operation_schedules[op_id] for op_id in sequence]
        intervals = [(record.start_time, record.completion_time) for record in records]
        tail = records[-1].completion_time if records else 0.0
        node_features["M"][island_id] = {
            "initial_configuration": instance.island_data[island_id].initial_config,
            "tail_configuration": records[-1].config_id if records else instance.island_data[island_id].initial_config,
            "total_processing_load": sum(record.processing_time for record in records) / horizon,
            "total_reconfiguration_time": sum(record.reconfiguration_end - record.reconfiguration_start for record in records) / horizon,
            "tail_completion_time": tail / horizon,
            "num_scheduled_operations": float(len(records)),
            "num_supported_configurations": float(len(instance.island_data[island_id].supported_configs)),
            "num_free_gaps": float(len(records) + 1),
        }
    for vehicle_id in instance.agvs_w:
        tasks = schedule.w_timelines[vehicle_id]
        intervals = [(task.empty_start, task.arrival_time) for task in tasks]
        node_features["W"][vehicle_id] = {
            "num_tasks": float(len(tasks)),
            "total_loaded_time": sum(task.loaded_travel_time for task in tasks) / horizon,
            "total_empty_time": sum(task.empty_travel_time for task in tasks) / horizon,
            "tail_completion_time": (tasks[-1].arrival_time if tasks else 0.0) / horizon,
            "largest_free_gap": _largest_gap(intervals, horizon) / horizon,
            "last_delivery_node": tasks[-1].destination if tasks else "WH",
        }
    for vehicle_id in instance.agvs_f:
        tasks = schedule.f_timelines[vehicle_id]
        intervals = [(task.departure_wh, task.return_wh) for task in tasks]
        node_features["F"][vehicle_id] = {
            "num_tasks": float(len(tasks)),
            "total_busy_time": sum(task.return_wh - task.departure_wh for task in tasks) / horizon,
            "tail_return_time": (tasks[-1].return_wh if tasks else 0.0) / horizon,
            "largest_free_gap": _largest_gap(intervals, horizon) / horizon,
        }

    edges: list[EdgeRecord] = []
    for product_id in instance.products:
        for op_id in instance.product_data[product_id].operations:
            edges.append(EdgeRecord("J", "contains", "O", product_id, op_id, {}))
            edges.append(EdgeRecord("O", "belongs_to", "J", op_id, product_id, {}))
        for source, target in instance.product_data[product_id].precedence:
            edges.append(EdgeRecord("O", "precedence", "O", source, target, {}))
            edges.append(EdgeRecord("O", "precedence_rev", "O", target, source, {}))
    decoder = InsertionDecoder(instance)
    unscheduled = set(instance.operations) - schedule.scheduled_operations
    om_features: dict[tuple[str, str], dict[str, float]] = {}
    for op_id in instance.operations:
        for island_id in instance.operation_data[op_id].eligible_islands:
            features = (
                _candidate_edge_features(instance, schedule, decoder, op_id, island_id)
                if op_id in unscheduled else
                {"processing_time": float(instance.processing_time[(op_id, island_id)])}
            )
            om_features[(op_id, island_id)] = features
            edges.append(EdgeRecord("O", "eligible_on", "M", op_id, island_id, features))
            edges.append(EdgeRecord("M", "can_process", "O", island_id, op_id, features))
    for source in instance.islands:
        for target in instance.islands:
            if source != target:
                edges.append(EdgeRecord("M", "spatial", "M", source, target, {
                    "distance": instance.distance[(source, target)] / max(1.0, max(instance.distance.values())),
                }))
    for vehicle_id in instance.agvs_w:
        for island_id in instance.islands:
            edges.append(EdgeRecord("W", "reachable_to", "M", vehicle_id, island_id, {}))
            edges.append(EdgeRecord("M", "reachable_by", "W", island_id, vehicle_id, {}))
    for vehicle_id in instance.agvs_f:
        for island_id in instance.islands:
            features = {
                "outbound_time": instance.f_outbound_time[(vehicle_id, island_id)] / horizon,
                "round_trip_time": (
                    instance.f_outbound_time[(vehicle_id, island_id)]
                    + instance.f_return_time[(vehicle_id, island_id)]
                ) / horizon,
            }
            edges.append(EdgeRecord("F", "deliver_to", "M", vehicle_id, island_id, features))
            edges.append(EdgeRecord("M", "served_by", "F", island_id, vehicle_id, features))
    for product_id, sequence in schedule.product_sequences.items():
        for source, target in zip(sequence, sequence[1:]):
            edges.append(EdgeRecord("O", "actual_product_prev", "O", source, target, {}))
    for island_id, sequence in schedule.island_timelines.items():
        for source, target in zip(sequence, sequence[1:]):
            edges.append(EdgeRecord("O", "machine_prev", "O", source, target, {"island": island_id}))

    operation_mask = {op_id: op_id in ready_set for op_id in instance.operations}
    island_masks = {
        op_id: {island_id: island_id in instance.operation_data[op_id].eligible_islands and op_id in ready_set for island_id in instance.islands}
        for op_id in instance.operations
    }
    w_masks: dict[tuple[str, str], tuple[str | None, ...]] = {}
    f_masks: dict[tuple[str, str], tuple[str, ...]] = {}
    for op_id in ready:
        product_sequence = schedule.product_sequences[instance.product_of[op_id]]
        pickup = "WH" if not product_sequence else schedule.operation_schedules[product_sequence[-1]].island_id
        for island_id in instance.operation_data[op_id].eligible_islands:
            w_masks[(op_id, island_id)] = (None,) if pickup == island_id else instance.agvs_w
            f_masks[(op_id, island_id)] = instance.agvs_f
    return GraphState(node_features, tuple(edges), ready, operation_mask, island_masks, w_masks, f_masks)
