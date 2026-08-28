"""Passive, deterministic feature extraction for Phase 6A ALNS diagnosis."""

from __future__ import annotations

from collections import Counter
import json
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.schedule import Schedule


BOTTLENECK_BY_BINDING = {
    "PRODUCT": "PRECEDENCE_SEQUENCE",
    "ISLAND": "ISLAND_PROCESSING_LOAD",
    "ISLAND_CONFIG": "RECONFIGURATION",
    "W_AGV": "W_LOGISTICS",
    "F_AGV": "F_LOGISTICS",
}


def schedule_features(instance: Instance, schedule: Schedule) -> dict[str, dict[str, object]]:
    """Return decision-time operation features without mutating the schedule."""
    records = schedule.operation_schedules
    makespan = max((record.completion_time for record in records.values()), default=0.0)
    island_load = Counter(record.island_id for record in records.values())
    max_load = max(island_load.values(), default=1)
    critical = _processing_critical_operations(schedule, makespan)
    resource_critical = {
        timeline[-1] for timeline in schedule.island_timelines.values() if timeline
    }
    result: dict[str, dict[str, object]] = {}
    for operation in instance.operations:
        record = records[operation]
        island_chain = schedule.island_timelines[record.island_id]
        island_position = island_chain.index(operation)
        next_reconfiguration = 0.0
        if island_position + 1 < len(island_chain):
            following = records[island_chain[island_position + 1]]
            next_reconfiguration = following.reconfiguration_end - following.reconfiguration_start
        w_position = _task_position(schedule.w_timelines, operation)
        f_position = _task_position(schedule.f_timelines, operation)
        w_delay = max(0.0, record.w_ready_time - record.product_ready_time)
        f_delay = max(0.0, record.f_ready_time - record.product_ready_time)
        sync_delay = max(0.0, record.start_time - max(
            record.product_ready_time, record.config_ready_time,
            record.w_ready_time, record.f_ready_time,
        ))
        reconfiguration = record.reconfiguration_end - record.reconfiguration_start
        slack = max(0.0, makespan - record.completion_time)
        result[operation] = {
            "operation_id": operation,
            "product_id": record.product_id,
            "assigned_island": record.island_id,
            "required_configuration": record.config_id,
            "processing_time": record.processing_time,
            "operation_start": record.start_time,
            "operation_end": record.completion_time,
            "operation_slack": slack,
            "is_on_processing_critical_path": operation in critical,
            "is_on_resource_critical_chain": operation in resource_critical,
            "criticality_score": 1.0 / (1.0 + slack),
            "number_of_predecessors": len(instance.predecessors[operation]),
            "number_of_successors": len(instance.successors[operation]),
            "eligible_island_count": len(instance.operation_data[operation].eligible_islands),
            "island_load_before": island_load[record.island_id],
            "island_relative_load": island_load[record.island_id] / max_load,
            "preceding_reconfiguration_time": reconfiguration,
            "following_reconfiguration_time": next_reconfiguration,
            "local_reconfiguration_contribution": reconfiguration + next_reconfiguration,
            "W_waiting_or_delay_contribution": w_delay,
            "F_waiting_or_delay_contribution": f_delay,
            "synchronization_wait_contribution": sync_delay,
            "position_in_product": schedule.product_sequences[record.product_id].index(operation),
            "position_in_island_chain": island_position,
            "position_in_W_chain": w_position,
            "position_in_F_chain": f_position,
        }
    return result


def _task_position(timelines, operation: str) -> int | None:
    for tasks in timelines.values():
        for position, task in enumerate(tasks):
            if task.operation_id == operation:
                return position
    return None


def _processing_critical_operations(schedule: Schedule, makespan: float) -> set[str]:
    """Zero-gap backward proxy on product and island processing arcs."""
    records = schedule.operation_schedules
    frontier = [op for op, record in records.items() if record.completion_time == makespan]
    critical = set(frontier)
    island_predecessor = {}
    for timeline in schedule.island_timelines.values():
        island_predecessor.update({op: timeline[index - 1] for index, op in enumerate(timeline) if index})
    while frontier:
        operation = frontier.pop()
        start = records[operation].start_time
        predecessors = list(schedule.product_predecessor.get(operation) and [schedule.product_predecessor[operation]] or [])
        if operation in island_predecessor:
            predecessors.append(island_predecessor[operation])
        for predecessor in predecessors:
            if records[predecessor].completion_time == start and predecessor not in critical:
                critical.add(predecessor)
                frontier.append(predecessor)
    return critical


def bottleneck_proxy(schedule: Schedule) -> str:
    """Classify the makespan terminal constraint; this is not causal ground truth."""
    makespan = max(record.completion_time for record in schedule.operation_schedules.values())
    terminal = [record for record in schedule.operation_schedules.values() if record.completion_time == makespan]
    bindings = {binding for record in terminal for binding in record.binding_resource}
    categories = {BOTTLENECK_BY_BINDING.get(binding, "MIXED_OR_UNCERTAIN") for binding in bindings}
    if not categories:
        return "MIXED_OR_UNCERTAIN"
    if len(categories) > 1:
        return "CROSS_RESOURCE_SYNCHRONIZATION"
    return next(iter(categories))


class Phase6AObserver:
    """Buffered observer used by the experiment runner."""

    def __init__(self, instance: Instance, run_metadata: Mapping[str, object]):
        self.instance = instance
        self.metadata = dict(run_metadata)
        self.transitions: list[dict[str, object]] = []
        self.targets: list[dict[str, object]] = []

    def __call__(self, event: Mapping[str, object]) -> None:
        current = event["current_before"]
        candidate = event["candidate"]
        best_before = event["best_before"]
        current_after = event["current_after"]
        best_after = event["best_after"]
        destroyed = event["destroyed_operation_ids"]
        immediate_delta = current.makespan - candidate.makespan
        target_features = schedule_features(self.instance, current.schedule)
        selected = [target_features[operation] for operation in destroyed]
        aggregate = self._aggregate(selected)
        accepted = bool(event["accepted"])
        new_best = bool(event["new_global_best"])
        if new_best:
            acceptance_type = "NEW_GLOBAL_BEST"
        elif not accepted:
            acceptance_type = "REJECTED"
        elif immediate_delta > 0:
            acceptance_type = "IMPROVING_ACCEPTED"
        elif immediate_delta == 0:
            acceptance_type = "EQUAL_ACCEPTED"
        else:
            acceptance_type = "WORSENING_ACCEPTED"
        base = {**self.metadata, "iteration": event["iteration"]}
        self.transitions.append({
            **base,
            "elapsed_time": event["elapsed_time"],
            "iteration_runtime": event["iteration_runtime"],
            "decoder_evaluations": event["decoder_evaluations"],
            "current_makespan_before": current.makespan,
            "best_makespan_before": best_before.makespan,
            "destroy_operator": event["destroy_operator"],
            "repair_operator": event["repair_operator"],
            "destroy_fraction": event["destroy_fraction"],
            "destroy_count": len(destroyed),
            "destroyed_operation_ids": json.dumps(destroyed),
            "candidate_makespan": candidate.makespan,
            "immediate_delta": immediate_delta,
            "relative_improvement": immediate_delta / current.makespan,
            "accepted": accepted,
            "acceptance_type": acceptance_type,
            "current_makespan_after": current_after.makespan,
            "best_makespan_after": best_after.makespan,
            "new_global_best": new_best,
            "operator_weights_before": json.dumps(event["operator_weights_before"], sort_keys=True),
            "operator_weights_after": json.dumps(event["operator_weights_after"], sort_keys=True),
            "number_of_operations_reinserted": len(destroyed),
            "number_of_candidate_insertions_evaluated": None,
            "repair_decoder_evaluations": event["repair_decoder_evaluations"],
            "repair_runtime": event["repair_runtime"],
            "candidate_trials_completed": event["candidate_trials_completed"],
            "temperature_before": event["temperature_before"],
            "bottleneck_type": bottleneck_proxy(current.schedule),
            **aggregate,
        })
        for feature in selected:
            self.targets.append({**base, **feature, "move_immediate_delta": immediate_delta,
                                 "move_accepted": accepted, "move_new_global_best": new_best})

    @staticmethod
    def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        def mean(field):
            values = [float(row[field]) for row in rows if row[field] is not None]
            return sum(values) / len(values) if values else None
        def ratio(field):
            return sum(bool(row[field]) for row in rows) / len(rows)
        return {
            "critical_path_overlap_ratio": ratio("is_on_processing_critical_path"),
            "critical_resource_overlap_ratio": ratio("is_on_resource_critical_chain"),
            "same_product_ratio": max(Counter(row["product_id"] for row in rows).values()) / len(rows),
            "same_island_ratio": max(Counter(row["assigned_island"] for row in rows).values()) / len(rows),
            "mean_slack": mean("operation_slack"),
            "mean_island_load": mean("island_load_before"),
            "mean_reconfiguration_contribution": mean("local_reconfiguration_contribution"),
            "mean_W_delay_contribution": mean("W_waiting_or_delay_contribution"),
            "mean_F_delay_contribution": mean("F_waiting_or_delay_contribution"),
            "mean_sync_delay_contribution": mean("synchronization_wait_contribution"),
            "configuration_diversity": len({row["required_configuration"] for row in rows}),
            "island_diversity": len({row["assigned_island"] for row in rows}),
            "product_diversity": len({row["product_id"] for row in rows}),
        }
