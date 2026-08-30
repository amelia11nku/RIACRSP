"""Leakage-safe CSG-1.0 node features from one complete decoded schedule."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Mapping

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.data.instance import Instance
from rcias_clgri.env.schedule import Schedule


EPSILON = 1e-9


def _positive_mean(values) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    return mean(positive) if positive else 1.0


def _position(index: int, length: int) -> float:
    return float(index) / max(length - 1, 1)


@dataclass(frozen=True)
class CSGNormalization:
    makespan_scale: float
    mean_processing_time: float
    mean_travel_time: float
    mean_reconfiguration_time: float
    operation_count: float
    island_count: float
    configuration_count: float
    w_resource_count: float
    f_resource_count: float

    @classmethod
    def from_state(cls, instance: Instance, schedule: Schedule) -> "CSGNormalization":
        makespan = max(record.completion_time for record in schedule.operation_schedules.values())
        processing = [
            instance.processing_time[(operation, island)]
            for operation in instance.operations
            for island in instance.operation_data[operation].eligible_islands
        ]
        travel = [
            *instance.w_loaded_time.values(), *instance.w_empty_time.values(),
            *instance.f_outbound_time.values(), *instance.f_return_time.values(),
        ]
        return cls(
            makespan_scale=max(float(makespan), 1.0),
            mean_processing_time=_positive_mean(processing),
            mean_travel_time=_positive_mean(travel),
            mean_reconfiguration_time=_positive_mean(instance.reconfiguration_time.values()),
            operation_count=float(max(len(instance.operations), 1)),
            island_count=float(max(len(instance.islands), 1)),
            configuration_count=float(max(len(instance.configurations), 1)),
            w_resource_count=float(max(len(instance.agvs_w), 1)),
            f_resource_count=float(max(len(instance.agvs_f), 1)),
        )

    def time(self, value: float) -> float:
        return float(value) / self.makespan_scale

    def processing(self, value: float) -> float:
        return float(value) / self.mean_processing_time

    def travel(self, value: float) -> float:
        return float(value) / self.mean_travel_time

    def reconfiguration(self, value: float) -> float:
        return float(value) / self.mean_reconfiguration_time

    def to_dict(self) -> dict[str, float]:
        return {
            "makespan_scale": self.makespan_scale,
            "mean_processing_time": self.mean_processing_time,
            "mean_travel_time": self.mean_travel_time,
            "mean_reconfiguration_time": self.mean_reconfiguration_time,
            "operation_count": self.operation_count,
            "island_count": self.island_count,
            "configuration_count": self.configuration_count,
            "w_resource_count": self.w_resource_count,
            "f_resource_count": self.f_resource_count,
        }


@dataclass(frozen=True)
class ReconfigurationContext:
    key: str
    operation_id: str
    island_id: str
    previous_operation: str | None
    source_config: str
    target_config: str
    start_time: float
    end_time: float
    features: Mapping[str, float]


@dataclass(frozen=True)
class ExtractedFeatures:
    nodes: Mapping[str, Mapping[str, Mapping[str, float]]]
    reconfiguration_contexts: tuple[ReconfigurationContext, ...]
    normalization: CSGNormalization


def extract_node_features(
    instance: Instance,
    schedule: Schedule,
) -> ExtractedFeatures:
    if set(schedule.operation_schedules) != set(instance.operations):
        raise ValueError("CSG requires a complete decoded schedule")
    norm = CSGNormalization.from_state(instance, schedule)
    current = schedule_features(instance, schedule)
    w_lookup = {
        task.operation_id: (task, position, len(tasks))
        for tasks in schedule.w_timelines.values()
        for position, task in enumerate(tasks)
    }
    f_lookup = {
        task.operation_id: (task, position, len(tasks))
        for tasks in schedule.f_timelines.values()
        for position, task in enumerate(tasks)
    }
    product_lookup = {
        operation: (position, len(sequence))
        for sequence in schedule.product_sequences.values()
        for position, operation in enumerate(sequence)
    }
    island_lookup = {
        operation: (position, len(sequence))
        for sequence in schedule.island_timelines.values()
        for position, operation in enumerate(sequence)
    }
    operation_nodes: dict[str, dict[str, float]] = {}
    for operation_id in sorted(instance.operations):
        record = schedule.operation_schedules[operation_id]
        feature = current[operation_id]
        operation = instance.operation_data[operation_id]
        product_position, product_length = product_lookup[operation_id]
        island_position, island_length = island_lookup[operation_id]
        w_entry = w_lookup.get(operation_id)
        f_entry = f_lookup.get(operation_id)
        mean_processing = mean(operation.processing_time.values())
        slack = float(feature["operation_slack"])
        operation_nodes[operation_id] = {
            "processing_time": float(record.processing_time),
            "processing_time_normalized": norm.processing(record.processing_time),
            "mean_eligible_processing_time_normalized": norm.processing(mean_processing),
            "eligible_island_count": float(len(operation.eligible_islands)),
            "eligible_island_fraction": len(operation.eligible_islands) / norm.island_count,
            "start_time": float(record.start_time),
            "start_time_normalized": norm.time(record.start_time),
            "completion_time": float(record.completion_time),
            "completion_time_normalized": norm.time(record.completion_time),
            "operation_slack": slack,
            "operation_slack_normalized": norm.time(slack),
            "criticality_proxy": float(feature["criticality_score"]),
            "is_processing_critical_proxy": float(bool(feature["is_on_processing_critical_path"])),
            "is_resource_terminal_proxy": float(bool(feature["is_on_resource_critical_chain"])),
            "w_delay": float(feature["W_waiting_or_delay_contribution"]),
            "w_delay_normalized": norm.time(float(feature["W_waiting_or_delay_contribution"])),
            "f_delay": float(feature["F_waiting_or_delay_contribution"]),
            "f_delay_normalized": norm.time(float(feature["F_waiting_or_delay_contribution"])),
            "synchronization_wait": float(feature["synchronization_wait_contribution"]),
            "synchronization_wait_normalized": norm.time(float(feature["synchronization_wait_contribution"])),
            "local_reconfiguration": float(feature["local_reconfiguration_contribution"]),
            "local_reconfiguration_normalized": norm.reconfiguration(float(feature["local_reconfiguration_contribution"])),
            "island_relative_load": float(feature["island_relative_load"]),
            "product_position_normalized": _position(product_position, product_length),
            "island_position_normalized": _position(island_position, island_length),
            "w_chain_position_normalized": 0.0 if w_entry is None else _position(w_entry[1], w_entry[2]),
            "f_chain_position_normalized": 0.0 if f_entry is None else _position(f_entry[1], f_entry[2]),
            "has_w_event": float(w_entry is not None),
            "has_f_event": float(f_entry is not None),
            "predecessor_count_normalized": len(instance.predecessors[operation_id]) / norm.operation_count,
            "successor_count_normalized": len(instance.successors[operation_id]) / norm.operation_count,
        }

    island_processing = {
        island: sum(schedule.operation_schedules[operation].processing_time for operation in schedule.island_timelines[island])
        for island in instance.islands
    }
    max_island_processing = max(island_processing.values(), default=1.0) or 1.0
    island_nodes: dict[str, dict[str, float]] = {}
    reconfiguration_contexts: list[ReconfigurationContext] = []
    for island in sorted(instance.islands):
        sequence = schedule.island_timelines[island]
        reconfiguration_time = 0.0
        reconfiguration_count = 0
        previous_config = instance.island_data[island].initial_config
        previous_operation = None
        for position, operation_id in enumerate(sequence):
            record = schedule.operation_schedules[operation_id]
            duration = float(record.reconfiguration_end - record.reconfiguration_start)
            if duration > EPSILON:
                reconfiguration_count += 1
                reconfiguration_time += duration
                reconfiguration_contexts.append(ReconfigurationContext(
                    key=f"R:{operation_id}",
                    operation_id=operation_id,
                    island_id=island,
                    previous_operation=previous_operation,
                    source_config=previous_config,
                    target_config=record.config_id,
                    start_time=float(record.reconfiguration_start),
                    end_time=float(record.reconfiguration_end),
                    features={
                        "start_time": float(record.reconfiguration_start),
                        "end_time": float(record.reconfiguration_end),
                        "duration": duration,
                        "duration_normalized": norm.reconfiguration(duration),
                        "resource_position_normalized": _position(position, len(sequence)),
                        "from_initial_configuration": float(previous_operation is None),
                    },
                ))
            previous_config = record.config_id
            previous_operation = operation_id
        processing_load = float(island_processing[island])
        busy = processing_load + reconfiguration_time
        last_completion = 0.0 if not sequence else schedule.operation_schedules[sequence[-1]].completion_time
        island_nodes[island] = {
            "assigned_processing_load": processing_load,
            "assigned_processing_load_normalized": norm.time(processing_load),
            "relative_processing_load": processing_load / max_island_processing,
            "scheduled_operation_count_normalized": len(sequence) / norm.operation_count,
            "capability_count_normalized": len(instance.island_data[island].supported_configs) / norm.configuration_count,
            "capability_fraction": len(instance.island_data[island].supported_configs) / norm.configuration_count,
            "reconfiguration_count_normalized": reconfiguration_count / norm.operation_count,
            "reconfiguration_time_burden": reconfiguration_time,
            "reconfiguration_time_normalized": norm.time(reconfiguration_time),
            "busy_time_normalized": norm.time(busy),
            "idle_time_normalized": norm.time(max(0.0, norm.makespan_scale - busy)),
            "last_completion_normalized": norm.time(last_completion),
        }

    supporting_island_count = Counter(
        config
        for island in instance.islands
        for config in instance.island_data[island].supported_configs
    )
    required_operation_count = Counter(
        instance.operation_data[operation].required_config
        for operation in instance.operations
    )
    config_nodes = {
        config: {
            "supporting_island_fraction": supporting_island_count[config] / norm.island_count,
            "required_operation_fraction": required_operation_count[config] / norm.operation_count,
        }
        for config in sorted(instance.configurations)
    }

    w_busy = {
        resource: sum(task.arrival_time - task.empty_start for task in schedule.w_timelines[resource])
        for resource in instance.agvs_w
    }
    max_w_busy = max(w_busy.values(), default=1.0) or 1.0
    w_nodes = {}
    w_event_nodes = {}
    for resource in sorted(instance.agvs_w):
        tasks = schedule.w_timelines[resource]
        travel = sum(task.empty_travel_time + task.loaded_travel_time for task in tasks)
        waiting = sum(max(0.0, task.loaded_start - task.empty_arrival) for task in tasks)
        w_nodes[resource] = {
            "task_count_normalized": len(tasks) / norm.operation_count,
            "busy_time": float(w_busy[resource]),
            "busy_time_normalized": norm.time(w_busy[resource]),
            "travel_time_normalized": norm.travel(travel),
            "waiting_burden_normalized": norm.time(waiting),
            "relative_load": w_busy[resource] / max_w_busy,
            "last_completion_normalized": norm.time(0.0 if not tasks else tasks[-1].arrival_time),
        }
        for position, task in enumerate(tasks):
            w_event_nodes[task.task_id] = {
                "start_time": float(task.empty_start),
                "end_time": float(task.arrival_time),
                "duration_normalized": norm.time(task.arrival_time - task.empty_start),
                "empty_travel_normalized": norm.travel(task.empty_travel_time),
                "pickup_wait_normalized": norm.time(max(0.0, task.loaded_start - task.empty_arrival)),
                "loaded_travel_normalized": norm.travel(task.loaded_travel_time),
                "resource_position_normalized": _position(position, len(tasks)),
                "warehouse_origin": float(task.pickup == "WH"),
                "first_product_transport": float(task.predecessor_op is None),
            }

    f_busy = {
        resource: sum(task.return_wh - task.departure_wh for task in schedule.f_timelines[resource])
        for resource in instance.agvs_f
    }
    max_f_busy = max(f_busy.values(), default=1.0) or 1.0
    f_nodes = {}
    f_event_nodes = {}
    for resource in sorted(instance.agvs_f):
        tasks = schedule.f_timelines[resource]
        travel = sum(task.outbound_time + task.return_time for task in tasks)
        f_nodes[resource] = {
            "task_count_normalized": len(tasks) / norm.operation_count,
            "busy_time": float(f_busy[resource]),
            "busy_time_normalized": norm.time(f_busy[resource]),
            "travel_time_normalized": norm.travel(travel),
            "relative_load": f_busy[resource] / max_f_busy,
            "last_completion_normalized": norm.time(0.0 if not tasks else tasks[-1].return_wh),
        }
        for position, task in enumerate(tasks):
            f_event_nodes[task.task_id] = {
                "start_time": float(task.departure_wh),
                "arrival_time": float(task.arrival_island),
                "end_time": float(task.return_wh),
                "duration_normalized": norm.time(task.return_wh - task.departure_wh),
                "outbound_travel_normalized": norm.travel(task.outbound_time),
                "return_travel_normalized": norm.travel(task.return_time),
                "resource_position_normalized": _position(position, len(tasks)),
            }

    nodes = {
        "OP": operation_nodes,
        "ISLAND": island_nodes,
        "CONFIG": config_nodes,
        "W_AGV": w_nodes,
        "F_AGV": f_nodes,
        "W_EVENT": w_event_nodes,
        "F_EVENT": f_event_nodes,
        "RECONF_EVENT": {context.key: context.features for context in reconfiguration_contexts},
    }
    return ExtractedFeatures(nodes, tuple(reconfiguration_contexts), norm)
