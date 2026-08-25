"""Objective calculation from immutable instance data and final schedule records."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rcias_clgri.data.instance import Instance

from .schedule import Schedule


@dataclass(frozen=True)
class ObjectiveBreakdown:
    makespan: float
    reconfiguration_cost: float
    w_loaded_cost: float
    w_empty_cost: float
    f_outbound_cost: float
    f_return_cost: float
    total_cost: float
    reconfiguration_count: int
    w_loaded_travel: float
    w_empty_travel: float
    f_travel: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calculate_objective(instance: Instance, schedule: Schedule) -> ObjectiveBreakdown:
    """Recompute makespan and every cost term from the schedule records."""

    makespan = max((record.completion_time for record in schedule.operation_schedules.values()), default=0.0)
    reconfiguration_cost = 0.0
    reconfiguration_count = 0
    for island_id, sequence in schedule.island_timelines.items():
        previous_config = instance.island_data[island_id].initial_config
        for op_id in sequence:
            config = schedule.operation_schedules[op_id].config_id
            reconfiguration_cost += instance.reconfiguration_cost[(island_id, previous_config, config)]
            if previous_config != config:
                reconfiguration_count += 1
            previous_config = config

    w_loaded_cost = 0.0
    w_empty_cost = 0.0
    w_loaded_travel = 0.0
    w_empty_travel = 0.0
    for vehicle_id, tasks in schedule.w_timelines.items():
        for task in tasks:
            w_loaded_travel += task.loaded_distance
            w_empty_travel += task.empty_distance
            w_loaded_cost += task.loaded_distance * instance.w_loaded_cost_per_distance[vehicle_id]
            w_empty_cost += task.empty_distance * instance.w_empty_cost_per_distance[vehicle_id]

    f_outbound_cost = 0.0
    f_return_cost = 0.0
    f_travel = 0.0
    for vehicle_id, tasks in schedule.f_timelines.items():
        for task in tasks:
            f_travel += task.outbound_distance + task.return_distance
            f_outbound_cost += task.outbound_distance * instance.f_outbound_cost_per_distance[vehicle_id]
            f_return_cost += task.return_distance * instance.f_return_cost_per_distance[vehicle_id]
    total_cost = reconfiguration_cost + w_loaded_cost + w_empty_cost + f_outbound_cost + f_return_cost
    return ObjectiveBreakdown(
        makespan=makespan,
        reconfiguration_cost=reconfiguration_cost,
        w_loaded_cost=w_loaded_cost,
        w_empty_cost=w_empty_cost,
        f_outbound_cost=f_outbound_cost,
        f_return_cost=f_return_cost,
        total_cost=total_cost,
        reconfiguration_count=reconfiguration_count,
        w_loaded_travel=w_loaded_travel,
        w_empty_travel=w_empty_travel,
        f_travel=f_travel,
    )
