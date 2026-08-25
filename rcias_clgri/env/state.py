"""Lightweight views of mutable resource state."""

from __future__ import annotations

from dataclasses import dataclass

from rcias_clgri.data.instance import Instance

from .schedule import Schedule


@dataclass(frozen=True)
class IslandState:
    island_id: str
    current_configuration: str
    available_time: float
    last_operation: str | None
    scheduled_intervals: tuple[tuple[float, float, str, str], ...]


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    current_location: str
    available_time: float
    task_ids: tuple[str, ...]


def get_island_state(instance: Instance, schedule: Schedule, island_id: str) -> IslandState:
    sequence = schedule.island_timelines[island_id]
    if sequence:
        last = schedule.operation_schedules[sequence[-1]]
        configuration = last.config_id
        available = last.completion_time
        last_operation = last.op_id
    else:
        configuration = instance.island_data[island_id].initial_config
        available = 0.0
        last_operation = None
    intervals: list[tuple[float, float, str, str]] = []
    for op_id in sequence:
        record = schedule.operation_schedules[op_id]
        if record.reconfiguration_end > record.reconfiguration_start:
            intervals.append((record.reconfiguration_start, record.reconfiguration_end, "RECONFIG", op_id))
        intervals.append((record.start_time, record.completion_time, "PROCESS", op_id))
    return IslandState(island_id, configuration, available, last_operation, tuple(intervals))


def get_w_vehicle_state(schedule: Schedule, vehicle_id: str) -> VehicleState:
    tasks = schedule.w_timelines[vehicle_id]
    if not tasks:
        return VehicleState(vehicle_id, "WH", 0.0, ())
    last = tasks[-1]
    return VehicleState(vehicle_id, last.destination, last.arrival_time, tuple(task.task_id for task in tasks))


def get_f_vehicle_state(schedule: Schedule, vehicle_id: str) -> VehicleState:
    tasks = schedule.f_timelines[vehicle_id]
    if not tasks:
        return VehicleState(vehicle_id, "WH", 0.0, ())
    return VehicleState(vehicle_id, "WH", tasks[-1].return_wh, tuple(task.task_id for task in tasks))
