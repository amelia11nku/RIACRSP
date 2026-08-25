"""Mutable schedule records produced by the deterministic decoder."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OperationSchedule:
    op_id: str
    product_id: str
    island_id: str
    config_id: str
    product_predecessor: str | None
    processing_time: int
    product_ready_time: float
    island_ready_time: float
    config_ready_time: float
    w_ready_time: float
    f_ready_time: float
    reconfiguration_start: float
    reconfiguration_end: float
    start_time: float
    completion_time: float
    binding_resource: tuple[str, ...]
    w_task_id: str | None
    f_task_id: str


@dataclass
class WTask:
    task_id: str
    vehicle_id: str
    product_id: str
    predecessor_op: str | None
    operation_id: str
    pickup: str
    destination: str
    release_time: float
    empty_origin: str
    empty_start: float
    empty_arrival: float
    loaded_start: float
    arrival_time: float
    empty_travel_time: float
    loaded_travel_time: float
    empty_distance: float
    loaded_distance: float


@dataclass
class FTask:
    task_id: str
    vehicle_id: str
    operation_id: str
    island_id: str
    departure_wh: float
    arrival_island: float
    return_wh: float
    outbound_time: float
    return_time: float
    outbound_distance: float
    return_distance: float


@dataclass
class Schedule:
    """One complete or partial constructive schedule."""

    instance_id: str
    operation_schedules: dict[str, OperationSchedule]
    product_sequences: dict[str, list[str]]
    product_predecessor: dict[str, str | None]
    product_successor: dict[str, str | None]
    island_timelines: dict[str, list[str]]
    w_timelines: dict[str, list[WTask]]
    f_timelines: dict[str, list[FTask]]
    accumulated_reconfiguration_cost: float = 0.0

    def clone(self) -> "Schedule":
        """Copy one small search state; production decoder probes remain copy-free."""

        return copy.deepcopy(self)

    @property
    def scheduled_operations(self) -> frozenset[str]:
        return frozenset(self.operation_schedules)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable schedule representation."""

        return {
            "instance_id": self.instance_id,
            "operations": {op_id: asdict(record) for op_id, record in self.operation_schedules.items()},
            "product_sequences": copy.deepcopy(self.product_sequences),
            "product_predecessor": dict(self.product_predecessor),
            "product_successor": dict(self.product_successor),
            "island_timelines": copy.deepcopy(self.island_timelines),
            "w_timelines": {
                vehicle: [asdict(task) for task in tasks] for vehicle, tasks in self.w_timelines.items()
            },
            "f_timelines": {
                vehicle: [asdict(task) for task in tasks] for vehicle, tasks in self.f_timelines.items()
            },
            "accumulated_reconfiguration_cost": self.accumulated_reconfiguration_cost,
        }
