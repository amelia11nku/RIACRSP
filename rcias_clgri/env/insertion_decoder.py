"""Deterministic, side-effect-free probe and local-commit decoder."""

from __future__ import annotations

from dataclasses import dataclass

from rcias_clgri.data.instance import Instance

from .schedule import OperationSchedule, Schedule
from .timelines import (
    FProbe,
    MachineProbe,
    WProbe,
    probe_f_insertion,
    probe_machine_insertion,
    probe_w_insertion,
    refresh_island_readiness,
    refresh_w_empty_legs,
)


@dataclass(frozen=True)
class Action:
    operation_id: str
    island_id: str
    w_agv_id: str | None
    f_agv_id: str


@dataclass(frozen=True)
class ActionProbe:
    action: Action
    product_predecessor: str | None
    product_ready: float
    w_ready: float
    f_probe: FProbe
    w_probe: WProbe | None
    machine_probe: MachineProbe

    @property
    def predicted_completion(self) -> float:
        return self.machine_probe.processing_end


class InsertionDecoder:
    """Construct schedules without hidden state or whole-schedule candidate copies."""

    def __init__(self, instance: Instance):
        self.instance = instance

    def empty_schedule(self) -> Schedule:
        return Schedule(
            instance_id=self.instance.instance_id,
            operation_schedules={},
            product_sequences={product_id: [] for product_id in self.instance.products},
            product_predecessor={},
            product_successor={},
            island_timelines={island_id: [] for island_id in self.instance.islands},
            w_timelines={agv_id: [] for agv_id in self.instance.agvs_w},
            f_timelines={agv_id: [] for agv_id in self.instance.agvs_f},
        )

    def probe_action(self, schedule: Schedule, action: Action) -> ActionProbe:
        """Evaluate an action without mutating the partial schedule."""

        op_id = action.operation_id
        if op_id in schedule.operation_schedules:
            raise ValueError(f"operation already scheduled: {op_id}")
        operation = self.instance.operation_data.get(op_id)
        if operation is None:
            raise ValueError(f"unknown operation: {op_id}")
        if action.island_id not in operation.eligible_islands:
            raise ValueError(f"ineligible island {action.island_id} for {op_id}")
        if action.f_agv_id not in self.instance.agvs_f:
            raise ValueError(f"unknown F-AGV: {action.f_agv_id}")
        product_sequence = schedule.product_sequences[operation.product_id]
        predecessor = product_sequence[-1] if product_sequence else None
        if predecessor is None:
            pickup, product_ready = "WH", 0.0
        else:
            predecessor_record = schedule.operation_schedules[predecessor]
            pickup = predecessor_record.island_id
            product_ready = predecessor_record.completion_time

        if pickup == action.island_id:
            if action.w_agv_id is not None:
                raise ValueError("W action must be NONE for a same-island product adjacency")
            w_probe = None
            w_ready = product_ready
        else:
            if action.w_agv_id not in self.instance.agvs_w:
                raise ValueError("a valid W-AGV is required for warehouse/cross-island transport")
            w_probe = probe_w_insertion(
                self.instance,
                schedule,
                op_id,
                operation.product_id,
                predecessor,
                pickup,
                action.island_id,
                product_ready,
                action.w_agv_id,
            )
            w_ready = w_probe.task.arrival_time
        f_probe = probe_f_insertion(
            self.instance, schedule, op_id, action.island_id, action.f_agv_id
        )
        base_ready = max(product_ready, w_ready, f_probe.task.arrival_island)
        machine_probe = probe_machine_insertion(
            self.instance, schedule, op_id, action.island_id, base_ready
        )
        return ActionProbe(
            action=action,
            product_predecessor=predecessor,
            product_ready=product_ready,
            w_ready=w_ready,
            f_probe=f_probe,
            w_probe=w_probe,
            machine_probe=machine_probe,
        )

    def commit_action(self, schedule: Schedule, probe: ActionProbe) -> OperationSchedule:
        """Commit exactly the local changes described by a previously computed probe."""

        action = probe.action
        op_id = action.operation_id
        operation = self.instance.operation_data[op_id]
        if op_id in schedule.operation_schedules:
            raise ValueError(f"operation already scheduled: {op_id}")

        f_task = probe.f_probe.task
        schedule.f_timelines[action.f_agv_id].insert(probe.f_probe.insert_index, f_task)
        w_task_id: str | None = None
        if probe.w_probe is not None:
            w_task = probe.w_probe.task
            schedule.w_timelines[action.w_agv_id].insert(probe.w_probe.insert_index, w_task)  # type: ignore[index]
            refresh_w_empty_legs(self.instance, schedule, action.w_agv_id)  # type: ignore[arg-type]
            w_task_id = w_task.task_id

        machine = probe.machine_probe
        record = OperationSchedule(
            op_id=op_id,
            product_id=operation.product_id,
            island_id=action.island_id,
            config_id=operation.required_config,
            product_predecessor=probe.product_predecessor,
            processing_time=self.instance.processing_time[(op_id, action.island_id)],
            product_ready_time=probe.product_ready,
            island_ready_time=machine.reconfiguration_start,
            config_ready_time=machine.reconfiguration_end,
            w_ready_time=probe.w_ready,
            f_ready_time=f_task.arrival_island,
            reconfiguration_start=machine.reconfiguration_start,
            reconfiguration_end=machine.reconfiguration_end,
            start_time=machine.processing_start,
            completion_time=machine.processing_end,
            binding_resource=(),
            w_task_id=w_task_id,
            f_task_id=f_task.task_id,
        )
        schedule.operation_schedules[op_id] = record
        schedule.island_timelines[action.island_id].insert(machine.insert_index, op_id)
        schedule.accumulated_reconfiguration_cost += machine.incremental_reconfiguration_cost

        product_sequence = schedule.product_sequences[operation.product_id]
        if probe.product_predecessor is not None:
            schedule.product_successor[probe.product_predecessor] = op_id
        product_sequence.append(op_id)
        schedule.product_predecessor[op_id] = probe.product_predecessor
        schedule.product_successor[op_id] = None
        refresh_island_readiness(self.instance, schedule, action.island_id)
        return record
