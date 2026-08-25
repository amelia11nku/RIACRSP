"""Stepwise topological construction environment for future learning policies."""

from __future__ import annotations

from typing import Any

from rcias_clgri.data.instance import Instance

from .insertion_decoder import Action, ActionProbe, InsertionDecoder
from .objective import ObjectiveBreakdown, calculate_objective
from .schedule import Schedule


class RCIASConstructionEnv:
    """Deterministic construction API with hard feasibility masks."""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.decoder = InsertionDecoder(instance)
        self.schedule = self.decoder.empty_schedule()

    def reset(self) -> Schedule:
        self.schedule = self.decoder.empty_schedule()
        return self.schedule

    def get_state(self) -> Schedule:
        return self.schedule

    def get_ready_operations(self) -> tuple[str, ...]:
        selected = self.schedule.scheduled_operations
        return tuple(
            op_id for op_id in self.instance.operations
            if op_id not in selected and self.instance.predecessors[op_id] <= selected
        )

    def get_feasible_actions(self) -> dict[str, tuple[str, ...]]:
        return {op_id: self.get_eligible_islands(op_id) for op_id in self.get_ready_operations()}

    def get_eligible_islands(self, op_id: str) -> tuple[str, ...]:
        if op_id not in self.get_ready_operations():
            return ()
        return self.instance.operation_data[op_id].eligible_islands

    def get_feasible_w_agvs(self, op_id: str, island_id: str) -> tuple[str | None, ...]:
        if island_id not in self.get_eligible_islands(op_id):
            return ()
        product_id = self.instance.product_of[op_id]
        sequence = self.schedule.product_sequences[product_id]
        pickup = "WH" if not sequence else self.schedule.operation_schedules[sequence[-1]].island_id
        return (None,) if pickup == island_id else self.instance.agvs_w

    def get_feasible_f_agvs(self, op_id: str, island_id: str) -> tuple[str, ...]:
        if island_id not in self.get_eligible_islands(op_id):
            return ()
        return self.instance.agvs_f

    def probe(self, action: Action) -> ActionProbe:
        if action.operation_id not in self.get_ready_operations():
            raise ValueError(f"operation is not topologically ready: {action.operation_id}")
        if action.w_agv_id not in self.get_feasible_w_agvs(action.operation_id, action.island_id):
            raise ValueError("W action violates the hard mask")
        if action.f_agv_id not in self.get_feasible_f_agvs(action.operation_id, action.island_id):
            raise ValueError("F action violates the hard mask")
        return self.decoder.probe_action(self.schedule, action)

    def step(self, action: Action) -> dict[str, Any]:
        probe = self.probe(action)
        record = self.decoder.commit_action(self.schedule, probe)
        return {
            "operation": record,
            "done": len(self.schedule.operation_schedules) == self.instance.num_operations,
            "ready_operations": self.get_ready_operations(),
        }

    @property
    def done(self) -> bool:
        return len(self.schedule.operation_schedules) == self.instance.num_operations

    def objective(self) -> ObjectiveBreakdown:
        """Return an independently recomputed objective breakdown."""

        return calculate_objective(self.instance, self.schedule)
