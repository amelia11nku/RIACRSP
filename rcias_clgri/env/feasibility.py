"""Independent, record-based schedule feasibility checker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from rcias_clgri.data.instance import Instance

from .objective import calculate_objective
from .schedule import Schedule

TOLERANCE = 1e-8


@dataclass(frozen=True)
class Violation:
    category: str
    resource: str
    operation_task_ids: tuple[str, ...]
    time_interval: tuple[float, float] | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def _violation(
    violations: list[Violation],
    category: str,
    resource: str,
    identifiers: Iterable[str],
    interval: tuple[float, float] | None,
    message: str,
) -> None:
    violations.append(Violation(category, resource, tuple(identifiers), interval, message))


def check_schedule(instance: Instance, schedule: Schedule) -> dict[str, object]:
    """Independently audit a partial or complete schedule against all model rules."""

    violations: list[Violation] = []
    scheduled = set(schedule.operation_schedules)
    expected = set(instance.operations)
    if scheduled != expected:
        missing = sorted(expected - scheduled)
        extra = sorted(scheduled - expected)
        _violation(
            violations, "COMPLETENESS", "SCHEDULE", [*missing, *extra], None,
            f"missing={missing}, extra={extra}",
        )

    # Operation records and synchronization.
    for op_id, record in schedule.operation_schedules.items():
        operation = instance.operation_data.get(op_id)
        if operation is None:
            continue
        interval = (record.start_time, record.completion_time)
        if record.product_id != operation.product_id:
            _violation(violations, "ID_CONSISTENCY", record.product_id, [op_id], interval, "wrong product")
        if record.island_id not in operation.eligible_islands:
            _violation(violations, "ASSIGNMENT", record.island_id, [op_id], interval, "ineligible island")
            continue
        if record.config_id != operation.required_config:
            _violation(violations, "CONFIGURATION", record.island_id, [op_id], interval, "wrong fixed configuration")
        duration = instance.processing_time[(op_id, record.island_id)]
        if not _close(record.completion_time - record.start_time, duration):
            _violation(violations, "PROCESSING_TIME", record.island_id, [op_id], interval, "incorrect duration")
        readiness = max(
            record.product_ready_time,
            record.config_ready_time,
            record.w_ready_time,
            record.f_ready_time,
        )
        if record.start_time + TOLERANCE < readiness:
            _violation(violations, "SYNCHRONIZATION", op_id, [op_id], interval, "operation starts before a readiness source")

    # Product realized paths and technological precedence.
    for product_id in instance.products:
        sequence = schedule.product_sequences.get(product_id, [])
        product_ops = set(instance.product_data[product_id].operations)
        if set(sequence) != product_ops or len(sequence) != len(product_ops):
            _violation(violations, "PRODUCT_PATH", product_id, sequence, None, "realized path does not cover product exactly once")
        position = {op_id: index for index, op_id in enumerate(sequence)}
        for source, target in instance.product_data[product_id].precedence:
            if source in position and target in position and position[source] >= position[target]:
                _violation(violations, "DAG_PRECEDENCE", product_id, [source, target], None, "realized order violates DAG")
            if source in scheduled and target in scheduled:
                left = schedule.operation_schedules[source]
                right = schedule.operation_schedules[target]
                if right.start_time + TOLERANCE < left.completion_time:
                    _violation(
                        violations, "DAG_PRECEDENCE", product_id, [source, target],
                        (left.completion_time, right.start_time), "temporal DAG precedence violated",
                    )
        for index, op_id in enumerate(sequence):
            if op_id not in scheduled:
                continue
            expected_predecessor = sequence[index - 1] if index else None
            if schedule.product_predecessor.get(op_id) != expected_predecessor:
                _violation(violations, "PRODUCT_PATH", product_id, [op_id], None, "incorrect direct predecessor")
            if expected_predecessor is not None and expected_predecessor in scheduled:
                predecessor = schedule.operation_schedules[expected_predecessor]
                current = schedule.operation_schedules[op_id]
                if current.start_time + TOLERANCE < predecessor.completion_time:
                    _violation(
                        violations, "PRODUCT_INDIVISIBILITY", product_id, [expected_predecessor, op_id],
                        (current.start_time, predecessor.completion_time), "same-product operations overlap",
                    )

    # Island processing and explicit reconfiguration occupation.
    for island_id in instance.islands:
        sequence = schedule.island_timelines.get(island_id, [])
        assigned = {op_id for op_id, record in schedule.operation_schedules.items() if record.island_id == island_id}
        if set(sequence) != assigned or len(sequence) != len(assigned):
            _violation(violations, "ISLAND_SEQUENCE", island_id, sequence, None, "timeline does not match assignments")
        previous_id: str | None = None
        previous_end = 0.0
        previous_config = instance.island_data[island_id].initial_config
        for op_id in sequence:
            if op_id not in scheduled:
                continue
            record = schedule.operation_schedules[op_id]
            setup = instance.reconfiguration_time[(island_id, previous_config, record.config_id)]
            expected_cost = instance.reconfiguration_cost[(island_id, previous_config, record.config_id)]
            if not _close(record.reconfiguration_start, previous_end) or not _close(record.reconfiguration_end, previous_end + setup):
                _violation(
                    violations, "RECONFIGURATION", island_id, [op_id],
                    (record.reconfiguration_start, record.reconfiguration_end), "incorrect transition interval",
                )
            if previous_config == record.config_id and (setup != 0 or expected_cost != 0):
                _violation(violations, "SAME_CONFIGURATION", island_id, [op_id], None, "same configuration is nonzero")
            if record.start_time + TOLERANCE < record.reconfiguration_end:
                _violation(
                    violations, "ISLAND_OCCUPANCY", island_id, [op_id],
                    (record.reconfiguration_end, record.start_time), "processing starts before setup completes",
                )
            if previous_id is not None and record.reconfiguration_start + TOLERANCE < previous_end:
                _violation(
                    violations, "ISLAND_OCCUPANCY", island_id, [previous_id, op_id],
                    (record.reconfiguration_start, previous_end), "island intervals overlap",
                )
            previous_id = op_id
            previous_end = record.completion_time
            previous_config = record.config_id

    # W tasks: exact generation from the realized product adjacency plus vehicle continuity.
    w_by_operation = {
        task.operation_id: task
        for tasks in schedule.w_timelines.values()
        for task in tasks
    }
    if sum(len(tasks) for tasks in schedule.w_timelines.values()) != len(w_by_operation):
        _violation(violations, "W_TASK", "W_FLEET", w_by_operation, None, "duplicate W task for an operation")
    for product_id, sequence in schedule.product_sequences.items():
        for index, op_id in enumerate(sequence):
            if op_id not in scheduled:
                continue
            predecessor_id = sequence[index - 1] if index else None
            pickup = "WH" if predecessor_id is None else schedule.operation_schedules[predecessor_id].island_id
            destination = schedule.operation_schedules[op_id].island_id
            task = w_by_operation.get(op_id)
            if pickup == destination:
                if task is not None:
                    _violation(violations, "W_SAME_ISLAND", destination, [op_id, task.task_id], None, "same-island task must not exist")
                if schedule.operation_schedules[op_id].w_task_id is not None:
                    _violation(violations, "W_SAME_ISLAND", destination, [op_id], None, "operation references a W task")
            elif task is None:
                _violation(violations, "W_CROSS_ISLAND", f"{pickup}->{destination}", [op_id], None, "required W task is missing")
            else:
                if task.pickup != pickup or task.destination != destination or task.predecessor_op != predecessor_id:
                    _violation(violations, "W_ROUTE", task.vehicle_id, [task.task_id], (task.loaded_start, task.arrival_time), "wrong realized route")
                release = 0.0 if predecessor_id is None else schedule.operation_schedules[predecessor_id].completion_time
                if task.loaded_start + TOLERANCE < release:
                    _violation(violations, "W_RELEASE", task.vehicle_id, [task.task_id], (task.loaded_start, release), "pickup before workpiece release")
                if schedule.operation_schedules[op_id].start_time + TOLERANCE < task.arrival_time:
                    _violation(violations, "W_SYNCHRONIZATION", task.vehicle_id, [task.task_id, op_id], None, "operation starts before W arrival")
    for vehicle_id, tasks in schedule.w_timelines.items():
        previous_location = "WH"
        previous_arrival = 0.0
        for task in tasks:
            expected_empty = instance.w_empty_time[(vehicle_id, previous_location, task.pickup)]
            expected_loaded = instance.w_loaded_time[(vehicle_id, task.pickup, task.destination)]
            if task.empty_origin != previous_location or not _close(task.empty_start, previous_arrival):
                _violation(violations, "W_EMPTY_REPOSITION", vehicle_id, [task.task_id], (task.empty_start, task.loaded_start), "location/time discontinuity")
            if not _close(task.empty_arrival, previous_arrival + expected_empty):
                _violation(violations, "W_EMPTY_REPOSITION", vehicle_id, [task.task_id], (previous_arrival, task.empty_arrival), "wrong empty travel")
            if task.loaded_start + TOLERANCE < task.empty_arrival:
                _violation(violations, "W_CAPACITY", vehicle_id, [task.task_id], (task.loaded_start, task.empty_arrival), "vehicle overlap")
            if not _close(task.arrival_time, task.loaded_start + expected_loaded):
                _violation(violations, "W_LOADED_TRAVEL", vehicle_id, [task.task_id], (task.loaded_start, task.arrival_time), "wrong loaded travel")
            previous_location = task.destination
            previous_arrival = task.arrival_time

    # F tasks: full round trip occupies the vehicle, arrival alone releases the operation.
    f_by_operation = {
        task.operation_id: task
        for tasks in schedule.f_timelines.values()
        for task in tasks
    }
    if sum(len(tasks) for tasks in schedule.f_timelines.values()) != len(f_by_operation):
        _violation(violations, "F_TASK", "F_FLEET", f_by_operation, None, "duplicate F task for an operation")
    for op_id, record in schedule.operation_schedules.items():
        task = f_by_operation.get(op_id)
        if task is None:
            _violation(violations, "F_TASK", op_id, [op_id], None, "component-kit task missing")
            continue
        if task.island_id != record.island_id:
            _violation(violations, "F_ROUTE", task.vehicle_id, [task.task_id, op_id], None, "kit delivered to wrong island")
        if record.start_time + TOLERANCE < task.arrival_island:
            _violation(violations, "F_SYNCHRONIZATION", task.vehicle_id, [task.task_id, op_id], None, "operation starts before kit arrival")
    for vehicle_id, tasks in schedule.f_timelines.items():
        previous_return = 0.0
        for task in tasks:
            outbound = instance.f_outbound_time[(vehicle_id, task.island_id)]
            return_time = instance.f_return_time[(vehicle_id, task.island_id)]
            if task.departure_wh + TOLERANCE < previous_return:
                _violation(violations, "F_CAPACITY", vehicle_id, [task.task_id], (task.departure_wh, previous_return), "round trips overlap")
            if not _close(task.arrival_island, task.departure_wh + outbound):
                _violation(violations, "F_OUTBOUND", vehicle_id, [task.task_id], None, "wrong kit arrival")
            if not _close(task.return_wh, task.arrival_island + return_time):
                _violation(violations, "F_RETURN", vehicle_id, [task.task_id], None, "wrong return availability")
            previous_return = task.return_wh

    objective = calculate_objective(instance, schedule)
    return {
        "feasible": not violations,
        "violations": [violation.to_dict() for violation in violations],
        "makespan": objective.makespan,
        "cost": objective.total_cost,
        "objective": objective.to_dict(),
    }
