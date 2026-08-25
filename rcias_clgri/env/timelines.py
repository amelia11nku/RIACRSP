"""Side-effect-free insertion probes for island and AGV timelines."""

from __future__ import annotations

from dataclasses import dataclass

from rcias_clgri.data.instance import Instance

from .schedule import FTask, OperationSchedule, Schedule, WTask

EPSILON = 1e-9


@dataclass(frozen=True)
class FProbe:
    insert_index: int
    task: FTask


@dataclass(frozen=True)
class WProbe:
    insert_index: int
    task: WTask


@dataclass(frozen=True)
class MachineProbe:
    insert_index: int
    island_id: str
    reconfiguration_start: float
    reconfiguration_end: float
    processing_start: float
    processing_end: float
    setup_before: float
    setup_after: float
    incremental_reconfiguration_cost: float
    previous_operation: str | None
    next_operation: str | None


def probe_f_insertion(
    instance: Instance,
    schedule: Schedule,
    op_id: str,
    island_id: str,
    vehicle_id: str,
) -> FProbe:
    """Find the earliest free round-trip gap for a component-kit task."""

    outbound = instance.f_outbound_time[(vehicle_id, island_id)]
    return_time = instance.f_return_time[(vehicle_id, island_id)]
    duration = outbound + return_time
    tasks = schedule.f_timelines[vehicle_id]
    earliest = 0.0
    index = len(tasks)
    for candidate_index, next_task in enumerate(tasks):
        if earliest + duration <= next_task.departure_wh + EPSILON:
            index = candidate_index
            break
        earliest = max(earliest, next_task.return_wh)
    task = FTask(
        task_id=f"F:{op_id}",
        vehicle_id=vehicle_id,
        operation_id=op_id,
        island_id=island_id,
        departure_wh=earliest,
        arrival_island=earliest + outbound,
        return_wh=earliest + duration,
        outbound_time=float(outbound),
        return_time=float(return_time),
        outbound_distance=instance.distance[("WH", island_id)],
        return_distance=instance.distance[(island_id, "WH")],
    )
    return FProbe(index, task)


def probe_w_insertion(
    instance: Instance,
    schedule: Schedule,
    op_id: str,
    product_id: str,
    predecessor_op: str | None,
    pickup: str,
    destination: str,
    release_time: float,
    vehicle_id: str,
) -> WProbe:
    """Find the earliest feasible W task insertion, including empty reposition."""

    if pickup == destination:
        raise ValueError("same-island adjacency must not create a W task")
    tasks = schedule.w_timelines[vehicle_id]
    loaded_duration = instance.w_loaded_time[(vehicle_id, pickup, destination)]
    for index in range(len(tasks) + 1):
        previous = tasks[index - 1] if index else None
        following = tasks[index] if index < len(tasks) else None
        empty_origin = previous.destination if previous else "WH"
        previous_arrival = previous.arrival_time if previous else 0.0
        empty_duration = instance.w_empty_time[(vehicle_id, empty_origin, pickup)]
        empty_arrival = previous_arrival + empty_duration
        loaded_start = max(release_time, empty_arrival)
        arrival = loaded_start + loaded_duration
        if following is not None:
            empty_to_following = instance.w_empty_time[(vehicle_id, destination, following.pickup)]
            if arrival + empty_to_following > following.loaded_start + EPSILON:
                continue
        task = WTask(
            task_id=f"W:{op_id}",
            vehicle_id=vehicle_id,
            product_id=product_id,
            predecessor_op=predecessor_op,
            operation_id=op_id,
            pickup=pickup,
            destination=destination,
            release_time=release_time,
            empty_origin=empty_origin,
            empty_start=previous_arrival,
            empty_arrival=empty_arrival,
            loaded_start=loaded_start,
            arrival_time=arrival,
            empty_travel_time=float(empty_duration),
            loaded_travel_time=float(loaded_duration),
            empty_distance=instance.distance[(empty_origin, pickup)],
            loaded_distance=instance.distance[(pickup, destination)],
        )
        return WProbe(index, task)
    raise RuntimeError(f"no W insertion gap for {op_id} on {vehicle_id}")


def probe_machine_insertion(
    instance: Instance,
    schedule: Schedule,
    op_id: str,
    island_id: str,
    base_ready: float,
) -> MachineProbe:
    """Find the earliest processing gap while preserving both adjacent setups."""

    config = instance.operation_data[op_id].required_config
    processing = instance.processing_time[(op_id, island_id)]
    sequence = schedule.island_timelines[island_id]
    for index in range(len(sequence) + 1):
        previous_id = sequence[index - 1] if index else None
        next_id = sequence[index] if index < len(sequence) else None
        previous_end = schedule.operation_schedules[previous_id].completion_time if previous_id else 0.0
        previous_config = (
            schedule.operation_schedules[previous_id].config_id
            if previous_id else instance.island_data[island_id].initial_config
        )
        setup_before = instance.reconfiguration_time[(island_id, previous_config, config)]
        config_ready = previous_end + setup_before
        start = max(base_ready, config_ready)
        end = start + processing
        if next_id is None:
            setup_after = 0.0
        else:
            next_record = schedule.operation_schedules[next_id]
            setup_after = instance.reconfiguration_time[(island_id, config, next_record.config_id)]
            if end + setup_after > next_record.start_time + EPSILON:
                continue

        new_before_cost = instance.reconfiguration_cost[(island_id, previous_config, config)]
        if next_id is None:
            incremental_cost = new_before_cost
        else:
            next_config = schedule.operation_schedules[next_id].config_id
            new_after_cost = instance.reconfiguration_cost[(island_id, config, next_config)]
            old_cost = instance.reconfiguration_cost[(island_id, previous_config, next_config)]
            incremental_cost = new_before_cost + new_after_cost - old_cost
        return MachineProbe(
            insert_index=index,
            island_id=island_id,
            reconfiguration_start=previous_end,
            reconfiguration_end=config_ready,
            processing_start=start,
            processing_end=end,
            setup_before=float(setup_before),
            setup_after=float(setup_after),
            incremental_reconfiguration_cost=float(incremental_cost),
            previous_operation=previous_id,
            next_operation=next_id,
        )
    raise RuntimeError(f"no machine insertion gap for {op_id}@{island_id}")


def refresh_w_empty_legs(instance: Instance, schedule: Schedule, vehicle_id: str) -> None:
    """Recompute empty-leg records after a local middle insertion."""

    previous_location = "WH"
    previous_arrival = 0.0
    for task in schedule.w_timelines[vehicle_id]:
        empty_duration = instance.w_empty_time[(vehicle_id, previous_location, task.pickup)]
        task.empty_origin = previous_location
        task.empty_start = previous_arrival
        task.empty_arrival = previous_arrival + empty_duration
        task.empty_travel_time = float(empty_duration)
        task.empty_distance = instance.distance[(previous_location, task.pickup)]
        previous_location = task.destination
        previous_arrival = task.arrival_time


def refresh_island_readiness(instance: Instance, schedule: Schedule, island_id: str) -> None:
    """Refresh direct-transition and binding records after a machine insertion."""

    previous_id: str | None = None
    previous_config = instance.island_data[island_id].initial_config
    previous_end = 0.0
    for op_id in schedule.island_timelines[island_id]:
        record = schedule.operation_schedules[op_id]
        setup = instance.reconfiguration_time[(island_id, previous_config, record.config_id)]
        record.island_ready_time = previous_end
        record.reconfiguration_start = previous_end
        record.reconfiguration_end = previous_end + setup
        record.config_ready_time = record.reconfiguration_end
        readiness = {
            "PRODUCT": record.product_ready_time,
            "ISLAND_CONFIG": record.config_ready_time,
            "W_AGV": record.w_ready_time,
            "F_AGV": record.f_ready_time,
        }
        binding = tuple(name for name, value in readiness.items() if abs(value - record.start_time) <= EPSILON)
        record.binding_resource = binding or ("INSERTION_SLOT",)
        previous_id = op_id
        previous_config = record.config_id
        previous_end = record.completion_time
