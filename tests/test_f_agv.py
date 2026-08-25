from __future__ import annotations

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.env.schedule import FTask
from rcias_clgri.env.timelines import probe_f_insertion


def test_f_arrival_and_vehicle_return_are_distinct(controlled_env):
    records = controlled_env.schedule.operation_schedules
    tasks = controlled_env.schedule.f_timelines["F1"]
    assert all(task.arrival_island < task.return_wh for task in tasks)
    assert any(records[task.operation_id].start_time < task.return_wh for task in tasks)


def test_f_round_trips_do_not_overlap(controlled_env):
    tasks = controlled_env.schedule.f_timelines["F1"]
    assert all(left.return_wh <= right.departure_wh for left, right in zip(tasks, tasks[1:]))


def test_f_probe_uses_middle_gap(fjsp_instance):
    schedule = InsertionDecoder(fjsp_instance).empty_schedule()
    vehicle, island = "F1", "M1"
    outbound = fjsp_instance.f_outbound_time[(vehicle, island)]
    return_time = fjsp_instance.f_return_time[(vehicle, island)]
    duration = outbound + return_time
    schedule.f_timelines[vehicle] = [
        FTask("old1", vehicle, "x", island, 0.0, float(outbound), float(duration), float(outbound), float(return_time), 0.0, 0.0),
        FTask("old2", vehicle, "y", island, float(3 * duration), float(3 * duration + outbound), float(4 * duration), float(outbound), float(return_time), 0.0, 0.0),
    ]
    probe = probe_f_insertion(fjsp_instance, schedule, "z", island, vehicle)
    assert probe.insert_index == 1
    assert probe.task.departure_wh == duration
