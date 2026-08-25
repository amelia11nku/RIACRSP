from __future__ import annotations

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.env.timelines import probe_w_insertion


def test_same_island_has_no_w_and_cross_island_has_w(controlled_env):
    records = controlled_env.schedule.operation_schedules
    assert records["o12"].w_task_id is None
    assert records["o22"].w_task_id is None
    assert records["o13"].w_task_id == "W:o13"
    assert records["o23"].w_task_id == "W:o23"


def test_w_empty_reposition_is_recorded(controlled_env):
    tasks = controlled_env.schedule.w_timelines["W1"]
    assert any(task.empty_distance > 0 for task in tasks[1:])
    previous_location = "WH"
    for task in tasks:
        assert task.empty_origin == previous_location
        previous_location = task.destination


def test_w_probe_can_insert_into_middle_gap(fjsp_instance):
    schedule = InsertionDecoder(fjsp_instance).empty_schedule()
    first = probe_w_insertion(
        fjsp_instance, schedule, "a", "J1", None, "WH", "M1", 0.0, "W1"
    )
    schedule.w_timelines["W1"].append(first.task)
    late = probe_w_insertion(
        fjsp_instance, schedule, "c", "J2", None, "WH", "M2", 200.0, "W1"
    )
    schedule.w_timelines["W1"].append(late.task)
    middle = probe_w_insertion(
        fjsp_instance, schedule, "b", "J1", "a", "M1", "M2", first.task.arrival_time, "W1"
    )
    assert middle.insert_index == 1
    assert middle.task.arrival_time + fjsp_instance.w_empty_time[("W1", "M2", "WH")] <= late.task.loaded_start
