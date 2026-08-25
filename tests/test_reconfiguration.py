from __future__ import annotations


def test_configuration_change_and_same_configuration(controlled_env):
    instance = controlled_env.instance
    schedule = controlled_env.schedule
    first = schedule.operation_schedules["o21"]
    second = schedule.operation_schedules["o22"]
    assert first.config_id == second.config_id == "C4"
    assert second.reconfiguration_end == second.reconfiguration_start
    changed = schedule.operation_schedules["o12"]
    assert changed.config_id != schedule.operation_schedules["o11"].config_id
    assert changed.reconfiguration_end - changed.reconfiguration_start == instance.reconfiguration_time[("M1", "C1", "C2")]


def test_island_processing_and_reconfiguration_intervals_do_not_overlap(controlled_env):
    schedule = controlled_env.schedule
    for sequence in schedule.island_timelines.values():
        previous_end = 0.0
        for op_id in sequence:
            record = schedule.operation_schedules[op_id]
            assert record.reconfiguration_start >= previous_end
            assert record.start_time >= record.reconfiguration_end
            previous_end = record.completion_time
