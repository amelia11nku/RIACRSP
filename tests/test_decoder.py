from __future__ import annotations

from rcias_clgri.env.feasibility import check_schedule


def test_complete_decoder_records_all_readiness_sources(controlled_env):
    schedule = controlled_env.schedule
    assert len(schedule.operation_schedules) == controlled_env.instance.num_operations
    for record in schedule.operation_schedules.values():
        assert record.start_time >= max(
            record.product_ready_time,
            record.config_ready_time,
            record.w_ready_time,
            record.f_ready_time,
        )
        assert record.binding_resource
        assert record.completion_time - record.start_time == record.processing_time
    assert check_schedule(controlled_env.instance, schedule)["feasible"]


def test_product_predecessor_is_realized_not_technological(controlled_env):
    schedule = controlled_env.schedule
    # o12 is not a technological predecessor of o13, but is its realized direct predecessor.
    assert "o12" not in controlled_env.instance.predecessors["o13"]
    assert schedule.product_predecessor["o13"] == "o12"
    assert schedule.product_successor["o12"] == "o13"
