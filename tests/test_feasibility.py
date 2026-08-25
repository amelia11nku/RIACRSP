from __future__ import annotations

from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.objective import calculate_objective


def test_independent_checker_and_objective_agree(controlled_env):
    result = check_schedule(controlled_env.instance, controlled_env.schedule)
    objective = calculate_objective(controlled_env.instance, controlled_env.schedule)
    assert result["feasible"]
    assert result["makespan"] == objective.makespan == controlled_env.objective().makespan
    assert result["cost"] == objective.total_cost == controlled_env.objective().total_cost


def test_checker_rejects_product_overlap(controlled_env):
    schedule = controlled_env.schedule.clone()
    current = schedule.operation_schedules["o12"]
    predecessor = schedule.operation_schedules["o11"]
    current.start_time = predecessor.start_time
    current.completion_time = current.start_time + current.processing_time
    result = check_schedule(controlled_env.instance, schedule)
    assert not result["feasible"]
    categories = {violation["category"] for violation in result["violations"]}
    assert "PRODUCT_INDIVISIBILITY" in categories or "DAG_PRECEDENCE" in categories
