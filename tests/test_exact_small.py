from __future__ import annotations

import pytest

from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact
from rcias_clgri.heuristic.dispatching import solve_dispatching


@pytest.mark.parametrize("fixture_name,expected", [("fjsp_instance", 57.0), ("automotive_instance", 157.0)])
def test_exact_tiny_matches_best_decoder(request, fixture_name, expected):
    instance = request.getfixturevalue(fixture_name)
    exact = solve_tiny_exact(instance, time_limit_seconds=30.0)
    heuristic = solve_dispatching(instance, "H2")
    assert exact.status == "OPTIMAL"
    assert exact.best_value == expected
    assert heuristic.objective.makespan == exact.objective.makespan
    assert check_schedule(instance, exact.schedule)["feasible"]
