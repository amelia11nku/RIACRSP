from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.exact.general_gurobi import solve_general_gurobi


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def general_results():
    gp = pytest.importorskip("gurobipy")
    results = {}
    for name, optimum in (("tiny_01", 157.0), ("tiny_02", 57.0), ("tiny_03", 36.0)):
        instance = load_instance(ROOT / "instances/tiny" / f"{name}.json")
        try:
            result = solve_general_gurobi(
                instance, time_limit_seconds=30.0, seed=23, threads=1
            )
        except gp.GurobiError as error:
            if "license" in str(error).lower():
                pytest.skip(f"Gurobi license unavailable: {error}")
            raise
        assert result.solver_makespan == optimum
        results[name] = (instance, result)
    return results


def test_general_gurobi_proves_all_frozen_tiny_optima(general_results):
    for instance, result in general_results.values():
        assert result.status == "OPTIMAL"
        assert result.optimality_proven
        assert result.gap == 0.0
        assert result.solver_makespan == result.replay_makespan
        assert result.best_bound == pytest.approx(result.solver_makespan, abs=1e-6)
        assert result.runtime_seconds >= 0.0
        assert result.total_runtime_seconds >= result.runtime_seconds
        assert result.node_count >= 0.0
        assert result.h1_mip_start_used
        assert result.replay_feasible
        assert result.action_replay_feasible
        assert result.action_replay_matches_solver
        assert check_schedule(instance, result.schedule)["feasible"]
        assert json.loads(json.dumps(result.to_dict()))["status"] == "OPTIMAL"


def test_general_gurobi_handles_arbitrary_dag_and_island_choices(general_results):
    instance, result = general_results["tiny_01"]
    for product, sequence in result.product_sequences.items():
        position = {operation: index for index, operation in enumerate(sequence)}
        for source, target in instance.product_data[product].precedence:
            assert position[source] < position[target]
    assert all(
        island in instance.operation_data[operation].eligible_islands
        for operation, island in result.island_assignments.items()
    )
    assert any(vehicle is None for vehicle in result.w_assignments.values())
    assert any(len(sequence) > 1 for sequence in result.island_sequences.values())


def test_general_gurobi_covers_fixed_native_profile_without_special_case(general_results):
    instance, result = general_results["tiny_03"]
    assert result.variable_count > 0
    assert result.constraint_count > 0
    assert set(result.island_assignments.values()) == set(instance.islands)
    assert set(result.w_assignments.values()) == set(instance.agvs_w)
    assert set(result.f_assignments.values()) == set(instance.agvs_f)
