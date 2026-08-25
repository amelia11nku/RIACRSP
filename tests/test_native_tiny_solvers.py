from __future__ import annotations

import json
from pathlib import Path

from rcias_clgri.data.generation import deterministic_json_text
from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.exact.native_tiny_solvers import solve_with_cp_sat
from scripts.generate_tiny_suite import build_multi_vehicle_tiny

ROOT = Path(__file__).resolve().parents[1]
INSTANCE_PATH = ROOT / "instances" / "tiny" / "tiny_03.json"
RUN = ROOT / "outputs" / "validation" / "tiny_03" / "run_1"


def test_tiny03_dimensions_and_reproducibility():
    instance = load_instance(INSTANCE_PATH)
    assert (len(instance.islands), len(instance.agvs_w), len(instance.agvs_f)) == (4, 2, 2)
    assert len(instance.operations) == 4
    regenerated = build_multi_vehicle_tiny(seed=23)
    assert deterministic_json_text(regenerated).encode("utf-8") == INSTANCE_PATH.read_bytes()


def test_cp_sat_proves_optimal_and_replays_feasibly():
    instance = load_instance(INSTANCE_PATH)
    result = solve_with_cp_sat(instance, time_limit_seconds=30.0, seed=23)
    assert result.status == "OPTIMAL"
    assert result.solver_makespan == result.best_bound == result.replay_makespan == 36.0
    assert result.gap == 0.0
    assert result.replay_feasible
    assert check_schedule(instance, result.schedule)["feasible"]
    assert set(result.w_assignments.values()) == set(instance.agvs_w)
    assert set(result.f_assignments.values()) == set(instance.agvs_f)


def test_gurobi_and_cp_sat_formal_outputs_agree():
    info = json.loads((RUN / "final_info.json").read_text(encoding="utf-8"))
    assert info["tiny_exact_validated"]
    assert info["dimensions"] == {
        "products": 2, "operations": 4, "islands": 4, "w_agvs": 2, "f_agvs": 2,
    }
    assert {row["backend"] for row in info["solvers"]} == {
        "gurobi-milp", "ortools-cp-sat",
    }
    assert all(row["status"] == "OPTIMAL" for row in info["solvers"])
    assert all(row["solver_makespan"] == row["best_bound"] == 36.0 for row in info["solvers"])
    assert all(row["gap"] == 0.0 and row["replay_feasible"] for row in info["solvers"])
    assert info["comparison"]["objective_equal"]
    assert info["comparison"]["decoder_replay_equal"]
    assert info["comparison"]["independent_feasibility"]
    assert info["comparison"]["same_replay_schedule"]
    assert info["exhaustive_reference"]["makespan"] == 36.0
    for relative in (
        "Figure_1_objective_bound.png", "Figure_2_solver_runtime.png",
        "agreed_schedule/gantt.png",
    ):
        assert (RUN / relative).stat().st_size > 30_000
