from __future__ import annotations

import csv
import json
from pathlib import Path

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "validation" / "tiny_01"


def test_tiny_exact_replays_to_same_objective():
    instance = load_instance(ROOT / "instances" / "tiny" / "tiny_01.json")
    exact = solve_tiny_exact(instance, time_limit_seconds=30.0)
    replay = RCIASConstructionEnv(instance)
    for action in exact.actions:
        replay.step(action)
    assert exact.status == "OPTIMAL"
    assert exact.objective.makespan == replay.objective().makespan == 157.0
    assert check_schedule(instance, replay.schedule)["feasible"]


def test_exported_csvs_come_from_solution_schedule():
    solution = json.loads((OUTPUT / "solution.json").read_text(encoding="utf-8"))
    with (OUTPUT / "operation_schedule.csv").open(encoding="utf-8", newline="") as handle:
        operations = list(csv.DictReader(handle))
    with (OUTPUT / "resource_timeline.csv").open(encoding="utf-8", newline="") as handle:
        resources = list(csv.DictReader(handle))
    assert len(operations) == len(solution["schedule"]["operations"]) == 6
    assert any(row["activity_type"] == "EMPTY_REPOSITION" for row in resources)
    assert any(row["activity_type"] == "RECONFIGURATION" for row in resources)
    same_island = next(row for row in operations if row["operation"] == "o21")
    assert same_island["W_required"] == "False"
    assert same_island["W_AGV"] == "NONE"


def test_tiny_feasibility_categories_all_pass():
    report = json.loads((OUTPUT / "feasibility_report.json").read_text(encoding="utf-8"))
    assert report["feasible"]
    assert len(report["categories"]) == 13
    assert all(item["status"] == "PASS" for item in report["categories"])
    assert report["exact_replay_equal"]


def test_gantt_outputs_exist_and_are_nonempty():
    assert (OUTPUT / "gantt.png").stat().st_size > 50_000
    assert (OUTPUT / "gantt.pdf").stat().st_size > 10_000
