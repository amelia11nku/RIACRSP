#!/usr/bin/env python3
"""Solve tiny_01 exactly, replay it, export records, and plot one auditable Gantt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.export import ResourceTimelineExporter
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.objective import calculate_objective
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.exact.tiny_exact_solver import gurobi_available, solve_tiny_exact
from scripts.plot_schedule_gantt import plot_resource_timeline

INSTANCE_PATH = ROOT / "instances" / "tiny" / "tiny_01.json"
OUTPUT = ROOT / "outputs" / "validation" / "tiny_01"

CATEGORIES = (
    "DAG precedence", "product indivisibility", "island non-overlap",
    "reconfiguration correctness", "same-configuration zero-reconfiguration",
    "W same-island no transport", "W cross-island transport", "W empty reposition",
    "W vehicle non-overlap", "F vehicle non-overlap", "F arrival synchronization",
    "processing start synchronization", "objective recomputation",
)


def _summary(instance, schedule, objective, exact) -> str:
    lines = ["# Tiny 01 Exact Schedule Validation", "", f"Solver: `{exact.backend}`", "",
             f"Status: `{exact.status}`", "", f"Optimal makespan: `{objective.makespan}`", ""]
    w_by_op = {task.operation_id: task for tasks in schedule.w_timelines.values() for task in tasks}
    f_by_op = {task.operation_id: task for tasks in schedule.f_timelines.values() for task in tasks}
    for product_id, sequence in schedule.product_sequences.items():
        lines.extend([f"## {product_id}", ""])
        for op_id in sequence:
            record = schedule.operation_schedules[op_id]
            w_task = w_by_op.get(op_id)
            f_task = f_by_op[op_id]
            lines.append(f"- `{op_id} @ {record.island_id}`; config `{record.config_id}`; process `[{record.start_time}, {record.completion_time}]`.")
            lines.append(
                f"  W: `{'same island — NONE' if w_task is None else f'{w_task.pickup} → {w_task.destination} by {w_task.vehicle_id} [{w_task.loaded_start}, {w_task.arrival_time}]'}`."
            )
            lines.append(
                f"  F: `WH → {record.island_id} → WH` by `{f_task.vehicle_id}`; arrival `{f_task.arrival_island}`, return `{f_task.return_wh}`."
            )
        lines.append("")
    lines.extend([
        "## Metrics", "",
        f"- Makespan: {objective.makespan}",
        f"- Reconfiguration count/time: {objective.reconfiguration_count} / "
        f"{sum(r.reconfiguration_end-r.reconfiguration_start for r in schedule.operation_schedules.values())}",
        f"- W loaded/empty travel: {objective.w_loaded_travel} / {objective.w_empty_travel}",
        f"- F travel: {objective.f_travel}",
        f"- Total cost: {objective.total_cost:.6f}",
        "- Feasible: TRUE", "",
        "`TINY_EXACT_VALIDATED = TRUE`", "",
    ])
    return "\n".join(lines)


def main() -> None:
    print("This run proves tiny_01 optimal, replays its actions, exports one schedule object, and audits every resource.")
    instance = load_instance(INSTANCE_PATH)
    exact = solve_tiny_exact(instance, time_limit_seconds=60.0)
    if exact.status != "OPTIMAL":
        raise RuntimeError(f"tiny exact status is not OPTIMAL: {exact.status}")
    replay = RCIASConstructionEnv(instance)
    for action in exact.actions:
        replay.step(action)
    exact_objective = calculate_objective(instance, exact.schedule)
    replay_objective = calculate_objective(instance, replay.schedule)
    if exact_objective.makespan != replay_objective.makespan:
        raise RuntimeError("exact and decoder replay makespans differ")
    audit = check_schedule(instance, replay.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"replayed tiny schedule is infeasible: {audit['violations']}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exporter = ResourceTimelineExporter(instance, replay.schedule)
    paths = exporter.export(OUTPUT)
    solution = {
        "instance": instance.instance_id,
        "solver": exact.backend,
        "solver_status": exact.status,
        "objective": exact.best_value,
        "runtime_seconds": exact.runtime_seconds,
        "optimal": True,
        "gap": 0.0,
        "explored_nodes": exact.explored_nodes,
        "actions": [action.__dict__ for action in exact.actions],
        "exact_makespan": exact_objective.makespan,
        "replay_makespan": replay_objective.makespan,
        "objective_breakdown": replay_objective.to_dict(),
        "schedule": replay.schedule.to_dict(),
    }
    write_json(solution, OUTPUT / "solution.json")
    feasibility = {
        "instance": instance.instance_id,
        "feasible": True,
        "categories": [{"constraint": category, "status": "PASS", "details": []} for category in CATEGORIES],
        "violations": audit["violations"],
        "makespan": audit["makespan"],
        "cost": audit["cost"],
        "exact_replay_equal": exact_objective.makespan == replay_objective.makespan,
    }
    write_json(feasibility, OUTPUT / "feasibility_report.json")
    plot_resource_timeline(paths["resource"], OUTPUT / "gantt.png", OUTPUT / "gantt.pdf")
    (OUTPUT / "validation_summary.md").write_text(
        _summary(instance, replay.schedule, replay_objective, exact), encoding="utf-8"
    )
    print(
        f"TINY_EXACT_VALIDATED = TRUE | solver={exact.backend} status={exact.status} "
        f"makespan={exact.best_value} runtime={exact.runtime_seconds:.4f}s gurobi={gurobi_available()}"
    )


if __name__ == "__main__":
    main()
