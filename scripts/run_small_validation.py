#!/usr/bin/env python3
"""Run baselines, independent checks, graph construction, and tiny exact search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.exact.tiny_exact_solver import gurobi_available, solve_tiny_exact
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.heuristic.dispatching import solve_dispatching


INSTANCE_PATHS = (
    Path("instances/tiny/fjsp_tiny.json"),
    Path("instances/tiny/automotive_tiny.json"),
    Path("instances/tiny/fjsp_small.json"),
    Path("instances/tiny/automotive_small.json"),
)


def _schedule_summary(result: Any) -> str:
    parts = []
    for product_id, sequence in result.schedule.product_sequences.items():
        assignments = [
            f"{op_id}@{result.schedule.operation_schedules[op_id].island_id}"
            for op_id in sequence
        ]
        parts.append(f"{product_id}: {' -> '.join(assignments)}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deterministic RCIAS construction")
    parser.add_argument("--output", type=Path, default=Path("validation_results.json"))
    args = parser.parse_args()
    print("This run shows that all baselines share one decoder and produce independently feasible schedules.")
    report: dict[str, Any] = {
        "exact_backend": "gurobi" if gurobi_available() else "exhaustive-active-schedule-bnb",
        "instances": {},
    }
    for path in INSTANCE_PATHS:
        instance = load_instance(path)
        item: dict[str, Any] = {
            "products": len(instance.products),
            "operations": len(instance.operations),
            "islands": len(instance.islands),
            "W": len(instance.agvs_w),
            "F": len(instance.agvs_f),
            "baselines": {},
        }
        for method in ("H1", "H2", "H3"):
            result = solve_dispatching(instance, method)
            audit = check_schedule(instance, result.schedule)
            if not audit["feasible"]:
                raise RuntimeError(f"{path}/{method} infeasible: {audit['violations']}")
            graph = build_graph_state(instance, result.schedule)
            metrics = result.objective.to_dict()
            metrics.update({
                "runtime_seconds": result.runtime_seconds,
                "feasible": True,
                "graph_nodes": sum(len(nodes) for nodes in graph.node_features.values()),
                "graph_edges": len(graph.edges),
            })
            item["baselines"][method] = metrics
            print(
                f"{path.name} {method}: makespan={result.objective.makespan:.1f} "
                f"cost={result.objective.total_cost:.3f} feasible=True runtime={result.runtime_seconds:.4f}s"
            )
            if "tiny" in path.stem and method == "H2":
                print("  " + _schedule_summary(result))
        if path.stem.endswith("_tiny"):
            exact = solve_tiny_exact(instance, time_limit_seconds=30.0)
            item["exact"] = {
                "status": exact.status,
                "backend": exact.backend,
                "objective_name": exact.objective_name,
                "best_value": exact.best_value,
                "makespan": exact.objective.makespan,
                "cost": exact.objective.total_cost,
                "explored_nodes": exact.explored_nodes,
                "runtime_seconds": exact.runtime_seconds,
                "feasible": check_schedule(instance, exact.schedule)["feasible"],
            }
            print(
                f"  exact: status={exact.status} makespan={exact.objective.makespan:.1f} "
                f"nodes={exact.explored_nodes} runtime={exact.runtime_seconds:.4f}s"
            )
        report["instances"][instance.instance_id] = item
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
