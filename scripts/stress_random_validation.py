#!/usr/bin/env python3
"""Generate and decode many seeded tiny instances to stress deterministic feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_automotive_semantic import build_tiny_instance as build_automotive_tiny
from generate_fjsp_reconfigurable import build_tiny_instance as build_fjsp_tiny
from rcias_clgri.data.loader import load_instance_dict
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.heuristic.dispatching import solve_dispatching


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test random RCIAS tiny instances")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/reports/stress_random_validation.json"),
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    print(f"This run checks that {args.count} seeded generated instances decode with 100% feasibility.")
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    makespans: list[float] = []
    for index in range(args.count):
        seed = args.seed + index
        raw = build_fjsp_tiny(seed) if index % 2 == 0 else build_automotive_tiny(seed)
        instance = load_instance_dict(raw)
        result = solve_dispatching(instance, "H1")
        audit = check_schedule(instance, result.schedule)
        makespans.append(result.objective.makespan)
        if not audit["feasible"]:
            failures.append({"seed": seed, "instance": instance.instance_id, "violations": audit["violations"]})
            break
    runtime = time.perf_counter() - started
    report = {
        "requested": args.count,
        "completed": len(makespans),
        "feasible": len(makespans) - len(failures),
        "feasible_rate": (len(makespans) - len(failures)) / len(makespans),
        "mean_makespan": sum(makespans) / len(makespans),
        "runtime_seconds": runtime,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise RuntimeError("stress validation found an infeasible schedule")


if __name__ == "__main__":
    main()
