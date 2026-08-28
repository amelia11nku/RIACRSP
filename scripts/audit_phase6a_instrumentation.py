#!/usr/bin/env python3
"""Verify that detailed Phase 6A observation preserves the ALNS trajectory."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import Phase6AObserver
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns


def main():
    with (ROOT / "instances/controlled/RCIAS-CB1/manifests/dev_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [next(row for row in rows if row["scale"] == scale) for scale in ("S", "M", "L")]
    cases = []
    for index, row in enumerate(selected):
        instance = load_instance(ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"])
        seed = 619001 + index
        config = ALNSConfig(candidate_trials=8, iteration_limit=40)
        compact = []
        start = time.perf_counter()
        off = solve_alns(instance, 3600.0, seed, config, lambda event: compact.append(_signature(event)))
        off_runtime = time.perf_counter() - start
        observer = Phase6AObserver(instance, {
            "run_id": f"regression_{row['scale']}", "instance_id": instance.instance_id,
            "suite": "DEV", "scale": row["scale"], "CF_level": row["CF_level"], "seed": seed,
        })
        start = time.perf_counter()
        on = solve_alns(instance, 3600.0, seed, config, observer)
        on_runtime = time.perf_counter() - start
        detailed = [(
            item["destroy_operator"], item["repair_operator"], item["accepted"],
            item["new_global_best"], item["candidate_makespan"],
            item["current_makespan_after"], item["best_makespan_after"],
            item["decoder_evaluations"],
        ) for item in observer.transitions]
        cases.append({
            "instance_id": instance.instance_id, "scale": row["scale"], "seed": seed,
            "same_best_makespan": off.best.makespan == on.best.makespan,
            "same_accepted_rejected_sequence": compact == detailed,
            "same_operator_sequence": [item[:2] for item in compact] == [item[:2] for item in detailed],
            "same_convergence_trajectory": [item[6] for item in compact] == [item[6] for item in detailed],
            "same_decoder_evaluation_count": off.decoder_evaluations == on.decoder_evaluations,
            "logging_disabled_runtime": off_runtime, "logging_enabled_runtime": on_runtime,
            "runtime_overhead_percentage": 100.0 * (on_runtime / off_runtime - 1.0),
            "transition_rows": len(observer.transitions), "target_rows": len(observer.targets),
        })
    passed = all(all(case[field] for field in (
        "same_best_makespan", "same_accepted_rejected_sequence", "same_operator_sequence",
        "same_convergence_trajectory", "same_decoder_evaluation_count",
    )) for case in cases)
    output = {
        "schema": "phase6a-instrumentation-regression-v1", "fixed_iteration_budget": 40,
        "cases": cases, "mean_runtime_overhead_percentage": statistics.mean(
            case["runtime_overhead_percentage"] for case in cases
        ), "INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR": not passed,
    }
    path = ROOT / "outputs/phase6a/diagnostics/instrumentation_regression.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("Phase 6A observer changed the ALNS trajectory")
    print("INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR = FALSE")


def _signature(event):
    return (
        event["destroy_operator"], event["repair_operator"], event["accepted"],
        event["new_global_best"], event["candidate"].makespan,
        event["current_after"].makespan, event["best_after"].makespan,
        event["decoder_evaluations"],
    )


if __name__ == "__main__":
    main()
