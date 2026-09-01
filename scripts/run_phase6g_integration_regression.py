#!/usr/bin/env python3
"""Prove that disabled CSG-NI exactly reproduces frozen baseline ALNS."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def signature(event: dict) -> dict:
    candidate = event["candidate"]
    return {
        "iteration": event["iteration"],
        "decoder_evaluations": event["decoder_evaluations"],
        "current_before": event["current_before"].makespan,
        "best_before": event["best_before"].makespan,
        "destroy_operator": event["destroy_operator"],
        "repair_operator": event["repair_operator"],
        "destroyed_operation_ids": event["destroyed_operation_ids"],
        "candidate_makespan": candidate.makespan,
        "candidate": asdict(candidate.candidate),
        "accepted": event["accepted"],
        "new_global_best": event["new_global_best"],
        "current_after": event["current_after"].makespan,
        "best_after": event["best_after"].makespan,
        "operator_weights_before": event["operator_weights_before"],
        "operator_weights_after": event["operator_weights_after"],
        "temperature_before": event["temperature_before"],
    }


def trajectory(result) -> list[tuple[int, float]]:
    return [
        (point.decoder_evaluations, point.current_best_makespan)
        for point in result.convergence_trace
    ]


def main() -> None:
    freeze_path = ROOT / "outputs/phase6g/environment/phase6g_environment_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    for relative, expected in freeze["frozen_source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed before integration regression: {relative}")
    cases = [
        ("instances/tiny/tiny_01.json", 670001),
        ("instances/tiny/tiny_03.json", 670002),
        ("instances/controlled/RCIAS-CB1/dev/CB1_DEV_S_CF1_R01.json", 670003),
        ("instances/controlled/RCIAS-CB1/dev/CB1_DEV_M_CF2_R01.json", 670004),
        ("instances/controlled/RCIAS-CB1/dev/CB1_DEV_L_CF3_R01.json", 670005),
    ]
    config = ALNSConfig(candidate_trials=8, iteration_limit=25)
    records = []
    for relative, seed in cases:
        instance = load_instance(ROOT / relative)
        baseline_events, wrapped_events = [], []
        baseline = solve_alns(instance, 10**9, seed, config, baseline_events.append)
        wrapped = solve_csgni(
            instance, 10**9, seed, None,
            alns_config=config,
            csgni_config=CSGNIConfig(intervention_rate=0),
            observer=wrapped_events.append,
        )
        checks = {
            "initial_makespan_equal": baseline_events[0]["current_before"].makespan == wrapped_events[0]["current_before"].makespan,
            "event_sequence_equal": [signature(row) for row in baseline_events] == [signature(row) for row in wrapped_events],
            "best_trajectory_equal": trajectory(baseline) == trajectory(wrapped),
            "final_best_makespan_equal": baseline.best.makespan == wrapped.best.makespan,
            "final_best_candidate_equal": baseline.best.candidate == wrapped.best.candidate,
            "decoder_evaluation_count_equal": baseline.decoder_evaluations == wrapped.decoder_evaluations,
            "iteration_count_equal": baseline.iterations == wrapped.iterations,
        }
        records.append({
            "instance_id": instance.instance_id,
            "seed": seed,
            "iterations": baseline.iterations,
            "initial_makespan": baseline_events[0]["current_before"].makespan,
            "final_best_makespan": baseline.best.makespan,
            "decoder_evaluations": baseline.decoder_evaluations,
            "checks": checks,
            "passed": all(checks.values()),
        })
        print(instance.instance_id, "PASS" if all(checks.values()) else "FAIL", flush=True)
    passed = all(record["passed"] for record in records)
    payload = {
        "schema": "phase6g-zero-intervention-regression-v1",
        "environment_freeze_hash": freeze["freeze_hash"],
        "stopping_rule": {"iteration_limit": 25, "candidate_trials": 8},
        "cases": records,
        "NI_DISABLED_EQUALS_ALNS": passed,
        "status": "PASS" if passed else "FAIL",
    }
    output = ROOT / "outputs/phase6g/integration_regression/zero_intervention_regression.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("zero-intervention regression failed")
    print("NI_DISABLED_EQUALS_ALNS = TRUE")


if __name__ == "__main__":
    main()
