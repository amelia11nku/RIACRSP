#!/usr/bin/env python3
"""Validate and summarize final exact-validation results without touching raw data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "paper_experiments/configs/exact_validation/final_protocol.json"
OUTPUT_ROOT = ROOT / "paper_experiments/processed_data/exact_validation"
METHOD_DIRS = {
    "GA": "GA",
    "Adapted DCGA": "ADAPTED_DCGA",
    "DABC-RIACRSP": "DABC_RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA_RIACRSP_V2_N4M",
    "CSG-NI Phase6H provisional": "CSG_NI_PROVISIONAL_PHASE6H",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    )


def validate_protocol(protocol: dict[str, Any]) -> tuple[str, str]:
    if protocol.get("status") != "FROZEN_BEFORE_FINAL_EXACT_HEURISTIC_EVALUATION":
        raise RuntimeError("final exact protocol is not frozen")
    if protocol.get("primary_methods") != list(METHOD_DIRS):
        raise RuntimeError("exact method list does not match collector")
    for record in protocol["frozen_files"]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen exact artifact mismatch: {record['path']}")
    implementation = ROOT / protocol["implementation_manifest"]
    if sha256(implementation) != protocol["implementation_manifest_sha256"]:
        raise RuntimeError("exact implementation manifest hash mismatch")
    audit = load_json(ROOT / protocol["benchmark_audit"])
    if not audit.get("all_primary_gates_passed"):
        raise RuntimeError("exact benchmark primary audit gate failed")
    return sha256(PROTOCOL_PATH), sha256(implementation)


def load_references(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    with (ROOT / protocol["gurobi_results"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    references = {row["instance_id"]: row for row in rows}
    if len(references) != 10 or len(rows) != 10:
        raise RuntimeError("exact Gurobi reference must contain 10 unique instances")
    for instance_id, row in references.items():
        checks = [
            row["status"] == "OPTIMAL",
            row["optimality_proven"].lower() == "true",
            close(row["reported_gap"], 0.0),
            close(row["objective_makespan"], row["best_bound"]),
            row["replay_feasible"].lower() == "true",
            row["native_replay_equal"].lower() == "true",
        ]
        if not all(checks):
            raise RuntimeError(f"invalid exact reference: {instance_id}")
    return references


def validate_result(
    path: Path,
    payload: dict[str, Any],
    method: str,
    instance_id: str,
    seed: int,
    reference: dict[str, str],
    protocol: dict[str, Any],
    protocol_hash: str,
    implementation_hash: str,
) -> dict[str, object]:
    optimum = float(reference["objective_makespan"])
    operations = int(reference["operation_count"])
    expected_limit = float(protocol["wall_clock_seconds_per_operation"]) * operations
    expected_path = (
        ROOT / protocol["output_root"] / "runs" / METHOD_DIRS[method]
        / instance_id / f"seed_{seed}.json"
    )
    if path != expected_path:
        raise ValueError("result path does not match method/instance/seed")
    checks = [
        payload.get("schema") == "initial-manuscript-final-exact-run-v1",
        payload.get("status") == "COMPLETE",
        payload.get("method") == method,
        payload.get("instance_id") == instance_id,
        int(payload.get("seed")) == seed,
        payload.get("protocol_sha256") == protocol_hash,
        payload.get("implementation_manifest_sha256") == implementation_hash,
        payload.get("instance_sha256") == reference["instance_sha256"],
        close(payload.get("proven_optimum"), optimum),
        close(payload.get("time_limit_seconds"), expected_limit),
        payload.get("feasible") is True,
        payload.get("independent_feasibility_audit", {}).get("feasible") is True,
        not payload.get("independent_feasibility_audit", {}).get("violations"),
    ]
    if not all(checks):
        raise ValueError("one or more identity/protocol/feasibility checks failed")
    objective = float(payload["best_makespan"])
    runtime = float(payload["runtime_seconds"])
    best_time = float(payload["best_found_time_seconds"])
    gap = 100.0 * (objective - optimum) / optimum
    reached = bool(payload["optimum_reached"])
    first_optimum = payload.get("first_optimum_time_seconds")
    if not all(math.isfinite(value) for value in (objective, runtime, best_time, gap)):
        raise ValueError("non-finite objective/runtime/gap")
    if objective <= 0 or runtime < 0 or best_time < 0 or best_time > runtime + 1e-6:
        raise ValueError("invalid objective or timing")
    if not close(payload["gap_to_proven_optimum_percent"], gap):
        raise ValueError("stored exact gap mismatch")
    if reached != (objective <= optimum + 1e-6):
        raise ValueError("optimum-hit flag mismatch")
    if reached:
        if first_optimum is None or not close(first_optimum, best_time):
            raise ValueError("successful run lacks actual first-optimum time")
        if payload.get("right_censored_without_optimum") is not False:
            raise ValueError("successful run incorrectly right-censored")
    else:
        if first_optimum is not None:
            raise ValueError("unsuccessful run has a fabricated optimum time")
        if payload.get("right_censored_without_optimum") is not True:
            raise ValueError("unsuccessful run lacks right-censor flag")
    return {
        "algorithm": method,
        "algorithm_version": payload.get("algorithm_version", method),
        "experiment_status": payload.get("experiment_status", "FROZEN_BASELINE"),
        "instance_id": instance_id,
        "source_instance_id": payload["source_instance_id"],
        "source_suite": payload["source_suite"],
        "operation_count": operations,
        "seed": seed,
        "run_id": f"{instance_id}__{seed}",
        "exact_optimum": optimum,
        "final_objective": objective,
        "exact_gap_percent": gap,
        "optimum_reached": reached,
        "first_optimum_time_seconds": first_optimum,
        "right_censored_without_optimum": not reached,
        "best_found_time_seconds": best_time,
        "runtime_seconds": runtime,
        "time_limit_seconds": expected_limit,
        "decoder_evaluations": int(payload["decoder_evaluations"]),
        "iterations": int(payload["iterations"]),
        "feasible": True,
        "git_commit": payload["started_from_commit"],
        "protocol_sha256": protocol_hash,
        "implementation_manifest_sha256": implementation_hash,
        "result_sha256": sha256(path),
        "result_path": str(path.relative_to(ROOT)),
    }


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in records:
        grouped.setdefault(
            (str(row["algorithm"]), str(row["instance_id"])), []
        ).append(row)
    summaries = []
    for (method, instance_id), rows in sorted(grouped.items()):
        objectives = [float(row["final_objective"]) for row in rows]
        gaps = [float(row["exact_gap_percent"]) for row in rows]
        hit_times = [
            float(row["first_optimum_time_seconds"])
            for row in rows if row["first_optimum_time_seconds"] != ""
            and row["first_optimum_time_seconds"] is not None
        ]
        hits = sum(bool(row["optimum_reached"]) for row in rows)
        summaries.append({
            "algorithm": method,
            "instance_id": instance_id,
            "operation_count": rows[0]["operation_count"],
            "exact_optimum": rows[0]["exact_optimum"],
            "observed_runs": len(rows),
            "expected_runs": 5,
            "best_objective": min(objectives),
            "mean_objective": statistics.fmean(objectives),
            "median_objective": statistics.median(objectives),
            "std_objective": statistics.stdev(objectives) if len(objectives) > 1 else 0.0,
            "best_gap_percent": min(gaps),
            "mean_gap_percent": statistics.fmean(gaps),
            "median_gap_percent": statistics.median(gaps),
            "optimum_hit_count": hits,
            "optimum_hit_rate": hits / len(rows),
            "best_time_to_optimum_seconds": min(hit_times) if hit_times else "",
            "mean_time_to_optimum_seconds": statistics.fmean(hit_times) if hit_times else "",
            "median_time_to_optimum_seconds": statistics.median(hit_times) if hit_times else "",
            "feasibility_rate": statistics.fmean(bool(row["feasible"]) for row in rows),
        })
    return summaries


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    protocol_hash, implementation_hash = validate_protocol(protocol)
    references = load_references(protocol)
    seeds = [int(seed) for seed in protocol["seeds"]]
    records: list[dict[str, object]] = []
    completion: list[dict[str, object]] = []
    errors: list[str] = []
    raw_root = ROOT / protocol["output_root"] / "runs"
    for method, directory in METHOD_DIRS.items():
        for instance_id in sorted(references):
            observed = []
            for seed in seeds:
                path = raw_root / directory / instance_id / f"seed_{seed}.json"
                if not path.is_file():
                    continue
                observed.append(seed)
                try:
                    records.append(validate_result(
                        path, load_json(path), method, instance_id, seed,
                        references[instance_id], protocol, protocol_hash,
                        implementation_hash,
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"{path.relative_to(ROOT)}: {error}")
            missing = [seed for seed in seeds if seed not in observed]
            completion.append({
                "algorithm": method,
                "instance_id": instance_id,
                "expected_runs": len(seeds),
                "observed_files": len(observed),
                "missing_seeds": ";".join(map(str, missing)),
                "complete": not missing,
            })

    keys = [(row["algorithm"], row["instance_id"], row["seed"]) for row in records]
    if len(keys) != len(set(keys)):
        errors.append("duplicate algorithm/instance/seed keys detected")
    records.sort(key=lambda row: (
        str(row["algorithm"]), str(row["instance_id"]), int(row["seed"])
    ))
    summaries = summarize(records)
    expected_total = int(protocol["expected_total_runs"])
    complete = len(records) == expected_total and not errors and all(
        bool(row["complete"]) for row in completion
    )

    reference_rows = [{
        "instance_id": instance_id,
        "source_instance_id": row["source_instance_id"],
        "operation_count": int(row["operation_count"]),
        "gurobi_optimum": float(row["objective_makespan"]),
        "gurobi_best_bound": float(row["best_bound"]),
        "gurobi_gap": float(row["reported_gap"]),
        "gurobi_runtime_seconds": float(row["solver_runtime_seconds"]),
        "gurobi_node_count": float(row["node_count"]),
        "native_replay_feasible": row["replay_feasible"].lower() == "true",
        "native_replay_makespan": float(row["replay_makespan"]),
        "action_replay_feasible": row["action_replay_feasible"].lower() == "true",
        "action_replay_makespan": float(row["action_replay_makespan"]),
        "action_replay_matches_solver": row["action_replay_matches_solver"].lower() == "true",
        "instance_sha256": row["instance_sha256"],
    } for instance_id, row in sorted(references.items())]

    run_fields = list(records[0]) if records else []
    summary_fields = list(summaries[0]) if summaries else []
    atomic_csv(OUTPUT_ROOT / "exact_validation_runs.csv", records, run_fields)
    atomic_csv(OUTPUT_ROOT / "exact_validation_summary.csv", summaries, summary_fields)
    atomic_csv(
        OUTPUT_ROOT / "exact_completion_matrix.csv", completion,
        list(completion[0]),
    )
    atomic_csv(
        OUTPUT_ROOT / "exact_reference.csv", reference_rows,
        list(reference_rows[0]),
    )
    audit = {
        "schema": "initial-manuscript-final-exact-result-inventory-v1",
        "status": "PASS_COMPLETE" if complete else "PASS_PARTIAL_WAITING_FOR_CSGNI" if not errors else "FAIL",
        "protocol_sha256": protocol_hash,
        "implementation_manifest_sha256": implementation_hash,
        "expected_methods": list(METHOD_DIRS),
        "expected_instances": sorted(references),
        "expected_seeds": seeds,
        "expected_total_runs": expected_total,
        "valid_unique_runs": len(records),
        "invalid_runs": len(errors),
        "first_errors": errors[:20],
        "completed_by_method": {
            method: sum(row["algorithm"] == method for row in records)
            for method in METHOD_DIRS
        },
        "all_runs_feasible": all(bool(row["feasible"]) for row in records),
        "all_gurobi_references_optimal": True,
        "final_exact_table_authorized": complete,
    }
    atomic_json(OUTPUT_ROOT / "exact_result_inventory.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
