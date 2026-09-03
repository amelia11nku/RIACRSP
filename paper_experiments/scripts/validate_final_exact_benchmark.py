#!/usr/bin/env python3
"""Audit the final 10-instance exact benchmark against imported Gurobi evidence."""

from __future__ import annotations

import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.env.objective import calculate_objective  # noqa: E402
from rcias_clgri.env.schedule import FTask, OperationSchedule, Schedule, WTask  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402


BENCHMARK_ROOT = (
    ROOT / "paper_experiments/benchmarks/exact_validation_10_final_gurobi"
)
INSTANCE_SUITE_ROOT = BENCHMARK_ROOT / "exact_validation_10_final"
MANIFEST_PATH = INSTANCE_SUITE_ROOT / "candidate_manifest.json"
CHECKSUM_PATH = INSTANCE_SUITE_ROOT / "checksums.sha256"
GUROBI_CSV_PATH = BENCHMARK_ROOT / "gurobi_results.csv"
MAPPING_PATH = BENCHMARK_ROOT / "id_mapping.csv"
AUDIT_PATH = (
    ROOT / "paper_experiments/processed_data/exact_validation"
    / "exact_benchmark_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def near(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def dataclass_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    return {item.name: raw[item.name] for item in fields(cls)}


def schedule_from_dict(raw: dict[str, Any]) -> Schedule:
    operations = {}
    for operation_id, record in raw["operations"].items():
        values = dataclass_kwargs(OperationSchedule, record)
        values["binding_resource"] = tuple(values["binding_resource"])
        operations[operation_id] = OperationSchedule(**values)
    return Schedule(
        instance_id=raw["instance_id"],
        operation_schedules=operations,
        product_sequences=raw["product_sequences"],
        product_predecessor=raw["product_predecessor"],
        product_successor=raw["product_successor"],
        island_timelines=raw["island_timelines"],
        w_timelines={
            vehicle: [WTask(**dataclass_kwargs(WTask, task)) for task in tasks]
            for vehicle, tasks in raw["w_timelines"].items()
        },
        f_timelines={
            vehicle: [FTask(**dataclass_kwargs(FTask, task)) for task in tasks]
            for vehicle, tasks in raw["f_timelines"].items()
        },
        accumulated_reconfiguration_cost=float(
            raw["accumulated_reconfiguration_cost"]
        ),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path.lstrip("* ")] = digest
    return checksums


def add_failure(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    manifest = load_json(MANIFEST_PATH)
    gurobi_rows = read_csv(GUROBI_CSV_PATH)
    mapping_rows = read_csv(MAPPING_PATH)
    checksum_rows = read_checksums(CHECKSUM_PATH)
    gurobi_by_id = {row["instance_id"]: row for row in gurobi_rows}
    mapping_by_id = {row["new_instance_id"]: row for row in mapping_rows}

    expected_ids = [f"RIACRSP_E{index:02d}" for index in range(1, 11)]
    manifest_ids = [row["instance_id"] for row in manifest.get("instances", [])]
    failures: list[str] = []
    diagnostics: list[str] = []
    add_failure(failures, manifest.get("instance_count") == 10, "manifest count != 10")
    add_failure(failures, manifest_ids == expected_ids, "manifest IDs/order mismatch")
    add_failure(failures, sorted(gurobi_by_id) == expected_ids, "Gurobi IDs mismatch")
    add_failure(failures, sorted(mapping_by_id) == expected_ids, "mapping IDs mismatch")
    add_failure(
        failures,
        manifest.get("status") == "FINAL_ALL_GUROBI_OPTIMAL_REPLAY_AUDITED",
        "manifest is not final",
    )

    instance_audits = []
    for entry in manifest.get("instances", []):
        instance_id = entry["instance_id"]
        instance_path = BENCHMARK_ROOT / entry["relative_path"]
        result_path = BENCHMARK_ROOT / "runs" / instance_id / "result.json"
        solution_path = BENCHMARK_ROOT / "runs" / instance_id / "solution.json"
        feasibility_path = BENCHMARK_ROOT / "runs" / instance_id / "feasibility.json"
        log_path = BENCHMARK_ROOT / "logs" / f"{instance_id}.log"
        local_failures: list[str] = []

        required = [instance_path, result_path, solution_path, feasibility_path, log_path]
        add_failure(
            local_failures,
            all(path.is_file() for path in required),
            "one or more required files are missing",
        )
        if local_failures:
            failures.extend(f"{instance_id}: {message}" for message in local_failures)
            instance_audits.append({"instance_id": instance_id, "failures": local_failures})
            continue

        instance_digest = sha256(instance_path)
        instance = load_instance(instance_path)
        result = load_json(result_path)
        solution = load_json(solution_path)
        feasibility = load_json(feasibility_path)
        csv_row = gurobi_by_id[instance_id]
        mapping = mapping_by_id[instance_id]
        schedule = schedule_from_dict(solution["schedule"])
        replay_audit = check_schedule(instance, schedule)
        replay_objective = calculate_objective(instance, schedule)
        action_schedule = schedule_from_dict(solution["action_replay_schedule"])
        action_audit = check_schedule(instance, action_schedule)
        action_objective = calculate_objective(instance, action_schedule)
        h1 = solve_dispatching(instance, "H1")
        h1_audit = check_schedule(instance, h1.schedule)

        checksum_key = f"instances/{instance_path.name}"
        add_failure(local_failures, instance.instance_id == instance_id, "loaded ID mismatch")
        add_failure(local_failures, instance.num_operations <= 12, "operation cap exceeded")
        add_failure(
            local_failures,
            instance.num_operations == int(entry["operation_count"]),
            "operation count mismatch",
        )
        add_failure(local_failures, instance_digest == entry["sha256"], "manifest hash mismatch")
        add_failure(
            local_failures,
            checksum_rows.get(checksum_key) == instance_digest,
            "checksum file mismatch",
        )
        add_failure(local_failures, result["instance_sha256"] == instance_digest, "result hash mismatch")
        add_failure(local_failures, csv_row["instance_sha256"] == instance_digest, "CSV hash mismatch")
        add_failure(local_failures, result["status"] == "OPTIMAL", "result status is not OPTIMAL")
        add_failure(local_failures, csv_row["status"] == "OPTIMAL", "CSV status is not OPTIMAL")
        add_failure(local_failures, as_bool(result["optimality_proven"]), "result optimality not proven")
        add_failure(local_failures, as_bool(csv_row["optimality_proven"]), "CSV optimality not proven")
        add_failure(local_failures, near(result["reported_gap"], 0.0), "nonzero result gap")
        add_failure(local_failures, near(csv_row["reported_gap"], 0.0), "nonzero CSV gap")
        add_failure(
            local_failures,
            near(result["objective_makespan"], result["best_bound"]),
            "objective/bound mismatch",
        )
        add_failure(local_failures, replay_audit["feasible"], "native replay infeasible")
        add_failure(local_failures, action_audit["feasible"], "action replay infeasible")
        add_failure(local_failures, as_bool(feasibility["feasible"]), "feasibility file reports false")
        add_failure(
            local_failures,
            near(replay_objective.makespan, result["objective_makespan"]),
            "native replay objective mismatch",
        )
        add_failure(
            local_failures,
            near(feasibility["makespan"], replay_objective.makespan),
            "feasibility objective mismatch",
        )
        add_failure(local_failures, h1_audit["feasible"], "local H1 infeasible")
        add_failure(
            local_failures,
            near(h1.objective.makespan, result["h1_makespan"]),
            "local/imported H1 mismatch",
        )
        add_failure(
            local_failures,
            mapping["source_instance_id"] == entry["source_instance_id"],
            "source mapping mismatch",
        )
        add_failure(
            local_failures,
            near(mapping["objective_makespan"], result["objective_makespan"]),
            "mapping objective mismatch",
        )
        add_failure(
            local_failures,
            "Optimal solution found" in log_path.read_text(encoding="utf-8", errors="replace"),
            "Gurobi log lacks optimal completion marker",
        )

        action_matches = near(action_objective.makespan, result["objective_makespan"])
        if not action_matches:
            diagnostics.append(
                f"{instance_id}: feasible action replay makespan "
                f"{action_objective.makespan:g} differs from solver/native replay "
                f"{float(result['objective_makespan']):g}"
            )
        failures.extend(f"{instance_id}: {message}" for message in local_failures)
        instance_audits.append({
            "instance_id": instance_id,
            "instance_path": str(instance_path.relative_to(ROOT)),
            "instance_sha256": instance_digest,
            "result_sha256": sha256(result_path),
            "solution_sha256": sha256(solution_path),
            "feasibility_sha256": sha256(feasibility_path),
            "gurobi_log_sha256": sha256(log_path),
            "source_instance_id": entry["source_instance_id"],
            "source_suite": entry["source_suite"],
            "operation_count": instance.num_operations,
            "objective_makespan": float(result["objective_makespan"]),
            "best_bound": float(result["best_bound"]),
            "reported_gap": float(result["reported_gap"]),
            "native_replay_feasible": bool(replay_audit["feasible"]),
            "native_replay_makespan": replay_objective.makespan,
            "action_replay_feasible": bool(action_audit["feasible"]),
            "action_replay_makespan": action_objective.makespan,
            "action_replay_matches_solver": action_matches,
            "local_h1_makespan": h1.objective.makespan,
            "gurobi_runtime_seconds": float(result["solver_runtime_seconds"]),
            "failures": local_failures,
        })

    payload = {
        "schema": "riacrsp-final-exact-benchmark-audit-v1",
        "status": (
            "FAIL" if failures else
            "PASS_WITH_ACTION_REPLAY_DIAGNOSTIC" if diagnostics else "PASS"
        ),
        "benchmark_root": str(BENCHMARK_ROOT.relative_to(ROOT)),
        "instance_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "instance_manifest_sha256": sha256(MANIFEST_PATH),
        "checksums_sha256": sha256(CHECKSUM_PATH),
        "gurobi_results_sha256": sha256(GUROBI_CSV_PATH),
        "id_mapping_sha256": sha256(MAPPING_PATH),
        "expected_instance_count": 10,
        "audited_instance_count": len(instance_audits),
        "all_primary_gates_passed": not failures,
        "diagnostics": diagnostics,
        "failures": failures,
        "instances": instance_audits,
    }
    atomic_json(payload, AUDIT_PATH)
    print(json.dumps({
        "status": payload["status"],
        "audited_instance_count": payload["audited_instance_count"],
        "failure_count": len(failures),
        "diagnostic_count": len(diagnostics),
        "audit_path": str(AUDIT_PATH.relative_to(ROOT)),
    }, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
