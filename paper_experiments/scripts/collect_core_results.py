#!/usr/bin/env python3
"""Audit and inventory existing Core results without mutating source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
PROTOCOL_PATH = PAPER_ROOT / "configs/main_core/protocol.json"
SOURCE_MANIFEST_PATH = PAPER_ROOT / "configs/main_core/source_manifest.json"
OUTPUT_ROOT = PAPER_ROOT / "processed_data/core"
MAIN_OUTPUT_ROOT = PAPER_ROOT / "processed_data/main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    fieldnames = list(rows[0]) if rows else [
        "method", "algorithm_version", "experiment_status", "instance_id",
        "scale", "CF_level", "seed", "run_id", "operation_count",
        "analysis_scope", "best_makespan", "best_found_time_seconds",
        "runtime_seconds", "time_limit_seconds", "decoder_evaluations",
        "iterations", "feasible", "git_commit", "config_path",
        "config_sha256", "implementation_manifest_sha256", "timestamp",
        "hardware_identifier", "instance_sha256", "result_sha256", "result_path",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path.relative_to(ROOT)}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: expected={expected} observed={observed}")


def load_instances(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    manifest_path = ROOT / protocol["instance_manifest"]
    verify_file(manifest_path, protocol["instance_manifest_sha256"], "instance manifest")
    checksums_path = ROOT / protocol["instance_checksums"]
    verify_file(checksums_path, protocol["instance_checksums_sha256"], "instance checksums")
    verify_file(
        ROOT / protocol["seed_manifest"],
        protocol["seed_manifest_sha256"],
        "seed manifest",
    )
    checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = digest
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != protocol["instance_count"]:
        raise RuntimeError(f"expected {protocol['instance_count']} Core instances, found {len(rows)}")
    instances: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["instance_id"] in instances:
            raise RuntimeError(f"duplicate Core instance: {row['instance_id']}")
        relative_path = row["relative_path"]
        if relative_path not in checksums:
            raise RuntimeError(f"missing frozen checksum for {relative_path}")
        instance_path = manifest_path.parent.parent / relative_path
        verify_file(instance_path, checksums[relative_path], f"instance {row['instance_id']}")
        row["sha256"] = checksums[relative_path]
        instances[row["instance_id"]] = row
    return instances


def validate_source_files(source: dict[str, Any]) -> None:
    for path_key, hash_key in (
        ("formal_manifest", "formal_manifest_sha256"),
        ("config_manifest", "config_manifest_sha256"),
        ("implementation_manifest", "implementation_manifest_sha256"),
    ):
        if path_key in source:
            verify_file(ROOT / source[path_key], source[hash_key], path_key.replace("_", " "))


def collect_source(
    source: dict[str, Any],
    instances: dict[str, dict[str, str]],
    primary_seeds: set[int],
    supplementary_seeds: set[int],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    validate_source_files(source)
    method = source["method"]
    root = ROOT / source["root"]
    allowed_seeds = primary_seeds | supplementary_seeds
    records: list[dict[str, object]] = []
    errors: list[str] = []
    excluded_seed_records = 0
    seen: set[tuple[str, int]] = set()
    paths = sorted(root.rglob("seed_*.json")) if root.is_dir() else []
    for path in paths:
        try:
            payload = load_json(path)
            instance_id = str(payload["instance_id"])
            seed = int(payload["seed"])
            if payload.get("schema") != source["schema"]:
                raise ValueError(f"unexpected schema {payload.get('schema')}")
            label = payload.get("algorithm", payload.get("method"))
            if label != source["result_label"]:
                raise ValueError(f"unexpected method label {label}")
            if instance_id not in instances:
                raise ValueError(f"instance outside frozen Core: {instance_id}")
            if seed not in allowed_seeds:
                excluded_seed_records += 1
                continue
            key = (instance_id, seed)
            if key in seen:
                raise ValueError(f"duplicate key {key}")
            seen.add(key)
            if path.parent.name != instance_id or path.stem != f"seed_{seed}":
                raise ValueError("path does not match result key")
            row = instances[instance_id]
            expected_limit = 2.0 * int(row["number_of_operations"])
            observed_limit = float(payload["time_limit_seconds"])
            if not math.isclose(observed_limit, expected_limit, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"time limit {observed_limit} != {expected_limit}")
            if not payload.get("feasible", False):
                raise ValueError("result is infeasible")
            independent = payload.get("independent_feasibility_audit")
            if independent is not None and (
                not independent.get("feasible", False) or independent.get("violations")
            ):
                raise ValueError("independent feasibility audit failed")
            makespan = float(payload["best_makespan"])
            runtime = float(payload["runtime"])
            best_found_time = float(payload["best_found_time"])
            decoder_evaluations = int(payload["decoder_evaluations"])
            iterations = int(payload["iterations"])
            if not math.isfinite(makespan) or makespan <= 0:
                raise ValueError("invalid makespan")
            if not math.isfinite(runtime) or runtime < 0:
                raise ValueError("invalid runtime")
            if (
                not math.isfinite(best_found_time)
                or best_found_time < 0
                or best_found_time > runtime + 1e-6
            ):
                raise ValueError("invalid time-to-best")
            if decoder_evaluations < 1 or iterations < 0:
                raise ValueError("invalid evaluation/iteration count")
            if "instance_sha256" in payload and payload["instance_sha256"] != row["sha256"]:
                raise ValueError("instance hash mismatch")
            if "formal_manifest_sha256" in source and (
                payload.get("formal_manifest_sha256") != source["formal_manifest_sha256"]
            ):
                raise ValueError("formal manifest hash mismatch")
            if "config_manifest_sha256" in source and (
                payload.get("config_sha256") != source["config_manifest_sha256"]
            ):
                raise ValueError("config manifest hash mismatch")
            if "implementation_manifest_sha256" in source and (
                payload.get("implementation_manifest_sha256")
                != source["implementation_manifest_sha256"]
            ):
                raise ValueError("implementation manifest hash mismatch")
            records.append({
                "method": method,
                "algorithm_version": payload.get("algorithm_version", label),
                "experiment_status": payload.get(
                    "experiment_status", "FROZEN_BASELINE"
                ),
                "instance_id": instance_id,
                "scale": row["scale"],
                "CF_level": row["CF_level"],
                "seed": seed,
                "run_id": f"{instance_id}__{seed}",
                "operation_count": int(row["number_of_operations"]),
                "analysis_scope": "PRIMARY" if seed in primary_seeds else "SUPPLEMENTARY",
                "best_makespan": makespan,
                "best_found_time_seconds": best_found_time,
                "runtime_seconds": runtime,
                "time_limit_seconds": observed_limit,
                "decoder_evaluations": decoder_evaluations,
                "iterations": iterations,
                "feasible": True,
                "git_commit": payload.get("git_commit", ""),
                "config_path": payload.get("config_path", ""),
                "config_sha256": payload.get("config_sha256", ""),
                "implementation_manifest_sha256": payload.get(
                    "implementation_manifest_sha256", ""
                ),
                "timestamp": payload.get("started_at_utc", ""),
                "hardware_identifier": json.dumps(
                    payload.get("environment", payload.get("compute", {})),
                    sort_keys=True,
                ),
                "instance_sha256": payload.get("instance_sha256", row["sha256"]),
                "result_sha256": sha256(path),
                "result_path": str(path.relative_to(ROOT)),
            })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")

    primary_expected = {(instance_id, seed) for instance_id in instances for seed in primary_seeds}
    supplementary_expected = {
        (instance_id, seed) for instance_id in instances for seed in supplementary_seeds
    }
    observed = {(str(row["instance_id"]), int(row["seed"])) for row in records}
    primary_missing = sorted(primary_expected - observed)
    supplementary_missing = sorted(supplementary_expected - observed)
    summary = {
        "method": method,
        "source_root": source["root"],
        "source_root_exists": root.is_dir(),
        "observed_json_files": len(paths),
        "excluded_seed_records": excluded_seed_records,
        "valid_unique_records": len(records),
        "invalid_records": len(errors),
        "primary_expected_records": len(primary_expected),
        "primary_observed_records": len(primary_expected & observed),
        "primary_missing_records": len(primary_missing),
        "primary_matrix_complete": not errors and not primary_missing,
        "supplementary_expected_records": len(supplementary_expected),
        "supplementary_observed_records": len(supplementary_expected & observed),
        "supplementary_missing_records": len(supplementary_missing),
        "all_seed_expected_records": source["expected_all_seed_records"],
        "all_seed_matrix_complete": not errors and len(records) == source["expected_all_seed_records"],
        "first_primary_missing": [list(key) for key in primary_missing[:10]],
        "first_supplementary_missing": [list(key) for key in supplementary_missing[:10]],
        "first_errors": errors[:10],
        "blocked_by": source.get("blocked_by"),
        "reporting_note": source.get("reporting_note"),
    }
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return nonzero unless all five primary matrices are complete",
    )
    args = parser.parse_args()
    protocol = load_json(PROTOCOL_PATH)
    sources = load_json(SOURCE_MANIFEST_PATH)["sources"]
    instances = load_instances(protocol)
    primary_seeds = set(map(int, protocol["primary_seeds"]))
    supplementary_seeds = set(map(int, protocol["supplementary_seeds"]))
    if primary_seeds & supplementary_seeds:
        raise RuntimeError("primary and supplementary seeds overlap")

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, object]] = []
    for source in sources:
        summary, source_records = collect_source(
            source, instances, primary_seeds, supplementary_seeds
        )
        summaries.append(summary)
        records.extend(source_records)
    records.sort(key=lambda row: (str(row["method"]), str(row["instance_id"]), int(row["seed"])))
    complete = all(summary["all_seed_matrix_complete"] for summary in summaries)
    observed_primary: dict[tuple[str, str], set[int]] = {}
    for record in records:
        if record["analysis_scope"] != "PRIMARY":
            continue
        key = (str(record["method"]), str(record["instance_id"]))
        observed_primary.setdefault(key, set()).add(int(record["seed"]))
    completion_rows = []
    for source in sources:
        method = source["method"]
        for instance_id in sorted(instances):
            observed_seeds = observed_primary.get((method, instance_id), set())
            missing_seeds = primary_seeds - observed_seeds
            completion_rows.append({
                "method": method,
                "instance_id": instance_id,
                "expected_seed_count": len(primary_seeds),
                "observed_seed_count": len(observed_seeds),
                "observed_seeds": json.dumps(sorted(observed_seeds)),
                "missing_seeds": json.dumps(sorted(missing_seeds)),
                "complete": not missing_seeds,
            })
    seed_manifest = {
        "schema": "initial-manuscript-main-seed-manifest-v1",
        "status": "FROZEN_FIVE_MATCHED_SEEDS",
        "primary_seeds": sorted(primary_seeds),
        "supplementary_seeds": sorted(supplementary_seeds),
        "excluded_available_seeds": protocol.get("excluded_available_seeds", []),
        "selection_rule": protocol["seed_selection_rule"],
        "source_manifest": protocol["seed_manifest"],
        "source_manifest_sha256": protocol["seed_manifest_sha256"],
    }
    audit = {
        "schema": "initial-manuscript-core-source-inventory-v1",
        "status": "PASS_COMPLETE" if complete else "INCOMPLETE_BLOCK_BKS_RPD",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST_PATH),
        "instance_count": len(instances),
        "primary_seeds": sorted(primary_seeds),
        "supplementary_seeds": sorted(supplementary_seeds),
        "methods": summaries,
        "observed_valid_records": len(records),
        "bks_rpd_authorized": complete,
        "derived_outputs": [
            "paper_experiments/processed_data/core/observed_runs.csv",
            "paper_experiments/processed_data/main/main_completion_matrix.csv",
            "paper_experiments/processed_data/main/main_seed_manifest.json",
        ],
    }
    atomic_csv(OUTPUT_ROOT / "observed_runs.csv", records)
    atomic_csv(MAIN_OUTPUT_ROOT / "main_completion_matrix.csv", completion_rows)
    atomic_json(MAIN_OUTPUT_ROOT / "main_seed_manifest.json", seed_manifest)
    atomic_json(OUTPUT_ROOT / "source_inventory.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 1 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
