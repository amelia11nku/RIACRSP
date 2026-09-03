#!/usr/bin/env python3
"""Assemble the audited Phase6H CSG-NI versus ALNS efficiency evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "outputs/phase6h_validation"
OUTPUT_ROOT = ROOT / "paper_experiments/processed_data/efficiency"
INTEGRITY_PATH = SOURCE_ROOT / "audit/analysis_integrity.json"
ARTIFACT_MANIFEST_PATH = SOURCE_ROOT / "audit/artifact_manifest.json"
METHODS = {"ALNS", "PHASE6H_CSGNI"}
SOURCES = {
    "efficiency_runs.csv": SOURCE_ROOT / "validation_run_summary.csv",
    "efficiency_instance_summary.csv": SOURCE_ROOT / "statistics/instance_method_summary.csv",
    "efficiency_method_summary.csv": SOURCE_ROOT / "statistics/method_summary.csv",
    "efficiency_pairwise.csv": SOURCE_ROOT / "statistics/pairwise_statistics.csv",
    "anytime_curves.csv": SOURCE_ROOT / "anytime/normalized_budget_checkpoints.csv",
    "anytime_runs.csv": SOURCE_ROOT / "anytime/run_anytime_summary.csv",
    "anytime_summary.csv": SOURCE_ROOT / "anytime/method_anytime_summary.csv",
    "target_hit_summary.csv": SOURCE_ROOT / "anytime/target_hit_summary.csv",
    "csgni_runtime_efficiency_by_run.csv": SOURCE_ROOT / "statistics/csgni_runtime_efficiency_by_run.csv",
    "csgni_runtime_efficiency_summary.csv": SOURCE_ROOT / "statistics/csgni_runtime_efficiency_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing Phase6H efficiency source: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty efficiency table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select(name: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if name == "efficiency_pairwise.csv":
        return [
            row for row in rows
            if row.get("method_a") == "PHASE6H_CSGNI" and row.get("method_b") == "ALNS"
        ]
    if name.startswith("csgni_runtime_efficiency"):
        return [row for row in rows if row.get("method") == "PHASE6H_CSGNI"]
    return [row for row in rows if row.get("method") in METHODS]


def unique_count(rows: list[dict[str, str]], field: str) -> int:
    return len({row[field] for row in rows})


def main() -> int:
    integrity: dict[str, Any] = json.loads(INTEGRITY_PATH.read_text(encoding="utf-8"))
    if not all((
        integrity.get("status") == "PASS",
        integrity.get("all_incumbent_traces_validated") is True,
        integrity.get("all_schedules_feasible") is True,
        integrity.get("cal_holdout_used_for_selection") is False,
        integrity.get("common_reference_type") == "POOLED_BKS",
    )):
        raise RuntimeError("Phase6H efficiency integrity gate is not PASS")
    artifact_manifest: dict[str, Any] = json.loads(
        ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if artifact_manifest.get("status") != "PASS":
        raise RuntimeError("Phase6H artifact manifest is not PASS")
    for relative, expected in artifact_manifest["key_files"].items():
        if relative in {
            "outputs/phase6h_validation/validation_run_summary.csv",
            "outputs/phase6h_validation/statistics/method_summary.csv",
        }:
            path = ROOT / relative
            if sha256(path) != expected:
                raise RuntimeError(f"frozen Phase6H source hash mismatch: {relative}")

    selected: dict[str, list[dict[str, str]]] = {}
    source_hashes: dict[str, str] = {}
    for output_name, source_path in SOURCES.items():
        rows = read_csv(source_path)
        selected[output_name] = select(output_name, rows)
        source_hashes[str(source_path.relative_to(ROOT))] = sha256(source_path)

    expected_counts = {
        "efficiency_runs.csv": 90,
        "efficiency_instance_summary.csv": 18,
        "efficiency_method_summary.csv": 2,
        "efficiency_pairwise.csv": 1,
        "anytime_curves.csv": 540,
        "anytime_runs.csv": 90,
        "anytime_summary.csv": 2,
        "target_hit_summary.csv": 8,
        "csgni_runtime_efficiency_by_run.csv": 45,
        "csgni_runtime_efficiency_summary.csv": 1,
    }
    observed_counts = {name: len(rows) for name, rows in selected.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Phase6H efficiency matrix mismatch: expected={expected_counts} observed={observed_counts}"
        )
    runs = selected["efficiency_runs.csv"]
    if unique_count(runs, "instance_id") != 9:
        raise RuntimeError("efficiency evidence must contain nine CAL-HOLDOUT instances")
    for method in METHODS:
        method_rows = [row for row in runs if row["method"] == method]
        if len(method_rows) != 45 or any(row["feasible"] != "True" for row in method_rows):
            raise RuntimeError(f"invalid 9 instance x 5 seed efficiency matrix for {method}")
        keys = [(row["instance_id"], row["seed"]) for row in method_rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate efficiency run keys for {method}")

    for name, rows in selected.items():
        atomic_csv(OUTPUT_ROOT / name, rows)
    inventory = {
        "schema": "initial-manuscript-phase6h-alns-efficiency-v1",
        "status": "PASS_REUSED_PHASE6H_CAL_HOLDOUT",
        "methods": ["ALNS", "PHASE6H_CSGNI"],
        "manuscript_identity": {
            "PHASE6H_CSGNI": "CSG-NI Phase6H provisional",
            "ALNS": "ALNS efficiency comparator only",
        },
        "instance_count": 9,
        "matched_runs_per_method": 45,
        "all_schedules_feasible": True,
        "all_incumbent_traces_validated": True,
        "common_reference_type": "POOLED_BKS",
        "cal_holdout_used_for_selection": False,
        "live_state_drift": integrity.get("live_state_drift"),
        "reporting_limit": (
            "Phase6H remains provisional; ALNS evidence is for efficiency/anytime only, "
            "not the five-method Core ranking"
        ),
        "source_integrity_sha256": sha256(INTEGRITY_PATH),
        "source_artifact_manifest_sha256": sha256(ARTIFACT_MANIFEST_PATH),
        "source_hashes": source_hashes,
        "output_row_counts": observed_counts,
    }
    atomic_json(OUTPUT_ROOT / "efficiency_inventory.json", inventory)
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
