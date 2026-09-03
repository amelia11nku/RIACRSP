#!/usr/bin/env python3
"""Compute manuscript Core BKS/RPD only after the frozen matrix is complete."""

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
PAPER_ROOT = ROOT / "paper_experiments"
INPUT_ROOT = PAPER_ROOT / "processed_data/core"
INVENTORY_PATH = INPUT_ROOT / "source_inventory.json"
RUNS_PATH = INPUT_ROOT / "observed_runs.csv"
OUTPUT_ROOT = PAPER_ROOT / "processed_data/main"
EXPECTED_METHODS = (
    "GA",
    "Adapted DCGA",
    "DABC-RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M",
    "CSG-NI Phase6H provisional",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    position = 1
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        average = (position + (position + end - index - 1)) / 2.0
        for method, _ in ordered[index:end]:
            ranks[method] = average
        position += end - index
        index = end
    return ranks


def main() -> int:
    if not INVENTORY_PATH.is_file() or not RUNS_PATH.is_file():
        raise RuntimeError("run collect_core_results.py before Core BKS/RPD")
    inventory: dict[str, Any] = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if inventory.get("status") != "PASS_COMPLETE" or not inventory.get(
        "bks_rpd_authorized", False
    ):
        raise RuntimeError(
            "Core matrix is incomplete; BKS/RPD remain blocked by source_inventory.json"
        )
    with RUNS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if row["analysis_scope"] == "PRIMARY"]
    expected_instances = int(inventory["instance_count"])
    expected_seeds = len(inventory["primary_seeds"])
    expected_total = expected_instances * expected_seeds * len(EXPECTED_METHODS)
    if len(primary) != expected_total:
        raise RuntimeError(f"expected {expected_total} primary rows, found {len(primary)}")
    keys = [(row["method"], row["instance_id"], row["seed"]) for row in primary]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Core method/instance/seed keys")
    if set(row["method"] for row in primary) != set(EXPECTED_METHODS):
        raise RuntimeError("Core method set differs from manuscript protocol")

    by_instance: dict[str, list[dict[str, str]]] = {}
    by_instance_method: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in primary:
        by_instance.setdefault(row["instance_id"], []).append(row)
        by_instance_method.setdefault((row["instance_id"], row["method"]), []).append(row)
    if len(by_instance) != expected_instances:
        raise RuntimeError("Core instance count mismatch")

    bks: dict[str, float] = {}
    bks_rows: list[dict[str, object]] = []
    for instance_id, instance_rows in sorted(by_instance.items()):
        value = min(float(row["best_makespan"]) for row in instance_rows)
        bks[instance_id] = value
        sources = sorted({
            row["method"] for row in instance_rows
            if math.isclose(float(row["best_makespan"]), value, abs_tol=1e-9)
        })
        bks_rows.append({
            "instance_id": instance_id,
            "scale": instance_rows[0]["scale"],
            "CF_level": instance_rows[0]["CF_level"],
            "operation_count": int(instance_rows[0]["operation_count"]),
            "draft_bks": value,
            "attaining_methods": ";".join(sources),
            "status": "DRAFT_PROVISIONAL_PHASE6H",
        })

    run_rows: list[dict[str, object]] = []
    for row in sorted(primary, key=lambda item: (item["instance_id"], item["method"], int(item["seed"]))):
        value = float(row["best_makespan"])
        reference = bks[row["instance_id"]]
        output: dict[str, object] = dict(row)
        output["draft_bks"] = reference
        output["rpd_percent"] = 100.0 * (value - reference) / reference
        output["bks_status"] = "DRAFT_PROVISIONAL_PHASE6H"
        run_rows.append(output)

    instance_summaries: list[dict[str, object]] = []
    ranks_by_instance: dict[str, dict[str, float]] = {}
    for instance_id in sorted(by_instance):
        medians = {
            method: statistics.median(
                float(row["best_makespan"])
                for row in by_instance_method[(instance_id, method)]
            )
            for method in EXPECTED_METHODS
        }
        ranks_by_instance[instance_id] = average_ranks(medians)
        for method in EXPECTED_METHODS:
            group = by_instance_method[(instance_id, method)]
            if len(group) != expected_seeds:
                raise RuntimeError(f"incomplete Core group: {method} / {instance_id}")
            objectives = [float(row["best_makespan"]) for row in group]
            rpds = [100.0 * (value - bks[instance_id]) / bks[instance_id] for value in objectives]
            instance_summaries.append({
                "method": method,
                "instance_id": instance_id,
                "scale": group[0]["scale"],
                "CF_level": group[0]["CF_level"],
                "operation_count": int(group[0]["operation_count"]),
                "run_count": len(group),
                "draft_bks": bks[instance_id],
                "best_makespan": min(objectives),
                "mean_makespan": statistics.fmean(objectives),
                "median_makespan": statistics.median(objectives),
                "std_makespan": statistics.stdev(objectives),
                "mean_rpd_percent": statistics.fmean(rpds),
                "median_rpd_percent": statistics.median(rpds),
                "median_objective_rank": ranks_by_instance[instance_id][method],
                "attains_bks": min(objectives) <= bks[instance_id] + 1e-9,
                "feasibility_rate": statistics.fmean(row["feasible"] == "True" for row in group),
            })

    scale_rows: list[dict[str, object]] = []
    for scale in ("S", "M", "L", "Overall"):
        for method in EXPECTED_METHODS:
            selected = [
                row for row in instance_summaries
                if row["method"] == method and (scale == "Overall" or row["scale"] == scale)
            ]
            scale_rows.append({
                "scale": scale,
                "method": method,
                "instance_count": len(selected),
                "mean_of_instance_mean_rpd_percent": statistics.fmean(
                    float(row["mean_rpd_percent"]) for row in selected
                ),
                "median_of_instance_median_rpd_percent": statistics.median(
                    float(row["median_rpd_percent"]) for row in selected
                ),
                "average_rank": statistics.fmean(
                    float(row["median_objective_rank"]) for row in selected
                ),
                "bks_attainment_count": sum(bool(row["attains_bks"]) for row in selected),
                "mean_runtime_seconds": statistics.fmean(
                    float(row["runtime_seconds"]) for row in run_rows
                    if row["method"] == method and (scale == "Overall" or row["scale"] == scale)
                ),
            })

    atomic_csv(OUTPUT_ROOT / "draft_bks_manifest.csv", bks_rows)
    atomic_csv(OUTPUT_ROOT / "main_runs.csv", run_rows)
    atomic_csv(OUTPUT_ROOT / "main_instance_summary.csv", instance_summaries)
    atomic_csv(OUTPUT_ROOT / "main_scale_summary.csv", scale_rows)
    manifest = {
        "schema": "initial-manuscript-core-bks-rpd-v1",
        "status": "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H",
        "source_inventory_sha256": sha256(INVENTORY_PATH),
        "observed_runs_sha256": sha256(RUNS_PATH),
        "method_count": len(EXPECTED_METHODS),
        "instance_count": len(by_instance),
        "run_count": len(run_rows),
        "bks_definition": "minimum makespan over five methods and five matched primary seeds",
        "rank_basis": "per-instance median makespan with average ties",
        "outputs": [
            "draft_bks_manifest.csv", "main_runs.csv",
            "main_instance_summary.csv", "main_scale_summary.csv",
        ],
    }
    atomic_json(OUTPUT_ROOT / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
