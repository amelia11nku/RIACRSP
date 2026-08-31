#!/usr/bin/env python3
"""Generate and validate the isolated 81-instance R06 revision holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.instances.controlled_generator import (
    acceptance_failures,
    configuration_entropy,
    generate_candidate,
    scale_sensitivity_variant,
)


TARGET = ROOT / "instances" / "controlled" / "RCIAS-CB1-TRAIN-R06"
MANIFESTS = TARGET / "manifests"
OUT = ROOT / "outputs" / "phase6f" / "revision_holdout" / "instances"
CONFIG_PATH = ROOT / "configs" / "phase6f_revision.json"
GENERATION_SPEC_PATH = ROOT / "configs" / "rcias_cb1_generation.json"
SCALES = ("S", "M", "L")
CFS = ("CF1", "CF2", "CF3")
RIS = ("RI1", "RI2", "RI3")
TIS = ("TI1", "TI2", "TI3")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_base(
    instance_id: str,
    scale: str,
    cf: str,
    base_seed: int,
    spec: dict,
) -> tuple[dict, int, list[dict]]:
    history = []
    for attempt in range(1, int(spec["max_attempts"]) + 1):
        final_seed = base_seed * 1000 + attempt
        raw = generate_candidate(instance_id, "TRAIN_REVISION_HOLDOUT_BASE", scale, cf, final_seed, spec)
        failures = acceptance_failures(raw, scale, cf, spec)
        history.append({"attempt": attempt, "final_seed": final_seed, "failures": failures})
        if not failures:
            return raw, final_seed, history
    raise RuntimeError(f"{instance_id} failed structural acceptance: {history[-1]}")


def historical_instance_hashes() -> set[str]:
    values = set()
    for path in (ROOT / "instances").rglob("*.json"):
        if TARGET in path.parents or "manifests" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(raw, dict) and isinstance(raw.get("meta"), dict) and "instance_id" in raw["meta"]:
            values.add(digest(path))
    return values


def historical_seed_values() -> set[int]:
    values: set[int] = set()
    paths = [
        ROOT / "instances" / "controlled" / "RCIAS-CB1-TRAIN" / "manifests" / "train_instance_manifest.csv",
        ROOT / "instances" / "controlled" / "RCIAS-CB1" / "manifests" / "benchmark_manifest.csv",
    ]
    for path in paths:
        frame = pd.read_csv(path)
        for column in (
            "base_generation_seed",
            "final_generation_seed",
            "trajectory_seed",
            "state_sampling_seed",
            "base_seed",
            "final_seed",
        ):
            if column in frame:
                values.update(frame[column].dropna().astype(int))
    for path in (ROOT / "instances" / "canonical" / "RCIAS-2.0").rglob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        seed = raw.get("meta", {}).get("seed") if isinstance(raw, dict) else None
        if seed is not None:
            values.add(int(seed))
    values.update(range(610001, 610011))
    return values


def generate() -> None:
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        raise RuntimeError("R06 revision instances already exist; use --verify-only")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(GENERATION_SPEC_PATH.read_text(encoding="utf-8"))
    base_namespace = int(config["seed_namespaces"]["instance_base_generation"])
    (TARGET / "revision_holdout").mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}
    for scale_index, scale in enumerate(SCALES, 1):
        for cf_index, cf in enumerate(CFS, 1):
            base_id = f"CB1_TRAIN_BASE_{scale}_{cf}_R06"
            base_seed = base_namespace + scale_index * 100 + cf_index * 10 + 6
            base, final_seed, history = accepted_base(base_id, scale, cf, base_seed, spec)
            histories[base_id] = history
            for ri in RIS:
                for ti in TIS:
                    instance_id = f"CB1_TRAIN_{scale}_{cf}_{ri}_{ti}_R06"
                    raw = scale_sensitivity_variant(base, instance_id, ri, ti, spec)
                    raw["meta"].update({
                        "suite": "TRAIN_REVISION_HOLDOUT_ONLY",
                        "training_split": "REVISION_HOLDOUT",
                        "base_structure": base_id,
                    })
                    path = TARGET / "revision_holdout" / f"{instance_id}.json"
                    write_json(raw, path)
                    instance = load_instance(path)
                    metrics = benchmark_metrics(instance)
                    row = {
                        "instance_id": instance_id,
                        "training_split": "REVISION_HOLDOUT",
                        "scale": scale,
                        "CF_level": cf,
                        "RI_level": ri,
                        "TI_level": ti,
                        "replicate": "R06",
                        "base_structure": base_id,
                        "base_generation_seed": base_seed,
                        "final_generation_seed": final_seed,
                        "trajectory_seed_namespace": config["seed_namespaces"]["trajectory"],
                        "state_sampling_seed_namespace": config["seed_namespaces"]["state_sampling"],
                        "counterfactual_arm_seed_namespace": config["seed_namespaces"]["arm_generation"],
                        "repair_seed_namespace": config["seed_namespaces"]["repair"],
                        "relative_path": str(path.relative_to(TARGET)),
                        "sha256": digest(path),
                    }
                    rows.append(row)
                    metric_rows.append({
                        **{
                            key: row[key]
                            for key in (
                                "instance_id",
                                "training_split",
                                "scale",
                                "CF_level",
                                "RI_level",
                                "TI_level",
                                "replicate",
                                "base_structure",
                            )
                        },
                        **metrics,
                        "configuration_entropy": configuration_entropy(raw),
                    })

    manifest = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values("instance_id").reset_index(drop=True)
    atomic_write_csv(manifest, MANIFESTS / "revision_instance_manifest.csv")
    atomic_write_csv(manifest, OUT / "revision_instance_manifest.csv")
    atomic_write_csv(metrics, OUT / "revision_structural_metrics.csv")
    atomic_write_json(histories, MANIFESTS / "generation_history.json")
    atomic_write_json(
        {
            "schema": "rcias-cb1-train-r06-generation-v1",
            "suite": config["revision_holdout"]["suite"],
            "replicate": "R06",
            "base_seed_namespace": base_namespace,
            "source_spec_sha256": digest(GENERATION_SPEC_PATH),
            "phase6f_config_sha256": digest(CONFIG_PATH),
            "factorial_design": {
                "scale": list(SCALES),
                "CF": list(CFS),
                "RI": list(RIS),
                "TI": list(TIS),
                "replicates": 1,
            },
            "training_use": "FORBIDDEN; final untouched revision holdout only",
        },
        MANIFESTS / "generation_spec.json",
    )
    verify()
    print("PHASE6F_R06_INSTANCES_CREATED count=81 cells=81")


def verify() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = MANIFESTS / "revision_instance_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("R06 instance manifest is missing")
    manifest = pd.read_csv(manifest_path)
    cells = manifest.groupby(["scale", "CF_level", "RI_level", "TI_level"]).size()
    historical_hashes = historical_instance_hashes()
    historical_seeds = historical_seed_values()
    failures = []
    feasibility = []
    for row in manifest.to_dict("records"):
        path = TARGET / row["relative_path"]
        checksum = digest(path)
        instance = load_instance(path)
        result = solve_dispatching(instance, "H1")
        audit = check_schedule(instance, result.schedule)
        feasibility.append({
            "instance_id": instance.instance_id,
            "h1_makespan": result.objective.makespan,
            "h1_feasible": bool(audit["feasible"]),
        })
        if checksum != row["sha256"] or checksum in historical_hashes:
            failures.append(row["instance_id"])

    metrics = pd.read_csv(OUT / "revision_structural_metrics.csv")
    seed_values = set(manifest["base_generation_seed"].astype(int)) | set(
        manifest["final_generation_seed"].astype(int)
    ) | set(int(value) for value in config["seed_namespaces"].values())
    checks = {
        "exactly_81_instances": len(manifest) == 81 and manifest.instance_id.nunique() == 81,
        "exactly_81_structural_cells": len(cells) == 81 and set(cells) == {1},
        "revision_split_only": set(manifest.training_split) == {"REVISION_HOLDOUT"},
        "replicate_r06_only": set(manifest.replicate) == {"R06"},
        "all_files_loadable_and_hash_exact": not failures,
        "zero_historical_instance_hash_overlap": not failures,
        "seed_namespaces_disjoint": not bool(seed_values & historical_seeds),
        "all_h1_schedules_feasible": all(row["h1_feasible"] for row in feasibility),
        "ri_ordering_valid": metrics.groupby("RI_level").RI.mean().reindex(RIS).is_monotonic_increasing,
        "w_ti_ordering_valid": (
            metrics.groupby("TI_level").W_transport_intensity.mean().reindex(TIS).is_monotonic_increasing
        ),
        "f_ti_ordering_valid": (
            metrics.groupby("TI_level").F_transport_intensity.mean().reindex(TIS).is_monotonic_increasing
        ),
        "phase6f_config_hash_matches_generation_spec": (
            json.loads((MANIFESTS / "generation_spec.json").read_text(encoding="utf-8"))[
                "phase6f_config_sha256"
            ]
            == digest(CONFIG_PATH)
        ),
    }
    atomic_write_csv(pd.DataFrame(feasibility), OUT / "revision_feasibility.csv")
    checksum_lines = "".join(
        f"{row['sha256']}  {row['relative_path']}\n"
        for row in manifest.sort_values("relative_path").to_dict("records")
    )
    (MANIFESTS / "checksums.sha256").write_text(checksum_lines, encoding="utf-8")
    audit_record = {
        "schema": "phase6f-r06-instance-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "instance_count": len(manifest),
        "structural_cell_count": len(cells),
        "manifest_sha256": digest(manifest_path),
        "historical_hash_overlap": failures,
    }
    audit_path = OUT / "revision_instance_audit.json"
    atomic_write_json(audit_record, audit_path)
    if audit_record["status"] != "PASS":
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    print("PHASE6F_R06_INSTANCES_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else generate()


if __name__ == "__main__":
    main()
