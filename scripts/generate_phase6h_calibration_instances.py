#!/usr/bin/env python3
"""Generate and verify the isolated Phase 6H CAL-FIT/CAL-HOLDOUT instances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics  # noqa: E402
from rcias_clgri.data.generation import write_json  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json  # noqa: E402
from rcias_clgri.instances.controlled_generator import (  # noqa: E402
    acceptance_failures,
    configuration_entropy,
    generate_candidate,
    scale_sensitivity_variant,
)


CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
GENERATION_SPEC_PATH = ROOT / "configs/rcias_cb1_generation.json"
TARGET = ROOT / "instances/controlled/RCIAS-CB1-CAL"
MANIFESTS = TARGET / "manifests"
SPLITS = {7: "CAL_FIT", 8: "CAL_HOLDOUT"}
SCALES = ("S", "M", "L")
CFS = ("CF1", "CF2", "CF3")


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
        raw = generate_candidate(
            instance_id, "TRAIN_CALIBRATION_BASE", scale, cf, final_seed, spec
        )
        failures = acceptance_failures(raw, scale, cf, spec)
        history.append({
            "attempt": attempt,
            "final_seed": final_seed,
            "failures": failures,
        })
        if not failures:
            return raw, final_seed, history
    raise RuntimeError(f"{instance_id} failed structural acceptance: {history[-1]}")


def historical_instance_hashes() -> set[str]:
    hashes = set()
    for path in (ROOT / "instances").rglob("*.json"):
        if TARGET in path.parents or "manifests" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(raw, dict) and "instance_id" in raw.get("meta", {}):
            hashes.add(digest(path))
    return hashes


def historical_ids_and_seeds() -> tuple[set[str], set[int]]:
    ids: set[str] = set()
    seeds: set[int] = set()
    for path in (ROOT / "instances").rglob("*.json"):
        if TARGET in path.parents or "manifests" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
        if meta.get("instance_id") is not None:
            ids.add(str(meta["instance_id"]))
        if meta.get("seed") is not None:
            seeds.add(int(meta["seed"]))
    for path in (ROOT / "instances").rglob("*.csv"):
        if TARGET in path.parents:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "instance_id" in frame:
            ids.update(frame["instance_id"].dropna().astype(str))
        for column in (
            "base_seed", "final_seed", "base_generation_seed",
            "final_generation_seed", "trajectory_seed", "state_sampling_seed",
        ):
            if column in frame:
                seeds.update(frame[column].dropna().astype(int))
    return ids, seeds


def generate() -> None:
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        raise RuntimeError("RCIAS-CB1-CAL already exists; use --verify-only")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(GENERATION_SPEC_PATH.read_text(encoding="utf-8"))
    base_namespace = int(config["calibration_instances"]["base_seed_namespace"])
    for folder in ("cal_fit", "cal_holdout"):
        (TARGET / folder).mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}
    for replicate, split in SPLITS.items():
        for scale_index, scale in enumerate(SCALES, 1):
            for cf_index, cf in enumerate(CFS, 1):
                base_id = f"CB1_CAL_BASE_{scale}_{cf}_R{replicate:02d}"
                base_seed = base_namespace + scale_index * 100 + cf_index * 10 + replicate
                base, final_seed, history = accepted_base(
                    base_id, scale, cf, base_seed, spec
                )
                histories[base_id] = history
                instance_id = f"CB1_CAL_{scale}_{cf}_RI2_TI2_R{replicate:02d}"
                raw = scale_sensitivity_variant(base, instance_id, "RI2", "TI2", spec)
                raw["meta"].update({
                    "suite": "TRAIN_CALIBRATION_ONLY",
                    "calibration_split": split,
                    "base_structure": base_id,
                })
                folder = split.lower()
                path = TARGET / folder / f"{instance_id}.json"
                write_json(raw, path)
                instance = load_instance(path)
                metrics = benchmark_metrics(instance)
                row = {
                    "instance_id": instance_id,
                    "calibration_split": split,
                    "scale": scale,
                    "CF_level": cf,
                    "RI_level": "RI2",
                    "TI_level": "TI2",
                    "replicate": f"R{replicate:02d}",
                    "base_structure": base_id,
                    "base_generation_seed": base_seed,
                    "final_generation_seed": final_seed,
                    "relative_path": str(path.relative_to(TARGET)),
                    "sha256": digest(path),
                }
                rows.append(row)
                metric_rows.append({
                    **{key: row[key] for key in (
                        "instance_id", "calibration_split", "scale", "CF_level",
                        "RI_level", "TI_level", "replicate", "base_structure",
                    )},
                    **metrics,
                    "configuration_entropy": configuration_entropy(raw),
                })

    manifest = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values("instance_id").reset_index(drop=True)
    atomic_write_csv(manifest, MANIFESTS / "calibration_instance_manifest.csv")
    atomic_write_csv(metrics, MANIFESTS / "calibration_structural_metrics.csv")
    atomic_write_json(histories, MANIFESTS / "generation_history.json")
    atomic_write_json({
        "schema": "rcias-cb1-cal-generation-v1",
        "source_spec_sha256": digest(GENERATION_SPEC_PATH),
        "phase6h_config_sha256": digest(CONFIG_PATH),
        "base_seed_namespace": base_namespace,
        "factorial_design": {
            "scale": list(SCALES),
            "CF": list(CFS),
            "RI": ["RI2"],
            "TI": ["TI2"],
            "replicate_split": {f"R{key:02d}": value for key, value in SPLITS.items()},
        },
        "cal_holdout_use": "FORBIDDEN_UNTIL_PHASE6H_POLICY_ARTIFACT_IS_FROZEN",
    }, MANIFESTS / "generation_spec.json")
    verify()
    print("PHASE6H_CAL_INSTANCES_CREATED count=18 cal_fit=9 cal_holdout=9")


def verify() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = MANIFESTS / "calibration_instance_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("Phase 6H calibration manifest is missing")
    manifest = pd.read_csv(manifest_path)
    historical_hashes = historical_instance_hashes()
    historical_ids, historical_seeds = historical_ids_and_seeds()
    failures: list[str] = []
    for row in manifest.to_dict("records"):
        path = TARGET / row["relative_path"]
        checksum = digest(path)
        instance = load_instance(path)
        if (
            checksum != row["sha256"]
            or checksum in historical_hashes
            or instance.instance_id != row["instance_id"]
            or row["instance_id"] in historical_ids
            or int(row["base_generation_seed"]) in historical_seeds
            or int(row["final_generation_seed"]) in historical_seeds
        ):
            failures.append(row["instance_id"])
    split_counts = manifest.calibration_split.value_counts().to_dict()
    cell_counts = manifest.groupby(["calibration_split", "scale", "CF_level"]).size()
    expected_search_seeds = {
        int(seed) for values in config["seeds"].values() for seed in values
    }
    generation_seeds = set(manifest.base_generation_seed.astype(int)) | set(
        manifest.final_generation_seed.astype(int)
    )
    checks = {
        "exactly_18_unique_instances": len(manifest) == 18 and manifest.instance_id.nunique() == 18,
        "exactly_9_per_split": split_counts == {"CAL_FIT": 9, "CAL_HOLDOUT": 9},
        "balanced_scale_cf_cells": len(cell_counts) == 18 and set(cell_counts) == {1},
        "fixed_ri2_ti2": set(manifest.RI_level) == {"RI2"} and set(manifest.TI_level) == {"TI2"},
        "replicate_split_exact": (
            set(manifest.loc[manifest.calibration_split == "CAL_FIT", "replicate"]) == {"R07"}
            and set(manifest.loc[manifest.calibration_split == "CAL_HOLDOUT", "replicate"]) == {"R08"}
        ),
        "all_loadable_and_hash_exact": not failures,
        "zero_historical_id_or_hash_overlap": not failures,
        "generation_seeds_unique": manifest.base_generation_seed.nunique() == 18 and manifest.final_generation_seed.nunique() == 18,
        "generation_and_search_seeds_disjoint": not bool(generation_seeds & expected_search_seeds),
        "phase6h_config_hash_matches": (
            json.loads((MANIFESTS / "generation_spec.json").read_text(encoding="utf-8"))[
                "phase6h_config_sha256"
            ] == digest(CONFIG_PATH)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "failures": failures})
    checksum_lines = "".join(
        f"{row['sha256']}  {row['relative_path']}\n"
        for row in manifest.sort_values("relative_path").to_dict("records")
    )
    (MANIFESTS / "checksums.sha256").write_text(checksum_lines, encoding="utf-8")
    atomic_write_json({
        "schema": "phase6h-calibration-instance-audit-v1",
        "checks": checks,
        "split_counts": split_counts,
        "status": "PASS",
    }, MANIFESTS / "calibration_instance_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else generate()
    if args.verify_only:
        print("PHASE6H_CAL_INSTANCES_VERIFY = TRUE")


if __name__ == "__main__":
    main()
