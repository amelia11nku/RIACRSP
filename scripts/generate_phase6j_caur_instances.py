#!/usr/bin/env python3
"""Generate and verify the disjoint Phase 6J R12/R13/R14 instance suite."""

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


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
GENERATION_SPEC_PATH = ROOT / "configs/rcias_cb1_generation.json"
TARGET = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14"
MANIFESTS = TARGET / "manifests"
SPLITS = {
    12: ("CAUR_FIT", "r12_caur_fit"),
    13: ("CAUR_SELECT", "r13_caur_select"),
    14: ("CAUR_HOLDOUT", "r14_caur_holdout"),
}
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
            instance_id, "TRAIN_CAUR_BASE", scale, cf, final_seed, spec
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


def historical_inventory() -> tuple[set[str], set[str], set[int]]:
    """Inventory every prior suite while excluding only the Phase 6J target."""
    hashes: set[str] = set()
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
            hashes.add(digest(path))
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
            "base_seed",
            "final_seed",
            "base_generation_seed",
            "final_generation_seed",
            "trajectory_seed",
            "state_sampling_seed",
        ):
            if column in frame:
                seeds.update(frame[column].dropna().astype(int))
    return hashes, ids, seeds


def generate() -> None:
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        raise RuntimeError("Phase 6J suite already exists; use --verify-only")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(GENERATION_SPEC_PATH.read_text(encoding="utf-8"))
    for _, folder in SPLITS.values():
        (TARGET / folder).mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    metrics: list[dict] = []
    histories: dict[str, list[dict]] = {}
    namespace = int(config["instance_suite"]["base_seed_namespace"])
    second_offset = int(config["instance_suite"]["second_cell_replicate_seed_offset"])
    for cell_replicate in config["instance_suite"]["cell_replicates"]:
        suffix = "" if cell_replicate == "C01" else "_C02"
        offset = 0 if cell_replicate == "C01" else second_offset
        for replicate, (split, folder) in SPLITS.items():
            for scale_index, scale in enumerate(SCALES, 1):
                for cf_index, cf in enumerate(CFS, 1):
                    base_id = f"CB1_CAUR_BASE_{scale}_{cf}_R{replicate:02d}{suffix}"
                    base_seed = namespace + offset + scale_index * 100 + cf_index * 10 + replicate
                    base, final_seed, history = accepted_base(
                        base_id, scale, cf, base_seed, spec
                    )
                    histories[base_id] = history
                    instance_id = f"CB1_CAUR_{scale}_{cf}_RI2_TI2_R{replicate:02d}{suffix}"
                    raw = scale_sensitivity_variant(base, instance_id, "RI2", "TI2", spec)
                    raw["meta"].update({
                        "suite": "TRAIN_CAUR_ONLY",
                        "caur_split": split,
                        "base_structure": base_id,
                        "cell_replicate": cell_replicate,
                    })
                    path = TARGET / folder / f"{instance_id}.json"
                    write_json(raw, path)
                    row = {
                        "instance_id": instance_id,
                        "caur_split": split,
                        "scale": scale,
                        "CF_level": cf,
                        "RI_level": "RI2",
                        "TI_level": "TI2",
                        "replicate": f"R{replicate:02d}",
                        "cell_replicate": cell_replicate,
                        "base_structure": base_id,
                        "base_generation_seed": base_seed,
                        "final_generation_seed": final_seed,
                        "relative_path": str(path.relative_to(TARGET)),
                        "sha256": digest(path),
                        "size_bytes": path.stat().st_size,
                    }
                    rows.append(row)
                    if split != "CAUR_HOLDOUT":
                        instance = load_instance(path)
                        metrics.append({
                            **{key: row[key] for key in (
                                "instance_id", "caur_split", "scale", "CF_level",
                                "RI_level", "TI_level", "replicate", "cell_replicate",
                                "base_structure",
                            )},
                            **benchmark_metrics(instance),
                            "configuration_entropy": configuration_entropy(raw),
                        })

    manifest = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    structural = pd.DataFrame(metrics).sort_values("instance_id").reset_index(drop=True)
    atomic_write_csv(manifest, MANIFESTS / "phase6j_instance_manifest.csv")
    atomic_write_csv(structural, MANIFESTS / "r12_r13_structural_metrics.csv")
    atomic_write_json(histories, MANIFESTS / "generation_history.json")
    atomic_write_json({
        "schema": "phase6j-caur-generation-spec-v1",
        "source_generation_spec_sha256": digest(GENERATION_SPEC_PATH),
        "base_seed_namespace": namespace,
        "second_cell_replicate_seed_offset": second_offset,
        "factorial_design": {
            "scale": list(SCALES),
            "CF": list(CFS),
            "RI": ["RI2"],
            "TI": ["TI2"],
            "cell_replicate": ["C01", "C02"],
            "replicate_split": {f"R{key:02d}": value[0] for key, value in SPLITS.items()},
        },
        "r14_pre_freeze_metrics": "WITHHELD",
        "r14_boundary": "instance payload access is locked until deployable artifact freeze",
    }, MANIFESTS / "generation_spec.json")
    verify()
    print("PHASE6J_CAUR_INSTANCES_CREATED count=54 r12=18 r13=18 r14=18")


def verify() -> None:
    manifest_path = MANIFESTS / "phase6j_instance_manifest.csv"
    if not manifest_path.is_file():
        raise RuntimeError("Phase 6J instance manifest is missing")
    manifest = pd.read_csv(manifest_path)
    historical_hashes, historical_ids, historical_seeds = historical_inventory()
    failures: list[str] = []
    parsed_splits: set[str] = set()
    for row in manifest.to_dict("records"):
        path = TARGET / row["relative_path"]
        checksum = digest(path)
        common_failure = (
            checksum != row["sha256"]
            or int(path.stat().st_size) != int(row["size_bytes"])
            or checksum in historical_hashes
            or row["instance_id"] in historical_ids
            or int(row["base_generation_seed"]) in historical_seeds
            or int(row["final_generation_seed"]) in historical_seeds
        )
        instance_failure = False
        if row["caur_split"] != "CAUR_HOLDOUT":
            instance = load_instance(path)
            parsed_splits.add(str(row["caur_split"]))
            instance_failure = instance.instance_id != row["instance_id"]
        if common_failure or instance_failure:
            failures.append(str(row["instance_id"]))

    split_counts = manifest.caur_split.value_counts().to_dict()
    cell_counts = manifest.groupby(["caur_split", "scale", "CF_level"]).size()
    split_hashes = {
        split: set(group.sha256) for split, group in manifest.groupby("caur_split")
    }
    generation_seeds = set(manifest.base_generation_seed.astype(int)) | set(
        manifest.final_generation_seed.astype(int)
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_seeds = {
        int(value)
        for key, value in config["rng"].items()
        if key.endswith("_namespace")
    } | {
        int(seed)
        for key, values in config["rng"].items()
        if key.endswith("_seeds")
        for seed in values
    } | {int(config["rng"]["grouped_bootstrap_seed"])}
    checks = {
        "exactly_54_unique_instances": len(manifest) == manifest.instance_id.nunique() == 54,
        "exactly_18_per_split": split_counts == {
            "CAUR_FIT": 18, "CAUR_SELECT": 18, "CAUR_HOLDOUT": 18,
        },
        "two_instances_per_scale_cf_cell": len(cell_counts) == 27 and set(cell_counts.tolist()) == {2},
        "cell_replicates_exact": (
            set(manifest.cell_replicate) == {"C01", "C02"}
            and manifest.groupby(["caur_split", "scale", "CF_level"])
            .cell_replicate.apply(set).eq({"C01", "C02"}).all()
        ),
        "fixed_ri2_ti2": set(manifest.RI_level) == {"RI2"} and set(manifest.TI_level) == {"TI2"},
        "split_hashes_disjoint": all(
            left.isdisjoint(right)
            for index, left in enumerate(split_hashes.values())
            for right in list(split_hashes.values())[index + 1 :]
        ),
        "historical_ids_hashes_seeds_disjoint": not failures,
        "experiment_and_generation_seeds_disjoint": generation_seeds.isdisjoint(experiment_seeds),
        "r14_not_loaded_for_metrics": parsed_splits == {"CAUR_FIT", "CAUR_SELECT"},
        "manifest_hashes_unique": manifest.sha256.nunique() == len(manifest),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema": "phase6j-caur-instance-integrity-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failures": failures,
        "manifest_sha256": digest(manifest_path),
        "r14_access_scope": "MANIFEST_ID_HASH_SIZE_ONLY",
    }
    atomic_write_json(audit, MANIFESTS / "phase6j_integrity_audit.json")
    if audit["status"] != "PASS":
        raise RuntimeError(f"Phase 6J instance verification failed: {audit}")
    print("PHASE6J_CAUR_INSTANCES_VERIFIED status=PASS count=54")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else generate()


if __name__ == "__main__":
    main()
