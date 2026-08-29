#!/usr/bin/env python3
"""Generate and freeze the isolated 405-instance RCIAS-CB1-TRAIN distribution."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance, load_instance_dict
from rcias_clgri.instances.controlled_generator import (
    acceptance_failures, configuration_entropy, generate_candidate, scale_sensitivity_variant,
)

TARGET = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
MANIFESTS = TARGET / "manifests"
OUT = ROOT / "outputs/phase6b/train_distribution"
BASE_NAMESPACE = 661000
SCALES = ("S", "M", "L")
CFS = ("CF1", "CF2", "CF3")
RIS = ("RI1", "RI2", "RI3")
TIS = ("TI1", "TI2", "TI3")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(replicate: int) -> str:
    return "TRAIN" if replicate <= 3 else ("TRAIN_VALIDATION" if replicate == 4 else "TRAIN_INTERNAL_HOLDOUT")


def accepted_base(instance_id, scale, cf, base_seed, spec):
    history = []
    for attempt in range(1, int(spec["max_attempts"]) + 1):
        final_seed = base_seed * 1000 + attempt
        raw = generate_candidate(instance_id, "TRAIN_BASE", scale, cf, final_seed, spec)
        failures = acceptance_failures(raw, scale, cf, spec)
        history.append({"attempt": attempt, "final_seed": final_seed, "failures": failures})
        if not failures:
            return raw, final_seed, history
    raise RuntimeError(f"{instance_id} failed structural acceptance: {history[-1]}")


def generate():
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        raise RuntimeError("RCIAS-CB1-TRAIN already exists; use --verify-only")
    spec = json.loads((ROOT / "configs/rcias_cb1_generation.json").read_text())
    for split in ("train", "train_validation", "train_internal_holdout"):
        (TARGET / split).mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    rows, metrics_rows, histories = [], [], {}
    for scale_index, scale in enumerate(SCALES, 1):
        for cf_index, cf in enumerate(CFS, 1):
            for replicate in range(1, 6):
                base_id = f"CB1_TRAIN_BASE_{scale}_{cf}_R{replicate:02d}"
                base_seed = BASE_NAMESPACE + scale_index * 100 + cf_index * 10 + replicate
                base, final_seed, history = accepted_base(base_id, scale, cf, base_seed, spec)
                histories[base_id] = history
                for ri in RIS:
                    for ti in TIS:
                        instance_id = f"CB1_TRAIN_{scale}_{cf}_{ri}_{ti}_R{replicate:02d}"
                        raw = scale_sensitivity_variant(base, instance_id, ri, ti, spec)
                        split = split_for(replicate)
                        raw["meta"].update({"suite": "TRAIN_ONLY", "training_split": split, "base_structure": base_id})
                        folder = split.lower()
                        path = TARGET / folder / f"{instance_id}.json"
                        write_json(raw, path)
                        instance = load_instance(path); metrics = benchmark_metrics(instance)
                        row = {
                            "instance_id": instance_id, "training_split": split, "scale": scale,
                            "CF_level": cf, "RI_level": ri, "TI_level": ti,
                            "replicate": f"R{replicate:02d}", "base_structure": base_id,
                            "base_generation_seed": base_seed, "final_generation_seed": final_seed,
                            "trajectory_seed": 662000000 + scale_index * 10000 + cf_index * 1000 + replicate * 100 + RIS.index(ri) * 10 + TIS.index(ti),
                            "state_sampling_seed": 663000000 + scale_index * 10000 + cf_index * 1000 + replicate * 100 + RIS.index(ri) * 10 + TIS.index(ti),
                            "counterfactual_arm_seed_namespace": 664000000,
                            "repair_seed_namespace": 665000000,
                            "relative_path": str(path.relative_to(TARGET)), "sha256": digest(path),
                        }
                        rows.append(row)
                        metrics_rows.append({**{key: row[key] for key in ("instance_id", "training_split", "scale", "CF_level", "RI_level", "TI_level", "replicate", "base_structure")},
                                             **metrics, "configuration_entropy": configuration_entropy(raw)})
    manifest = pd.DataFrame(rows); metrics = pd.DataFrame(metrics_rows)
    manifest.to_csv(MANIFESTS / "train_instance_manifest.csv", index=False)
    manifest.to_csv(OUT / "train_instance_manifest.csv", index=False)
    metrics.to_csv(OUT / "train_structural_metrics.csv", index=False)
    numeric = metrics.select_dtypes("number").columns
    metrics[numeric].corr().to_csv(OUT / "train_structural_metric_correlations.csv")
    (MANIFESTS / "generation_spec.json").write_text(json.dumps({
        "schema": "rcias-cb1-train-generation-v1", "base_seed_namespace": BASE_NAMESPACE,
        "source_spec_sha256": digest(ROOT / "configs/rcias_cb1_generation.json"),
        "factorial_design": {"scale": list(SCALES), "CF": list(CFS), "RI": list(RIS), "TI": list(TIS), "replicates": 5},
        "split_rule": {"R01-R03": "TRAIN", "R04": "TRAIN_VALIDATION", "R05": "TRAIN_INTERNAL_HOLDOUT"},
        "ri_factors": spec["sensitivity"]["ri_factors"], "ti_factors": spec["sensitivity"]["ti_factors"],
    }, indent=2, sort_keys=True) + "\n")
    (MANIFESTS / "generation_history.json").write_text(json.dumps(histories, indent=2, sort_keys=True) + "\n")
    verify()
    print("RCIAS_CB1_TRAIN_CREATED count=405 train=243 validation=81 holdout=81")


def verify():
    manifest_path = MANIFESTS / "train_instance_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("training manifest missing")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 405 or manifest.instance_id.nunique() != 405:
        raise RuntimeError("expected 405 unique training instances")
    expected_splits = {"TRAIN": 243, "TRAIN_VALIDATION": 81, "TRAIN_INTERNAL_HOLDOUT": 81}
    if manifest.training_split.value_counts().to_dict() != expected_splits:
        raise RuntimeError("training split count mismatch")
    cells = manifest.groupby(["training_split", "scale", "CF_level", "RI_level", "TI_level"]).size()
    if len(cells) != 243 or set(cells.loc["TRAIN"]) != {3} or set(cells.loc["TRAIN_VALIDATION"]) != {1} or set(cells.loc["TRAIN_INTERNAL_HOLDOUT"]) != {1}:
        raise RuntimeError("factorial split imbalance")
    evaluation_hashes = {
        digest(path) for root in (ROOT / "instances/canonical/RCIAS-2.0", ROOT / "instances/controlled/RCIAS-CB1")
        for path in root.rglob("*.json")
    }
    failures = []
    for row in manifest.to_dict("records"):
        path = TARGET / row["relative_path"]
        if digest(path) != row["sha256"] or digest(path) in evaluation_hashes:
            failures.append(row["instance_id"])
        load_instance(path)
    if failures:
        raise RuntimeError(f"training leakage/hash failures: {failures[:5]}")
    metrics = pd.read_csv(OUT / "train_structural_metrics.csv")
    required_metrics = {"number_of_operations", "F_route_mean", "F_cap_mean", "R_full_op", "R_full_island",
                        "processing_CV_mean", "RI", "W_transport_intensity", "F_transport_intensity",
                        "configuration_entropy"}
    if len(metrics) != 405 or not required_metrics <= set(metrics):
        raise RuntimeError("structural metrics audit incomplete")
    if not metrics.groupby("RI_level").RI.mean().reindex(RIS).is_monotonic_increasing:
        raise RuntimeError("RI factor ordering failed")
    if not metrics.groupby("TI_level").W_transport_intensity.mean().reindex(TIS).is_monotonic_increasing:
        raise RuntimeError("W transport factor ordering failed")
    if not metrics.groupby("TI_level").F_transport_intensity.mean().reindex(TIS).is_monotonic_increasing:
        raise RuntimeError("F transport factor ordering failed")
    test_seeds = set()
    for path in [ROOT / "instances/controlled/RCIAS-CB1/manifests/benchmark_manifest.csv"]:
        test_seeds.update(pd.read_csv(path).final_seed.astype(int))
    for path in (ROOT / "instances/canonical/RCIAS-2.0").rglob("*.json"):
        if path.name not in {"manifest.json", "generation_config.json"}:
            seed = json.loads(path.read_text()).get("meta", {}).get("seed")
            if seed is not None: test_seeds.add(int(seed))
    seed_values = set(manifest.final_generation_seed.astype(int)) | set(manifest.trajectory_seed.astype(int)) | set(manifest.state_sampling_seed.astype(int))
    if seed_values & test_seeds or set(range(610001, 610011)) & seed_values:
        raise RuntimeError("Phase 6B seed namespace overlaps frozen generation or Phase 6A")
    sample = manifest.sort_values("instance_id").iloc[0]
    spec = json.loads((ROOT / "configs/rcias_cb1_generation.json").read_text())
    regenerated_base, _, _ = accepted_base(sample.base_structure, sample.scale, sample.CF_level, int(sample.base_generation_seed), spec)
    regenerated = scale_sensitivity_variant(regenerated_base, sample.instance_id, sample.RI_level, sample.TI_level, spec)
    regenerated["meta"].update({"suite": "TRAIN_ONLY", "training_split": sample.training_split, "base_structure": sample.base_structure})
    if regenerated != json.loads((TARGET / sample.relative_path).read_text()):
        raise RuntimeError("deterministic training regeneration failed")
    checksums = "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in manifest.sort_values("relative_path").to_dict("records"))
    (MANIFESTS / "checksums.sha256").write_text(checksums)
    audit = {
        "schema": "rcias-cb1-train-audit-v1", "instance_count": 405,
        "split_counts": expected_splits, "all_81_cells_in_every_split": True,
        "all_loadable": True, "frozen_test_hash_overlap": False,
        "all_generation_seeds_unique": manifest.final_generation_seed.nunique() == 45,
        "all_phase6b_seeds_disjoint_from_frozen_sets": True,
        "RI_ordering_valid": True, "W_TI_ordering_valid": True, "F_TI_ordering_valid": True,
        "structural_metrics_present": sorted(required_metrics), "DAG_validity": "PASS_VIA_LOAD_INSTANCE",
        "factorial_independence": "exactly balanced Scale x CF x RI x TI design",
        "deterministic_regeneration": True,
    }
    path = ROOT / "outputs/phase6b/audit/train_distribution_audit.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify-only", action="store_true"); args = parser.parse_args()
    verify() if args.verify_only else generate()
    if args.verify_only: print("RCIAS_CB1_TRAIN_VERIFY = TRUE")


if __name__ == "__main__": main()
