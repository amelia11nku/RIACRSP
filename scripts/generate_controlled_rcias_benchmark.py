#!/usr/bin/env python3
"""Generate the native RCIAS-CB1 DEV, CORE, and paired sensitivity suites."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance_dict
from rcias_clgri.instances.controlled_generator import (
    acceptance_failures, configuration_entropy, generate_candidate, scale_sensitivity_variant,
)

TARGET = ROOT / "instances/controlled/RCIAS-CB1"
MANIFESTS = TARGET / "manifests"


def _accepted(instance_id, suite, scale, cf, base_seed, spec):
    history = []
    for attempt in range(1, int(spec["max_attempts"]) + 1):
        seed = base_seed * 1000 + attempt
        raw = generate_candidate(instance_id, suite, scale, cf, seed, spec)
        failures = acceptance_failures(raw, scale, cf, spec)
        history.append({"attempt": attempt, "seed": seed, "failures": failures})
        if not failures:
            return raw, attempt, seed, history
    raise RuntimeError(f"{instance_id} failed {spec['max_attempts']} attempts; last={history[-1]}")


def _row(raw, suite, scale, cf, replicate, base_seed, attempt, final_seed, spec, ri="BASE", ti="BASE", base_structure=""):
    metrics = benchmark_metrics(load_instance_dict(raw)); cf_rule = spec["cf_levels"][cf]
    return {
        "instance_id": raw["meta"]["instance_id"], "suite": suite, "scale": scale,
        "CF_level": cf, "RI_level": ri, "TI_level": ti, "replicate": replicate,
        "base_structure": base_structure, "base_seed": base_seed, "accepted_attempt": attempt,
        "final_seed": final_seed, "target_F_cap": cf_rule["cap_center"],
        "realized_F_cap": metrics["F_cap_mean"], "target_F_route": cf_rule["route_center"],
        "realized_F_route": metrics["F_route_mean"], "target_processing_CV": "0.20-0.45",
        "realized_processing_CV": metrics["processing_CV_mean"],
        "target_RI": spec["core_targets"]["ri_center"] if ri == "BASE" else {"RI1": .2, "RI2": .4, "RI3": .8}[ri],
        "realized_RI": metrics["RI"], "realized_W_TI": metrics["W_transport_intensity"],
        "realized_F_TI": metrics["F_transport_intensity"],
        "configuration_entropy": configuration_entropy(raw), "R_full_op": metrics["R_full_op"],
        "R_full_island": metrics["R_full_island"], "number_of_operations": metrics["number_of_operations"],
        "relative_path": "",
    }


def _write_manifest(name, rows):
    path = MANIFESTS / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-unfrozen", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/rcias_cb1_generation.json").read_text())
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        if not args.regenerate_unfrozen:
            raise RuntimeError("RCIAS-CB1 candidate files already exist; use the audit/verify path, not silent regeneration")
        if (ROOT / "outputs/phase5c/controlled_benchmark_audit/freeze_record.json").exists() or (MANIFESTS / "checksums.sha256").exists():
            raise RuntimeError("RCIAS-CB1 is frozen and cannot be regenerated")
    for directory in (TARGET / "dev", TARGET / "core", TARGET / "sensitivity", MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)
    all_rows, histories = [], {}
    scale_index = {"S": 1, "M": 2, "L": 3}; cf_index = {"CF1": 1, "CF2": 2, "CF3": 3}
    for suite, replicates, folder in (("DEV", 2, "dev"), ("CORE", 5, "core")):
        for scale in ("S", "M", "L"):
            for cf in ("CF1", "CF2", "CF3"):
                for replicate in range(1, replicates + 1):
                    instance_id = f"CB1_{suite}_{scale}_{cf}_R{replicate:02d}"
                    base_seed = spec["seed_ranges"][suite] + scale_index[scale] * 100 + cf_index[cf] * 10 + replicate
                    raw, attempt, final_seed, history = _accepted(instance_id, suite, scale, cf, base_seed, spec)
                    path = TARGET / folder / f"{instance_id}.json"; write_json(raw, path)
                    row = _row(raw, suite, scale, cf, f"R{replicate:02d}", base_seed, attempt, final_seed, spec)
                    row["relative_path"] = str(path.relative_to(TARGET)); all_rows.append(row); histories[instance_id] = history
                    print(f"accepted {instance_id} attempt={attempt}")
    for replicate in range(1, 6):
        base_name = f"SENS_R{replicate:02d}"
        base_seed = spec["seed_ranges"]["SENS"] + replicate
        raw, attempt, final_seed, history = _accepted(base_name, "SENS_BASE", "M", "CF2", base_seed, spec)
        histories[base_name] = history
        for ri in ("RI1", "RI2", "RI3"):
            for ti in ("TI1", "TI2", "TI3"):
                instance_id = f"CB1_SENS_M_CF2_{ri}_{ti}_R{replicate:02d}"
                variant = scale_sensitivity_variant(raw, instance_id, ri, ti, spec)
                path = TARGET / "sensitivity" / f"{instance_id}.json"; write_json(variant, path)
                row = _row(variant, "SENS", "M", "CF2", f"R{replicate:02d}", base_seed, attempt, final_seed, spec, ri, ti, base_name)
                row["relative_path"] = str(path.relative_to(TARGET)); all_rows.append(row)
        print(f"accepted {base_name} attempt={attempt} variants=9")
    dev = [row for row in all_rows if row["suite"] == "DEV"]
    core = [row for row in all_rows if row["suite"] == "CORE"]
    sens = [row for row in all_rows if row["suite"] == "SENS"]
    assert (len(dev), len(core), len(sens)) == (18, 45, 45)
    _write_manifest("dev_manifest.csv", dev); _write_manifest("core_manifest.csv", core)
    _write_manifest("sensitivity_manifest.csv", sens); _write_manifest("benchmark_manifest.csv", all_rows)
    (MANIFESTS / "generation_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (MANIFESTS / "generation_history.json").write_text(json.dumps(histories, indent=2, sort_keys=True) + "\n")
    (TARGET / "README.md").write_text("# RCIAS-CB1\n\nNative controlled RCIAS-2.0 benchmark. DEV is calibration-only; CORE and sensitivity are formal test sets. See manifests/generation_spec.json.\n")
    print("RCIAS_CB1_GENERATED DEV=18 CORE=45 SENS=45 TOTAL=108")


if __name__ == "__main__":
    main()
