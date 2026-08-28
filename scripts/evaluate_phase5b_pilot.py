#!/usr/bin/env python3
"""Evaluate the frozen Phase 5B downstream pilot and apply its holdout gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import evaluate_policy, load_operation_anchored_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gap(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b_training.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase5b/downstream_pilot/selected_best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/downstream_pilot/evaluation"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    model, tensorizer, metadata = load_operation_anchored_checkpoint(args.checkpoint, device=device)
    factory = make_factory(ROOT, config)
    hybrid = json.loads((ROOT / "outputs/phase5b/hybrid_diagnosis/final_info.json").read_text())["records"]
    rows = []
    specifications = [
        ("phase5b_holdout", level, int(seed), level, None)
        for level, seeds in config["phase5b_holdout_seeds"].items()
        for seed in seeds
    ]
    for scenario in config["phase5b_structural_scenarios"]["names"]:
        specifications.extend(
            ("phase5b_structural", "M", int(seed), scenario, scenario)
            for seed in config["phase5b_structural_scenarios"]["seeds"]
        )
    for index, (split, level, seed, group, scenario) in enumerate(specifications, start=1):
        instance = factory.sample(seed, level, scenario=scenario) if scenario else factory.sample(seed, level)
        result = evaluate_policy(model, tensorizer, instance, device=device, method="PHASE5B_DOWNSTREAM_PPO")
        references = [
            row for row in hybrid
            if row["split"] == split and int(row["instance_seed"]) == seed and row["group"] == group
        ]
        if len(references) != 3:
            raise RuntimeError("expected three Phase 5A reference rows per evaluation instance")
        bc = float(references[0]["bc_makespan"])
        rows.append({
            "instance": instance.instance_id,
            "split": split,
            "group": group,
            "level": level,
            "seed": seed,
            "bc_makespan": bc,
            "phase5a_ppo_mean_makespan": mean(float(row["ppo_makespan"]) for row in references),
            "frozen_o_hybrid_mean_makespan": mean(float(row["bc_o_ppo_mwf"]) for row in references),
            "phase5b_downstream_makespan": result.makespan,
            "phase5b_gap_to_bc_percent": _gap(result.makespan, bc),
            "feasible": result.feasible,
        })
        if index % 10 == 0 or index == len(specifications):
            print(f"pilot_evaluated={index}/{len(specifications)}")
    summary = {}
    for split in ("phase5b_holdout", "phase5b_structural"):
        groups = sorted({row["group"] for row in rows if row["split"] == split})
        for group in groups + ["Overall"]:
            selected = [row for row in rows if row["split"] == split and (group == "Overall" or row["group"] == group)]
            summary[f"{split}|{group}"] = {
                "instances": len(selected),
                "bc_mean_makespan": mean(float(row["bc_makespan"]) for row in selected),
                "phase5a_ppo_mean_makespan": mean(float(row["phase5a_ppo_mean_makespan"]) for row in selected),
                "frozen_o_hybrid_mean_makespan": mean(float(row["frozen_o_hybrid_mean_makespan"]) for row in selected),
                "phase5b_downstream_mean_makespan": mean(float(row["phase5b_downstream_makespan"]) for row in selected),
                "phase5b_gap_to_bc_percent": mean(float(row["phase5b_gap_to_bc_percent"]) for row in selected),
                "feasibility_rate": mean(float(row["feasible"]) for row in selected),
            }
    thresholds = config["downstream_pilot_gate"]
    checks = {
        "holdout_not_worse_than_bc": summary["phase5b_holdout|Overall"]["phase5b_gap_to_bc_percent"] <= float(thresholds["holdout_gap_to_bc_percent_max"]),
        **{
            f"level_{level}_within_limit": summary[f"phase5b_holdout|{level}"]["phase5b_gap_to_bc_percent"] <= float(thresholds["per_level_gap_to_bc_percent_max"])
            for level in ("S", "M", "L")
        },
        "high_travel_within_limit": summary["phase5b_structural|high_travel"]["phase5b_gap_to_bc_percent"] <= float(thresholds["high_travel_gap_to_bc_percent_max"]),
        "high_reconfiguration_within_limit": summary["phase5b_structural|high_reconfiguration"]["phase5b_gap_to_bc_percent"] <= float(thresholds["high_reconfiguration_gap_to_bc_percent_max"]),
        "feasibility_100": all(row["feasible"] for row in rows),
    }
    passed = all(checks.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json({
        "passed": passed,
        "checks": checks,
        "thresholds": thresholds,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_metadata": metadata,
        "canonical_instances": 0,
        "summary": summary,
        "records": rows,
    }, args.out_dir / "final_info.json")
    print(f"PHASE5B_DOWNSTREAM_PILOT_GATE={'PASS' if passed else 'FAIL'} canonical=0")


if __name__ == "__main__":
    main()
