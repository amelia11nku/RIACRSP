#!/usr/bin/env python3
"""Evaluate three frozen Phase 5B downstream PPO seeds and apply the formal gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev
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
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b_final.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/structural_evaluation"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    checkpoint_paths = [
        ROOT / f"outputs/phase5b/downstream_seed_{index}/selected_best.pt"
        for index in (1, 2, 3)
    ]
    policies = []
    frozen = []
    for path in checkpoint_paths:
        model, tensorizer, metadata = load_operation_anchored_checkpoint(path, device=device)
        policies.append((int(metadata["training_seed"]), model, tensorizer))
        frozen.append({
            "training_seed": int(metadata["training_seed"]),
            "checkpoint": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "best_update": int(metadata["update"]),
        })
    factory = make_factory(ROOT, config)
    reference_rows = json.loads(
        (ROOT / "outputs/phase5b/hybrid_diagnosis/final_info.json").read_text()
    )["records"]
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
    rows = []
    for index, (split, level, instance_seed, group, scenario) in enumerate(specifications, start=1):
        instance = factory.sample(instance_seed, level, scenario=scenario) if scenario else factory.sample(instance_seed, level)
        references = [
            row for row in reference_rows
            if row["split"] == split and int(row["instance_seed"]) == instance_seed and row["group"] == group
        ]
        if len(references) != 3:
            raise RuntimeError("expected three Phase 5A references per instance")
        bc = float(references[0]["bc_makespan"])
        for training_seed, model, tensorizer in policies:
            result = evaluate_policy(model, tensorizer, instance, device=device, method="PHASE5B_DOWNSTREAM_PPO")
            rows.append({
                "instance": instance.instance_id,
                "instance_seed": instance_seed,
                "split": split,
                "group": group,
                "level": level,
                "training_seed": training_seed,
                "bc_makespan": bc,
                "phase5a_ppo_mean_makespan": mean(float(row["ppo_makespan"]) for row in references),
                "frozen_o_hybrid_mean_makespan": mean(float(row["bc_o_ppo_mwf"]) for row in references),
                "phase5b_makespan": result.makespan,
                "phase5b_gap_to_bc_percent": _gap(result.makespan, bc),
                "feasible": result.feasible,
            })
        if index % 10 == 0 or index == len(specifications):
            print(f"formal_evaluated={index}/{len(specifications)}")
    summary = {}
    for split in ("phase5b_holdout", "phase5b_structural"):
        groups = sorted({row["group"] for row in rows if row["split"] == split}) + ["Overall"]
        for group in groups:
            selected = [row for row in rows if row["split"] == split and (group == "Overall" or row["group"] == group)]
            seed_gaps = {
                str(training_seed): mean(
                    float(row["phase5b_gap_to_bc_percent"])
                    for row in selected if int(row["training_seed"]) == training_seed
                )
                for training_seed, _, _ in policies
            }
            values = list(seed_gaps.values())
            summary[f"{split}|{group}"] = {
                "instance_seed_runs": len(selected),
                "seed_gaps_to_bc_percent": seed_gaps,
                "mean_gap_to_bc_percent": mean(values),
                "std_gap_to_bc_percentage_points": pstdev(values),
                "median_gap_to_bc_percent": median(values),
                "best_seed_gap_to_bc_percent": min(values),
                "worst_seed_gap_to_bc_percent": max(values),
                "feasibility_rate": mean(float(row["feasible"]) for row in selected),
            }
    holdout = summary["phase5b_holdout|Overall"]
    checks = {
        "mean_downstream_ppo_not_worse_than_bc": holdout["mean_gap_to_bc_percent"] <= 0.0,
        "seed_std_at_most_three_percentage_points": holdout["std_gap_to_bc_percentage_points"] <= 3.0,
        "feasibility_100": all(row["feasible"] for row in rows),
    }
    passed = all(checks.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "three_seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json({
        "passed": passed,
        "checks": checks,
        "frozen_checkpoints": frozen,
        "canonical_instances": 0,
        "summary": summary,
        "records": rows,
    }, args.out_dir / "final_info.json")
    print(f"PHASE5B_THREE_SEED_GATE={'PASS' if passed else 'FAIL'} canonical=0")


if __name__ == "__main__":
    main()
