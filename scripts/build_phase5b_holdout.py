#!/usr/bin/env python3
"""Freeze the independent Phase 5B holdout and structural stress references."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import evaluate_baselines, evaluate_policy, load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seed_boundaries(config: dict[str, object]) -> None:
    holdout = {int(seed) for seeds in config["phase5b_holdout_seeds"].values() for seed in seeds}
    structural = {int(seed) for seed in config["phase5b_structural_scenarios"]["seeds"]}
    excluded_groups = [
        config["development_seeds"],
        config["historical_validation_seeds"],
    ]
    excluded = {
        int(seed)
        for group in excluded_groups
        for seeds in group.values()
        for seed in seeds
    }
    excluded.update(int(seed) for seed in config["structural_scenarios"]["seeds"])
    excluded.update(int(seed) for seed in config["training_seed_policy"]["independent_training_seeds"])
    if len(holdout) != 60:
        raise ValueError("Phase 5B holdout must contain 60 unique seeds")
    if len(structural) != 10:
        raise ValueError("Phase 5B structural set must contain ten shared scenario seeds")
    if holdout & structural or holdout & excluded or structural & excluded:
        raise ValueError("Phase 5B evaluation seeds overlap an earlier or training seed set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/expanded_holdout"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    validate_seed_boundaries(config)
    device = resolve_device(args.device)
    checkpoint = ROOT / config["teacher_checkpoint"]
    bc_model, tensorizer, metadata = load_checkpoint(checkpoint, device=device)
    factory = make_factory(ROOT, config)
    specifications = [
        ("expanded_holdout", level, int(seed), None)
        for level, seeds in config["phase5b_holdout_seeds"].items()
        for seed in seeds
    ]
    for scenario in config["phase5b_structural_scenarios"]["names"]:
        specifications.extend(
            ("structural_stress", "M", int(seed), scenario)
            for seed in config["phase5b_structural_scenarios"]["seeds"]
        )
    rows = []
    for split, level, seed, scenario in specifications:
        instance = factory.sample(seed, level, scenario=scenario) if scenario else factory.sample(seed, level)
        baselines = {result.method: result for result in evaluate_baselines(instance)}
        bc = evaluate_policy(bc_model, tensorizer, instance, device=device, method="BC_GREEDY")
        rows.append({
            "instance": instance.instance_id,
            "split": split,
            "group": scenario or level,
            "level": level,
            "seed": seed,
            "h1_makespan": baselines["H1"].makespan,
            "h2_makespan": baselines["H2"].makespan,
            "bc_makespan": bc.makespan,
            "bc_feasible": bc.feasible,
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json({
        "instance_count": len(rows),
        "expanded_holdout_instances": 60,
        "structural_stress_instances": 30,
        "canonical_instances": 0,
        "seed_boundaries_validated": True,
        "bc_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "bc_checkpoint_sha256": _sha256(checkpoint),
        "bc_checkpoint_metadata": metadata,
        "all_bc_feasible": all(row["bc_feasible"] for row in rows),
        "records": rows,
    }, args.out_dir / "final_info.json")
    print("PHASE5B_HOLDOUT_FROZEN = TRUE | holdout=60 structural=30 canonical=0")


if __name__ == "__main__":
    main()
