#!/usr/bin/env python3
"""Recheck Phase 5A stagewise hybrids on Phase 5B evaluation sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import evaluate_policy, load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device
from rcias_clgri.learning.stagewise import collect_hybrid_episode


METRICS = (
    "bc_makespan",
    "ppo_makespan",
    "bc_o_ppo_mwf",
    "ppo_o_bc_mwf",
    "bc_om_ppo_wf",
    "ppo_om_bc_wf",
    "bc_omw_ppo_f",
    "ppo_omw_bc_f",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _specifications(config):
    specs = [
        ("phase5a_development", level, int(seed), level, None)
        for level, seeds in config["development_seeds"].items()
        for seed in seeds
    ]
    for scenario in config["structural_scenarios"]["names"]:
        specs.extend(
            ("phase5a_structural", "M", int(seed), scenario, scenario)
            for seed in config["structural_scenarios"]["seeds"]
        )
    specs.extend(
        ("phase5b_holdout", level, int(seed), level, None)
        for level, seeds in config["phase5b_holdout_seeds"].items()
        for seed in seeds
    )
    for scenario in config["phase5b_structural_scenarios"]["names"]:
        specs.extend(
            ("phase5b_structural", "M", int(seed), scenario, scenario)
            for seed in config["phase5b_structural_scenarios"]["seeds"]
        )
    return specs


def _aggregate(rows):
    result = {}
    keys = sorted({(row["split"], row["group"], row["ppo_seed"]) for row in rows})
    keys.extend(sorted({(row["split"], "Overall", row["ppo_seed"]) for row in rows}))
    for split, group, seed in keys:
        selected = [
            row for row in rows
            if row["split"] == split
            and str(row["ppo_seed"]) == str(seed)
            and (group == "Overall" or row["group"] == group)
        ]
        result[f"{split}|{group}|{seed}"] = {
            "instances": len(selected),
            **{metric: mean(float(row[metric]) for row in selected) for metric in METRICS},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5b_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/hybrid_diagnosis"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    bc_path = ROOT / config["teacher_checkpoint"]
    bc, tensorizer, _ = load_checkpoint(bc_path, device=device)
    checkpoints = [ROOT / path for path in config["phase5a_oracle_checkpoints"]]
    factory = make_factory(ROOT, config)
    specifications = _specifications(config)
    rows = []
    frozen = []
    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        ppo, ppo_tensorizer, metadata = load_checkpoint(checkpoint, device=device)
        if tensorizer.to_schema() != ppo_tensorizer.to_schema():
            raise RuntimeError("BC and PPO tensorizer schemas differ")
        seed = int(metadata.get("training_seed", 510100 + checkpoint_index))
        frozen.append({
            "training_seed": seed,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": _sha256(checkpoint),
        })
        for instance_index, (split, level, instance_seed, group, scenario) in enumerate(specifications, start=1):
            instance = factory.sample(instance_seed, level, scenario=scenario) if scenario else factory.sample(instance_seed, level)
            bc_result = evaluate_policy(bc, tensorizer, instance, device=device, method="BC")
            ppo_result = evaluate_policy(ppo, tensorizer, instance, device=device, method="PPO")
            hybrids = {
                "bc_o_ppo_mwf": collect_hybrid_episode(bc, ppo, tensorizer, instance, prefix_stages=1, device=device),
                "ppo_o_bc_mwf": collect_hybrid_episode(ppo, bc, tensorizer, instance, prefix_stages=1, device=device),
                "bc_om_ppo_wf": collect_hybrid_episode(bc, ppo, tensorizer, instance, prefix_stages=2, device=device),
                "ppo_om_bc_wf": collect_hybrid_episode(ppo, bc, tensorizer, instance, prefix_stages=2, device=device),
                "bc_omw_ppo_f": collect_hybrid_episode(bc, ppo, tensorizer, instance, prefix_stages=3, device=device),
                "ppo_omw_bc_f": collect_hybrid_episode(ppo, bc, tensorizer, instance, prefix_stages=3, device=device),
            }
            rows.append({
                "instance": instance.instance_id,
                "instance_seed": instance_seed,
                "level": level,
                "split": split,
                "group": group,
                "ppo_seed": seed,
                "bc_makespan": bc_result.makespan,
                "ppo_makespan": ppo_result.makespan,
                **{name: result.makespan for name, result in hybrids.items()},
                "all_feasible": bc_result.feasible and ppo_result.feasible and all(result.feasible for result in hybrids.values()),
            })
            if instance_index % 10 == 0 or instance_index == len(specifications):
                print(f"checkpoint={checkpoint_index}/3 instances={instance_index}/{len(specifications)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "hybrid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    phase5a_rows = [row for row in rows if row["split"].startswith("phase5a_")]
    phase5a_dir = ROOT / "outputs/phase5b/phase5a_oracle_recheck"
    phase5a_dir.mkdir(parents=True, exist_ok=True)
    write_json({
        "canonical_instances": 0,
        "checkpoints": frozen,
        "all_feasible": all(row["all_feasible"] for row in phase5a_rows),
        "records": phase5a_rows,
        "summary": _aggregate(phase5a_rows),
    }, phase5a_dir / "final_info.json")
    write_json({
        "canonical_instances": 0,
        "checkpoints": frozen,
        "instance_count_per_checkpoint": len(specifications),
        "all_feasible": all(row["all_feasible"] for row in rows),
        "records": rows,
        "summary": _aggregate(rows),
    }, args.out_dir / "final_info.json")
    print(f"PHASE5B_HYBRID_DIAGNOSIS_COMPLETE rows={len(rows)} canonical=0")


if __name__ == "__main__":
    main()
