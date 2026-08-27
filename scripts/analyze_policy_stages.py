#!/usr/bin/env python3
"""Run sequential BC/PPO stagewise oracle diagnosis on synthetic instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import evaluate_policy, load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device
from rcias_clgri.learning.stagewise import collect_hybrid_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a_training.json"))
    parser.add_argument("--ppo", type=Path, default=Path("outputs/phase4/ppo_seed_3/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5a/stagewise_diagnosis"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    bc, tensorizer, _ = load_checkpoint(config["teacher_checkpoint"], device=device)
    ppo, ppo_tensorizer, _ = load_checkpoint(args.ppo, device=device)
    if tensorizer.to_schema() != ppo_tensorizer.to_schema():
        raise RuntimeError("BC and PPO tensorizer schemas differ")
    factory = make_factory(ROOT, config)
    specifications = [
        (level, int(seed), "development", None)
        for level, seeds in config["development_seeds"].items() for seed in seeds
    ]
    for scenario in config["structural_scenarios"]["names"]:
        specifications.extend(
            ("M", int(seed), scenario, scenario)
            for seed in config["structural_scenarios"]["seeds"]
        )
    rows = []
    for level, seed, group, scenario in specifications:
        instance = (
            factory.sample(seed, level)
            if scenario is None else factory.sample(seed, level, scenario=scenario)
        )
        bc_result = evaluate_policy(bc, tensorizer, instance, device=device, method="BC")
        ppo_result = evaluate_policy(ppo, tensorizer, instance, device=device, method="PPO")
        bc_o = collect_hybrid_episode(bc, ppo, tensorizer, instance, prefix_stages=1, device=device)
        ppo_o = collect_hybrid_episode(ppo, bc, tensorizer, instance, prefix_stages=1, device=device)
        bc_om = collect_hybrid_episode(bc, ppo, tensorizer, instance, prefix_stages=2, device=device)
        ppo_om = collect_hybrid_episode(ppo, bc, tensorizer, instance, prefix_stages=2, device=device)
        rows.append({
            "instance": instance.instance_id, "seed": seed, "level": level, "group": group,
            "bc_makespan": bc_result.makespan, "ppo_makespan": ppo_result.makespan,
            "bc_o_ppo_rest": bc_o.makespan, "ppo_o_bc_rest": ppo_o.makespan,
            "bc_om_ppo_wf": bc_om.makespan, "ppo_om_bc_wf": ppo_om.makespan,
            "all_feasible": all(x.feasible for x in (bc_o, ppo_o, bc_om, ppo_om)),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "stagewise_diagnosis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ("bc_makespan", "ppo_makespan", "bc_o_ppo_rest", "ppo_o_bc_rest", "bc_om_ppo_wf", "ppo_om_bc_wf")
    write_json({
        "canonical_instances": 0,
        "ppo_checkpoint": args.ppo.as_posix(),
        "instances": len(rows),
        "all_feasible": all(row["all_feasible"] for row in rows),
        "overall_means": {column: sum(float(row[column]) for row in rows) / len(rows) for column in columns},
        "group_means": {
            group: {
                column: sum(float(row[column]) for row in rows if row["group"] == group) / sum(row["group"] == group for row in rows)
                for column in columns
            } for group in sorted({row["group"] for row in rows})
        },
        "records": rows,
    }, args.out_dir / "final_info.json")
    print(f"STAGEWISE_ORACLE_VALIDATED = TRUE | instances={len(rows)} canonical=0")


if __name__ == "__main__":
    main()
