#!/usr/bin/env python3
"""Evaluate frozen Phase 5A checkpoints without touching the canonical suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import (
    evaluate_baselines,
    evaluate_policy,
    load_checkpoint,
)
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_specs(selection: str) -> list[tuple[str, str, Path]]:
    phase5_name = "best_mean.pt" if selection == "mean" else "best_robust.pt"
    return [
        ("BC_GREEDY", "BC", ROOT / "outputs/phase4/bc_large/best.pt"),
        *[
            ("PHASE4_PPO", str(seed), ROOT / f"outputs/phase4/ppo_seed_{index}/best.pt")
            for index, seed in enumerate((410101, 410102, 410103), start=1)
        ],
        *[
            ("PHASE5A_PPO", str(seed), ROOT / f"outputs/phase5a/seed_{index}/{phase5_name}")
            for index, seed in enumerate((510101, 510102, 510103), start=1)
        ],
    ]


def _evaluate(instance, split: str, group: str, policies, device) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in evaluate_baselines(instance):
        if result.method not in {"H1", "H2"}:
            continue
        rows.append({
            **result.to_dict(),
            "data_split": split,
            "group": group,
            "training_seed": "NA",
        })
    for method, seed, model, tensorizer in policies:
        result = evaluate_policy(model, tensorizer, instance, device=device, method=method)
        rows.append({
            **result.to_dict(),
            "data_split": split,
            "group": group,
            "training_seed": seed,
        })
    bc_makespan = next(float(row["makespan"]) for row in rows if row["method"] == "BC_GREEDY")
    for row in rows:
        row["gap_to_bc_percent"] = 100.0 * (float(row["makespan"]) - bc_makespan) / bc_makespan
    return rows


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        for group in (str(row["group"]), "Overall"):
            key = (str(row["data_split"]), group, str(row["method"]))
            groups.setdefault(key, []).append(row)
    records = []
    for (split, group, method), values in sorted(groups.items()):
        makespans = [float(value["makespan"]) for value in values]
        gaps = [float(value["gap_to_bc_percent"]) for value in values]
        records.append({
            "data_split": split,
            "group": group,
            "method": method,
            "runs": len(values),
            "mean_makespan": mean(makespans),
            "std_makespan": pstdev(makespans) if len(makespans) > 1 else 0.0,
            "mean_gap_to_bc_percent": mean(gaps),
            "std_gap_to_bc_percent": pstdev(gaps) if len(gaps) > 1 else 0.0,
            "feasibility_rate": mean(float(bool(value["feasible"])) for value in values),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a_final.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection", choices=("mean", "robust"), default="mean")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5a/structural_evaluation"))
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    specs = _checkpoint_specs(args.selection)
    missing = [str(path) for _, _, path in specs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen evaluation checkpoints are missing: {missing}")

    policies = []
    frozen = []
    for method, seed, path in specs:
        model, tensorizer, metadata = load_checkpoint(path, device=device)
        policies.append((method, seed, model, tensorizer))
        frozen.append({
            "method": method,
            "training_seed": seed,
            "checkpoint": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "metadata": metadata,
        })

    factory = make_factory(ROOT, config)
    rows: list[dict[str, object]] = []
    for level, seeds in config["development_seeds"].items():
        for seed in seeds:
            rows.extend(_evaluate(factory.sample(int(seed), level), "development", level, policies, device))
    for level, seeds in config["historical_validation_seeds"].items():
        for seed in seeds:
            rows.extend(_evaluate(
                factory.sample(int(seed), level), "historical_validation", level, policies, device
            ))
    structural = config["structural_scenarios"]
    for scenario in structural["names"]:
        for seed in structural["seeds"]:
            rows.extend(_evaluate(
                factory.sample(int(seed), "M", scenario=scenario),
                "structural_generalization",
                scenario,
                policies,
                device,
            ))

    summary = _aggregate(rows)
    output_dir = ROOT / args.out_dir
    _write_csv(output_dir / f"{args.selection}_results.csv", rows)
    _write_csv(output_dir / f"{args.selection}_summary.csv", summary)
    write_json({
        "checkpoint_selection": args.selection,
        "canonical_instances": 0,
        "training_or_tuning_instances_from_canonical": 0,
        "checkpoints": frozen,
        "records": rows,
        "summary": summary,
    }, output_dir / f"{args.selection}_final_info.json")
    print(f"PHASE5A_SYNTHETIC_EVALUATION_COMPLETE rows={len(rows)} canonical_instances=0")


if __name__ == "__main__":
    main()
