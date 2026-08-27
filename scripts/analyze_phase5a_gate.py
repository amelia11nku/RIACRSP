#!/usr/bin/env python3
"""Apply the documented Phase 5A synthetic gate to frozen evaluation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_gap(rows, *, split: str, method: str, group: str | None = None, seed=None) -> float:
    values = [
        float(row["gap_to_bc_percent"])
        for row in rows
        if row["data_split"] == split
        and row["method"] == method
        and (group is None or row["group"] == group)
        and (seed is None or str(row["training_seed"]) == str(seed))
    ]
    return mean(values)


def main() -> None:
    evaluation_path = ROOT / "outputs/phase5a/structural_evaluation/mean_final_info.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows = evaluation["records"]

    phase4_seed_gaps = [
        _mean_gap(rows, split="development", method="PHASE4_PPO", seed=seed)
        for seed in (410101, 410102, 410103)
    ]
    phase5_seed_gaps = [
        _mean_gap(rows, split="development", method="PHASE5A_PPO", seed=seed)
        for seed in (510101, 510102, 510103)
    ]
    level_gaps = {
        split: {
            level: _mean_gap(
                rows, split=split, group=level, method="PHASE5A_PPO"
            )
            for level in ("S", "M", "L")
        }
        for split in ("development", "historical_validation")
    }
    structural_gaps = {
        group: _mean_gap(
            rows,
            split="structural_generalization",
            group=group,
            method="PHASE5A_PPO",
        )
        for group in ("fleet_scarcity", "high_reconfiguration", "high_travel")
    }

    trajectories = {}
    late_collapse = False
    checkpoints = []
    for index, seed in enumerate((510101, 510102, 510103), start=1):
        directory = ROOT / f"outputs/phase5a/seed_{index}"
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        best = float(metrics["best_validation_normalized_makespan"])
        final = float(metrics["final_validation"]["mean_normalized_makespan"])
        regression = 100.0 * (final - best) / best
        collapsed = regression > 10.0
        late_collapse = late_collapse or collapsed
        trajectories[str(seed)] = {
            "best_update": metrics["best_update"],
            "best_normalized_makespan": best,
            "final_normalized_makespan": final,
            "final_regression_from_best_percent": regression,
            "late_catastrophic_collapse": collapsed,
            "feasibility_100": metrics["feasibility_100"],
        }
        checkpoint = directory / "best_mean.pt"
        checkpoints.append({
            "training_seed": seed,
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": _sha256(checkpoint),
        })

    thresholds = {
        "development_overall_gap_to_bc_percent_max": 0.0,
        "per_level_gap_to_bc_percent_severe": 10.0,
        "structural_group_gap_to_bc_percent_severe": 10.0,
        "final_regression_from_best_percent_late_collapse": 10.0,
        "seed_gap_std_must_be_below_phase4": True,
    }
    development_gap = _mean_gap(rows, split="development", method="PHASE5A_PPO")
    seed_std_phase4 = pstdev(phase4_seed_gaps)
    seed_std_phase5 = pstdev(phase5_seed_gaps)
    all_feasible = all(item["feasibility_100"] for item in trajectories.values())
    per_level_ok = max(value for groups in level_gaps.values() for value in groups.values()) <= 10.0
    structural_ok = max(structural_gaps["high_reconfiguration"], structural_gaps["high_travel"]) <= 10.0
    checks = {
        "ppo_not_worse_than_bc": development_gap <= 0.0,
        "all_seeds_feasible": all_feasible,
        "seed_variance_below_phase4": seed_std_phase5 < seed_std_phase4,
        "no_late_catastrophic_collapse": not late_collapse,
        "no_severe_per_level_collapse": per_level_ok,
        "high_reconfiguration_and_travel_no_severe_regression": structural_ok,
    }
    passed = all(checks.values())
    write_json({
        "gate_name": "phase5a_canonical_gate",
        "passed": passed,
        "checks": checks,
        "thresholds": thresholds,
        "development_gap_to_bc_percent": development_gap,
        "development_seed_gaps_to_bc_percent": phase5_seed_gaps,
        "phase4_development_seed_gaps_to_bc_percent": phase4_seed_gaps,
        "development_seed_gap_std_phase5a": seed_std_phase5,
        "development_seed_gap_std_phase4": seed_std_phase4,
        "per_level_gaps_to_bc_percent": level_gaps,
        "structural_gaps_to_bc_percent": structural_gaps,
        "training_trajectories": trajectories,
        "frozen_checkpoints": checkpoints,
        "canonical_instances": 130 if passed else 0,
        "canonical_evaluation_authorized": passed,
    }, ROOT / "outputs/phase5a/canonical_gate.json")
    print(f"PHASE5A_CANONICAL_GATE={'PASS' if passed else 'FAIL'} canonical_instances={130 if passed else 0}")


if __name__ == "__main__":
    main()
