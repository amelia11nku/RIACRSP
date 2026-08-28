#!/usr/bin/env python3
"""Apply the Phase 5B operation-drift gate to frozen hybrid results."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json


def _gap(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def main() -> None:
    source = ROOT / "outputs/phase5b/hybrid_diagnosis/final_info.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["records"]
    holdout = [row for row in rows if row["split"] == "phase5b_holdout"]
    seeds = sorted({int(row["ppo_seed"]) for row in holdout})
    seed_results = {}
    hybrid_beats_full_count = 0
    for seed in seeds:
        selected = [row for row in holdout if int(row["ppo_seed"]) == seed]
        full_gap = mean(_gap(float(row["ppo_makespan"]), float(row["bc_makespan"])) for row in selected)
        hybrid_gap = mean(_gap(float(row["bc_o_ppo_mwf"]), float(row["bc_makespan"])) for row in selected)
        hybrid_beats_full = hybrid_gap < full_gap
        hybrid_beats_full_count += int(hybrid_beats_full)
        seed_results[str(seed)] = {
            "full_ppo_gap_to_bc_percent": full_gap,
            "bc_o_ppo_mwf_gap_to_bc_percent": hybrid_gap,
            "hybrid_beats_full_ppo": hybrid_beats_full,
        }
    aggregate_holdout_gap = mean(
        _gap(float(row["bc_o_ppo_mwf"]), float(row["bc_makespan"])) for row in holdout
    )
    structural_gaps = {}
    for group in ("high_reconfiguration", "high_travel", "fleet_scarcity"):
        selected = [
            row for row in rows
            if row["split"] == "phase5b_structural" and row["group"] == group
        ]
        structural_gaps[group] = mean(
            _gap(float(row["bc_o_ppo_mwf"]), float(row["bc_makespan"])) for row in selected
        )
    checks = {
        "majority_of_phase5a_seeds_hybrid_beats_full_ppo": hybrid_beats_full_count >= 2,
        "aggregate_holdout_hybrid_not_worse_than_bc": aggregate_holdout_gap <= 0.0,
        "high_reconfiguration_no_severe_regression": structural_gaps["high_reconfiguration"] <= 5.0,
        "high_travel_no_severe_regression": structural_gaps["high_travel"] <= 5.0,
        "all_hybrids_feasible": bool(data["all_feasible"]),
    }
    confirmed = all(checks.values())
    write_json({
        "operation_drift_confirmed": confirmed,
        "checks": checks,
        "thresholds": {
            "majority_seed_count": 2,
            "holdout_gap_to_bc_percent_max": 0.0,
            "structural_severe_regression_percent": 5.0,
        },
        "seed_results": seed_results,
        "aggregate_holdout_hybrid_gap_to_bc_percent": aggregate_holdout_gap,
        "structural_hybrid_gaps_to_bc_percent": structural_gaps,
        "canonical_instances": 0,
    }, ROOT / "outputs/phase5b/operation_drift_gate.json")
    print(f"OPERATION_DRIFT_CONFIRMED = {str(confirmed).upper()} | canonical=0")


if __name__ == "__main__":
    main()
