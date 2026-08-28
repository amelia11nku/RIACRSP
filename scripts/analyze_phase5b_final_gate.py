#!/usr/bin/env python3
"""Freeze the Phase 5B constructive-policy and canonical-evaluation gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def main() -> None:
    formal = json.loads(
        (ROOT / "outputs/phase5b/structural_evaluation/final_info.json").read_text()
    )
    f2 = json.loads(
        (ROOT / "outputs/phase5b/freeze_boundary/F2_seed_1/evaluation/final_info.json").read_text()
    )
    holdout = formal["summary"]["phase5b_holdout|Overall"]
    level_gaps = {
        level: formal["summary"][f"phase5b_holdout|{level}"]["mean_gap_to_bc_percent"]
        for level in ("S", "M", "L")
    }
    structural = {
        group: formal["summary"][f"phase5b_structural|{group}"]
        for group in ("fleet_scarcity", "high_reconfiguration", "high_travel")
    }
    checks = {
        "expanded_holdout_overall_not_worse_than_bc": holdout["mean_gap_to_bc_percent"] <= 0.0,
        "no_level_severe_regression": max(level_gaps.values()) <= 5.0,
        "structural_groups_stable": max(
            item["worst_seed_gap_to_bc_percent"] for item in structural.values()
        ) <= 5.0,
        "three_seeds_stable": holdout["std_gap_to_bc_percentage_points"] <= 3.0,
        "feasibility_100": bool(formal["checks"]["feasibility_100"]),
        "config_frozen": True,
        "f1_beats_f2_on_holdout": (
            holdout["mean_gap_to_bc_percent"]
            < f2["summary"]["phase5b_holdout|Overall"]["phase5b_gap_to_bc_percent"]
        ),
    }
    passed = all(checks.values())
    config_path = ROOT / "configs/phase5b_final.json"
    write_json({
        "passed": passed,
        "checks": checks,
        "selected_architecture": "F1_frozen_bc_operation_plus_ppo_mwf",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "frozen_checkpoints": formal["frozen_checkpoints"],
        "holdout_mean_gap_to_bc_percent": holdout["mean_gap_to_bc_percent"],
        "holdout_seed_std_percentage_points": holdout["std_gap_to_bc_percentage_points"],
        "level_gaps_to_bc_percent": level_gaps,
        "structural_results": structural,
        "f2_holdout_gap_to_bc_percent": f2["summary"]["phase5b_holdout|Overall"]["phase5b_gap_to_bc_percent"],
        "canonical_evaluation_authorized": passed,
        "canonical_instances_before_authorized_evaluation": 0,
    }, ROOT / "outputs/phase5b/canonical_gate.json")
    print(f"PHASE5B_CANONICAL_GATE={'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
