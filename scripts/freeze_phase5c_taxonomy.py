#!/usr/bin/env python3
"""Create and freeze the Phase 5C taxonomy from structural metrics only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "phase5c" / "benchmark_audit"
METRICS = AUDIT / "instance_metrics.csv"
PROPOSAL = AUDIT / "taxonomy_proposal.csv"
RULES = AUDIT / "taxonomy_rules.json"
FREEZE = AUDIT / "taxonomy_freeze.json"

CORE_ROUTE_MAX = 0.30
STRESS_ROUTE_MIN = 0.425
STRESS_CAP_MIN = 1.0
STRESS_FULL_ISLAND_MIN = 1.0


def classify(row: pd.Series) -> tuple[str, str, str]:
    route = float(row["F_route_mean"])
    cap = float(row["F_cap_mean"])
    full_island = float(row["R_full_island"])
    if route < CORE_ROUTE_MAX:
        return (
            "CORE_CANDIDATE",
            f"F_route_mean={route:.6f} < {CORE_ROUTE_MAX:.3f}",
            "routing flexibility lies below the first empirical separation",
        )
    if (
        route >= STRESS_ROUTE_MIN
        and cap >= STRESS_CAP_MIN
        and full_island >= STRESS_FULL_ISLAND_MIN
    ):
        return (
            "EXTREME_FLEXIBILITY_STRESS",
            f"F_route_mean={route:.6f} >= {STRESS_ROUTE_MIN:.3f}",
            f"F_cap_mean={cap:.6f} and R_full_island={full_island:.6f} are both full",
        )
    return (
        "TRANSITIONAL",
        f"F_route_mean={route:.6f} lies between the core and stress regimes",
        f"F_cap_mean={cap:.6f}; R_full_island={full_island:.6f}",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frame = pd.read_csv(METRICS)
    assigned = frame.apply(classify, axis=1, result_type="expand")
    assigned.columns = ["classification", "primary_reason", "secondary_reason"]
    proposal = pd.concat(
        [frame[["instance_id", "family"]], assigned], axis=1
    ).sort_values("instance_id")
    proposal.to_csv(PROPOSAL, index=False)

    route = frame["F_route_mean"].astype(float)
    rules = {
        "version": "phase5c-taxonomy-v1",
        "performance_metrics_used": False,
        "inputs": ["F_route_mean", "F_cap_mean", "R_full_island"],
        "rules_in_order": [
            {
                "classification": "CORE_CANDIDATE",
                "condition": f"F_route_mean < {CORE_ROUTE_MAX}",
            },
            {
                "classification": "EXTREME_FLEXIBILITY_STRESS",
                "condition": (
                    f"F_route_mean >= {STRESS_ROUTE_MIN} and "
                    f"F_cap_mean >= {STRESS_CAP_MIN} and "
                    f"R_full_island >= {STRESS_FULL_ISLAND_MIN}"
                ),
            },
            {"classification": "TRANSITIONAL", "condition": "otherwise"},
        ],
        "threshold_justification": {
            "core_boundary": (
                "0.30 lies in the observed empty interval between 0.252500 and "
                "0.348485; it separates the lower-routing-flexibility regime."
            ),
            "stress_boundary": (
                "0.425 lies in the observed empty interval between 0.412000 and "
                "0.426887; full capability and full-island gates prevent routing "
                "flexibility alone from defining the stress regime."
            ),
        },
        "sensitivity": {
            "window": 0.025,
            "instances_within_core_boundary_window": int(
                ((route - CORE_ROUTE_MAX).abs() <= 0.025).sum()
            ),
            "instances_within_stress_boundary_window": int(
                ((route - STRESS_ROUTE_MIN).abs() <= 0.025).sum()
            ),
            "core_alternative_counts": {
                str(value): int((route < value).sum())
                for value in (0.275, 0.30, 0.325)
            },
            "route_stress_alternative_counts_before_gates": {
                str(value): int((route >= value).sum())
                for value in (0.40, 0.425, 0.45)
            },
        },
        "classification_counts": proposal["classification"].value_counts().to_dict(),
    }
    RULES.write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n")

    files = [METRICS, PROPOSAL, RULES]
    freeze = {
        "version": "phase5c-taxonomy-freeze-v1",
        "performance_metrics_used": False,
        "files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    canonical = json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()
    freeze["taxonomy_hash"] = hashlib.sha256(canonical).hexdigest()
    FREEZE.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(
        "PHASE5C_TAXONOMY_FROZEN",
        rules["classification_counts"],
        freeze["taxonomy_hash"],
    )


if __name__ == "__main__":
    main()
