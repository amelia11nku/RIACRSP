#!/usr/bin/env python3
"""Build descriptive Core45 heterogeneity, stability, and budget diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
MAIN_ROOT = PAPER_ROOT / "processed_data/main"
OUTPUT_ROOT = PAPER_ROOT / "processed_data/supplementary"
CSG = "CSG-NI Phase6H provisional"
METHODS = (
    "GA",
    "Adapted DCGA",
    "DABC-RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M",
    CSG,
)
DISPLAY = {
    "GA": "GA",
    "Adapted DCGA": "DCGA",
    "DABC-RIACRSP": "DABC",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA",
    CSG: "CSG-NI",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main() -> int:
    manifest = json.loads(
        (MAIN_ROOT / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H":
        raise RuntimeError("complete Core45 analysis gate is not PASS")

    runs = read_csv(MAIN_ROOT / "main_runs.csv")
    instances = read_csv(MAIN_ROOT / "main_instance_summary.csv")
    if len(runs) != 1125 or len(instances) != 225:
        raise RuntimeError("expected 1,125 runs and 225 method-instance summaries")
    if {row["method"] for row in runs} != set(METHODS):
        raise RuntimeError("unexpected Core45 method identity")
    if any(row["feasible"] != "True" for row in runs):
        raise RuntimeError("supplementary analysis requires all schedules to be feasible")

    by_instance_method = {
        (row["instance_id"], row["method"]): row for row in instances
    }
    instance_ids = sorted({row["instance_id"] for row in instances})
    if len(by_instance_method) != 225 or len(instance_ids) != 45:
        raise RuntimeError("Core45 method-instance matrix is not unique and complete")

    paired_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for instance_id in instance_ids:
        csg_row = by_instance_method[(instance_id, CSG)]
        csg_rpd = float(csg_row["median_rpd_percent"])
        baselines = []
        for method in METHODS[:-1]:
            baseline = by_instance_method[(instance_id, method)]
            baseline_rpd = float(baseline["median_rpd_percent"])
            advantage = baseline_rpd - csg_rpd
            paired_rows.append({
                "instance_id": instance_id,
                "scale": csg_row["scale"],
                "CF_level": csg_row["CF_level"],
                "operation_count": int(csg_row["operation_count"]),
                "competitor": method,
                "display_competitor": DISPLAY[method],
                "competitor_median_rpd_percent": baseline_rpd,
                "csgni_median_rpd_percent": csg_rpd,
                "advantage_percent_points_positive_favors_csgni": advantage,
                "outcome": "win" if advantage > 0 else "loss" if advantage < 0 else "tie",
            })
            baselines.append((baseline_rpd, method))
        best_rpd, best_method = min(baselines)
        best_rows.append({
            "instance_id": instance_id,
            "scale": csg_row["scale"],
            "CF_level": csg_row["CF_level"],
            "operation_count": int(csg_row["operation_count"]),
            "best_baseline": best_method,
            "display_best_baseline": DISPLAY[best_method],
            "best_baseline_median_rpd_percent": best_rpd,
            "csgni_median_rpd_percent": csg_rpd,
            "advantage_percent_points_positive_favors_csgni": best_rpd - csg_rpd,
        })

    grouped_advantage: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    grouped_outcomes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in paired_rows:
        key = (str(row["scale"]), str(row["CF_level"]), str(row["competitor"]))
        grouped_advantage[key].append(
            float(row["advantage_percent_points_positive_favors_csgni"])
        )
        grouped_outcomes[key].append(str(row["outcome"]))
    advantage_summary = []
    for scale in ("S", "M", "L"):
        for competitor in METHODS[:-1]:
            for cf_level in ("CF1", "CF2", "CF3"):
                key = (scale, cf_level, competitor)
                values = grouped_advantage[key]
                outcomes = grouped_outcomes[key]
                if len(values) != 5:
                    raise RuntimeError(f"expected five instances for {key}, got {len(values)}")
                advantage_summary.append({
                    "scale": scale,
                    "competitor": competitor,
                    "display_competitor": DISPLAY[competitor],
                    "CF_level": cf_level,
                    "instance_count": len(values),
                    "median_advantage_percent_points": float(np.median(values)),
                    "mean_advantage_percent_points": float(np.mean(values)),
                    "q25_advantage_percent_points": percentile(values, 25),
                    "q75_advantage_percent_points": percentile(values, 75),
                    "wins": outcomes.count("win"),
                    "ties": outcomes.count("tie"),
                    "losses": outcomes.count("loss"),
                })

    run_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        run_groups[(row["instance_id"], row["method"])].append(row)
    variability_rows: list[dict[str, object]] = []
    for (instance_id, method), group in sorted(run_groups.items()):
        if len(group) != 5 or len({row["seed"] for row in group}) != 5:
            raise RuntimeError(f"expected five unique seeds for {(instance_id, method)}")
        rpd = [float(row["rpd_percent"]) for row in group]
        variability_rows.append({
            "instance_id": instance_id,
            "scale": group[0]["scale"],
            "CF_level": group[0]["CF_level"],
            "method": method,
            "display_method": DISPLAY[method],
            "seed_count": len(group),
            "seed_rpd_sd_percent_points": float(np.std(rpd, ddof=1)),
            "seed_rpd_iqr_percent_points": percentile(rpd, 75) - percentile(rpd, 25),
        })
    variability_summary = []
    for scale in ("S", "M", "L"):
        for method in METHODS:
            values = [
                float(row["seed_rpd_sd_percent_points"])
                for row in variability_rows
                if row["scale"] == scale and row["method"] == method
            ]
            if len(values) != 15:
                raise RuntimeError(f"expected 15 variability values for {(scale, method)}")
            variability_summary.append({
                "scale": scale,
                "method": method,
                "display_method": DISPLAY[method],
                "instance_count": len(values),
                "median_seed_rpd_sd_percent_points": float(np.median(values)),
                "mean_seed_rpd_sd_percent_points": float(np.mean(values)),
                "q25_seed_rpd_sd_percent_points": percentile(values, 25),
                "q75_seed_rpd_sd_percent_points": percentile(values, 75),
            })

    runtime_rows = []
    for method in METHODS:
        group = [row for row in runs if row["method"] == method]
        ratios = [float(row["runtime_seconds"]) / float(row["time_limit_seconds"]) for row in group]
        evaluations = [float(row["decoder_evaluations"]) for row in group]
        time_to_best = [
            float(row["best_found_time_seconds"]) / float(row["time_limit_seconds"])
            for row in group
        ]
        iterations = [float(row["iterations"]) for row in group]
        if len(group) != 225 or not all(math.isfinite(value) for value in ratios):
            raise RuntimeError(f"invalid runtime evidence for {method}")
        runtime_rows.append({
            "method": method,
            "display_method": DISPLAY[method],
            "run_count": len(group),
            "median_runtime_budget_fraction": float(np.median(ratios)),
            "mean_runtime_budget_fraction": float(np.mean(ratios)),
            "q25_runtime_budget_fraction": percentile(ratios, 25),
            "q75_runtime_budget_fraction": percentile(ratios, 75),
            "median_decoder_evaluations": float(np.median(evaluations)),
            "mean_decoder_evaluations": float(np.mean(evaluations)),
            "median_time_to_best_budget_fraction": float(np.median(time_to_best)),
            "median_iterations": float(np.median(iterations)),
            "termination_note": (
                "original-paper MAXGEN=100 cap or common maximum wall-clock budget"
                if method == "LG_HGA-RIACRSP-v2-N4M"
                else "common maximum wall-clock budget"
            ),
        })

    operation_counts = np.asarray([int(row["operation_count"]) for row in best_rows])
    best_advantages = np.asarray([
        float(row["advantage_percent_points_positive_favors_csgni"])
        for row in best_rows
    ])
    rho, p_value = spearmanr(operation_counts, best_advantages)
    analysis = {
        "schema": "initial-manuscript-core-supplementary-analysis-v1",
        "status": "PASS_DESCRIPTIVE_EXPLORATORY",
        "source": "paper_experiments/processed_data/main/main_runs.csv",
        "runs": len(runs),
        "independent_instances": len(instance_ids),
        "matched_seeds_per_method_instance": 5,
        "definitions": {
            "paired_advantage": "competitor median RPD minus CSG-NI median RPD; positive favors CSG-NI",
            "seed_variability": "sample standard deviation of RPD across five matched seeds within each method-instance cell",
            "runtime_budget_fraction": "observed runtime divided by the preregistered maximum wall-clock budget",
        },
        "exploratory_association": {
            "variables": "operation count and CSG-NI advantage over the best baseline selected separately on each instance",
            "spearman_rho": float(rho),
            "two_sided_p_value_unadjusted": float(p_value),
            "n_instances": len(best_rows),
            "interpretation_boundary": "post-hoc descriptive association confounded with scale and instance structure; not a preregistered causal or predictive claim",
        },
        "multiplicity_boundary": "scale-by-CF cells and variability summaries are descriptive; no cellwise hypothesis tests were performed",
        "exclusions": "none",
    }

    atomic_csv(OUTPUT_ROOT / "core_paired_advantage_by_instance.csv", paired_rows)
    atomic_csv(OUTPUT_ROOT / "core_best_baseline_advantage.csv", best_rows)
    atomic_csv(OUTPUT_ROOT / "core_paired_advantage_scale_cf.csv", advantage_summary)
    atomic_csv(OUTPUT_ROOT / "core_seed_variability.csv", variability_rows)
    atomic_csv(OUTPUT_ROOT / "core_seed_variability_summary.csv", variability_summary)
    atomic_csv(OUTPUT_ROOT / "core_runtime_utilization.csv", runtime_rows)
    atomic_json(OUTPUT_ROOT / "supplementary_analysis.json", analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
