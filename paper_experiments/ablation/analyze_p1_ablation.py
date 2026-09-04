#!/usr/bin/env python3
"""Validate and analyze the completed three-arm P1 ablation matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.env.insertion_decoder import Action  # noqa: E402
from rcias_clgri.env.rcias_env import RCIASConstructionEnv  # noqa: E402


ABLATION_ROOT = ROOT / "paper_experiments/ablation"
CONFIG_PATH = ABLATION_ROOT / "configs/p1_ablation_protocol.json"
MANIFEST_PATH = ABLATION_ROOT / "ablation_instance_manifest.csv"
OUTPUT_ROOT = ABLATION_ROOT / "processed_data"
TABLE_ROOT = ABLATION_ROOT / "tables"
SNIPPET_ROOT = ROOT / "paper_experiments/reports/snippets"
REFERENCE_MANIFEST_PATH = ABLATION_ROOT / "audit/canonical_reference_manifest.csv"
ARMS = (
    "CSG-NI Full",
    "Uniform full-bank selection",
    "No NI (ALNS-H1)",
)
ARM_ROOTS = {
    "CSG-NI Full": ROOT / "paper_experiments/raw_results/core45/CSG_NI_PROVISIONAL_PHASE6H/runs",
    "Uniform full-bank selection": ABLATION_ROOT / "raw_results/random_full_bank_frozen_gate/runs",
    "No NI (ALNS-H1)": ABLATION_ROOT / "raw_results/no_ni_alns_h1_equivalence/runs",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def replay(instance_path: Path, payload: dict) -> float:
    instance = load_instance(instance_path)
    environment = RCIASConstructionEnv(instance)
    for raw in payload["best_actions"]:
        environment.step(Action(**raw))
    audit = check_schedule(instance, environment.schedule)
    makespan = environment.objective().makespan
    if not environment.done or not audit["feasible"] or makespan != float(payload["best_makespan"]):
        raise RuntimeError(f"stored action replay failed: {instance.instance_id}")
    return makespan


def paired_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    return (positive - negative) / (positive + negative)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, label in enumerate(ordered):
        value = min(1.0, raw[label] * (total - index))
        running = max(running, value)
        adjusted[label] = running
    return adjusted


def latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{float(mantissa):.2f}\\times 10^{{{int(exponent)}}}"


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    progress_path = ABLATION_ROOT / "raw_results/random_full_bank_frozen_gate/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "COMPLETE" or progress.get("completed_runs") != 90:
        raise RuntimeError("P1 random-selection arm is not complete (90/90 required)")
    manifest = read_csv(MANIFEST_PATH)
    if len(manifest) != 18:
        raise RuntimeError("P1 manifest must contain 18 instances")
    reference_rows = read_csv(REFERENCE_MANIFEST_PATH)
    reference_lookup = {
        (row["arm"], row["instance_id"], int(row["seed"])): row
        for row in reference_rows
    }
    if len(reference_rows) != 180 or len(reference_lookup) != 180:
        raise RuntimeError("P1 canonical reference manifest must contain 180 unique reused results")

    runs: list[dict[str, object]] = []
    for instance_row in manifest:
        instance_id = instance_row["instance_id"]
        instance_path = ROOT / instance_row["instance_path"]
        for arm in ARMS:
            for seed in config["seeds"]:
                path = ARM_ROOTS[arm] / instance_id / f"seed_{seed}.json"
                reference = reference_lookup.get((arm, instance_id, int(seed)))
                relative_path = str(path.relative_to(ROOT))
                result_hash = digest(path)
                if arm != "Uniform full-bank selection":
                    if (
                        reference is None
                        or reference["canonical_result_path"] != relative_path
                        or reference["canonical_result_sha256"] != result_hash
                    ):
                        raise RuntimeError(f"P1 canonical result manifest mismatch: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("instance_id") != instance_id
                    or int(payload.get("seed")) != int(seed)
                    or payload.get("feasible") is not True
                ):
                    raise RuntimeError(f"invalid P1 result identity: {path}")
                replay(instance_path, payload)
                runs.append({
                    "arm": arm,
                    "instance_id": instance_id,
                    "scale": instance_row["scale"],
                    "CF_level": instance_row["CF_level"],
                    "seed": int(seed),
                    "operation_count": int(instance_row["number_of_operations"]),
                    "best_makespan": float(payload["best_makespan"]),
                    "runtime_seconds": float(payload["runtime"]),
                    "time_limit_seconds": float(payload["time_limit_seconds"]),
                    "best_found_time_seconds": float(payload["best_found_time"]),
                    "decoder_evaluations": int(payload["decoder_evaluations"]),
                    "iterations": int(payload["iterations"]),
                    "feasible": True,
                    "replay_feasible": True,
                    "result_path": relative_path,
                    "result_sha256": result_hash,
                })
    if len(runs) != 270:
        raise RuntimeError("P1 requires 270 total arm-instance-seed records")

    pooled_best = {
        instance_id: min(float(row["best_makespan"]) for row in runs if row["instance_id"] == instance_id)
        for instance_id in {str(row["instance_id"]) for row in runs}
    }
    for row in runs:
        bks = pooled_best[str(row["instance_id"])]
        row["pooled_best_makespan"] = bks
        row["rpd_percent"] = 100.0 * (float(row["best_makespan"]) - bks) / bks

    instance_rows: list[dict[str, object]] = []
    for instance_row in manifest:
        instance_id = instance_row["instance_id"]
        temporary = []
        for arm in ARMS:
            group = [row for row in runs if row["instance_id"] == instance_id and row["arm"] == arm]
            median_rpd = float(np.median([float(row["rpd_percent"]) for row in group]))
            temporary.append((arm, group, median_rpd))
        ranks = rankdata([item[2] for item in temporary], method="average")
        for rank, (arm, group, median_rpd) in zip(ranks, temporary):
            instance_rows.append({
                "arm": arm,
                "instance_id": instance_id,
                "scale": instance_row["scale"],
                "CF_level": instance_row["CF_level"],
                "operation_count": int(instance_row["number_of_operations"]),
                "seed_count": len(group),
                "median_rpd_percent": median_rpd,
                "mean_rpd_percent": float(np.mean([float(row["rpd_percent"]) for row in group])),
                "objective_rank": float(rank),
                "attains_pooled_best": any(float(row["best_makespan"]) == pooled_best[instance_id] for row in group),
                "median_decoder_evaluations": float(np.median([float(row["decoder_evaluations"]) for row in group])),
                "median_time_to_best_seconds": float(np.median([float(row["best_found_time_seconds"]) for row in group])),
                "median_time_to_best_budget_fraction": float(np.median([
                    float(row["best_found_time_seconds"]) / float(row["time_limit_seconds"]) for row in group
                ])),
                "median_runtime_seconds": float(np.median([float(row["runtime_seconds"]) for row in group])),
                "feasibility_rate": float(np.mean([bool(row["feasible"]) for row in group])),
            })

    full_lookup = {
        row["instance_id"]: float(row["median_rpd_percent"])
        for row in instance_rows if row["arm"] == "CSG-NI Full"
    }
    differences: list[dict[str, object]] = []
    for row in instance_rows:
        if row["arm"] == "CSG-NI Full":
            continue
        value = float(row["median_rpd_percent"]) - full_lookup[str(row["instance_id"])]
        differences.append({
            "ablation": row["arm"],
            "instance_id": row["instance_id"],
            "scale": row["scale"],
            "CF_level": row["CF_level"],
            "paired_rpd_difference_positive_favors_full": value,
            "full_outcome": "win" if value > 0 else "loss" if value < 0 else "tie",
        })

    matrices = {
        arm: np.asarray([
            float(next(row for row in instance_rows if row["arm"] == arm and row["instance_id"] == item["instance_id"])["median_rpd_percent"])
            for item in manifest
        ])
        for arm in ARMS
    }
    friedman = friedmanchisquare(*(matrices[arm] for arm in ARMS))
    raw_p: dict[str, float] = {}
    effects: dict[str, float] = {}
    for arm in ARMS[1:]:
        delta = matrices[arm] - matrices["CSG-NI Full"]
        raw_p[arm] = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
        effects[arm] = paired_rank_biserial(delta)
    adjusted = holm_adjust(raw_p) if friedman.pvalue < 0.05 else {arm: raw_p[arm] for arm in raw_p}

    table_rows: list[dict[str, object]] = []
    for arm in ARMS:
        group = [row for row in instance_rows if row["arm"] == arm]
        comparison = [row for row in differences if row["ablation"] == arm]
        table_rows.append({
            "arm": arm,
            "instance_count": len(group),
            "mean_instance_median_rpd_percent": float(np.mean([float(row["median_rpd_percent"]) for row in group])),
            "median_instance_median_rpd_percent": float(np.median([float(row["median_rpd_percent"]) for row in group])),
            "average_rank": float(np.mean([float(row["objective_rank"]) for row in group])),
            "pooled_best_attainment_count": sum(bool(row["attains_pooled_best"]) for row in group),
            "median_decoder_evaluations": float(np.median([float(row["median_decoder_evaluations"]) for row in group])),
            "median_time_to_best_budget_fraction": float(np.median([float(row["median_time_to_best_budget_fraction"]) for row in group])),
            "feasibility_rate": float(np.mean([float(row["feasibility_rate"]) for row in group])),
            "full_wins": sum(row["full_outcome"] == "win" for row in comparison) if comparison else "",
            "ties": sum(row["full_outcome"] == "tie" for row in comparison) if comparison else "",
            "full_losses": sum(row["full_outcome"] == "loss" for row in comparison) if comparison else "",
            "wilcoxon_p_raw_vs_full": raw_p.get(arm, ""),
            "holm_p_vs_full": adjusted.get(arm, ""),
            "paired_rank_biserial_positive_favors_full": effects.get(arm, ""),
        })

    mechanism_rows: list[dict[str, object]] = []
    selection_events: list[dict[str, object]] = []
    full_live_root = ROOT / "paper_experiments/raw_results/core45/CSG_NI_PROVISIONAL_PHASE6H/live_logs"
    random_live_root = ABLATION_ROOT / "raw_results/random_full_bank_frozen_gate/live_logs"
    for arm, live_root in (("CSG-NI Full", full_live_root), ("Uniform full-bank selection", random_live_root)):
        for instance_row in manifest:
            instance_id = instance_row["instance_id"]
            for seed in config["seeds"]:
                frame = pd.read_parquet(live_root / instance_id / f"seed_{seed}.parquet")
                eligible = frame[frame["ni_eligible"].astype(bool)]
                if eligible.empty or not (eligible["requested_bank_size"] == 24).all():
                    raise RuntimeError(f"invalid live candidate-bank evidence: {(arm, instance_id, seed)}")
                result = next(
                    row for row in runs
                    if row["arm"] == arm and row["instance_id"] == instance_id and row["seed"] == int(seed)
                )
                mechanism_rows.append({
                    "arm": arm,
                    "instance_id": instance_id,
                    "scale": instance_row["scale"],
                    "CF_level": instance_row["CF_level"],
                    "seed": int(seed),
                    "eligible_decisions": len(eligible),
                    "interventions": int(eligible["ni_intervention"].astype(bool).sum()),
                    "intervention_coverage": float(eligible["ni_intervention"].astype(bool).mean()),
                    "median_unique_bank_size": float(eligible["candidate_bank_size"].median()),
                    "median_duplicate_rule_count": float(eligible["duplicate_bank_size"].median()),
                    "median_ni_overhead_ms_per_eligible_decision": float(eligible["ni_overhead_ms"].median()),
                    "cumulative_ni_overhead_fraction": float(eligible["ni_overhead_ms"].sum() / 1000.0 / float(result["runtime_seconds"])),
                    "median_uniform_rebuild_ms": (
                        float(eligible["uniform_selection_bank_rebuild_ms"].median())
                        if "uniform_selection_bank_rebuild_ms" in eligible else 0.0
                    ),
                })
                intervened = eligible[eligible["ni_intervention"].astype(bool)]
                for family, count in intervened["selected_origin_family"].value_counts().items():
                    selection_events.append({
                        "arm": arm,
                        "scale": instance_row["scale"],
                        "origin_family": str(family),
                        "selection_count": int(count),
                    })

    selection_rows: list[dict[str, object]] = []
    families = sorted({str(row["origin_family"]) for row in selection_events})
    for arm in ("CSG-NI Full", "Uniform full-bank selection"):
        for scale in ("S", "M", "L"):
            subset = [row for row in selection_events if row["arm"] == arm and row["scale"] == scale]
            total = sum(int(row["selection_count"]) for row in subset)
            for family in families:
                count = sum(
                    int(row["selection_count"])
                    for row in subset if row["origin_family"] == family
                )
                selection_rows.append({
                    "arm": arm,
                    "scale": scale,
                    "origin_family": family,
                    "selection_count": count,
                    "selection_fraction": count / total if total else 0.0,
                    "total_interventions": total,
                    "reporting_boundary": "pooled state-level process diagnostic; not an inferential replicate count",
                })

    stats = {
        "schema": "initial-manuscript-p1-ablation-analysis-v1",
        "status": "PASS_COMPLETE_REPLAY_VALIDATED",
        "independent_unit": "18 instance-level medians over five matched seeds",
        "friedman": {
            "chi_square": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "arm_count": 3,
            "instance_count": 18,
        },
        "post_hoc": {
            arm: {
                "wilcoxon_two_sided_p_raw": raw_p[arm],
                "holm_adjusted_p": adjusted[arm],
                "paired_rank_biserial_positive_favors_full": effects[arm],
            }
            for arm in ARMS[1:]
        },
        "multiplicity": "Holm correction across the two Full-versus-ablation paired comparisons",
        "pooled_best_definition": "minimum makespan across all three arms and five seeds on each of the 18 instances",
        "candidate_bank_rule_count": 24,
        "schedule_replay": "270/270 feasible and exact-makespan reproducing",
    }
    atomic_csv(OUTPUT_ROOT / "ablation_runs.csv", runs)
    atomic_csv(OUTPUT_ROOT / "ablation_instance_summary.csv", instance_rows)
    atomic_csv(OUTPUT_ROOT / "ablation_paired_differences.csv", differences)
    atomic_csv(OUTPUT_ROOT / "ablation_method_summary.csv", table_rows)
    atomic_csv(OUTPUT_ROOT / "mechanism_run_summary.csv", mechanism_rows)
    atomic_csv(OUTPUT_ROOT / "mechanism_selection_family.csv", selection_rows)
    atomic_text(OUTPUT_ROOT / "ablation_statistics.json", json.dumps(stats, indent=2, sort_keys=True) + "\n")
    atomic_csv(TABLE_ROOT / "table_p1_ablation.csv", table_rows)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation analysis of the principal CSG-NI mechanisms on 18 independent instances, using five matched seeds per arm. Best is the number of pooled-best attainments; Dec. evals is the median decoder-evaluation count; W/T/L reports Full wins, ties, and losses against each ablation. Holm-adjusted two-sided Wilcoxon $p$ values and paired rank-biserial effects use instance medians; positive effects favor Full. All 270 schedules were feasible on replay.}",
        "\\label{tab:p1_ablation}",
        "\\footnotesize",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Arm & Mean RPD & Median RPD & Rank & Best & Dec. evals & W/T/L & Holm $p$ & $r_{rb}$ \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        p = "--" if row["holm_p_vs_full"] == "" else f"{float(row['holm_p_vs_full']):.3g}"
        effect = "--" if row["paired_rank_biserial_positive_favors_full"] == "" else f"{float(row['paired_rank_biserial_positive_favors_full']):+.2f}"
        outcome = (
            "--"
            if row["full_wins"] == ""
            else f"{row['full_wins']}/{row['ties']}/{row['full_losses']}"
        )
        display_arm = str(row["arm"]).replace("_", "\\_")
        lines.append(
            f"{display_arm} & {float(row['mean_instance_median_rpd_percent']):.2f} & "
            f"{float(row['median_instance_median_rpd_percent']):.2f} & {float(row['average_rank']):.2f} & "
            f"{row['pooled_best_attainment_count']} & {float(row['median_decoder_evaluations']):.0f} & "
            f"{outcome} & {p} & {effect} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    atomic_text(TABLE_ROOT / "table_p1_ablation.tex", "\n".join(lines))

    full = table_rows[0]
    random_arm = table_rows[1]
    no_ni = table_rows[2]
    paragraph = (
        "\\subsection{Mechanism ablation} On the 18-instance balanced Core subset, Full CSG-NI "
        f"achieved a mean instance-median RPD of {float(full['mean_instance_median_rpd_percent']):.2f}\\%, "
        f"compared with {float(random_arm['mean_instance_median_rpd_percent']):.2f}\\% after replacing "
        "learned target prioritization by uniform full-bank selection and "
        f"{float(no_ni['mean_instance_median_rpd_percent']):.2f}\\% without NI. The three-arm Friedman "
        f"test gave $\\chi^2={float(friedman.statistic):.2f}$ "
        f"($p={latex_scientific(float(friedman.pvalue))}$). "
        "Holm-adjusted paired results and effect sizes are reported in Table~\\ref{tab:p1_ablation}; "
        "mechanism claims are restricted to this ablation subset and to components that were cleanly separable.\n"
    )
    atomic_text(SNIPPET_ROOT / "p1_ablation_results.tex", paragraph)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
