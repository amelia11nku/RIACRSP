#!/usr/bin/env python3
"""Audit and analyze completed Phase 6G DEV-HOLDOUT live-search results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASE = ROOT / "outputs/phase6g"
HOLDOUT = PHASE / "dev_holdout"
STATS = PHASE / "statistics"
PROFILING = PHASE / "profiling"
FIGURES = PHASE / "figures"
AUDIT = PHASE / "audit"
METHOD_ORDER = ("H1", "ALNS", "GA", "CSGNI")
STAGES = ("early", "mid", "late")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=180)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    runs = pd.read_csv(HOLDOUT / "dev_holdout_run_results.csv")
    instance = pd.read_csv(HOLDOUT / "dev_holdout_instance_summary.csv")
    payloads = [json.loads(path.read_text()) for path in sorted((HOLDOUT / "runs").rglob("*.json"))]
    log_paths = sorted((PHASE / "live_logs/dev_holdout").rglob("*.parquet"))
    logs = pd.concat((pd.read_parquet(path) for path in log_paths), ignore_index=True)
    logs["normalized_time"] = logs["elapsed_time"] / logs.groupby(
        ["instance_id", "seed"]
    )["elapsed_time"].transform("max")
    logs["search_stage"] = pd.cut(
        logs["normalized_time"], [-np.inf, 1 / 3, 2 / 3, np.inf], labels=STAGES
    ).astype(str)
    return runs, instance, payloads, logs


def integrity_audit(runs: pd.DataFrame, payloads: list[dict], logs: pd.DataFrame) -> dict:
    config = json.loads((ROOT / "configs/phase6g_live_solver.json").read_text())
    split = pd.read_csv(PHASE / "environment/dev_split.csv")
    split = split[split.split == "DEV_HOLDOUT"]
    operation_count = dict(zip(split.instance_id, split.number_of_operations))
    seeds = set(config["seeds"]["DEV_HOLDOUT"])
    expected = {"H1": 9, "ALNS": 90, "GA": 90, "CSGNI": 90}
    actual = runs.method.value_counts().to_dict()
    stochastic = runs[runs.method != "H1"]
    keys_unique = not stochastic.duplicated(["method", "instance_id", "seed"]).any()
    seed_sets_valid = all(
        set(part.seed.astype(int)) == seeds
        for _, part in stochastic.groupby(["method", "instance_id"])
    )
    budgets_valid = all(
        math.isclose(float(row.time_limit_seconds), 2.0 * operation_count[row.instance_id])
        for row in stochastic.itertuples(index=False)
    )
    log_keys = logs[["instance_id", "seed"]].drop_duplicates()
    csg_keys = stochastic[stochastic.method == "CSGNI"][["instance_id", "seed"]]
    log_coverage_valid = len(log_keys.merge(csg_keys, how="outer", indicator=True).query(
        "_merge != 'both'"
    )) == 0
    checks = {
        "result_count_279": len(runs) == 279 == len(payloads),
        "method_counts": actual == expected,
        "unique_stochastic_keys": keys_unique,
        "preregistered_seed_sets": seed_sets_valid,
        "wall_clock_budgets_2N": budgets_valid,
        "all_status_complete": all(item.get("status") == "COMPLETE" for item in payloads),
        "all_schedules_feasible": bool(runs.feasible.all()),
        "csgni_log_count_90": len(log_keys) == 90,
        "csgni_log_key_coverage": log_coverage_valid,
        "csgni_logs_nonempty": len(logs) > 0,
    }
    payload = {
        "schema": "phase6g-dev-holdout-integrity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_count": len(runs),
        "method_counts": actual,
        "csgni_iteration_rows": len(logs),
        "csgni_feasibility_rate": float(runs[runs.method == "CSGNI"].feasible.mean()),
    }
    atomic_json(payload, AUDIT / "dev_holdout_integrity.json")
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    return payload


def paired_statistics(instance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = instance.pivot(index=["instance_id", "scale", "CF_level"], columns="method", values="mean").reset_index()
    rng = np.random.default_rng(670501)
    rows = []
    for baseline in ("H1", "ALNS", "GA"):
        improvement = (means[baseline] - means["CSGNI"]) / means[baseline]
        boot = np.array([
            rng.choice(improvement.to_numpy(), size=len(improvement), replace=True).mean()
            for _ in range(20000)
        ])
        test = wilcoxon(means["CSGNI"], means[baseline], alternative="less", method="auto")
        delta = means["CSGNI"] - means[baseline]
        rows.append({
            "method_a": "CSGNI",
            "method_b": baseline,
            "paired_unit": "per_instance_mean_makespan",
            "instance_count": len(means),
            "mean_relative_improvement": float(improvement.mean()),
            "bootstrap_95ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_95ci_high": float(np.quantile(boot, 0.975)),
            "wins": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta > 0).sum()),
            "wilcoxon_alternative": "CSGNI_less_makespan",
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_p": float(test.pvalue),
        })
        means[f"improvement_vs_{baseline}"] = improvement
    pairwise = pd.DataFrame(rows)
    atomic_csv(pairwise, STATS / "pairwise_statistics.csv")

    subgroup_rows = []
    for group_name in ("scale", "CF_level"):
        for value, part in means.groupby(group_name):
            subgroup_rows.append({
                "group_type": group_name,
                "group_value": value,
                "instance_count": len(part),
                **{
                    f"mean_improvement_vs_{baseline}": float(part[f"improvement_vs_{baseline}"].mean())
                    for baseline in ("H1", "ALNS", "GA")
                },
            })
    subgroup = pd.DataFrame(subgroup_rows)
    atomic_csv(subgroup, STATS / "subgroup_improvement.csv")
    atomic_csv(means, STATS / "paired_instance_means.csv")
    return pairwise, means


def intervention_analysis(logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    groupings = [("overall", "ALL", logs)]
    for column in ("scale", "CF_level", "search_stage"):
        groupings.extend((column, str(value), part) for value, part in logs.groupby(column))
    for group_type, group_value, part in groupings:
        eligible = part[part.ni_eligible]
        ni = part[part.ni_intervention]
        fallback = part[part.fallback]
        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "iterations": len(part),
            "ni_eligible_iterations": len(eligible),
            "ni_interventions": int(part.ni_intervention.sum()),
            "fallbacks": int(part.fallback.sum()),
            "intervention_coverage": float(part.ni_intervention.sum() / max(len(eligible), 1)),
            "fallback_rate": float(part.fallback.sum() / max(len(eligible), 1)),
            "ni_immediate_improvement_rate": float((ni.immediate_relative_utility > 0).mean()),
            "fallback_immediate_improvement_rate": float((fallback.immediate_relative_utility > 0).mean()),
            "ni_acceptance_rate": float(ni.accepted.mean()),
            "fallback_acceptance_rate": float(fallback.accepted.mean()),
            "ni_global_best_rate": float(ni.new_global_best.mean()),
            "fallback_global_best_rate": float(fallback.new_global_best.mean()),
            "ni_mean_immediate_relative_utility": float(ni.immediate_relative_utility.mean()),
            "fallback_mean_immediate_relative_utility": float(fallback.immediate_relative_utility.mean()),
        })
    summary = pd.DataFrame(rows)
    atomic_csv(summary, STATS / "live_intervention_summary.csv")

    stage = summary[summary.group_type == "search_stage"].set_index("group_value").reindex(STAGES)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(stage.index, stage.intervention_coverage * 100, color="#4472C4")
    ax.set(ylabel="Intervention coverage (%)", xlabel="Search stage", ylim=(0, 100))
    ax.set_title("CSG-NI interventions by search stage")
    save_figure(fig, "interventions_by_search_stage")

    overall = summary.iloc[0]
    labels = ["Improvement", "Acceptance", "Global best"]
    ni_values = [overall.ni_immediate_improvement_rate, overall.ni_acceptance_rate, overall.ni_global_best_rate]
    fallback_values = [overall.fallback_immediate_improvement_rate, overall.fallback_acceptance_rate, overall.fallback_global_best_rate]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - .18, ni_values, .36, label="NI", color="#4472C4")
    ax.bar(x + .18, fallback_values, .36, label="Fallback", color="#ED7D31")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Rate", ylim=(0, 1))
    ax.legend()
    ax.set_title("NI vs fallback move outcomes")
    save_figure(fig, "ni_vs_fallback_move_quality")
    return summary, stage.reset_index()


def calibration_analysis(logs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    evaluated = logs[logs.ni_intervention].dropna(
        subset=["predicted_probability", "predicted_utility", "immediate_relative_utility"]
    ).copy()
    evaluated["realized_positive"] = (evaluated.immediate_relative_utility > 0).astype(float)
    evaluated["confidence_bin"] = pd.cut(
        evaluated.predicted_probability,
        bins=np.linspace(0, 1, 11), include_lowest=True,
    ).astype(str)
    try:
        evaluated["utility_decile"] = pd.qcut(
            evaluated.predicted_utility, 10, duplicates="drop"
        ).astype(str)
    except ValueError:
        evaluated["utility_decile"] = "all"

    rows = []
    for group_type, column in (("confidence_bin", "confidence_bin"), ("utility_decile", "utility_decile"), ("scale", "scale"), ("search_stage", "search_stage")):
        for value, part in evaluated.groupby(column, observed=True):
            rows.append({
                "group_type": group_type,
                "group_value": str(value),
                "count": len(part),
                "mean_predicted_probability": float(part.predicted_probability.mean()),
                "realized_positive_fraction": float(part.realized_positive.mean()),
                "mean_predicted_utility": float(part.predicted_utility.mean()),
                "mean_realized_utility": float(part.immediate_relative_utility.mean()),
                "probability_absolute_error": float(
                    abs(part.predicted_probability.mean() - part.realized_positive.mean())
                ),
                "utility_absolute_error": float(
                    abs(part.predicted_utility.mean() - part.immediate_relative_utility.mean())
                ),
            })
    summary = pd.DataFrame(rows)
    atomic_csv(summary, STATS / "live_calibration_summary.csv")

    confidence = summary[summary.group_type == "confidence_bin"]
    ece = float(np.average(confidence.probability_absolute_error, weights=confidence["count"]))
    utility_mae = float(np.mean(np.abs(
        evaluated.predicted_utility - evaluated.immediate_relative_utility
    )))
    utility_corr = spearmanr(
        evaluated.predicted_utility, evaluated.immediate_relative_utility,
        nan_policy="omit",
    )
    audit = {
        "schema": "phase6g-live-calibration-audit-v1",
        "evaluated_interventions": len(evaluated),
        "expected_calibration_error": ece,
        "utility_mae": utility_mae,
        "utility_spearman_r": float(utility_corr.statistic),
        "utility_spearman_p": float(utility_corr.pvalue),
        "realized_positive_fraction": float(evaluated.realized_positive.mean()),
        "mean_predicted_probability": float(evaluated.predicted_probability.mean()),
        "mean_predicted_utility": float(evaluated.predicted_utility.mean()),
        "mean_realized_utility": float(evaluated.immediate_relative_utility.mean()),
    }
    atomic_json(audit, AUDIT / "live_calibration_audit.json")

    deciles = summary[summary.group_type == "utility_decile"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(evaluated.predicted_utility, evaluated.immediate_relative_utility, s=3, alpha=.08, color="#A5A5A5")
    ax.plot(deciles.mean_predicted_utility, deciles.mean_realized_utility, "o-", color="#4472C4", label="Decile mean")
    low = min(evaluated.predicted_utility.min(), evaluated.immediate_relative_utility.min())
    high = max(evaluated.predicted_utility.max(), evaluated.immediate_relative_utility.max())
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1, label="Ideal")
    ax.set(xlabel="Predicted utility", ylabel="Realized immediate utility")
    ax.legend()
    ax.set_title("Predicted vs realized NI utility")
    save_figure(fig, "predicted_vs_realized_utility")
    return summary, audit


def runtime_profile(runs: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    components = {
        "CSG build": "csg_build_ms",
        "Target bank": "target_bank_ms",
        "Tensorization": "tensorization_ms",
        "GPU forward": "model_inference_ms",
        "Target scoring": "action_scoring_ms",
        "Calibration gate": "calibration_gate_ms",
        "Selected repair+decoder": "repair_time_ms",
    }
    total_runtime_ms = runs[runs.method == "CSGNI"].runtime.sum() * 1000.0
    rows = []
    for name, column in components.items():
        values = logs[column].astype(float)
        rows.append({
            "component": name,
            "event_count": len(values),
            "mean_ms": float(values.mean()),
            "median_ms": float(values.median()),
            "p95_ms": float(values.quantile(.95)),
            "total_seconds": float(values.sum() / 1000.0),
            "fraction_of_csgni_solver_runtime": float(values.sum() / total_runtime_ms),
        })
    overhead = logs.ni_overhead_ms.sum()
    rows.append({
        "component": "Total neural decision overhead",
        "event_count": len(logs),
        "mean_ms": float(logs.ni_overhead_ms.mean()),
        "median_ms": float(logs.ni_overhead_ms.median()),
        "p95_ms": float(logs.ni_overhead_ms.quantile(.95)),
        "total_seconds": float(overhead / 1000.0),
        "fraction_of_csgni_solver_runtime": float(overhead / total_runtime_ms),
    })
    profile = pd.DataFrame(rows)
    atomic_csv(profile, PROFILING / "runtime_profile.csv")

    chart = profile[profile.component.isin(components)]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(chart.component, chart.total_seconds, color="#4472C4")
    ax.tick_params(axis="x", rotation=35)
    ax.set(ylabel="Total time (seconds)")
    ax.set_title("CSG-NI runtime component totals")
    save_figure(fig, "runtime_overhead_breakdown")
    return profile


def _trace_at(trace: list[dict], x: np.ndarray, domain: str, denominator: float) -> np.ndarray:
    key = "elapsed_time" if domain == "time" else "decoder_evaluations"
    points_x = np.array([float(point[key]) / denominator for point in trace])
    points_y = np.array([float(point["current_best_makespan"]) for point in trace])
    indices = np.searchsorted(points_x, x, side="right") - 1
    indices = np.clip(indices, 0, len(points_y) - 1)
    return points_y[indices]


def trajectory_analysis(payloads: list[dict], runs: pd.DataFrame) -> pd.DataFrame:
    grid = np.linspace(0, 1, 21)
    rows = []
    payload_map = {
        (item["method"], item["instance_id"], item.get("seed")): item for item in payloads
    }
    h1 = runs[runs.method == "H1"].set_index("instance_id").best_makespan
    for domain in ("time", "evaluations"):
        values = {method: [] for method in ("ALNS", "CSGNI")}
        for method in values:
            for item in payloads:
                if item["method"] != method:
                    continue
                denominator = (
                    float(item["time_limit_seconds"])
                    if domain == "time" else float(item["decoder_evaluations"])
                )
                values[method].append(_trace_at(item["convergence_trace"], grid, domain, denominator))
            matrix = np.stack(values[method])
            for index, position in enumerate(grid):
                rows.append({
                    "domain": domain,
                    "normalized_position": position,
                    "method": method,
                    "mean_best_makespan": float(matrix[:, index].mean()),
                    "median_best_makespan": float(np.median(matrix[:, index])),
                })

    for method in ("ALNS", "CSGNI"):
        for item in payloads:
            if item["method"] != method:
                continue
            initial = float(h1[item["instance_id"]])
            first = next((
                point["elapsed_time"] for point in item["convergence_trace"]
                if point["current_best_makespan"] < initial
            ), None)
            rows.append({
                "domain": "run_timing",
                "normalized_position": np.nan,
                "method": method,
                "instance_id": item["instance_id"],
                "seed": item["seed"],
                "time_to_first_improvement_over_h1": first,
                "time_to_final_best": item["best_found_time"],
                "normalized_time_to_final_best": item["best_found_time"] / item["time_limit_seconds"],
            })
    trajectory = pd.DataFrame(rows)
    atomic_csv(trajectory, STATS / "trajectory_summary.csv")

    for domain, stem, xlabel in (
        ("time", "alns_vs_csgni_convergence_by_time", "Normalized wall-clock time"),
        ("evaluations", "alns_vs_csgni_convergence_by_evaluations", "Normalized decoder evaluations"),
    ):
        part = trajectory[trajectory.domain == domain]
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for method, color in (("ALNS", "#ED7D31"), ("CSGNI", "#4472C4")):
            curve = part[part.method == method]
            ax.plot(curve.normalized_position, curve.mean_best_makespan, label=method, color=color)
        ax.set(xlabel=xlabel, ylabel="Mean best makespan")
        ax.legend()
        ax.set_title("ALNS vs CSG-NI convergence")
        save_figure(fig, stem)
    return trajectory


def result_figures(runs: pd.DataFrame, paired: pd.DataFrame) -> None:
    stochastic = runs[runs.method != "H1"]
    data = [stochastic[stochastic.method == method].best_makespan for method in ("ALNS", "GA", "CSGNI")]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.boxplot(data, tick_labels=("ALNS", "GA", "CSG-NI"), showmeans=True)
    ax.set(ylabel="Final makespan")
    ax.set_title("DEV-HOLDOUT final makespan distribution")
    save_figure(fig, "dev_holdout_makespan_comparison")

    part = paired.sort_values(["scale", "CF_level"])
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = ["#4472C4" if value >= 0 else "#C00000" for value in part.improvement_vs_ALNS]
    ax.bar(part.instance_id, part.improvement_vs_ALNS * 100, color=colors)
    ax.axhline(0, color="black", linewidth=.8)
    ax.tick_params(axis="x", rotation=55)
    ax.set(ylabel="CSG-NI improvement over ALNS (%)")
    ax.set_title("Per-instance mean improvement")
    save_figure(fig, "csgni_improvement_over_alns")


def main() -> None:
    runs, instance, payloads, logs = load_results()
    integrity = integrity_audit(runs, payloads, logs)
    pairwise, paired = paired_statistics(instance)
    intervention, _ = intervention_analysis(logs)
    _, calibration = calibration_analysis(logs)
    profile = runtime_profile(runs, logs)
    trajectory_analysis(payloads, runs)
    result_figures(runs, paired)
    atomic_json({
        "schema": "phase6g-live-analysis-status-v1",
        "status": "COMPLETE_EXCEPT_STATE_DRIFT",
        "integrity": integrity["status"],
        "csgni_improvement_over_alns": float(
            pairwise.loc[pairwise.method_b == "ALNS", "mean_relative_improvement"].iloc[0]
        ),
        "csgni_wilcoxon_vs_alns_p": float(
            pairwise.loc[pairwise.method_b == "ALNS", "wilcoxon_p"].iloc[0]
        ),
        "intervention_coverage": float(intervention.iloc[0].intervention_coverage),
        "fallback_rate": float(intervention.iloc[0].fallback_rate),
        "calibration": calibration,
        "neural_overhead_fraction": float(
            profile.loc[
                profile.component == "Total neural decision overhead",
                "fraction_of_csgni_solver_runtime",
            ].iloc[0]
        ),
        "state_drift_status": "PENDING_FEATURE_CAPTURE_AUDIT",
    }, AUDIT / "live_analysis_status.json")
    print("PHASE6G_LIVE_ANALYSIS_COMPLETE_EXCEPT_STATE_DRIFT")


if __name__ == "__main__":
    main()
