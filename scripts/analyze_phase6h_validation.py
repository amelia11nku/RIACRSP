#!/usr/bin/env python3
"""Reconstruct Phase 6H final-quality, calibration, drift, and anytime results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import (  # noqa: E402
    first_common_target_hit,
    normalized_gap_auc,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.ni.calibration import calibration_metrics, reliability_table  # noqa: E402
from scripts.analyze_phase6g_drift import FEATURES, compare  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
OUT = ROOT / "outputs/phase6h_validation"
STATISTICS = OUT / "statistics"
ANYTIME = OUT / "anytime"
AUDIT = OUT / "audit"
METHODS = ("H1", "ALNS", "GA", "DCGA", "PHASE6G_CSGNI", "PHASE6H_CSGNI")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_path(row) -> Path:
    filename = "result.json" if pd.isna(row.seed) else f"seed_{int(row.seed)}.json"
    return OUT / "runs" / row.method / row.instance_id / filename


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def final_quality_statistics(runs: pd.DataFrame, config: dict) -> pd.DataFrame:
    instance_rows = []
    for (method, instance_id), part in runs.groupby(["method", "instance_id"]):
        instance_rows.append({
            "method": method,
            "instance_id": instance_id,
            "scale": part.scale.iloc[0],
            "CF_level": part.CF_level.iloc[0],
            "run_count": len(part),
            "mean_final_makespan": float(part.final_best.mean()),
            "best_final_makespan": float(part.final_best.min()),
            "std_final_makespan": float(part.final_best.std(ddof=0)),
            "feasibility_rate": float(part.feasible.mean()),
            "mean_runtime": float(part.total_runtime.mean()),
            "mean_decoder_evaluations": float(part.total_decoder_evals.mean()),
            "median_time_to_best": float(part.time_to_best.median()),
            "median_evals_to_best": float(part.evals_to_best.median()),
        })
    instances = pd.DataFrame(instance_rows)
    atomic_csv(instances, STATISTICS / "instance_method_summary.csv")
    pivot = instances.pivot(index="instance_id", columns="method", values="mean_final_makespan")
    method_rows = []
    for method, part in instances.groupby("method"):
        method_rows.append({
            "method": method,
            "instance_count": len(part),
            "mean_of_instance_means": float(part.mean_final_makespan.mean()),
            "mean_of_instance_best": float(part.best_final_makespan.mean()),
            "mean_instance_std": float(part.std_final_makespan.mean()),
            "feasibility_rate": float(part.feasibility_rate.mean()),
            "mean_runtime": float(part.mean_runtime.mean()),
            "mean_decoder_evaluations": float(part.mean_decoder_evaluations.mean()),
            "median_time_to_best": float(part.median_time_to_best.median()),
            "median_evals_to_best": float(part.median_evals_to_best.median()),
            "mean_improvement_vs_alns": (
                None if method == "H1" else float(
                    ((pivot["ALNS"] - pivot[method]) / pivot["ALNS"]).mean()
                )
            ),
        })
    atomic_csv(pd.DataFrame(method_rows), STATISTICS / "method_summary.csv")

    primary = pivot["PHASE6H_CSGNI"]
    pairwise_rows = []
    bootstrap_seed = int(config["seeds"]["ANALYSIS_BOOTSTRAP"][0])
    for index, baseline in enumerate(("H1", "ALNS", "GA", "DCGA", "PHASE6G_CSGNI")):
        relative = ((pivot[baseline] - primary) / pivot[baseline]).to_numpy(dtype=float)
        differences = pivot[baseline] - primary
        low, high = bootstrap_interval(relative, bootstrap_seed + index)
        pairwise_rows.append({
            "method_a": "PHASE6H_CSGNI",
            "method_b": baseline,
            "mean_relative_improvement": float(relative.mean()),
            "bootstrap_95_low": low,
            "bootstrap_95_high": high,
            "wins": int((differences > 0).sum()),
            "ties": int((differences == 0).sum()),
            "losses": int((differences < 0).sum()),
            "wilcoxon_one_sided_p": float(wilcoxon(
                primary, pivot[baseline], alternative="less", zero_method="wilcox"
            ).pvalue),
        })
    atomic_csv(pd.DataFrame(pairwise_rows), STATISTICS / "pairwise_statistics.csv")

    subgroup_rows = []
    joined = instances.merge(
        instances[instances.method == "PHASE6H_CSGNI"][["instance_id", "mean_final_makespan"]]
        .rename(columns={"mean_final_makespan": "phase6h_mean"}),
        on="instance_id",
    )
    for baseline in ("ALNS", "GA", "DCGA", "PHASE6G_CSGNI"):
        part = joined[joined.method == baseline].copy()
        part["relative_improvement"] = (
            part.mean_final_makespan - part.phase6h_mean
        ) / part.mean_final_makespan
        for grouping in ("scale", "CF_level"):
            for group, cell in part.groupby(grouping):
                subgroup_rows.append({
                    "baseline": baseline,
                    "grouping": grouping,
                    "group": group,
                    "instance_count": len(cell),
                    "mean_relative_improvement": float(cell.relative_improvement.mean()),
                    "minimum_relative_improvement": float(cell.relative_improvement.min()),
                })
    atomic_csv(pd.DataFrame(subgroup_rows), STATISTICS / "subgroup_improvement.csv")
    return instances


def anytime_statistics(runs: pd.DataFrame, config: dict) -> pd.DataFrame:
    references = runs.groupby("instance_id").final_best.min().rename("reference_makespan")
    atomic_csv(references.reset_index().assign(reference_type="POOLED_BKS"), ANYTIME / "common_references.csv")
    budget_by_instance = (
        runs.dropna(subset=["time_limit_seconds"])
        .groupby("instance_id").time_limit_seconds.first()
    )
    run_rows = []
    checkpoint_rows = []
    target_rows = []
    for row in runs.itertuples(index=False):
        payload = load_json(run_path(row))
        trace = validate_incumbent_trace(
            payload["incumbent_trace"], final_best=float(row.final_best)
        )
        budget = float(budget_by_instance[row.instance_id])
        reference = float(references[row.instance_id])
        auc = normalized_gap_auc(trace, budget=budget, reference_makespan=reference)
        run_rows.append({
            "method": row.method,
            "instance_id": row.instance_id,
            "seed": row.seed,
            "reference_makespan": reference,
            "reference_type": "POOLED_BKS",
            "normalized_gap_auc": auc,
            "time_to_best": row.time_to_best,
            "evals_to_best": row.evals_to_best,
        })
        for sampled in sample_incumbent_trace(
            trace,
            budget=budget,
            fractions=config["anytime"]["normalized_budget_fractions"],
        ):
            checkpoint_rows.append({
                "method": row.method,
                "instance_id": row.instance_id,
                "seed": row.seed,
                **sampled,
                "relative_gap_to_reference": (
                    None if sampled["best_makespan"] is None else max(
                        0.0, float(sampled["best_makespan"]) / reference - 1.0
                    )
                ),
            })
        for target_gap in config["anytime"]["common_target_gaps"]:
            hit = first_common_target_hit(
                trace, target_makespan=reference * (1.0 + float(target_gap))
            )
            target_rows.append({
                "method": row.method,
                "instance_id": row.instance_id,
                "seed": row.seed,
                "target_gap": float(target_gap),
                "target_makespan": reference * (1.0 + float(target_gap)),
                **hit,
            })
    anytime_runs = pd.DataFrame(run_rows)
    checkpoints = pd.DataFrame(checkpoint_rows)
    targets = pd.DataFrame(target_rows)
    atomic_csv(anytime_runs, ANYTIME / "run_anytime_summary.csv")
    atomic_csv(checkpoints, ANYTIME / "normalized_budget_checkpoints.csv")
    atomic_csv(targets, ANYTIME / "common_target_hitting_times.csv")
    target_summary = targets.groupby(["method", "target_gap"]).agg(
        run_count=("reached", "size"),
        target_hit_rate=("reached", "mean"),
        median_hit_time=("elapsed_wall_time", "median"),
        median_hit_evaluations=("decoder_evaluations", "median"),
    ).reset_index()
    atomic_csv(target_summary, ANYTIME / "target_hit_summary.csv")
    anytime_method = anytime_runs.groupby("method").agg(
        mean_normalized_gap_auc=("normalized_gap_auc", "mean"),
        median_normalized_gap_auc=("normalized_gap_auc", "median"),
        median_time_to_best=("time_to_best", "median"),
        median_evals_to_best=("evals_to_best", "median"),
    ).reset_index()
    atomic_csv(anytime_method, ANYTIME / "method_anytime_summary.csv")
    return anytime_method


def calibration_and_drift(
    config: dict, runs: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    calibration_rows = []
    reliability_rows = []
    intervention_rows = []
    runtime_rows = []
    phase6h_live = []
    for method in ("PHASE6G_CSGNI", "PHASE6H_CSGNI"):
        paths = sorted((OUT / "live_logs" / method).rglob("*.parquet"))
        live = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        eligible = live[live.ni_eligible].copy()
        selected = eligible[eligible.ni_intervention].copy()
        if selected.empty:
            raise RuntimeError(f"{method} has no evaluated NI interventions")
        metrics = calibration_metrics(
            selected.calibrated_probability,
            selected.realized_positive,
            bins=int(config["probability_calibration"]["reliability_bins"]),
        )
        rank = spearmanr(
            selected.calibrated_utility, selected.realized_immediate_utility
        ).statistic
        calibration_rows.append({
            "method": method,
            "eligible_state_count": len(eligible),
            "evaluated_intervention_count": len(selected),
            "intervention_coverage": float(eligible.ni_intervention.mean()),
            "fallback_coverage": float(eligible.fallback.mean()),
            **metrics,
            "utility_mae": float(np.mean(np.abs(
                selected.calibrated_utility - selected.realized_immediate_utility
            ))),
            "utility_rmse": float(np.sqrt(np.mean(
                (selected.calibrated_utility - selected.realized_immediate_utility) ** 2
            ))),
            "utility_spearman_r": float(rank) if np.isfinite(rank) else 0.0,
            "intervention_acceptance_rate": float(selected.accepted.mean()),
            "intervention_global_best_rate": float(selected.new_global_best.mean()),
            "mean_intervention_utility": float(selected.realized_immediate_utility.mean()),
        })
        table = reliability_table(
            selected.calibrated_probability,
            selected.realized_positive,
            bins=int(config["probability_calibration"]["reliability_bins"]),
        )
        table.insert(0, "method", method)
        reliability_rows.append(table)
        groups = [("overall", "ALL", eligible)]
        groups.extend(
            ("scale", str(group), part) for group, part in eligible.groupby("scale")
        )
        groups.extend(
            ("CF_level", str(group), part)
            for group, part in eligible.groupby("CF_level")
        )
        for grouping, group, part in groups:
            interventions = part[part.ni_intervention]
            intervention_rows.append({
                "method": method,
                "grouping": grouping,
                "group": group,
                "eligible_state_count": len(part),
                "intervention_count": len(interventions),
                "fallback_count": int(part.fallback.sum()),
                "intervention_coverage": float(part.ni_intervention.mean()),
                "fallback_coverage": float(part.fallback.mean()),
                "immediate_positive_rate": float(interventions.realized_positive.mean()),
                "acceptance_rate": float(interventions.accepted.mean()),
                "global_best_hit_rate": float(interventions.new_global_best.mean()),
                "mean_immediate_utility": float(
                    interventions.realized_immediate_utility.mean()
                ),
                "mean_predicted_probability": float(
                    interventions.calibrated_probability.mean()
                ),
                "mean_predicted_utility": float(
                    interventions.calibrated_utility.mean()
                ),
            })
        runtime_lookup = runs[runs.method == method].set_index(
            ["instance_id", "seed"]
        ).total_runtime
        for (instance_id, seed), part in live.groupby(["instance_id", "seed"]):
            solver_runtime = float(runtime_lookup.loc[(instance_id, int(seed))])
            total_overhead = float(part.ni_overhead_ms.sum())
            runtime_rows.append({
                "method": method,
                "instance_id": instance_id,
                "seed": int(seed),
                "eligible_decisions": len(part),
                "interventions": int(part.ni_intervention.sum()),
                "fallbacks": int(part.fallback.sum()),
                "solver_runtime_seconds": solver_runtime,
                "total_decision_overhead_ms": total_overhead,
                "decision_overhead_fraction": total_overhead / (1000.0 * solver_runtime),
                "total_csg_build_ms": float(part.csg_build_ms.sum()),
                "total_model_inference_ms": float(part.model_inference_ms.sum()),
                "total_calibration_gate_ms": float(part.calibration_gate_ms.sum()),
                "mean_decision_overhead_ms": float(part.ni_overhead_ms.mean()),
            })
        if method == "PHASE6H_CSGNI":
            phase6h_live.append(live[["scale", "CF_level", *FEATURES]])
    calibration = pd.DataFrame(calibration_rows)
    atomic_csv(calibration, STATISTICS / "holdout_calibration_summary.csv")
    atomic_csv(pd.concat(reliability_rows, ignore_index=True), STATISTICS / "holdout_reliability_bins.csv")
    atomic_csv(pd.DataFrame(intervention_rows), STATISTICS / "intervention_diagnostics.csv")
    runtime = pd.DataFrame(runtime_rows)
    runtime_summary = runtime.groupby("method").agg(
        run_count=("seed", "size"),
        mean_eligible_decisions=("eligible_decisions", "mean"),
        mean_interventions=("interventions", "mean"),
        mean_fallbacks=("fallbacks", "mean"),
        mean_solver_runtime_seconds=("solver_runtime_seconds", "mean"),
        mean_total_decision_overhead_ms=("total_decision_overhead_ms", "mean"),
        mean_decision_overhead_fraction=("decision_overhead_fraction", "mean"),
        mean_total_csg_build_ms=("total_csg_build_ms", "mean"),
        mean_total_model_inference_ms=("total_model_inference_ms", "mean"),
        mean_total_calibration_gate_ms=("total_calibration_gate_ms", "mean"),
        mean_per_decision_overhead_ms=("mean_decision_overhead_ms", "mean"),
    ).reset_index()
    atomic_csv(runtime, STATISTICS / "csgni_runtime_efficiency_by_run.csv")
    atomic_csv(runtime_summary, STATISTICS / "csgni_runtime_efficiency_summary.csv")

    live_states = pd.concat(phase6h_live, ignore_index=True)
    reference = pd.read_parquet(
        ROOT / "outputs/phase6g/drift_audit/phase6c_training_state_reference.parquet"
    )
    drift = compare(reference, live_states)
    atomic_csv(drift, STATISTICS / "live_state_drift_summary.csv")
    overall = drift[drift.group_type == "overall"]
    order = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    classification = max(overall.severity, key=order.__getitem__)
    return calibration, classification


def main() -> None:
    config = load_json(CONFIG_PATH)
    progress = load_json(OUT / "progress.json")
    if progress.get("status") != "COMPLETE":
        raise RuntimeError("Phase 6H CAL-HOLDOUT validation is incomplete")
    runs = pd.read_csv(OUT / "validation_run_summary.csv")
    expected = 9 + len(config["seeds"]["CAL_HOLDOUT"]) * 9 * (len(METHODS) - 1)
    if len(runs) != expected or set(runs.method) != set(METHODS):
        raise RuntimeError("Phase 6H validation run table is incomplete")
    if not runs.feasible.all():
        raise RuntimeError("at least one Phase 6H validation schedule is infeasible")
    instances = final_quality_statistics(runs, config)
    anytime = anytime_statistics(runs, config)
    calibration, drift = calibration_and_drift(config, runs)
    atomic_json({
        "schema": "phase6h-validation-analysis-integrity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "run_count": len(runs),
        "instance_count": runs.instance_id.nunique(),
        "method_count": runs.method.nunique(),
        "all_schedules_feasible": True,
        "all_incumbent_traces_validated": True,
        "common_reference_type": "POOLED_BKS",
        "calibration_methods_evaluated": calibration.method.tolist(),
        "live_state_drift": drift,
        "cal_holdout_used_for_selection": False,
    }, AUDIT / "analysis_integrity.json")
    print(
        f"PHASE6H_VALIDATION_ANALYSIS_COMPLETE runs={len(runs)} "
        f"instances={instances.instance_id.nunique()} drift={drift} "
        f"anytime_methods={len(anytime)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
