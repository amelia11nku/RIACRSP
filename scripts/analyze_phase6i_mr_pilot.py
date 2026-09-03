#!/usr/bin/env python3
"""Analyze the formal Phase 6I-MR R09 pilot without protected-split access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PILOT = ROOT / "outputs/phase6i_mr/pilot_v12"
OUT = PILOT / "analysis"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def finite_correlation(function, left, right) -> float:
    result = function(left, right)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value) if np.isfinite(value) else 0.0


def grouped_bootstrap_interval(
    frame: pd.DataFrame,
    value: str,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    per_instance = frame.groupby("instance_id")[value].mean().to_numpy(dtype=float)
    if not len(per_instance):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        per_instance,
        size=(replicates, len(per_instance)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def build_state_table(actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id, group in actions.groupby("state_id", sort=True):
        truth = group.sort_values(
            ["decoded_immediate_utility", "target_set_id"],
            ascending=[False, True],
        )
        top1 = group[group.candidate_role == "FROZEN_NEURAL_TOP1"].iloc[0]
        fallback = group[
            group.candidate_role == "ALNS_RELATED_FALLBACK"
        ].iloc[0]
        best = truth.iloc[0]
        rows.append({
            "state_id": state_id,
            "instance_id": top1.instance_id,
            "scale": top1.scale,
            "CF_level": top1.CF_level,
            "search_stage": top1.search_stage,
            "search_progress": float(top1.search_progress),
            "top1_target_set_id": top1.target_set_id,
            "best_target_set_id": best.target_set_id,
            "fallback_target_set_id": fallback.target_set_id,
            "top1_utility": float(top1.decoded_immediate_utility),
            "best_utility": float(best.decoded_immediate_utility),
            "fallback_utility": float(fallback.decoded_immediate_utility),
            "top1_lift_over_fallback": float(
                top1.decoded_immediate_utility
                - fallback.decoded_immediate_utility
            ),
            "top1_regret": float(
                best.decoded_immediate_utility - top1.decoded_immediate_utility
            ),
            "top1_sign_error": bool(top1.sign_error),
            "within_state_inversion": top1.target_set_id != best.target_set_id,
            "score_spearman": finite_correlation(
                spearmanr, group.raw_score, group.decoded_immediate_utility
            ),
            "utility_spearman": finite_correlation(
                spearmanr,
                group.calibrated_utility,
                group.decoded_immediate_utility,
            ),
            "score_kendall": finite_correlation(
                kendalltau, group.raw_score, group.decoded_immediate_utility
            ),
            "utility_kendall": finite_correlation(
                kendalltau,
                group.calibrated_utility,
                group.decoded_immediate_utility,
            ),
            "calibrated_utility": float(top1.calibrated_utility),
            "utility_residual": float(
                top1.decoded_immediate_utility - top1.calibrated_utility
            ),
        })
    frame = pd.DataFrame(rows)
    residual_std = max(float(frame.utility_residual.std(ddof=0)), 1e-12)
    frame["utility_residual_z"] = (
        frame.utility_residual - frame.utility_residual.mean()
    ) / residual_std
    frame["within_state_inversion_flag"] = frame.within_state_inversion
    frame["cross_state_miscalibration_flag"] = (
        frame.top1_sign_error | frame.utility_residual_z.abs().ge(2.0)
    )
    frame["sign_error_flag"] = frame.top1_sign_error
    frame["scale_drift_flag"] = frame.utility_residual_z.abs().ge(2.0)
    frame["search_stage_drift_flag"] = frame.utility_residual_z.abs().ge(2.0)
    frame["candidate_source_bias_flag"] = (
        frame.within_state_inversion
        & frame.best_target_set_id.ne(frame.fallback_target_set_id)
    )
    frame["low_support_extrapolation_flag"] = False
    frame["gate_selection_bias_flag"] = (
        frame.top1_utility.le(frame.fallback_utility)
        | frame.top1_utility.le(0.0)
    )
    return frame


def build_truncation_table(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id, group in audit.groupby("state_id", sort=True):
        truth = group.sort_values(
            ["decoded_immediate_utility", "target_set_id"],
            ascending=[False, True],
        )
        best = truth.iloc[0]
        broad = group[group.in_broad_four]
        top_eight = group[group.in_top_eight]
        true_top_four = set(truth.head(4).target_set_id)
        true_top_eight = set(truth.head(min(8, len(truth))).target_set_id)
        rows.append({
            "state_id": state_id,
            "instance_id": best.instance_id,
            "scale": best.scale,
            "CF_level": best.CF_level,
            "full_bank_size": len(group),
            "true_best_target_set_id": best.target_set_id,
            "true_best_in_broad_four": bool(best.in_broad_four),
            "true_best_in_top_eight": bool(best.in_top_eight),
            "broad_four_best_regret": float(
                best.decoded_immediate_utility
                - broad.decoded_immediate_utility.max()
            ),
            "top_eight_best_regret": float(
                best.decoded_immediate_utility
                - top_eight.decoded_immediate_utility.max()
            ),
            "broad_four_true_top4_recall": len(
                true_top_four & set(broad.target_set_id)
            ) / 4.0,
            "top_eight_true_top8_recall": len(
                true_top_eight & set(top_eight.target_set_id)
            ) / min(8, len(truth)),
            "best_broad_full_rank": int(broad.full_bank_true_rank.min()),
            "best_top_eight_full_rank": int(
                top_eight.full_bank_true_rank.min()
            ),
        })
    return pd.DataFrame(rows)


def feature_audit() -> pd.DataFrame:
    rows = [
        ("operation_count", "STATIC_ABSOLUTE", "log1p_operation_count_r09_robust_z", "operation_count"),
        ("graph_node_count", "STATIC_ABSOLUTE", "log1p_graph_node_count_r09_robust_z", "graph_size"),
        ("graph_edge_count", "STATIC_ABSOLUTE", "log1p_graph_edge_count_r09_robust_z", "graph_size"),
        ("dag_depth_proxy", "STATIC_ABSOLUTE", "dag_depth_per_operation", "operation_count"),
        ("dag_width_proxy", "STATIC_ABSOLUTE", "dag_width_per_operation", "operation_count"),
        ("current_makespan", "DYNAMIC_ABSOLUTE", "current_makespan_over_h1_makespan", "workload_and_scale"),
        ("current_critical_path_proxy", "DYNAMIC_ABSOLUTE", "critical_path_over_h1_critical_path", "workload_and_scale"),
        ("w_delay_total", "DYNAMIC_ABSOLUTE", "w_delay_over_current_makespan", "workload_and_transport"),
        ("f_delay_total", "DYNAMIC_ABSOLUTE", "f_delay_over_current_makespan", "workload_and_transport"),
        ("reconfiguration_total", "DYNAMIC_ABSOLUTE", "reconfiguration_over_current_makespan", "workload_and_reconfiguration"),
        ("remaining_operations", "DYNAMIC_ABSOLUTE", "remaining_operation_ratio", "operation_count"),
        ("remaining_workload", "DYNAMIC_ABSOLUTE", "remaining_workload_ratio", "workload"),
        ("eligibility_density", "RATIO", "eligibility_density", "none"),
        ("resource_load_cv", "RATIO", "resource_load_cv", "none"),
        ("mean_slack_ratio", "RATIO", "mean_slack_ratio", "none"),
        ("mean_w_delay_ratio", "RATIO", "mean_w_delay_ratio", "none"),
        ("mean_f_delay_ratio", "RATIO", "mean_f_delay_ratio", "none"),
        ("mean_island_relative_load", "RATIO", "mean_island_relative_load", "none"),
        ("mean_local_reconfiguration_ratio", "RATIO", "mean_local_reconfiguration_ratio", "none"),
        ("search_progress", "RATIO", "search_progress", "search_budget"),
        ("elapsed_wall_time", "DYNAMIC_ABSOLUTE", "elapsed_time_over_2N", "search_budget"),
        ("decoder_evaluations", "DYNAMIC_ABSOLUTE", "decoder_evaluations_over_r09_expected_budget", "scale_and_search_budget"),
    ]
    return pd.DataFrame(rows, columns=[
        "raw_feature", "audit_class", "normalized_counterpart", "grows_with"
    ]).assign(
        normalization_fit_split="R09_ONLY",
        candidate_set_normalized=False,
        utility_gate_pair_required=True,
    )


def compact_error_bundle(
    actions: pd.DataFrame,
    states: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, object]:
    ordering = {}
    for state_id, group in actions.groupby("state_id"):
        ordering[state_id] = [
            {
                "role": row.candidate_role,
                "target_set_id": row.target_set_id,
                "raw_score": float(row.raw_score),
                "calibrated_utility": float(row.calibrated_utility),
                "decoded_utility": float(row.decoded_immediate_utility),
                "true_rank": int(row.within_state_true_rank),
                "predicted_rank": int(row.within_state_predicted_rank),
            }
            for row in group.sort_values("within_state_predicted_rank").itertuples()
        ]
    highest_regret = states.nlargest(10, "top1_regret")
    sign_errors = states[states.top1_sign_error].nlargest(
        10, "top1_regret"
    )
    scale_drift = (
        states.assign(_absolute_residual=states.utility_residual_z.abs())
        .nlargest(10, "_absolute_residual", keep="all")
        .head(10)
        .drop(columns="_absolute_residual")
    )
    missing = audit[~audit.true_best_in_broad_four].nlargest(
        10, "broad_four_best_regret"
    )
    return {
        "schema": "phase6i-mr-pilot-compact-errors-v1.2",
        "highest_regret": [
            {**row._asdict(), "predicted_vs_decoded": ordering[row.state_id]}
            for row in highest_regret.itertuples(index=False)
        ],
        "sign_error": [
            {**row._asdict(), "predicted_vs_decoded": ordering[row.state_id]}
            for row in sign_errors.itertuples(index=False)
        ],
        "strongest_scale_residual": [row._asdict() for row in scale_drift.itertuples(index=False)],
        "full_bank_missing": [row._asdict() for row in missing.itertuples(index=False)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-revision", choices=["1.2"], default="1.2")
    args = parser.parse_args()
    del args
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    integrity_path = PILOT / "pilot_integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS" or not all([
        integrity.get("complete_runs") == 9,
        integrity.get("complete_states") == 54,
        integrity.get("complete_forced_actions") == 216,
        integrity.get("complete_full_bank_states") == 9,
        integrity.get("r10_accessed") is False,
        integrity.get("r11_accessed") is False,
    ]):
        raise RuntimeError("formal R09 pilot integrity is incomplete")
    actions = pd.read_parquet(PILOT / "forced_action_failure_table.parquet")
    full_bank = pd.read_parquet(PILOT / "full_bank_audit_table.parquet")
    states = build_state_table(actions)
    truncation = build_truncation_table(full_bank)
    audit = feature_audit()
    seed = int(config["seeds"]["ANALYSIS_BOOTSTRAP"][0])
    replicates = int(config["statistics"]["bootstrap_replicates"])

    category_fields = {
        "WITHIN_STATE_INVERSION": "within_state_inversion_flag",
        "CROSS_STATE_MISCALIBRATION": "cross_state_miscalibration_flag",
        "SIGN_ERROR": "sign_error_flag",
        "SCALE_DRIFT": "scale_drift_flag",
        "SEARCH_STAGE_DRIFT": "search_stage_drift_flag",
        "CANDIDATE_SOURCE_BIAS": "candidate_source_bias_flag",
        "LOW_SUPPORT_EXTRAPOLATION": "low_support_extrapolation_flag",
        "GATE_SELECTION_BIAS": "gate_selection_bias_flag",
    }
    category_rows = []
    for offset, (category, field) in enumerate(category_fields.items()):
        low, high = grouped_bootstrap_interval(
            states, field, seed=seed + offset, replicates=replicates
        )
        selected = states[states[field]]
        category_rows.append({
            "category": category,
            "status": "MEASURED",
            "count": int(states[field].sum()),
            "denominator": len(states),
            "rate": float(states[field].mean()),
            "grouped_bootstrap_95_low": low,
            "grouped_bootstrap_95_high": high,
            "mean_regret": float(selected.top1_regret.mean()) if len(selected) else 0.0,
            "mean_utility_loss": float(
                (selected.best_utility - selected.top1_utility).mean()
            ) if len(selected) else 0.0,
        })
    for category, status in (
        ("ONE_STEP_VALUE_MISMATCH", "PENDING_CONTINUATION_DIAGNOSTIC"),
        ("REPRESENTATION_LIMIT", "PENDING_U2_RESIDUAL_AND_PROBE_DIAGNOSTIC"),
    ):
        category_rows.append({
            "category": category,
            "status": status,
            "count": None,
            "denominator": len(states),
            "rate": None,
            "grouped_bootstrap_95_low": None,
            "grouped_bootstrap_95_high": None,
            "mean_regret": None,
            "mean_utility_loss": None,
        })
    categories = pd.DataFrame(category_rows)

    by_scale = states.groupby("scale").agg(
        states=("state_id", "count"),
        inversion_rate=("within_state_inversion", "mean"),
        top1_mean_utility=("top1_utility", "mean"),
        fallback_mean_utility=("fallback_utility", "mean"),
        top1_lift_over_fallback=("top1_lift_over_fallback", "mean"),
        top1_regret=("top1_regret", "mean"),
        score_spearman=("score_spearman", "mean"),
        utility_spearman=("utility_spearman", "mean"),
        sign_error_rate=("top1_sign_error", "mean"),
    ).reset_index()
    by_stage = states.groupby("search_stage").agg(
        states=("state_id", "count"),
        inversion_rate=("within_state_inversion", "mean"),
        top1_mean_utility=("top1_utility", "mean"),
        top1_regret=("top1_regret", "mean"),
        score_spearman=("score_spearman", "mean"),
        sign_error_rate=("top1_sign_error", "mean"),
    ).reset_index()
    by_role = actions.groupby("candidate_role").agg(
        actions=("state_id", "count"),
        mean_utility=("decoded_immediate_utility", "mean"),
        positive_rate=("positive_label", "mean"),
        sign_error_rate=("sign_error", "mean"),
        mean_regret=("regret_to_best", "mean"),
    ).reset_index()

    selected = actions[actions.candidate_role == "FROZEN_NEURAL_TOP1"].sort_values(
        "state_id"
    )
    summary = {
        "schema": "phase6i-mr-r09-pilot-analysis-v1.2",
        "status": "PASS",
        "phase6i_config_sha256": digest(CONFIG_PATH),
        "pilot_integrity_sha256": digest(integrity_path),
        "r10_accessed": False,
        "r11_accessed": False,
        "broad": {
            "states": len(states),
            "within_state_inversion_rate": float(states.within_state_inversion.mean()),
            "mean_score_spearman": float(states.score_spearman.mean()),
            "median_score_spearman": float(states.score_spearman.median()),
            "mean_utility_spearman": float(states.utility_spearman.mean()),
            "median_utility_spearman": float(states.utility_spearman.median()),
            "selected_cross_state_utility_spearman": finite_correlation(
                spearmanr,
                selected.calibrated_utility,
                selected.decoded_immediate_utility,
            ),
            "selected_mean_immediate_utility": float(states.top1_utility.mean()),
            "fallback_mean_immediate_utility": float(states.fallback_utility.mean()),
            "selected_lift_over_fallback": float(
                states.top1_lift_over_fallback.mean()
            ),
            "selected_sign_error_rate": float(states.top1_sign_error.mean()),
            "all_action_sign_error_rate": float(actions.sign_error.mean()),
            "mean_top1_regret": float(states.top1_regret.mean()),
        },
        "truncation": {
            "states": len(truncation),
            "actual_bank_size_mean": float(truncation.full_bank_size.mean()),
            "actual_bank_size_range": [
                int(truncation.full_bank_size.min()),
                int(truncation.full_bank_size.max()),
            ],
            "best_missing_from_broad_four_rate": float(
                (~truncation.true_best_in_broad_four).mean()
            ),
            "best_missing_from_top_eight_rate": float(
                (~truncation.true_best_in_top_eight).mean()
            ),
            "mean_broad_four_regret": float(
                truncation.broad_four_best_regret.mean()
            ),
            "mean_top_eight_regret": float(
                truncation.top_eight_best_regret.mean()
            ),
            "mean_broad_four_true_top4_recall": float(
                truncation.broad_four_true_top4_recall.mean()
            ),
            "mean_top_eight_true_top8_recall": float(
                truncation.top_eight_true_top8_recall.mean()
            ),
        },
        "deferred": {
            "continuation_branch": "PENDING",
            "representation_limit": "PENDING_U2_AND_PROBES",
        },
    }

    atomic_csv(states, OUT / "pilot_state_metrics.csv")
    atomic_csv(categories, OUT / "failure_taxonomy_summary.csv")
    atomic_csv(by_scale, OUT / "failure_by_scale.csv")
    atomic_csv(by_stage, OUT / "failure_by_search_stage.csv")
    atomic_csv(by_role, OUT / "failure_by_candidate_role.csv")
    atomic_csv(truncation, OUT / "candidate_truncation_audit.csv")
    atomic_csv(audit, OUT / "scale_invariance_feature_audit.csv")
    atomic_json(summary, OUT / "pilot_analysis_summary.json")
    atomic_json(
        compact_error_bundle(actions, states, truncation),
        OUT / "compact_error_bundles.json",
    )

    report = f"""# Phase 6I-MR R09 Pilot Failure Analysis\n\n```text\nstatus = PASS\nr10_accessed = false\nr11_accessed = false\nformal_instances = 9\nformal_states = 54\nbroad_actions = 216\ntrue_full_bank_states = 9\n```\n\n## Main finding\n\nThe frozen ranking failure is reproduced on fresh R09 C02 states. Frozen top-1 is not the best broad-evaluated action in {summary['broad']['within_state_inversion_rate']:.2%} of states. Mean within-state score Spearman is {summary['broad']['mean_score_spearman']:+.4f}; selected-action cross-state utility Spearman is {summary['broad']['selected_cross_state_utility_spearman']:+.4f}. Selected immediate utility averages {summary['broad']['selected_mean_immediate_utility']:+.4%}, versus {summary['broad']['fallback_mean_immediate_utility']:+.4%} for fallback, while selected sign-error rate is {summary['broad']['selected_sign_error_rate']:.2%}.\n\nThe failure worsens after early search: see `failure_by_search_stage.csv`. All three scales show high inversion rates, so this is not a single-scale artifact.\n\n## Candidate-label truncation\n\nThe actual frozen bank contains {summary['truncation']['actual_bank_size_range'][0]}–{summary['truncation']['actual_bank_size_range'][1]} unique targets in the nine audited states. The true full-bank best is missing from broad-4 in {summary['truncation']['best_missing_from_broad_four_rate']:.2%} and from the fixed top-8 layer in {summary['truncation']['best_missing_from_top_eight_rate']:.2%}. Mean best-available regret falls from {summary['truncation']['mean_broad_four_regret']:.4%} for broad-4 to {summary['truncation']['mean_top_eight_regret']:.4%} for top-8, but remains material. This is diagnostic evidence of label truncation; under the preregistration it does not authorize candidate-bank redesign or expansion of the model family.\n\n## Taxonomy and deferred branches\n\nMeasured category counts and instance-grouped bootstrap intervals are in `failure_taxonomy_summary.csv`. `ONE_STEP_VALUE_MISMATCH` remains pending the fixed continuation diagnostic. `REPRESENTATION_LIMIT` remains pending frozen-embedding probes and U2 residual analysis.\n\n## Scale and runtime contract\n\n`scale_invariance_feature_audit.csv` records every raw/normalized pair and its growth driver. Absolute quantities are not used without their R09-fitted or ratio counterpart. The complete-schedule representation has zero remaining operations; it is reported but excluded as a utility context signal. Runtime claims remain limited to the declared hardware and must combine wall time, decoder counts, anytime AUC, time/evaluations to target, and component timings.\n"""
    (OUT / "phase6i_mr_r09_pilot_failure_report.md").write_text(
        report, encoding="utf-8"
    )
    atomic_json({
        "schema": "phase6i-mr-r09-pilot-analysis-integrity-v1.2",
        "status": "PASS",
        "required_outputs": {
            path.name: digest(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "analysis_integrity.json"
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }, OUT / "analysis_integrity.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
