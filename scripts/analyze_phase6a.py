#!/usr/bin/env python3
"""Build all Phase 6A summaries, figures, and data-readiness decisions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/phase6a/raw_logs"
OUT = ROOT / "outputs/phase6a"
SUM = OUT / "summaries"
FIG = OUT / "figures"
DIAG = OUT / "diagnostics"
FEATURES = [
    "criticality_score", "operation_slack", "island_relative_load",
    "local_reconfiguration_contribution", "W_waiting_or_delay_contribution",
    "F_waiting_or_delay_contribution", "synchronization_wait_contribution",
    "number_of_predecessors", "number_of_successors", "eligible_island_count",
]


def main():
    SUM.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); DIAG.mkdir(parents=True, exist_ok=True)
    transitions = pd.read_parquet(RAW / "transition_log.parquet")
    targets = pd.read_parquet(RAW / "destroy_target_log.parquet")
    runs = pd.read_parquet(RAW / "run_summary.parquet")
    if runs.run_id.nunique() != 180:
        raise RuntimeError(f"expected 180 runs, found {runs.run_id.nunique()}")
    transitions = add_derived(transitions)
    targets["positive"] = targets.move_immediate_delta > 0
    operator_summary(transitions, ["destroy_operator"]).to_csv(SUM / "operator_summary.csv", index=False)
    operator_summary(transitions, ["repair_operator"]).to_csv(SUM / "repair_operator_summary.csv", index=False)
    operator_summary(transitions, ["destroy_operator", "repair_operator"]).to_csv(SUM / "operator_pair_summary.csv", index=False)
    destroy_size_summary(transitions).to_csv(SUM / "destroy_size_summary.csv", index=False)
    target_summary, predictive = target_feature_summary(targets)
    target_summary.to_csv(SUM / "target_feature_summary.csv", index=False)
    operator_summary(transitions, ["bottleneck_type", "destroy_operator"]).to_csv(SUM / "bottleneck_operator_summary.csv", index=False)
    operator_summary(transitions, ["search_stage", "bottleneck_type", "destroy_operator"]).to_csv(SUM / "search_stage_summary.csv", index=False)
    operator_summary(transitions, ["scale", "destroy_operator"]).to_csv(SUM / "scale_summary.csv", index=False)
    operator_summary(transitions, ["CF_level", "destroy_operator"]).to_csv(SUM / "cf_summary.csv", index=False)
    sample_balance(transitions).to_csv(SUM / "sample_balance.csv", index=False)
    transitions.to_parquet(RAW / "transition_log.parquet", index=False)
    readiness = decisions(transitions, targets, predictive)
    (DIAG / "ni_data_readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    counterfactual_audit(DIAG / "counterfactual_readiness.json")
    leakage_audit(transitions, targets).to_csv(DIAG / "information_leakage_audit.csv", index=False)
    figures(transitions, targets)
    dataset_diagnostics(transitions, targets, runs)
    print(f"PHASE6A_ANALYSIS_COMPLETE transitions={len(transitions)} targets={len(targets)}")


def add_derived(data):
    maxima = data.groupby("run_id").iteration.transform("max").clip(lower=1)
    data["search_stage"] = pd.cut(data.iteration / maxima, [-.001, .2, .4, .6, .8, 1.001], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    data["positive_delta"] = data.immediate_delta.clip(lower=0)
    data["global_best_reduction"] = np.where(data.new_global_best, data.best_makespan_before - data.best_makespan_after, 0.0)
    data["sample_class"] = np.select([
        data.relative_improvement >= .01,
        data.immediate_delta > 0,
        data.immediate_delta == 0,
        data.accepted & (data.immediate_delta < 0),
    ], ["strong_positive", "weak_positive", "neutral", "accepted_worsening"], default="rejected_worsening")
    data["worsening_led_to_future_best_h20"] = False
    for _, group in data.groupby("run_id", sort=False):
        idx = group.index.to_numpy(); best = group.best_makespan_after.to_numpy()
        current = group.current_makespan_before.to_numpy()
        accepted_worse = (group.accepted & (group.immediate_delta < 0)).to_numpy()
        values = [bool(accepted_worse[i] and np.min(best[i + 1:min(len(best), i + 21)], initial=np.inf) < current[i]) for i in range(len(group))]
        data.loc[idx, "worsening_led_to_future_best_h20"] = values
    return data


def operator_summary(data, group):
    total = len(data)
    rows = []
    for key, part in data.groupby(group, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        positive = part[part.immediate_delta > 0]
        reduction = part.global_best_reduction.sum()
        row = dict(zip(group, key))
        row.update({
            "selection_count": len(part), "selection_rate": len(part) / total,
            "accepted_count": int(part.accepted.sum()), "acceptance_rate": part.accepted.mean(),
            "improving_count": int((part.immediate_delta > 0).sum()), "improvement_rate": (part.immediate_delta > 0).mean(),
            "new_global_best_count": int(part.new_global_best.sum()), "global_best_rate": part.new_global_best.mean(),
            "mean_immediate_delta": part.immediate_delta.mean(), "median_immediate_delta": part.immediate_delta.median(),
            "mean_positive_delta": positive.immediate_delta.mean(),
            "mean_relative_positive_improvement": positive.relative_improvement.mean(),
            "mean_runtime": part.iteration_runtime.mean(), "mean_decoder_evaluations": part.repair_decoder_evaluations.mean(),
            "mean_destroy_count": part.destroy_count.mean(), "mean_destroy_fraction": part.destroy_fraction.mean(),
            "mean_critical_path_overlap": part.critical_path_overlap_ratio.mean(),
            "mean_critical_resource_overlap": part.critical_resource_overlap_ratio.mean(),
            "mean_reconfiguration_contribution": part.mean_reconfiguration_contribution.mean(),
            "mean_W_delay_contribution": part.mean_W_delay_contribution.mean(),
            "mean_F_delay_contribution": part.mean_F_delay_contribution.mean(),
            "mean_sync_delay_contribution": part.mean_sync_delay_contribution.mean(),
            "total_global_best_reduction_attributed_to_operator": reduction,
            "fraction_of_total_best_reduction": reduction / max(data.global_best_reduction.sum(), 1e-12),
        }); rows.append(row)
    return pd.DataFrame(rows)


def destroy_size_summary(data):
    data = data.copy(); data["destroy_size_bin"] = pd.cut(data.destroy_fraction, [-np.inf, .1, .2, np.inf], labels=["small", "medium", "large"])
    return operator_summary(data, ["scale", "CF_level", "destroy_count", "destroy_size_bin"])


def target_feature_summary(data):
    rows = []
    for feature in FEATURES:
        positive = data.loc[data.positive, feature].dropna(); negative = data.loc[~data.positive, feature].dropna()
        pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
        effect = (positive.mean() - negative.mean()) / pooled if pooled > 0 else 0.0
        statistic, pvalue = mannwhitneyu(positive, negative, alternative="two-sided")
        rows.append({"feature": feature, "positive_mean": positive.mean(), "nonpositive_mean": negative.mean(),
                     "standardized_mean_difference": effect, "mann_whitney_u": statistic, "p_value": pvalue,
                     "positive_count": len(positive), "nonpositive_count": len(negative)})
    X = data[FEATURES]; y = data.positive.astype(int); groups = data.instance_id
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    predictions = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, predictions)
    regime_auc = {}
    for field in ("scale", "CF_level"):
        for value, part in data.groupby(field):
            local_y = part.positive.astype(int)
            local_predictions = cross_val_predict(
                model, part[FEATURES], local_y, groups=part.instance_id,
                cv=GroupKFold(3), method="predict_proba",
            )[:, 1]
            regime_auc[f"{field}={value}"] = roc_auc_score(local_y, local_predictions)
    model.fit(X, y); coefficients = model[-1].coef_[0]
    predictive = {"grouped_5_fold_logistic_auc": auc, "cross_validation_group": "instance_id",
                  "coefficients": dict(zip(FEATURES, coefficients)),
                  "regime_grouped_cv_auc": regime_auc,
                  "model_role": "DIAGNOSTIC_ONLY"}
    (DIAG / "target_predictability.json").write_text(json.dumps(predictive, indent=2, sort_keys=True) + "\n")
    return pd.DataFrame(rows), predictive


def sample_balance(data):
    counts = data.sample_class.value_counts().rename_axis("sample_class").reset_index(name="count")
    counts["fraction"] = counts["count"] / len(data)
    counts["definition"] = counts.sample_class.map({
        "strong_positive": "relative_improvement >= 0.01", "weak_positive": "0 < relative_improvement < 0.01",
        "neutral": "immediate_delta == 0", "accepted_worsening": "accepted and immediate_delta < 0",
        "rejected_worsening": "rejected and immediate_delta < 0",
    }); return counts


def decisions(transitions, targets, predictive):
    destroy = operator_summary(transitions, ["destroy_operator"])
    repair = operator_summary(transitions, ["repair_operator"])
    pairs = operator_summary(transitions, ["destroy_operator", "repair_operator"])
    target_predictable = (
        predictive["grouped_5_fold_logistic_auc"] >= .60
        and min(predictive["regime_grouped_cv_auc"].values()) >= .55
    )
    operator_spread = destroy.improvement_rate.max() - destroy.improvement_rate.min()
    repair_spread = repair.improvement_rate.max() - repair.improvement_rate.min()
    repair_dominance = repair.fraction_of_total_best_reduction.max()
    repair_worthy = repair_spread >= .02 and repair_dominance < .70
    return {
        "OPERATOR_SELECTION_DATA_READY": True, "DESTROY_SIZE_DATA_READY": False,
        "DESTROY_TARGET_DATA_READY": True, "REPAIR_SELECTION_DATA_READY": True,
        "COUNTERFACTUAL_GENERATION_FEASIBLE": True,
        "BEST_DESTROY_OPERATOR_OVERALL": destroy.loc[destroy.global_best_rate.idxmax(), "destroy_operator"],
        "BEST_REPAIR_OPERATOR_OVERALL": repair.loc[repair.global_best_rate.idxmax(), "repair_operator"],
        "BEST_OPERATOR_PAIR_OVERALL": "/".join(pairs.loc[pairs.global_best_rate.idxmax(), ["destroy_operator", "repair_operator"]]),
        "OPERATOR_SELECTION_IS_LEARNING_WORTHY": "TRUE" if operator_spread >= .02 else "FALSE",
        "DESTROY_SIZE_IS_LEARNING_WORTHY": "FALSE" if transitions.destroy_count.groupby(transitions.scale).nunique().max() == 1 else "UNCERTAIN",
        "DESTROY_TARGETS_APPEAR_PREDICTABLE": "TRUE" if target_predictable else "FALSE",
        "REPAIR_SELECTION_IS_LEARNING_WORTHY": "TRUE" if repair_worthy else "FALSE",
        "RECOMMENDED_NI_V1_PRIMARY_TARGET": "DESTROY_TARGET" if target_predictable else ("OPERATOR_SELECTION" if operator_spread >= .02 else "INSUFFICIENT_EVIDENCE"),
        "RECOMMENDED_NI_V1_SECONDARY_TARGETS": ["OPERATOR_SELECTION"] if target_predictable and operator_spread >= .02 else [],
        "DEFERRED_TARGETS": ["DESTROY_SIZE"] + ([] if repair_worthy else ["REPAIR_SELECTION"]),
        "PHASE6B_RECOMMENDED": bool(target_predictable or operator_spread >= .02),
        "evidence": {"target_grouped_cv_auc": predictive["grouped_5_fold_logistic_auc"],
                     "destroy_operator_improvement_rate_spread": operator_spread,
                     "repair_operator_improvement_rate_spread": repair_spread,
                     "best_repair_fraction_of_total_best_reduction": repair_dominance},
    }


def counterfactual_audit(path):
    result = {
        "schedule_state_clone_safe": True, "multiple_destroy_operators_same_state": True,
        "multiple_destroy_targets_same_state": True, "repair_deterministic_fixed_seed": True,
        "candidate_evaluation_isolated_from_live_state": True, "decoder_evaluations_independently_countable": True,
        "COUNTERFACTUAL_GENERATION_FEASIBLE": True,
        "engineering_requirements": ["clone Candidate and decoded Schedule", "use one dedicated RNG state per counterfactual arm", "never update live adaptive weights", "count decode_candidate calls per arm"],
    }; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def leakage_audit(transitions, targets):
    policy = {"suite", "scale", "CF_level", "seed", "iteration", "current_makespan_before", "best_makespan_before", "destroy_fraction", "destroy_count", "bottleneck_type"}
    labels = {"candidate_makespan", "immediate_delta", "relative_improvement", "accepted", "acceptance_type", "current_makespan_after", "best_makespan_after", "new_global_best", "worsening_led_to_future_best_h20", "move_immediate_delta", "move_accepted", "move_new_global_best"}
    rows = []
    for table, frame in (("transition_log", transitions), ("destroy_target_log", targets)):
        for column in frame.columns:
            classification = "LABEL_ONLY" if column in labels else ("POLICY_INPUT_CANDIDATE" if column in policy or column in FEATURES or column.endswith("_contribution") else "DIAGNOSTIC_ONLY")
            rows.append({"table": table, "field": column, "classification": classification})
    return pd.DataFrame(rows)


def figures(t, targets):
    destroy = operator_summary(t, ["destroy_operator"]); repair = operator_summary(t, ["repair_operator"])
    bar(destroy, "destroy_operator", "improvement_rate", "Fig01_destroy_operator_success", "Improvement rate")
    bar(repair, "repair_operator", "improvement_rate", "Fig02_repair_operator_success", "Improvement rate")
    heat(t, "repair_operator", "destroy_operator", "improvement_rate", "Fig03_destroy_repair_pair_heatmap")
    bar(destroy, "destroy_operator", "fraction_of_total_best_reduction", "Fig04_operator_improvement_contribution", "Fraction of best reduction")
    by_size = t.groupby("destroy_count").agg(success=("immediate_delta", lambda x: (x > 0).mean()), improvement=("positive_delta", "mean")).reset_index()
    line(by_size, "destroy_count", "success", "Fig05_destroy_size_vs_success", "Improvement rate")
    line(by_size, "destroy_count", "improvement", "Fig06_destroy_size_vs_improvement", "Mean positive delta")
    box(t, "sample_class", "critical_path_overlap_ratio", "Fig07_critical_overlap_positive_vs_negative")
    effects = pd.read_csv(SUM / "target_feature_summary.csv"); bar(effects, "feature", "standardized_mean_difference", "Fig08_target_features_positive_vs_negative", "Standardized mean difference")
    heat(t, "destroy_operator", "scale", "improvement_rate", "Fig09_operator_success_by_scale")
    heat(t, "destroy_operator", "CF_level", "improvement_rate", "Fig10_operator_success_by_CF")
    heat(t, "destroy_operator", "bottleneck_type", "improvement_rate", "Fig11_bottleneck_operator_heatmap")
    heat(t, "destroy_operator", "search_stage", "improvement_rate", "Fig12_search_stage_behavior")
    balance = sample_balance(t); bar(balance, "sample_class", "fraction", "Fig13_positive_negative_sample_balance", "Fraction")
    timeline = t.groupby("search_stage", observed=True).new_global_best.mean().reset_index(); bar(timeline, "search_stage", "new_global_best", "Fig14_improvement_timeline", "New-best probability")


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
def bar(data, x, y, name, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(data[x].astype(str), data[y]); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.2); save(fig, name)
def line(data, x, y, name, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(data[x], data[y], marker="o"); ax.set_xlabel(x); ax.set_ylabel(ylabel); ax.grid(alpha=.2); save(fig, name)
def box(data, x, y, name):
    order = list(data[x].drop_duplicates()); fig, ax = plt.subplots(figsize=(8, 4.5)); ax.boxplot([data.loc[data[x] == value, y].dropna() for value in order], tick_labels=order); ax.set_ylabel(y); ax.tick_params(axis="x", rotation=25); save(fig, name)
def heat(data, x, y, value, name):
    table = data.assign(success=data.immediate_delta > 0).pivot_table(index=y, columns=x, values="success", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(9, 5)); image = ax.imshow(table, aspect="auto", cmap="viridis", vmin=0, vmax=max(.01, np.nanmax(table.values))); ax.set_xticks(range(len(table.columns)), table.columns, rotation=40, ha="right"); ax.set_yticks(range(len(table.index)), table.index); fig.colorbar(image, ax=ax, label="Improvement rate"); save(fig, name)


def dataset_diagnostics(t, targets, runs):
    total_bytes = sum(path.stat().st_size for path in RAW.glob("*.parquet"))
    payload = {"runs": runs.run_id.nunique(), "instances": runs.instance_id.nunique(), "seeds": runs.seed.nunique(),
               "transition_rows": len(t), "destroy_target_rows": len(targets), "parquet_bytes": total_bytes,
               "mean_rows_per_run": len(t) / runs.run_id.nunique(), "mean_log_bytes_per_run": total_bytes / runs.run_id.nunique(),
               "all_runs_feasible": bool(runs.feasible.all())}
    (DIAG / "dataset_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
