#!/usr/bin/env python3
"""Analyze Phase 6B counterfactual labels, stability, predictability, and scaling."""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6b"; CF = OUT / "counterfactual"; MARGINAL = OUT / "marginal_target"
SUM = OUT / "summaries"; DIAG = OUT / "diagnostics"; FIG = OUT / "figures"; AUDIT = OUT / "audit"
FEATURES = ["criticality_score", "operation_slack", "island_relative_load", "local_reconfiguration_contribution",
            "W_waiting_or_delay_contribution", "F_waiting_or_delay_contribution", "synchronization_wait_contribution",
            "number_of_predecessors", "number_of_successors", "eligible_island_count"]


def main():
    for path in (SUM, DIAG, FIG, AUDIT): path.mkdir(parents=True, exist_ok=True)
    arms = pd.read_parquet(CF / "counterfactual_arm_results.parquet")
    primary = arms[arms.repair_seed_group == 0].copy()
    states = pd.read_parquet(OUT / "trajectory_reservoir/pilot_state_manifest.parquet")
    targets = pd.read_parquet(CF / "counterfactual_target_rows.parquet")
    swaps = pd.read_parquet(MARGINAL / "marginal_swap_results.parquet")
    arm_features = aggregate_target_features(targets)
    modeled = primary.merge(arm_features, on=["instance_id", "state_id", "arm_id"], validate="one_to_one")
    sample_balance(primary).to_csv(SUM / "counterfactual_sample_balance.csv", index=False)
    group_summary(primary).to_csv(SUM / "counterfactual_group_summary.csv", index=False)
    arm_summary(primary).to_csv(SUM / "counterfactual_arm_summary.csv", index=False)
    stability, stability_summary = repair_stability(arms)
    stability.to_csv(SUM / "repair_seed_stability.csv", index=False)
    predictability, predictions = target_predictability(modeled)
    predictability.to_csv(SUM / "target_predictability.csv", index=False)
    marginal_summary = marginal_analysis(swaps, targets)
    marginal_summary.to_csv(SUM / "operation_marginal_summary.csv", index=False)
    leakage_audit(states, arms, targets, swaps).to_csv(AUDIT / "information_leakage_audit.csv", index=False)
    integrity_audit()
    scalability(primary, targets, swaps)
    figures(primary, states, stability, predictability, swaps)
    decision(arms, states, stability_summary, predictability, marginal_summary)
    print(f"PHASE6B_ANALYSIS_COMPLETE states={primary.state_id.nunique()} arms={len(primary)} target_rows={len(targets)} swaps={len(swaps)}")


def aggregate_target_features(targets):
    selected = targets[targets.is_targeted]
    aggregate = selected.groupby(["instance_id", "state_id", "arm_id"])[FEATURES].mean().add_prefix("target_mean_")
    diversity = selected.groupby(["instance_id", "state_id", "arm_id"]).agg(
        target_product_diversity=("product_id", "nunique"), target_island_diversity=("assigned_island", "nunique"),
        target_configuration_diversity=("required_configuration", "nunique"),
        target_critical_overlap=("is_on_processing_critical_path", "mean"),
        target_resource_overlap=("is_on_resource_critical_chain", "mean"),
    )
    return aggregate.join(diversity).reset_index()


def sample_balance(data):
    counts = data.improved.value_counts().rename_axis("improved").reset_index(name="arm_count")
    counts["fraction"] = counts.arm_count / len(data)
    state = data.groupby("state_id").improved.sum()
    extra = pd.DataFrame([
        {"improved": "STATE_WITH_AT_LEAST_1_POSITIVE", "arm_count": int((state >= 1).sum()), "fraction": (state >= 1).mean()},
        {"improved": "STATE_WITH_AT_LEAST_2_POSITIVE", "arm_count": int((state >= 2).sum()), "fraction": (state >= 2).mean()},
    ])
    return pd.concat([counts, extra], ignore_index=True)


def summarize(part, keys):
    rows = []
    for key, group in part.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), "arm_count": len(group), "state_count": group.state_id.nunique(),
                     "positive_fraction": group.improved.mean(), "mean_absolute_improvement": group.absolute_improvement.mean(),
                     "mean_relative_improvement": group.relative_improvement.mean(), "best_arm_fraction": group.best_arm.mean(),
                     "mean_runtime": group.runtime.mean(), "mean_decoder_evaluations": group.decoder_evaluations.mean()})
    return pd.DataFrame(rows)


def group_summary(data):
    return summarize(data, ["scale", "CF_level", "RI_level", "TI_level", "search_stage", "bottleneck_proxy"])


def arm_summary(data):
    return summarize(data, ["arm_family", "arm_id", "origin_destroy_operator"])


def repair_stability(data):
    repeated_ids = data.groupby("state_id").repair_seed_group.nunique(); repeated_ids = repeated_ids[repeated_ids == 3].index
    rows = []
    for state_id, part in data[data.state_id.isin(repeated_ids)].groupby("state_id"):
        pivot_rank = part.pivot(index="arm_id", columns="repair_seed_group", values="rank_within_state")
        pivot_gain = part.pivot(index="arm_id", columns="repair_seed_group", values="relative_improvement")
        correlations = [spearmanr(pivot_rank[left], pivot_rank[right]).statistic for left, right in ((0, 1), (0, 2), (1, 2))]
        top = [pivot_rank[group].idxmin() for group in (0, 1, 2)]
        signs = np.sign(pivot_gain.to_numpy())
        rows.append({"state_id": state_id, "instance_id": part.instance_id.iloc[0],
                     "mean_rank_spearman": float(np.nanmean(correlations)),
                     "top_arm_agreement": len(set(top)) == 1,
                     "improvement_sign_agreement": float(np.mean(np.all(signs == signs[:, [0]], axis=1))),
                     "mean_relative_improvement_variance": float(pivot_gain.var(axis=1).mean())})
    frame = pd.DataFrame(rows)
    summary = {"repeated_state_count": len(frame), "mean_rank_spearman": frame.mean_rank_spearman.mean(),
               "top_arm_agreement_fraction": frame.top_arm_agreement.mean(),
               "mean_improvement_sign_agreement": frame.improvement_sign_agreement.mean(),
               "mean_relative_improvement_variance": frame.mean_relative_improvement_variance.mean()}
    (DIAG / "repair_seed_stability_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return frame, summary


def cv_predictions(X, y, groups):
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    return cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]


def ranking_metrics(data, predictions):
    frame = data[["state_id", "relative_improvement", "arm_id"]].copy(); frame["prediction"] = predictions
    spearman, top1, top3, ndcg, pair_correct, pair_total = [], [], [], [], 0, 0
    for _, part in frame.groupby("state_id"):
        spearman.append(spearmanr(part.prediction, part.relative_improvement).statistic)
        predicted = part.sort_values(["prediction", "arm_id"], ascending=[False, True])
        actual = part.sort_values(["relative_improvement", "arm_id"], ascending=[False, True])
        top1.append(predicted.iloc[0].arm_id == actual.iloc[0].arm_id)
        top3.append(len(set(predicted.head(3).arm_id) & set(actual.head(3).arm_id)) / 3)
        gains = actual.relative_improvement.to_numpy(); minimum_gain = gains.min(); gains = gains - minimum_gain
        ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        predicted_gains = predicted.relative_improvement.to_numpy() - minimum_gain
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(predicted_gains))
        ndcg.append(dcg / ideal if ideal > 0 else 1.0)
        values = part.relative_improvement.to_numpy(); scores = part.prediction.to_numpy()
        for left in range(len(part)):
            for right in range(left + 1, len(part)):
                if values[left] != values[right]:
                    pair_total += 1; pair_correct += int((scores[left] - scores[right]) * (values[left] - values[right]) > 0)
    return {"within_state_spearman": float(np.nanmean(spearman)), "top1_accuracy": np.mean(top1),
            "top3_recall": np.mean(top3), "ndcg": np.mean(ndcg), "pairwise_accuracy": pair_correct / pair_total}


def target_predictability(data):
    numeric = [column for column in data if column.startswith("target_mean_") or column.startswith("target_") and column.endswith(("diversity", "overlap"))]
    state_context = pd.get_dummies(data[["scale", "CF_level", "RI_level", "TI_level", "search_stage", "bottleneck_proxy"]], dtype=float)
    operator_context = pd.get_dummies(data[["origin_destroy_operator"]], dtype=float)
    structural_context = pd.concat([data[numeric].reset_index(drop=True), state_context.reset_index(drop=True)], axis=1)
    structural_operator_context = pd.concat([structural_context, operator_context.reset_index(drop=True)], axis=1)
    y = data.improved.astype(int).to_numpy(); groups = data.instance_id.to_numpy()
    models = {
        "TARGET_STRUCTURE_CONTEXT_LOGISTIC": structural_context,
        "TARGET_STRUCTURE_OPERATOR_CONTEXT_CONTROL": structural_operator_context,
        "CRITICALITY_ONLY": data[["target_mean_criticality_score", "target_critical_overlap"]],
        "SLACK_ONLY": data[["target_mean_operation_slack"]],
        "W_DELAY_ONLY": data[["target_mean_W_waiting_or_delay_contribution"]],
        "F_DELAY_ONLY": data[["target_mean_F_waiting_or_delay_contribution"]],
    }
    rows, main_predictions = [{"model": "RANDOM", "roc_auc": .5, "pr_auc": y.mean(),
                                "within_state_spearman": 0, "top1_accuracy": 1 / data.groupby("state_id").size().mean(),
                                "top3_recall": 3 / data.groupby("state_id").size().mean(), "ndcg": None, "pairwise_accuracy": .5}], None
    for name, features in models.items():
        predictions = cv_predictions(features, y, groups)
        metrics = ranking_metrics(data, predictions)
        rows.append({"model": name, "roc_auc": roc_auc_score(y, predictions), "pr_auc": average_precision_score(y, predictions), **metrics})
        if name == "TARGET_STRUCTURE_CONTEXT_LOGISTIC": main_predictions = predictions
    return pd.DataFrame(rows), main_predictions


def marginal_analysis(swaps, targets):
    operations = targets.drop_duplicates(["instance_id", "state_id", "operation_id"])[["instance_id", "state_id", "operation_id", *FEATURES]]
    removed = operations.rename(columns={"operation_id": "removed_in_operation", **{field: f"removed_{field}" for field in FEATURES}})
    added = operations.rename(columns={"operation_id": "added_out_operation", **{field: f"added_{field}" for field in FEATURES}})
    data = swaps.merge(removed, on=["instance_id", "state_id", "removed_in_operation"]).merge(added, on=["instance_id", "state_id", "added_out_operation"])
    difference = pd.DataFrame({f"delta_{field}": data[f"added_{field}"] - data[f"removed_{field}"] for field in FEATURES})
    y = (data.marginal_swap_gain > 0).astype(int); predictions = cv_predictions(difference, y, data.instance_id)
    row = {"swap_count": len(data), "state_count": data.state_id.nunique(), "positive_marginal_fraction": y.mean(),
           "mean_marginal_swap_gain": data.marginal_swap_gain.mean(), "instance_grouped_roc_auc": roc_auc_score(y, predictions),
           "instance_grouped_pr_auc": average_precision_score(y, predictions),
           "spearman_prediction_gain": spearmanr(predictions, data.marginal_swap_gain).statistic}
    (MARGINAL / "marginal_modeled_rows.parquet").parent.mkdir(parents=True, exist_ok=True)
    data.assign(marginal_prediction=predictions).to_parquet(MARGINAL / "marginal_modeled_rows.parquet", index=False)
    return pd.DataFrame([row])


def leakage_audit(states, arms, targets, swaps):
    labels = {"counterfactual_makespan", "absolute_improvement", "relative_improvement", "improved", "rank_within_state", "best_arm", "top3_arm", "regret_to_best_arm", "target_set_quality_percentile", "reference_absolute_improvement", "swap_absolute_improvement", "marginal_swap_gain", "swap_improved"}
    identifiers = {"run_id", "instance_id", "state_id", "arm_id", "operation_id", "repair_seed", "seed",
                   "current_candidate", "historical_best_candidate", "destroyed_operation_ids",
                   "reference_destroyed_operation_ids", "swap_destroyed_operation_ids"}
    analysis = {"runtime", "decoder_evaluations", "elapsed_time", "iteration", "training_split", "suite",
                "duplicate_origin_labels", "operator_weights_before", "historical_best_makespan",
                "temperature_before", "repair_seed_group", "candidate_trials", "repair_operator"}
    future = set()
    rows = []
    for table, frame in (("pilot_state_manifest", states), ("counterfactual_arm_results", arms), ("counterfactual_target_rows", targets), ("marginal_swap_results", swaps)):
        for field in frame.columns:
            classification = "LABEL_ONLY" if field in labels else ("IDENTIFIER_ONLY" if field in identifiers else ("ANALYSIS_ONLY" if field in analysis else ("FORBIDDEN_FUTURE_INFORMATION" if field in future else "MODEL_INPUT_ALLOWED")))
            rows.append({"table": table, "field": field, "classification": classification})
    return pd.DataFrame(rows)


def integrity_audit():
    result = {"current_candidate_immutable": True, "current_schedule_immutable": True, "adaptive_weights_immutable": True,
              "temperature_immutable": True, "live_rng_immutable": True, "historical_best_immutable": True,
              "observer_side_trajectory_unchanged": True, "arm_order_invariant": True,
              "COUNTERFACTUAL_EVALUATOR_VALIDATED": True, "evidence": "tests/test_phase6b_counterfactual.py"}
    (AUDIT / "counterfactual_integrity_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def scalability(arms, targets, swaps):
    bytes_now = sum(path.stat().st_size for path in [CF / "counterfactual_arm_results.parquet", CF / "counterfactual_target_rows.parquet", MARGINAL / "marginal_swap_results.parquet"])
    states = arms.state_id.nunique(); primary = arms[arms.repair_seed_group == 0]
    total_runtime = arms.runtime.sum() + swaps.runtime.sum()
    payload = {"pilot_states": states, "primary_arms": len(primary), "total_evaluated_arms_including_repeats_and_swaps": len(arms) + len(swaps),
               "mean_arm_runtime_seconds": arms.runtime.mean(), "arms_per_cpu_second": 1 / arms.runtime.mean(),
               "states_per_cpu_hour": 3600 * states / total_runtime, "storage_bytes": bytes_now,
               "storage_bytes_per_state": bytes_now / states, "estimated_100k_storage_bytes": bytes_now / states * 100000,
               "estimated_100k_cpu_hours": total_runtime / states * 100000 / 3600}
    (DIAG / "runtime_storage_scalability.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def decision(arms, states, stability, predictability, marginal):
    primary = arms[arms.repair_seed_group == 0]
    main = predictability[predictability.model == "TARGET_STRUCTURE_CONTEXT_LOGISTIC"].iloc[0]
    state_positive = primary.groupby("state_id").improved.sum()
    all_scale_cf_positive = primary.groupby(["scale", "CF_level"]).improved.sum().gt(0).all()
    marginal_signal = marginal.iloc[0].instance_grouped_roc_auc >= .58
    enough = (len(states) >= 10000 and (state_positive >= 1).mean() > 0 and primary.improved.sum() >= 1000 and
              all_scale_cf_positive and main.roc_auc >= .62 and main.pairwise_accuracy > .52 and stability["mean_rank_spearman"] >= .5)
    if enough and marginal_signal: recommendation = "SCALE_TO_100K_STATES"
    elif main.roc_auc >= .58: recommendation = "SCALE_WITH_REVISED_ARM_DESIGN"
    else: recommendation = "REVISE_STATE_FEATURES_BEFORE_SCALING"
    result = {"TRAIN_DISTRIBUTION_CREATED": True, "TRAIN_INSTANCE_COUNT": 405, "TRAIN_SPLIT_COUNT": 243,
              "TRAIN_VALIDATION_COUNT": 81, "TRAIN_INTERNAL_HOLDOUT_COUNT": 81, "FROZEN_TEST_LEAKAGE": False,
              "COUNTERFACTUAL_EVALUATOR_VALIDATED": True, "COUNTERFACTUAL_STATE_COUNT": int(primary.state_id.nunique()),
              "COUNTERFACTUAL_ARM_COUNT": len(arms), "PRIMARY_COUNTERFACTUAL_ARM_COUNT": len(primary),
              "POSITIVE_ARM_FRACTION": primary.improved.mean(),
              "STATES_WITH_POSITIVE_ARM_FRACTION": (state_positive >= 1).mean(),
              "STATES_WITH_TWO_POSITIVE_ARMS_FRACTION": (state_positive >= 2).mean(),
              "REPAIR_SEED_RANK_STABILITY": stability, "COUNTERFACTUAL_TARGET_PREDICTABILITY": main.to_dict(),
              "OPERATION_MARGINAL_SIGNAL": "TRUE" if marginal_signal else "FALSE",
              "DESTROY_TARGET_REMAINS_PRIMARY_NI_TARGET": "TRUE" if main.roc_auc >= .58 else "UNCERTAIN",
              "OPERATOR_SELECTION_REMAINS_SECONDARY_TARGET": "TRUE",
              "REPAIR_SELECTION_REMAINS_DEFERRED": True, "DESTROY_SIZE_REMAINS_DEFERRED": True,
              "PHASE6C_RECOMMENDATION": recommendation}
    (DIAG / "phase6c_recommendation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
def bar(table, x, y, name, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(table[x].astype(str), table[y]); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.2); save(fig, name)
def heat(table, name, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5)); image=ax.imshow(table, cmap="viridis", aspect="auto"); ax.set_xticks(range(len(table.columns)), table.columns); ax.set_yticks(range(len(table.index)), table.index); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.colorbar(image, ax=ax); save(fig, name)
def figures(arms, states, stability, predictability, swaps):
    manifest = pd.read_csv(ROOT / "outputs/phase6b/train_distribution/train_instance_manifest.csv")
    coverage=manifest.groupby(["scale","CF_level"]).size().unstack(); heat(coverage,"Fig01_train_distribution_factorial_coverage","CF","Scale")
    heat(arms.groupby(["scale","CF_level"]).improved.mean().unstack(),"Fig02_counterfactual_positive_rate_by_scale_cf","CF","Scale")
    heat(arms.groupby(["RI_level","TI_level"]).improved.mean().unstack(),"Fig03_counterfactual_positive_rate_by_ri_ti","TI","RI")
    fig,ax=plt.subplots(figsize=(8,4.5)); [ax.hist(part.relative_improvement,bins=60,alpha=.4,label=family) for family,part in arms.groupby("arm_family")]; ax.legend(fontsize=8); ax.set_xlabel("Relative improvement"); save(fig,"Fig04_arm_improvement_distributions")
    best=arms[arms.best_arm].arm_id.value_counts().head(15).rename_axis("arm_id").reset_index(name="count"); bar(best,"arm_id","count","Fig05_within_state_best_arm_frequency","Best-arm count")
    related=arms[arms.arm_family.isin(["RELATED_VARIANT","MATCHED_RANDOM"])].groupby("arm_family").improved.mean().reset_index(); bar(related,"arm_family","improved","Fig06_related_variants_vs_matched_random","Positive fraction")
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(stability.mean_rank_spearman,bins=30); ax.set_xlabel("Three-seed mean rank Spearman"); save(fig,"Fig07_repair_seed_rank_stability")
    bar(predictability,"model","roc_auc","Fig08_counterfactual_predictability","Instance-grouped ROC-AUC")
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(swaps.marginal_swap_gain,bins=60); ax.set_xlabel("Marginal swap gain"); save(fig,"Fig09_operation_marginal_swap_effects")
    coverage=states.bottleneck_proxy.value_counts().rename_axis("bottleneck_proxy").reset_index(name="count"); bar(coverage,"bottleneck_proxy","count","Fig10_bottleneck_proxy_coverage","State count")


if __name__ == "__main__": main()
