#!/usr/bin/env python3
"""Analyze the frozen Phase 6C dataset and produce diagnostics and figures."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from rcias_clgri.search.counterfactual import stable_seed

OUT = ROOT / "outputs/phase6c"
DATASET = OUT / "dataset"
SUMMARY = OUT / "summaries"
DIAGNOSTICS = OUT / "diagnostics"
FIGURES = OUT / "figures"
SPLIT_ORDER = ["TRAIN", "TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"]
NUMERIC_OPERATION_FEATURES = [
    "criticality_score", "operation_slack", "W_waiting_or_delay_contribution",
    "F_waiting_or_delay_contribution", "island_relative_load",
    "local_reconfiguration_contribution", "eligible_island_count",
    "synchronization_wait_contribution",
]


def json_safe(value):
    """Replace non-finite diagnostic metrics with JSON null values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def load_shards(filename: str, columns=None) -> pd.DataFrame:
    paths = sorted(DATASET.glob(f"*/*/{filename}"))
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def save_figure(fig, number: int, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"Fig{number:02d}_{name}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def grouped_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    result = frame.groupby(groups, dropna=False).agg(
        arm_count=("target_set_id", "size"), state_count=("state_id", "nunique"),
        positive_arm_fraction=("mean_relative_improvement", lambda values: values.gt(0).mean()),
        robust_positive_fraction=("positive_under_2_of_3", "mean"),
        mean_relative_improvement=("mean_relative_improvement", "mean"),
        mean_improvement_probability=("improvement_probability", "mean"),
        mean_regret=("regret_to_best", "mean"), top_arm_frequency=("top1", "mean"),
    ).reset_index()
    return result


def build_feature_caches() -> tuple[pd.DataFrame, pd.DataFrame]:
    target_cache = DIAGNOSTICS / "target_set_preaction_features.parquet"
    pair_cache = DIAGNOSTICS / "operation_pair_preaction_features.parquet"
    if target_cache.exists() and pair_cache.exists():
        return pd.read_parquet(target_cache), pd.read_parquet(pair_cache)
    target_frames, pair_frames = [], []
    for membership_path in sorted(DATASET.glob("*/*/target_membership.parquet")):
        membership = pd.read_parquet(membership_path)
        targeted = membership[membership.is_targeted]
        grouped = targeted.groupby(["state_id", "target_set_id"]).agg(
            target_mean_criticality=("criticality_score", "mean"),
            target_max_criticality=("criticality_score", "max"),
            target_mean_slack=("operation_slack", "mean"),
            target_min_slack=("operation_slack", "min"),
            target_mean_W_delay=("W_waiting_or_delay_contribution", "mean"),
            target_mean_F_delay=("F_waiting_or_delay_contribution", "mean"),
            target_mean_island_load=("island_relative_load", "mean"),
            target_mean_reconfiguration=("local_reconfiguration_contribution", "mean"),
            target_mean_eligible_islands=("eligible_island_count", "mean"),
            target_mean_sync_delay=("synchronization_wait_contribution", "mean"),
            target_critical_fraction=("is_on_processing_critical_path", "mean"),
            target_resource_critical_fraction=("is_on_resource_critical_chain", "mean"),
            target_product_diversity=("product_id", "nunique"),
            target_island_diversity=("assigned_island", "nunique"),
        ).reset_index()
        target_frames.append(grouped)

        base = membership.drop_duplicates(["state_id", "operation_id"])
        lookup = {
            (row.state_id, row.operation_id): {field: getattr(row, field) for field in NUMERIC_OPERATION_FEATURES}
            for row in base.itertuples(index=False)
        }
        pairs = pd.read_parquet(membership_path.parent / "operation_pairs.parquet")
        records = []
        for pair in pairs.itertuples(index=False):
            removed = json.loads(pair.removed_operations)
            added = json.loads(pair.added_operations)
            record = pair._asdict()
            for field in NUMERIC_OPERATION_FEATURES:
                removed_mean = float(np.mean([lookup[(pair.state_id, operation)][field] for operation in removed]))
                added_mean = float(np.mean([lookup[(pair.state_id, operation)][field] for operation in added]))
                record[f"delta_{field}"] = added_mean - removed_mean
            records.append(record)
        pair_frames.append(pd.DataFrame(records))
    target_features = pd.concat(target_frames, ignore_index=True)
    pair_features = pd.concat(pair_frames, ignore_index=True)
    atomic_write_parquet(target_features, target_cache)
    atomic_write_parquet(pair_features, pair_cache)
    return target_features, pair_features


def stability_summary(raw: pd.DataFrame, aggregates: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    state_ids = sorted(aggregates.state_id.unique(), key=lambda value: stable_seed(value, "stability_audit"))[:10000]
    aggregate_sample = aggregates[aggregates.state_id.isin(state_ids)]
    raw_sample = raw[raw.state_id.isin(state_ids)]
    aggregate_rank = aggregate_sample.set_index(["state_id", "target_set_id"])["rank_within_state"]
    aggregate_top = aggregate_sample.loc[aggregate_sample.groupby("state_id").rank_within_state.idxmin()].set_index("state_id").target_set_id
    aggregate_top3 = aggregate_sample[aggregate_sample.rank_within_state <= 3].groupby("state_id").target_set_id.apply(set)
    correlations, top1, top3 = [], [], []
    for (state_id, group), part in raw_sample.groupby(["state_id", "repair_seed_group"]):
        ranks = part.relative_improvement.rank(ascending=False, method="average")
        mean_ranks = part.target_set_id.map(lambda target: aggregate_rank.loc[(state_id, target)])
        correlation = spearmanr(ranks, mean_ranks).statistic
        if not math.isnan(correlation):
            correlations.append(correlation)
        seed_order = part.sort_values(["relative_improvement", "target_set_id"], ascending=[False, True])
        top1.append(seed_order.target_set_id.iloc[0] == aggregate_top.loc[state_id])
        top3.append(len(set(seed_order.target_set_id.head(3)) & aggregate_top3.loc[state_id]) / 3)
    joined = raw_sample.merge(
        aggregate_sample[["state_id", "target_set_id", "mean_relative_improvement"]],
        on=["state_id", "target_set_id"], how="left",
    )
    sign_agreement = (joined.relative_improvement.gt(0) == joined.mean_relative_improvement.gt(0)).mean()
    pair_sample = pairs[pairs.state_id.isin(state_ids)]
    indexed = raw_sample.set_index(["state_id", "target_set_id", "repair_seed_group"])["relative_improvement"]
    pair_agreements = []
    for pair in pair_sample.itertuples(index=False):
        for group in range(3):
            difference = indexed.loc[(pair.state_id, pair.perturbed_target_set_id, group)] - indexed.loc[(pair.state_id, pair.reference_target_set_id, group)]
            sign = 1 if difference > 0 else -1 if difference < 0 else 0
            pair_agreements.append(sign == pair.pairwise_preference)
    return pd.DataFrame([{
        "sampled_state_count": len(state_ids),
        "single_seed_vs_aggregate_rank_spearman": float(np.mean(correlations)),
        "single_seed_vs_aggregate_top1_agreement": float(np.mean(top1)),
        "single_seed_vs_aggregate_top3_overlap": float(np.mean(top3)),
        "single_seed_vs_aggregate_sign_agreement": float(sign_agreement),
        "pairwise_preference_stability": float(np.mean(pair_agreements)),
        "phase6b_rank_spearman": 0.6436327213707315,
        "phase6b_top1_agreement": 0.1739441660701503,
        "phase6b_sign_agreement": 0.9323642237028062,
    }])


def make_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    return Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(max_iter=300, class_weight="balanced", random_state=670001)),
    ])


def ranking_metrics(frame: pd.DataFrame, score_column: str, max_states: int = 10000) -> dict[str, float]:
    state_ids = sorted(frame.state_id.unique(), key=lambda value: stable_seed(value, "ranking_metrics"))[:max_states]
    spearman, ndcg, top1, top3, pairwise = [], [], [], [], []
    for _, part in frame[frame.state_id.isin(state_ids)].groupby("state_id"):
        if len(part) < 2:
            continue
        truth = part.mean_relative_improvement.to_numpy()
        score = part[score_column].to_numpy()
        correlation = spearmanr(truth, score).statistic
        if not math.isnan(correlation):
            spearman.append(correlation)
        relevance = truth - truth.min()
        ndcg.append(ndcg_score([relevance], [score]))
        true_best = int(np.argmax(truth))
        predicted_order = np.argsort(-score)
        top1.append(predicted_order[0] == true_best)
        top3.append(true_best in predicted_order[:3])
        concordant, comparable = 0, 0
        for left in range(len(part)):
            for right in range(left + 1, len(part)):
                truth_delta = truth[left] - truth[right]
                score_delta = score[left] - score[right]
                if truth_delta == 0:
                    continue
                comparable += 1
                concordant += truth_delta * score_delta > 0
        if comparable:
            pairwise.append(concordant / comparable)
    return {
        "within_state_spearman": float(np.mean(spearman)), "ndcg": float(np.mean(ndcg)),
        "top1_accuracy": float(np.mean(top1)), "top3_recall": float(np.mean(top3)),
        "pairwise_accuracy": float(np.mean(pairwise)), "ranking_state_count": len(state_ids),
    }


def classification_metrics(frame: pd.DataFrame, label: str, score: str) -> dict[str, float]:
    y = frame[label].astype(int)
    if y.nunique() < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "positive_fraction": float(y.mean())}
    return {"roc_auc": roc_auc_score(y, frame[score]), "pr_auc": average_precision_score(y, frame[score]),
            "positive_fraction": float(y.mean())}


def diagnostic_models(aggregates: pd.DataFrame, target_features: pd.DataFrame,
                      pair_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    modeled = aggregates.merge(target_features, on=["state_id", "target_set_id"], validate="one_to_one")
    categorical = ["scale", "CF_level", "RI_level", "TI_level", "search_stage", "bottleneck_proxy", "arm_family"]
    numeric = [
        "current_makespan", "destroy_count", "target_mean_criticality", "target_max_criticality",
        "target_mean_slack", "target_min_slack", "target_mean_W_delay", "target_mean_F_delay",
        "target_mean_island_load", "target_mean_reconfiguration", "target_mean_eligible_islands",
        "target_mean_sync_delay", "target_critical_fraction", "target_resource_critical_fraction",
        "target_product_diversity", "target_island_diversity",
    ]
    train = modeled[modeled.training_split == "TRAIN"]
    train_sample = train.sample(n=min(250000, len(train)), random_state=670001)
    target_model = make_pipeline(categorical, numeric)
    target_model.fit(train_sample[categorical + numeric], train_sample.positive_under_2_of_3)
    results = []
    scored_parts = []
    for split in ("TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"):
        part = modeled[modeled.training_split == split].copy()
        part["diagnostic_score"] = target_model.predict_proba(part[categorical + numeric])[:, 1]
        metrics = classification_metrics(part, "positive_under_2_of_3", "diagnostic_score") | ranking_metrics(part, "diagnostic_score")
        results.append({"model": "SET_STRUCTURE_CONTEXT_LOGISTIC", "evaluation_split": split,
                        "regime_dimension": "ALL", "regime_value": "ALL", **metrics})
        scored_parts.append(part)
        for dimension in ("scale", "CF_level", "RI_level", "TI_level"):
            for value, group in part.groupby(dimension):
                metrics = classification_metrics(group, "positive_under_2_of_3", "diagnostic_score")
                results.append({"model": "SET_STRUCTURE_CONTEXT_LOGISTIC", "evaluation_split": split,
                                "regime_dimension": dimension, "regime_value": value, **metrics})

    original = modeled[modeled.origin_rules.str.contains('"operator_')].copy()
    operator_categorical = ["scale", "CF_level", "RI_level", "TI_level", "search_stage", "bottleneck_proxy", "origin_destroy_operator"]
    operator_numeric = ["current_makespan"]
    operator_train = original[original.training_split == "TRAIN"].sample(
        n=min(200000, (original.training_split == "TRAIN").sum()), random_state=670002,
    )
    operator_model = make_pipeline(operator_categorical, operator_numeric)
    operator_model.fit(operator_train[operator_categorical + operator_numeric], operator_train.positive_under_2_of_3)
    for split in ("TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"):
        part = original[original.training_split == split].copy()
        part["diagnostic_score"] = operator_model.predict_proba(part[operator_categorical + operator_numeric])[:, 1]
        metrics = classification_metrics(part, "positive_under_2_of_3", "diagnostic_score") | ranking_metrics(part, "diagnostic_score")
        results.append({"model": "OPERATOR_CONTEXT_CONTROL", "evaluation_split": split,
                        "regime_dimension": "ALL", "regime_value": "ALL", **metrics})

    pair_frame = pair_features[pair_features.pairwise_preference != 0].copy()
    pair_frame["label"] = pair_frame.pairwise_preference.gt(0)
    pair_categorical = ["pair_rule"]
    pair_numeric = [f"delta_{field}" for field in NUMERIC_OPERATION_FEATURES]
    pair_train = pair_frame[pair_frame.training_split == "TRAIN"]
    pair_model = make_pipeline(pair_categorical, pair_numeric)
    pair_model.fit(pair_train[pair_categorical + pair_numeric], pair_train.label)
    for split in ("TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"):
        part = pair_frame[pair_frame.training_split == split].copy()
        part["diagnostic_score"] = pair_model.predict_proba(part[pair_categorical + pair_numeric])[:, 1]
        metrics = classification_metrics(part, "label", "diagnostic_score")
        metrics["accuracy"] = float(((part.diagnostic_score >= .5) == part.label).mean())
        results.append({"model": "CONDITIONAL_OPERATION_PAIR_LOGISTIC", "evaluation_split": split,
                        "regime_dimension": "ALL", "regime_value": "ALL", **metrics})
    summary = pd.DataFrame(results)
    holdout = summary[(summary.evaluation_split == "TRAIN_INTERNAL_HOLDOUT") & (summary.regime_dimension == "ALL")]
    return summary, {
        row.model: row._asdict() for row in holdout.itertuples(index=False)
    }


def figures(states: pd.DataFrame, aggregates: pd.DataFrame, cell: pd.DataFrame,
            arm: pd.DataFrame, stability: pd.DataFrame, predictability: pd.DataFrame,
            pairs: pd.DataFrame, shard_manifest: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4)); states.training_split.value_counts().reindex(SPLIT_ORDER).plot.bar(ax=ax); ax.set_ylabel("states"); save_figure(fig, 1, "state_coverage_by_split")
    fig, ax = plt.subplots(figsize=(8, 5)); cell.pivot_table(index=["scale", "CF_level"], columns="training_split", values="state_count").plot.bar(ax=ax); ax.set_ylabel("states"); save_figure(fig, 2, "structural_cell_coverage")
    fig, ax = plt.subplots(figsize=(8, 4)); arm.groupby("arm_family").positive_arm_fraction.mean().sort_values().plot.barh(ax=ax); ax.set_xlabel("positive fraction"); save_figure(fig, 3, "arm_family_positive_yield")
    row = stability.iloc[0]; fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Phase6B single", "Phase6C aggregate"], [row.phase6b_rank_spearman, row.single_seed_vs_aggregate_rank_spearman]); ax.set_ylabel("rank Spearman"); save_figure(fig, 4, "single_seed_vs_three_seed_rank_stability")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(aggregates.improvement_probability, bins=np.linspace(0, 1, 5)); ax.set_xlabel("improvement probability"); save_figure(fig, 5, "improvement_probability_distribution")
    diversity = aggregates.groupby("state_id").mean_relative_improvement.std(); fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(diversity, bins=40); ax.set_xlabel("within-state improvement std"); save_figure(fig, 6, "within_state_rank_diversity")
    value = arm[arm.arm_family.isin(["ORIGINAL_OPERATOR", "LOCAL_PERTURBATION", "STRUCTURED_NEAR_NEIGHBOR"])].groupby("arm_family").positive_arm_fraction.mean(); fig, ax = plt.subplots(figsize=(7, 4)); value.plot.bar(ax=ax); ax.set_ylabel("positive fraction"); save_figure(fig, 7, "local_swap_near_neighbor_value")
    overall = predictability[predictability.regime_dimension.eq("ALL") & predictability.model.eq("SET_STRUCTURE_CONTEXT_LOGISTIC")]; fig, ax = plt.subplots(figsize=(6, 4)); overall.set_index("evaluation_split")[["roc_auc", "pr_auc"]].plot.bar(ax=ax); save_figure(fig, 8, "predictability_by_split")
    regime = predictability[predictability.regime_dimension.isin(["scale", "CF_level"]) & predictability.evaluation_split.eq("TRAIN_INTERNAL_HOLDOUT")]; fig, ax = plt.subplots(figsize=(8, 4)); regime.plot.bar(x="regime_value", y="roc_auc", ax=ax); save_figure(fig, 9, "predictability_by_scale_cf")
    regime = predictability[predictability.regime_dimension.isin(["RI_level", "TI_level"]) & predictability.evaluation_split.eq("TRAIN_INTERNAL_HOLDOUT")]; fig, ax = plt.subplots(figsize=(8, 4)); regime.plot.bar(x="regime_value", y="roc_auc", ax=ax); save_figure(fig, 10, "predictability_by_ri_ti")
    fig, ax = plt.subplots(figsize=(7, 4)); pairs.groupby("pair_rule").pairwise_preference.apply(lambda values: values.ne(0).mean()).plot.bar(ax=ax); ax.set_ylabel("non-tie fraction"); save_figure(fig, 11, "operation_pair_signal")
    profile = shard_manifest.groupby("split").agg(state_count=("state_count", "sum"), runtime=("runtime_seconds", "sum")); fig, ax = plt.subplots(figsize=(6, 4)); profile.plot.scatter(x="state_count", y="runtime", ax=ax); save_figure(fig, 12, "compute_storage_profile")


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True); DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(OUT / "manifests/state_manifest.csv")
    aggregates = load_shards("target_set_aggregates.parquet")
    raw = load_shards("repair_seed_outcomes.parquet")
    pairs = load_shards("operation_pairs.parquet")
    overall = grouped_summary(aggregates.assign(dataset="ALL"), ["dataset"])
    overall["states_with_positive_arm_fraction"] = aggregates.groupby("state_id").mean_relative_improvement.max().gt(0).mean()
    overall["distinct_state_count"] = aggregates.state_id.nunique()
    split = grouped_summary(aggregates, ["training_split"])
    cell = grouped_summary(aggregates, ["training_split", "scale", "CF_level", "RI_level", "TI_level"])
    arm = grouped_summary(aggregates, ["training_split", "arm_family"])
    stability = stability_summary(raw, aggregates, pairs)
    pair_summary = pairs.groupby(["training_split", "pair_rule"]).agg(
        pair_count=("state_id", "size"), positive_preference_fraction=("pairwise_preference", lambda values: values.gt(0).mean()),
        tie_fraction=("pairwise_preference", lambda values: values.eq(0).mean()), mean_gain_difference=("mean_gain_difference", "mean"),
    ).reset_index()
    target_features, pair_features = build_feature_caches()
    predictability, holdout_models = diagnostic_models(aggregates, target_features, pair_features)
    for frame, name in ((overall, "dataset_overall_summary.csv"), (split, "split_summary.csv"),
                        (cell, "structural_cell_summary.csv"), (arm, "arm_family_summary.csv"),
                        (stability, "repair_seed_aggregation_summary.csv"),
                        (pair_summary, "operation_pair_summary.csv"), (predictability, "predictability_summary.csv")):
        atomic_write_csv(frame, SUMMARY / name)
    thresholds = json.loads((ROOT / "configs/phase6c_counterfactual.json").read_text())["readiness_thresholds"]
    target_holdout = holdout_models["SET_STRUCTURE_CONTEXT_LOGISTIC"]
    operator_holdout = holdout_models["OPERATOR_CONTEXT_CONTROL"]
    pair_holdout = holdout_models["CONDITIONAL_OPERATION_PAIR_LOGISTIC"]
    holdout_regimes = predictability[(predictability.model == "SET_STRUCTURE_CONTEXT_LOGISTIC") &
                                     (predictability.evaluation_split == "TRAIN_INTERNAL_HOLDOUT") &
                                     (predictability.regime_dimension != "ALL")]
    regime_signal = bool(holdout_regimes.positive_fraction.min() >= thresholds["minimum_structural_regime_positive_fraction"])
    set_ready = bool(target_holdout["roc_auc"] >= thresholds["minimum_set_level_roc_auc"] and
                     target_holdout["pairwise_accuracy"] >= thresholds["minimum_pairwise_accuracy"] and regime_signal)
    operator_ready = bool(operator_holdout["roc_auc"] >= thresholds["minimum_set_level_roc_auc"] and
                          operator_holdout["pairwise_accuracy"] >= thresholds["minimum_pairwise_accuracy"])
    noise_reduced = bool(stability.iloc[0].single_seed_vs_aggregate_rank_spearman >= thresholds["minimum_repair_aggregate_rank_spearman"] and
                         stability.iloc[0].single_seed_vs_aggregate_rank_spearman > stability.iloc[0].phase6b_rank_spearman)
    pair_metric = pair_holdout["accuracy"]
    pair_signal = "READY" if pair_metric >= thresholds["minimum_pairwise_accuracy"] else "WEAK"
    recommendation = "PROCEED_TO_CSG_DEFINITION" if set_ready and noise_reduced and regime_signal else "REVISE_COUNTERFACTUAL_ARMS"
    conclusions = {
        "SCALED_DATASET_COMPLETE": True, "DISTINCT_STATE_COUNT": int(aggregates.state_id.nunique()),
        "TRAIN_STATE_COUNT": int(states.training_split.eq("TRAIN").sum()),
        "VALIDATION_STATE_COUNT": int(states.training_split.eq("TRAIN_VALIDATION").sum()),
        "INTERNAL_HOLDOUT_STATE_COUNT": int(states.training_split.eq("TRAIN_INTERNAL_HOLDOUT").sum()),
        "THREE_REPAIR_SEED_AGGREGATION_COMPLETE": True,
        "REPAIR_NOISE_REDUCED_BY_AGGREGATION": "TRUE" if noise_reduced else "FALSE",
        "SET_LEVEL_TARGET_SIGNAL_READY": "TRUE" if set_ready else "FALSE",
        "OPERATOR_SELECTION_SIGNAL_READY": "TRUE" if operator_ready else "FALSE",
        "OPERATION_PAIR_SIGNAL": pair_signal, "DIRECT_OPERATION_SCALAR_LABEL_READY": False,
        "DESTROY_SIZE_REMAINS_DEFERRED": True, "REPAIR_SELECTION_REMAINS_DEFERRED": True,
        "DATASET_FROZEN": False, "PHASE6D_RECOMMENDATION": recommendation,
        "diagnostic_internal_holdout": holdout_models,
        "three_seed_stability": stability.iloc[0].to_dict(),
        "all_structural_regimes_have_signal": regime_signal,
    }
    conclusions = json_safe(conclusions)
    atomic_write_json(conclusions, DIAGNOSTICS / "phase6d_recommendation.json")
    shard_manifest = pd.read_csv(OUT / "manifests/shard_manifest.csv")
    figures(states, aggregates, cell, arm, stability, predictability, pairs, shard_manifest)
    print("PHASE6C_ANALYSIS_COMPLETE", json.dumps(conclusions, sort_keys=True))


if __name__ == "__main__":
    main()
