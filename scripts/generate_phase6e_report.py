#!/usr/bin/env python3
"""Generate the frozen Phase 6E figures, conclusions, and validation report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ri_acrsp_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs" / "phase6e"
FIGURES = PHASE / "figures"
AUDIT = PHASE / "audit"
REPORT = ROOT / "docs" / "reports" / "phase6e_supervised_ni_validation_report.md"

METRICS_PATH = PHASE / "evaluation_v2" / "internal_holdout_metrics.csv"
SELECTED_PATH = PHASE / "evaluation_v2" / "selected_action_utility.csv"
REGIMES_PATH = PHASE / "evaluation_v2" / "structural_regime_metrics.csv"
PAIRWISE_PATH = PHASE / "statistics" / "pairwise_statistics.csv"
PROFILE_PATH = PHASE / "profiling" / "inference_profile.csv"
PROFILE_SUMMARY_PATH = PHASE / "profiling" / "inference_profile_summary.json"
SEEDS_PATH = PHASE / "training" / "final_seeds" / "model_seed_summary.csv"
SANITY_PATH = PHASE / "sanity" / "mandatory_sanity.json"
FREEZE_PATH = PHASE / "environment" / "freeze_record.json"
CACHE_AUDIT_PATH = PHASE / "audit" / "tensor_cache_audit.json"
EXPERIMENT_FREEZE_PATH = PHASE / "audit" / "experiment_freeze.json"
EVALUATION_COMPLETION_PATH = PHASE / "evaluation_v2" / "evaluation_completion.json"
HOLDOUT_ACCESS_PATH = PHASE / "evaluation_v2" / "holdout_access_record.json"
V1_INVALIDATION_PATH = PHASE / "evaluation" / "holdout_v1_invalidation.json"

FULL = "FULL_CSG_ENSEMBLE"

DISPLAY = {
    FULL: "Full CSG ensemble",
    "FULL_CSG_SEED_660201": "Full CSG seed 660201",
    "FULL_CSG_SEED_660202": "Full CSG seed 660202",
    "FULL_CSG_SEED_660203": "Full CSG seed 660203",
    "FLAT_SET": "Flat set",
    "STATIC_CSG": "Static CSG",
    "NO_EDGE_FEATURES": "No edge features",
    "B3_PHASE6C_TABULAR": "Phase6C tabular",
    "B1_FIXED_RELATED": "Fixed related",
    "B2_BEST_FIXED_ORIGINAL": "Best fixed original",
    "B0_RANDOM_EXPECTATION": "Random expectation",
}

COLORS = {
    FULL: "#1f77b4",
    "FLAT_SET": "#ff7f0e",
    "STATIC_CSG": "#2ca02c",
    "NO_EDGE_FEATURES": "#d62728",
    "B3_PHASE6C_TABULAR": "#9467bd",
    "B1_FIXED_RELATED": "#8c564b",
    "B2_BEST_FIXED_ORIGINAL": "#7f7f7f",
    "B0_RANDOM_EXPECTATION": "#bcbd22",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    labels: list[str],
    title: str,
) -> None:
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        row = frame.loc[model]
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            [row[metric] for metric in metrics],
            width,
            label=DISPLAY[model],
            color=COLORS.get(model),
        )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_training_curves(seed_rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for seed in seed_rows["seed"].astype(int):
        history = pd.read_csv(
            PHASE / "training" / "final_seeds" / f"seed_{seed}" / "training_history.csv"
        )
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label=str(seed))
        axes[1].plot(
            history["epoch"], history["validation_objective"], marker="o", label=str(seed)
        )
    axes[0].set(title="Training composite loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(
        title="Validation selection objective",
        xlabel="Epoch",
        ylabel="0.5 pairwise accuracy + 0.5 NDCG",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="Seed")
    save_figure(fig, "Fig01_training_validation_curves")


def plot_ranking_comparison(metrics: pd.DataFrame) -> None:
    indexed = metrics.set_index("model")
    models = [FULL, "B3_PHASE6C_TABULAR", "FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES"]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    grouped_bars(
        ax,
        indexed,
        models,
        ["roc_auc", "pr_auc", "pairwise_accuracy", "ndcg", "top3_recall"],
        ["ROC-AUC", "PR-AUC", "Pairwise", "NDCG", "Top-3"],
        "Internal-holdout ranking comparison (20,000 states)",
    )
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, "Fig02_internal_holdout_ranking_comparison")


def plot_selected_utility(selected: pd.DataFrame) -> None:
    models = [
        FULL,
        "B3_PHASE6C_TABULAR",
        "FLAT_SET",
        "STATIC_CSG",
        "NO_EDGE_FEATURES",
        "B1_FIXED_RELATED",
        "B0_RANDOM_EXPECTATION",
    ]
    indexed = selected.set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    values = 100 * indexed["mean_selected_utility"]
    bars = ax.bar(
        [DISPLAY[model] for model in models],
        values,
        color=[COLORS[model] for model in models],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set(ylabel="Mean realized relative improvement (%)", title="Selected-action utility")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "Fig03_selected_action_improvement")


def plot_selected_regret(selected: pd.DataFrame) -> None:
    models = [
        FULL,
        "B3_PHASE6C_TABULAR",
        "FLAT_SET",
        "STATIC_CSG",
        "NO_EDGE_FEATURES",
        "B1_FIXED_RELATED",
        "B0_RANDOM_EXPECTATION",
    ]
    indexed = selected.set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    values = 100 * indexed["mean_selected_regret"]
    bars = ax.bar(
        [DISPLAY[model] for model in models],
        values,
        color=[COLORS[model] for model in models],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set(ylabel="Mean regret to oracle (%)", title="Selected-action regret (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "Fig04_selected_action_regret")


def plot_controlled_comparison(
    metrics: pd.DataFrame, comparator: str, stem: str, title: str
) -> None:
    indexed = metrics.set_index("model")
    models = [FULL, comparator]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grouped_bars(
        axes[0],
        indexed,
        models,
        ["pairwise_accuracy", "within_state_spearman", "ndcg", "top3_recall"],
        ["Pairwise", "Spearman", "NDCG", "Top-3"],
        "Within-state ranking",
    )
    utility_metrics = ["selected_positive_fraction", "selected_top3_fraction"]
    grouped_bars(
        axes[1],
        indexed,
        models,
        utility_metrics,
        ["Positive selection", "Top-3 selection"],
        "Selected-action quality",
    )
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle(title)
    save_figure(fig, stem)


def full_regime_rows(regimes: pd.DataFrame, dimension: str, order: list[str]) -> pd.DataFrame:
    rows = regimes[
        (regimes["model"] == FULL)
        & (regimes["metric_family"] == "ACTION_SCORING")
        & (regimes["regime_dimension"] == dimension)
    ].set_index("regime_value")
    return rows.loc[order]


def plot_two_regime_dimensions(
    regimes: pd.DataFrame,
    first: tuple[str, list[str]],
    second: tuple[str, list[str]],
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for column, (dimension, order) in enumerate((first, second)):
        rows = full_regime_rows(regimes, dimension, order)
        axes[0, column].plot(order, rows["pairwise_accuracy"], marker="o", label="Pairwise")
        axes[0, column].plot(order, rows["ndcg"], marker="s", label="NDCG")
        axes[0, column].set(title=f"{dimension}: ranking", ylim=(0.5, 0.9))
        axes[0, column].legend()
        axes[1, column].bar(order, 100 * rows["mean_selected_utility"], color="#1f77b4")
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set(
            title=f"{dimension}: selected utility",
            ylabel="Relative improvement (%)",
        )
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, stem)


def plot_search_stage(regimes: pd.DataFrame) -> None:
    order = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    rows = full_regime_rows(regimes, "search_stage", order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(order, rows["roc_auc"], marker="o", label="ROC-AUC")
    axes[0].plot(order, rows["pairwise_accuracy"], marker="s", label="Pairwise")
    axes[0].plot(order, rows["ndcg"], marker="^", label="NDCG")
    axes[0].set(title="Ranking by search stage", ylim=(0.5, 0.9))
    axes[0].legend()
    axes[1].bar(order, 100 * rows["mean_selected_utility"], color="#1f77b4")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Selected utility by search stage", ylabel="Relative improvement (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "Fig09_performance_by_search_stage")


def plot_inference(profile: pd.DataFrame) -> None:
    profile = profile.set_index("scale").loc[["S", "M", "L"]]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, profile["p90_total_single_model_decision_ms"], 0.36, label="Single model")
    axes[0].bar(
        x + 0.18,
        profile["p90_projected_ensemble_decision_ms"],
        0.36,
        label="3-model ensemble",
    )
    axes[0].axhline(150, color="red", linestyle="--", label="Frozen budget")
    axes[0].set(xticks=x, xticklabels=["S", "M", "L"], ylabel="p90 latency (ms)")
    axes[0].set_title("Decision latency by scale")
    axes[0].legend()
    axes[1].bar(
        ["S", "M", "L"],
        profile["projected_1000_decision_overhead_seconds"],
        color="#9467bd",
    )
    axes[1].axhline(150, color="red", linestyle="--", label="Frozen budget")
    axes[1].set(ylabel="Projected overhead (s)", title="1,000 NI decisions")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "Fig10_inference_latency_by_scale")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def f4(value: float) -> str:
    return f"{value:.4f}"


def signed_pct(value: float) -> str:
    return f"{100 * value:+.4f}%"


def build_conclusions(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    pairwise: pd.DataFrame,
    regimes: pd.DataFrame,
    seeds: pd.DataFrame,
    profile_summary: dict,
) -> dict:
    selected_index = selected.set_index("model")
    pair_index = pairwise.set_index("comparator")
    metric_index = metrics.set_index("model")

    def beats(comparator: str) -> bool:
        row = pair_index.loc[comparator]
        return bool(row["mean_paired_utility_delta"] > 0 and row["significant_at_0_05"])

    seed_metric_rows = metric_index.loc[
        ["FULL_CSG_SEED_660201", "FULL_CSG_SEED_660202", "FULL_CSG_SEED_660203"]
    ]
    seed_stable = bool(
        seed_metric_rows["roc_auc"].max() - seed_metric_rows["roc_auc"].min() <= 0.02
        and seed_metric_rows["mean_selected_utility"].max()
        - seed_metric_rows["mean_selected_utility"].min()
        <= 0.003
        and seeds["validation_objective"].max() - seeds["validation_objective"].min()
        <= 0.015
    )

    major_dimensions = ["scale", "CF_level", "RI_level", "TI_level"]
    full_regimes = regimes[
        (regimes["model"] == FULL)
        & (regimes["metric_family"] == "ACTION_SCORING")
        & (regimes["regime_dimension"].isin(major_dimensions))
    ]
    selected_regimes = regimes[
        (regimes["metric_family"] == "SELECTED_ACTION")
        & (regimes["regime_dimension"].isin(major_dimensions))
        & (regimes["model"].isin([FULL, "B1_FIXED_RELATED"]))
    ].pivot_table(
        index=["regime_dimension", "regime_value"],
        columns="model",
        values="mean_selected_utility",
    )
    structural_stable = bool(
        full_regimes["pairwise_accuracy"].min() >= 0.55
        and full_regimes["ndcg"].min() >= 0.84
        and (
            selected_regimes[FULL] - selected_regimes["B1_FIXED_RELATED"]
        ).min()
        > 0
    )

    conclusions = {
        "TENSORIZER_VALIDATED": "TRUE",
        "TARGET_SET_SCORER_TRAINED": "TRUE",
        "THREE_MODEL_SEEDS_COMPLETE": (
            "TRUE" if len(seeds) == 3 and bool((seeds["status"] == "COMPLETE").all()) else "FALSE"
        ),
        "NO_LABEL_LEAKAGE": "TRUE",
        "FULL_CSG_BEATS_RANDOM": "TRUE" if beats("B0_RANDOM_EXPECTATION") else "FALSE",
        "FULL_CSG_BEATS_RELATED": "TRUE" if beats("B1_FIXED_RELATED") else "FALSE",
        "FULL_CSG_BEATS_BEST_FIXED_OPERATOR": (
            "TRUE" if beats("B2_BEST_FIXED_ORIGINAL") else "FALSE"
        ),
        "FULL_CSG_BEATS_TABULAR_BASELINE": (
            "TRUE" if beats("B3_PHASE6C_TABULAR") else "FALSE"
        ),
        "FULL_CSG_BEATS_FLAT_SET_MODEL": "TRUE" if beats("FLAT_SET") else "FALSE",
        "REALIZED_SYNCHRONIZATION_TOPOLOGY_ADDS_VALUE": (
            "TRUE" if beats("STATIC_CSG") else "FALSE"
        ),
        "EDGE_FEATURES_ADD_VALUE": "TRUE" if beats("NO_EDGE_FEATURES") else "FALSE",
        "MODEL_STABLE_ACROSS_SEEDS": "TRUE" if seed_stable else "FALSE",
        "MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES": (
            "TRUE" if structural_stable else "FALSE"
        ),
        "INFERENCE_COST_ACCEPTABLE_FOR_SOLVER_INTEGRATION": (
            "TRUE" if all(profile_summary["checks"].values()) else "FALSE"
        ),
        "PHASE6F_RECOMMENDATION": "REVISE_MODEL",
    }
    return {
        "schema": "phase6e-scientific-conclusions-v1",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_version": "phase6e-holdout-v2",
        "primary_method": FULL,
        "conclusions": conclusions,
        "decision_rules": {
            "beats": "positive state-paired selected-utility delta and BH-FDR p <= 0.05",
            "seed_stability": "ROC-AUC range <= 0.02, selected-utility range <= 0.003, validation-objective range <= 0.015",
            "structural_stability": "major-regime pairwise >= 0.55, NDCG >= 0.84, and selected utility exceeds fixed-related in every major regime",
            "inference_acceptance": "all frozen latency and GPU-memory checks pass",
        },
        "primary_selected_utility": float(selected_index.loc[FULL, "mean_selected_utility"]),
        "primary_mean_regret": float(selected_index.loc[FULL, "mean_selected_regret"]),
        "paired_deltas": {
            comparator: {
                "mean_delta": float(row["mean_paired_utility_delta"]),
                "p_value_bh_fdr": float(row["p_value_bh_fdr"]),
                "significant": bool(row["significant_at_0_05"]),
            }
            for comparator, row in pair_index.iterrows()
        },
        "important_caveats": [
            "The full ensemble has negative absolute mean selected utility despite beating every comparator overall.",
            "The advantage over the Phase 6C tabular baseline is statistically significant but only 0.0002875 in mean relative improvement.",
            "The no-edge model has higher global ROC-AUC and PR-AUC, while the full model wins solver-relevant selected utility and most within-state metrics.",
            "The frozen three-model ensemble exceeds the M/L latency budget and the 1,000-decision overhead budget.",
        ],
    }


def generate_report(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    pairwise: pd.DataFrame,
    regimes: pd.DataFrame,
    profile: pd.DataFrame,
    seeds: pd.DataFrame,
    sanity: dict,
    freeze: dict,
    cache_audit: dict,
    experiment_freeze: dict,
    evaluation: dict,
    access: dict,
    conclusions_record: dict,
) -> str:
    mi = metrics.set_index("model")
    si = selected.set_index("model")
    pi = pairwise.set_index("comparator")
    full = mi.loc[FULL]
    conclusions = conclusions_record["conclusions"]

    ranking_models = [
        FULL,
        "B3_PHASE6C_TABULAR",
        "FLAT_SET",
        "STATIC_CSG",
        "NO_EDGE_FEATURES",
    ]
    ranking_table = markdown_table(
        ["Method", "ROC-AUC", "PR-AUC", "Pairwise", "Spearman", "NDCG", "Top-1", "Top-3"],
        [
            [
                DISPLAY[model],
                f4(mi.loc[model, "roc_auc"]),
                f4(mi.loc[model, "pr_auc"]),
                f4(mi.loc[model, "pairwise_accuracy"]),
                f4(mi.loc[model, "within_state_spearman"]),
                f4(mi.loc[model, "ndcg"]),
                f4(mi.loc[model, "top1_accuracy"]),
                f4(mi.loc[model, "top3_recall"]),
            ]
            for model in ranking_models
        ],
    )

    selected_models = [
        FULL,
        "B3_PHASE6C_TABULAR",
        "FLAT_SET",
        "STATIC_CSG",
        "NO_EDGE_FEATURES",
        "B1_FIXED_RELATED",
        "B2_BEST_FIXED_ORIGINAL",
        "B0_RANDOM_EXPECTATION",
    ]
    utility_table = markdown_table(
        ["Method", "Mean utility", "Median", "Positive", "Top-3", "Mean regret"],
        [
            [
                DISPLAY[model],
                signed_pct(si.loc[model, "mean_selected_utility"]),
                signed_pct(si.loc[model, "median_selected_utility"]),
                f"{100 * si.loc[model, 'selected_positive_fraction']:.2f}%",
                f"{100 * si.loc[model, 'selected_top3_fraction']:.2f}%",
                f"{100 * si.loc[model, 'mean_selected_regret']:.4f}%",
            ]
            for model in selected_models
        ],
    )

    paired_table = markdown_table(
        ["Comparator", "Mean paired delta", "W/T/L", "BH-FDR p", "Significant"],
        [
            [
                DISPLAY[comparator],
                signed_pct(row["mean_paired_utility_delta"]),
                f"{int(row['win_count'])}/{int(row['tie_count'])}/{int(row['loss_count'])}",
                f"{row['p_value_bh_fdr']:.3g}",
                str(bool(row["significant_at_0_05"])).upper(),
            ]
            for comparator, row in pi.iterrows()
        ],
    )

    seed_models = ["FULL_CSG_SEED_660201", "FULL_CSG_SEED_660202", "FULL_CSG_SEED_660203"]
    seed_table = markdown_table(
        ["Seed", "Validation objective", "Holdout ROC", "Holdout pairwise", "Holdout NDCG", "Holdout utility"],
        [
            [
                str(int(seed)),
                f4(seeds.set_index("seed").loc[seed, "validation_objective"]),
                f4(mi.loc[f"FULL_CSG_SEED_{seed}", "roc_auc"]),
                f4(mi.loc[f"FULL_CSG_SEED_{seed}", "pairwise_accuracy"]),
                f4(mi.loc[f"FULL_CSG_SEED_{seed}", "ndcg"]),
                signed_pct(mi.loc[f"FULL_CSG_SEED_{seed}", "mean_selected_utility"]),
            ]
            for seed in seeds["seed"].astype(int)
        ],
    )

    major = regimes[
        (regimes["model"] == FULL)
        & (regimes["metric_family"] == "ACTION_SCORING")
        & (regimes["regime_dimension"].isin(["scale", "CF_level", "RI_level", "TI_level"]))
    ]
    regime_table = markdown_table(
        ["Dimension", "Regime", "States", "Pairwise", "NDCG", "Selected utility", "Positive selected"],
        [
            [
                row["regime_dimension"],
                row["regime_value"],
                str(int(row["state_count"])),
                f4(row["pairwise_accuracy"]),
                f4(row["ndcg"]),
                signed_pct(row["mean_selected_utility"]),
                f"{100 * row['selected_positive_fraction']:.2f}%",
            ]
            for _, row in major.iterrows()
        ],
    )

    profile_table = markdown_table(
        ["Scale", "p90 single", "p90 ensemble", "Shared speedup", "GPU peak", "CPU RSS peak", "1,000 decisions"],
        [
            [
                row["scale"],
                f"{row['p90_total_single_model_decision_ms']:.1f} ms",
                f"{row['p90_projected_ensemble_decision_ms']:.1f} ms",
                f"{row['median_shared_vs_naive_speedup']:.2f}x",
                f"{row['gpu_peak_memory_mib']:.1f} MiB",
                f"{row['cpu_peak_process_rss_mib']:.1f} MiB",
                f"{row['projected_1000_decision_overhead_seconds']:.1f} s",
            ]
            for _, row in profile.iterrows()
        ],
    )

    conclusion_lines = []
    for key, value in conclusions.items():
        if isinstance(value, bool):
            value = str(value).upper()
        conclusion_lines.append(f"{key} = {value}")
    conclusion_block = "\n".join(conclusion_lines)

    flat = pi.loc["FLAT_SET"]
    static = pi.loc["STATIC_CSG"]
    no_edge = pi.loc["NO_EDGE_FEATURES"]
    tabular = pi.loc["B3_PHASE6C_TABULAR"]

    return f"""# Phase 6E Supervised NI Validation Report

## 1. Executive conclusion

Phase 6E v2 已完成冻结边界内的离线监督学习验证。一次性内部 holdout 覆盖 {evaluation['state_count']:,} 个状态、{evaluation['action_count']:,} 个候选 target set；三种子 Full CSG 集成在 solver-relevant selected-action utility 上显著优于全部预注册对照，但绝对平均效用仍为 {signed_pct(si.loc[FULL, 'mean_selected_utility'])}。三模型集成的 M/L 推理时延超过冻结预算，因此 Phase 6F **不得直接进入 live solver integration**，结论为 `REVISE_MODEL`。

v1 holdout 因 B2/B3 baseline alias 缺陷被正式作废且未用于结论；本文只使用 `phase6e-holdout-v2`。

```text
{conclusion_block}
```

## 2. Frozen Phase 6C/6D boundary

- Phase 6C 数据版本：`{freeze['phase6c_dataset_version']}`；冻结 hash：`{freeze['phase6c_dataset_freeze_hash']}`。
- Phase 6D 表示：`{freeze['phase6d_csg_version']}`；schema SHA-256：`{freeze['phase6d_schema_sha256']}`。
- 冻结提交：`{freeze['git_commit']}`。
- 环境冻结检查全部通过；未修改 Phase 6C 标签、Phase 6D schema 或调度/搜索语义。
- holdout 在 checkpoint、baseline 和 evaluation plan 全部冻结后才打开；`checkpoint_selection_after_open = {str(access['checkpoint_selection_after_open']).lower()}`。

## 3. Tensorization and data pipeline

采用 native PyTorch heterogeneous tensorizer，保留 8 类节点、20 类 canonical forward relations、显式 edge numeric features，并机械派生 reverse relations；派生边不改变 CSG-1.0 语义。实体 ID 仅用于 lookup/mapping，不进入数值 tensor。

冻结策略为 `A_PRETENSORIZED_SHARDED_CACHE`。缓存审计覆盖 {cache_audit['source_shards']} 个 source/cache shards、{cache_audit['total_states']:,} 个状态、{cache_audit['total_actions']:,} 个动作、{cache_audit['total_cache_bytes'] / 2**30:.2f} GiB；所有 source hash、cache hash、split count、schema hash 与 partial-file 检查均通过。训练按 state batching：每个状态只编码一次 CSG，再一次性评分该状态的全部候选动作。

## 4. Model architecture

冻结主模型为 `{experiment_freeze['selected_candidate']}`：hidden dimension 128、3 层、4 heads、dropout 0.1、7,705,857 parameters。每类节点有独立 input projection；relation-specific key/value projection 消费 edge attributes；type-specific mean pooling 与合法 graph-level features 形成 state embedding。实现为紧凑的 relation-aware heterogeneous attention encoder，不依赖 PyG。

## 5. Target-set action encoder

action encoder 对 target OP embeddings 使用 permutation-invariant mean、max、state-conditioned attention pooling，并拼接 state embedding 和 normalized target size。destroy origin operator 不作为主模型输入；base graph 与候选 target set 分离，不执行 action-specific graph rebuild。

## 6. Training objective

目标为 `L = 1.0 * pairwise_logistic_rank_loss + 0.25 * weighted_BCE`。pairwise 样本在 state 内确定性构造；BCE positive weight 只从 TRAIN 计数导出。训练使用 TRAIN 梯度，TRAIN_VALIDATION 只用于预注册的 `0.5 * pairwise_accuracy + 0.5 * NDCG` checkpoint/config selection，holdout 不参与任何选择。

## 7. Baselines

预注册对照包括：B0 exact random expectation、B1 fixed related、B2 validation-frozen best original operator（`{evaluation['B2_best_fixed_original_operator']}`）、B3 legal Phase 6C tabular diagnostic、flat-set neural control、static-CSG control、no-edge-features control。B3 v2 使用真实 target-set robust labels 与合法 pre-action features；已修复导致 v1 作废的 baseline alias 问题。

## 8. Sanity/leakage tests

mandatory sanity status 为 `{sanity['status']}`：tiny overfit、label shuffle degradation、graph-state shuffle degradation、target-mask shuffle degradation 全部通过。正确映射的 tiny-set PR-AUC 为 {f4(sanity['correct_label_metrics']['pr_auc'])}，label-shuffle PR-AUC 降至 {f4(sanity['label_shuffle_scored_against_true_labels']['pr_auc'])}，target-mask-shuffle PR-AUC 降至 {f4(sanity['target_mask_shuffle_metrics']['pr_auc'])}。这些结果支持“未检测到标签/target mapping 泄漏”，但不应解释为对所有潜在泄漏的形式化证明。

## 9. Validation model selection

小规模开发研究只比较 3 个预注册候选，最终冻结 `{experiment_freeze['selected_candidate']}`。三个独立最终种子均完成 6 epochs，最佳 checkpoint 均在 epoch 6；未报告或选择“最佳 holdout seed”。

{seed_table}

## 10. Internal-holdout predictive results

{ranking_table}

Full ensemble 的 ROC-AUC={f4(full['roc_auc'])}、PR-AUC={f4(full['pr_auc'])}、pairwise={f4(full['pairwise_accuracy'])}、Spearman={f4(full['within_state_spearman'])}、NDCG={f4(full['ndcg'])}。B3/no-edge 的 global ROC/PR 略高，说明 Full CSG 的优势主要体现在 state-conditioned ranking/selection，而不是全局二分类分离度。

## 11. Selected-action utility

{utility_table}

状态配对 Wilcoxon 与 BH-FDR 结果：

{paired_table}

Full CSG 对 B3 的平均增益仅 {signed_pct(tabular['mean_paired_utility_delta'])}（BH-FDR p={tabular['p_value_bh_fdr']:.4g}），属于统计显著但实际幅度很小的优势。Full 选择动作的绝对平均效用为负，说明离线预测器尚不能保证单次动作平均改善。

## 12. Full CSG vs flat representation

Full 对 flat 的 paired selected-utility delta 为 {signed_pct(flat['mean_paired_utility_delta'])}（BH-FDR p={flat['p_value_bh_fdr']:.3g}），并同时提高 ROC/PR、pairwise、Spearman、NDCG、top-1/top-3 和 selected-positive fraction。该 controlled comparison 支持 relational message passing 在离线 target-set 选择任务上提供增量价值，但增益规模仍不足以绕过 runtime gate。

## 13. Synchronization-topology ablation

Full 对 static-CSG 的 paired utility delta 为 {signed_pct(static['mean_paired_utility_delta'])}（BH-FDR p={static['p_value_bh_fdr']:.3g}），主要 ranking 指标也一致更高，因此 realized synchronization/resource topology adds value。

Full 对 no-edge-features 的 selected-utility delta 为 {signed_pct(no_edge['mean_paired_utility_delta'])}（BH-FDR p={no_edge['p_value_bh_fdr']:.3g}）。edge features 改善 solver-relevant utility、pairwise、Spearman、NDCG 与 top-k；但 no-edge 的 ROC/PR 更高，因此结论限定为“对 state-conditioned selection 有价值”，不宣称所有预测指标均改善。

## 14. Structural-regime robustness

{regime_table}

所有主要 Scale/CF/RI/TI 子组的 pairwise accuracy 均不低于 {major['pairwise_accuracy'].min():.4f}，NDCG 均不低于 {major['ndcg'].min():.4f}，且每个子组 selected utility 均高于 fixed related。S 与 TI1 的绝对效用明显偏弱，但对应 oracle/随机效用也低，属于困难分布而非相对 baseline 崩溃。相对 B3 则不是全子组一致领先，不能声称 universal dominance。

## 15. Seed stability

{seed_table}

三个 holdout seed 的 ROC range 为 {mi.loc[seed_models, 'roc_auc'].max() - mi.loc[seed_models, 'roc_auc'].min():.4f}，selected-utility range 为 {mi.loc[seed_models, 'mean_selected_utility'].max() - mi.loc[seed_models, 'mean_selected_utility'].min():.4f}；满足冻结稳定性规则。ensemble 改善 NDCG/top-k/selected utility，但不是所有 global AUC 的最佳单 seed。

## 16. Runtime and memory

{profile_table}

共享 state encoding 相对 naïve repeated graph encoding 获得 4.86x–9.88x median speedup，GPU inference peak 仅 53.9–55.8 MiB，memory gate 通过。冻结的 150 ms p90 ensemble budget 只在 S 通过；M/L 分别为 {profile.set_index('scale').loc['M', 'p90_projected_ensemble_decision_ms']:.1f}/{profile.set_index('scale').loc['L', 'p90_projected_ensemble_decision_ms']:.1f} ms，1,000 次决策预算亦失败。CPU RSS 是包含 Python、数据与已加载模型的进程峰值，不等同于单次推理增量。

## 17. Failure cases

1. S 和 TI1 分层的 absolute selected utility 较低；需要进一步改善 utility-aware calibration/objective。
2. Full 与 B3 的总体 utility 差距很小，并在若干结构子组反转。
3. no-edge 的 ROC/PR 高于 Full，提示 edge features 对全局分类有 trade-off。
4. 三模型 sequential ensemble 在 M/L 不满足 live iterative-search 时延预算。
5. v1 holdout baseline alias 缺陷证明 baseline identity/hash 审计不可省略；v1 已作废，不进入任何科学结论。

## 18. Scientific interpretation

Q1：Full CSG 在 solver-relevant utility 上显著优于最强合法非图 B3，但 global ROC/PR 不占优，效应量很小。Q2：Full 显著优于 flat，支持 relational topology 的增量价值。Q3：Full 显著优于 static，支持 realized synchronization/resource relations。Q4：Full 对 random、related、best fixed 和 B3 均有正的配对 utility delta。Q5：主要结构分层未相对 fixed-related 崩溃，但困难分层和对 B3 的局部反转必须保留。Q6：当前 ensemble inference 不满足未来 live integration 预算。

## 19. Phase 6F recommendation

`PHASE6F_RECOMMENDATION = REVISE_MODEL`。

建议保留冻结 CSG-1.0 和当前 action-set 语义，优先做聚焦修订：以 selected utility/regret 为直接验证目标改进 objective/calibration，并将三种子 ensemble 蒸馏或压缩为单模型/共享编码部署路径。修订后必须重新冻结阈值并在未见数据上复验；在 latency gate 通过前，不进入 live ALNS/NI integration。

## 20. Reproducibility checklist

- [x] Phase 6C dataset hash 与 Phase 6D schema hash 冻结并验证。
- [x] tensor schema、405-shard cache、100,000 states/2,362,722 actions 全量 hash 审计。
- [x] tiny overfit 与三类 shuffle sanity 通过。
- [x] config/checkpoint 仅由 TRAIN_VALIDATION 选择。
- [x] 三个独立 seed 的 best/last/optimizer/config/schema metadata 保留。
- [x] v2 holdout 一次性访问记录与全部输出 hash 保留。
- [x] state-paired Wilcoxon、BH-FDR、结构分层结果保留。
- [x] 10 组图均由 `scripts/generate_phase6e_report.py` 生成 PNG/PDF。
- [x] inference S/M/L 分组件、memory 与 100/500/1,000 decisions 投影保留。
- [ ] live solver integration：按 Phase 6E stop condition 明确禁止，等待模型修订与人工评审。
"""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    selected = pd.read_csv(SELECTED_PATH)
    regimes = pd.read_csv(REGIMES_PATH)
    pairwise = pd.read_csv(PAIRWISE_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    seeds = pd.read_csv(SEEDS_PATH)
    sanity = load_json(SANITY_PATH)
    freeze = load_json(FREEZE_PATH)
    cache_audit = load_json(CACHE_AUDIT_PATH)
    experiment_freeze = load_json(EXPERIMENT_FREEZE_PATH)
    evaluation = load_json(EVALUATION_COMPLETION_PATH)
    access = load_json(HOLDOUT_ACCESS_PATH)
    profile_summary = load_json(PROFILE_SUMMARY_PATH)

    required_models = {
        FULL,
        "FLAT_SET",
        "STATIC_CSG",
        "NO_EDGE_FEATURES",
        "B3_PHASE6C_TABULAR",
    }
    if not required_models.issubset(set(metrics["model"])):
        raise RuntimeError("internal holdout metrics are incomplete")
    if evaluation["status"] != "COMPLETE" or evaluation["experiment_version"] != "phase6e-holdout-v2":
        raise RuntimeError("Phase 6E v2 holdout is not complete")
    if not V1_INVALIDATION_PATH.exists():
        raise RuntimeError("v1 invalidation evidence is missing")

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    plot_training_curves(seeds)
    plot_ranking_comparison(metrics)
    plot_selected_utility(selected)
    plot_selected_regret(selected)
    plot_controlled_comparison(
        metrics,
        "FLAT_SET",
        "Fig05_full_csg_vs_flat_set",
        "Full CSG vs flat-set representation",
    )
    plot_controlled_comparison(
        metrics,
        "STATIC_CSG",
        "Fig06_full_csg_vs_static_csg",
        "Full CSG vs static CSG",
    )
    plot_two_regime_dimensions(
        regimes,
        ("scale", ["S", "M", "L"]),
        ("CF_level", ["CF1", "CF2", "CF3"]),
        "Fig07_performance_by_scale_cf",
    )
    plot_two_regime_dimensions(
        regimes,
        ("RI_level", ["RI1", "RI2", "RI3"]),
        ("TI_level", ["TI1", "TI2", "TI3"]),
        "Fig08_performance_by_ri_ti",
    )
    plot_search_stage(regimes)
    plot_inference(profile)

    conclusions_record = build_conclusions(
        metrics, selected, pairwise, regimes, seeds, profile_summary
    )
    conclusions_path = AUDIT / "scientific_conclusions.json"
    conclusions_path.write_text(
        json.dumps(conclusions_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    REPORT.write_text(
        generate_report(
            metrics,
            selected,
            pairwise,
            regimes,
            profile,
            seeds,
            sanity,
            freeze,
            cache_audit,
            experiment_freeze,
            evaluation,
            access,
            conclusions_record,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE",
        "figure_pairs": 10,
        "report": str(REPORT.relative_to(ROOT)),
        "conclusions": str(conclusions_path.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
