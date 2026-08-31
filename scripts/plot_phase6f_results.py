#!/usr/bin/env python3
"""Generate the ten required reproducible Phase 6F figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6f/figures"
PRIMARY = "PHASE6F_REVISED_HYBRID_SEED_660301"
OLD = "PHASE6E_FULL_CSG_ENSEMBLE"
COLORS = {"old": "#708090", "new": "#1f77b4", "fallback": "#d95f02"}


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def short_model(name: str) -> str:
    labels = {
        PRIMARY: "Phase6F hybrid",
        OLD: "Phase6E ensemble",
        "PHASE6E_DEPLOYABLE_SINGLE_SEED_660201": "Phase6E single",
        "PHASE6C_TABULAR": "Phase6C tabular",
        "B1_FIXED_RELATED": "Fixed related",
        "PHASE6E_FLAT_SET": "Flat set",
        "B0_RANDOM_EXPECTATION": "Random expectation",
    }
    return labels.get(name, name.replace("PHASE6F_REVISED_SEED_", "P6F seed "))


def objective_comparison() -> None:
    data = pd.read_csv(ROOT / "outputs/phase6f/objectives/objective_validation_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [value.split("_")[0] for value in data["objective"]]
    values = 100.0 * data["mean_selected_utility"]
    bars = ax.bar(names, values, color=["#999999", "#66c2a5", "#3288bd"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Validation mean selected utility (%)")
    ax.set_title("Utility-aware objective comparison")
    ax.axhline(0, color="black", linewidth=0.8)
    save(fig, "Fig01_utility_objective_comparison")


def compact_frontier() -> None:
    data = pd.read_csv(ROOT / "outputs/phase6f/compact_models/compact_model_validation_summary.csv")
    latency = pd.read_csv(ROOT / "outputs/phase6f/profiling/latency_profile.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(data["selected"], COLORS["new"], "#aaaaaa")
    ax.scatter(data["parameter_count"] / 1e6, 100 * data["mean_selected_utility"],
               s=100, c=colors)
    for row in data.itertuples(index=False):
        ax.annotate(row.model_candidate,
                    (row.parameter_count / 1e6, 100 * row.mean_selected_utility),
                    xytext=(5, 5), textcoords="offset points")
    l_p90 = float(latency.loc[latency["scale"].eq("L"), "p90_model_decision_ms"].iloc[0])
    ax.text(0.02, 0.03, f"Selected model measured L p90: {l_p90:.2f} ms",
            transform=ax.transAxes)
    ax.set_xlabel("Parameters (millions; compactness proxy)")
    ax.set_ylabel("Validation mean selected utility (%)")
    ax.set_title("Compact-model quality / deployment-cost frontier")
    save(fig, "Fig02_compact_model_quality_latency_frontier")


def calibration_reliability() -> None:
    data = pd.read_csv(ROOT / "outputs/phase6f/calibration/reliability_bins.csv")
    if "method" not in data:
        data["method"] = "ISOTONIC"
    fig, ax = plt.subplots(figsize=(6, 6))
    for method, group in data.groupby("method", sort=False):
        group = group[group["count"] > 0]
        ax.plot(group["mean_confidence"], group["positive_fraction"], marker="o", label=method)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Mean calibrated probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Validation calibration reliability")
    ax.legend()
    save(fig, "Fig03_calibration_reliability")


def coverage_tradeoff() -> None:
    data = pd.read_csv(ROOT / "outputs/phase6f/calibration/selective_policy_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(100 * data["coverage"], 100 * data["hybrid_selected_utility"],
               s=25, alpha=0.45, color="#7b3294")
    chosen = data[data["selected"]]
    ax.scatter(100 * chosen["coverage"], 100 * chosen["hybrid_selected_utility"],
               s=130, color=COLORS["new"], label="Frozen threshold")
    ax.set_xlabel("Intervention coverage (%)")
    ax.set_ylabel("Validation hybrid utility (%)")
    ax.set_title("Intervention coverage / utility tradeoff")
    ax.legend()
    save(fig, "Fig04_intervention_coverage_utility_tradeoff")


def selected_utility() -> None:
    data = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_selected_action_summary.csv"
    )
    methods = [PRIMARY, OLD, "PHASE6E_DEPLOYABLE_SINGLE_SEED_660201",
               "PHASE6C_TABULAR", "PHASE6E_FLAT_SET", "B1_FIXED_RELATED",
               "B0_RANDOM_EXPECTATION"]
    data = data.set_index("model").loc[methods].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([short_model(value) for value in data["model"]],
                  100 * data["mean_selected_utility"],
                  color=[COLORS["new"]] + [COLORS["old"]] * 4
                  + [COLORS["fallback"], "#bbbbbb"])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylabel("R06 mean selected utility (%)")
    ax.set_title("Revision-holdout selected utility")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "Fig05_revision_holdout_selected_utility")


def regret() -> None:
    data = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_selected_action_summary.csv"
    )
    methods = [PRIMARY, OLD, "PHASE6E_DEPLOYABLE_SINGLE_SEED_660201",
               "PHASE6C_TABULAR", "PHASE6E_FLAT_SET", "B1_FIXED_RELATED"]
    data = data.set_index("model").loc[methods].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([short_model(value) for value in data["model"]],
                  100 * data["mean_selected_regret"],
                  color=[COLORS["new"]] + [COLORS["old"]] * 4 + [COLORS["fallback"]])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylabel("R06 mean regret (%) — lower is better")
    ax.set_title("Revision-holdout selected-action regret")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "Fig06_revision_holdout_regret")


def paired_old_new() -> None:
    data = pd.read_parquet(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_state_selected_actions.parquet"
    )
    old = data[data["model"].eq(OLD)][["state_id", "selected_utility"]].rename(
        columns={"selected_utility": "old"}
    )
    new = data[data["model"].eq(PRIMARY)][["state_id", "selected_utility"]].rename(
        columns={"selected_utility": "new"}
    )
    paired = old.merge(new, on="state_id", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(100 * paired["old"], 100 * paired["new"], gridsize=45,
              mincnt=1, cmap="Blues")
    limits = [100 * min(paired["old"].min(), paired["new"].min()),
              100 * max(paired["old"].max(), paired["new"].max())]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Phase6E ensemble selected utility (%)")
    ax.set_ylabel("Phase6F selected utility (%)")
    ax.set_title("State-paired old vs revised model")
    save(fig, "Fig07_old_vs_revised_model")


def grouped_delta(name: str, dimensions: tuple[str, str], figure: str) -> None:
    data = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_structural_summary.csv"
    )
    old = data[data["model"].eq(OLD)]
    new = data[data["model"].eq(PRIMARY)]
    joined = new.merge(old, on=["regime_dimension", "regime_value"],
                       suffixes=("_new", "_old"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dimension in zip(axes, dimensions):
        part = joined[joined["regime_dimension"].eq(dimension)].copy()
        part["delta"] = 100 * (
            part["mean_selected_utility_new"] - part["mean_selected_utility_old"]
        )
        colors = np.where(part["delta"] >= 0, COLORS["new"], COLORS["fallback"])
        bars = ax.bar(part["regime_value"], part["delta"], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(dimension)
        ax.set_ylabel("Phase6F − Phase6E utility (pp)")
    fig.suptitle(name)
    save(fig, figure)


def latency() -> None:
    data = pd.read_csv(ROOT / "outputs/phase6f/profiling/latency_profile.csv")
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, data["p90_model_decision_ms"], width,
           label="Model decision", color=COLORS["new"])
    ax.bar(x + width / 2, data["p90_end_to_end_decision_ms"], width,
           label="End-to-end", color="#80b1d3")
    ax.axhline(150, color="#d73027", linestyle="--", label="150 ms hard gate")
    ax.set_xticks(x, data["scale"])
    ax.set_ylabel("p90 latency (ms)")
    ax.set_title("Frozen deployment latency on R06")
    ax.legend()
    save(fig, "Fig10_deployment_latency")


def main() -> None:
    objective_comparison()
    compact_frontier()
    calibration_reliability()
    coverage_tradeoff()
    selected_utility()
    regret()
    paired_old_new()
    grouped_delta(
        "Revision utility delta by scale and CF",
        ("scale", "CF_level"),
        "Fig08_revision_performance_by_scale_cf",
    )
    grouped_delta(
        "Revision utility delta by RI and TI",
        ("RI_level", "TI_level"),
        "Fig09_revision_performance_by_ri_ti",
    )
    latency()
    print(f"PHASE6F_FIGURES_COMPLETE={len(list(OUT.glob('*.png')))}")


if __name__ == "__main__":
    main()
