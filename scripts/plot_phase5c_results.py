#!/usr/bin/env python3
"""Generate the twelve required Phase 5C figures from frozen CSV/JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5c"
FIG = OUT / "figures"
COLORS = {"GA": "#4C78A8", "Adapted DCGA": "#F58518", "ALNS-H1": "#54A24B"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def scatter(summary, metric, ylabel, name):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, part in summary.groupby("algorithm"):
        ax.scatter(part[metric], part.improvement_over_h1_percent, s=20, alpha=.65, label=method, color=COLORS[method])
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel=ylabel, ylabel="Improvement over H1 (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig, name)


def convergence(x_key, name, xlabel):
    grid = np.linspace(0, 1, 101)
    curves = {method: [] for method in COLORS}
    for path in (OUT / "search/formal").rglob("seed_*.json"):
        raw = json.loads(path.read_text())
        method = raw["algorithm"]
        trace = raw["convergence_trace"]
        if x_key == "elapsed_time":
            x = np.array([point[x_key] / raw["time_limit_seconds"] for point in trace])
        else:
            maximum = max(1, raw["decoder_evaluations"])
            x = np.array([point[x_key] / maximum for point in trace])
        y = np.array([point["current_best_makespan"] for point in trace])
        baseline = y[0]
        normalized = 100 * (baseline - y) / baseline
        curves[method].append(normalized[np.maximum(0, np.searchsorted(x, grid, side="right") - 1)])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, values in curves.items():
        matrix = np.asarray(values)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        ax.plot(grid, mean, label=method, color=COLORS[method])
        ax.fill_between(grid, mean - std, mean + std, alpha=.15, color=COLORS[method])
    ax.set(xlabel=xlabel, ylabel="Improvement from initial search solution (%)")
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save(fig, name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(OUT / "benchmark_audit/instance_metrics.csv")
    constructive = pd.read_csv(OUT / "comparisons/constructive_results.csv")
    summary = pd.read_csv(OUT / "comparisons/search_instance_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for family, part in audit.groupby("family"):
        axes[0].hist(part.F_route_mean, bins=np.linspace(0, .75, 16), alpha=.45, label=family)
        axes[1].hist(part.F_cap_mean, bins=np.linspace(.75, 1.01, 14), alpha=.45, label=family)
    axes[0].set(xlabel="Routing flexibility", ylabel="Instances"); axes[1].set(xlabel="Capability coverage")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "Fig01_structural_flexibility_profiles")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for family, part in audit.groupby("family"):
        ax.scatter(part.F_route_mean, part.F_cap_mean, s=26, alpha=.7, label=family)
    ax.set(xlabel="F_route_mean", ylabel="F_cap_mean"); ax.legend(frameon=False); ax.grid(alpha=.2)
    save(fig, "Fig02_route_vs_capability_density")

    selected = constructive[constructive.method.isin(["H2", "BC_GREEDY", "PPO_MEAN"])]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = [selected[selected.method == method].gap_to_h1_percent for method in ("H2", "BC_GREEDY", "PPO_MEAN")]
    ax.boxplot(data, tick_labels=["H2", "BC", "PPO mean"], showmeans=True)
    ax.axhline(0, color="black", lw=.8); ax.set(ylabel="Gap to H1 (%)"); ax.grid(axis="y", alpha=.2)
    save(fig, "Fig03_constructive_gap_to_H1")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    methods = list(COLORS)
    ax.boxplot([summary[summary.algorithm == method]["mean"] for method in methods], tick_labels=methods, showmeans=True)
    ax.set(ylabel="Mean makespan across 10 runs"); ax.grid(axis="y", alpha=.2)
    save(fig, "Fig04_metaheuristic_solution_quality")

    for column, name, ylabel in [
        ("improvement_over_h1_percent", "Fig05_improvement_over_H1", "Improvement over H1 (%)"),
        ("improvement_over_ppo_percent", "Fig06_improvement_over_PPO", "Improvement over PPO mean (%)")]:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.boxplot([summary[summary.algorithm == method][column] for method in methods], tick_labels=methods, showmeans=True)
        ax.axhline(0, color="black", lw=.8); ax.set(ylabel=ylabel); ax.grid(axis="y", alpha=.2)
        save(fig, name)

    convergence("elapsed_time", "Fig07_convergence_wall_clock", "Normalized wall-clock T/Tmax")
    convergence("decoder_evaluations", "Fig08_convergence_decoder_evaluations", "Normalized decoder evaluations")
    scatter(summary, "F_route_mean", "Routing flexibility F_route_mean", "Fig09_performance_vs_routing_flexibility")
    scatter(summary, "F_cap_mean", "Capability coverage F_cap_mean", "Fig10_performance_vs_capability_coverage")

    heat = summary.groupby(["family", "algorithm"]).improvement_over_h1_percent.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=20); ax.set_yticks(range(len(heat.index)), heat.index)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)): ax.text(j, i, f"{heat.iloc[i,j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Improvement over H1 (%)")
    save(fig, "Fig11_family_method_heatmap")

    tax = summary.groupby(["taxonomy", "algorithm"]).improvement_over_h1_percent.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.4)); x=np.arange(len(tax.index)); width=.24
    for j, method in enumerate(methods): ax.bar(x+(j-1)*width, tax[method], width, label=method, color=COLORS[method])
    ax.set_xticks(x, [item.replace("_", "\n") for item in tax.index]); ax.axhline(0,color="black",lw=.8)
    ax.set(ylabel="Mean improvement over H1 (%)"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    save(fig, "Fig12_core_vs_extreme_comparison")
    print("PHASE5C_FIGURES_COMPLETE figures=12 formats=PNG,PDF")


if __name__ == "__main__":
    main()
