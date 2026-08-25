#!/usr/bin/env python3
"""Generate the five publication-ready Phase 3 figures from formal outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "purple": "#CC79A7", "sky": "#56B4E9", "black": "#222222",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _histories(root):
    histories = []
    for index in (1, 2, 3):
        path = root / f"ppo_seed_{index}" / "training_history.json"
        histories.append(json.loads(path.read_text(encoding="utf-8"))["updates"])
    return histories


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_training_curves(histories, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for index, (history, color) in enumerate(zip(histories, colors), start=1):
        steps = [row["environment_steps"] for row in history]
        axes[0].plot(steps, row_values(history, "normalized_return"), color=color, lw=1.4, label=f"Seed {index}")
        axes[1].plot(
            steps, row_values(history, "selection_validation_normalized_makespan"),
            color=color, lw=1.4, label=f"Seed {index}",
        )
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Mean normalized return")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Held-out normalized makespan")
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    _save(fig, output, "Figure_1_training_return_makespan")


def row_values(history, key):
    return [float(row[key]) for row in history]


def figure_losses(histories, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65))
    colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for index, (history, color) in enumerate(zip(histories, colors), start=1):
        steps = [row["environment_steps"] for row in history]
        axes[0].plot(steps, row_values(history, "policy_loss"), color=color, lw=1.2, label=f"Seed {index}")
        axes[1].plot(steps, row_values(history, "value_loss"), color=color, lw=1.2)
        axes[2].plot(steps, row_values(history, "approx_kl"), color=color, lw=1.2)
    axes[0].set_ylabel("Policy loss")
    axes[1].set_ylabel("Value loss")
    axes[2].set_ylabel("Approximate KL")
    for axis in axes:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.5)
    _save(fig, output, "Figure_2_losses_kl")


def figure_entropy(histories, output):
    fig, axis = plt.subplots(figsize=(4.4, 3.0))
    keys = (
        ("operation_entropy", "O", COLORS["blue"]),
        ("island_entropy", "M", COLORS["orange"]),
        ("w_entropy", "W", COLORS["green"]),
        ("f_entropy", "F", COLORS["purple"]),
    )
    min_length = min(len(history) for history in histories)
    steps = np.mean([
        [history[index]["environment_steps"] for index in range(min_length)]
        for history in histories
    ], axis=0)
    for key, label, color in keys:
        values = np.asarray([
            [float(history[index][key]) for index in range(min_length)]
            for history in histories
        ])
        average = values.mean(axis=0)
        deviation = values.std(axis=0)
        axis.plot(steps, average, color=color, lw=1.6, label=f"{label} stage")
        axis.fill_between(steps, average - deviation, average + deviation, color=color, alpha=0.13)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Stage entropy")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, output, "Figure_3_stage_entropy")


def figure_validation_gap(rows, output):
    order = ("H1", "H2", "H3", "BC_GREEDY", "PPO_GREEDY")
    labels = ("H1", "H2", "H3", "BC", "PPO")
    groups = [[float(row["gap_percent"]) for row in rows if row["method"] == method] for method in order]
    means = [np.mean(values) for values in groups]
    errors = [np.std(values) for values in groups]
    asymmetric_errors = np.asarray([
        [min(center, spread) for center, spread in zip(means, errors)],
        errors,
    ])
    fig, axis = plt.subplots(figsize=(4.6, 3.0))
    bars = axis.bar(
        labels, means, yerr=asymmetric_errors, capsize=3,
        color=[COLORS["sky"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["blue"]],
        edgecolor="white", linewidth=0.7,
    )
    axis.set_ylabel("Gap to best compared method (%)")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, output, "Figure_4_validation_gaps")


def figure_canonical_boxplot(rows, output):
    families = ("Brandimarte", "Hurink E", "Hurink R", "Hurink V")
    methods = ("H1", "H2", "H3", "BC_GREEDY", "PPO_GREEDY")
    method_labels = ("H1", "H2", "H3", "BC", "PPO")
    colors = (COLORS["sky"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["blue"])
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.8), sharey=True)
    for axis, family in zip(axes, families):
        values = [
            [float(row["gap_percent"]) for row in rows if row["group"] == family and row["method"] == method]
            for method in methods
        ]
        box = axis.boxplot(values, patch_artist=True, widths=0.65, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        axis.set_xticks(range(1, 6), method_labels, rotation=35, ha="right")
        axis.set_xlabel(family)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axes[0].set_ylabel("Gap to best compared method (%)")
    fig.tight_layout(w_pad=0.8)
    _save(fig, output, "Figure_5_canonical_family_gaps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase3/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    histories = _histories(args.results_dir)
    figure_training_curves(histories, args.output_dir)
    figure_losses(histories, args.output_dir)
    figure_entropy(histories, args.output_dir)
    synthetic = _read_csv(args.results_dir / "validation" / "synthetic_results.csv")
    figure_validation_gap(synthetic, args.output_dir)
    canonical_path = args.results_dir / "canonical_evaluation" / "canonical_results.csv"
    if canonical_path.exists():
        figure_canonical_boxplot(_read_csv(canonical_path), args.output_dir)
    print(f"Wrote Phase 3 PNG/PDF figures to {args.output_dir}")


if __name__ == "__main__":
    main()
