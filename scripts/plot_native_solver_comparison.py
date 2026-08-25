#!/usr/bin/env python3
"""Plot objective/bound agreement and runtime for tiny_03 native solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#0072B2", "#E69F00")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "savefig.dpi": 320,
    })


def plot_comparison(input_path: Path, output_dir: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    solvers = data["solvers"]
    labels = [item["label"] for item in solvers]
    objectives = [item["solver_makespan"] for item in solvers]
    bounds = [item["best_bound"] for item in solvers]
    runtimes = [1000.0 * item["runtime_seconds"] for item in solvers]
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    positions = np.arange(len(labels))
    width = 0.32
    fig, axis = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    bars_objective = axis.bar(positions - width / 2, objectives, width, label="Objective",
                              color=COLORS[0], edgecolor="#222222", linewidth=0.6)
    bars_bound = axis.bar(positions + width / 2, bounds, width, label="Best bound",
                          color=COLORS[1], edgecolor="#222222", linewidth=0.6, hatch="//")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Makespan")
    axis.set_ylim(0, max(objectives + bounds) * 1.2)
    axis.legend(frameon=False, ncol=2)
    for bars in (bars_objective, bars_bound):
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    fig.savefig(output_dir / "Figure_1_objective_bound.png")
    fig.savefig(output_dir / "Figure_1_objective_bound.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    bars = axis.bar(labels, runtimes, color=COLORS, edgecolor="#222222", linewidth=0.6)
    axis.set_ylabel("Solver runtime (ms)")
    axis.set_ylim(0, max(runtimes) * 1.25)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in runtimes], padding=3, fontsize=8)
    fig.savefig(output_dir / "Figure_2_solver_runtime.png")
    fig.savefig(output_dir / "Figure_2_solver_runtime.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent
    plot_comparison(args.input, output)
    print(f"Wrote native solver comparison figures to {output}")


if __name__ == "__main__":
    main()
