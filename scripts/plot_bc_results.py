#!/usr/bin/env python3
"""Plot Behavior Cloning loss and imitation accuracy diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "total": "#0072B2", "operation": "#D55E00", "island": "#009E73",
    "w": "#CC79A7", "f": "#E69F00", "joint": "#0072B2",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 130,
        "savefig.dpi": 320,
    })


def plot_history(history_path: Path, output_dir: Path) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
    epochs = [item["epoch"] for item in history]
    _style()
    fig, axis = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for key in ("total", "operation", "island"):
        axis.plot(epochs, [item["losses"][key] for item in history], label=key.title(),
                  color=COLORS[key], linewidth=1.8)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Tiny exact-trajectory Behavior Cloning")
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output_dir / "Figure_1_bc_loss.png")
    fig.savefig(output_dir / "Figure_1_bc_loss.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for key in ("operation", "island", "w", "f", "joint"):
        axis.plot(epochs, [100.0 * item["accuracy"][key] for item in history],
                  label=key.upper() if key in {"w", "f"} else key.title(),
                  color=COLORS[key], linewidth=1.7,
                  linestyle="--" if key in {"w", "f"} else "-")
    axis.set_ylim(-2, 102)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Expert reproduction (%)")
    axis.set_title("Hard-masked autoregressive imitation accuracy")
    axis.legend(frameon=False, ncol=5, fontsize=8)
    fig.savefig(output_dir / "Figure_2_bc_accuracy.png")
    fig.savefig(output_dir / "Figure_2_bc_accuracy.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.history.parent
    output.mkdir(parents=True, exist_ok=True)
    plot_history(args.history, output)
    print(f"Wrote BC figures to {output}")


if __name__ == "__main__":
    main()
