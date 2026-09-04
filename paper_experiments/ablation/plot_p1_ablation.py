#!/usr/bin/env python3
"""Render the P1 scale-stratified paired-effect ablation figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from paper_experiments.scripts.paper_figure_style import (
    FINAL_WIDTH_MM,
    PALETTE,
    POINTS_PER_MM,
    add_panel_label,
    apply_publication_style,
    file_sha256,
    plt,
    save_publication_figure,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = ROOT / "paper_experiments/ablation"
DATA_PATH = ABLATION_ROOT / "processed_data/ablation_paired_differences.csv"
STATS_PATH = ABLATION_ROOT / "processed_data/ablation_statistics.json"
FIGURE_ROOT = ABLATION_ROOT / "figures"
ABLATIONS = ("Uniform full-bank selection", "No NI (ALNS-H1)")
LABELS = ("Uniform full-bank", "No NI (ALNS-H1)")
SCALES = (("S", "Small"), ("M", "Medium"), ("L", "Large"))


def main() -> int:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    if stats.get("status") != "PASS_COMPLETE_REPLAY_VALIDATED":
        raise RuntimeError("P1 analysis gate is not PASS")
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36:
        raise RuntimeError("P1 paired figure requires 18 instances x two ablations")
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, 70.0 * POINTS_PER_MM / 72.0),
        sharey=True,
        constrained_layout=True,
    )
    colors = (PALETTE["dcga"], PALETTE["neutral"])
    for axis, (scale, title), panel_id in zip(axes, SCALES, ("a", "b", "c")):
        values = [
            np.asarray([
                float(row["paired_rpd_difference_positive_favors_full"])
                for row in rows if row["scale"] == scale and row["ablation"] == ablation
            ])
            for ablation in ABLATIONS
        ]
        if any(len(group) != 6 for group in values):
            raise RuntimeError(f"P1 scale {scale} must contain six paired instances per ablation")
        box = axis.boxplot(
            values,
            positions=(1, 2),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE["black"], "linewidth": 1.1},
            whiskerprops={"color": PALETTE["neutral"], "linewidth": 0.8},
            capprops={"color": PALETTE["neutral"], "linewidth": 0.8},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)
        for position, group, color in zip((1, 2), values, colors):
            jitter = np.linspace(-0.10, 0.10, num=len(group))
            axis.scatter(
                position + jitter,
                group,
                s=14,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        axis.axhline(0.0, color=PALETTE["black"], linewidth=0.75, linestyle="--", zorder=1)
        axis.set_xticks((1, 2), LABELS, rotation=12, ha="right", rotation_mode="anchor")
        axis.set_title(f"{title} (n = 6)")
        style_axis(axis)
        add_panel_label(axis, panel_id)
    axes[0].set_ylabel("Ablation RPD minus Full RPD (percentage points)")
    figure.supxlabel("Positive values favor Full CSG-NI", fontsize=7)
    base_path = FIGURE_ROOT / "figure_p1_ablation_effect_by_scale"
    save_publication_figure(
        figure,
        axes,
        ("a", "b", "c"),
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": "P1 ablation figure",
            "core_conclusion": "The quality cost of removing learned selection or NI is evaluated as an instance-paired contrast across scale.",
            "archetype": "quantitative grid",
            "source_data": str(DATA_PATH.relative_to(ROOT)),
            "source_sha256": file_sha256(DATA_PATH),
            "replicate_unit": "independent Core instance; each paired difference uses arm medians over five matched seeds",
            "center_and_spread": "boxplots show median, IQR and 1.5-IQR whiskers; all six instance differences per scale are shown",
            "exclusions": "none",
            "reporting_boundary": "separate controlled ablation analysis; positive differences favor Full and do not imply universal per-seed superiority",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
