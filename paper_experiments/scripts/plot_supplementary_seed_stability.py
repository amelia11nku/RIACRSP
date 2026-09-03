#!/usr/bin/env python3
"""Plot Supplementary Figure 2: seed-to-seed RPD variability on Core45."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from paper_figure_style import (
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
PAPER_ROOT = ROOT / "paper_experiments"
DATA_ROOT = PAPER_ROOT / "processed_data/supplementary"
FIGURE_ROOT = PAPER_ROOT / "figures"
METHODS = (
    ("GA", PALETTE["ga"]),
    ("Adapted DCGA", PALETTE["dcga"]),
    ("DABC", PALETTE["dabc"]),
    ("LG_HGA", PALETTE["lghga"]),
    ("CSG-NI", PALETTE["csgni"]),
)
TICK_LABELS = {"Adapted DCGA": "DCGA"}
SCALES = (("S", "Small"), ("M", "Medium"), ("L", "Large"))


def load_rows() -> tuple[list[dict[str, str]], Path]:
    analysis_path = DATA_ROOT / "supplementary_analysis.json"
    source_path = DATA_ROOT / "core_seed_variability.csv"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "PASS_DESCRIPTIVE_EXPLORATORY":
        raise RuntimeError("supplementary Core45 analysis gate is not PASS")
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["instance_id"], row["display_method"]) for row in rows}
    if len(rows) != 225 or len(keys) != 225:
        raise RuntimeError("expected one seed-variability value per method and Core instance")
    for row in rows:
        value = float(row["seed_rpd_sd_percent_points"])
        if not math.isfinite(value) or value < 0 or int(row["seed_count"]) != 5:
            raise RuntimeError("invalid seed-variability source value")
    return rows, source_path


def main() -> int:
    rows, source_path = load_rows()
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, 68.0 * POINTS_PER_MM / 72.0),
        sharey=True,
        constrained_layout=True,
    )
    panel_ids = ("a", "b", "c")
    for axis, (scale, title), panel_id in zip(axes, SCALES, panel_ids):
        values = [
            np.asarray([
                float(row["seed_rpd_sd_percent_points"])
                for row in rows
                if row["scale"] == scale and row["display_method"] == method
            ])
            for method, _ in METHODS
        ]
        if any(len(method_values) != 15 for method_values in values):
            raise RuntimeError(f"expected 15 Core instances per method for scale {scale}")
        boxes = axis.boxplot(
            values,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE["black"], "linewidth": 1.0},
            whiskerprops={"color": PALETTE["neutral"], "linewidth": 0.7},
            capprops={"color": PALETTE["neutral"], "linewidth": 0.7},
        )
        point_offsets = np.linspace(-0.11, 0.11, 15)
        for method_index, ((_, color), method_values) in enumerate(zip(METHODS, values), start=1):
            boxes["boxes"][method_index - 1].set(
                facecolor=color,
                edgecolor=PALETTE["black"],
                linewidth=0.7,
            )
            axis.scatter(
                method_index + point_offsets,
                np.sort(method_values),
                s=7,
                facecolor=color,
                edgecolor=PALETTE["black"],
                linewidth=0.25,
                zorder=3,
            )
        axis.set_xticks(
            range(1, len(METHODS) + 1),
            [TICK_LABELS.get(method, method) for method, _ in METHODS],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_title(f"{title} (n = 15)")
        add_panel_label(axis, panel_id)
        style_axis(axis)
    axes[0].set_ylabel("Seed-to-seed RPD s.d. (percentage points)")
    base_path = FIGURE_ROOT / "supplementary_figure2_seed_stability"
    save_publication_figure(
        figure,
        axes,
        panel_ids,
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": "Supplementary Figure 2",
            "core_conclusion": "Seed sensitivity differs by method and scale, qualifying quality comparisons with a direct robustness diagnostic.",
            "archetype": "quantitative grid",
            "source_data": str(source_path.relative_to(ROOT)),
            "source_sha256": file_sha256(source_path),
            "source_rows": len(rows),
            "replicate_unit": "Core instance; each point is the sample standard deviation of RPD across five matched seeds",
            "center_and_spread": "boxplot median and interquartile range; whiskers use the Matplotlib 1.5-IQR rule",
            "exclusions": "none; boxplot outlier symbols are suppressed because every instance is shown as a point",
            "reporting_boundary": "descriptive robustness analysis; lower variability does not imply better solution quality",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
