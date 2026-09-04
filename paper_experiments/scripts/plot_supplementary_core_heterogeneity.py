#!/usr/bin/env python3
"""Plot Supplementary Figure 1: paired CSG-NI advantage by scale and CF."""

from __future__ import annotations

import csv
import json
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
)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
DATA_ROOT = PAPER_ROOT / "processed_data/supplementary"
FIGURE_ROOT = PAPER_ROOT / "figures"
COMPETITORS = ("GA", "DCGA", "DABC", "LG_HGA")
CF_LEVELS = ("CF1", "CF2", "CF3")
SCALES = (("S", "Small"), ("M", "Medium"), ("L", "Large"))


def load_rows() -> tuple[list[dict[str, str]], Path]:
    analysis_path = DATA_ROOT / "supplementary_analysis.json"
    source_path = DATA_ROOT / "core_paired_advantage_scale_cf.csv"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "PASS_DESCRIPTIVE_EXPLORATORY":
        raise RuntimeError("supplementary Core45 analysis gate is not PASS")
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (row["scale"], row["display_competitor"], row["CF_level"])
        for row in rows
    }
    if len(rows) != 36 or len(keys) != 36:
        raise RuntimeError("expected 3 scales x 4 competitors x 3 CF levels")
    if any(int(row["instance_count"]) != 5 for row in rows):
        raise RuntimeError("every heatmap cell must summarize five Core instances")
    return rows, source_path


def main() -> int:
    rows, source_path = load_rows()
    lookup = {
        (row["scale"], row["display_competitor"], row["CF_level"]): row
        for row in rows
    }
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, 66.0 * POINTS_PER_MM / 72.0),
        sharey=True,
        constrained_layout=True,
    )
    colormap = LinearSegmentedColormap.from_list(
        "csgni_advantage",
        ("#5D79A6", "#F4F4F2", "#B64342"),
    )
    norm = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=40.0)
    image = None
    panel_ids = ("a", "b", "c")
    for axis, (scale, title), panel_id in zip(axes, SCALES, panel_ids):
        values = np.asarray([
            [
                float(lookup[(scale, competitor, cf)]["median_advantage_percent_points"])
                for cf in CF_LEVELS
            ]
            for competitor in COMPETITORS
        ])
        image = axis.imshow(values, cmap=colormap, norm=norm, aspect="auto")
        for row_index, competitor in enumerate(COMPETITORS):
            for column_index, cf_level in enumerate(CF_LEVELS):
                row = lookup[(scale, competitor, cf_level)]
                value = float(row["median_advantage_percent_points"])
                text_color = "white" if value < -2.5 or value > 20 else PALETTE["black"]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.1f} ({row['wins']}/{row['losses']})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=5.5,
                )
        axis.set_xticks(range(len(CF_LEVELS)), CF_LEVELS)
        axis.set_yticks(range(len(COMPETITORS)), COMPETITORS)
        axis.set_title(f"{title} (n = 5 per cell)")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        add_panel_label(axis, panel_id)
    if image is None:
        raise RuntimeError("heatmap image was not created")
    colorbar = figure.colorbar(image, ax=axes, location="bottom", shrink=0.58, pad=0.12, aspect=35)
    colorbar.set_label("Median paired RPD advantage (percentage points; positive favors CSG-NI)")
    colorbar.outline.set_linewidth(0.5)
    base_path = FIGURE_ROOT / "supplementary_figure1_core_heterogeneity"
    save_publication_figure(
        figure,
        axes,
        panel_ids,
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": "Supplementary Figure 1",
            "core_conclusion": "The paired CSG-NI quality advantage is strongly scale dependent and is not uniform on Small instances.",
            "archetype": "quantitative grid",
            "source_data": str(source_path.relative_to(ROOT)),
            "source_sha256": file_sha256(source_path),
            "source_rows": len(rows),
            "replicate_unit": "Core instance; five independent instances per scale-by-CF cell, each summarized over five matched seeds",
            "center_and_spread": "cell value is the median paired RPD difference; parenthetical annotations give CSG-NI wins/losses among five instances",
            "exclusions": "none",
            "reporting_boundary": "post-hoc descriptive stratification; no cellwise hypothesis tests and no causal interpretation",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
