#!/usr/bin/env python3
"""Render the P3 Core45 quality--runtime trade-off figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper_experiments"
DATA_ROOT = PAPER_ROOT / "processed_data/runtime"
FIGURE_ROOT = PAPER_ROOT / "figures"
METHODS = ("GA", "DCGA", "DABC", "LG_HGA", "CSG-NI")
COLORS = {
    "GA": PALETTE["ga"],
    "DCGA": PALETTE["dcga"],
    "DABC": PALETTE["dabc"],
    "LG_HGA": PALETTE["lghga"],
    "CSG-NI": PALETTE["csgni"],
}
SCALE_ORDER = ("S", "M", "L")
MARKERS = {"S": "o", "M": "s", "L": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def error_extent(row: dict[str, str], stem: str) -> tuple[list[float], list[float]]:
    median = float(row[f"median_{stem}"])
    return [median - float(row[f"q25_{stem}"])], [float(row[f"q75_{stem}"]) - median]


def main() -> int:
    summary_path = DATA_ROOT / "p3_runtime_summary.json"
    overall_path = DATA_ROOT / "core_quality_runtime_overall.csv"
    scale_path = DATA_ROOT / "core_quality_runtime_by_scale.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS_DESCRIPTIVE_EXISTING_CORE45":
        raise RuntimeError("P3 processed-data gate is not PASS")
    overall = read_csv(overall_path)
    by_scale = read_csv(scale_path)
    if len(overall) != 5 or len(by_scale) != 15:
        raise RuntimeError("P3 requires five overall and fifteen scale summaries")

    overall_lookup = {row["display_method"]: row for row in overall}
    scale_lookup = {(row["display_method"], row["scale"]): row for row in by_scale}
    if set(overall_lookup) != set(METHODS):
        raise RuntimeError("unexpected P3 manuscript method identities")

    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(width_in, 74.0 * POINTS_PER_MM / 72.0),
        constrained_layout=True,
    )
    overall_axis, scale_axis = axes

    for method in METHODS:
        row = overall_lookup[method]
        x = float(row["median_runtime_budget_fraction"])
        y = float(row["median_rpd_percent"])
        xerr = error_extent(row, "runtime_budget_fraction")
        yerr = error_extent(row, "rpd_percent")
        overall_axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=4.8 if method == "CSG-NI" else 4.2,
            mfc=COLORS[method],
            mec="white",
            mew=0.55,
            ecolor=COLORS[method],
            elinewidth=0.85,
            capsize=2.0,
            zorder=4 if method == "CSG-NI" else 3,
        )
    overall_axis.axvline(1.0, color=PALETTE["neutral"], linewidth=0.7, linestyle="--", zorder=1)
    overall_axis.set_xlim(0.0, 1.08)
    overall_axis.set_ylim(bottom=-0.5)
    overall_axis.set_xlabel("Median realized budget fraction")
    overall_axis.set_ylabel("Median instance-level RPD (%)")
    overall_axis.set_title("Overall trade-off (45 instances)")
    style_axis(overall_axis)
    add_panel_label(overall_axis, "a")

    for method in METHODS:
        rows = [scale_lookup[(method, scale)] for scale in SCALE_ORDER]
        x = [float(row["median_runtime_seconds"]) for row in rows]
        y = [float(row["median_rpd_percent"]) for row in rows]
        scale_axis.plot(x, y, color=COLORS[method], linewidth=1.1, zorder=2)
        for scale, row, x_value, y_value in zip(SCALE_ORDER, rows, x, y):
            xerr = error_extent(row, "runtime_seconds")
            yerr = error_extent(row, "rpd_percent")
            scale_axis.errorbar(
                x_value,
                y_value,
                xerr=xerr,
                yerr=yerr,
                fmt=MARKERS[scale],
                ms=4.7 if method == "CSG-NI" else 4.0,
                mfc=COLORS[method],
                mec="white",
                mew=0.5,
                ecolor=COLORS[method],
                elinewidth=0.65,
                capsize=1.6,
                zorder=4 if method == "CSG-NI" else 3,
            )
    scale_axis.set_xscale("log")
    scale_axis.set_xlim(8, 520)
    scale_axis.set_xticks((10, 100, 300), ("10", "100", "300"))
    scale_axis.minorticks_off()
    scale_axis.tick_params(axis="x", length=0)
    scale_axis.set_ylim(bottom=-0.5)
    scale_axis.set_yticks((0, 10, 20, 35))
    scale_axis.set_xlabel("Median realized runtime (s, log scale)")
    # Panel (a) carries the shared response-axis label; omitting the duplicate
    # keeps the inter-panel gutter clear at final journal width.
    scale_axis.set_title("Scale-dependent trajectories (15 instances each)")
    style_axis(scale_axis)
    scale_axis.grid(False)
    scale_axis.tick_params(axis="y", length=0, pad=4.0)
    add_panel_label(scale_axis, "b")

    method_handles = [
        Line2D([0], [0], marker="o", color=COLORS[method], markerfacecolor=COLORS[method],
               markeredgecolor="white", markeredgewidth=0.5, label=method, linewidth=1.1)
        for method in METHODS
    ]
    scale_handles = [
        Line2D([0], [0], marker=MARKERS[scale], color=PALETTE["neutral"],
               markerfacecolor=PALETTE["neutral"], markeredgecolor="white",
               markeredgewidth=0.5, label={"S": "Small", "M": "Medium", "L": "Large"}[scale],
               linewidth=0)
        for scale in SCALE_ORDER
    ]
    figure.legend(
        handles=method_handles + scale_handles,
        loc="outside upper center",
        ncol=8,
        columnspacing=1.05,
        handletextpad=0.4,
    )

    base_path = FIGURE_ROOT / "figure3_quality_runtime_tradeoff"
    save_publication_figure(
        figure,
        axes,
        ("a", "b"),
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": "Figure 3",
            "core_conclusion": "Core45 quality differences are interpreted jointly with realized wall-clock cost and source-compatible stopping rules.",
            "archetype": "quantitative grid",
            "source_data": [
                str(overall_path.relative_to(ROOT)),
                str(scale_path.relative_to(ROOT)),
            ],
            "source_sha256": {
                str(overall_path.relative_to(ROOT)): file_sha256(overall_path),
                str(scale_path.relative_to(ROOT)): file_sha256(scale_path),
            },
            "replicate_unit": "Core instance; each point summarizes instance medians over five matched seeds",
            "center_and_spread": "median and interquartile range across 45 overall or 15 scale-specific instances",
            "exclusions": "none",
            "reporting_boundary": "descriptive quality-runtime analysis; decoder evaluations excluded from the common plot because evaluation semantics differ across algorithm families; no Pareto claim",
            "termination_disclosure": "LG_HGA retains the original MAXGEN=100 cap and commonly terminates before the wall-clock ceiling",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
