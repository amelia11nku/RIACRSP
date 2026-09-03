#!/usr/bin/env python3
"""Plot Figure 2 from audited Phase6H CSG-NI and ALNS efficiency data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from paper_figure_style import (
    FINAL_WIDTH_MM,
    FuncFormatter,
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
DATA_ROOT = PAPER_ROOT / "processed_data/efficiency"
FIGURE_ROOT = PAPER_ROOT / "figures"
METHODS = (
    ("ALNS", "ALNS", PALETTE["alns"], "o"),
    ("PHASE6H_CSGNI", "CSG-NI (Phase6H provisional)", PALETTE["csgni"], "s"),
)
BAND_COLORS = {"ALNS": "#E5E9F0", "PHASE6H_CSGNI": "#F1DFDF"}
ERROR_COLORS = {"ALNS": "#A7B0C1", "PHASE6H_CSGNI": "#D49E9D"}


def format_decoder_evaluations(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    if value >= 1:
        return f"{value:g}"
    return ""


def load_rows() -> tuple[list[dict[str, str]], Path]:
    inventory_path = DATA_ROOT / "efficiency_inventory.json"
    curves_path = DATA_ROOT / "anytime_curves.csv"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "PASS_REUSED_PHASE6H_CAL_HOLDOUT":
        raise RuntimeError("efficiency data integrity gate is not PASS")
    with curves_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (row["method"], row["instance_id"], row["seed"], row["budget_fraction"])
        for row in rows
    }
    if len(rows) != 540 or len(keys) != 540:
        raise RuntimeError("expected 540 unique audited anytime checkpoint rows")
    if {row["method"] for row in rows} != {method for method, _, _, _ in METHODS}:
        raise RuntimeError("Figure 2 contains methods outside the CSG-NI/ALNS efficiency contract")
    return rows, curves_path


def aggregate(
    rows: list[dict[str, str]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fractions = sorted(
        {float(row["budget_fraction"]) for row in rows if row["method"] == method}
    )
    if fractions != [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError(f"unexpected normalized budget grid for {method}: {fractions}")
    median_gap, lower_gap, upper_gap = [], [], []
    median_evaluations, lower_evaluations, upper_evaluations = [], [], []
    for fraction in fractions:
        group = [
            row
            for row in rows
            if row["method"] == method
            and float(row["budget_fraction"]) == fraction
            and row["incumbent_available"] == "True"
        ]
        if len(group) != 45:
            raise RuntimeError(f"incomplete anytime checkpoint for {method} at {fraction}")
        gaps = np.asarray(
            [100.0 * float(row["relative_gap_to_reference"]) for row in group],
            dtype=float,
        )
        evaluations = np.asarray(
            [float(row["decoder_evaluations"]) for row in group], dtype=float
        )
        if (
            not np.all(np.isfinite(gaps))
            or not np.all(np.isfinite(evaluations))
            or np.any(gaps < -1e-9)
            or np.any(evaluations <= 0)
        ):
            raise RuntimeError("invalid gap or decoder-evaluation value in Figure 2 source")
        lower_gap.append(float(np.percentile(gaps, 25)))
        median_gap.append(float(np.median(gaps)))
        upper_gap.append(float(np.percentile(gaps, 75)))
        lower_evaluations.append(float(np.percentile(evaluations, 25)))
        median_evaluations.append(float(np.median(evaluations)))
        upper_evaluations.append(float(np.percentile(evaluations, 75)))
    arrays = tuple(
        np.asarray(values)
        for values in (
            fractions,
            median_gap,
            lower_gap,
            upper_gap,
            median_evaluations,
            lower_evaluations,
            upper_evaluations,
        )
    )
    if np.any(np.diff(arrays[4]) < 0):
        raise RuntimeError(f"median decoder evaluations are not monotone for {method}")
    return arrays  # type: ignore[return-value]


def main() -> int:
    rows, curves_path = load_rows()
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(width_in, 68.0 * POINTS_PER_MM / 72.0),
        sharey=True,
        constrained_layout=True,
    )
    handles = []
    for raw, display, color, marker in METHODS:
        (
            fraction,
            gap,
            gap_low,
            gap_high,
            evaluations,
            eval_low,
            eval_high,
        ) = aggregate(rows, raw)
        handle = axes[0].plot(
            fraction,
            gap,
            color=color,
            marker=marker,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=display,
            zorder=3,
        )[0]
        handles.append(handle)
        axes[0].fill_between(
            fraction,
            gap_low,
            gap_high,
            facecolor=BAND_COLORS[raw],
            edgecolor=color,
            linewidth=0.35,
            zorder=1,
        )
        axes[1].plot(
            evaluations,
            gap,
            color=color,
            marker=marker,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        axes[1].errorbar(
            evaluations,
            gap,
            xerr=np.vstack((evaluations - eval_low, eval_high - evaluations)),
            yerr=np.vstack((gap - gap_low, gap_high - gap)),
            fmt="none",
            ecolor=ERROR_COLORS[raw],
            elinewidth=0.55,
            capsize=1.6,
            zorder=2,
        )
    axes[0].set_xlabel("Normalized runtime, t/T")
    axes[0].set_ylabel("Median gap to pooled BKS (%)")
    axes[0].set_title("Anytime convergence (median and IQR)")
    axes[1].set_xlabel("Median decoder evaluations (log scale)")
    axes[1].set_title("Search effort (median and IQR)")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_decoder_evaluations))
    for axis, panel_id in zip(axes, ("a", "b")):
        add_panel_label(axis, panel_id)
        style_axis(axis)
    figure.legend(
        handles,
        [display for _, display, _, _ in METHODS],
        loc="outside upper center",
        ncols=2,
        handlelength=1.8,
        columnspacing=1.4,
    )
    base_path = FIGURE_ROOT / "figure2_anytime_efficiency"
    save_publication_figure(
        figure,
        axes,
        ("a", "b"),
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": 2,
            "core_conclusion": "Phase6H CSG-NI is compared with ALNS in both wall-clock anytime convergence and decoder-search effort.",
            "archetype": "quantitative grid",
            "source_data": str(curves_path.relative_to(ROOT)),
            "source_sha256": file_sha256(curves_path),
            "source_rows": len(rows),
            "replicate_unit": "45 matched runs: nine CAL-HOLDOUT instances x five seeds per method",
            "center_and_spread": "median and interquartile range at six preregistered normalized-budget checkpoints",
            "exclusions": "none; every method/checkpoint group contains all 45 runs",
            "display_transform": "relative gap multiplied by 100; decoder-evaluation axis is logarithmic; no interpolation or sampling",
            "reporting_boundary": "descriptive efficiency evidence on CAL-HOLDOUT; ALNS is not part of the five-method Core ranking and Phase6H CSG-NI remains provisional",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
