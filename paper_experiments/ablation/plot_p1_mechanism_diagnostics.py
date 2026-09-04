#!/usr/bin/env python3
"""Render non-inferential process diagnostics for Full and random-selection CSG-NI."""

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
RUN_PATH = ABLATION_ROOT / "processed_data/mechanism_run_summary.csv"
FAMILY_PATH = ABLATION_ROOT / "processed_data/mechanism_selection_family.csv"
STATS_PATH = ABLATION_ROOT / "processed_data/ablation_statistics.json"
FIGURE_ROOT = ABLATION_ROOT / "figures"
ARMS = ("CSG-NI Full", "Uniform full-bank selection")
SCALES = ("S", "M", "L")
COLORS = {"CSG-NI Full": PALETTE["csgni"], "Uniform full-bank selection": PALETTE["dcga"]}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_boxplot(axis, rows, field: str, ylabel: str) -> None:
    positions = []
    values = []
    colors = []
    for scale_index, scale in enumerate(SCALES, 1):
        for arm_index, arm in enumerate(ARMS):
            positions.append(scale_index + (-0.16 if arm_index == 0 else 0.16))
            group = [float(row[field]) for row in rows if row["scale"] == scale and row["arm"] == arm]
            if len(group) != 30:
                raise RuntimeError(f"expected 30 run summaries for {(scale, arm)}")
            values.append(group)
            colors.append(COLORS[arm])
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["black"], "linewidth": 0.9},
        whiskerprops={"color": PALETTE["neutral"], "linewidth": 0.7},
        capprops={"color": PALETTE["neutral"], "linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.42)
    axis.set_xticks((1, 2, 3), ("Small", "Medium", "Large"))
    axis.set_ylabel(ylabel)


def main() -> int:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    if stats.get("status") != "PASS_COMPLETE_REPLAY_VALIDATED":
        raise RuntimeError("P1 mechanism gate is not PASS")
    run_rows = read_csv(RUN_PATH)
    family_rows = read_csv(FAMILY_PATH)
    if len(run_rows) != 180:
        raise RuntimeError("mechanism diagnostics require 180 Full/random run summaries")
    families = sorted({row["origin_family"] for row in family_rows})
    palette = ("#B4C0D8", "#8495B8", "#6E9D97", "#D0A15C", "#B64342", "#767676")
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, 72.0 * POINTS_PER_MM / 72.0),
    )
    # Keep the three data panels on an explicit equal-width grid.  Automatic
    # layout expands the first gutter to accommodate its longer y-axis label,
    # which breaks the publication alignment contract even though the subplot
    # specification itself is regular.
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.88, wspace=0.42)

    family_axis = axes[0]
    bar_positions = np.arange(6)
    bottoms = np.zeros(6)
    labels = []
    for scale in SCALES:
        labels.extend((f"{scale}\nFull", f"{scale}\nUniform"))
    for family, color in zip(families, palette):
        fractions = []
        for scale in SCALES:
            for arm in ARMS:
                row = next(
                    item for item in family_rows
                    if item["scale"] == scale and item["arm"] == arm and item["origin_family"] == family
                )
                fractions.append(float(row["selection_fraction"]))
        family_axis.bar(bar_positions, fractions, bottom=bottoms, color=color, width=0.72, label=family)
        bottoms += np.asarray(fractions)
    family_axis.set_xticks(bar_positions, labels)
    family_axis.set_ylim(0, 1.0)
    family_axis.set_ylabel("Executed target-family fraction")
    family_axis.set_title("Target provenance")
    style_axis(family_axis)
    family_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, fontsize=5.4)
    add_panel_label(family_axis, "a")

    grouped_boxplot(axes[1], run_rows, "intervention_coverage", "Intervention coverage")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Frozen-gate use")
    style_axis(axes[1])
    add_panel_label(axes[1], "b")

    grouped_boxplot(axes[2], run_rows, "cumulative_ni_overhead_fraction", "NI overhead / runtime")
    axes[2].set_title("Mechanism overhead")
    style_axis(axes[2])
    add_panel_label(axes[2], "c")
    figure.text(
        0.755,
        0.12,
        "Full",
        color=COLORS[ARMS[0]],
        ha="center",
        va="center",
    )
    figure.text(
        0.885,
        0.12,
        "Uniform",
        color=COLORS[ARMS[1]],
        ha="center",
        va="center",
    )

    base_path = FIGURE_ROOT / "supplementary_p1_mechanism_diagnostics"
    save_publication_figure(
        figure,
        axes,
        ("a", "b", "c"),
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": "P1 supplementary mechanism diagnostics",
            "core_conclusion": "The learned-selection contrast is interpreted alongside target provenance, frozen-gate coverage and measured NI overhead.",
            "archetype": "quantitative grid",
            "source_data": [str(RUN_PATH.relative_to(ROOT)), str(FAMILY_PATH.relative_to(ROOT))],
            "source_sha256": {str(RUN_PATH.relative_to(ROOT)): file_sha256(RUN_PATH), str(FAMILY_PATH.relative_to(ROOT)): file_sha256(FAMILY_PATH)},
            "replicate_unit": "run summaries for panels b-c; panel a is a pooled state-level process composition",
            "center_and_spread": "panels b-c show median, IQR and 1.5-IQR whiskers over 30 runs per arm and scale",
            "exclusions": "none",
            "reporting_boundary": "descriptive process audit; state-level selection counts are not independent inferential replicates",
            "candidate_bank_boundary": "both arms request all 24 production rules and deduplicate only identical target sets",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
