#!/usr/bin/env python3
"""Plot Figure 1 after the complete Core BKS/RPD integrity gate passes."""

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
DATA_ROOT = PAPER_ROOT / "processed_data/main"
FIGURE_ROOT = PAPER_ROOT / "figures"
METHODS = (
    ("GA", "GA", PALETTE["ga"]),
    ("Adapted DCGA", "DCGA", PALETTE["dcga"]),
    ("DABC-RIACRSP", "DABC", PALETTE["dabc"]),
    ("LG_HGA-RIACRSP-v2-N4M", "LG_HGA", PALETTE["lghga"]),
    ("CSG-NI Phase6H provisional", "CSG-NI", PALETTE["csgni"]),
)


def load_rows() -> tuple[list[dict[str, str]], Path]:
    manifest_path = DATA_ROOT / "analysis_manifest.json"
    summary_path = DATA_ROOT / "main_instance_summary.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("complete Core BKS/RPD before generating Figure 1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H":
        raise RuntimeError("Core BKS/RPD gate is not PASS")
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_methods = {method for method, _, _ in METHODS}
    keys = {(row["method"], row["instance_id"]) for row in rows}
    if (
        len(rows) != 225
        or len(keys) != 225
        or {row["method"] for row in rows} != expected_methods
        or {row["scale"] for row in rows} != {"S", "M", "L"}
    ):
        raise RuntimeError("expected one complete 5 method x 45 instance Core summary")
    for row in rows:
        values = (
            float(row["median_rpd_percent"]),
            float(row["feasibility_rate"]),
            float(row["draft_bks"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("non-finite Figure 1 source value")
        if values[0] < -1e-9 or values[1] != 1.0 or values[2] <= 0:
            raise RuntimeError("invalid RPD, feasibility, or BKS in Figure 1 source")
    return rows, summary_path


def main() -> int:
    rows, summary_path = load_rows()
    font_path = apply_publication_style()
    width_in = FINAL_WIDTH_MM * POINTS_PER_MM / 72.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, 69.0 * POINTS_PER_MM / 72.0),
        sharey=True,
        constrained_layout=True,
    )
    panel_ids = ("a", "b", "c")
    scale_labels = {"S": "Small", "M": "Medium", "L": "Large"}
    for axis, panel_id, scale in zip(axes, panel_ids, ("S", "M", "L")):
        scale_rows = [row for row in rows if row["scale"] == scale]
        ordered_instances = sorted({row["instance_id"] for row in scale_rows})
        if len(ordered_instances) != 15:
            raise RuntimeError(f"expected 15 {scale} instances")
        by_key = {(row["method"], row["instance_id"]): row for row in scale_rows}
        values = [
            [float(by_key[(method, instance)]["median_rpd_percent"]) for instance in ordered_instances]
            for method, _, _ in METHODS
        ]
        boxes = axis.boxplot(
            values,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE["black"], "linewidth": 1.0},
            whiskerprops={"color": PALETTE["neutral"], "linewidth": 0.7},
            capprops={"color": PALETTE["neutral"], "linewidth": 0.7},
        )
        point_offsets = np.linspace(-0.11, 0.11, len(ordered_instances))
        for method_index, ((_, _, color), method_values) in enumerate(zip(METHODS, values), start=1):
            boxes["boxes"][method_index - 1].set(
                facecolor=color,
                edgecolor=PALETTE["black"],
                linewidth=0.7,
            )
            axis.scatter(
                method_index + point_offsets,
                method_values,
                s=7,
                facecolor=color,
                edgecolor=PALETTE["black"],
                linewidth=0.25,
                zorder=3,
            )
        axis.axhline(
            0,
            color=PALETTE["neutral"],
            linewidth=0.65,
            linestyle=(0, (3, 2)),
        )
        axis.set_xticks(
            range(1, len(METHODS) + 1),
            [label for _, label, _ in METHODS],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_title(f"{scale_labels[scale]} (n = 15)")
        add_panel_label(axis, panel_id)
        style_axis(axis)
    axes[0].set_ylabel("Per-instance median RPD (%)")
    base_path = FIGURE_ROOT / "figure1_quality_distribution"
    save_publication_figure(
        figure,
        axes,
        panel_ids,
        base_path,
        {
            "schema": "initial-manuscript-figure-qa-v1",
            "figure": 1,
            "core_conclusion": "Relative solution-quality differences among the five methods are assessed across all three Core45 scales.",
            "archetype": "quantitative grid",
            "source_data": str(summary_path.relative_to(ROOT)),
            "source_sha256": file_sha256(summary_path),
            "source_rows": len(rows),
            "replicate_unit": "Core instance; each point is the median RPD across five matched seeds",
            "center_and_spread": "boxplot median and interquartile range; whiskers use the Matplotlib 1.5-IQR rule",
            "exclusions": "none",
            "display_transform": "deterministic horizontal offsets expose all 15 instance summaries per method; no sampling",
            "reporting_boundary": "Phase6H CSG-NI is provisional; formal inference is reported separately using 45 paired instances",
            "font_file": font_path.name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
