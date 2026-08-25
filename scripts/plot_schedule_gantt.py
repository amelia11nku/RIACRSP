#!/usr/bin/env python3
"""Plot a resource-level validation Gantt from resource_timeline.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {
    "RECONFIGURATION": "#E69F00",
    "PROCESSING": "#0072B2",
    "EMPTY_REPOSITION": "#999999",
    "LOADED_TRANSPORT": "#009E73",
    "OUTBOUND_DELIVERY": "#CC79A7",
    "RETURN_TO_WH": "#56B4E9",
}
HATCHES = {"RECONFIGURATION": "///", "EMPTY_REPOSITION": "..", "RETURN_TO_WH": "\\\\"}


def plot_resource_timeline(csv_path: str | Path, png_path: str | Path, pdf_path: str | Path) -> None:
    """Render the exact CSV activities; no schedule times are recomputed here."""

    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("resource timeline is empty")
    type_order = {"ASSEMBLY_ISLAND": 0, "W_AGV": 1, "F_AGV": 2}
    resources = sorted(
        {(row["resource_type"], row["resource_id"]) for row in rows},
        key=lambda item: (type_order[item[0]], item[1]),
    )
    position = {resource: index for index, resource in enumerate(resources)}
    figure_height = max(4.5, 0.68 * len(resources) + 1.7)
    fig, ax = plt.subplots(figsize=(12.5, figure_height))
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        duration = end - start
        if duration <= 0:
            continue
        activity = row["activity_type"]
        y = position[(row["resource_type"], row["resource_id"])]
        ax.barh(
            y, duration, left=start, height=0.58, color=COLORS[activity],
            edgecolor="#222222", linewidth=0.55, hatch=HATCHES.get(activity), zorder=3,
        )
        if activity == "RECONFIGURATION":
            label = f"{row['from_configuration']}→{row['to_configuration']}"
        elif activity in {"PROCESSING", "LOADED_TRANSPORT", "OUTBOUND_DELIVERY"}:
            label = row["operation"]
        else:
            label = ""
        if label and (duration >= 3 or activity == "RECONFIGURATION"):
            short_reconfiguration = activity == "RECONFIGURATION" and duration < 7
            ax.text(
                start + duration / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6.5 if short_reconfiguration else 7.5,
                rotation=90 if short_reconfiguration else 0,
                color="white" if activity in {"PROCESSING", "LOADED_TRANSPORT"} else "black",
                zorder=4,
            )
    labels = [
        f"Island {resource_id}" if resource_type == "ASSEMBLY_ISLAND" else f"{resource_type.replace('_', '-')} {resource_id}"
        for resource_type, resource_id in resources
    ]
    ax.set_yticks(range(len(resources)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    legend_order = [
        "RECONFIGURATION", "PROCESSING", "EMPTY_REPOSITION", "LOADED_TRANSPORT",
        "OUTBOUND_DELIVERY", "RETURN_TO_WH",
    ]
    handles = [
        Patch(facecolor=COLORS[item], edgecolor="#222222", hatch=HATCHES.get(item), label=item.replace("_", " ").title())
        for item in legend_order if any(row["activity_type"] == item for row in rows)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3,
              frameon=False, fontsize=8)
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.input.parent
    plot_resource_timeline(
        args.input,
        args.png or output_dir / "gantt.png",
        args.pdf or output_dir / "gantt.pdf",
    )
    print(f"Wrote {args.png or output_dir / 'gantt.png'} and {args.pdf or output_dir / 'gantt.pdf'}")


if __name__ == "__main__":
    main()
