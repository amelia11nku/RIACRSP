#!/usr/bin/env python3
"""Generate the six compact Phase 6D CSG validation figures."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

_MPL_CACHE = Path("/tmp/ri_acrsp_matplotlib")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs/csg_v1_schema.json"
VALIDATION = ROOT / "outputs/phase6d/validation"
PROFILING = ROOT / "outputs/phase6d/profiling"
EXAMPLES = ROOT / "outputs/phase6d/examples"
DEFAULT_OUTPUT = ROOT / "outputs/phase6d/figures"
COLORS = {
    "OP": "#3666A7", "ISLAND": "#E18A3B", "CONFIG": "#A35FA5",
    "W_AGV": "#399E5A", "F_AGV": "#5C9EAD", "W_EVENT": "#7BC87C",
    "F_EVENT": "#87C7D4", "RECONF_EVENT": "#D85C5C",
}
CLASS_COLORS = {
    "STATIC_CONTEXT": "#8B8B8B", "REALIZED_RESOURCE_ORDER": "#2D6A9F",
    "TEMPORAL_CAUSAL": "#C47A16", "SYNCHRONIZATION": "#C33C54",
}


def save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def schema_overview(schema: dict, output: Path) -> None:
    positions = {
        "CONFIG": (0.12, 0.80), "ISLAND": (0.38, 0.80), "OP": (0.72, 0.62),
        "RECONF_EVENT": (0.38, 0.45), "W_EVENT": (0.60, 0.25), "F_EVENT": (0.83, 0.25),
        "W_AGV": (0.50, 0.06), "F_AGV": (0.88, 0.06),
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for node_type, (x, y) in positions.items():
        feature_count = len(schema["node_types"][node_type]["features"])
        ax.text(
            x, y, f"{node_type}\n{feature_count} features", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS[node_type], "edgecolor": "white"},
            zorder=3,
        )
    for edge in schema["edge_types"]:
        source = positions[edge["source"]]; target = positions[edge["target"]]
        if source == target:
            continue
        arrow = FancyArrowPatch(
            source, target, arrowstyle="-|>", mutation_scale=8,
            color=CLASS_COLORS[edge["class"]], alpha=0.30, linewidth=1,
            connectionstyle="arc3,rad=0.08", zorder=1,
        )
        ax.add_patch(arrow)
    handles = [plt.Line2D([0], [0], color=color, lw=3, label=name.replace("_", " ").title())
               for name, color in CLASS_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2)
    ax.set_title("CSG-1.0: static alternatives and realized synchronization", fontsize=15, pad=15)
    save(fig, output, "Fig01_CSG_schema_overview")


def small_example(output: Path) -> None:
    manifest = pd.read_csv(EXAMPLES / "example_manifest.csv")
    directory = ROOT / manifest.loc[manifest.role == "small_state", "directory"].iloc[0]
    graph = json.loads((directory / "tables/graph.json").read_text())
    node_counts = {key: len(value) for key, value in graph["nodes"].items()}
    relation = pd.read_csv(directory / "relation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(list(node_counts), list(node_counts.values()), color=[COLORS[key] for key in node_counts])
    axes[0].tick_params(axis="x", rotation=55); axes[0].set_ylabel("Node count")
    axes[0].set_title("Small-state node composition")
    relation = relation.sort_values("edge_count", ascending=True)
    axes[1].barh(relation.edge_type.str.split("__").str[1], relation.edge_count, color="#577590")
    axes[1].set_xlabel("Edge count"); axes[1].set_title("Canonical relation distribution")
    save(fig, output, "Fig02_small_state_CSG_example")


def synchronization_neighborhood(output: Path) -> None:
    manifest = pd.read_csv(EXAMPLES / "example_manifest.csv")
    row = manifest[manifest.role == "cross_resource_synchronization"].iloc[0]
    graph = json.loads((ROOT / row.directory / "tables/graph.json").read_text())
    fan_in = Counter(
        edge["target_key"]
        for edge_type, edges in graph["edges"].items()
        if edge_type.endswith("__ENABLES__OP")
        for edge in edges
    )
    operation = max(fan_in, key=lambda key: (fan_in[key], key))
    selected_edges = []
    for edge_type, edges in graph["edges"].items():
        source_type, relation, target_type = edge_type.split("__")
        if relation not in {"ENABLES", "PRECEDES", "PRODUCT_NEXT", "ISLAND_NEXT"}:
            continue
        for edge in edges:
            if (target_type == "OP" and edge["target_key"] == operation) or (
                source_type == "OP" and edge["source_key"] == operation
            ):
                selected_edges.append((source_type, edge["source_key"], relation, target_type, edge["target_key"], edge["features"]))
    nodes = {("OP", operation)}
    for source_type, source, _, target_type, target, _ in selected_edges:
        nodes.add((source_type, source)); nodes.add((target_type, target))
    neighbors = sorted(nodes - {("OP", operation)})
    positions = {("OP", operation): (0.5, 0.5)}
    for index, node in enumerate(neighbors):
        angle = 2 * np.pi * index / max(len(neighbors), 1)
        positions[node] = (0.5 + 0.40 * np.cos(angle), 0.5 + 0.40 * np.sin(angle))
    fig, ax = plt.subplots(figsize=(9, 8)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for source_type, source, relation, target_type, target, features in selected_edges:
        left, right = positions[(source_type, source)], positions[(target_type, target)]
        ax.add_patch(FancyArrowPatch(left, right, arrowstyle="-|>", mutation_scale=12, color="#666", alpha=.65))
        middle = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        gap = features.get("temporal_gap")
        label = relation if gap is None else f"{relation}\ngap={gap:g}"
        ax.text(*middle, label, fontsize=7, ha="center", va="center", backgroundcolor="white")
    for node, position in positions.items():
        node_type, key = node
        ax.scatter(*position, s=850 if node_type == "OP" else 520, color=COLORS[node_type], edgecolor="white", zorder=3)
        ax.text(*position, f"{node_type}\n{key}", color="white", fontsize=7, ha="center", va="center", zorder=4)
    ax.set_title("Incoming synchronization and local causal neighborhood")
    save(fig, output, "Fig03_synchronization_neighborhood_example")


def scaling_figures(output: Path) -> None:
    detail = pd.read_csv(PROFILING / "profiling_per_state.csv")
    colors = detail.scale.map({"S": "#4C78A8", "M": "#F58518", "L": "#E45756"})
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(detail.operation_count, detail.node_count, c=colors, label="nodes", marker="o")
    ax.scatter(detail.operation_count, detail.edge_count, c=colors, marker="^", alpha=.75, label="edges")
    ax.set_xlabel("Operations"); ax.set_ylabel("Graph elements"); ax.legend(frameon=False)
    ax.set_title("CSG size scaling")
    save(fig, output, "Fig04_node_edge_scaling")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(detail.operation_count, detail.construction_seconds * 1000, c=colors)
    fit = np.polyfit(detail.operation_count, detail.construction_seconds * 1000, 1)
    x = np.linspace(detail.operation_count.min(), detail.operation_count.max(), 100)
    ax.plot(x, np.polyval(fit, x), color="#333", linestyle="--", label="linear trend")
    ax.set_xlabel("Operations"); ax.set_ylabel("Construction time (ms)"); ax.legend(frameon=False)
    ax.set_title("Framework-neutral construction time")
    save(fig, output, "Fig05_construction_time_scaling")


def relation_distribution(output: Path) -> None:
    edges = pd.read_csv(VALIDATION / "edge_type_summary.csv").sort_values("total_count")
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = edges.edge_type.str.split("__").str[1] + " (" + edges.edge_type.str.split("__").str[0] + ")"
    ax.barh(labels, edges.total_count, color="#577590")
    ax.set_xlabel("Edges across validation sample")
    ax.set_title("CSG relation-type distribution")
    save(fig, output, "Fig06_relation_type_distribution")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA_PATH.read_text())
    schema_overview(schema, args.output)
    small_example(args.output)
    synchronization_neighborhood(args.output)
    scaling_figures(args.output)
    relation_distribution(args.output)
    print("PHASE6D_CSG_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()
