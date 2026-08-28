#!/usr/bin/env python3
"""Audit structural properties of all frozen canonical instances."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.loader import load_instance

OUT = ROOT / "outputs/phase5c/benchmark_audit"


DEFINITIONS = {
    "F_route_mean": "Mean across operations of eligible-island count divided by total islands.",
    "R_full_op": "Fraction of operations eligible on every island.",
    "R_high_op": "Fraction of operations eligible on at least 80% of islands.",
    "F_cap_mean": "Mean across islands of supported-configuration count divided by total configurations.",
    "R_full_island": "Fraction of islands supporting every configuration.",
    "capability_heterogeneity": "Population standard deviation of island capability-coverage fractions.",
    "processing_CV_mean": "Mean eligible-processing-time coefficient of variation over flexible operations.",
    "RI": "Mean nonzero reconfiguration time divided by mean eligible processing time.",
    "W_transport_intensity": "Mean loaded W travel time divided by mean eligible processing time.",
    "F_transport_intensity": "Mean F round-trip time divided by mean eligible processing time.",
    "mean_ready_set_size": "Mean ready-set size in lexicographic precedence-only release simulation.",
    "resource_pressure_proxies": "Operations/islands per AGV are structural scarcity proxies, not utilization.",
}


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _box(rows, metric, ylabel, name):
    families = ("Brandimarte", "Hurink E", "Hurink R", "Hurink V")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.boxplot([[float(row[metric]) for row in rows if row["family"] == family] for family in families], tick_labels=families, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, name)


def main() -> None:
    paths = sorted(
        path for path in (ROOT / "instances/canonical/RCIAS-2.0").rglob("*.json")
        if path.name not in {"manifest.json", "generation_config.json"}
    )
    if len(paths) != 130:
        raise RuntimeError(f"expected 130 canonical instances, found {len(paths)}")
    rows = [benchmark_metrics(load_instance(path)) for path in paths]
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "instance_metrics.csv", rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    families = sorted({row["family"] for row in rows})
    family_rows = []
    for family in families:
        selected = [row for row in rows if row["family"] == family]
        family_rows.append({
            "family": family,
            "instances": len(selected),
            **{f"{metric}_mean": mean(float(row[metric]) for row in selected) for metric in numeric},
            **{f"{metric}_median": median(float(row[metric]) for row in selected) for metric in numeric},
        })
    _write_csv(OUT / "family_summary.csv", family_rows)
    (OUT / "metric_definitions.json").write_text(json.dumps(DEFINITIONS, indent=2, sort_keys=True) + "\n")
    summary = {
        "canonical_instances": 130,
        "performance_metrics_used": False,
        "families": {family: sum(row["family"] == family for row in rows) for family in families},
        "family_key_means": {
            family: {
                metric: mean(float(row[metric]) for row in rows if row["family"] == family)
                for metric in ("F_route_mean", "R_full_op", "R_high_op", "F_cap_mean", "R_full_island", "processing_CV_mean", "RI", "W_transport_intensity", "F_transport_intensity")
            } for family in families
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    plt.rcParams.update({"font.size": 10})
    _box(rows, "F_route_mean", "Mean routing flexibility", "01_routing_flexibility_by_family")
    _box(rows, "R_full_op", "Full-flexibility operation ratio", "02_full_flexibility_ratio_by_family")
    _box(rows, "F_cap_mean", "Mean island capability coverage", "03_capability_coverage_by_family")
    _box(rows, "processing_CV_mean", "Mean processing-time CV", "04_processing_heterogeneity_by_family")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = {"Brandimarte": "#2563eb", "Hurink E": "#059669", "Hurink R": "#f59e0b", "Hurink V": "#dc2626"}
    for family in families:
        selected = [row for row in rows if row["family"] == family]
        ax.scatter([row["F_route_mean"] for row in selected], [row["F_cap_mean"] for row in selected], label=family, alpha=0.75, color=colors[family])
    ax.set_xlabel("Mean routing flexibility")
    ax.set_ylabel("Mean capability coverage")
    ax.legend(frameon=False)
    _save(fig, "05_route_vs_capability_scatter")
    _box(rows, "RI", "Reconfiguration intensity RI", "06_reconfiguration_intensity_by_family")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(families))
    ax.bar(x - 0.18, [mean(row["W_transport_intensity"] for row in rows if row["family"] == family) for family in families], 0.36, label="W")
    ax.bar(x + 0.18, [mean(row["F_transport_intensity"] for row in rows if row["family"] == family) for family in families], 0.36, label="F")
    ax.set_xticks(x, families)
    ax.set_ylabel("Travel time / processing time")
    ax.legend(frameon=False)
    _save(fig, "07_transport_intensity_by_family")
    correlation_metrics = ("F_route_mean", "R_full_op", "F_cap_mean", "R_full_island", "processing_CV_mean", "RI", "W_transport_intensity", "F_transport_intensity")
    matrix = np.corrcoef(np.array([[float(row[m]) for m in correlation_metrics] for row in rows]), rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation_metrics)), correlation_metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation_metrics)), correlation_metrics)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    _save(fig, "08_structural_metric_correlation")
    print("PHASE5C_BENCHMARK_AUDIT_COMPLETE instances=130 figures=8")


if __name__ == "__main__":
    main()
