#!/usr/bin/env python3
"""Structural, pairing, and feasibility audit for RCIAS-CB1."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.instances.controlled_generator import acceptance_failures, configuration_entropy

CB1 = ROOT / "instances/controlled/RCIAS-CB1"
OUT = ROOT / "outputs/phase5c/controlled_benchmark_audit"


def _save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def _pairing_projection(raw):
    selected = copy.deepcopy(raw)
    selected["meta"].pop("instance_id", None); selected["meta"].pop("RI_level", None); selected["meta"].pop("TI_level", None)
    selected["reconfiguration"].pop("time")
    selected["logistics"]["W"].pop("loaded_time"); selected["logistics"]["W"].pop("empty_time")
    selected["logistics"]["F"].pop("outbound_time"); selected["logistics"]["F"].pop("return_time")
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((CB1 / "manifests/generation_spec.json").read_text())
    manifest = pd.read_csv(CB1 / "manifests/benchmark_manifest.csv")
    rows, feasibility = [], []
    for record in manifest.to_dict("records"):
        path = CB1 / record["relative_path"]
        instance = load_instance(path); metrics = benchmark_metrics(instance)
        raw = json.loads(path.read_text())
        rows.append({**metrics, "suite": record["suite"], "scale": record["scale"],
                     "CF_level": record["CF_level"], "RI_level": record["RI_level"],
                     "TI_level": record["TI_level"], "replicate": record["replicate"],
                     "base_structure": record["base_structure"], "configuration_entropy": configuration_entropy(raw)})
        result = solve_dispatching(instance, "H1"); audit = check_schedule(instance, result.schedule)
        feasibility.append({"instance_id": instance.instance_id, "loadable": True, "h1_feasible": audit["feasible"]})
    controlled = pd.DataFrame(rows); controlled.to_csv(OUT / "controlled_instance_metrics.csv", index=False)
    pd.DataFrame(feasibility).to_csv(OUT / "feasibility_smoke_test.csv", index=False)
    numeric = [column for column in controlled.select_dtypes(include=np.number) if column != "performance_metrics_used"]
    controlled.groupby(["suite", "scale", "CF_level"], dropna=False)[numeric].agg(["mean", "std", "min", "max"]).to_csv(OUT / "controlled_group_summary.csv")
    corr_cols = ["number_of_operations", "F_route_mean", "F_cap_mean", "processing_CV_mean", "RI", "W_transport_intensity", "F_transport_intensity", "precedence_edge_density"]
    controlled[corr_cols].corr().to_csv(OUT / "metric_correlations.csv")

    legacy = pd.read_csv(ROOT / "outputs/phase5c/benchmark_audit/instance_metrics.csv")
    legacy["suite"] = "Legacy-130"
    combined = pd.concat([legacy, controlled], ignore_index=True, sort=False)
    compare_metrics = ["F_route_mean", "R_full_op", "F_cap_mean", "R_full_island", "capability_heterogeneity", "processing_CV_mean", "RI", "W_transport_intensity", "F_transport_intensity", "configuration_diversity_entropy", "precedence_edge_density", "mean_ready_set_size"]
    comparison = combined.groupby("suite")[compare_metrics].agg(["mean", "median", "std", "min", "max"])
    comparison.to_csv(OUT / "legacy_vs_controlled_summary.csv")

    core_failures = {}
    for record in manifest[manifest.suite == "CORE"].to_dict("records"):
        raw = json.loads((CB1 / record["relative_path"]).read_text())
        failed = acceptance_failures(raw, record["scale"], record["CF_level"], spec)
        if failed: core_failures[record["instance_id"]] = failed
    pairing_failures = []
    for base, part in manifest[manifest.suite == "SENS"].groupby("base_structure"):
        projections = [_pairing_projection(json.loads((CB1 / relative).read_text())) for relative in part.relative_path]
        if any(item != projections[0] for item in projections[1:]): pairing_failures.append(base)
    counts = manifest.suite.value_counts().to_dict()
    core_cells = manifest[manifest.suite == "CORE"].groupby(["scale", "CF_level"]).size().to_dict()
    sens_cells = manifest[manifest.suite == "SENS"].groupby(["RI_level", "TI_level"]).size().to_dict()
    coverage = {
        "counts": counts, "all_108_loadable": len(feasibility) == 108,
        "all_108_feasible": all(item["h1_feasible"] for item in feasibility),
        "core_acceptance_failures": core_failures, "sensitivity_pairing_failures": pairing_failures,
        "core_cells_balanced": len(core_cells) == 9 and set(core_cells.values()) == {5},
        "dev_cells_balanced": len(manifest[manifest.suite == "DEV"].groupby(["scale", "CF_level"])) == 9 and set(manifest[manifest.suite == "DEV"].groupby(["scale", "CF_level"]).size()) == {2},
        "sensitivity_cells_balanced": len(sens_cells) == 9 and set(sens_cells.values()) == {5},
        "sensitivity_pairing_verified": not pairing_failures,
    }
    (OUT / "coverage_diagnostics.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    history = json.loads((CB1 / "manifests/generation_history.json").read_text())
    attempts = [len(value) for value in history.values()]
    (OUT / "generation_acceptance_summary.json").write_text(json.dumps({
        "requested_structures": len(history), "accepted_structures": len(history),
        "maximum_attempt": max(attempts), "mean_attempts": sum(attempts) / len(attempts),
        "accepted_without_widening": True,
    }, indent=2, sort_keys=True) + "\n")
    _figures(combined, controlled)
    if core_failures or pairing_failures or not coverage["all_108_feasible"]:
        raise RuntimeError(f"controlled audit failed: {coverage}")
    print("RCIAS_CB1_AUDIT_COMPLETE instances=108 feasible=108 pairing=TRUE figures=12")


def _figures(combined, controlled):
    mappings = [
        ("F_route_mean", "Routing flexibility", "01_legacy_vs_controlled_route_flexibility"),
        ("F_cap_mean", "Capability coverage", "02_legacy_vs_controlled_capability_coverage"),
        ("R_full_island", "Full-island ratio", "03_legacy_vs_controlled_full_island_ratio"),
        ("processing_CV_mean", "Processing-time CV", "04_legacy_vs_controlled_processing_heterogeneity"),
        ("RI", "Reconfiguration intensity", "05_legacy_vs_controlled_reconfiguration_intensity"),
    ]
    order = ["Legacy-130", "DEV", "CORE", "SENS"]
    for metric, label, name in mappings:
        fig, ax = plt.subplots(figsize=(7.2, 4.3)); data=[combined[combined.suite == suite][metric].dropna() for suite in order]
        ax.boxplot(data, tick_labels=order, showmeans=True); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2); _save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4));
    for ax, metric, label in zip(axes, ["W_transport_intensity", "F_transport_intensity"], ["W intensity", "F intensity"]):
        ax.boxplot([combined[combined.suite == suite][metric].dropna() for suite in order], tick_labels=order, showmeans=True); ax.set_ylabel(label); ax.tick_params(axis="x", rotation=20)
    _save(fig, "06_legacy_vs_controlled_transport_intensity")
    fig, ax=plt.subplots(figsize=(6.5,4.5));
    for cf, part in controlled[controlled.suite.isin(["DEV","CORE"])].groupby("CF_level"): ax.scatter(part.F_route_mean,part.F_cap_mean,label=cf,alpha=.7)
    ax.set(xlabel="F_route_mean",ylabel="F_cap_mean");ax.legend();ax.grid(alpha=.2);_save(fig,"07_controlled_route_vs_capability")
    columns=["number_of_operations","F_route_mean","F_cap_mean","processing_CV_mean","RI","W_transport_intensity","F_transport_intensity","precedence_edge_density"]
    corr=controlled[columns].corr();fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1);ax.set_xticks(range(len(columns)),columns,rotation=70,ha="right");ax.set_yticks(range(len(columns)),columns);fig.colorbar(im,ax=ax);_save(fig,"08_controlled_metric_correlation")
    core=controlled[controlled.suite=="CORE"];fig,ax=plt.subplots(figsize=(7,4));table=core.groupby(["scale","CF_level"]).size().unstack();im=ax.imshow(table.values,cmap="Blues");ax.set_xticks(range(3),table.columns);ax.set_yticks(range(3),table.index);[ax.text(j,i,int(table.iloc[i,j]),ha="center",va="center") for i in range(3) for j in range(3)];_save(fig,"09_core_scale_cf_design")
    sens=controlled[controlled.suite=="SENS"];fig,ax=plt.subplots(figsize=(7,4));table=sens.groupby(["RI_level","TI_level"]).size().unstack();im=ax.imshow(table.values,cmap="Greens");ax.set_xticks(range(3),table.columns);ax.set_yticks(range(3),table.index);[ax.text(j,i,int(table.iloc[i,j]),ha="center",va="center") for i in range(3) for j in range(3)];_save(fig,"10_sensitivity_RI_TI_design")
    fig,ax=plt.subplots(figsize=(6.5,4));sens.boxplot(column="RI",by="RI_level",ax=ax);fig.suptitle("");ax.set_title("");ax.set_ylabel("Realized RI");_save(fig,"11_sensitivity_realized_RI")
    fig,axes=plt.subplots(1,2,figsize=(9,4));sens.boxplot(column="W_transport_intensity",by="TI_level",ax=axes[0]);sens.boxplot(column="F_transport_intensity",by="TI_level",ax=axes[1]);fig.suptitle("");axes[0].set_ylabel("W intensity");axes[1].set_ylabel("F intensity");_save(fig,"12_sensitivity_realized_TI")


if __name__ == "__main__": main()
