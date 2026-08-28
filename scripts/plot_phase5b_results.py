#!/usr/bin/env python3
"""Generate the required Phase 5B validation figures as PNG and PDF."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5b/figures"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _gap(value, bc):
    return 100.0 * (float(value) - float(bc)) / float(bc)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    hybrid = json.loads((ROOT / "outputs/phase5b/hybrid_diagnosis/final_info.json").read_text())["records"]
    holdout = [row for row in hybrid if row["split"] == "phase5b_holdout"]
    labels = ("BC", "Phase 5A PPO", "BC-O + PPO-MWF")
    values = (
        0.0,
        mean(_gap(row["ppo_makespan"], row["bc_makespan"]) for row in holdout),
        mean(_gap(row["bc_o_ppo_mwf"], row["bc_makespan"]) for row in holdout),
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(labels, values, color=("#64748b", "#dc2626", "#2563eb"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean gap to BC (%)")
    _save(fig, "Figure_1_bc_phase5a_hybrid")

    methods = (
        "ppo_makespan", "bc_o_ppo_mwf", "ppo_o_bc_mwf", "bc_om_ppo_wf",
        "ppo_om_bc_wf", "bc_omw_ppo_f", "ppo_omw_bc_f",
    )
    method_labels = ("PPO", "BC-O\nPPO-MWF", "PPO-O\nBC-MWF", "BC-OM\nPPO-WF", "PPO-OM\nBC-WF", "BC-OMW\nPPO-F", "PPO-OMW\nBC-F")
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.bar(method_labels, [mean(_gap(row[m], row["bc_makespan"]) for row in holdout) for m in methods], color="#2563eb")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Holdout gap to BC (%)")
    _save(fig, "Figure_2_phase5a_oracle_decomposition")

    histories = [
        json.loads((ROOT / f"outputs/phase5b/downstream_seed_{i}/training_history.json").read_text())["updates"]
        for i in (1, 2, 3)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, history in enumerate(histories, start=1):
        ax.plot([r["update"] for r in history], [r["validation_normalized_makespan"] for r in history], label=f"seed {index}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Development normalized makespan")
    ax.legend(frameon=False)
    _save(fig, "Figure_3_frozen_o_training_curve")

    best = [min(r["validation_normalized_makespan"] for r in h) for h in histories]
    last = [h[-1]["validation_normalized_makespan"] for h in histories]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(x - 0.18, best, width=0.36, label="best", color="#059669")
    ax.bar(x + 0.18, last, width=0.36, label="last", color="#f59e0b")
    ax.set_xticks(x, ("seed 1", "seed 2", "seed 3"))
    ax.set_ylabel("Development normalized makespan")
    ax.legend(frameon=False)
    _save(fig, "Figure_4_best_vs_last")

    rollback_counts = [sum(int(r["rollback_applied"]) for r in h) for h in histories]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(("seed 1", "seed 2", "seed 3"), rollback_counts, color="#7c3aed")
    ax.set_ylabel("Rollback events")
    ax.set_ylim(0, 1)
    ax.text(1, 0.45, "No trajectory crossed the\n3 consecutive × 4% trigger", ha="center")
    _save(fig, "Figure_5_rollback_events")

    formal = json.loads((ROOT / "outputs/phase5b/structural_evaluation/final_info.json").read_text())["summary"]
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    levels = ("S", "M", "L")
    ax.bar(levels, [formal[f"phase5b_holdout|{level}"]["mean_gap_to_bc_percent"] for level in levels], color="#059669")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Three-seed mean gap to BC (%)")
    _save(fig, "Figure_6_sml_holdout_gaps")

    scenarios = ("fleet_scarcity", "high_reconfiguration", "high_travel")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(("fleet scarcity", "high reconfiguration", "high travel"), [formal[f"phase5b_structural|{s}"]["mean_gap_to_bc_percent"] for s in scenarios], color="#2563eb")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Three-seed mean gap to BC (%)")
    _save(fig, "Figure_7_structural_scenario_gaps")

    canonical = json.loads((ROOT / "outputs/phase5b/canonical_evaluation/final_info.json").read_text())["records"]
    families = ("Brandimarte", "Hurink E", "Hurink R", "Hurink V")
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.bar(families, [mean(float(r["gap_to_bc_percent"]) for r in canonical if r["family"] == family and r["method"] == "PHASE5B_DOWNSTREAM_PPO") for family in families], color="#dc2626")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Canonical three-seed gap to BC (%)")
    _save(fig, "Figure_8_canonical_family_gaps")
    print("PHASE5B_FIGURES_COMPLETE count=8 formats=png,pdf")


if __name__ == "__main__":
    main()
