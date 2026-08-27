#!/usr/bin/env python3
"""Generate the required Phase 5A diagnostic figures as PNG and PDF."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a/figures"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _histories():
    return [
        json.loads((ROOT / f"outputs/phase5a/seed_{index}/training_history.json").read_text())["updates"]
        for index in (1, 2, 3)
    ]


def _line_figure(histories, keys, labels, name, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
    history = histories[0]
    for key, label, color in zip(keys, labels, colors):
        ax.plot([row["update"] for row in history], [row[key] for row in history], label=label, color=color)
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    _save(fig, name)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    histories = _histories()
    evaluation = json.loads((ROOT / "outputs/phase5a/structural_evaluation/mean_final_info.json").read_text())
    rows = evaluation["records"]

    methods = ("BC_GREEDY", "PHASE4_PPO", "PHASE5A_PPO")
    labels = ("BC", "Phase 4 PPO", "Phase 5A PPO")
    gaps = [np.mean([float(r["gap_to_bc_percent"]) for r in rows if r["data_split"] == "development" and r["method"] == method]) for method in methods]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, gaps, color=("#64748b", "#dc2626", "#059669"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean gap to BC (%)")
    _save(fig, "Figure_1_bc_phase4_phase5a_gap")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, history in enumerate(histories, start=1):
        ax.plot([r["update"] for r in history], [r["validation_normalized_makespan"] for r in history], label=f"seed {index}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Development normalized makespan")
    ax.legend(frameon=False)
    _save(fig, "Figure_2_development_score_vs_update")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, history in enumerate(histories, start=1):
        ax.plot([r["update"] for r in history], [r["teacher_kl"] for r in history], label=f"seed {index}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("KL(current || BC)")
    ax.legend(frameon=False)
    _save(fig, "Figure_3_teacher_kl_vs_update")

    _line_figure(
        histories,
        ("teacher_kl_operation", "teacher_kl_island", "teacher_kl_w", "teacher_kl_f"),
        ("operation", "island", "W-AGV", "F-AGV"),
        "Figure_4_per_stage_kl",
        "KL(current || BC)",
    )
    _line_figure(
        histories,
        ("normalized_operation_entropy", "normalized_island_entropy", "normalized_w_entropy", "normalized_f_entropy"),
        ("operation", "island", "W-AGV", "F-AGV"),
        "Figure_5_normalized_stage_entropy",
        "Normalized entropy",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, history in enumerate(histories, start=1):
        ax.plot([r["update"] for r in history], [r["encoder_parameter_drift_from_bc"] for r in history], label=f"seed {index}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Encoder parameter drift from BC")
    ax.legend(frameon=False)
    _save(fig, "Figure_6_encoder_drift")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.24
    for index, history in enumerate(histories):
        updates = np.array([r["update"] for r in history])
        values = np.array([r["validation_normalized_makespan"] for r in history])
        ax.scatter(updates + (index - 1) * width, values, s=12, alpha=0.65, label=f"seed {index + 1}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Seed-wise development score")
    ax.legend(frameon=False)
    _save(fig, "Figure_7_seedwise_validation_trajectories")

    stagewise = json.loads((ROOT / "outputs/phase5a/stagewise_diagnosis/final_info.json").read_text())["group_means"]["development"]
    stage_keys = ("bc_makespan", "ppo_makespan", "bc_o_ppo_rest", "ppo_o_bc_rest", "bc_om_ppo_wf", "ppo_om_bc_wf")
    stage_labels = ("BC", "P4 PPO", "BC-O\nPPO-rest", "PPO-O\nBC-rest", "BC-OM\nPPO-WF", "PPO-OM\nBC-WF")
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(stage_labels, [stagewise[key] for key in stage_keys], color="#2563eb")
    ax.set_ylabel("Mean makespan")
    _save(fig, "Figure_8_stagewise_oracle_decomposition")

    scenarios = ("fleet_scarcity", "high_reconfiguration", "high_travel")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(scenarios))
    for offset, method, label, color in zip((-0.25, 0, 0.25), methods, labels, ("#64748b", "#dc2626", "#059669")):
        values = [np.mean([float(r["gap_to_bc_percent"]) for r in rows if r["group"] == scenario and r["method"] == method]) for scenario in scenarios]
        ax.bar(x + offset, values, width=0.24, label=label, color=color)
    ax.set_xticks(x, ("fleet scarcity", "high reconfiguration", "high travel"))
    ax.set_ylabel("Mean gap to BC (%)")
    ax.legend(frameon=False)
    _save(fig, "Figure_9_structural_scenario_performance")
    print("PHASE5A_FIGURES_COMPLETE count=9 formats=png,pdf")


if __name__ == "__main__":
    main()
