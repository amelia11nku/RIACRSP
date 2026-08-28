#!/usr/bin/env python3
"""Build machine-readable Phase 5C comparisons and paired statistics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5c"


def _holm(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["p_value"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def main():
    comparisons = OUT / "comparisons"
    statistics = OUT / "statistics"
    comparisons.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)
    taxonomy = pd.read_csv(OUT / "benchmark_audit/taxonomy_proposal.csv").rename(columns={"classification": "taxonomy"})
    metrics = pd.read_csv(OUT / "benchmark_audit/instance_metrics.csv")
    annotations = taxonomy[["instance_id", "taxonomy"]].merge(metrics, on="instance_id", how="left")

    constructive = pd.read_csv(ROOT / "outputs/phase5b/canonical_evaluation/canonical_results.csv")
    constructive["method"] = constructive["method"].replace({"PHASE5B_DOWNSTREAM_PPO": "PPO"})
    means = constructive[constructive.method == "PPO"].groupby("instance_id", as_index=False).agg(
        makespan=("makespan", "mean"), runtime_seconds=("runtime_seconds", "mean"),
        inference_seconds=("inference_seconds", "mean"), feasible=("feasible", "all"), family=("family", "first")
    )
    means["method"] = "PPO_MEAN"
    means["training_seed"] = "THREE_SEED_MEAN"
    constructive = pd.concat([constructive, means], ignore_index=True, sort=False)
    constructive = constructive.merge(annotations[["instance_id", "taxonomy", "number_of_operations"]], on="instance_id", how="left")
    h1 = constructive[constructive.method == "H1"].set_index("instance_id").makespan
    constructive["gap_to_h1_percent"] = constructive.apply(lambda row: 100 * (row.makespan - h1[row.instance_id]) / h1[row.instance_id], axis=1)
    constructive.to_csv(comparisons / "constructive_results.csv", index=False)

    records = []
    for path in sorted((OUT / "search/formal").rglob("seed_*.json")):
        raw = json.loads(path.read_text())
        records.append({
            "instance_id": raw["instance_id"], "algorithm": raw["algorithm"], "seed": raw["seed"],
            "time_limit": raw["time_limit_seconds"], "runtime": raw["runtime"],
            "time_to_best": raw["best_found_time"], "decoder_evaluations": raw["decoder_evaluations"],
            "makespan": raw["best_makespan"], "feasible": raw["feasible"], "result_path": str(path.relative_to(ROOT)),
        })
    runs = pd.DataFrame(records)
    expected = 130 * 10 * 3
    if len(runs) != expected:
        raise RuntimeError(f"formal results incomplete: {len(runs)}/{expected}")
    runs = runs.merge(annotations, on="instance_id", how="left")
    runs.to_csv(comparisons / "search_run_results.csv", index=False)
    summary = runs.groupby(["instance_id", "family", "taxonomy", "algorithm"], as_index=False).agg(
        best=("makespan", "min"), mean=("makespan", "mean"), median=("makespan", "median"),
        standard_deviation=("makespan", "std"), worst=("makespan", "max"), runtime=("runtime", "mean"),
        time_to_best=("time_to_best", "mean"), decoder_evaluations=("decoder_evaluations", "mean"),
        feasibility_rate=("feasible", "mean"), time_limit=("time_limit", "first"),
    )
    summary["h1_makespan"] = summary.instance_id.map(h1)
    ppo = constructive[constructive.method == "PPO_MEAN"].set_index("instance_id").makespan
    summary["ppo_mean_makespan"] = summary.instance_id.map(ppo)
    summary["improvement_over_h1_percent"] = 100 * (summary.h1_makespan - summary["mean"]) / summary.h1_makespan
    summary["improvement_over_ppo_percent"] = 100 * (summary.ppo_mean_makespan - summary["mean"]) / summary.ppo_mean_makespan
    summary.to_csv(comparisons / "search_instance_summary.csv", index=False)

    family = summary.groupby(["family", "algorithm"], as_index=False).agg(
        instances=("instance_id", "size"), mean_makespan=("mean", "mean"),
        mean_improvement_over_h1_percent=("improvement_over_h1_percent", "mean"),
        mean_improvement_over_ppo_percent=("improvement_over_ppo_percent", "mean"), feasibility_rate=("feasibility_rate", "mean"))
    family.to_csv(comparisons / "family_summary.csv", index=False)
    groups = []
    for label, selected in [("Legacy-130", summary), *[(name, summary[summary.taxonomy == name]) for name in sorted(summary.taxonomy.unique())]]:
        for algorithm, part in selected.groupby("algorithm"):
            groups.append({"taxonomy": label, "algorithm": algorithm, "instances": len(part),
                "mean_makespan": part["mean"].mean(), "mean_improvement_over_h1_percent": part.improvement_over_h1_percent.mean(),
                "mean_improvement_over_ppo_percent": part.improvement_over_ppo_percent.mean(), "feasibility_rate": part.feasibility_rate.mean()})
    pd.DataFrame(groups).to_csv(comparisons / "taxonomy_summary.csv", index=False)
    summary[["instance_id", "family", "taxonomy", "algorithm", "improvement_over_h1_percent"]].to_csv(comparisons / "h1_improvement_summary.csv", index=False)
    summary[["instance_id", "family", "taxonomy", "algorithm", "improvement_over_ppo_percent"]].to_csv(comparisons / "ppo_improvement_summary.csv", index=False)

    methods = sorted(summary.algorithm.unique())
    tests = []
    pivot = summary.pivot(index="instance_id", columns="algorithm", values="mean")
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            delta = pivot[left] - pivot[right]
            statistic, p_value = wilcoxon(pivot[left], pivot[right], zero_method="pratt")
            tests.append({"left": left, "right": right, "instances": len(delta), "statistic": statistic,
                "p_value": p_value, "median_difference": delta.median(), "wins_left": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()), "losses_left": int((delta > 0).sum())})
    _holm(tests)
    pd.DataFrame(tests).to_csv(statistics / "pairwise_tests.csv", index=False)
    print(f"PHASE5C_SUMMARY_COMPLETE runs={len(runs)} feasible={runs.feasible.mean():.6f}")


if __name__ == "__main__":
    main()
