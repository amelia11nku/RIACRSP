#!/usr/bin/env python3
"""Validate and summarize the frozen RCIAS-CB1 Core search experiment."""

from __future__ import annotations

import json
from itertools import combinations
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5c"
ALGORITHMS = ("GA", "Adapted DCGA", "ALNS-H1")


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def _wilcoxon(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    differences = left - right
    nonzero = differences[differences != 0]
    ranks = _rank(nonzero.abs())
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    statistic = min(positive, negative)
    count = len(nonzero)
    mean = count * (count + 1) / 4
    tie_counts = nonzero.abs().value_counts()
    variance = count * (count + 1) * (2 * count + 1) / 24 - float(((tie_counts ** 3 - tie_counts).sum()) / 48)
    if variance == 0:
        return statistic, 1.0
    z = (abs(statistic - mean) - 0.5) / math.sqrt(variance)
    return statistic, math.erfc(z / math.sqrt(2))


def _friedman(pivot: pd.DataFrame) -> tuple[float, float]:
    ranks = pivot.rank(axis=1, method="average")
    instances, algorithms = ranks.shape
    statistic = 12 / (instances * algorithms * (algorithms + 1)) * float((ranks.sum() ** 2).sum()) - 3 * instances * (algorithms + 1)
    # Three algorithms imply two degrees of freedom, whose chi-square survival is exp(-x/2).
    return statistic, math.exp(-statistic / 2)


def _holm(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("p_value").reset_index(drop=True)
    adjusted, running, count = [], 0.0, len(frame)
    for index, p_value in enumerate(frame.p_value):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted.append(running)
    frame["p_holm"] = adjusted
    frame["reject_0_05"] = frame.p_holm < 0.05
    return frame


def main() -> None:
    comparisons = OUT / "comparisons"
    statistics = OUT / "statistics"
    audit_dir = OUT / "search/cb1_core"
    comparisons.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "instances/controlled/RCIAS-CB1/manifests/core_manifest.csv")
    freeze_hash = json.loads((OUT / "controlled_benchmark_audit/freeze_record.json").read_text())["freeze_hash"]
    records = []
    for path in sorted((audit_dir / "formal").rglob("seed_*.json")):
        raw = json.loads(path.read_text())
        records.append({
            "instance_id": raw["instance_id"], "algorithm": raw["algorithm"], "seed": raw["seed"],
            "makespan": raw["best_makespan"], "runtime": raw["runtime"],
            "time_limit": raw["time_limit_seconds"], "time_to_best": raw["best_found_time"],
            "decoder_evaluations": raw["decoder_evaluations"], "feasible": raw["feasible"],
            "suite": raw["suite"], "freeze_hash": raw["taxonomy_hash"],
            "result_path": str(path.relative_to(ROOT)),
        })
    runs = pd.DataFrame(records)
    duplicate_count = int(runs.duplicated(["instance_id", "algorithm", "seed"]).sum())
    counts = runs.groupby("algorithm").size().to_dict()
    seed_counts = runs.groupby(["instance_id", "algorithm"]).seed.nunique()
    checks = {
        "result_count_1350": len(runs) == 1350,
        "algorithm_counts_450_each": counts == {algorithm: 450 for algorithm in ALGORITHMS},
        "instance_count_45": runs.instance_id.nunique() == 45,
        "ten_unique_seeds_per_instance_algorithm": len(seed_counts) == 135 and bool((seed_counts == 10).all()),
        "no_duplicate_instance_algorithm_seed": duplicate_count == 0,
        "all_feasible": bool(runs.feasible.all()),
        "suite_cb1_core": set(runs.suite) == {"cb1_core"},
        "freeze_hash_matches": set(runs.freeze_hash) == {freeze_hash},
        "manifest_instances_match": set(runs.instance_id) == set(manifest.instance_id),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CB1-Core audit failed: {checks}")

    runs = runs.merge(manifest[["instance_id", "scale", "CF_level", "number_of_operations"]], on="instance_id", how="left")
    runs["runtime_ratio"] = runs.runtime / runs.time_limit
    runs.to_csv(comparisons / "cb1_search_run_results.csv", index=False)

    constructive = pd.read_csv(comparisons / "cb1_constructive_results.csv")
    h1 = constructive[constructive.method == "H1"].set_index("instance_id").makespan
    ppo = constructive[constructive.method == "PPO"].groupby("instance_id").makespan.mean()
    summary = runs.groupby(["instance_id", "scale", "CF_level", "algorithm"], as_index=False).agg(
        best=("makespan", "min"), mean=("makespan", "mean"), median=("makespan", "median"),
        standard_deviation=("makespan", "std"), worst=("makespan", "max"), runtime=("runtime", "mean"),
        time_to_best=("time_to_best", "mean"), decoder_evaluations=("decoder_evaluations", "mean"),
        feasibility_rate=("feasible", "mean"), time_limit=("time_limit", "first"),
    )
    summary["h1_makespan"] = summary.instance_id.map(h1)
    summary["ppo_mean_makespan"] = summary.instance_id.map(ppo)
    for statistic in ("mean", "best"):
        summary[f"{statistic}_improvement_over_h1_percent"] = 100 * (summary.h1_makespan - summary[statistic]) / summary.h1_makespan
        summary[f"{statistic}_improvement_over_ppo_percent"] = 100 * (summary.ppo_mean_makespan - summary[statistic]) / summary.ppo_mean_makespan
    summary.to_csv(comparisons / "cb1_search_instance_summary.csv", index=False)

    def grouped(columns: list[str]) -> pd.DataFrame:
        return summary.groupby(columns + ["algorithm"], as_index=False).agg(
            instances=("instance_id", "size"), mean_makespan=("mean", "mean"),
            mean_improvement_over_h1_percent=("mean_improvement_over_h1_percent", "mean"),
            best_improvement_over_h1_percent=("best_improvement_over_h1_percent", "mean"),
            mean_improvement_over_ppo_percent=("mean_improvement_over_ppo_percent", "mean"),
            best_improvement_over_ppo_percent=("best_improvement_over_ppo_percent", "mean"),
            feasibility_rate=("feasibility_rate", "mean"), mean_runtime=("runtime", "mean"))

    grouped(["scale"]).to_csv(comparisons / "cb1_scale_summary.csv", index=False)
    grouped(["CF_level"]).to_csv(comparisons / "cb1_cf_summary.csv", index=False)
    grouped(["scale", "CF_level"]).to_csv(comparisons / "cb1_scale_cf_summary.csv", index=False)
    summary[["instance_id", "scale", "CF_level", "algorithm", "mean_improvement_over_h1_percent", "best_improvement_over_h1_percent"]].to_csv(comparisons / "h1_improvement_summary.csv", index=False)
    summary[["instance_id", "scale", "CF_level", "algorithm", "mean_improvement_over_ppo_percent", "best_improvement_over_ppo_percent"]].to_csv(comparisons / "ppo_improvement_summary.csv", index=False)

    pivot = summary.pivot(index="instance_id", columns="algorithm", values="mean")
    tests, wtl = [], []
    for left, right in combinations(ALGORITHMS, 2):
        difference = pivot[left] - pivot[right]
        statistic, p_value = _wilcoxon(pivot[left], pivot[right])
        row = {"left": left, "right": right, "instances": len(difference), "statistic": statistic,
               "p_value": p_value, "median_difference": difference.median(),
               "wins_left": int((difference < 0).sum()), "ties": int((difference == 0).sum()),
               "losses_left": int((difference > 0).sum())}
        tests.append(row); wtl.append({key: row[key] for key in ("left", "right", "instances", "wins_left", "ties", "losses_left")})
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(statistics / "cb1_pairwise_wilcoxon.csv", index=False)
    _holm(tests_frame).to_csv(statistics / "cb1_holm_corrected.csv", index=False)
    pd.DataFrame(wtl).to_csv(statistics / "cb1_win_tie_loss.csv", index=False)
    friedman_statistic, friedman_p_value = _friedman(pivot[list(ALGORITHMS)])
    (statistics / "cb1_friedman.json").write_text(json.dumps({"algorithms": ALGORITHMS, "instances": 45,
        "statistic": friedman_statistic, "p_value": friedman_p_value}, indent=2) + "\n")

    runtime = runs.groupby("algorithm").runtime_ratio.agg(["mean", "median", lambda x: x.quantile(.95), "max"]).reset_index()
    runtime.columns = ["algorithm", "mean", "median", "p95", "max"]
    overall = grouped([]).sort_values("mean_makespan")
    audit = {"schema": "phase5c-cb1-core-audit-v1", "status": "PASS", "checks": checks,
             "algorithm_counts": counts, "duplicate_count": duplicate_count,
             "runtime_ratio_by_algorithm": runtime.to_dict(orient="records"),
             "best_conventional_baseline_by_mean_makespan": overall.iloc[0].algorithm}
    (audit_dir / "formal_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    overall.to_csv(comparisons / "cb1_overall_summary.csv", index=False)
    print(f"PHASE5C_CB1_SUMMARY_COMPLETE runs={len(runs)} best={overall.iloc[0].algorithm}")


if __name__ == "__main__":
    main()
