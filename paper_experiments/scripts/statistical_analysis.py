#!/usr/bin/env python3
"""Run the preregistered paired Core tests after complete BKS/RPD assembly."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "paper_experiments/processed_data/main"
MANIFEST_PATH = DATA_ROOT / "analysis_manifest.json"
SUMMARY_PATH = DATA_ROOT / "main_instance_summary.csv"
CSG = "CSG-NI Phase6H provisional"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def paired_rank_biserial(differences: list[float]) -> float:
    nonzero = [value for value in differences if not math.isclose(value, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 0.0
    ranks = stats.rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    return float((positive - negative) / (positive + negative))


def main() -> int:
    if not MANIFEST_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise RuntimeError("run compute_core_bks_rpd.py after the Core matrix completes")
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_COMPLETE_DRAFT_PROVISIONAL_PHASE6H":
        raise RuntimeError("Core BKS/RPD manifest is not complete")
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = sorted({row["method"] for row in rows})
    instances = sorted({row["instance_id"] for row in rows})
    if len(methods) != 5 or len(instances) != 45 or len(rows) != 225 or CSG not in methods:
        raise RuntimeError("expected a complete 5 method x 45 instance summary")
    values = {
        (row["method"], row["instance_id"]): float(row["median_rpd_percent"])
        for row in rows
    }
    if len(values) != len(rows):
        raise RuntimeError("duplicate method/instance rows in Core summary")
    friedman = stats.friedmanchisquare(*[
        [values[(method, instance)] for instance in instances] for method in methods
    ])
    global_significant = bool(friedman.pvalue < 0.05)

    raw_p: dict[str, float] = {}
    comparisons: dict[str, dict[str, object]] = {}
    for competitor in methods:
        if competitor == CSG:
            continue
        csg_values = [values[(CSG, instance)] for instance in instances]
        competitor_values = [values[(competitor, instance)] for instance in instances]
        differences = [other - csg for csg, other in zip(csg_values, competitor_values)]
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        ties = len(differences) - wins - losses
        try:
            test = stats.wilcoxon(
                csg_values, competitor_values, alternative="two-sided",
                zero_method="wilcox", method="auto",
            )
            statistic = float(test.statistic)
            pvalue = float(test.pvalue)
        except ValueError:
            statistic = 0.0
            pvalue = 1.0
        raw_p[competitor] = pvalue
        comparisons[competitor] = {
            "comparison": f"{CSG} vs {competitor}",
            "reference_method": CSG,
            "competitor": competitor,
            "instance_count": len(instances),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_raw": pvalue,
            "paired_rank_biserial_positive_favors_csg": paired_rank_biserial(differences),
        }
    adjusted = holm_adjust(raw_p)
    pairwise_rows = []
    for competitor in sorted(comparisons):
        row = comparisons[competitor]
        row["holm_adjusted_p"] = adjusted[competitor]
        row["reject_at_0_05"] = global_significant and adjusted[competitor] < 0.05
        row["posthoc_authorized_by_friedman"] = global_significant
        pairwise_rows.append(row)

    atomic_csv(DATA_ROOT / "statistical_tests.csv", pairwise_rows)
    report = {
        "schema": "initial-manuscript-core-statistics-v1",
        "status": "PASS",
        "source_summary_sha256": sha256(SUMMARY_PATH),
        "analysis_unit": "per-instance median RPD across five matched seeds",
        "friedman": {
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "alpha": 0.05,
            "significant": global_significant,
        },
        "posthoc": "two-sided paired Wilcoxon with Holm correction",
        "effect_size": "paired rank-biserial; positive favors Phase6H CSG-NI",
        "pairwise_output": "statistical_tests.csv",
    }
    atomic_json(DATA_ROOT / "statistical_analysis.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
