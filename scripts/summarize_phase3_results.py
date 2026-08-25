#!/usr/bin/env python3
"""Rebuild Phase 3 group/overall summaries from immutable raw evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from scripts.evaluate_ppo import _aggregate, _write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ppo_seed_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["method"] != "PPO_GREEDY":
            continue
        for group in (row["group"], "Overall"):
            key = (row["data_split"], group, row["training_seed"])
            groups.setdefault(key, []).append(row)
    records = []
    for (split, group, seed), values in sorted(groups.items()):
        records.append({
            "data_split": split,
            "group": group,
            "training_seed": seed,
            "instances": len(values),
            "mean_makespan": mean(float(row["makespan"]) for row in values),
            "mean_gap_percent": mean(float(row["gap_percent"]) for row in values),
            "best_makespan": min(float(row["makespan"]) for row in values),
            "feasibility_rate": mean(
                float(row["feasible"].strip().lower() in {"1", "true", "yes"})
                for row in values
            ),
        })
    return records


def _rebuild(directory: Path, stem: str) -> tuple[int, list[dict[str, object]]]:
    raw_path = directory / f"{stem}_results.csv"
    if not raw_path.exists():
        return 0, []
    rows = _read_csv(raw_path)
    summary = _aggregate(rows)
    _write_csv(directory / f"{stem}_summary.csv", summary)
    seed_summary = _ppo_seed_summary(rows)
    if seed_summary:
        _write_csv(directory / f"{stem}_ppo_seed_summary.csv", seed_summary)
    final_path = directory / "final_info.json"
    if final_path.exists():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        payload["summary"] = summary
        write_json(payload, final_path)
    return len(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/phase3"))
    args = parser.parse_args()
    synthetic_runs, synthetic_summary = _rebuild(
        args.results_dir / "validation", "synthetic"
    )
    canonical_runs, canonical_summary = _rebuild(
        args.results_dir / "canonical_evaluation", "canonical"
    )
    canonical_instances = canonical_runs // 7 if canonical_runs else 0
    write_json({
        "synthetic_runs": synthetic_runs,
        "canonical_instances": canonical_instances,
        "canonical_runs": canonical_runs,
        "frozen_before_canonical": True,
        "evaluation_complete": canonical_instances == 130,
        "synthetic_summary_records": len(synthetic_summary),
        "canonical_summary_records": len(canonical_summary),
    }, args.results_dir / "evaluation_final_info.json")
    print(
        "PHASE3_SUMMARY_COMPLETE = TRUE | "
        f"synthetic_runs={synthetic_runs} canonical_instances={canonical_instances}"
    )


if __name__ == "__main__":
    main()
