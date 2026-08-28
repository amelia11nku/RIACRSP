#!/usr/bin/env python3
"""Evaluate the deterministic H1 reference on frozen CB1 sensitivity cases."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching


def main() -> None:
    manifest_path = ROOT / "instances/controlled/RCIAS-CB1/manifests/sensitivity_manifest.csv"
    output = ROOT / "outputs/phase5c/comparisons/sensitivity_h1_results.csv"
    with manifest_path.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    records = []
    for row in manifest:
        instance = load_instance(ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"])
        started = time.perf_counter()
        result = solve_dispatching(instance, "H1")
        records.append({
            "instance_id": instance.instance_id, "base_structure": row["base_structure"],
            "RI_level": row["RI_level"], "TI_level": row["TI_level"], "method": "H1",
            "makespan": result.objective.makespan, "runtime_seconds": time.perf_counter() - started,
            # The constructive environment only returns after every operation has been decoded.
            "feasible": len(result.schedule.operation_schedules) == instance.num_operations,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f"CB1_SENSITIVITY_H1_COMPLETE rows={len(records)} feasible={all(row['feasible'] for row in records)}")


if __name__ == "__main__":
    main()
