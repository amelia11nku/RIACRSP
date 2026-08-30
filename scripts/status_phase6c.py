#!/usr/bin/env python3
"""Report actual persistent Phase 6C production progress from verified artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c"


def statuses(pattern: str):
    rows = []
    for path in sorted(OUT.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
            if payload.get("status") == "COMPLETE":
                rows.append(payload)
        except json.JSONDecodeError:
            pass
    return rows


def main():
    job_path = OUT / "environment/production_job_status.json"
    job = json.loads(job_path.read_text()) if job_path.exists() else {"status": "NOT_LAUNCHED"}
    reservoir = statuses("reservoir/raw/*/status.json")
    dataset = statuses("dataset/*/*/status.json")
    payload = {
        "job": job,
        "reservoir_completed_shards": len(reservoir),
        "reservoir_total_shards": 810,
        "reservoir_available_states": sum(row.get("state_count", 0) for row in reservoir),
        "selected_state_manifest_exists": (OUT / "manifests/state_manifest.csv").exists(),
        "dataset_completed_shards": len(dataset),
        "dataset_total_shards": 405,
        "dataset_completed_states": sum(row.get("state_count", 0) for row in dataset),
        "dataset_completed_arms": sum(row.get("arm_count", 0) for row in dataset),
        "dataset_completed_repair_rows": sum(row.get("repair_seed_row_count", 0) for row in dataset),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
