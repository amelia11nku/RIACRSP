#!/usr/bin/env python3
"""Verify Phase 6A–6F and freeze the Phase 6G development protocol."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6g_live_solver.json"
OUT = ROOT / "outputs/phase6g"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config = load(CONFIG)
    freeze_path = OUT / "environment/phase6g_environment_freeze.json"
    split_path = OUT / "environment/dev_split.csv"
    if freeze_path.exists() and split_path.exists():
        frozen = load(freeze_path)
        if frozen["phase6g_config_sha256"] != digest(CONFIG):
            raise RuntimeError("Phase 6G config changed after protocol freeze")
        print("PHASE6G_ENVIRONMENT_FREEZE_VERIFIED", frozen["freeze_hash"])
        return

    gate_paths = {
        "phase6a": ROOT / "outputs/phase6a/diagnostics/completion_gate.json",
        "phase6b": ROOT / "outputs/phase6b/audit/completion_gate.json",
        "phase6c": ROOT / "outputs/phase6c/audit/completion_gate.json",
        "phase6d": ROOT / "outputs/phase6d/audit/completion_gate.json",
        "phase6e": ROOT / "outputs/phase6e/audit/completion_gate.json",
        "phase6f": ROOT / "outputs/phase6f/audit/completion_gate.json",
    }
    gates = {name: load(path) for name, path in gate_paths.items()}
    experiment_freeze_path = ROOT / config["frozen_phase6f"]["experiment_freeze"]
    experiment_freeze = load(experiment_freeze_path)
    checkpoint = Path(experiment_freeze["selected_checkpoint_path"])
    checks = {
        **{
            f"{phase}_complete": bool(gates[phase][f"{phase.upper()}_COMPLETE"])
            for phase in gates
        },
        "phase6g_recommended": bool(gates["phase6f"]["PHASE6G_LIVE_INTEGRATION_RECOMMENDED"]),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_hash_exact": checkpoint.exists() and digest(checkpoint) == config["frozen_phase6f"]["checkpoint_sha256"],
        "experiment_freeze_hash_agrees": experiment_freeze["selected_checkpoint_sha256"] == config["frozen_phase6f"]["checkpoint_sha256"],
        "destroy_fraction_frozen": config["search"]["destroy_fraction"] == 0.15,
        "candidate_trials_frozen": config["search"]["candidate_trials"] == 8,
        "rates_exact": config["search"]["intervention_rates"] == [20, 50, 100],
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    manifest_path = ROOT / config["development_split"]["manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    split_rows = []
    for row in manifest:
        replicate = row["replicate"]
        if replicate == config["development_split"]["DEV_TUNE_replicate"]:
            split = "DEV_TUNE"
        elif replicate == config["development_split"]["DEV_HOLDOUT_replicate"]:
            split = "DEV_HOLDOUT"
        else:
            raise RuntimeError(f"unexpected DEV replicate: {replicate}")
        split_rows.append({
            "instance_id": row["instance_id"],
            "split": split,
            "scale": row["scale"],
            "CF_level": row["CF_level"],
            "replicate": replicate,
            "number_of_operations": row["number_of_operations"],
            "relative_path": row["relative_path"],
        })
    for split in ("DEV_TUNE", "DEV_HOLDOUT"):
        selected = [row for row in split_rows if row["split"] == split]
        cells = {(row["scale"], row["CF_level"]) for row in selected}
        if len(selected) != 9 or len(cells) != 9:
            raise RuntimeError(f"{split} must contain one instance from each of 9 cells")

    source_paths = [
        ROOT / "rcias_clgri/search/alns.py",
        ROOT / "rcias_clgri/search/ga.py",
        ROOT / "rcias_clgri/heuristic/dispatching.py",
        ROOT / "rcias_clgri/search/common.py",
        ROOT / "configs/phase5c_alns.json",
        ROOT / "configs/phase5c_ga.json",
        ROOT / "configs/csg_v1_schema.json",
    ]
    payload = {
        "schema": "phase6g-environment-freeze-v1",
        "status": "FROZEN_BEFORE_DEV_TUNE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "phase6g_config_sha256": digest(CONFIG),
        "phase6f_experiment_freeze_sha256": digest(experiment_freeze_path),
        "checkpoint_sha256": digest(checkpoint),
        "phase_completion_gate_sha256": {
            phase: digest(path) for phase, path in gate_paths.items()
        },
        "frozen_source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_paths
        },
        "dev_manifest_sha256": digest(manifest_path),
        "checks": checks,
    }
    payload["freeze_hash"] = canonical_hash(payload)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with split_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    print("PHASE6G_ENVIRONMENT_FROZEN", payload["freeze_hash"])


if __name__ == "__main__":
    main()
