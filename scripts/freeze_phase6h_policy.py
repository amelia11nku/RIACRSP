#!/usr/bin/env python3
"""Freeze the CAL-FIT-selected policy before any CAL-HOLDOUT solver run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
CALIBRATION = ROOT / "outputs/phase6h_calibration/calibration"
GATE = ROOT / "outputs/phase6h_calibration/gate_study"
OUT = ROOT / "outputs/phase6h_calibration/frozen"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    config = load_json(CONFIG_PATH)
    fit_integrity = load_json(CALIBRATION / "fit_integrity.json")
    gate_integrity = load_json(GATE / "gate_study_integrity.json")
    if fit_integrity.get("status") != "PASS" or gate_integrity.get("status") != "PASS":
        raise RuntimeError("calibration fit and gate study must both pass")
    if fit_integrity.get("cal_holdout_opened") is not False or gate_integrity.get("cal_holdout_opened") is not False:
        raise RuntimeError("CAL-HOLDOUT was opened before the policy freeze")
    policies = pd.read_csv(GATE / "gate_study_policy_summary.csv")
    selected = policies[policies.selected]
    if len(selected) != 1 or str(selected.iloc[0].policy_name) != gate_integrity["selected_policy"]:
        raise RuntimeError("gate selection is missing or ambiguous")
    manifest = pd.read_csv(CALIBRATION / "candidate_policy_manifest.csv")
    candidate_row = manifest[manifest.policy_name == gate_integrity["selected_policy"]]
    if len(candidate_row) != 1:
        raise RuntimeError("selected candidate is absent from the calibration manifest")
    candidate_path = ROOT / candidate_row.iloc[0].relative_path
    if digest(candidate_path) != candidate_row.iloc[0].sha256:
        raise RuntimeError("selected candidate policy hash mismatch")
    candidate = load_json(candidate_path)
    frozen = {
        **candidate,
        "schema": "phase6h-frozen-policy-v1",
        "status": "FROZEN_BEFORE_CAL_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase6h_config_sha256": digest(CONFIG_PATH),
        "calibration_fit_artifact_sha256": fit_integrity["artifact_sha256"],
        "gate_study_integrity_sha256": digest(GATE / "gate_study_integrity.json"),
        "gate_study_policy_summary_sha256": digest(GATE / "gate_study_policy_summary.csv"),
        "selected_gate_metrics": json.loads(selected.iloc[0].to_json()),
        "selection_data": "CAL_FIT_ONLY",
        "cal_holdout_opened": False,
    }
    policy_path = OUT / "phase6h_policy.json"
    atomic_json(frozen, policy_path)
    atomic_json({
        "schema": "phase6h-policy-freeze-record-v1",
        "status": "FROZEN_BEFORE_CAL_HOLDOUT",
        "policy_name": frozen["policy_name"],
        "policy_relative_path": str(policy_path.relative_to(ROOT)),
        "policy_sha256": digest(policy_path),
        "checkpoint_sha256": config["frozen_phase6f"]["checkpoint_sha256"],
        "cal_holdout_opened": False,
    }, OUT / "freeze_record.json")
    print(
        f"PHASE6H_POLICY_FROZEN policy={frozen['policy_name']} "
        f"sha256={digest(policy_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
