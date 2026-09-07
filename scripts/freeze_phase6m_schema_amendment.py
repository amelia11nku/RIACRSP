#!/usr/bin/env python3
"""Freeze the pre-optimizer categorical correction for bottleneck_proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/phase6m_selective_confidence_v1.json"
BASE_PREREG = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/preregistration.json"
REPORT = ROOT / "docs/reports/phase6m_m1_schema_amendment.md"
IMPLEMENTATION = ROOT / "outputs/phase6m_selective_confidence_v1/implementation"
OUT = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/amendment_m2_schema.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def immutable_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text() == text, f"refusing to replace frozen evidence: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def main() -> None:
    config = json.loads(BASE_CONFIG.read_text())
    require("bottleneck_proxy" in config["selector_features"]["structural_numeric"],
            "base schema no longer contains the audited type error")
    require(not (ROOT / "outputs/phase6m_selective_confidence_v1/training/inner_ranker").exists(),
            "optimizer output exists before schema correction")
    invalid_protocol = IMPLEMENTATION / "implementation_protocol.json"
    invalid_schema = IMPLEMENTATION / "feature_schema.json"
    require(invalid_protocol.is_file() and invalid_schema.is_file(),
            "invalidated M2 artifacts must be preserved")
    failure = {
        "schema": "phase6m-m2-preflight-failure-v1",
        "status": "INVALIDATED_BEFORE_FIRST_OPTIMIZER_STEP",
        "command": "/home/liulei/miniconda3/envs/gnn311/bin/python scripts/train_phase6m_selective_confidence.py --device cpu --preflight",
        "error_type": "ValueError",
        "error": "could not convert string to float: W_LOGISTICS",
        "root_cause": "bottleneck_proxy was preregistered as structural_numeric but is categorical in the frozen input",
        "observed_categories": [
            "CROSS_RESOURCE_SYNCHRONIZATION", "F_LOGISTICS", "RECONFIGURATION", "W_LOGISTICS"
        ],
        "invalid_implementation_protocol_sha256": digest(invalid_protocol),
        "invalid_feature_schema_sha256": digest(invalid_schema),
        "optimizer_steps_started": False, "outer_oof_generated": False,
        "r13_accessed": False, "r14_accessed": False,
    }
    failure_path = IMPLEMENTATION / "preflight_failure_v1.json"
    immutable_json(failure_path, failure)
    payload = {
        "schema": "phase6m-preregistration-schema-amendment-v1",
        "status": "FROZEN_SCHEMA_CORRECTION_BEFORE_FIRST_OPTIMIZER_STEP",
        "reason": "correct observed categorical dtype without changing scientific model choices",
        "base_config_sha256": digest(BASE_CONFIG),
        "base_preregistration_sha256": digest(BASE_PREREG),
        "amendment_report_sha256": digest(REPORT),
        "freeze_code_sha256": digest(Path(__file__)),
        "preflight_failure": {
            "path": str(failure_path.relative_to(ROOT)), "sha256": digest(failure_path),
        },
        "active_feature_patch": {
            "remove_from_structural_numeric": ["bottleneck_proxy"],
            "add_to_categorical_one_hot": ["bottleneck_proxy"],
        },
        "all_other_protocol_fields_unchanged": True,
        "optimizer_steps_started": False, "outer_oof_generated": False,
        "r13_accessed": False, "r14_accessed": False,
    }
    immutable_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"], "amendment_sha256": digest(OUT),
        "preflight_failure_sha256": digest(failure_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
