#!/usr/bin/env python3
"""Freeze Phase 6M M2 code and feature schema before the first optimizer step."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rcias_clgri.ni.phase6m_selective_risk import (
    BANK_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    FAMILY,
    FORBIDDEN_ONLINE_COLUMNS,
    RANKER_FEATURE_COLUMNS,
    SELECTOR_NUMERIC_COLUMNS,
    STRUCTURAL_NUMERIC_COLUMNS,
    SUPPORT_FEATURE_COLUMNS,
    SUPPORT_NUMERIC_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/preregistration.json"
AMENDMENT = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/amendment_m2_schema.json"
IMPL = ROOT / "outputs/phase6m_selective_confidence_v1/implementation"


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
    prereg = json.loads(PREREG.read_text())
    require(prereg["status"] == "FROZEN_BEFORE_PHASE6M_OUTER_OOF_OR_OPTIMIZER_STEP",
            "Phase 6M M1 preregistration is invalid")
    amendment = json.loads(AMENDMENT.read_text())
    require(amendment["status"] == "FROZEN_SCHEMA_CORRECTION_BEFORE_FIRST_OPTIMIZER_STEP",
            "Phase 6M M1 schema amendment is invalid")
    forbidden = [
        ROOT / "outputs/phase6m_selective_confidence_v1/training/inner_ranker",
        ROOT / "outputs/phase6m_selective_confidence_v1/training/selector",
        ROOT / "outputs/phase6m_selective_confidence_v1/training/oof_predictions.parquet",
    ]
    require(not any(path.exists() for path in forbidden),
            "Phase 6M optimizer output exists before implementation freeze")
    schema = {
        "schema": "phase6m-selective-risk-feature-schema-v1",
        "model_family": FAMILY,
        "ranker_features": list(RANKER_FEATURE_COLUMNS),
        "bank_features": list(BANK_FEATURE_COLUMNS),
        "support_features": list(SUPPORT_FEATURE_COLUMNS),
        "support_numeric_columns": list(SUPPORT_NUMERIC_COLUMNS),
        "structural_numeric_columns": list(STRUCTURAL_NUMERIC_COLUMNS),
        "selector_numeric_columns": list(SELECTOR_NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "forbidden_online_columns": list(FORBIDDEN_ONLINE_COLUMNS),
        "selector_numeric_normalization": "training median/IQR floor 0.001, clip [-10,10]",
        "categorical_encoding": "training vocabulary one-hot with explicit UNK index zero",
        "candidate_order": "state_id then target_set_id stable lexical order",
        "historical_score_online_forward_calls": 0,
    }
    schema["schema_amendment_sha256"] = digest(AMENDMENT)
    schema_path = IMPL / "feature_schema_v2.json"
    immutable_json(schema_path, schema)
    code = [
        ROOT / "rcias_clgri/ni/phase6m_selective_risk.py",
        ROOT / "scripts/train_phase6m_selective_confidence.py",
        ROOT / "scripts/launch_phase6m_training.py",
        ROOT / "scripts/freeze_phase6m_schema_amendment.py",
        Path(__file__),
    ]
    inputs = [
        ROOT / "configs/phase6m_selective_confidence_v1.json",
        PREREG,
        AMENDMENT,
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/training_protocol.json",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/oof_predictions.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/selected_winners.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_seed_labels.parquet",
    ]
    protected_path = ROOT / prereg["protected_phase6l_manifest"]["path"]
    payload = {
        "schema": "phase6m-selective-risk-implementation-protocol-v2",
        "status": "FROZEN_BEFORE_FIRST_PHASE6M_OPTIMIZER_STEP",
        "model_family": FAMILY,
        "preregistration_sha256": digest(PREREG),
        "schema_amendment_sha256": digest(AMENDMENT),
        "feature_schema": {"path": str(schema_path.relative_to(ROOT)), "sha256": digest(schema_path)},
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in code},
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "protected_phase6l_manifest": {
            "path": str(protected_path.relative_to(ROOT)), "sha256": digest(protected_path),
        },
        "inner_ranker_runs": 18, "selector_runs": 9,
        "optimizer_steps_started": False, "outer_oof_generated": False,
        "historical_score_online_forward_calls": 0,
        "r13_accessed": False, "r14_accessed": False,
    }
    protocol_path = IMPL / "implementation_protocol_v2.json"
    immutable_json(protocol_path, payload)
    print(json.dumps({
        "status": payload["status"], "feature_schema_sha256": digest(schema_path),
        "implementation_protocol_sha256": digest(protocol_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
