#!/usr/bin/env python3
"""Audit and record the first authorized opening of sealed R06 labels."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_FREEZE = ROOT / "outputs/phase6f/audit/experiment_freeze.json"
LABEL_FREEZE = ROOT / "outputs/phase6f/revision_holdout/sealed_label_freeze.json"
STATE_FREEZE = ROOT / "outputs/phase6f/revision_holdout/state_freeze.json"
OUTPUT = ROOT / "outputs/phase6f/audit/revision_holdout_label_open.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash_without(payload: dict, key: str) -> str:
    value = {name: item for name, item in payload.items() if name != key}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    model = read_json(MODEL_FREEZE)
    labels = read_json(LABEL_FREEZE)
    states = read_json(STATE_FREEZE)
    opened_at = datetime.now(timezone.utc)
    frozen_at = datetime.fromisoformat(model["frozen_at_utc"])
    checks = {
        "model_configuration_frozen": model.get("status") == "FROZEN_BEFORE_R06_LABEL_OPEN",
        "model_freeze_hash_valid": canonical_hash_without(model, "freeze_hash")
        == model.get("freeze_hash"),
        "selected_checkpoint_hash_valid": sha256_file(
            Path(model["selected_checkpoint_path"])
        ) == model.get("selected_checkpoint_sha256"),
        "sealed_labels_complete": labels.get("status") == "SEALED_COMPLETE",
        "sealed_labels_not_previously_opened": not labels.get(
            "label_files_opened_for_analysis"
        ),
        "label_freeze_hash_valid": canonical_hash_without(labels, "freeze_hash")
        == labels.get("freeze_hash"),
        "state_freeze_bound": labels.get("state_freeze_hash") == states.get("freeze_hash"),
        "model_freeze_bound": model.get("sealed_label_freeze_hash")
        == labels.get("freeze_hash"),
        "revision_holdout_labels_opened_after_model_freeze": opened_at > frozen_at,
    }
    payload = {
        "schema": "phase6f-r06-label-open-v1",
        "status": "OPENED_FOR_FROZEN_EVALUATION" if all(checks.values()) else "FAIL",
        "model_frozen_at_utc": model["frozen_at_utc"],
        "labels_opened_at_utc": opened_at.isoformat(),
        "model_freeze_hash": model["freeze_hash"],
        "selected_checkpoint_sha256": model["selected_checkpoint_sha256"],
        "sealed_label_freeze_hash": labels["freeze_hash"],
        "state_freeze_hash": states["freeze_hash"],
        "revision_holdout_labels_opened_after_model_freeze": bool(
            checks["revision_holdout_labels_opened_after_model_freeze"]
        ),
        "authorized_scope": "OFFLINE_FROZEN_PHASE6F_EVALUATION_ONLY",
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] == "FAIL":
        raise SystemExit(1)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
