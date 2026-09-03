"""Leakage guard for Phase 6I-MR fresh-holdout instance access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rcias_clgri.data.loader import load_instance


R11_SPLIT = "LIVE_REV_HOLDOUT"
R11_PATH_PART = "r11_live_rev_holdout"


class Phase6IHoldoutAccessError(RuntimeError):
    """Raised when R11 content is requested before the selected artifact freeze."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_phase6i_r11_path(path: Path) -> bool:
    """Identify R11 without opening the instance payload."""
    return R11_PATH_PART in path.parts or "_R11" in path.stem


def verify_phase6i_r11_unlock(
    freeze_record_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Validate the immutable selected-artifact record that unlocks R11."""
    if not freeze_record_path.is_file() or not artifact_path.is_file():
        raise Phase6IHoldoutAccessError(
            "Phase 6I-MR selected artifact and freeze record are required before R11 access"
        )
    record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != "phase6i-mr-artifact-freeze-v1"
        or record.get("status") != "FROZEN_BEFORE_R11"
        or record.get("r11_content_accessed") is not False
        or record.get("artifact_sha256") != sha256_file(artifact_path)
    ):
        raise Phase6IHoldoutAccessError("invalid Phase 6I-MR R11 freeze record")
    return record


def load_phase6i_instance(
    path: Path,
    *,
    freeze_record_path: Path | None = None,
    artifact_path: Path | None = None,
):
    """Load a Phase 6I instance, rejecting fresh holdout access until freeze."""
    path = Path(path)
    if is_phase6i_r11_path(path):
        if freeze_record_path is None or artifact_path is None:
            raise Phase6IHoldoutAccessError(
                "R11 content is locked until the Phase 6I-MR artifact is frozen"
            )
        verify_phase6i_r11_unlock(Path(freeze_record_path), Path(artifact_path))
    return load_instance(path)
