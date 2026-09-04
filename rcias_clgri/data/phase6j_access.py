"""Leakage-safe access controls for the Phase 6J CAUR splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c_io import atomic_write_json


R12_SPLIT = "CAUR_FIT"
R13_SPLIT = "CAUR_SELECT"
R14_SPLIT = "CAUR_HOLDOUT"

_SPLIT_PATH_PARTS = {
    R12_SPLIT: "r12_caur_fit",
    R13_SPLIT: "r13_caur_select",
    R14_SPLIT: "r14_caur_holdout",
}
_SPLIT_STEM_TOKENS = {R12_SPLIT: "_R12", R13_SPLIT: "_R13", R14_SPLIT: "_R14"}
_LOCK_CONTRACTS = {
    R13_SPLIT: ("phase6j-caur-r13-freeze-v1", "FROZEN_BEFORE_R13"),
    R14_SPLIT: ("phase6j-caur-r14-freeze-v1", "FROZEN_BEFORE_R14"),
}


class Phase6JAccessError(RuntimeError):
    """Raised when a Phase 6J split is accessed outside its frozen boundary."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_forbidden_phase6i_r11_path(path: Path) -> bool:
    """Identify legacy R11 payloads without opening them."""
    normalized = path.as_posix().lower()
    return (
        "r11_live_rev_holdout" in path.parts
        or "outputs/phase6i_mr/r11_validation" in normalized
        or "_r11" in path.stem.lower()
    )


def phase6j_split_for_path(path: Path) -> str | None:
    """Return the CAUR split encoded in a path, without reading the payload."""
    path = Path(path)
    matches = [
        split
        for split, part in _SPLIT_PATH_PARTS.items()
        if part in path.parts or _SPLIT_STEM_TOKENS[split] in path.stem.upper()
    ]
    if len(matches) > 1:
        raise Phase6JAccessError(f"ambiguous Phase 6J split path: {path}")
    return matches[0] if matches else None


def verify_phase6j_unlock(
    split: str,
    freeze_record_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Verify the immutable artifact record required for R13 or R14."""
    if split not in _LOCK_CONTRACTS:
        raise Phase6JAccessError(f"no unlock contract exists for split {split}")
    freeze_record_path = Path(freeze_record_path)
    artifact_path = Path(artifact_path)
    if not freeze_record_path.is_file() or not artifact_path.is_file():
        raise Phase6JAccessError(f"{split} remains locked until its artifact is frozen")
    record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    expected_schema, expected_status = _LOCK_CONTRACTS[split]
    checks = (
        record.get("schema") == expected_schema,
        record.get("status") == expected_status,
        record.get("split") == split,
        record.get("content_accessed") is False,
        record.get("artifact_sha256") == sha256_file(artifact_path),
    )
    if not all(checks):
        raise Phase6JAccessError(f"invalid or stale {split} freeze record")
    return record


def begin_one_time_split_access(
    split: str,
    *,
    freeze_record_path: Path,
    artifact_path: Path,
    ledger_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Open or resume the single authorized R13/R14 access pass.

    A run can resume with the same ``run_id``. A different run, or a completed
    ledger, cannot reopen the split.
    """
    freeze = verify_phase6j_unlock(split, freeze_record_path, artifact_path)
    ledger_path = Path(ledger_path)
    expected = {
        "schema": "phase6j-caur-one-time-access-ledger-v1",
        "split": split,
        "run_id": str(run_id),
        "freeze_record_sha256": sha256_file(Path(freeze_record_path)),
        "artifact_sha256": sha256_file(Path(artifact_path)),
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if any(ledger.get(key) != value for key, value in expected.items()):
            raise Phase6JAccessError(f"{split} was already opened by another pass")
        if ledger.get("status") != "ACTIVE":
            raise Phase6JAccessError(f"{split} one-time access is already complete")
        return ledger

    ledger = {
        **expected,
        "status": "ACTIVE",
        "content_accessed": True,
        "freeze_status": freeze["status"],
    }
    atomic_write_json(ledger, ledger_path)
    return ledger


def complete_one_time_split_access(ledger_path: Path, *, run_id: str) -> dict[str, Any]:
    """Close an active split ledger; closed ledgers cannot be reopened."""
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        raise Phase6JAccessError("cannot complete a missing split-access ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "ACTIVE" or ledger.get("run_id") != str(run_id):
        raise Phase6JAccessError("split-access ledger is not active for this run")
    completed = {**ledger, "status": "COMPLETE"}
    atomic_write_json(completed, ledger_path)
    return completed


def load_phase6j_instance(
    path: Path,
    *,
    freeze_record_path: Path | None = None,
    artifact_path: Path | None = None,
    ledger_path: Path | None = None,
    run_id: str | None = None,
):
    """Load R12 freely and reject R13/R14 unless the one-time pass is active."""
    path = Path(path)
    if is_forbidden_phase6i_r11_path(path):
        raise Phase6JAccessError("Phase 6J code may never read Phase 6I-MR R11 payloads")
    split = phase6j_split_for_path(path)
    if split is None:
        raise Phase6JAccessError("path is not a registered Phase 6J split")
    if split in _LOCK_CONTRACTS:
        if None in (freeze_record_path, artifact_path, ledger_path, run_id):
            raise Phase6JAccessError(f"{split} content is locked")
        verify_phase6j_unlock(
            split, Path(freeze_record_path), Path(artifact_path)  # type: ignore[arg-type]
        )
        ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))  # type: ignore[arg-type]
        if (
            ledger.get("schema") != "phase6j-caur-one-time-access-ledger-v1"
            or ledger.get("split") != split
            or ledger.get("run_id") != str(run_id)
            or ledger.get("status") != "ACTIVE"
            or ledger.get("artifact_sha256") != sha256_file(Path(artifact_path))  # type: ignore[arg-type]
            or ledger.get("freeze_record_sha256")
            != sha256_file(Path(freeze_record_path))  # type: ignore[arg-type]
        ):
            raise Phase6JAccessError(f"{split} access ledger is invalid")
    return load_instance(path)
