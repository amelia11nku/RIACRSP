"""Atomic deterministic file helpers for Phase 6C shards and manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partial_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial.{os.getpid()}")


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def remove_partial_files(directory: Path) -> int:
    removed = 0
    if not directory.exists():
        return removed
    for path in directory.rglob(".*.partial.*"):
        path.unlink()
        removed += 1
    return removed
