"""Versioned sharded cache for pre-tensorized Phase 6E state samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .dataset import NIStateSample


CACHE_SCHEMA = "phase6e-pre-tensorized-shard-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shard_cache(
    path: Path,
    samples: Sequence[NIStateSample],
    *,
    instance_id: str,
    training_split: str,
    source_shard_sha256: str,
) -> dict[str, object]:
    if not samples:
        raise ValueError("refusing to write an empty tensorized shard")
    schema_hashes = {sample.graph.tensor_schema_hash for sample in samples}
    if len(schema_hashes) != 1:
        raise ValueError("tensorized shard mixes incompatible schemas")
    if any(sample.graph.instance_id != instance_id for sample in samples):
        raise ValueError("tensorized shard mixes instance IDs")
    state_ids = [sample.graph.state_id for sample in samples]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("tensorized shard contains duplicate state IDs")
    action_count = sum(sample.actions.action_count for sample in samples)
    payload = {
        "schema": CACHE_SCHEMA,
        "instance_id": instance_id,
        "training_split": training_split,
        "source_shard_sha256": source_shard_sha256,
        "tensor_schema_hash": next(iter(schema_hashes)),
        "state_count": len(samples),
        "action_count": action_count,
        "positive_count": sum(int(sample.actions.positive.sum().item()) for sample in samples),
        "samples": list(samples),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "schema": CACHE_SCHEMA,
        "instance_id": instance_id,
        "training_split": training_split,
        "source_shard_sha256": source_shard_sha256,
        "tensor_schema_hash": next(iter(schema_hashes)),
        "state_count": len(samples),
        "action_count": action_count,
        "positive_count": sum(int(sample.actions.positive.sum().item()) for sample in samples),
        "membership_count": sum(sample.actions.membership_count for sample in samples),
        "cache_path": str(path),
        "cache_bytes": path.stat().st_size,
        "cache_sha256": file_sha256(path),
        "status": "COMPLETE",
    }


def load_shard_cache(
    path: Path,
    *,
    expected_tensor_schema_hash: str | None = None,
    expected_source_shard_sha256: str | None = None,
) -> tuple[list[NIStateSample], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported Phase 6E cache schema in {path}")
    if (
        expected_tensor_schema_hash is not None
        and payload.get("tensor_schema_hash") != expected_tensor_schema_hash
    ):
        raise ValueError(f"tensor schema mismatch in {path}")
    if (
        expected_source_shard_sha256 is not None
        and payload.get("source_shard_sha256") != expected_source_shard_sha256
    ):
        raise ValueError(f"source shard mismatch in {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != int(payload["state_count"]):
        raise ValueError(f"state count mismatch in {path}")
    if sum(sample.actions.action_count for sample in samples) != int(payload["action_count"]):
        raise ValueError(f"action count mismatch in {path}")
    metadata = {key: value for key, value in payload.items() if key != "samples"}
    return samples, metadata


def valid_cache_record(
    record_path: Path,
    *,
    source_shard_sha256: str,
    tensor_schema_hash: str,
) -> dict[str, object] | None:
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cache_path = Path(str(record.get("cache_path", "")))
    if not cache_path.is_absolute():
        cache_path = record_path.parents[3] / cache_path
    expected = (
        record.get("status") == "COMPLETE"
        and record.get("source_shard_sha256") == source_shard_sha256
        and record.get("tensor_schema_hash") == tensor_schema_hash
        and "positive_count" in record
        and cache_path.exists()
        and cache_path.stat().st_size == int(record.get("cache_bytes", -1))
    )
    return record if expected else None
