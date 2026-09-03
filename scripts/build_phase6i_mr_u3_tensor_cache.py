#!/usr/bin/env python3
"""Build resumable R09 graph-tensor shards for conditional Phase 6I U3."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg_from_schedule  # noqa: E402
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache, write_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts.build_phase6i_mr_embedding_cache import (  # noqa: E402
    records_for_tensorization,
    replay_index,
)
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_csv,
    atomic_json,
    digest,
    load_json,
)
from scripts.run_phase6i_mr_pilot import _candidate_from_dict, _search_stage  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
SOURCE_PATH = ROOT / "outputs/phase6i_mr/training_data/r09_actions.parquet"
REPLAY_ROOT = ROOT / "outputs/phase6i_mr/collection/r09/state_replays"
OUT = ROOT / "outputs/phase6i_mr/u3_tensor_cache"


def valid(record_path: Path, source_hash: str, schema_hash: str) -> dict | None:
    cache_path = record_path.with_suffix(".pt")
    if not record_path.is_file() or not cache_path.is_file():
        return None
    try:
        record = load_json(record_path)
        _, metadata = load_shard_cache(
            cache_path,
            expected_tensor_schema_hash=schema_hash,
            expected_source_shard_sha256=source_hash,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if all([
        record.get("status") == "COMPLETE",
        record.get("cache_sha256") == digest(cache_path),
        metadata["state_count"] == record["state_count"],
        metadata["action_count"] == record["action_count"],
        record.get("r10_accessed") is False,
        record.get("r11_accessed") is False,
    ]):
        return record
    return None


def main() -> None:
    config = load_json(CONFIG_PATH)
    protocol = load_json(PROTOCOL_PATH)
    if not all([
        protocol.get("r10_accessed") is False,
        protocol.get("r11_accessed") is False,
        protocol["u3_activation_rule"],
    ]):
        raise RuntimeError("U3 tensorization must precede protected-split access")
    activation = load_json(ROOT / "outputs/phase6i_mr/model_training/u3_activation_decision.json")
    if activation.get("u3_activated") is not True:
        raise RuntimeError("U3 was not activated by the frozen R09 rule")
    frame = pd.read_parquet(SOURCE_PATH)
    replays = replay_index(REPLAY_ROOT)
    tensorizer = CSGTensorizer()
    source_hash = digest(SOURCE_PATH)
    records = []
    started = time.perf_counter()
    for instance_id, instance_rows in frame.groupby("instance_id", sort=True):
        cache_path = OUT / f"{instance_id}.pt"
        record_path = OUT / f"{instance_id}.json"
        existing = valid(record_path, source_hash, tensorizer.tensor_schema_hash)
        if existing is not None:
            records.append(existing)
            print({"event": "u3_tensor_skip", "instance_id": instance_id}, flush=True)
            continue
        instance_path = ROOT / config["instance_suite"]["root"] / str(instance_rows.instance_relative_path.iloc[0])
        instance = load_phase6i_instance(instance_path)
        samples = []
        for state_id, group in instance_rows.groupby("state_id", sort=True):
            replay = load_json(replays[str(state_id)])
            current = decode_candidate(instance, _candidate_from_dict(replay["current_candidate"]))
            if not math.isclose(current.makespan, float(replay["current_makespan"]), rel_tol=0, abs_tol=1e-9):
                raise RuntimeError(f"R09 replay mismatch: {state_id}")
            progress = float(replay["search_progress"])
            graph = build_csg_from_schedule(
                instance,
                current.schedule,
                state_id=str(state_id),
                search_progress=progress,
                search_stage=_search_stage(progress),
            )
            ordered = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
            samples.append(NIStateSample(
                tensorizer.tensorize(graph),
                tensorize_action_records(graph, records_for_tensorization(ordered)),
                {
                    "training_split": "R09",
                    "scale": str(ordered.scale.iloc[0]),
                    "CF_level": str(ordered.CF_level.iloc[0]),
                    "search_stage": str(ordered.search_stage.iloc[0]),
                    "oof_fold": str(ordered.oof_fold.iloc[0]),
                },
            ))
        record = write_shard_cache(
            cache_path,
            samples,
            instance_id=instance_id,
            training_split="R09",
            source_shard_sha256=source_hash,
        )
        record.update({
            "instance_sha256": digest(instance_path),
            "training_protocol_sha256": digest(PROTOCOL_PATH),
            "r10_accessed": False,
            "r11_accessed": False,
        })
        atomic_json(record, record_path)
        records.append(record)
        print({"event": "u3_tensor_complete", "instance_id": instance_id, "states": record["state_count"]}, flush=True)
    manifest = pd.DataFrame(records).sort_values("instance_id")
    atomic_csv(manifest, OUT / "u3_tensor_manifest.csv")
    checks = {
        "expected_shards": len(manifest) == 18,
        "expected_states": int(manifest.state_count.sum()) == 1620,
        "expected_actions": int(manifest.action_count.sum()) == 6480,
        "tensor_schema_singleton": manifest.tensor_schema_hash.nunique() == 1,
        "r10_not_accessed": True,
        "r11_not_accessed": True,
    }
    integrity = {
        "schema": "phase6i-mr-u3-tensor-cache-integrity-v1.2",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "shards": len(manifest),
        "states": int(manifest.state_count.sum()),
        "actions": int(manifest.action_count.sum()),
        "tensor_schema_hash": tensorizer.tensor_schema_hash,
        "manifest_sha256": digest(OUT / "u3_tensor_manifest.csv"),
        "source_sha256": source_hash,
        "runtime_seconds": time.perf_counter() - started,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(integrity, OUT / "u3_tensor_integrity.json")
    print(integrity, flush=True)
    if integrity["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
