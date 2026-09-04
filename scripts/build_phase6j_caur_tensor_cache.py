#!/usr/bin/env python3
"""Build resumable R12 graph-tensor shards for CAUR J1/J2 training."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import grouped_oof_fold  # noqa: E402
from rcias_clgri.csg import build_csg_from_schedule  # noqa: E402
from rcias_clgri.data.phase6j_access import (  # noqa: E402
    load_phase6j_instance,
    verify_r12_collection_authorization,
)
from rcias_clgri.ni.cache import load_shard_cache, write_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts.run_phase6j_caur_pilot import (  # noqa: E402
    atomic_csv,
    atomic_json,
    candidate_from_dict,
    digest,
    load_json,
    search_stage,
)


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
COLLECTOR_PATH = ROOT / "scripts/run_phase6j_caur_collection.py"
FREEZE_PATH = ROOT / "outputs/phase6j_caur/frozen/r12_horizon_freeze.json"
COLLECTION = ROOT / "outputs/phase6j_caur/r12_collection"
SOURCE_PATH = COLLECTION / "r12_grouped_labels.parquet"
REPLAY_ROOT = COLLECTION / "state_replays"
OUT = ROOT / "outputs/phase6j_caur/tensor_cache"


def records_for_tensorization(group: pd.DataFrame) -> list[dict[str, object]]:
    ordered = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
    truth = ordered.continuation_advantage_mean.to_numpy(dtype=float)
    target_ids = ordered.target_set_id.astype(str).to_numpy()
    truth_order = np.lexsort((target_ids, -truth))
    ranks = np.empty(len(ordered), dtype=int)
    ranks[truth_order] = np.arange(1, len(ordered) + 1)
    best = float(np.max(truth))
    rows = []
    for index, row in enumerate(ordered.itertuples(index=False)):
        rank = int(ranks[index])
        rows.append({
            "state_id": str(row.state_id),
            "target_set_id": str(row.target_set_id),
            "destroyed_operation_ids": str(row.target_operation_ids),
            "mean_relative_improvement": float(row.continuation_advantage_mean),
            "rank_within_state": rank,
            "rank_percentile": (rank - 1) / max(len(ordered) - 1, 1),
            "regret_to_best": best - float(row.continuation_advantage_mean),
            "top1": rank == 1,
            "top3": rank <= 3,
            "arm_family": str(row.origin_family),
            "origin_destroy_operator": str(row.origin_destroy_operator),
            "origin_rules": str(row.origin_rules),
            "origin_families": str(row.origin_families),
        })
    return rows


def replay_index() -> dict[str, Path]:
    result = {path.stem: path for path in REPLAY_ROOT.glob("*.json")}
    if len(result) != 288:
        raise RuntimeError("R12 tensorization requires 288 unique replay files")
    return result


def valid_record(
    record_path: Path,
    *,
    source_sha256: str,
    tensor_schema_hash: str,
    collection_integrity_sha256: str,
) -> dict | None:
    cache_path = record_path.with_suffix(".pt")
    if not record_path.is_file() or not cache_path.is_file():
        return None
    try:
        record = load_json(record_path)
        _, metadata = load_shard_cache(
            cache_path,
            expected_tensor_schema_hash=tensor_schema_hash,
            expected_source_shard_sha256=source_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    checks = (
        record.get("schema") == "phase6j-caur-r12-tensor-shard-v1",
        record.get("status") == "COMPLETE",
        record.get("collection_integrity_sha256") == collection_integrity_sha256,
        record.get("cache_sha256") == digest(cache_path),
        metadata.get("state_count") == record.get("state_count") == 16,
        metadata.get("action_count") == record.get("action_count"),
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    )
    return record if all(checks) else None


def validate_collection() -> tuple[dict, str]:
    verify_r12_collection_authorization(
        FREEZE_PATH,
        project_root=ROOT,
        config_path=CONFIG_PATH,
        collection_script_path=COLLECTOR_PATH,
    )
    integrity_path = COLLECTION / "collection_integrity.json"
    integrity = load_json(integrity_path)
    checks = (
        integrity.get("status") == "PASS",
        all(integrity.get("checks", {}).values()),
        integrity.get("states") == 288,
        integrity.get("grouped_rows") == 6809,
        integrity.get("r13_accessed") is False,
        integrity.get("r14_accessed") is False,
        digest(SOURCE_PATH) == integrity.get("r12_grouped_labels_sha256"),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("R12 collection is not a valid training source")
    return integrity, digest(integrity_path)


def main() -> None:
    integrity, integrity_sha256 = validate_collection()
    config = load_json(CONFIG_PATH)
    frame = pd.read_parquet(SOURCE_PATH)
    replays = replay_index()
    tensorizer = CSGTensorizer()
    source_sha256 = digest(SOURCE_PATH)
    records = []
    started = time.perf_counter()
    for instance_id, instance_rows in frame.groupby("instance_id", sort=True):
        cache_path = OUT / f"{instance_id}.pt"
        record_path = OUT / f"{instance_id}.json"
        existing = valid_record(
            record_path,
            source_sha256=source_sha256,
            tensor_schema_hash=tensorizer.tensor_schema_hash,
            collection_integrity_sha256=integrity_sha256,
        )
        if existing is not None:
            records.append(existing)
            print(json.dumps({
                "event": "phase6j_tensor_skip",
                "instance_id": instance_id,
            }), flush=True)
            continue
        first = instance_rows.iloc[0]
        instance_path = (
            ROOT / config["instance_suite"]["root"] / str(first.instance_relative_path)
        )
        instance = load_phase6j_instance(instance_path)
        if digest(instance_path) != str(first.instance_sha256):
            raise RuntimeError(f"R12 tensor instance hash mismatch: {instance_id}")
        samples = []
        for state_id, group in instance_rows.groupby("state_id", sort=True):
            replay = load_json(replays[str(state_id)])
            snapshot = replay["snapshot"]
            current = decode_candidate(
                instance, candidate_from_dict(snapshot["current_candidate"])
            )
            if not math.isclose(
                current.makespan,
                float(snapshot["current_makespan"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(f"R12 tensor replay mismatch: {state_id}")
            progress = float(snapshot["search_progress"])
            graph = build_csg_from_schedule(
                instance,
                current.schedule,
                state_id=str(state_id),
                search_progress=progress,
                search_stage=search_stage(progress),
            )
            actions = records_for_tensorization(group)
            samples.append(NIStateSample(
                tensorizer.tensorize(graph),
                tensorize_action_records(graph, actions),
                {
                    "training_split": "R12_CAUR_FIT",
                    "scale": str(group.scale.iloc[0]),
                    "CF_level": str(group.CF_level.iloc[0]),
                    "search_stage": str(group.search_stage.iloc[0]),
                    "oof_fold": str(grouped_oof_fold(
                        str(group.scale.iloc[0]), str(group.CF_level.iloc[0])
                    )),
                    "cell_replicate": str(group.cell_replicate.iloc[0]),
                    "trajectory_seed": str(group.trajectory_seed.iloc[0]),
                },
            ))
        record = write_shard_cache(
            cache_path,
            samples,
            instance_id=str(instance_id),
            training_split="R12_CAUR_FIT",
            source_shard_sha256=source_sha256,
        )
        record.update({
            "schema": "phase6j-caur-r12-tensor-shard-v1",
            "collection_integrity_sha256": integrity_sha256,
            "instance_sha256": digest(instance_path),
            "r12_horizon_freeze_sha256": integrity["r12_horizon_freeze_sha256"],
            "r13_accessed": False,
            "r14_accessed": False,
        })
        atomic_json(record, record_path)
        records.append(record)
        print(json.dumps({
            "event": "phase6j_tensor_complete",
            "instance_id": instance_id,
            "states": record["state_count"],
            "actions": record["action_count"],
        }), flush=True)

    manifest = pd.DataFrame(records).sort_values("instance_id")
    atomic_csv(manifest, OUT / "tensor_manifest.csv")
    checks = {
        "expected_shards": len(manifest) == 18,
        "expected_states": int(manifest.state_count.sum()) == 288,
        "expected_actions": int(manifest.action_count.sum()) == 6809,
        "tensor_schema_singleton": manifest.tensor_schema_hash.nunique() == 1,
        "source_hash_exact": set(manifest.source_shard_sha256) == {source_sha256},
        "collection_integrity_hash_exact": set(manifest.collection_integrity_sha256)
        == {integrity_sha256},
        "r13_r14_not_accessed": not manifest.r13_accessed.any()
        and not manifest.r14_accessed.any(),
    }
    result = {
        "schema": "phase6j-caur-r12-tensor-cache-integrity-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "shards": len(manifest),
        "states": int(manifest.state_count.sum()),
        "actions": int(manifest.action_count.sum()),
        "tensor_schema_hash": str(manifest.tensor_schema_hash.iloc[0]),
        "manifest_sha256": digest(OUT / "tensor_manifest.csv"),
        "source_sha256": source_sha256,
        "collection_integrity_sha256": integrity_sha256,
        "runtime_seconds": time.perf_counter() - started,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(result, OUT / "tensor_cache_integrity.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
