#!/usr/bin/env python3
"""Build the Phase 6N combined 864-state CSG tensor cache."""

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

from rcias_clgri.csg import build_csg_from_schedule  # noqa: E402
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache, write_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts import run_phase6n_data_generation as generation  # noqa: E402
from scripts.run_phase6j_caur_pilot import atomic_csv, atomic_json, digest, search_stage  # noqa: E402


CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
SOURCE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
DATA_INTEGRITY = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/data_integrity.json"
NEW_REPLAYS = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/new_r12_collection/state_replays"
OLD_CACHE = ROOT / "outputs/phase6j_caur/tensor_cache"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def records_for_tensorization(group: pd.DataFrame) -> list[dict[str, object]]:
    ordered = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
    truth = ordered.continuation_advantage_mean.to_numpy(dtype=float)
    target_ids = ordered.target_set_id.astype(str).to_numpy()
    truth_order = np.lexsort((target_ids, -truth))
    ranks = np.empty(len(ordered), dtype=int)
    ranks[truth_order] = np.arange(1, len(ordered) + 1)
    best = float(np.max(truth))
    return [
        {
            "state_id": str(row.state_id),
            "target_set_id": str(row.target_set_id),
            "destroyed_operation_ids": str(row.target_operation_ids),
            "mean_relative_improvement": float(row.continuation_advantage_mean),
            "rank_within_state": int(ranks[index]),
            "rank_percentile": (int(ranks[index]) - 1) / max(len(ordered) - 1, 1),
            "regret_to_best": best - float(row.continuation_advantage_mean),
            "top1": int(ranks[index]) == 1,
            "top3": int(ranks[index]) <= 3,
            "arm_family": str(row.origin_family),
            "origin_destroy_operator": str(row.origin_destroy_operator),
            "origin_rules": str(row.origin_rules),
            "origin_families": str(row.origin_families),
        }
        for index, row in enumerate(ordered.itertuples(index=False))
    ]


def _old_samples() -> dict[str, NIStateSample]:
    manifest = pd.read_csv(OLD_CACHE / "tensor_manifest.csv")
    result: dict[str, NIStateSample] = {}
    for row in manifest.itertuples(index=False):
        samples, _ = load_shard_cache(
            Path(row.cache_path),
            expected_tensor_schema_hash=str(row.tensor_schema_hash),
            expected_source_shard_sha256=str(row.source_shard_sha256),
        )
        for sample in samples:
            result[sample.graph.state_id] = sample
    if len(result) != 288:
        raise RuntimeError("Phase 6J cache no longer contains 288 original states")
    return result


def validate_inputs() -> tuple[dict, dict, str, str]:
    config = load_json(CONFIG)
    integrity = load_json(DATA_INTEGRITY)
    source_sha256 = digest(SOURCE)
    integrity_sha256 = digest(DATA_INTEGRITY)
    checks = (
        integrity.get("status") == "PASS",
        integrity.get("combined_states") == 864,
        integrity.get("combined_candidate_rows") == 20441,
        integrity.get("artifacts", {}).get(str(SOURCE.relative_to(ROOT))) == source_sha256,
        integrity.get("historical_score_calls") == 0,
        integrity.get("r13_accessed") is False,
        integrity.get("r14_accessed") is False,
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6N tensor-cache input boundary failed")
    return config, integrity, source_sha256, integrity_sha256


def _valid_record(
    record_path: Path,
    *,
    source_sha256: str,
    tensor_schema_hash: str,
    integrity_sha256: str,
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
        record.get("schema") == "phase6n-combined-tensor-shard-v1",
        record.get("status") == "COMPLETE",
        record.get("data_integrity_sha256") == integrity_sha256,
        record.get("cache_sha256") == digest(cache_path),
        metadata.get("state_count") == record.get("state_count") == 48,
        metadata.get("action_count") == record.get("action_count"),
        record.get("historical_score_calls") == 0,
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    )
    return record if all(checks) else None


def build_new_sample(
    instance,
    group: pd.DataFrame,
    replay: dict,
    tensorizer: CSGTensorizer,
) -> NIStateSample:
    state_id = str(group.state_id.iloc[0])
    snapshot = replay["snapshot"]
    current = decode_candidate(
        instance, generation.candidate_from_dict(snapshot["current_candidate"])
    )
    if not math.isclose(
        current.makespan, float(snapshot["current_makespan"]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(f"Phase 6N tensor replay mismatch: {state_id}")
    progress = float(snapshot["search_progress"])
    graph = build_csg_from_schedule(
        instance,
        current.schedule,
        state_id=state_id,
        search_progress=progress,
        search_stage=search_stage(progress),
    )
    return NIStateSample(
        tensorizer.tensorize(graph),
        tensorize_action_records(graph, records_for_tensorization(group)),
        {
            "training_split": "R12_CAUR_FIT",
            "scale": str(group.scale.iloc[0]),
            "CF_level": str(group.CF_level.iloc[0]),
            "search_stage": str(group.search_stage.iloc[0]),
            "oof_fold": str(group.oof_fold.iloc[0]),
            "cell_replicate": str(group.cell_replicate.iloc[0]),
            "trajectory_seed": str(group.trajectory_seed.iloc[0]),
            "phase6n_data_origin": "NEW_ALNS_EXPANSION",
        },
    )


def main() -> None:
    config, integrity, source_sha256, integrity_sha256 = validate_inputs()
    frame = pd.read_parquet(SOURCE)
    old = _old_samples()
    replays = {path.stem: path for path in NEW_REPLAYS.glob("*.json")}
    if len(replays) != 576:
        raise RuntimeError("Phase 6N requires 576 new replay files")
    tensorizer = CSGTensorizer()
    records = []
    started = time.perf_counter()
    for instance_id, instance_rows in frame.groupby("instance_id", sort=True):
        record_path = OUT / f"{instance_id}.json"
        cache_path = OUT / f"{instance_id}.pt"
        existing = _valid_record(
            record_path,
            source_sha256=source_sha256,
            tensor_schema_hash=tensorizer.tensor_schema_hash,
            integrity_sha256=integrity_sha256,
        )
        if existing is not None:
            records.append(existing)
            print(json.dumps({"event": "phase6n_tensor_skip", "instance_id": instance_id}), flush=True)
            continue
        first = instance_rows.iloc[0]
        instance_path = ROOT / config["locked_inputs"]["instance_root"] / str(
            first.instance_relative_path
        )
        if digest(instance_path) != str(first.instance_sha256):
            raise RuntimeError(f"Phase 6N instance hash mismatch: {instance_id}")
        instance = load_phase6j_instance(instance_path)
        samples = []
        for state_id, group in instance_rows.groupby("state_id", sort=True):
            group = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
            origin = str(group.phase6n_data_origin.iloc[0])
            if origin == "ORIGINAL_PHASE6J_CAUR":
                sample = old[str(state_id)]
            elif origin == "NEW_ALNS_EXPANSION":
                sample = build_new_sample(
                    instance, group, load_json(replays[str(state_id)]), tensorizer
                )
            else:
                raise RuntimeError(f"unknown Phase 6N data origin: {origin}")
            if sample.actions.target_set_ids != tuple(group.target_set_id.astype(str)):
                raise RuntimeError(f"Phase 6N candidate identity/order mismatch: {state_id}")
            samples.append(sample)
        record = write_shard_cache(
            cache_path,
            samples,
            instance_id=str(instance_id),
            training_split="R12_CAUR_FIT_EXPANDED",
            source_shard_sha256=source_sha256,
        )
        record.update({
            "schema": "phase6n-combined-tensor-shard-v1",
            "data_integrity_sha256": integrity_sha256,
            "instance_sha256": digest(instance_path),
            "original_states": sum(
                sample.graph.state_id in old for sample in samples
            ),
            "new_states": sum(sample.graph.state_id not in old for sample in samples),
            "historical_score_calls": 0,
            "r13_accessed": False,
            "r14_accessed": False,
        })
        atomic_json(record, record_path)
        records.append(record)
        print(json.dumps({
            "event": "phase6n_tensor_complete",
            "instance_id": instance_id,
            "states": record["state_count"],
            "actions": record["action_count"],
        }), flush=True)

    manifest = pd.DataFrame(records).sort_values("instance_id")
    atomic_csv(manifest, OUT / "tensor_manifest.csv")
    checks = {
        "expected_shards": len(manifest) == 18,
        "expected_states": int(manifest.state_count.sum()) == 864,
        "expected_actions": int(manifest.action_count.sum()) == 20441,
        "original_states": int(manifest.original_states.sum()) == 288,
        "new_states": int(manifest.new_states.sum()) == 576,
        "tensor_schema_singleton": manifest.tensor_schema_hash.nunique() == 1,
        "source_hash_exact": set(manifest.source_shard_sha256) == {source_sha256},
        "data_integrity_hash_exact": set(manifest.data_integrity_sha256) == {integrity_sha256},
        "zero_historical_score_calls": int(manifest.historical_score_calls.sum()) == 0,
        "r13_r14_locked": not manifest.r13_accessed.any() and not manifest.r14_accessed.any(),
    }
    result = {
        "schema": "phase6n-combined-tensor-cache-integrity-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "states": int(manifest.state_count.sum()),
        "actions": int(manifest.action_count.sum()),
        "tensor_schema_hash": str(manifest.tensor_schema_hash.iloc[0]),
        "source_sha256": source_sha256,
        "data_integrity_sha256": integrity_sha256,
        "manifest_sha256": digest(OUT / "tensor_manifest.csv"),
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(result, OUT / "tensor_cache_integrity.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("Phase 6N tensor-cache integrity failed")


if __name__ == "__main__":
    main()
