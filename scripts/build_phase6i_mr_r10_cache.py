#!/usr/bin/env python3
"""Build immutable frozen-embedding and graph-tensor caches for one-time R10."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg_from_schedule  # noqa: E402
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache, write_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts.build_phase6i_mr_embedding_cache import (  # noqa: E402
    append_embeddings,
    records_for_tensorization,
    replay_index,
)
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    add_context,
    apply_normalization,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)
from scripts.run_phase6i_mr_pilot import _candidate_from_dict, _search_stage  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PRE_R10_FREEZE = ROOT / "outputs/phase6i_mr/pre_r10/pre_r10_freeze.json"
AUTHORIZATION = ROOT / "outputs/phase6i_mr/frozen/r10_collection_authorization.json"
R10_COLLECTION = ROOT / "outputs/phase6i_mr/collection/r10"
TRAINING_CONSTANTS = ROOT / "outputs/phase6i_mr/training_data/training_constants.json"
OUT = ROOT / "outputs/phase6i_mr/r10_selection/cache"


def _validate_unlock() -> tuple[dict, dict, dict]:
    config = load_json(CONFIG_PATH)
    freeze = load_json(PRE_R10_FREEZE)
    authorization = load_json(AUTHORIZATION)
    collection = load_json(R10_COLLECTION / "collection_integrity.json")
    if not all([
        freeze.get("status") == "PASS",
        freeze.get("r10_accessed") is False,
        freeze.get("r11_accessed") is False,
        authorization.get("status") == "FROZEN_BEFORE_ONE_TIME_R10_ACCESS",
        authorization.get("pre_r10_freeze_sha256") == digest(PRE_R10_FREEZE),
        authorization.get("code_hashes", {}).get(
            "scripts/build_phase6i_mr_r10_cache.py"
        ) == digest(Path(__file__)),
        collection.get("status") == "PASS",
        collection.get("r10_accessed") is True,
        collection.get("r11_accessed") is False,
    ]):
        raise RuntimeError("R10 cache access boundary or code freeze is invalid")
    return config, freeze, collection


def _save_shard(
    instance_id: str,
    frame: pd.DataFrame,
    samples: list[NIStateSample],
    *,
    tensor_schema_hash: str,
    source_hash: str,
    checkpoint_hash: str,
    replay_error: float,
    runtime_seconds: float,
) -> dict[str, object]:
    embedding_path = OUT / "embeddings" / f"{instance_id}.parquet"
    tensor_path = OUT / "tensors" / f"{instance_id}.pt"
    record_path = OUT / "records" / f"{instance_id}.json"
    atomic_parquet(frame, embedding_path)
    write_shard_cache(
        tensor_path,
        samples,
        instance_id=instance_id,
        training_split="R10",
        source_shard_sha256=source_hash,
    )
    _, metadata = load_shard_cache(
        tensor_path,
        expected_tensor_schema_hash=tensor_schema_hash,
        expected_source_shard_sha256=source_hash,
    )
    record = {
        "schema": "phase6i-mr-r10-selection-cache-shard-v1.2",
        "status": "COMPLETE",
        "instance_id": instance_id,
        "state_count": int(frame.state_id.nunique()),
        "action_count": len(frame),
        "embedding_dim": len([c for c in frame if c.startswith("embedding_")]),
        "tensor_schema_hash": tensor_schema_hash,
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": source_hash,
        "embedding_path": str(embedding_path.relative_to(ROOT)),
        "embedding_sha256": digest(embedding_path),
        "tensor_cache_path": str(tensor_path.relative_to(ROOT)),
        "tensor_cache_sha256": digest(tensor_path),
        "tensor_state_count": int(metadata["state_count"]),
        "tensor_action_count": int(metadata["action_count"]),
        "maximum_frozen_score_replay_error": replay_error,
        "runtime_seconds": runtime_seconds,
        "r10_accessed": True,
        "r11_accessed": False,
    }
    atomic_json(record, record_path)
    record["record_path"] = str(record_path.relative_to(ROOT))
    record["record_sha256"] = digest(record_path)
    return record


def _valid_shard(
    instance_id: str,
    *,
    tensor_schema_hash: str,
    source_hash: str,
    checkpoint_hash: str,
) -> dict[str, object] | None:
    record_path = OUT / "records" / f"{instance_id}.json"
    if not record_path.is_file():
        return None
    try:
        record = load_json(record_path)
        embedding_path = ROOT / record["embedding_path"]
        tensor_path = ROOT / record["tensor_cache_path"]
        _, metadata = load_shard_cache(
            tensor_path,
            expected_tensor_schema_hash=tensor_schema_hash,
            expected_source_shard_sha256=source_hash,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if all([
        record.get("status") == "COMPLETE",
        record.get("instance_id") == instance_id,
        record.get("tensor_schema_hash") == tensor_schema_hash,
        record.get("source_sha256") == source_hash,
        record.get("checkpoint_sha256") == checkpoint_hash,
        record.get("embedding_sha256") == digest(embedding_path),
        record.get("tensor_cache_sha256") == digest(tensor_path),
        int(record.get("state_count")) == int(metadata["state_count"]),
        int(record.get("action_count")) == int(metadata["action_count"]),
        record.get("r10_accessed") is True,
        record.get("r11_accessed") is False,
    ]):
        result = dict(record)
        result["record_path"] = str(record_path.relative_to(ROOT))
        result["record_sha256"] = digest(record_path)
        return result
    return None


def main() -> None:
    config, freeze, collection = _validate_unlock()
    labels_path = R10_COLLECTION / "forced_action_labels.parquet"
    source_hash = digest(labels_path)
    frame = add_context(pd.read_parquet(labels_path))
    constants = load_json(TRAINING_CONSTANTS)
    frame = apply_normalization(frame, constants["context_normalization"])
    policy = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device="cuda",
        proposal_seed_namespace=config["rng_namespaces"]["frozen_live_proposal"],
        deployment_artifact=ROOT / config["locked_inputs"]["phase6h_policy"],
    )
    policy.model.eval()
    checkpoint_hash = config["locked_inputs"]["phase6f_checkpoint_sha256"]
    if policy.checkpoint_sha256 != checkpoint_hash:
        raise RuntimeError("frozen Phase6F checkpoint hash mismatch")
    replays = replay_index(R10_COLLECTION / "state_replays")
    started = time.perf_counter()
    records = []
    for instance_id, instance_rows in frame.groupby("instance_id", sort=True):
        existing = _valid_shard(
            str(instance_id),
            tensor_schema_hash=policy.tensorizer.tensor_schema_hash,
            source_hash=source_hash,
            checkpoint_hash=checkpoint_hash,
        )
        if existing is not None:
            records.append(existing)
            print(json.dumps({"event": "r10_cache_skip", "instance_id": instance_id}), flush=True)
            continue
        shard_started = time.perf_counter()
        instance_path = (
            ROOT / config["instance_suite"]["root"]
            / str(instance_rows.instance_relative_path.iloc[0])
        )
        instance = load_phase6i_instance(instance_path)
        output_rows = []
        samples = []
        maximum_error = 0.0
        for state_id, group in instance_rows.groupby("state_id", sort=True):
            replay = load_json(replays[str(state_id)])
            current = decode_candidate(
                instance, _candidate_from_dict(replay["current_candidate"])
            )
            if not math.isclose(
                current.makespan,
                float(replay["current_makespan"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(f"R10 state replay mismatch: {state_id}")
            progress = float(replay["search_progress"])
            graph = build_csg_from_schedule(
                instance,
                current.schedule,
                state_id=str(state_id),
                search_progress=progress,
                search_stage=_search_stage(progress),
            )
            ordered = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
            sample = NIStateSample(
                policy.tensorizer.tensorize(graph),
                tensorize_action_records(graph, records_for_tensorization(ordered)),
                {
                    "training_split": "R10",
                    "scale": str(ordered.scale.iloc[0]),
                    "CF_level": str(ordered.CF_level.iloc[0]),
                    "search_stage": str(ordered.search_stage.iloc[0]),
                },
            )
            batch = batch_state_samples([sample]).to(policy.device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if policy.device.type == "cuda" else nullcontext()
            )
            with torch.inference_mode(), autocast:
                output = policy.model(batch)
            embeddings = output.action_embeddings.detach().float().cpu().numpy()
            scores = output.scores.detach().float().cpu().numpy()
            utilities = output.utility_predictions.detach().float().cpu().numpy()
            error = float(np.max(np.abs(scores - ordered.raw_score.to_numpy(dtype=float))))
            maximum_error = max(maximum_error, error)
            if error > 0.01:
                raise RuntimeError(f"R10 frozen score replay error {error}: {state_id}")
            output_rows.append(append_embeddings(ordered, embeddings, scores, utilities))
            samples.append(sample)
        output = pd.concat(output_rows, ignore_index=True)
        record = _save_shard(
            str(instance_id),
            output,
            samples,
            tensor_schema_hash=policy.tensorizer.tensor_schema_hash,
            source_hash=source_hash,
            checkpoint_hash=checkpoint_hash,
            replay_error=maximum_error,
            runtime_seconds=time.perf_counter() - shard_started,
        )
        records.append(record)
        print(json.dumps({
            "event": "r10_cache_complete",
            "instance_id": instance_id,
            "states": record["state_count"],
        }), flush=True)

    manifest = pd.DataFrame(records).sort_values("instance_id")
    manifest_path = OUT / "cache_manifest.csv"
    atomic_csv(manifest, manifest_path)
    checks = {
        "r10_collection_pass": collection["status"] == "PASS",
        "expected_shards": len(manifest) == 18,
        "expected_states": int(manifest.state_count.sum()) == 1620,
        "expected_actions": int(manifest.action_count.sum()) == 6480,
        "embedding_dim_128": bool(manifest.embedding_dim.eq(128).all()),
        "tensor_schema_singleton": manifest.tensor_schema_hash.nunique() == 1,
        "score_replay_within_0_01": bool(
            manifest.maximum_frozen_score_replay_error.le(0.01).all()
        ),
        "pre_r10_freeze_hash_exact": (
            load_json(AUTHORIZATION)["pre_r10_freeze_sha256"] == digest(PRE_R10_FREEZE)
        ),
        "r10_accessed_once": True,
        "r11_not_accessed": True,
    }
    integrity = {
        "schema": "phase6i-mr-r10-selection-cache-integrity-v1.2",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "pre_r10_freeze_sha256": digest(PRE_R10_FREEZE),
        "r10_collection_integrity_sha256": digest(
            R10_COLLECTION / "collection_integrity.json"
        ),
        "source_sha256": source_hash,
        "tensor_schema_hash": policy.tensorizer.tensor_schema_hash,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": digest(manifest_path),
        "runtime_seconds": time.perf_counter() - started,
        "r10_accessed": True,
        "r11_accessed": False,
    }
    atomic_json(integrity, OUT / "cache_integrity.json")
    print(json.dumps(integrity, indent=2, sort_keys=True), flush=True)
    if integrity["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
