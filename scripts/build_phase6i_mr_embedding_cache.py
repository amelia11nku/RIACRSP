#!/usr/bin/env python3
"""Build resumable frozen-embedding caches for Phase 6I-MR head training."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg, build_csg_from_schedule  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.data.phase6c import reconstruct_state_from_instance  # noqa: E402
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    CONTEXT_SOURCES,
    add_context,
    apply_normalization,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)
from scripts.run_phase6i_mr_pilot import (  # noqa: E402
    _candidate_from_dict,
    _search_stage,
    _state_context,
)
from rcias_clgri.analysis.phase6i_mr import _state_feature_summary  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
DATA_ROOT = ROOT / "outputs/phase6i_mr/training_data"
DATA_FREEZE_PATH = DATA_ROOT / "training_data_freeze.json"
R09_COLLECTION = ROOT / "outputs/phase6i_mr/collection/r09"
PILOT = ROOT / "outputs/phase6i_mr/pilot_v12"
OLD_INSTANCE_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
OUT = ROOT / "outputs/phase6i_mr/embedding_cache"


def atomic_record(payload: dict, path: Path) -> None:
    atomic_json(payload, path)


def records_for_tensorization(group: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    width = len(group)
    for row in group.itertuples(index=False):
        rank = int(row.within_state_true_rank)
        rows.append({
            "state_id": row.state_id,
            "target_set_id": row.target_set_id,
            "destroyed_operation_ids": row.target_operation_ids,
            "mean_relative_improvement": float(row.decoded_immediate_utility),
            "rank_within_state": rank,
            "rank_percentile": (rank - 1) / max(width - 1, 1),
            "regret_to_best": float(row.regret_to_best),
            "top1": rank == 1,
            "top3": rank <= 3,
            "arm_family": row.origin_family,
            "origin_destroy_operator": row.origin_destroy_operator,
            "origin_rules": row.origin_rules,
            "origin_families": row.origin_family,
        })
    return rows


def infer(policy: FrozenLiveInference, sample: NIStateSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = batch_state_samples([sample]).to(policy.device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if policy.device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast:
        output = policy.model(batch)
    return (
        output.action_embeddings.detach().float().cpu().numpy(),
        output.scores.detach().float().cpu().numpy(),
        output.utility_predictions.detach().float().cpu().numpy(),
    )


def append_embeddings(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    scores: np.ndarray,
    utilities: np.ndarray,
) -> pd.DataFrame:
    if len(frame) != len(embeddings):
        raise RuntimeError("action embedding count mismatch")
    result = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(
                embeddings,
                columns=[
                    f"embedding_{index:03d}"
                    for index in range(embeddings.shape[1])
                ],
            ),
        ],
        axis=1,
    )
    result["frozen_reference_score"] = scores
    result["frozen_reference_utility"] = utilities
    return result


def replay_index(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*.json"):
        if path.stem in result:
            raise RuntimeError(f"duplicate replay state: {path.stem}")
        result[path.stem] = path
    return result


def cache_paths(source: str, instance_id: str) -> tuple[Path, Path]:
    directory = OUT / source.lower()
    return directory / f"{instance_id}.parquet", directory / f"{instance_id}.json"


def valid_record(
    source: str,
    instance_id: str,
    *,
    data_freeze_hash: str,
    checkpoint_hash: str,
) -> dict | None:
    table_path, record_path = cache_paths(source, instance_id)
    if not table_path.is_file() or not record_path.is_file():
        return None
    try:
        record = load_json(record_path)
    except (OSError, json.JSONDecodeError):
        return None
    if all([
        record.get("status") == "COMPLETE",
        record.get("source") == source,
        record.get("instance_id") == instance_id,
        record.get("training_data_freeze_sha256") == data_freeze_hash,
        record.get("checkpoint_sha256") == checkpoint_hash,
        record.get("table_sha256") == digest(table_path),
        record.get("r10_accessed") is False,
        record.get("r11_accessed") is False,
    ]):
        return record
    return None


def save_shard(
    source: str,
    instance_id: str,
    frame: pd.DataFrame,
    *,
    state_count: int,
    data_freeze_hash: str,
    checkpoint_hash: str,
    tensor_schema_hash: str,
    source_hashes: dict[str, str],
    score_max_abs_error: float | None,
    runtime_seconds: float,
) -> dict:
    embedding_columns = [column for column in frame if column.startswith("embedding_")]
    if not all([
        len(frame) > 0,
        frame.state_id.nunique() == state_count,
        len(embedding_columns) == 128,
        np.isfinite(frame[embedding_columns].to_numpy(dtype=float)).all(),
    ]):
        raise RuntimeError(f"invalid embedding shard: {source}/{instance_id}")
    table_path, record_path = cache_paths(source, instance_id)
    atomic_parquet(frame, table_path)
    record = {
        "schema": "phase6i-mr-frozen-embedding-shard-v1.2",
        "status": "COMPLETE",
        "source": source,
        "instance_id": instance_id,
        "state_count": state_count,
        "action_count": len(frame),
        "embedding_dim": len(embedding_columns),
        "tensor_schema_hash": tensor_schema_hash,
        "training_data_freeze_sha256": data_freeze_hash,
        "checkpoint_sha256": checkpoint_hash,
        "source_hashes": source_hashes,
        "score_max_abs_error": score_max_abs_error,
        "runtime_seconds": runtime_seconds,
        "table_path": str(table_path),
        "table_sha256": digest(table_path),
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_record(record, record_path)
    return record


def build_live_source(
    source: str,
    frame: pd.DataFrame,
    replay_root: Path,
    instance_root: Path,
    policy: FrozenLiveInference,
    *,
    data_freeze_hash: str,
    checkpoint_hash: str,
) -> list[dict]:
    replays = replay_index(replay_root)
    records = []
    for instance_id, instance_rows in frame.groupby("instance_id", sort=True):
        existing = valid_record(
            source,
            instance_id,
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
        )
        if existing is not None:
            records.append(existing)
            print(json.dumps({"event": "cache_skip", **existing}), flush=True)
            continue
        started = time.perf_counter()
        relative_path = str(instance_rows.instance_relative_path.iloc[0])
        instance_path = instance_root / relative_path
        instance = load_phase6i_instance(instance_path)
        output_rows = []
        maximum_error = 0.0
        for state_id, group in instance_rows.groupby("state_id", sort=True):
            replay_path = replays.get(str(state_id))
            if replay_path is None:
                raise RuntimeError(f"missing replay: {state_id}")
            replay = load_json(replay_path)
            current = decode_candidate(
                instance, _candidate_from_dict(replay["current_candidate"])
            )
            if not math.isclose(
                current.makespan,
                float(replay["current_makespan"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(f"replay mismatch: {state_id}")
            progress = float(replay["search_progress"])
            graph = build_csg_from_schedule(
                instance,
                current.schedule,
                state_id=str(state_id),
                search_progress=progress,
                search_stage=_search_stage(progress),
            )
            ordered = group.sort_values("candidate_role").reset_index(drop=True)
            actions = tensorize_action_records(
                graph, records_for_tensorization(ordered)
            )
            sample = NIStateSample(
                policy.tensorizer.tensorize(graph),
                actions,
                {
                    "training_split": source,
                    "scale": str(ordered.scale.iloc[0]),
                    "CF_level": str(ordered.CF_level.iloc[0]),
                    "search_stage": str(ordered.search_stage.iloc[0]),
                },
            )
            embeddings, scores, utilities = infer(policy, sample)
            error = float(np.max(np.abs(
                scores - ordered.raw_score.to_numpy(dtype=float)
            )))
            maximum_error = max(maximum_error, error)
            if error > 0.01:
                raise RuntimeError(
                    f"frozen score replay error {error:.6f}: {state_id}"
                )
            output_rows.append(
                append_embeddings(ordered, embeddings, scores, utilities)
            )
        result = pd.concat(output_rows, ignore_index=True)
        record = save_shard(
            source,
            instance_id,
            result,
            state_count=instance_rows.state_id.nunique(),
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
            tensor_schema_hash=policy.tensorizer.tensor_schema_hash,
            source_hashes={
                "instance_sha256": digest(instance_path),
                "source_frame_sha256": digest(
                    DATA_ROOT / (
                        "r09_actions.parquet"
                        if source == "R09" else "continuation_actions.parquet"
                    )
                ),
            },
            score_max_abs_error=maximum_error,
            runtime_seconds=time.perf_counter() - started,
        )
        records.append(record)
        print(json.dumps({"event": "cache_complete", **record}), flush=True)
    return records


def build_old_source(
    selection: pd.DataFrame,
    policy: FrozenLiveInference,
    constants: dict,
    *,
    data_freeze_hash: str,
    checkpoint_hash: str,
) -> list[dict]:
    records = []
    normalization = constants["context_normalization"]
    p99 = float(constants["immediate_utility_p99_absolute"])
    for instance_id, selected_rows in selection.groupby("instance_id", sort=True):
        existing = valid_record(
            "OLD_TRAIN",
            instance_id,
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
        )
        if existing is not None:
            records.append(existing)
            print(json.dumps({"event": "cache_skip", **existing}), flush=True)
            continue
        started = time.perf_counter()
        first = selected_rows.iloc[0]
        cache_path = Path(str(first.cache_path))
        samples, metadata = load_shard_cache(
            cache_path,
            expected_tensor_schema_hash=policy.tensorizer.tensor_schema_hash,
        )
        selected_ids = set(selected_rows.state_id)
        sample_map = {
            sample.graph.state_id: sample
            for sample in samples if sample.graph.state_id in selected_ids
        }
        if set(sample_map) != selected_ids:
            raise RuntimeError(f"old cache state selection mismatch: {instance_id}")
        states_path = Path(str(first.source_states_path))
        states = pd.read_parquet(states_path)
        states = states[states.state_id.isin(selected_ids)].set_index("state_id")
        instance = load_instance(
            OLD_INSTANCE_ROOT / str(states.iloc[0].instance_relative_path)
        )
        h1 = solve_dispatching(instance, "H1")
        output_rows = []
        for selected in selected_rows.sort_values("state_id").itertuples(index=False):
            sample = sample_map[selected.state_id]
            embeddings, scores, utilities = infer(policy, sample)
            state_record = states.loc[selected.state_id].to_dict()
            state_record["state_id"] = selected.state_id
            reconstructed = reconstruct_state_from_instance(instance, state_record)
            graph = build_csg(reconstructed, state_record)
            context = _state_context(
                instance,
                reconstructed.decoded,
                h1.schedule,
                SimpleNamespace(
                    graph=graph,
                    state_feature_summary=_state_feature_summary(graph),
                ),
            )
            context_frame = apply_normalization(
                add_context(pd.DataFrame([context])), normalization
            ).iloc[0]
            actions = sample.actions
            width = actions.action_count
            frame = pd.DataFrame({
                "source": ["OLD_PHASE6F_TRAIN"] * width,
                "instance_id": [instance_id] * width,
                "state_id": [selected.state_id] * width,
                "scale": [selected.scale] * width,
                "CF_level": [selected.CF_level] * width,
                "search_stage": [selected.search_stage] * width,
                "candidate_role": ["EARLIER_PHASE6F_BANK"] * width,
                "target_set_id": list(actions.target_set_ids),
                "origin_family": list(actions.arm_family),
                "origin_destroy_operator": list(actions.origin_destroy_operator),
                "origin_rules": list(actions.origin_rules),
                "decoded_immediate_utility": actions.utility.numpy(),
                "normalized_immediate_utility": np.clip(
                    actions.utility.numpy() / p99, -1.0, 1.0
                ),
                "positive_label": actions.positive.numpy().astype(bool),
                "within_state_true_rank": actions.rank_within_state.numpy(),
                "regret_to_best": actions.regret_to_best.numpy(),
            })
            for name in CONTEXT_SOURCES:
                frame[f"raw_context__{name}"] = context_frame[
                    f"raw_context__{name}"
                ]
                frame[f"context__{name}"] = context_frame[f"context__{name}"]
            output_rows.append(
                append_embeddings(frame, embeddings, scores, utilities)
            )
        result = pd.concat(output_rows, ignore_index=True)
        record = save_shard(
            "OLD_TRAIN",
            instance_id,
            result,
            state_count=len(selected_rows),
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
            tensor_schema_hash=policy.tensorizer.tensor_schema_hash,
            source_hashes={
                "old_tensor_cache_sha256": digest(cache_path),
                "old_states_sha256": digest(states_path),
                "old_selection_sha256": digest(
                    DATA_ROOT / "old_train_state_selection.csv"
                ),
            },
            score_max_abs_error=None,
            runtime_seconds=time.perf_counter() - started,
        )
        records.append(record)
        print(json.dumps({"event": "cache_complete", **record}), flush=True)
        del samples, sample_map
    return records


def write_progress(records: list[dict], expected_shards: int, started: float) -> None:
    atomic_json({
        "schema": "phase6i-mr-embedding-cache-progress-v1.2",
        "status": "COMPLETE" if len(records) == expected_shards else "RUNNING",
        "completed_shards": len(records),
        "expected_shards": expected_shards,
        "completed_states": sum(int(row["state_count"]) for row in records),
        "completed_actions": sum(int(row["action_count"]) for row in records),
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r10_accessed": False,
        "r11_accessed": False,
    }, OUT / "progress.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["R09", "CONTINUATION", "OLD_TRAIN"],
        default=["R09", "CONTINUATION", "OLD_TRAIN"],
    )
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    data_freeze = load_json(DATA_FREEZE_PATH)
    if not all([
        data_freeze.get("status") == "FROZEN_BEFORE_MODEL_FIT_AND_R10_ACCESS",
        data_freeze.get("r10_accessed") is False,
        data_freeze.get("r11_accessed") is False,
    ]):
        raise RuntimeError("training data is not frozen before R10 access")
    for record in data_freeze["frozen_outputs"].values():
        if digest(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("training-data freeze output hash mismatch")
    data_freeze_hash = digest(DATA_FREEZE_PATH)
    checkpoint_hash = config["locked_inputs"]["phase6f_checkpoint_sha256"]
    policy = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["frozen_live_proposal"],
        deployment_artifact=ROOT / config["locked_inputs"]["phase6h_policy"],
    )
    if policy.checkpoint_sha256 != checkpoint_hash:
        raise RuntimeError("frozen checkpoint hash mismatch")
    policy.model.eval()
    started = time.perf_counter()
    all_records = []
    expected_by_source = {"R09": 18, "CONTINUATION": 9, "OLD_TRAIN": 9}
    if "R09" in args.sources:
        r09 = pd.read_parquet(DATA_ROOT / "r09_actions.parquet")
        all_records.extend(build_live_source(
            "R09",
            r09,
            R09_COLLECTION / "state_replays",
            ROOT / config["instance_suite"]["root"],
            policy,
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
        ))
        write_progress(all_records, sum(expected_by_source[s] for s in args.sources), started)
    if "CONTINUATION" in args.sources:
        continuation = pd.read_parquet(DATA_ROOT / "continuation_actions.parquet")
        all_records.extend(build_live_source(
            "CONTINUATION",
            continuation,
            PILOT / "state_replays",
            ROOT / config["instance_suite"]["root"],
            policy,
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
        ))
        write_progress(all_records, sum(expected_by_source[s] for s in args.sources), started)
    if "OLD_TRAIN" in args.sources:
        selection = pd.read_csv(DATA_ROOT / "old_train_state_selection.csv")
        constants = load_json(DATA_ROOT / "training_constants.json")
        all_records.extend(build_old_source(
            selection,
            policy,
            constants,
            data_freeze_hash=data_freeze_hash,
            checkpoint_hash=checkpoint_hash,
        ))
        write_progress(all_records, sum(expected_by_source[s] for s in args.sources), started)

    manifest = pd.DataFrame(all_records).sort_values(["source", "instance_id"])
    atomic_csv(manifest, OUT / "embedding_cache_manifest.csv")
    expected_shards = sum(expected_by_source[source] for source in args.sources)
    expected_states = sum({
        "R09": 1620,
        "CONTINUATION": 27,
        "OLD_TRAIN": 540,
    }[source] for source in args.sources)
    checks = {
        "expected_shards": len(manifest) == expected_shards,
        "expected_states": int(manifest.state_count.sum()) == expected_states,
        "embedding_dim_128": bool(manifest.embedding_dim.eq(128).all()),
        "tensor_schema_singleton": manifest.tensor_schema_hash.nunique() == 1,
        "checkpoint_hash_exact": set(manifest.checkpoint_sha256) == {checkpoint_hash},
        "live_score_replay_tolerance": bool(
            manifest[manifest.score_max_abs_error.notna()]
            .score_max_abs_error.le(0.01).all()
        ),
        "r10_not_accessed": True,
        "r11_not_accessed": True,
    }
    integrity = {
        "schema": "phase6i-mr-embedding-cache-integrity-v1.2",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "training_data_freeze_sha256": data_freeze_hash,
        "checkpoint_sha256": checkpoint_hash,
        "tensor_schema_hash": policy.tensorizer.tensor_schema_hash,
        "completed_shards": len(manifest),
        "completed_states": int(manifest.state_count.sum()),
        "completed_actions": int(manifest.action_count.sum()),
        "checks": checks,
        "manifest_sha256": digest(OUT / "embedding_cache_manifest.csv"),
        "runtime_seconds": time.perf_counter() - started,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(integrity, OUT / "embedding_cache_integrity.json")
    write_progress(all_records, expected_shards, started)
    print(json.dumps(integrity, indent=2, sort_keys=True), flush=True)
    if integrity["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
