#!/usr/bin/env python3
"""Build/resume the frozen Strategy-A Phase 6E pre-tensorized cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.cache import (  # noqa: E402
    load_shard_cache,
    valid_cache_record,
    write_shard_cache,
)
from rcias_clgri.ni.dataset import iter_shard_samples, load_shard_frames  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


SPLIT_CACHE_DIRECTORIES = {
    "TRAIN": "train",
    "TRAIN_VALIDATION": "validation",
    "TRAIN_INTERNAL_HOLDOUT": "internal_holdout",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_paths(output_root: Path, instance_id: str, split: str) -> tuple[Path, Path]:
    directory = output_root / SPLIT_CACHE_DIRECTORIES[split]
    return directory / f"{instance_id}.pt", directory / f"{instance_id}.manifest.json"


def build_one(payload: tuple[dict[str, object], str, str, str]) -> dict[str, object]:
    shard, dataset_root_value, train_root_value, output_root_value = payload
    started = time.perf_counter()
    instance_id = str(shard["shard_id"])
    split = str(shard["split"])
    source_hash = str(shard["shard_sha256"])
    dataset_root = Path(dataset_root_value)
    train_root = Path(train_root_value)
    output_root = Path(output_root_value)
    cache_path, record_path = cache_paths(output_root, instance_id, split)
    tensorizer = CSGTensorizer(include_reverse=True)
    existing = valid_cache_record(
        record_path,
        source_shard_sha256=source_hash,
        tensor_schema_hash=tensorizer.tensor_schema_hash,
    )
    if existing is not None:
        return {**existing, "runtime_seconds": 0.0, "result": "SKIPPED_VALID"}
    try:
        states, actions = load_shard_frames(dataset_root, instance_id, split)
        samples = list(iter_shard_samples(
            states,
            actions,
            train_root=train_root,
            tensorizer=tensorizer,
        ))
        record = write_shard_cache(
            cache_path,
            samples,
            instance_id=instance_id,
            training_split=split,
            source_shard_sha256=source_hash,
        )
        loaded, metadata = load_shard_cache(
            cache_path,
            expected_tensor_schema_hash=tensorizer.tensor_schema_hash,
            expected_source_shard_sha256=source_hash,
        )
        if len(loaded) != len(states) or metadata["action_count"] != len(actions):
            raise ValueError("post-write cache coverage validation failed")
        runtime = time.perf_counter() - started
        record.update({
            "runtime_seconds": runtime,
            "states_per_second": len(samples) / runtime,
            "result": "BUILT",
        })
        write_json(record_path, record)
        return record
    except Exception as error:
        return {
            "schema": "phase6e-cache-failure-v1",
            "instance_id": instance_id,
            "training_split": split,
            "source_shard_sha256": source_hash,
            "status": "FAILED",
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--instance-id")
    parser.add_argument("--splits", nargs="*", choices=tuple(SPLIT_CACHE_DIRECTORIES))
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "outputs/phase6c/dataset"
    )
    parser.add_argument(
        "--train-root", type=Path, default=ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/tensorization/cache"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "outputs/phase6c/manifests/shard_manifest.csv"
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest = pd.read_csv(args.manifest)
    selected = manifest[manifest["status"] == "COMPLETE"]
    if args.splits:
        selected = selected[selected["split"].isin(args.splits)]
    if args.instance_id:
        selected = selected[selected["shard_id"] == args.instance_id]
    selected = selected.sort_values(["split", "shard_id"])
    if args.limit_shards is not None:
        selected = selected.head(args.limit_shards)
    if selected.empty:
        raise ValueError("no frozen Phase 6C shards selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    total_shards = len(selected)
    total_states = int(selected["state_count"].sum())
    jobs = [
        (row, str(args.dataset_root), str(args.train_root), str(args.output_root))
        for row in selected.to_dict("records")
    ]
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_one, job): str(job[0]["shard_id"]) for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            results.append(record)
            elapsed = time.perf_counter() - started
            completed_states = sum(
                int(item.get("state_count", 0))
                for item in results if item.get("status") == "COMPLETE"
            )
            rate = completed_states / elapsed if elapsed else 0.0
            eta = (total_states - completed_states) / rate if rate else None
            event = {
                "event": "shard_complete",
                "completed_shards": completed,
                "total_shards": total_shards,
                "completed_states": completed_states,
                "total_states": total_states,
                "states_per_second": rate,
                "eta_seconds": eta,
                **record,
            }
            print(json.dumps(event), flush=True)
            if record.get("status") != "COMPLETE":
                for pending in futures:
                    pending.cancel()
                break

    frame = pd.DataFrame(results).sort_values(["training_split", "instance_id"])
    global_records = []
    for record_path in sorted(args.output_root.glob("*/*.manifest.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "COMPLETE":
            global_records.append(record)
    pd.DataFrame(global_records).sort_values(
        ["training_split", "instance_id"]
    ).to_csv(args.output_root / "cache_manifest.csv", index=False)
    failed = frame[frame["status"] != "COMPLETE"]
    summary = {
        "schema": "phase6e-cache-build-summary-v1",
        "selected_strategy": "A_PRETENSORIZED_SHARDED_CACHE",
        "requested_shards": total_shards,
        "complete_shards": int((frame["status"] == "COMPLETE").sum()),
        "failed_shards": len(failed),
        "requested_states": total_states,
        "complete_states": int(frame.loc[frame["status"] == "COMPLETE", "state_count"].sum()),
        "cache_bytes": int(frame.loc[frame["status"] == "COMPLETE", "cache_bytes"].sum()),
        "runtime_seconds": time.perf_counter() - started,
        "tensor_schema_hash": CSGTensorizer().tensor_schema_hash,
        "status": "COMPLETE" if failed.empty and len(frame) == total_shards else "FAILED",
    }
    write_json(args.output_root / "cache_build_summary.json", summary)
    print(json.dumps({"event": "build_complete", **summary}), flush=True)
    if summary["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
