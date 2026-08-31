#!/usr/bin/env python3
"""Generate frozen Phase 6C labels for R06 without opening them for analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c_io import (
    atomic_write_csv,
    atomic_write_json,
    remove_partial_files,
    sha256_file,
)
from scripts.run_phase6c_dataset import run_shard, valid_status


INSTANCE_ROOT = ROOT / "instances" / "controlled" / "RCIAS-CB1-TRAIN-R06"
INSTANCE_MANIFEST = INSTANCE_ROOT / "manifests" / "revision_instance_manifest.csv"
REVISION = ROOT / "outputs" / "phase6f" / "revision_holdout"
STATE_MANIFEST = REVISION / "state_manifest.csv"
STATE_FREEZE = REVISION / "state_freeze.json"
SEALED_ROOT = REVISION / "sealed_labels"
SHARD_MANIFEST = REVISION / "sealed_label_shard_manifest.csv"
LABEL_FREEZE = REVISION / "sealed_label_freeze.json"
SMOKE_ROOT = REVISION / "smoke_labels"
CONFIG_PATH = ROOT / "configs" / "phase6f_revision.json"
DATA_FILES = (
    "states.parquet",
    "repair_seed_outcomes.parquet",
    "target_set_aggregates.parquet",
    "target_membership.parquet",
    "operation_pairs.parquet",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def phase6c_protocol(config: dict, state_freeze: dict) -> dict:
    revision = config["revision_holdout"]
    return {
        "label_generation_version": revision["label_generation_version"],
        "destroy_fraction": revision["destroy_fraction"],
        "repair_operator": revision["repair_operator"],
        "repair_seed_count": revision["repair_seed_count"],
        "candidate_trials": revision["candidate_trials"],
        "seed_namespaces": {
            "arm_generation": config["seed_namespaces"]["arm_generation"],
            "repair": config["seed_namespaces"]["repair"],
        },
        "production_freeze_hash": state_freeze["freeze_hash"],
    }


def status_path(output_root: Path, instance_id: str) -> Path:
    return output_root / "revision_holdout" / instance_id / "status.json"


def custom_valid_status(path: Path, state_freeze_hash: str) -> dict | None:
    status = valid_status(path)
    if status is None or status.get("production_freeze_hash") != state_freeze_hash:
        return None
    return status


def write_manifest(output_root: Path, destination: Path, state_freeze_hash: str) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.glob("*/*/status.json")):
        status = custom_valid_status(path, state_freeze_hash)
        if status is not None:
            rows.append({key: value for key, value in status.items() if key != "file_sha256"})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values("shard_id").reset_index(drop=True)
    atomic_write_csv(frame, destination)
    return frame


def completed_state_count(output_root: Path, state_freeze_hash: str) -> int:
    """Count resumable progress without re-hashing every completed shard."""
    total = 0
    for path in output_root.glob("*/*/status.json"):
        status = load_json(path)
        if (
            status.get("status") == "COMPLETE"
            and status.get("production_freeze_hash") == state_freeze_hash
        ):
            total += int(status.get("state_count", 0))
    return total


def freeze_sealed_labels(state_freeze: dict) -> dict:
    frame = write_manifest(SEALED_ROOT, SHARD_MANIFEST, state_freeze["freeze_hash"])
    states = pd.read_csv(STATE_MANIFEST)
    expected_instances = set(states.instance_id.unique())
    complete_instances = set(frame.shard_id) if len(frame) else set()
    checks = {
        "exact_instance_coverage": complete_instances == expected_instances,
        "exact_state_count": len(frame) > 0 and int(frame.state_count.sum()) == len(states) == 8100,
        "three_seed_rows_exact": len(frame) > 0
        and bool((frame.repair_seed_row_count == 3 * frame.arm_count).all()),
        "state_freeze_hash_bound": len(frame) > 0
        and all(
            load_json(path).get("production_freeze_hash") == state_freeze["freeze_hash"]
            for path in SEALED_ROOT.glob("*/*/status.json")
        ),
        "all_status_files_complete": len(frame) == 81,
        "model_freeze_not_required_for_sealed_generation": not state_freeze["labels_opened"],
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    collection = {
        row.shard_id: row.shard_sha256
        for row in frame.sort_values("shard_id").itertuples(index=False)
    }
    payload = {
        "schema": "phase6f-r06-sealed-label-freeze-v1",
        "status": "SEALED_COMPLETE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_freeze_hash": state_freeze["freeze_hash"],
        "state_manifest_sha256": sha256_file(STATE_MANIFEST),
        "instance_manifest_sha256": sha256_file(INSTANCE_MANIFEST),
        "shard_manifest_sha256": sha256_file(SHARD_MANIFEST),
        "label_shard_collection_sha256": canonical_hash(collection),
        "shard_count": len(frame),
        "state_count": int(frame.state_count.sum()),
        "arm_count": int(frame.arm_count.sum()),
        "repair_seed_row_count": int(frame.repair_seed_row_count.sum()),
        "label_files_opened_for_analysis": False,
        "model_config_frozen": False,
        "checks": checks,
    }
    payload["freeze_hash"] = canonical_hash(payload)
    atomic_write_json(payload, LABEL_FREEZE)
    return payload


def run_smoke(
    states: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: dict,
    state_freeze_hash: str,
    smoke_states: int,
) -> None:
    first_instance = states.sort_values(["scale", "instance_id", "state_id"]).instance_id.iloc[0]
    part = states[states.instance_id == first_instance].sort_values("state_id").head(smoke_states)
    record = manifest.set_index("instance_id").loc[first_instance].to_dict() | {
        "instance_id": first_instance
    }
    status = run_shard((record, part, SMOKE_ROOT, protocol, INSTANCE_ROOT))
    verified = custom_valid_status(status_path(SMOKE_ROOT, first_instance), state_freeze_hash)
    if verified is None or verified["state_count"] != smoke_states:
        raise RuntimeError("R06 sealed-label smoke validation failed")
    atomic_write_json(
        {
            "schema": "phase6f-r06-label-smoke-v1",
            "status": "PASS",
            "instance_id": first_instance,
            "state_count": smoke_states,
            "arm_count": status["arm_count"],
            "repair_seed_row_count": status["repair_seed_row_count"],
            "runtime_seconds": status["runtime_seconds"],
            "states_per_second": smoke_states / max(status["runtime_seconds"], 1e-9),
            "phase6c_protocol_reused": True,
        },
        REVISION / "label_smoke_audit.json",
    )
    print(
        f"PHASE6F_R06_LABEL_SMOKE_PASS states={smoke_states} "
        f"runtime={status['runtime_seconds']:.2f}s"
    )


def verify(state_freeze: dict) -> None:
    payload = freeze_sealed_labels(state_freeze)
    if payload["label_files_opened_for_analysis"]:
        raise RuntimeError("sealed R06 labels were marked as opened")
    print(
        f"PHASE6F_R06_SEALED_LABELS_VERIFIED shards={payload['shard_count']} "
        f"states={payload['state_count']} arms={payload['arm_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(24, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--shards", nargs="*")
    parser.add_argument("--smoke-states", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not STATE_FREEZE.exists():
        raise RuntimeError("R06 state identities must be frozen before label generation")
    config = load_json(CONFIG_PATH)
    state_freeze = load_json(STATE_FREEZE)
    if state_freeze["labels_opened"] or state_freeze["model_config_frozen"]:
        raise RuntimeError("unexpected R06 state-freeze access state before label generation")
    protocol = phase6c_protocol(config, state_freeze)
    states = pd.read_csv(STATE_MANIFEST)
    manifest = pd.read_csv(INSTANCE_MANIFEST)
    if args.verify_only:
        verify(state_freeze)
        return
    if args.smoke_states:
        if args.smoke_states < 1 or args.smoke_states > 10:
            raise ValueError("--smoke-states must be between 1 and 10")
        run_smoke(
            states,
            manifest,
            protocol,
            state_freeze["freeze_hash"],
            args.smoke_states,
        )
        return

    remove_partial_files(SEALED_ROOT)
    requested = set(args.shards or [])
    records = manifest.set_index("instance_id")
    tasks = []
    for instance_id, part in states.groupby("instance_id"):
        if requested and instance_id not in requested:
            continue
        path = status_path(SEALED_ROOT, instance_id)
        if custom_valid_status(path, state_freeze["freeze_hash"]) is None:
            record = records.loc[instance_id].to_dict() | {"instance_id": instance_id}
            tasks.append((record, part, SEALED_ROOT, protocol, INSTANCE_ROOT))
    print(
        f"PHASE6F_R06_SEALED_LABEL_START pending={len(tasks)} workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_shard, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            status = future.result()
            elapsed = time.perf_counter() - started
            completed_states = completed_state_count(
                SEALED_ROOT, state_freeze["freeze_hash"]
            )
            print(json.dumps({
                "event": "r06_label_shard",
                "completed_shards_this_run": index,
                "submitted_shards": len(futures),
                "completed_states_total": completed_states,
                "target_states": 8100,
                "shard_id": status["shard_id"],
                "runtime_seconds": status["runtime_seconds"],
                "states_per_wall_second": completed_states / max(elapsed, 1e-9),
            }), flush=True)
    write_manifest(SEALED_ROOT, SHARD_MANIFEST, state_freeze["freeze_hash"])
    if not requested:
        payload = freeze_sealed_labels(state_freeze)
        print(
            f"PHASE6F_R06_SEALED_LABEL_COMPLETE shards={payload['shard_count']} "
            f"states={payload['state_count']} arms={payload['arm_count']}"
        )


if __name__ == "__main__":
    main()
