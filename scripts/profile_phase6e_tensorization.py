#!/usr/bin/env python3
"""Profile Phase 6E graph construction/cache strategies on frozen TRAIN data."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import statistics
import sys
import time

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg  # noqa: E402
from rcias_clgri.data.phase6c import reconstruct_state  # noqa: E402
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.dataset import (  # noqa: E402
    NIStateSample,
    load_shard_frames,
    tensorize_action_records,
)
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return ordered[index]


def select_shards(manifest: pd.DataFrame) -> dict[str, pd.Series]:
    train = manifest[(manifest["split"] == "TRAIN") & (manifest["status"] == "COMPLETE")]
    selected = {}
    for scale in ("S", "M", "L"):
        matches = train[train["shard_id"].str.contains(f"_{scale}_")]
        if matches.empty:
            raise RuntimeError(f"no TRAIN shard found for scale {scale}")
        selected[scale] = matches.sort_values("shard_id").iloc[0]
    return selected


def profile_gpu_transfer(samples: list[NIStateSample], scale: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "scale": scale,
            "gpu_available": False,
            "gpu_transfer_batch_states": 0,
            "gpu_transfer_mib_per_second": None,
            "gpu_transfer_milliseconds": None,
        }
    copies = max(4, min(32, 8 * 1024 * 1024 // max(samples[0].graph.tensor_bytes(), 1)))
    repeated = [samples[index % len(samples)] for index in range(copies)]
    batch = batch_state_samples(repeated)
    for _ in range(2):
        batch.to("cuda")
    torch.cuda.synchronize()
    durations = []
    for _ in range(10):
        started = time.perf_counter()
        moved = batch.to("cuda")
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        del moved
    seconds = statistics.median(durations)
    mib = batch.tensor_bytes() / (1024**2)
    return {
        "scale": scale,
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_transfer_batch_states": copies,
        "gpu_transfer_batch_mib": mib,
        "gpu_transfer_mib_per_second": mib / seconds,
        "gpu_transfer_milliseconds": seconds * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-per-scale", type=int, default=3)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "outputs/phase6c/dataset")
    parser.add_argument(
        "--train-root", type=Path, default=ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/tensorization"
    )
    args = parser.parse_args()
    if args.states_per_scale < 1:
        raise ValueError("--states-per-scale must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "outputs/phase6c/manifests/shard_manifest.csv"
    freeze_path = ROOT / "outputs/phase6c/audit/dataset_freeze_record.json"
    manifest = pd.read_csv(manifest_path)
    tensorizer = CSGTensorizer(include_reverse=True)
    rows: list[dict[str, object]] = []
    scale_samples: dict[str, list[NIStateSample]] = {}

    for scale, shard in select_shards(manifest).items():
        states, actions = load_shard_frames(
            args.dataset_root, str(shard["shard_id"]), str(shard["split"])
        )
        action_groups = {
            state_id: frame for state_id, frame in actions.groupby("state_id", sort=False)
        }
        samples = []
        for state_record in states.head(args.states_per_scale).to_dict("records"):
            action_records = action_groups[str(state_record["state_id"])].to_dict("records")
            reconstructed, reconstruct_seconds = timed(
                lambda: reconstruct_state(state_record, args.train_root)
            )
            graph, graph_build_seconds = timed(lambda: build_csg(reconstructed, state_record))
            tensor_graph, tensorize_seconds = timed(lambda: tensorizer.tensorize(graph))
            action_set, action_seconds = timed(
                lambda: tensorize_action_records(graph, action_records)
            )
            sample = NIStateSample(
                tensor_graph,
                action_set,
                {
                    key: str(state_record[key])
                    for key in (
                        "training_split", "scale", "CF_level", "RI_level", "TI_level",
                        "search_stage", "bottleneck_proxy",
                    )
                },
            )
            samples.append(sample)

            strategy_a = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            strategy_b = pickle.dumps(
                (graph, action_records, sample.structural_metadata),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            source_payload = pickle.dumps(
                (state_record, action_records), protocol=pickle.HIGHEST_PROTOCOL
            )
            _, strategy_a_load = timed(lambda: pickle.loads(strategy_a))

            def load_b():
                loaded_graph, loaded_actions, metadata = pickle.loads(strategy_b)
                return NIStateSample(
                    tensorizer.tensorize(loaded_graph),
                    tensorize_action_records(loaded_graph, loaded_actions),
                    metadata,
                )

            _, strategy_b_load = timed(load_b)
            rows.append({
                "scale": scale,
                "instance_id": graph.instance_id,
                "state_id": graph.state_id,
                "node_count": graph.node_count,
                "canonical_edge_count": graph.edge_count,
                "action_count": action_set.action_count,
                "tensor_bytes": tensor_graph.tensor_bytes(),
                "reconstruction_seconds": reconstruct_seconds,
                "csg_build_seconds": graph_build_seconds,
                "tensorization_seconds": tensorize_seconds,
                "action_projection_seconds": action_seconds,
                "strategy_a_serialized_bytes": len(strategy_a),
                "strategy_a_load_seconds": strategy_a_load,
                "strategy_b_serialized_bytes": len(strategy_b),
                "strategy_b_load_tensorize_seconds": strategy_b_load,
                "strategy_c_existing_source_bytes": len(source_payload),
                "strategy_c_rebuild_tensorize_seconds": (
                    reconstruct_seconds + graph_build_seconds + tensorize_seconds + action_seconds
                ),
            })
        scale_samples[scale] = samples

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_root / "cache_strategy_profile.csv", index=False)
    gpu_profiles = [profile_gpu_transfer(scale_samples[scale], scale) for scale in ("S", "M", "L")]
    pd.DataFrame(gpu_profiles).to_csv(args.output_root / "gpu_transfer_profile.csv", index=False)

    total_states = int(manifest[manifest["status"] == "COMPLETE"]["state_count"].sum())
    scale_weights = (
        manifest[manifest["status"] == "COMPLETE"]
        .assign(scale=lambda frame: frame["shard_id"].str.extract(r"_(S|M|L)_"))
        .groupby("scale")["state_count"].sum()
    )
    weighted_a = sum(
        detail[detail["scale"] == scale]["strategy_a_serialized_bytes"].mean()
        * int(scale_weights[scale])
        for scale in ("S", "M", "L")
    ) / int(scale_weights.sum())
    weighted_b = sum(
        detail[detail["scale"] == scale]["strategy_b_serialized_bytes"].mean()
        * int(scale_weights[scale])
        for scale in ("S", "M", "L")
    ) / int(scale_weights.sum())
    estimated_a_gib = weighted_a * total_states / (1024**3)
    estimated_b_gib = weighted_b * total_states / (1024**3)
    disk = shutil.disk_usage(ROOT)
    allowed_a_gib = disk.free / (1024**3) * 0.45
    choice = "A_PRETENSORIZED_SHARDED_CACHE" if estimated_a_gib <= allowed_a_gib else "C_RECONSTRUCT_BOUNDED_LRU"
    reasons = (
        [
            "Estimated pre-tensorized cache fits within 45% of currently free workspace disk.",
            "Strategy A removes repeated deterministic reconstruction and CSG construction from every epoch.",
            "Cache remains sharded by frozen Phase 6C instance and stores all actions per state.",
        ]
        if choice.startswith("A_")
        else [
            "Estimated pre-tensorized cache would consume more than 45% of currently free workspace disk.",
            "Use deterministic reconstruction with a bounded shard LRU; no holdout result informed this choice.",
        ]
    )
    decision = {
        "schema": "phase6e-cache-strategy-decision-v1",
        "decision_frozen_before_internal_holdout": True,
        "selected_strategy": choice,
        "profile_train_only": True,
        "profile_states_per_scale": args.states_per_scale,
        "profile_state_count": len(detail),
        "phase6c_manifest_sha256": sha256(manifest_path),
        "phase6c_freeze_sha256": sha256(freeze_path),
        "tensor_schema_hash": tensorizer.tensor_schema_hash,
        "frozen_total_state_count": total_states,
        "estimated_strategy_a_gib": estimated_a_gib,
        "estimated_strategy_b_gib": estimated_b_gib,
        "strategy_c_additional_cache_gib": 0.0,
        "free_disk_gib_at_decision": disk.free / (1024**3),
        "strategy_a_disk_limit_gib": allowed_a_gib,
        "median_rebuild_tensorize_seconds": statistics.median(
            detail["strategy_c_rebuild_tensorize_seconds"].tolist()
        ),
        "p90_rebuild_tensorize_seconds": percentile(
            detail["strategy_c_rebuild_tensorize_seconds"].tolist(), 0.9
        ),
        "reasons": reasons,
        "holdout_metrics_observed": False,
        "gpu_profiles": gpu_profiles,
    }
    (args.output_root / "cache_strategy_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "tensor_schema.json").write_text(
        json.dumps(tensorizer.schema_record(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
