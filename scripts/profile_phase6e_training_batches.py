#!/usr/bin/env python3
"""Profile full forward/backward Phase 6E state batches on S/M/L caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.losses import NILossConfig, phase6e_loss  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


CANDIDATE_BATCHES = {
    "S": (32, 64, 96),
    "M": (16, 32, 48),
    "L": (8, 16, 24),
}


def one_shard(manifest: pd.DataFrame, scale: str) -> pd.Series:
    matches = manifest[
        manifest["training_split"].eq("TRAIN")
        & manifest["instance_id"].str.contains(f"_{scale}_")
    ]
    if matches.empty:
        raise ValueError(f"missing TRAIN cache for scale {scale}")
    return matches.sort_values("instance_id").iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/profiling"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training batch profile requires a visible CUDA GPU")
    manifest = pd.read_csv(args.manifest)
    tensorizer = CSGTensorizer()
    model_config = NIModelConfig(
        hidden_dim=128, layers=3, heads=4, dropout=0.1,
        use_edge_features=True, relation_mode="FULL_CSG", message_passing=True,
    )
    rows = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for scale in ("S", "M", "L"):
        record = one_shard(manifest, scale)
        samples, _ = load_shard_cache(Path(str(record["cache_path"])))
        for state_count in CANDIDATE_BATCHES[scale]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(660120)
            model = CSGTargetSetScorer(tensorizer, model_config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            batch_started = time.perf_counter()
            batch = batch_state_samples(samples[:state_count]).to(device)
            batch_seconds = time.perf_counter() - batch_started
            durations = []
            status = "PASS"
            error = ""
            try:
                for iteration in range(args.repetitions + 1):
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(batch)
                        losses = phase6e_loss(output.scores, batch, NILossConfig())
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    if iteration:
                        durations.append(time.perf_counter() - started)
            except torch.cuda.OutOfMemoryError as exc:
                status = "OOM"
                error = str(exc)
            median_seconds = float(pd.Series(durations).median()) if durations else None
            peak = torch.cuda.max_memory_allocated(device)
            rows.append({
                "scale": scale,
                "batch_states": state_count,
                "action_count": batch.action_count,
                "node_count": sum(value.shape[0] for value in batch.node_features.values()),
                "directed_edge_count": sum(edge.index.shape[1] for edge in batch.edges.values()),
                "batch_build_transfer_seconds": batch_seconds,
                "median_train_step_seconds": median_seconds,
                "states_per_second": state_count / median_seconds if median_seconds else None,
                "actions_per_second": batch.action_count / median_seconds if median_seconds else None,
                "gpu_peak_memory_mib": peak / (1024**2),
                "gpu_memory_fraction": peak / total_memory,
                "status": status,
                "error": error,
            })
            del batch, optimizer, model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    recommendations = {}
    for scale, group in frame[frame["status"].eq("PASS")].groupby("scale"):
        safe = group[group["gpu_memory_fraction"] <= 0.60]
        if safe.empty:
            safe = group.sort_values("gpu_memory_fraction").head(1)
        recommendations[scale] = int(
            safe.sort_values(["states_per_second", "batch_states"], ascending=False)
            .iloc[0]["batch_states"]
        )
    result = {
        "schema": "phase6e-training-batch-profile-v1",
        "status": "PASS" if len(recommendations) == 3 else "FAIL",
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_mib": total_memory / (1024**2),
        "profile_model": {
            "hidden_dim": 128, "layers": 3, "heads": 4, "mixed_precision": True
        },
        "safety_limit_gpu_memory_fraction": 0.60,
        "recommended_batch_states": recommendations,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "training_batch_profile.csv", index=False)
    (args.output_root / "training_batch_recommendation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(frame.to_json(orient="records", indent=2))
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
