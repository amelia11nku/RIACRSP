#!/usr/bin/env python3
"""Profile Phase 6E shared-state inference latency on validation S/M/L states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import pandas as pd
import psutil
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
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - started) * 1000.0


def percentile(values: list[float], quantile: float) -> float:
    return float(pd.Series(values).quantile(quantile, interpolation="linear"))


def selected_validation_shard(dataset_root: Path, scale: str) -> tuple[str, str]:
    paths = sorted(dataset_root.glob(f"validation/*_{scale}_*/states.parquet"))
    if not paths:
        raise ValueError(f"no validation shard for scale {scale}")
    return paths[0].parent.name, "TRAIN_VALIDATION"


def cuda_timed(function, device: torch.device) -> tuple[object, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs/phase6e_inference_profile.json"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "outputs/phase6c/dataset"
    )
    parser.add_argument(
        "--train-root", type=Path,
        default=ROOT / "instances/controlled/RCIAS-CB1-TRAIN",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "outputs/phase6e/training/final_seeds/seed_660201/checkpoint_best.pt",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/profiling"
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    states_per_scale = int(plan["states_per_scale"])
    repetitions = int(plan["gpu_repetitions_per_state"])
    warmups = int(plan["warmup_repetitions"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 6E inference profiling requires a visible CUDA GPU")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensorizer = CSGTensorizer()
    model = CSGTargetSetScorer(
        tensorizer, NIModelConfig(**checkpoint["model_config"])
    )
    if checkpoint["tensor_schema_hash"] != model.tensor_schema_hash:
        raise ValueError("checkpoint/tensor schema mismatch")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    del checkpoint
    process = psutil.Process()
    device_total_memory = torch.cuda.get_device_properties(device).total_memory
    detail_rows: list[dict[str, object]] = []

    for scale in ("S", "M", "L"):
        instance_id, split = selected_validation_shard(args.dataset_root, scale)
        states, actions = load_shard_frames(args.dataset_root, instance_id, split)
        action_groups = {
            state_id: part for state_id, part in actions.groupby("state_id", sort=False)
        }
        for state_record in states.head(states_per_scale).to_dict("records"):
            state_id = str(state_record["state_id"])
            action_records = action_groups[state_id].to_dict("records")
            rss_before = process.memory_info().rss
            tracemalloc.start()
            reconstructed, reconstruction_ms = timed(
                lambda: reconstruct_state(state_record, args.train_root)
            )
            graph, csg_build_ms = timed(lambda: build_csg(reconstructed, state_record))
            tensor_graph, tensorization_ms = timed(lambda: tensorizer.tensorize(graph))
            action_set, action_projection_ms = timed(
                lambda: tensorize_action_records(graph, action_records)
            )
            sample = NIStateSample(
                tensor_graph,
                action_set,
                {
                    name: str(state_record[name])
                    for name in (
                        "training_split", "scale", "CF_level", "RI_level",
                        "TI_level", "search_stage", "bottleneck_proxy",
                    )
                },
            )
            cpu_python_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            batch_cpu = batch_state_samples([sample])
            batch, transfer_ms = cuda_timed(lambda: batch_cpu.to(device), device)
            rss_after = process.memory_info().rss

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                for _ in range(warmups):
                    model(batch)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                encoder_times = []
                action_times = []
                for _ in range(repetitions):
                    encoded, encoder_ms = cuda_timed(
                        lambda: model.state_encoder(batch), device
                    )
                    node_embeddings, graph_embeddings = encoded

                    def score_actions():
                        action_embeddings = model.action_encoder(
                            node_embeddings["OP"], graph_embeddings, batch
                        )
                        return model.score_head(action_embeddings).squeeze(-1)

                    _, action_ms = cuda_timed(score_actions, device)
                    encoder_times.append(encoder_ms)
                    action_times.append(action_ms)
            encode_ms = statistics.median(encoder_times)
            score_ms = statistics.median(action_times)
            gpu_total_ms = encode_ms + score_ms
            cpu_pipeline_ms = (
                reconstruction_ms + csg_build_ms + tensorization_ms
                + action_projection_ms + transfer_ms
            )
            single_total_ms = cpu_pipeline_ms + gpu_total_ms
            ensemble_total_ms = cpu_pipeline_ms + 3.0 * gpu_total_ms
            naive_repeated_ms = cpu_pipeline_ms + action_set.action_count * encode_ms + score_ms
            detail_rows.append({
                "scale": scale,
                "instance_id": instance_id,
                "state_id": state_id,
                "node_count": graph.node_count,
                "canonical_edge_count": graph.edge_count,
                "candidate_action_count": action_set.action_count,
                "reconstruction_ms": reconstruction_ms,
                "csg_build_ms": csg_build_ms,
                "tensorization_ms": tensorization_ms,
                "action_projection_ms": action_projection_ms,
                "gpu_transfer_ms": transfer_ms,
                "gpu_graph_encoding_ms": encode_ms,
                "gpu_action_scoring_ms": score_ms,
                "gpu_shared_forward_ms": gpu_total_ms,
                "total_single_model_decision_ms": single_total_ms,
                "projected_ensemble_decision_ms": ensemble_total_ms,
                "projected_naive_repeated_encoding_ms": naive_repeated_ms,
                "shared_vs_naive_speedup": naive_repeated_ms / single_total_ms,
                "gpu_peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                "cpu_process_rss_mib": rss_after / (1024**2),
                "cpu_rss_delta_mib": max(rss_after - rss_before, 0) / (1024**2),
                "cpu_python_peak_mib": cpu_python_peak / (1024**2),
                "batch_tensor_mib": batch_cpu.tensor_bytes() / (1024**2),
            })
            del batch, batch_cpu, sample, tensor_graph, action_set, graph, reconstructed
            torch.cuda.empty_cache()
            print(json.dumps({
                "event": "profile_state_complete",
                "scale": scale,
                "completed_states": sum(row["scale"] == scale for row in detail_rows),
                "total_states": states_per_scale,
                "state_id": state_id,
                "single_model_ms": single_total_ms,
                "ensemble_ms": ensemble_total_ms,
            }), flush=True)

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for scale, group in detail.groupby("scale", sort=False):
        row: dict[str, object] = {
            "scale": scale,
            "profile_state_count": len(group),
            "median_candidate_action_count": float(group["candidate_action_count"].median()),
        }
        timing_columns = [
            "reconstruction_ms", "csg_build_ms", "tensorization_ms",
            "action_projection_ms", "gpu_transfer_ms", "gpu_graph_encoding_ms",
            "gpu_action_scoring_ms", "gpu_shared_forward_ms",
            "total_single_model_decision_ms", "projected_ensemble_decision_ms",
            "projected_naive_repeated_encoding_ms", "shared_vs_naive_speedup",
        ]
        for column in timing_columns:
            row[f"median_{column}"] = float(group[column].median())
            row[f"p90_{column}"] = percentile(group[column].tolist(), 0.9)
        row["gpu_peak_memory_mib"] = float(group["gpu_peak_memory_mib"].max())
        row["gpu_memory_fraction"] = row["gpu_peak_memory_mib"] * 1024**2 / device_total_memory
        row["cpu_peak_process_rss_mib"] = float(group["cpu_process_rss_mib"].max())
        row["cpu_peak_python_allocation_mib"] = float(group["cpu_python_peak_mib"].max())
        for count in plan["future_decision_counts"]:
            row[f"projected_{int(count)}_decision_overhead_seconds"] = (
                row["p90_projected_ensemble_decision_ms"] * int(count) / 1000.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    thresholds = plan["acceptance_thresholds"]
    checks = {
        "p90_ensemble_latency_within_budget": bool(
            summary["p90_projected_ensemble_decision_ms"].max()
            <= thresholds["maximum_p90_projected_ensemble_state_latency_ms"]
        ),
        "projected_1000_decisions_within_budget": bool(
            summary["projected_1000_decision_overhead_seconds"].max()
            <= thresholds["maximum_projected_1000_decision_overhead_seconds"]
        ),
        "gpu_memory_within_budget": bool(
            summary["gpu_memory_fraction"].max()
            <= thresholds["maximum_gpu_memory_fraction"]
        ),
    }
    result = {
        "schema": "phase6e-inference-profile-summary-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "profile_split": plan["profile_split"],
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_mib": device_total_memory / (1024**2),
        "states_per_scale": states_per_scale,
        "gpu_repetitions_per_state": repetitions,
        "checkpoint_seed": plan["checkpoint_seed"],
        "shared_state_encoding": True,
        "thresholds": thresholds,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_root / "inference_profile_detail.csv", index=False)
    summary.to_csv(args.output_root / "inference_profile.csv", index=False)
    (args.output_root / "inference_profile_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_json(orient="records"), flush=True)
    print(json.dumps({"event": "inference_profile_complete", **result}), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
