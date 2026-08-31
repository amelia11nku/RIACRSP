#!/usr/bin/env python3
"""Profile frozen Phase 6F model-decision and end-to-end R06 latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg  # noqa: E402
from rcias_clgri.data.phase6c import reconstruct_state  # noqa: E402
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.calibration import FrozenCalibrator  # noqa: E402
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


def cuda_timed(function, device: torch.device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def percentile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile, interpolation="linear"))


def selected_r06_shard(dataset_root: Path, scale: str) -> str:
    paths = sorted(dataset_root.glob(f"revision_holdout/*_{scale}_*/states.parquet"))
    if not paths:
        raise ValueError(f"no R06 shard for scale {scale}")
    return paths[0].parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=ROOT / "outputs/phase6f/revision_holdout/sealed_labels",
    )
    parser.add_argument(
        "--train-root", type=Path,
        default=ROOT / "instances/controlled/RCIAS-CB1-TRAIN-R06",
    )
    parser.add_argument(
        "--experiment-freeze", type=Path,
        default=ROOT / "outputs/phase6f/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/phase6f/profiling",
    )
    args = parser.parse_args()
    freeze = json.loads(args.experiment_freeze.read_text(encoding="utf-8"))
    protocol = freeze["latency_protocol"]
    states_per_scale = int(protocol["states_per_scale"])
    warmups = int(protocol["warmup_repetitions"])
    repetitions = int(protocol["timed_repetitions"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 6F inference profiling requires CUDA")

    checkpoint = torch.load(
        Path(str(freeze["selected_checkpoint_path"])),
        map_location="cpu",
        weights_only=False,
    )
    tensorizer = CSGTensorizer()
    model = CSGTargetSetScorer(
        tensorizer, NIModelConfig(**checkpoint["model_config"])
    )
    if checkpoint["tensor_schema_hash"] != model.tensor_schema_hash:
        raise ValueError("checkpoint/tensor schema mismatch")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    del checkpoint
    if model.utility_head is None:
        raise ValueError("frozen Phase 6F checkpoint has no utility head")
    probability = FrozenCalibrator(**freeze["probability_calibrator"])
    utility = FrozenCalibrator(**freeze["utility_calibrator"])
    thresholds = freeze["selective_intervention_thresholds"]
    process = psutil.Process()
    device_total_memory = torch.cuda.get_device_properties(device).total_memory
    detail_rows: list[dict[str, object]] = []

    for scale in ("S", "M", "L"):
        instance_id = selected_r06_shard(args.dataset_root, scale)
        states, actions = load_shard_frames(
            args.dataset_root, instance_id, "REVISION_HOLDOUT"
        )
        action_groups = {
            state_id: part for state_id, part in actions.groupby("state_id", sort=False)
        }
        for state_record in states.head(states_per_scale).to_dict("records"):
            state_id = str(state_record["state_id"])
            action_records = action_groups[state_id].to_dict("records")
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
            batch_cpu = batch_state_samples([sample])
            batch, transfer_ms = cuda_timed(lambda: batch_cpu.to(device), device)

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                for _ in range(warmups):
                    model(batch)
                encoder_times = []
                scoring_times = []
                calibration_times = []
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(repetitions):
                    encoded, encoder_ms = cuda_timed(
                        lambda: model.state_encoder(batch), device
                    )
                    node_embeddings, graph_embeddings = encoded

                    def score_actions():
                        action_embeddings = model.action_encoder(
                            node_embeddings["OP"], graph_embeddings, batch
                        )
                        return (
                            model.score_head(action_embeddings).squeeze(-1),
                            model.utility_head(action_embeddings).squeeze(-1),
                        )

                    (score, predicted_utility), scoring_ms = cuda_timed(
                        score_actions, device
                    )

                    def calibrate_and_gate():
                        raw_score = score.detach().float().cpu().numpy()
                        raw_utility = predicted_utility.detach().float().cpu().numpy()
                        calibrated_probability = probability.predict(raw_score)
                        calibrated_utility = utility.predict(raw_utility)
                        order = np.argsort(-raw_score, kind="stable")
                        best = int(order[0])
                        second_probability = (
                            float(calibrated_probability[int(order[1])])
                            if len(order) > 1 else 0.0
                        )
                        margin = float(calibrated_probability[best]) - second_probability
                        return bool(
                            calibrated_probability[best] >= thresholds["confidence"]
                            and calibrated_utility[best] >= thresholds["predicted_utility"]
                            and margin >= thresholds["decision_margin"]
                        )

                    _, calibration_ms = timed(calibrate_and_gate)
                    encoder_times.append(encoder_ms)
                    scoring_times.append(scoring_ms)
                    calibration_times.append(calibration_ms)

            encode_ms = statistics.median(encoder_times)
            score_ms = statistics.median(scoring_times)
            calibration_ms = statistics.median(calibration_times)
            model_decision_ms = transfer_ms + encode_ms + score_ms + calibration_ms
            preprocessing_ms = (
                reconstruction_ms + csg_build_ms + tensorization_ms
                + action_projection_ms
            )
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
                "calibration_gating_ms": calibration_ms,
                "model_decision_ms": model_decision_ms,
                "preprocessing_ms": preprocessing_ms,
                "end_to_end_decision_ms": preprocessing_ms + model_decision_ms,
                "gpu_peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                "cpu_process_rss_mib": process.memory_info().rss / (1024**2),
                "batch_tensor_mib": batch_cpu.tensor_bytes() / (1024**2),
            })
            del batch, batch_cpu, sample, tensor_graph, action_set, graph, reconstructed
            torch.cuda.empty_cache()
            print(json.dumps({
                "event": "phase6f_profile_state",
                "scale": scale,
                "completed_states": sum(row["scale"] == scale for row in detail_rows),
                "total_states": states_per_scale,
                "state_id": state_id,
                "model_decision_ms": model_decision_ms,
            }), flush=True)

    detail = pd.DataFrame(detail_rows)
    timing_columns = [
        "reconstruction_ms", "csg_build_ms", "tensorization_ms",
        "action_projection_ms", "gpu_transfer_ms", "gpu_graph_encoding_ms",
        "gpu_action_scoring_ms", "calibration_gating_ms", "model_decision_ms",
        "preprocessing_ms", "end_to_end_decision_ms",
    ]
    summary_rows = []
    for scale, group in detail.groupby("scale", sort=False):
        row: dict[str, object] = {
            "scale": scale,
            "profile_state_count": len(group),
            "median_candidate_action_count": float(group["candidate_action_count"].median()),
        }
        for column in timing_columns:
            row[f"median_{column}"] = float(group[column].median())
            row[f"p90_{column}"] = percentile(group[column], 0.9)
        row["gpu_peak_memory_mib"] = float(group["gpu_peak_memory_mib"].max())
        row["gpu_memory_fraction"] = row["gpu_peak_memory_mib"] * 1024**2 / device_total_memory
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    hard_ms = float(protocol["p90_hard_ms_each_scale"])
    checks = {
        f"p90_model_decision_{row.scale}_within_{hard_ms:g}ms": bool(
            row.p90_model_decision_ms <= hard_ms
        )
        for row in summary.itertuples(index=False)
    }
    result = {
        "schema": "phase6f-r06-inference-profile-v1",
        "status": "COMPLETE",
        "latency_gate_passed": all(checks.values()),
        "checks": checks,
        "profile_split": "REVISION_HOLDOUT",
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_mib": device_total_memory / (1024**2),
        "states_per_scale": states_per_scale,
        "warmup_repetitions": warmups,
        "timed_repetitions": repetitions,
        "model_decision_p90_ms": {
            str(row.scale): float(row.p90_model_decision_ms)
            for row in summary.itertuples(index=False)
        },
        "end_to_end_p90_ms": {
            str(row.scale): float(row.p90_end_to_end_decision_ms)
            for row in summary.itertuples(index=False)
        },
        "hard_gate_ms": hard_ms,
        "one_state_one_encoding_all_action_scoring": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_root / "latency_profile_detail.csv", index=False)
    summary.to_csv(args.output_root / "latency_profile.csv", index=False)
    (args.output_root / "latency_profile_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_json(orient="records"), flush=True)
    print(json.dumps({"event": "phase6f_inference_profile_complete", **result}), flush=True)


if __name__ == "__main__":
    main()
