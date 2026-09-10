#!/usr/bin/env python3
"""Profile immutable historical C0 without modifying its output namespace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.train import load_cache, model_for, prepare

C0_OUT = ROOT / "outputs/ngas_a1/critic_training_gpu_v1"
CACHE = ROOT / "outputs/ngas_a1/critic_training_v1/training_cache.json.gz"
CACHE_MANIFEST = ROOT / "outputs/ngas_a1/critic_training_v1/training_cache_manifest.json"
OUTPUT = ROOT / "outputs/ngas_a1/critic_training_rthgt_v2/audit/c0_reference_gpu_profile.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(samples: list[float], quantile: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * quantile)]


def configure_cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C0 reference profiling requires exactly one CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    configure_cuda()
    protocol = json.loads((C0_OUT / "training_protocol.json").read_text())
    config = protocol["config"]
    records = load_cache(CACHE, CACHE_MANIFEST)
    selected = {}
    for scale in ("S", "M", "L"):
        candidates = sorted(
            (row for row in records if row["scale"] == scale),
            key=lambda row: (
                len(row["state_features"]["node_features"]), row["state_id"]))
        selected[scale] = candidates[len(candidates) // 2]
    selected["MAX"] = max(records, key=lambda row: (
        len(row["state_features"]["node_features"]), len(row["actions"]),
        row["state_id"]))

    checkpoint_path = C0_OUT / "oof/seed_746101/fold_0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_for(config, int(checkpoint["seed"]), "cuda")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    profiles = {}
    for name, record in selected.items():
        batch, _ = prepare(record, "cuda")
        with torch.inference_mode():
            for _ in range(20):
                model(batch)
            torch.cuda.synchronize()
            first = model(batch)
            second = model(batch)
            deterministic = all(torch.equal(first[key], second[key]) for key in first)
            finite = all(torch.isfinite(value).all() for value in first.values())
            samples = []
            for _ in range(100):
                started = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
        profiles[name] = {
            "state_id": record["state_id"],
            "nodes": len(record["state_features"]["node_features"]),
            "edges_with_reverse": len(record["state_features"]["edge_index"]),
            "joint_actions": len(record["actions"]),
            "mean_ms": statistics.fmean(samples),
            "p50_ms": percentile(samples, 0.50),
            "p90_ms": percentile(samples, 0.90),
            "p99_ms": percentile(samples, 0.99),
            "finite_outputs": bool(finite),
            "repeated_inference_bitwise_equal": bool(deterministic),
        }

    runs = [
        json.loads(path.read_text())
        for path in sorted((C0_OUT / "oof").glob("seed_*/fold_*.json"))]
    states_per_second = [
        run["train_states"] * run["epochs"] / run["runtime_seconds"]
        for run in runs]
    actions_per_second = []
    for run in runs:
        actions = sum(
            len(row["actions"]) for row in records
            if row["fold"] != run["held_fold"])
        actions_per_second.append(
            actions * run["epochs"] / run["runtime_seconds"])
    feasibility = json.loads((C0_OUT / "gpu_feasibility.json").read_text())
    value = {
        "schema": "ngas-a13r-c0-reference-gpu-profile-v1",
        "status": "PASS" if all(
            row["finite_outputs"] and row["repeated_inference_bitwise_equal"]
            for row in profiles.values()) else "FAIL",
        "scope": "post-selection read-only profile of immutable historical C0",
        "selection_effect": "none",
        "device": torch.cuda.get_device_name(0),
        "precision": "FP32",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        "states": profiles,
        "worst_representative_p90_ms": max(row["p90_ms"] for row in profiles.values()),
        "peak_inference_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
        "historical_training": {
            "runs": len(runs),
            "mean_training_states_per_second": statistics.fmean(states_per_second),
            "mean_training_actions_per_second": statistics.fmean(actions_per_second),
            "formal_peak_reserved_memory_bytes": None,
            "formal_peak_memory_not_recorded": True,
            "preformal_smoke_peak_reserved_memory_bytes":
                feasibility["memory"]["peak_reserved_bytes"],
        },
        "checkpoint_sha256": digest(checkpoint_path),
        "historical_protocol_sha256": digest(C0_OUT / "training_protocol.json"),
        "historical_output_modified": False,
        "r13": "LOCKED", "r14": "LOCKED", "gurobi_run": False,
    }
    if value["status"] != "PASS":
        raise RuntimeError("C0 reference GPU profile failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "status": value["status"],
        "parameters": value["parameters"],
        "worst_representative_p90_ms": value["worst_representative_p90_ms"],
        "peak_inference_reserved_memory_bytes":
            value["peak_inference_reserved_memory_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
