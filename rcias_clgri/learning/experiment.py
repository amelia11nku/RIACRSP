"""Shared Phase 3 experiment setup and held-out validation helpers."""

from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path
import random
import socket
import subprocess
from typing import Iterable, Mapping

import numpy as np
import torch

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, ModelConfig, RCIASNeuralModel
from rcias_clgri.training import TrainingInstanceFactory

from .rollout import collect_episode


def run_metadata(
    config_path: str | Path, *, device: torch.device | str, training_seed: int,
) -> dict[str, object]:
    """Collect the reproducibility metadata required for a major training run."""
    resolved_device = torch.device(device)
    config_bytes = Path(config_path).read_bytes()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    try:
        import psutil
        ram_bytes = int(psutil.virtual_memory().total)
        cpu = platform.processor() or platform.machine()
    except ImportError:
        ram_bytes = 0
        cpu = platform.processor() or platform.machine()
    gpu = None
    vram_bytes = 0
    if resolved_device.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved_device)
        gpu = properties.name
        vram_bytes = int(properties.total_memory)
    return {
        "hostname": socket.gethostname(),
        "cpu": cpu,
        "ram_bytes": ram_bytes,
        "gpu": gpu,
        "vram_bytes": vram_bytes,
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "device": str(resolved_device),
        "training_seed": int(training_seed),
        "git_commit": git_commit,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def load_phase3_config(path: str | Path) -> dict[str, object]:
    target = Path(path)
    config = json.loads(target.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = load_phase3_config(target.parent / str(parent))

    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = (
                merge(result[key], value)
                if key in result and isinstance(result[key], dict) and isinstance(value, dict)
                else value
            )
        return result

    return merge(base, config)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def make_factory(root: Path, config: Mapping[str, object]) -> TrainingInstanceFactory:
    return TrainingInstanceFactory(
        config["curriculum_levels"],
        root / "instances" / "canonical" / "RCIAS-2.0" / "manifest.csv",
    )


def initialize_model(
    factory: TrainingInstanceFactory,
    config: Mapping[str, object],
    *,
    seed: int,
    device: torch.device | str,
) -> tuple[RCIASNeuralModel, GraphTensorizer]:
    seed_everything(seed)
    instance = factory.sample(seed + 9173, "S")
    graph_state = build_graph_state(
        instance, InsertionDecoder(instance).empty_schedule()
    )
    tensorizer = GraphTensorizer(graph_state)
    model = RCIASNeuralModel(tensorizer, ModelConfig(**config["model"]))
    return model.to(device), tensorizer


def validate_policy(
    model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    factory: TrainingInstanceFactory,
    validation_seeds: Mapping[str, Iterable[int]],
    *,
    levels: Iterable[str],
    device: torch.device | str,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for level in levels:
        for seed in validation_seeds[level]:
            instance = factory.sample(int(seed), level)
            episode = collect_episode(
                model,
                tensorizer,
                instance,
                device=device,
                deterministic=True,
                store_transitions=False,
            )
            records.append({
                "level": level,
                "seed": int(seed),
                "instance_id": instance.instance_id,
                "makespan": episode.makespan,
                "normalized_makespan": episode.makespan / episode.reward_scale,
                "feasible": episode.feasible,
                "normalized_entropy": episode.normalized_entropy,
            })
    return {
        "records": records,
        "mean_makespan": sum(float(row["makespan"]) for row in records) / len(records),
        "mean_normalized_makespan": (
            sum(float(row["normalized_makespan"]) for row in records) / len(records)
        ),
        "feasibility_rate": sum(float(bool(row["feasible"])) for row in records) / len(records),
        "mean_normalized_entropy": (
            sum(float(row["normalized_entropy"]) for row in records) / len(records)
        ),
    }
