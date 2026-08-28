"""Deterministic policy and constructive-baseline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch
from torch.torch_version import TorchVersion

from rcias_clgri.data.instance import Instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.nn.config import ModelConfig
from rcias_clgri.nn.model import RCIASNeuralModel
from rcias_clgri.nn.hierarchical_policy import OperationAnchoredModel
from rcias_clgri.nn.tensorizer import GraphTensorizer

from .rollout import collect_episode


@dataclass(frozen=True)
class EvaluationResult:
    instance_id: str
    method: str
    makespan: float
    runtime_seconds: float
    inference_seconds: float
    feasible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_policy(
    model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    instance: Instance,
    *,
    device: torch.device | str,
    method: str = "PPO_GREEDY",
) -> EvaluationResult:
    started = perf_counter()
    episode = collect_episode(
        model, tensorizer, instance, device=device, deterministic=True,
        store_transitions=False,
    )
    runtime = perf_counter() - started
    inference = episode.timing["forward"] + episode.timing["policy_scoring"]
    return EvaluationResult(
        instance.instance_id, method, episode.makespan, runtime, inference, episode.feasible
    )


def evaluate_baselines(instance: Instance) -> list[EvaluationResult]:
    results = []
    for method in ("H1", "H2", "H3"):
        result = solve_dispatching(instance, method)
        results.append(EvaluationResult(
            instance.instance_id,
            method,
            result.objective.makespan,
            result.runtime_seconds,
            result.runtime_seconds,
            True,
        ))
    return results


def checkpoint_payload(
    model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "tensorizer_schema": tensorizer.to_schema(),
        "metadata": metadata,
    }


def save_checkpoint(
    path: str | Path,
    model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    *,
    metadata: dict[str, object],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, tensorizer, metadata=metadata), target)


def save_operation_anchored_checkpoint(
    path: str | Path,
    model: OperationAnchoredModel,
    tensorizer: GraphTensorizer,
    *,
    frozen_operation_checkpoint: str,
    metadata: dict[str, object],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_type": "operation_anchored",
        "downstream_model_state": model.downstream.state_dict(),
        "model_config": model.config.to_dict(),
        "tensorizer_schema": tensorizer.to_schema(),
        "frozen_operation_checkpoint": frozen_operation_checkpoint,
        "frozen_prefix_stages": model.frozen_prefix_stages,
        "metadata": metadata,
    }, target)


def load_checkpoint(
    path: str | Path, *, device: torch.device | str,
) -> tuple[RCIASNeuralModel, GraphTensorizer, dict[str, object]]:
    # PyTorch exposes ``torch.__version__`` as TorchVersion. Older locally-created
    # checkpoints may contain that harmless metadata type; keep weights-only loading
    # and allowlist only this concrete standard-library-adjacent PyTorch class.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    tensorizer = GraphTensorizer.from_schema(payload["tensorizer_schema"])
    model = RCIASNeuralModel(tensorizer, ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tensorizer, payload.get("metadata", {})


def load_operation_anchored_checkpoint(
    path: str | Path, *, device: torch.device | str,
) -> tuple[OperationAnchoredModel, GraphTensorizer, dict[str, object]]:
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("checkpoint_type") != "operation_anchored":
        raise ValueError("checkpoint is not an operation-anchored Phase 5B policy")
    tensorizer = GraphTensorizer.from_schema(payload["tensorizer_schema"])
    downstream = RCIASNeuralModel(tensorizer, ModelConfig(**payload["model_config"]))
    downstream.load_state_dict(payload["downstream_model_state"])
    frozen, frozen_tensorizer, _ = load_checkpoint(
        payload["frozen_operation_checkpoint"], device=device
    )
    if frozen_tensorizer.to_schema() != tensorizer.to_schema():
        raise RuntimeError("frozen-operation and downstream tensorizer schemas differ")
    model = OperationAnchoredModel(
        frozen,
        downstream,
        frozen_prefix_stages=int(payload.get("frozen_prefix_stages", 1)),
    ).to(device)
    model.eval()
    return model, tensorizer, payload.get("metadata", {})


def evaluate_instances(
    instances: Iterable[Instance],
    *,
    policy_models: Iterable[tuple[str, RCIASNeuralModel, GraphTensorizer]],
    device: torch.device | str,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for instance in instances:
        results.extend(evaluate_baselines(instance))
        for method, model, tensorizer in policy_models:
            results.append(evaluate_policy(
                model, tensorizer, instance, device=device, method=method
            ))
    return results
