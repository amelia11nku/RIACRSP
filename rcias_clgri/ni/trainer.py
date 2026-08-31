"""Resumable supervised state-batched trainer for Phase 6E."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .batching import NIBatch, batch_state_samples
from .cache import load_shard_cache
from .encoder import NIModelConfig
from .losses import NILossConfig, phase6e_loss
from .metrics import evaluate_action_scores
from .scorer import CSGTargetSetScorer


@dataclass(frozen=True)
class NITrainingConfig:
    epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 1e-4
    checkpoint_every_shards: int = 5
    batch_states_s: int = 32
    batch_states_m: int = 16
    batch_states_l: int = 8

    def __post_init__(self) -> None:
        positive = {
            "epochs": self.epochs,
            "gradient_accumulation": self.gradient_accumulation,
            "checkpoint_every_shards": self.checkpoint_every_shards,
            "batch_states_s": self.batch_states_s,
            "batch_states_m": self.batch_states_m,
            "batch_states_l": self.batch_states_l,
        }
        if any(value < 1 for value in positive.values()):
            raise ValueError(f"training counts must be positive: {positive}")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer configuration")

    def batch_size(self, scale: str) -> int:
        try:
            return {"S": self.batch_states_s, "M": self.batch_states_m, "L": self.batch_states_l}[scale]
        except KeyError as error:
            raise ValueError(f"unknown graph scale: {scale}") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_fingerprint(
    model: NIModelConfig,
    loss: NILossConfig,
    training: NITrainingConfig,
    tensor_schema_hash: str,
) -> str:
    payload = {
        "model": asdict(model),
        "loss": asdict(loss),
        "training": asdict(training),
        "tensor_schema_hash": tensor_schema_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scored_action_frame(scores: torch.Tensor, batch: NIBatch) -> pd.DataFrame:
    state_index = batch.action_to_state.detach().cpu().tolist()
    metadata = [batch.structural_metadata[index] for index in state_index]
    return pd.DataFrame({
        "state_id": [batch.state_ids[index] for index in state_index],
        "instance_id": [batch.instance_ids[index] for index in state_index],
        "target_set_id": batch.target_set_ids,
        "arm_family": batch.arm_family,
        "origin_destroy_operator": batch.origin_destroy_operator,
        "mean_relative_improvement": batch.utility.detach().cpu().numpy(),
        "rank_within_state": batch.rank_within_state.detach().cpu().numpy(),
        "rank_percentile": batch.rank_percentile.detach().cpu().numpy(),
        "regret_to_best": batch.regret_to_best.detach().cpu().numpy(),
        "top1": batch.top1.detach().cpu().numpy(),
        "top3": batch.top3.detach().cpu().numpy(),
        "score": scores.detach().float().cpu().numpy(),
        **{
            name: [values.get(name, "") for values in metadata]
            for name in (
                "training_split", "scale", "CF_level", "RI_level", "TI_level",
                "search_stage", "bottleneck_proxy",
            )
        },
    })


class NITrainer:
    def __init__(
        self,
        model: CSGTargetSetScorer,
        *,
        model_config: NIModelConfig,
        loss_config: NILossConfig,
        training_config: NITrainingConfig,
        device: torch.device,
        seed: int,
        output_directory: Path,
    ) -> None:
        self.model = model.to(device)
        self.model_config = model_config
        self.loss_config = loss_config
        self.config = training_config
        self.device = device
        self.seed = seed
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = config_fingerprint(
            model_config, loss_config, training_config, model.tensor_schema_hash
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.amp_enabled = training_config.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)
        self.global_step = 0
        self.best_objective = -float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history: list[dict[str, object]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        next_shard_index: int,
        epoch_accumulator: Mapping[str, float],
        status: str,
    ) -> dict[str, object]:
        return {
            "schema": "phase6e-training-checkpoint-v1",
            "config_fingerprint": self.fingerprint,
            "tensor_schema_hash": self.model.tensor_schema_hash,
            "seed": self.seed,
            "epoch": epoch,
            "next_shard_index": next_shard_index,
            "global_step": self.global_step,
            "best_objective": self.best_objective,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "epoch_accumulator": dict(epoch_accumulator),
            "history": self.history,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "model_config": asdict(self.model_config),
            "loss_config": asdict(self.loss_config),
            "training_config": asdict(self.config),
            "status": status,
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        next_shard_index: int,
        epoch_accumulator: Mapping[str, float],
        status: str,
    ) -> None:
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(
            self._checkpoint_payload(
                epoch=epoch,
                next_shard_index=next_shard_index,
                epoch_accumulator=epoch_accumulator,
                status=status,
            ),
            temporary,
        )
        temporary.replace(path)

    def resume(self, path: Path) -> tuple[int, int, dict[str, float]]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("config_fingerprint") != self.fingerprint:
            raise ValueError("resume checkpoint configuration mismatch")
        if int(checkpoint.get("seed")) != self.seed:
            raise ValueError("resume checkpoint seed mismatch")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.global_step = int(checkpoint["global_step"])
        self.best_objective = float(checkpoint["best_objective"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        self.history = list(checkpoint["history"])
        return (
            int(checkpoint["epoch"]),
            int(checkpoint["next_shard_index"]),
            {key: float(value) for key, value in checkpoint["epoch_accumulator"].items()},
        )

    def _shard_order(self, records: pd.DataFrame, epoch: int) -> list[dict[str, object]]:
        values = records.sort_values("instance_id").to_dict("records")
        random.Random(self.seed + 1009 * epoch).shuffle(values)
        return values

    def _sample_batches(self, samples, *, epoch: int, instance_id: str):
        digest = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16)
        indices = list(range(len(samples)))
        random.Random(self.seed + 65537 * epoch + digest).shuffle(indices)
        scale = str(samples[0].structural_metadata["scale"])
        batch_size = self.config.batch_size(scale)
        for start in range(0, len(indices), batch_size):
            yield batch_state_samples([samples[index] for index in indices[start:start + batch_size]])

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def train_epoch(
        self,
        records: pd.DataFrame,
        *,
        epoch: int,
        start_shard_index: int,
        accumulator: dict[str, float] | None = None,
    ) -> dict[str, float]:
        values = dict(accumulator or {
            "weighted_loss": 0.0,
            "weighted_rank_loss": 0.0,
            "weighted_classification_loss": 0.0,
            "action_count": 0.0,
            "state_count": 0.0,
            "batch_count": 0.0,
        })
        shard_order = self._shard_order(records, epoch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_step = 0
        epoch_started = time.perf_counter()
        for shard_index in range(start_shard_index, len(shard_order)):
            record = shard_order[shard_index]
            samples, _ = load_shard_cache(
                Path(str(record["cache_path"])),
                expected_tensor_schema_hash=self.model.tensor_schema_hash,
                expected_source_shard_sha256=str(record["source_shard_sha256"]),
            )
            for batch_cpu in self._sample_batches(
                samples, epoch=epoch, instance_id=str(record["instance_id"])
            ):
                batch = batch_cpu.to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    output = self.model(batch)
                    losses = phase6e_loss(output.scores, batch, self.loss_config)
                    scaled_loss = losses["loss"] / self.config.gradient_accumulation
                self.scaler.scale(scaled_loss).backward()
                accumulation_step += 1
                if accumulation_step == self.config.gradient_accumulation:
                    self._optimizer_step()
                    accumulation_step = 0
                actions = batch.action_count
                values["weighted_loss"] += float(losses["loss"].detach()) * actions
                values["weighted_rank_loss"] += float(losses["rank_loss"].detach()) * actions
                values["weighted_classification_loss"] += (
                    float(losses["classification_loss"].detach()) * actions
                )
                values["action_count"] += actions
                values["state_count"] += batch.state_count
                values["batch_count"] += 1
            if accumulation_step:
                self._optimizer_step()
                accumulation_step = 0
            if (
                (shard_index + 1) % self.config.checkpoint_every_shards == 0
                or shard_index + 1 == len(shard_order)
            ):
                self.save_checkpoint(
                    self.output_directory / "checkpoint_last.pt",
                    epoch=epoch,
                    next_shard_index=shard_index + 1,
                    epoch_accumulator=values,
                    status="TRAINING",
                )
            elapsed = time.perf_counter() - epoch_started
            event = {
                "event": "train_shard",
                "epoch": epoch,
                "completed_shards": shard_index + 1,
                "total_shards": len(shard_order),
                "states": int(values["state_count"]),
                "actions": int(values["action_count"]),
                "states_per_second": values["state_count"] / max(elapsed, 1e-9),
                "global_step": self.global_step,
                "instance_id": record["instance_id"],
                "gpu_peak_memory_mib": (
                    torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    if self.device.type == "cuda" else 0.0
                ),
            }
            print(json.dumps(event), flush=True)
        count = max(values["action_count"], 1)
        return {
            **values,
            "train_loss": values["weighted_loss"] / count,
            "train_rank_loss": values["weighted_rank_loss"] / count,
            "train_classification_loss": values["weighted_classification_loss"] / count,
        }

    def evaluate(self, records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        self.model.eval()
        frames = []
        with torch.no_grad():
            for record in records.sort_values("instance_id").to_dict("records"):
                samples, _ = load_shard_cache(
                    Path(str(record["cache_path"])),
                    expected_tensor_schema_hash=self.model.tensor_schema_hash,
                    expected_source_shard_sha256=str(record["source_shard_sha256"]),
                )
                scale = str(samples[0].structural_metadata["scale"])
                batch_size = self.config.batch_size(scale)
                for start in range(0, len(samples), batch_size):
                    batch = batch_state_samples(samples[start:start + batch_size]).to(self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.amp_enabled,
                    ):
                        scores = self.model(batch).scores
                    frames.append(scored_action_frame(scores, batch))
        scored = pd.concat(frames, ignore_index=True)
        return scored, evaluate_action_scores(scored)

    def fit(
        self,
        train_records: pd.DataFrame,
        validation_records: pd.DataFrame,
        *,
        resume_path: Path | None = None,
    ) -> dict[str, object]:
        if set(train_records["training_split"]) != {"TRAIN"}:
            raise ValueError("gradient records must be TRAIN only")
        if set(validation_records["training_split"]) != {"TRAIN_VALIDATION"}:
            raise ValueError("tuning records must be TRAIN_VALIDATION only")
        seed_everything(self.seed)
        start_epoch, start_shard, accumulator = 1, 0, {}
        if resume_path is not None:
            start_epoch, start_shard, accumulator = self.resume(resume_path)
        for epoch in range(start_epoch, self.config.epochs + 1):
            training = self.train_epoch(
                train_records,
                epoch=epoch,
                start_shard_index=start_shard if epoch == start_epoch else 0,
                accumulator=accumulator if epoch == start_epoch else None,
            )
            scored, validation = self.evaluate(validation_records)
            objective = float(
                0.5 * validation["pairwise_accuracy"] + 0.5 * validation["ndcg"]
            )
            improved = objective > self.best_objective + self.config.early_stopping_min_delta
            if improved:
                self.best_objective = objective
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "global_step": self.global_step,
                **{key: value for key, value in training.items() if key.startswith("train_")},
                **{f"validation_{key}": value for key, value in validation.items()},
                "validation_objective": objective,
                "improved": improved,
            }
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(
                self.output_directory / "training_history.csv", index=False
            )
            scored.to_parquet(
                self.output_directory / f"validation_scores_epoch_{epoch:02d}.parquet",
                index=False,
            )
            if improved:
                self.save_checkpoint(
                    self.output_directory / "checkpoint_best.pt",
                    epoch=epoch + 1,
                    next_shard_index=0,
                    epoch_accumulator={},
                    status="BEST_VALIDATION",
                )
            self.save_checkpoint(
                self.output_directory / "checkpoint_last.pt",
                epoch=epoch + 1,
                next_shard_index=0,
                epoch_accumulator={},
                status="EPOCH_COMPLETE",
            )
            print(json.dumps({"event": "epoch_complete", **row}), flush=True)
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                break
            start_shard, accumulator = 0, {}
        summary = {
            "schema": "phase6e-training-summary-v1",
            "status": "COMPLETE",
            "seed": self.seed,
            "config_fingerprint": self.fingerprint,
            "best_epoch": self.best_epoch,
            "best_validation_objective": self.best_objective,
            "completed_epochs": len(self.history),
            "global_step": self.global_step,
            "parameter_count": self.model.parameter_count(),
            "amp_enabled": self.amp_enabled,
            "gpu_peak_memory_mib": (
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if self.device.type == "cuda" else 0.0
            ),
        }
        (self.output_directory / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
