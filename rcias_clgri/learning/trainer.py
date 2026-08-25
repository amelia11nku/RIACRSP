"""Constructive PPO rollout collection and minibatch optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from time import perf_counter
from typing import Mapping

import torch

from rcias_clgri.nn.model import RCIASNeuralModel
from rcias_clgri.nn.tensorizer import BatchGraphTensor, GraphTensorizer
from rcias_clgri.training.curriculum import CurriculumManager
from rcias_clgri.training.instance_factory import TrainingInstanceFactory

from .buffer import RolloutBuffer
from .ppo import clipped_ppo_loss
from .rollout import RolloutEpisode, collect_episode


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_transitions: int = 192

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PPOConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


class PPOTrainer:
    def __init__(
        self,
        model: RCIASNeuralModel,
        tensorizer: GraphTensorizer,
        config: PPOConfig,
        *,
        device: torch.device | str,
    ) -> None:
        self.model = model.to(device)
        self.tensorizer = tensorizer
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )

    def collect(
        self,
        factory: TrainingInstanceFactory,
        curriculum: CurriculumManager,
        *,
        seed_rng: random.Random,
        forbidden_seeds: set[int],
    ) -> tuple[RolloutBuffer, list[RolloutEpisode], float]:
        started = perf_counter()
        buffer = RolloutBuffer()
        episodes: list[RolloutEpisode] = []
        while len(buffer) < self.config.rollout_transitions:
            level = curriculum.sample_level(seed_rng)
            seed = seed_rng.randrange(1, 2**31 - 1)
            while seed in forbidden_seeds:
                seed = seed_rng.randrange(1, 2**31 - 1)
            instance = factory.sample(seed, level)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            episode = collect_episode(
                self.model,
                self.tensorizer,
                instance,
                device=self.device,
                deterministic=False,
                generator=generator,
                store_transitions=True,
            )
            for transition in episode.transitions:
                buffer.add(transition)
            episodes.append(episode)
        buffer.compute_advantages(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        return buffer, episodes, perf_counter() - started

    def update(self, buffer: RolloutBuffer, *, seed: int) -> dict[str, float | int]:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("rollout buffer advantages are missing")
        self.model.train()
        records: dict[str, list[float]] = {
            key: [] for key in (
                "total_loss", "policy_loss", "value_loss", "entropy",
                "operation_entropy", "island_entropy", "w_entropy", "f_entropy",
                "normalized_entropy", "approx_kl", "clip_fraction",
                "explained_variance", "gradient_norm_before", "gradient_norm_after",
            )
        }
        epochs_completed = 0
        started = perf_counter()
        for epoch in range(self.config.update_epochs):
            stopped_for_kl = False
            for indices in buffer.minibatches(
                self.config.minibatch_size, seed=seed + epoch
            ):
                transitions = [buffer.transitions[index] for index in indices]
                graph_batch = BatchGraphTensor.from_graphs(
                    [transition.graph for transition in transitions]
                ).to(self.device)
                hidden_batch = self.model.encode_batch(graph_batch)
                evaluations = [
                    self.model.policy.evaluate_action(graph, hidden, transition.action)
                    for graph, hidden, transition in zip(
                        graph_batch.graphs, hidden_batch, transitions
                    )
                ]
                new_log_prob = torch.stack([
                    evaluation.joint_log_prob for evaluation in evaluations
                ])
                values = torch.stack([
                    self.model.value(hidden) for hidden in hidden_batch
                ])
                joint_entropy = torch.stack([
                    evaluation.joint_entropy for evaluation in evaluations
                ])
                old_log_prob = torch.tensor(
                    [transition.old_joint_log_prob for transition in transitions],
                    dtype=torch.float32,
                    device=self.device,
                )
                index_tensor = torch.tensor(indices, dtype=torch.long)
                advantages = buffer.advantages[index_tensor].to(self.device)
                returns = buffer.returns[index_tensor].to(self.device)
                loss = clipped_ppo_loss(
                    new_joint_log_prob=new_log_prob,
                    old_joint_log_prob=old_log_prob,
                    advantages=advantages,
                    values=values,
                    returns=returns,
                    entropy=joint_entropy,
                    clip_epsilon=self.config.clip_epsilon,
                    value_coefficient=self.config.value_coefficient,
                    entropy_coefficient=self.config.entropy_coefficient,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.total_loss.backward()
                gradient_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(gradient_before):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                gradient_after = min(float(gradient_before), self.config.max_grad_norm)
                scalar = {
                    "total_loss": loss.total_loss,
                    "policy_loss": loss.policy_loss,
                    "value_loss": loss.value_loss,
                    "entropy": loss.entropy,
                    "approx_kl": loss.approx_kl,
                    "clip_fraction": loss.clip_fraction,
                    "explained_variance": loss.explained_variance,
                }
                for name, value in scalar.items():
                    records[name].append(float(value.detach().cpu()))
                for stage in ("operation", "island", "w", "f"):
                    records[f"{stage}_entropy"].append(_mean([
                        float(evaluation.stage_entropies[stage].detach().cpu())
                        for evaluation in evaluations
                    ]))
                records["normalized_entropy"].append(_mean([
                    float(evaluation.active_stage_normalized_entropy.detach().cpu())
                    for evaluation in evaluations
                ]))
                records["gradient_norm_before"].append(float(gradient_before))
                records["gradient_norm_after"].append(gradient_after)
                if records["approx_kl"][-1] > 1.5 * self.config.target_kl:
                    stopped_for_kl = True
                    break
            epochs_completed += 1
            if stopped_for_kl:
                break
        metrics: dict[str, float | int] = {
            key: _mean(values) for key, values in records.items()
        }
        metrics.update({
            "epochs_completed": epochs_completed,
            "minibatches": len(records["total_loss"]),
            "update_time": perf_counter() - started,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "parameter_norm": math.sqrt(sum(
                float(parameter.detach().pow(2).sum().cpu())
                for parameter in self.model.parameters()
            )),
        })
        return metrics

    @staticmethod
    def rollout_metrics(episodes: list[RolloutEpisode]) -> dict[str, object]:
        total_steps = sum(len(episode.actions) for episode in episodes)
        timing = {
            name: sum(episode.timing[name] for episode in episodes)
            for name in ("graph_build", "forward", "policy_scoring", "decoder")
        }
        return {
            "episodes": len(episodes),
            "environment_steps": total_steps,
            "mean_episode_makespan": _mean([episode.makespan for episode in episodes]),
            "mean_normalized_return": _mean([
                episode.normalized_return for episode in episodes
            ]),
            "feasibility_rate": _mean([
                float(episode.feasible) for episode in episodes
            ]),
            "joint_entropy": _mean([episode.joint_entropy for episode in episodes]),
            "normalized_entropy": _mean([
                episode.normalized_entropy for episode in episodes
            ]),
            "stage_entropy": {
                stage: _mean([episode.stage_entropy[stage] for episode in episodes])
                for stage in ("operation", "island", "w", "f")
            },
            "timing": timing,
            "action_statistics": {
                key: _mean([
                    float(episode.action_statistics[key]) for episode in episodes
                ])
                for key in (
                    "mean_ready_operations", "mean_eligible_islands", "w_none_ratio",
                    "same_configuration_selection_ratio", "reconfiguration_count",
                )
            },
        }
