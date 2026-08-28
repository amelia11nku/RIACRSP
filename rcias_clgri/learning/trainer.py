"""Constructive PPO rollout collection and minibatch optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from .teacher_anchor import active_stage_mean, anchor_beta, teacher_stage_kl


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
    minimum_complete_episodes: int = 1
    encoder_lr: float | None = None
    policy_head_lr: float | None = None
    value_head_lr: float | None = None
    critic_stop_gradient: bool = False
    entropy_mode: str = "raw_joint"
    entropy_stage_coefficients: Mapping[str, float] = field(
        default_factory=lambda: {stage: 1.0 for stage in ("operation", "island", "w", "f")}
    )
    hard_kl_multiplier: float = 2.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PPOConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def critic_hidden(
    hidden: Mapping[str, torch.Tensor], *, stop_gradient: bool,
) -> Mapping[str, torch.Tensor]:
    return (
        {key: value.detach() for key, value in hidden.items()}
        if stop_gradient else hidden
    )


class PPOTrainer:
    def __init__(
        self,
        model: RCIASNeuralModel,
        tensorizer: GraphTensorizer,
        config: PPOConfig,
        *,
        device: torch.device | str,
        teacher_model: RCIASNeuralModel | None = None,
        teacher_anchor_config: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.tensorizer = tensorizer
        self.config = config
        self.device = torch.device(device)
        self.teacher_model = teacher_model
        self.teacher_anchor_config = teacher_anchor_config or {"enabled": False}
        self.optimizer = self._make_optimizer([
            config.encoder_lr or config.learning_rate,
            config.policy_head_lr or config.learning_rate,
            config.value_head_lr or config.learning_rate,
        ])

    def _make_optimizer(self, learning_rates: list[float]) -> torch.optim.AdamW:
        return torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": learning_rates[0], "name": "encoder"},
            {"params": self.model.policy.parameters(), "lr": learning_rates[1], "name": "policy"},
            {"params": self.model.value.parameters(), "lr": learning_rates[2], "name": "value"},
        ], weight_decay=1e-5)

    def reduce_learning_rates(
        self, factor: float, *, reset_optimizer_state: bool = True,
    ) -> None:
        if not 0.0 < factor < 1.0:
            raise ValueError("learning-rate reduction factor must be between zero and one")
        learning_rates = [float(group["lr"]) * factor for group in self.optimizer.param_groups]
        if reset_optimizer_state:
            self.optimizer = self._make_optimizer(learning_rates)
        else:
            for group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
                group["lr"] = learning_rate

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
        while (
            len(buffer) < self.config.rollout_transitions
            or len(episodes) < self.config.minimum_complete_episodes
        ):
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

    def update(
        self, buffer: RolloutBuffer, *, seed: int, update_number: int = 0,
    ) -> dict[str, float | int]:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("rollout buffer advantages are missing")
        self.model.train()
        records: dict[str, list[float]] = {
            key: [] for key in (
                "total_loss", "policy_loss", "value_loss", "entropy",
                "operation_entropy", "island_entropy", "w_entropy", "f_entropy",
                "normalized_entropy", "approx_kl", "clip_fraction",
                "explained_variance", "gradient_norm_before", "gradient_norm_after",
                "teacher_kl", "teacher_kl_operation", "teacher_kl_island",
                "teacher_kl_w", "teacher_kl_f",
                "normalized_operation_entropy", "normalized_island_entropy",
                "normalized_w_entropy", "normalized_f_entropy",
            )
        }
        beta = anchor_beta(self.teacher_anchor_config, update_number)
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
                    self.model.evaluate_action_from_hidden(graph, hidden, transition.action)
                    for graph, hidden, transition in zip(
                        graph_batch.graphs, hidden_batch, transitions
                    )
                ]
                new_log_prob = torch.stack([
                    evaluation.joint_log_prob for evaluation in evaluations
                ])
                value_hidden = [
                    critic_hidden(hidden, stop_gradient=self.config.critic_stop_gradient)
                    for hidden in hidden_batch
                ]
                values = torch.stack([self.model.value(hidden) for hidden in value_hidden])
                joint_entropy = torch.stack([
                    evaluation.joint_entropy for evaluation in evaluations
                ])
                normalized_entropy = torch.stack([
                    (
                        evaluation.active_stage_normalized_entropy
                        if hasattr(self.model, "trainable_stages")
                        else active_stage_mean(
                            evaluation.stage_normalized_entropies,
                            evaluation.stage_candidate_counts,
                            self.config.entropy_stage_coefficients,
                        )
                    ) for evaluation in evaluations
                ])
                entropy_for_loss = (
                    normalized_entropy
                    if self.config.entropy_mode == "normalized_stage"
                    else joint_entropy
                )
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
                    entropy=entropy_for_loss,
                    clip_epsilon=self.config.clip_epsilon,
                    value_coefficient=self.config.value_coefficient,
                    entropy_coefficient=self.config.entropy_coefficient,
                )
                teacher_stage_values = {stage: [] for stage in ("operation", "island", "w", "f")}
                if self.teacher_model is not None and (
                    beta > 0.0 or bool(self.teacher_anchor_config.get("measure_kl", False))
                ):
                    for graph, hidden, transition in zip(graph_batch.graphs, hidden_batch, transitions):
                        values_by_stage = teacher_stage_kl(
                            self.model, self.teacher_model, graph, hidden, transition.action
                        )
                        for stage, value in values_by_stage.items():
                            teacher_stage_values[stage].append(value)
                    teacher_per_sample = [
                        active_stage_mean(
                            {stage: teacher_stage_values[stage][index] for stage in teacher_stage_values},
                            evaluations[index].stage_candidate_counts,
                        ) for index in range(len(evaluations))
                    ]
                    teacher_loss = torch.stack(teacher_per_sample).mean()
                else:
                    teacher_loss = loss.total_loss.new_zeros(())
                total_loss = loss.total_loss + beta * teacher_loss
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(gradient_before):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                gradient_after = min(float(gradient_before), self.config.max_grad_norm)
                scalar = {
                    "total_loss": total_loss,
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
                    records[f"normalized_{stage}_entropy"].append(_mean([
                        float(evaluation.stage_normalized_entropies[stage].detach().cpu())
                        for evaluation in evaluations
                    ]))
                records["normalized_entropy"].append(_mean([
                    float(evaluation.active_stage_normalized_entropy.detach().cpu())
                    for evaluation in evaluations
                ]))
                records["teacher_kl"].append(float(teacher_loss.detach().cpu()))
                for stage in ("operation", "island", "w", "f"):
                    records[f"teacher_kl_{stage}"].append(
                        _mean([float(value.detach().cpu()) for value in teacher_stage_values[stage]])
                    )
                records["gradient_norm_before"].append(float(gradient_before))
                records["gradient_norm_after"].append(gradient_after)
                if records["approx_kl"][-1] > 1.5 * self.config.target_kl:
                    stopped_for_kl = True
                    break
                if records["approx_kl"][-1] > self.config.hard_kl_multiplier * self.config.target_kl:
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
            "encoder_learning_rate": self.optimizer.param_groups[0]["lr"],
            "policy_learning_rate": self.optimizer.param_groups[1]["lr"],
            "value_learning_rate": self.optimizer.param_groups[2]["lr"],
            "parameter_norm": math.sqrt(sum(
                float(parameter.detach().pow(2).sum().cpu())
                for parameter in self.model.parameters()
            )),
            "teacher_beta": beta,
            "max_kl": max(records["approx_kl"], default=0.0),
            "p95_kl": (
                sorted(records["approx_kl"])[min(
                    len(records["approx_kl"]) - 1,
                    math.ceil(0.95 * len(records["approx_kl"])) - 1,
                )] if records["approx_kl"] else 0.0
            ),
            "advantage_mean": float(buffer.advantages.mean()),
            "advantage_std": float(buffer.advantages.std(unbiased=False)),
            "return_variance": float(buffer.returns.var(unbiased=False)),
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
            "unique_instances": len({episode.instance_id for episode in episodes}),
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
