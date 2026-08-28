"""Operation-anchored dual-branch policy for downstream M/W/F refinement."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from rcias_clgri.env.insertion_decoder import Action

from .model import RCIASNeuralModel
from .policy import PolicyActionEvaluation
from .tensorizer import BatchGraphTensor, GraphTensor


ALL_DOWNSTREAM_STAGES = ("island", "w", "f")


def _freeze(module: nn.Module) -> None:
    module.eval()
    module.requires_grad_(False)


class FrozenOperationBranch(nn.Module):
    """Freeze the complete graph-state to BC operation-distribution mapping."""

    def __init__(self, bc_model: RCIASNeuralModel) -> None:
        super().__init__()
        self.model = bc_model
        _freeze(self.model)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def distribution(self, graph: GraphTensor):
        hidden = self.model.encode(graph)
        return self.model.policy.operation_distribution(graph, hidden)

    @torch.no_grad()
    def prefix_distributions(self, graph: GraphTensor, action: Action):
        hidden = self.model.encode(graph)
        return {
            "operation": self.model.policy.operation_distribution(graph, hidden),
            "island": self.model.policy.island_distribution(
                graph, hidden, action.operation_id
            ),
        }


class OperationAnchoredModel(nn.Module):
    """Use frozen greedy BC sequencing and trainable downstream decisions/value."""

    def __init__(
        self,
        frozen_bc_model: RCIASNeuralModel,
        downstream_model: RCIASNeuralModel,
        *,
        frozen_prefix_stages: int = 1,
    ) -> None:
        super().__init__()
        if frozen_prefix_stages not in (1, 2):
            raise ValueError("Phase 5B frozen prefix must contain O or O+M")
        self.frozen_operation = FrozenOperationBranch(frozen_bc_model)
        self.downstream = downstream_model
        self.config = downstream_model.config
        self.frozen_prefix_stages = frozen_prefix_stages
        self.trainable_stages = (
            ALL_DOWNSTREAM_STAGES if frozen_prefix_stages == 1 else ("w", "f")
        )

    @property
    def encoder(self):
        return self.downstream.encoder

    @property
    def policy(self):
        return self.downstream.policy

    @property
    def value(self):
        return self.downstream.value

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_operation.eval()
        return self

    def encode(self, graph: GraphTensor | BatchGraphTensor):
        return self.downstream.encode(graph)

    def encode_batch(self, graph: BatchGraphTensor):
        return self.downstream.encode_batch(graph)

    def _evaluate(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
        *,
        temperature: float = 1.0,
    ) -> PolicyActionEvaluation:
        frozen_distributions = self.frozen_operation.prefix_distributions(graph, action)
        downstream_distributions = self.policy.action_distributions(graph, hidden, action)
        distributions = {
            "operation": frozen_distributions["operation"],
            "island": (
                frozen_distributions["island"]
                if self.frozen_prefix_stages == 2
                else downstream_distributions["island"]
            ),
            "w": downstream_distributions["w"],
            "f": downstream_distributions["f"],
        }
        selected = {
            "operation": action.operation_id,
            "island": action.island_id,
            "w": action.w_agv_id,
            "f": action.f_agv_id,
        }
        stage_log_probs = {
            stage: distribution.log_prob(selected[stage], temperature)
            for stage, distribution in distributions.items()
        }
        stage_entropies = {
            stage: distribution.entropy(temperature)
            for stage, distribution in distributions.items()
        }
        stage_normalized_entropies = {
            stage: distribution.normalized_entropy(temperature)
            for stage, distribution in distributions.items()
        }
        active_downstream = [
            stage_normalized_entropies[stage]
            for stage in self.trainable_stages
            if len(distributions[stage].candidate_ids) > 1
        ]
        zero = stage_log_probs["operation"].new_zeros(())
        return PolicyActionEvaluation(
            action=action,
            joint_log_prob=torch.stack([
                stage_log_probs[stage] for stage in self.trainable_stages
            ]).sum(),
            stage_log_probs=stage_log_probs,
            stage_entropies=stage_entropies,
            stage_normalized_entropies=stage_normalized_entropies,
            joint_entropy=torch.stack([
                stage_entropies[stage] for stage in self.trainable_stages
            ]).sum(),
            active_stage_normalized_entropy=(
                torch.stack(active_downstream).mean() if active_downstream else zero
            ),
            stage_candidate_counts={
                stage: len(distribution.candidate_ids)
                for stage, distribution in distributions.items()
            },
        )

    def sample_action_from_hidden(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> PolicyActionEvaluation:
        operation_id = self.frozen_operation.distribution(graph).argmax()
        if self.frozen_prefix_stages == 2:
            with torch.no_grad():
                frozen_hidden = self.frozen_operation.model.encode(graph)
                island = self.frozen_operation.model.policy.island_distribution(
                    graph, frozen_hidden, operation_id
                )
        else:
            island = self.policy.island_distribution(graph, hidden, operation_id)
        island_id = island.sample(
            deterministic=(True if self.frozen_prefix_stages == 2 else deterministic),
            temperature=temperature,
            generator=generator,
        )
        w_agv = self.policy.w_distribution(graph, hidden, operation_id, island_id)
        w_id = w_agv.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        f_agv = self.policy.f_distribution(graph, hidden, operation_id, island_id, w_id)
        f_id = f_agv.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        return self._evaluate(
            graph,
            hidden,
            Action(operation_id, island_id, w_id, f_id),
            temperature=temperature,
        )

    def evaluate_action_from_hidden(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
        *,
        temperature: float = 1.0,
    ) -> PolicyActionEvaluation:
        return self._evaluate(graph, hidden, action, temperature=temperature)

    def forward(self, graph: GraphTensor) -> torch.Tensor:
        return self.value(self.encode(graph))
