"""Hard-masked O -> M -> W -> F autoregressive policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

import torch
from torch import nn
from torch.nn import functional as F

from rcias_clgri.env.insertion_decoder import Action

from .tensorizer import GraphTensor, NODE_TYPES

CandidateId = TypeVar("CandidateId")


@dataclass(frozen=True)
class CandidateDistribution(Generic[CandidateId]):
    candidate_ids: tuple[CandidateId, ...]
    logits: torch.Tensor

    def argmax(self) -> CandidateId:
        if not self.candidate_ids:
            raise RuntimeError("cannot select from an empty hard-masked candidate set")
        return self.candidate_ids[int(torch.argmax(self.logits).item())]

    def probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self.candidate_ids:
            raise RuntimeError("cannot normalize an empty hard-masked candidate set")
        if len(self.candidate_ids) == 1:
            return torch.ones_like(self.logits)
        return torch.softmax(self.logits / temperature, dim=0)

    def probability(self, candidate_id: CandidateId, temperature: float = 1.0) -> torch.Tensor:
        if candidate_id not in self.candidate_ids:
            return self.logits.sum() * 0.0
        return self.probabilities(temperature)[self.candidate_ids.index(candidate_id)]

    def sample(
        self,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> CandidateId:
        if deterministic:
            return self.argmax()
        probabilities = self.probabilities(temperature)
        index = torch.multinomial(probabilities, 1, generator=generator)
        return self.candidate_ids[int(index.item())]

    def log_prob(self, candidate_id: CandidateId, temperature: float = 1.0) -> torch.Tensor:
        if candidate_id not in self.candidate_ids:
            raise ValueError(
                f"target {candidate_id!r} violates the hard mask {self.candidate_ids}"
            )
        if len(self.candidate_ids) == 1:
            return self.logits.sum() * 0.0
        index = self.candidate_ids.index(candidate_id)
        return torch.log_softmax(self.logits / temperature, dim=0)[index]

    def entropy(self, temperature: float = 1.0) -> torch.Tensor:
        if len(self.candidate_ids) <= 1:
            return self.logits.sum() * 0.0
        log_probabilities = torch.log_softmax(self.logits / temperature, dim=0)
        probabilities = torch.exp(log_probabilities)
        return -(probabilities * log_probabilities).sum()

    def normalized_entropy(self, temperature: float = 1.0) -> torch.Tensor:
        if len(self.candidate_ids) <= 1:
            return self.logits.sum() * 0.0
        scale = torch.log(self.logits.new_tensor(float(len(self.candidate_ids))))
        return self.entropy(temperature) / scale


@dataclass(frozen=True)
class PolicyActionEvaluation:
    action: Action
    joint_log_prob: torch.Tensor
    stage_log_probs: Mapping[str, torch.Tensor]
    stage_entropies: Mapping[str, torch.Tensor]
    stage_normalized_entropies: Mapping[str, torch.Tensor]
    joint_entropy: torch.Tensor
    active_stage_normalized_entropy: torch.Tensor
    stage_candidate_counts: Mapping[str, int]


def _scorer(input_dim: int, embedding_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1)
    )


class AutoregressivePolicy(nn.Module):
    """Score legal candidates stage-by-stage without a Cartesian action tensor."""

    def __init__(self, embedding_dim: int, candidate_dims: Mapping[str, int]) -> None:
        super().__init__()
        dim = embedding_dim
        self.none_w_embedding = nn.Parameter(torch.zeros(dim))
        self.operation_scorer = _scorer(2 * dim + candidate_dims["operation"], dim)
        self.island_scorer = _scorer(3 * dim + candidate_dims["island"], dim)
        self.w_scorer = _scorer(4 * dim + candidate_dims["w"], dim)
        self.f_scorer = _scorer(5 * dim + candidate_dims["f"], dim)

    @staticmethod
    def _global_context(hidden: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([hidden[kind].mean(dim=0) for kind in NODE_TYPES]).mean(dim=0)

    @staticmethod
    def _score(scorer: nn.Module, rows: list[torch.Tensor]) -> torch.Tensor:
        if not rows:
            return torch.empty(0)
        return scorer(torch.stack(rows)).squeeze(-1)

    def operation_distribution(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor],
    ) -> CandidateDistribution[str]:
        ids = graph.candidates.ready_operations
        context = self._global_context(hidden)
        rows = [
            torch.cat([
                hidden["O"][graph.node_index["O"][op_id]], context,
                graph.candidates.operation_features[op_id],
            ])
            for op_id in ids
        ]
        return CandidateDistribution(ids, self._score(self.operation_scorer, rows))

    def island_distribution(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor], op_id: str,
    ) -> CandidateDistribution[str]:
        mask = graph.candidates.island_masks[op_id]
        ids = tuple(island_id for island_id, allowed in mask.items() if allowed)
        context = self._global_context(hidden)
        op_embedding = hidden["O"][graph.node_index["O"][op_id]]
        rows = [
            torch.cat([
                hidden["M"][graph.node_index["M"][island_id]], op_embedding, context,
                graph.candidates.island_features[(op_id, island_id)],
            ])
            for island_id in ids
        ]
        return CandidateDistribution(ids, self._score(self.island_scorer, rows))

    def _w_embedding(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor], vehicle_id: str | None,
    ) -> torch.Tensor:
        if vehicle_id is None:
            return self.none_w_embedding
        return hidden["W"][graph.node_index["W"][vehicle_id]]

    def w_distribution(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor],
        op_id: str, island_id: str,
    ) -> CandidateDistribution[str | None]:
        ids = graph.candidates.w_masks[(op_id, island_id)]
        context = self._global_context(hidden)
        op_embedding = hidden["O"][graph.node_index["O"][op_id]]
        island_embedding = hidden["M"][graph.node_index["M"][island_id]]
        rows = [
            torch.cat([
                self._w_embedding(graph, hidden, vehicle_id), island_embedding,
                op_embedding, context,
                graph.candidates.w_features[(op_id, island_id, vehicle_id)],
            ])
            for vehicle_id in ids
        ]
        return CandidateDistribution(ids, self._score(self.w_scorer, rows))

    def f_distribution(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor],
        op_id: str, island_id: str, w_id: str | None,
    ) -> CandidateDistribution[str]:
        ids = graph.candidates.f_masks[(op_id, island_id)]
        context = self._global_context(hidden)
        op_embedding = hidden["O"][graph.node_index["O"][op_id]]
        island_embedding = hidden["M"][graph.node_index["M"][island_id]]
        w_embedding = self._w_embedding(graph, hidden, w_id)
        rows = [
            torch.cat([
                hidden["F"][graph.node_index["F"][vehicle_id]], w_embedding,
                island_embedding, op_embedding, context,
                graph.candidates.f_features[(op_id, island_id, vehicle_id)],
            ])
            for vehicle_id in ids
        ]
        return CandidateDistribution(ids, self._score(self.f_scorer, rows))

    @staticmethod
    def imitation_loss(distribution: CandidateDistribution, target) -> torch.Tensor:
        if target not in distribution.candidate_ids:
            raise ValueError(f"target {target!r} violates the hard mask {distribution.candidate_ids}")
        index = distribution.candidate_ids.index(target)
        return F.cross_entropy(
            distribution.logits.unsqueeze(0),
            torch.tensor([index], dtype=torch.long, device=distribution.logits.device),
        )

    def action_losses(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor], action: Action,
    ) -> dict[str, torch.Tensor]:
        operation = self.operation_distribution(graph, hidden)
        island = self.island_distribution(graph, hidden, action.operation_id)
        w_agv = self.w_distribution(graph, hidden, action.operation_id, action.island_id)
        f_agv = self.f_distribution(
            graph, hidden, action.operation_id, action.island_id, action.w_agv_id
        )
        losses = {
            "operation": self.imitation_loss(operation, action.operation_id),
            "island": self.imitation_loss(island, action.island_id),
            "w": self.imitation_loss(w_agv, action.w_agv_id),
            "f": self.imitation_loss(f_agv, action.f_agv_id),
        }
        losses["total"] = sum(losses.values())
        return losses

    def evaluate_action(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
        *,
        temperature: float = 1.0,
    ) -> PolicyActionEvaluation:
        distributions = self.action_distributions(graph, hidden, action)
        selected = {
            "operation": action.operation_id,
            "island": action.island_id,
            "w": action.w_agv_id,
            "f": action.f_agv_id,
        }
        stage_log_probs = {
            name: distribution.log_prob(selected[name], temperature)
            for name, distribution in distributions.items()
        }
        stage_entropies = {
            name: distribution.entropy(temperature)
            for name, distribution in distributions.items()
        }
        stage_normalized_entropies = {
            name: distribution.normalized_entropy(temperature)
            for name, distribution in distributions.items()
        }
        normalized = [
            stage_normalized_entropies[name]
            for name, distribution in distributions.items()
            if len(distribution.candidate_ids) > 1
        ]
        zero = next(iter(stage_log_probs.values())).new_zeros(())
        return PolicyActionEvaluation(
            action=action,
            joint_log_prob=torch.stack(tuple(stage_log_probs.values())).sum(),
            stage_log_probs=stage_log_probs,
            stage_entropies=stage_entropies,
            stage_normalized_entropies=stage_normalized_entropies,
            joint_entropy=torch.stack(tuple(stage_entropies.values())).sum(),
            active_stage_normalized_entropy=(
                torch.stack(normalized).mean() if normalized else zero
            ),
            stage_candidate_counts={
                name: len(distribution.candidate_ids)
                for name, distribution in distributions.items()
            },
        )

    def action_distributions(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
    ) -> dict[str, CandidateDistribution]:
        """Return all hard-masked stage distributions along an action path."""
        return {
            "operation": self.operation_distribution(graph, hidden),
            "island": self.island_distribution(graph, hidden, action.operation_id),
            "w": self.w_distribution(
                graph, hidden, action.operation_id, action.island_id
            ),
            "f": self.f_distribution(
                graph, hidden, action.operation_id, action.island_id, action.w_agv_id
            ),
        }

    def sample_action(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> PolicyActionEvaluation:
        operation = self.operation_distribution(graph, hidden)
        op_id = operation.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        island = self.island_distribution(graph, hidden, op_id)
        island_id = island.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        w_agv = self.w_distribution(graph, hidden, op_id, island_id)
        w_id = w_agv.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        f_agv = self.f_distribution(graph, hidden, op_id, island_id, w_id)
        f_id = f_agv.sample(
            deterministic=deterministic, temperature=temperature, generator=generator
        )
        return self.evaluate_action(
            graph, hidden, Action(op_id, island_id, w_id, f_id),
            temperature=temperature,
        )

    def action_log_prob(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self.evaluate_action(
            graph, hidden, action, temperature=temperature
        ).joint_log_prob

    def action_entropy(
        self,
        graph: GraphTensor,
        hidden: Mapping[str, torch.Tensor],
        action: Action,
        *,
        temperature: float = 1.0,
    ) -> PolicyActionEvaluation:
        return self.evaluate_action(graph, hidden, action, temperature=temperature)

    def greedy_action(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor],
    ) -> Action:
        return self.sample_action(graph, hidden, deterministic=True).action
