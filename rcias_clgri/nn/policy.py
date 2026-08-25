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

    def greedy_action(
        self, graph: GraphTensor, hidden: Mapping[str, torch.Tensor],
    ) -> Action:
        op_id = self.operation_distribution(graph, hidden).argmax()
        island_id = self.island_distribution(graph, hidden, op_id).argmax()
        w_id = self.w_distribution(graph, hidden, op_id, island_id).argmax()
        f_id = self.f_distribution(graph, hidden, op_id, island_id, w_id).argmax()
        return Action(op_id, island_id, w_id, f_id)
