"""Candidate-conditioned CSG critic for Phase 6N."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rcias_clgri.csg.schema import NODE_TYPE_ORDER

from .action_encoder import segment_max
from .batching import NIBatch
from .encoder import segment_mean, segment_softmax
from .scorer import CSGTargetSetScorer


FAMILY = "N1_CANDIDATE_CONDITIONED_CSG"
BOUNDARY_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "precedence": ("OP__PRECEDES__OP",),
    "island_resource": (
        "OP__ELIGIBLE_ON__ISLAND",
        "OP__ASSIGNED_TO__ISLAND",
        "OP__ISLAND_NEXT__OP",
    ),
    "reconfiguration": (
        "OP__REQUIRES__CONFIG",
        "RECONF_EVENT__ENABLES__OP",
        "OP__TRIGGERS_RECONF__RECONF_EVENT",
    ),
    "w_agv_transport": (
        "W_EVENT__ENABLES__OP",
        "OP__RELEASES_WORKPIECE_TO__W_EVENT",
    ),
    "f_agv_delivery": ("F_EVENT__ENABLES__OP",),
    "synchronization": (
        "OP__PRECEDES__OP",
        "OP__PRODUCT_NEXT__OP",
        "OP__ISLAND_NEXT__OP",
        "W_EVENT__ENABLES__OP",
        "F_EVENT__ENABLES__OP",
        "RECONF_EVENT__ENABLES__OP",
        "OP__TRIGGERS_RECONF__RECONF_EVENT",
        "OP__RELEASES_WORKPIECE_TO__W_EVENT",
    ),
}
DIRECTIONS = ("outgoing_from_target", "incoming_to_target")
SYNCHRONIZATION_RELATIONS = BOUNDARY_FAMILIES["synchronization"]


@dataclass(frozen=True)
class Phase6NOutput:
    advantage: torch.Tensor
    beats_fallback_logit: torch.Tensor
    candidate_embeddings: torch.Tensor


def _node_offsets(node_hidden: Mapping[str, torch.Tensor]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for node_type in NODE_TYPE_ORDER:
        offsets[node_type] = offset
        offset += len(node_hidden[node_type])
    return offsets


def _unique_boundary_values(
    node_hidden: Mapping[str, torch.Tensor],
    batch: NIBatch,
    membership: torch.Tensor,
    relation_keys: Sequence[str],
    direction: str,
    *,
    require_binding: bool,
    incident_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique (candidate, external node) boundary memberships."""
    if direction not in (*DIRECTIONS, "both"):
        raise ValueError(f"unknown boundary direction: {direction}")
    offsets = _node_offsets(node_hidden)
    all_hidden = torch.cat([node_hidden[key] for key in NODE_TYPE_ORDER], dim=0)
    action_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    directions = DIRECTIONS if direction == "both" else (direction,)
    for selected_direction in directions:
        for relation_key in relation_keys:
            edge = batch.edges[relation_key]
            if edge.spec.derived_reverse or edge.index.shape[1] == 0:
                continue
            outgoing = selected_direction == "outgoing_from_target"
            incident_type = edge.spec.source_type if outgoing else edge.spec.target_type
            external_type = edge.spec.target_type if outgoing else edge.spec.source_type
            if incident_type != "OP":
                continue
            incident = edge.index[0 if outgoing else 1]
            external = edge.index[1 if outgoing else 0]
            same_state = batch.action_to_state[:, None].eq(
                batch.node_batch_index["OP"][incident][None, :]
            )
            matches = same_state & membership[:, incident]
            if external_type == "OP":
                matches &= ~membership[:, external]
            if incident_mask is not None:
                matches &= incident_mask[incident][None, :]
            if require_binding:
                try:
                    binding_index = edge.spec.edge_feature_names.index("binding_indicator")
                except ValueError as error:
                    raise ValueError(
                        f"synchronization relation lacks binding_indicator: {relation_key}"
                    ) from error
                matches &= edge.features[:, binding_index].eq(1)[None, :]
            actions, edge_indices = matches.nonzero(as_tuple=True)
            if len(actions):
                action_parts.append(actions)
                node_parts.append(external[edge_indices] + offsets[external_type])
    if not action_parts:
        return (
            all_hidden.new_empty((0, all_hidden.shape[1])),
            torch.empty(0, dtype=torch.long, device=all_hidden.device),
        )
    actions = torch.cat(action_parts)
    nodes = torch.cat(node_parts)
    codes = actions * len(all_hidden) + nodes
    unique_codes = torch.unique(codes, sorted=True)
    unique_actions = torch.div(unique_codes, len(all_hidden), rounding_mode="floor")
    unique_nodes = unique_codes.remainder(len(all_hidden))
    return all_hidden[unique_nodes], unique_actions


def _pooled(
    values: torch.Tensor,
    action_indices: torch.Tensor,
    action_count: int,
    scalar: torch.Tensor,
) -> torch.Tensor:
    if len(values):
        mean = segment_mean(values, action_indices, action_count)
        maximum = segment_max(values, action_indices, action_count)
    else:
        mean = values.new_zeros((action_count, values.shape[-1]))
        maximum = values.new_zeros((action_count, values.shape[-1]))
    return torch.cat([mean, maximum, scalar.unsqueeze(-1)], dim=1)


class CandidateTargetPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        operation_hidden: torch.Tensor,
        graph_embedding: torch.Tensor,
        batch: NIBatch,
    ) -> torch.Tensor:
        selected = operation_hidden[batch.target_operation_indices]
        action_index = batch.target_action_index
        mean = segment_mean(selected, action_index, batch.action_count)
        maximum = segment_max(selected, action_index, batch.action_count)
        query = self.attention_query(graph_embedding[batch.action_to_state])
        keys = self.attention_key(selected)
        logits = (keys * query[action_index]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        weights = segment_softmax(logits, action_index)
        attention = selected.new_zeros((batch.action_count, self.hidden_dim))
        attention.index_add_(0, action_index, weights.unsqueeze(-1) * selected)
        sizes = torch.bincount(action_index, minlength=batch.action_count).to(selected.dtype)
        op_counts = (batch.node_ptr["OP"][1:] - batch.node_ptr["OP"][:-1]).to(
            selected.dtype
        )
        normalized_size = sizes / op_counts[batch.action_to_state].clamp_min(1)
        return self.projection(torch.cat(
            [mean, maximum, attention, normalized_size.unsqueeze(-1)], dim=1
        ))


class CandidateConditionedPooler(nn.Module):
    def __init__(
        self,
        categorical_sizes: tuple[int, int, int],
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_pool = CandidateTargetPool(hidden_dim, dropout)
        self.boundary_projections = nn.ModuleDict({
            f"{family}__{direction}": nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, 32), nn.GELU()
            )
            for family in BOUNDARY_FAMILIES
            for direction in DIRECTIONS
        })
        self.conditioned_projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(2 * hidden_dim + 1, 32), nn.GELU())
            for name in ("critical", "bottleneck", "critical_sync")
        })
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(size, 4) for size in categorical_sizes
        ])
        self.context_projection = nn.Sequential(nn.Linear(22, 32), nn.GELU())
        self.candidate_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        node_hidden: Mapping[str, torch.Tensor],
        graph_embedding: torch.Tensor,
        batch: NIBatch,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        critical_operation_mask: torch.Tensor,
        bottleneck_operation_mask: torch.Tensor,
    ) -> torch.Tensor:
        if categorical.shape != (batch.action_count, 3):
            raise ValueError("Phase 6N requires three categorical columns per candidate")
        if numeric.shape != (batch.action_count, 10):
            raise ValueError("Phase 6N requires ten numeric columns per candidate")
        op_count = len(node_hidden["OP"])
        if critical_operation_mask.shape != (op_count,):
            raise ValueError("critical operation mask is misaligned")
        if bottleneck_operation_mask.shape != (op_count,):
            raise ValueError("bottleneck operation mask is misaligned")

        membership = torch.zeros(
            (batch.action_count, op_count), dtype=torch.bool,
            device=node_hidden["OP"].device,
        )
        membership[batch.target_action_index, batch.target_operation_indices] = True
        state_node_counts = sum(
            torch.bincount(
                batch.node_batch_index[node_type], minlength=batch.state_count
            )
            for node_type in NODE_TYPE_ORDER
        ).to(node_hidden["OP"].dtype)
        action_denominator = torch.log1p(
            state_node_counts[batch.action_to_state].clamp_min(1)
        )

        target = self.target_pool(node_hidden["OP"], graph_embedding, batch)
        boundary_parts = []
        for family, relation_keys in BOUNDARY_FAMILIES.items():
            for direction in DIRECTIONS:
                values, actions = _unique_boundary_values(
                    node_hidden,
                    batch,
                    membership,
                    relation_keys,
                    direction,
                    require_binding=family == "synchronization",
                )
                counts = torch.bincount(actions, minlength=batch.action_count).to(
                    node_hidden["OP"].dtype
                )
                normalized_count = torch.log1p(counts) / action_denominator
                boundary_parts.append(self.boundary_projections[
                    f"{family}__{direction}"
                ](_pooled(values, actions, batch.action_count, normalized_count)))

        conditioned_parts = []
        selected_hidden = node_hidden["OP"][batch.target_operation_indices]
        for name, mask in (
            ("critical", critical_operation_mask),
            ("bottleneck", bottleneck_operation_mask),
        ):
            retained = mask[batch.target_operation_indices]
            actions = batch.target_action_index[retained]
            values = selected_hidden[retained]
            presence = torch.bincount(actions, minlength=batch.action_count).gt(0).to(
                selected_hidden.dtype
            )
            conditioned_parts.append(self.conditioned_projections[name](
                _pooled(values, actions, batch.action_count, presence)
            ))

        values, actions = _unique_boundary_values(
            node_hidden,
            batch,
            membership,
            SYNCHRONIZATION_RELATIONS,
            "both",
            require_binding=True,
            incident_mask=critical_operation_mask,
        )
        presence = torch.bincount(actions, minlength=batch.action_count).gt(0).to(
            selected_hidden.dtype
        )
        conditioned_parts.append(self.conditioned_projections["critical_sync"](
            _pooled(values, actions, batch.action_count, presence)
        ))

        categories = torch.cat([
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.categorical_embeddings)
        ], dim=1)
        cheap = self.context_projection(torch.cat([categories, numeric], dim=1))
        combined = torch.cat([
            target,
            *boundary_parts,
            *conditioned_parts,
            graph_embedding[batch.action_to_state],
            cheap,
        ], dim=1)
        if combined.shape[1] != 768:
            raise RuntimeError(f"Phase 6N candidate width drifted: {combined.shape[1]}")
        return self.candidate_projection(combined)


class CandidateConditionedCSGModel(nn.Module):
    family = FAMILY

    def __init__(
        self,
        base: CSGTargetSetScorer,
        categorical_sizes: tuple[int, int, int],
        *,
        family: str,
    ) -> None:
        super().__init__()
        if family != FAMILY:
            raise ValueError(f"unsupported Phase 6N family: {family}")
        if len(base.state_encoder.layers) != 2:
            raise ValueError("Phase 6N requires the frozen two-block Phase 6F encoder")
        self.state_encoder = base.state_encoder
        self.pooler = CandidateConditionedPooler(
            categorical_sizes,
            hidden_dim=base.config.hidden_dim,
            dropout=base.config.dropout,
        )
        self.fallback_fusion = nn.Sequential(
            nn.Linear(3 * base.config.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(base.config.dropout),
            nn.Linear(256, base.config.hidden_dim),
            nn.LayerNorm(base.config.hidden_dim),
        )

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(base.config.hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(base.config.dropout),
                nn.Linear(64, 1),
            )

        self.continuation_advantage_head = head()
        self.beats_fallback_head = head()
        for parameter in self.state_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.state_encoder.layers[-1].parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "CandidateConditionedCSGModel":
        super().train(mode)
        self.state_encoder.eval()
        if mode:
            self.state_encoder.layers[-1].train()
        self.pooler.train(mode)
        self.fallback_fusion.train(mode)
        self.continuation_advantage_head.train(mode)
        self.beats_fallback_head.train(mode)
        return self

    def forward(
        self,
        batch: NIBatch,
        *,
        fallback_action_indices: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        critical_operation_mask: torch.Tensor,
        bottleneck_operation_mask: torch.Tensor,
    ) -> Phase6NOutput:
        node_hidden, graph_embedding = self.state_encoder(batch)
        candidate = self.pooler(
            node_hidden,
            graph_embedding,
            batch,
            categorical,
            numeric,
            critical_operation_mask,
            bottleneck_operation_mask,
        )
        fallback = candidate[fallback_action_indices][batch.action_to_state]
        fused = self.fallback_fusion(torch.cat(
            [candidate, fallback, candidate - fallback], dim=1
        ))
        return Phase6NOutput(
            self.continuation_advantage_head(fused).squeeze(-1),
            self.beats_fallback_head(fused).squeeze(-1),
            candidate,
        )

    def parameter_counts(self) -> tuple[int, int]:
        return (
            sum(parameter.numel() for parameter in self.parameters()),
            sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
        )


def candidate_conditioned_loss(
    advantage_prediction: torch.Tensor,
    beats_fallback_logit: torch.Tensor,
    advantage_target: torch.Tensor,
    beats_fallback_target: torch.Tensor,
    action_ptr: torch.Tensor,
    *,
    pair_gap_scale: float,
    huber_delta: float,
    pairwise_weight: float = 1.0,
    listnet_weight: float = 0.75,
    advantage_huber_weight: float = 0.5,
    beats_bce_weight: float = 0.25,
    gap_weight_clip: tuple[float, float] = (0.25, 4.0),
) -> dict[str, torch.Tensor]:
    tensors = (
        advantage_prediction,
        beats_fallback_logit,
        advantage_target,
        beats_fallback_target,
    )
    if any(tensor.ndim != 1 for tensor in tensors) or len({len(x) for x in tensors}) != 1:
        raise ValueError("Phase 6N loss inputs must be aligned vectors")
    if pair_gap_scale <= 0 or huber_delta <= 0:
        raise ValueError("Phase 6N objective scales must be positive")
    if action_ptr.ndim != 1 or action_ptr[0] != 0 or action_ptr[-1] != len(tensors[0]):
        raise ValueError("invalid Phase 6N action pointers")
    terms: dict[str, list[torch.Tensor]] = {
        "pairwise_loss": [],
        "listnet_loss": [],
        "advantage_huber_loss": [],
        "beats_fallback_bce_loss": [],
    }
    pair_count = 0
    for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist()):
        prediction = advantage_prediction[start:stop]
        target = advantage_target[start:stop]
        prediction_z = (prediction - prediction.mean()) / prediction.std(
            unbiased=False
        ).clamp_min(1e-6)
        target_z = (target - target.mean()) / target.std(unbiased=False).clamp_min(1e-6)
        gaps = target.unsqueeze(1) - target.unsqueeze(0)
        better = gaps > 1e-12
        pair_count += int(better.sum().item())
        if bool(better.any()):
            predicted_gaps = prediction_z.unsqueeze(1) - prediction_z.unsqueeze(0)
            weights = (gaps.abs() / pair_gap_scale).clamp(*gap_weight_clip)
            pairwise = (weights[better] * F.softplus(-predicted_gaps[better])).mean()
        else:
            pairwise = prediction.sum() * 0.0
        terms["pairwise_loss"].append(pairwise)
        terms["listnet_loss"].append(-(
            torch.softmax(target_z, dim=0) * torch.log_softmax(prediction_z, dim=0)
        ).sum())
        terms["advantage_huber_loss"].append(F.huber_loss(
            prediction, target, delta=huber_delta
        ))
        terms["beats_fallback_bce_loss"].append(F.binary_cross_entropy_with_logits(
            beats_fallback_logit[start:stop], beats_fallback_target[start:stop]
        ))
    means = {name: torch.stack(values).mean() for name, values in terms.items()}
    total = (
        pairwise_weight * means["pairwise_loss"]
        + listnet_weight * means["listnet_loss"]
        + advantage_huber_weight * means["advantage_huber_loss"]
        + beats_bce_weight * means["beats_fallback_bce_loss"]
    )
    return {"loss": total, **means, "pair_count": torch.as_tensor(pair_count)}
