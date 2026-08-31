"""State-based heterogeneous batching for offline Neural Improvement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from rcias_clgri.csg.schema import NODE_TYPE_ORDER

from .dataset import NIStateSample
from .tensorize import NIEdgeTensor


@dataclass(frozen=True)
class NIBatch:
    tensor_schema_hash: str
    state_ids: tuple[str, ...]
    instance_ids: tuple[str, ...]
    graph_hashes: tuple[str, ...]
    node_features: Mapping[str, torch.Tensor]
    node_batch_index: Mapping[str, torch.Tensor]
    node_ptr: Mapping[str, torch.Tensor]
    edges: Mapping[str, NIEdgeTensor]
    graph_numeric: torch.Tensor
    graph_categorical: torch.Tensor
    action_to_state: torch.Tensor
    action_ptr: torch.Tensor
    target_operation_indices: torch.Tensor
    target_action_index: torch.Tensor
    rank_better_index: torch.Tensor
    rank_worse_index: torch.Tensor
    utility: torch.Tensor
    positive: torch.Tensor
    rank_within_state: torch.Tensor
    rank_percentile: torch.Tensor
    regret_to_best: torch.Tensor
    top1: torch.Tensor
    top3: torch.Tensor
    target_set_ids: tuple[str, ...]
    arm_family: tuple[str, ...]
    origin_destroy_operator: tuple[str, ...]
    origin_rules: tuple[str, ...]
    origin_families: tuple[str, ...]
    structural_metadata: tuple[Mapping[str, str], ...]

    @property
    def state_count(self) -> int:
        return len(self.state_ids)

    @property
    def action_count(self) -> int:
        return len(self.target_set_ids)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "NIBatch":
        tensor_fields = (
            "graph_numeric", "graph_categorical", "action_to_state", "action_ptr",
            "target_operation_indices", "target_action_index", "utility", "positive",
            "rank_better_index", "rank_worse_index",
            "rank_within_state", "rank_percentile", "regret_to_best", "top1", "top3",
        )
        return replace(
            self,
            node_features={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.node_features.items()
            },
            node_batch_index={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.node_batch_index.items()
            },
            node_ptr={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.node_ptr.items()
            },
            edges={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.edges.items()
            },
            **{
                name: getattr(self, name).to(device, non_blocking=non_blocking)
                for name in tensor_fields
            },
        )

    def tensor_bytes(self) -> int:
        tensors = [
            *self.node_features.values(),
            *self.node_batch_index.values(),
            *self.node_ptr.values(),
            *(item for edge in self.edges.values() for item in (edge.index, edge.features)),
            self.graph_numeric,
            self.graph_categorical,
            self.action_to_state,
            self.action_ptr,
            self.target_operation_indices,
            self.target_action_index,
            self.rank_better_index,
            self.rank_worse_index,
            self.utility,
            self.positive,
            self.rank_within_state,
            self.rank_percentile,
            self.regret_to_best,
            self.top1,
            self.top3,
        ]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _prefix_sum(counts: Sequence[int]) -> list[int]:
    values = [0]
    for count in counts:
        values.append(values[-1] + count)
    return values


def _informative_pairs(
    utility: torch.Tensor,
    action_offsets: Sequence[int],
    top_bottom_count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    better: list[int] = []
    worse: list[int] = []
    for start, stop in zip(action_offsets, action_offsets[1:]):
        local = utility[start:stop]
        order = torch.argsort(local, descending=True, stable=True).tolist()
        pairs: set[tuple[int, int]] = set()
        width = min(top_bottom_count, len(order) // 2)
        for high in order[:width]:
            for low in order[-width:]:
                if float(local[high]) > float(local[low]):
                    pairs.add((start + high, start + low))
        for high, low in zip(order, order[1:]):
            if float(local[high]) > float(local[low]):
                pairs.add((start + high, start + low))
        positive = [index for index in order if float(local[index]) > 0]
        non_positive = [index for index in reversed(order) if float(local[index]) <= 0]
        for high, low in zip(positive, non_positive):
            pairs.add((start + high, start + low))
        for high, low in sorted(pairs):
            better.append(high)
            worse.append(low)
    return torch.tensor(better, dtype=torch.long), torch.tensor(worse, dtype=torch.long)


def batch_state_samples(samples: Sequence[NIStateSample]) -> NIBatch:
    if not samples:
        raise ValueError("cannot create an empty state batch")
    schema_hashes = {sample.graph.tensor_schema_hash for sample in samples}
    if len(schema_hashes) != 1:
        raise ValueError("state batch mixes incompatible tensor schemas")

    node_counts = {
        node_type: [sample.graph.node_features[node_type].shape[0] for sample in samples]
        for node_type in NODE_TYPE_ORDER
    }
    node_offsets = {
        node_type: _prefix_sum(counts) for node_type, counts in node_counts.items()
    }
    node_features = {
        node_type: torch.cat(
            [sample.graph.node_features[node_type] for sample in samples], dim=0
        )
        for node_type in NODE_TYPE_ORDER
    }
    node_batch_index = {
        node_type: torch.cat([
            torch.full((count,), state_index, dtype=torch.long)
            for state_index, count in enumerate(node_counts[node_type])
        ])
        for node_type in NODE_TYPE_ORDER
    }
    node_ptr = {
        node_type: torch.tensor(offsets, dtype=torch.long)
        for node_type, offsets in node_offsets.items()
    }

    relation_keys = tuple(samples[0].graph.edges)
    if any(tuple(sample.graph.edges) != relation_keys for sample in samples[1:]):
        raise ValueError("state batch mixes incompatible relation schemas")
    edges: dict[str, NIEdgeTensor] = {}
    for relation_key in relation_keys:
        spec = samples[0].graph.edges[relation_key].spec
        indices = []
        features = []
        for state_index, sample in enumerate(samples):
            edge = sample.graph.edges[relation_key]
            if edge.spec != spec:
                raise ValueError(f"relation spec mismatch for {relation_key}")
            offset = torch.tensor(
                [
                    node_offsets[spec.source_type][state_index],
                    node_offsets[spec.target_type][state_index],
                ],
                dtype=torch.long,
            ).view(2, 1)
            indices.append(edge.index + offset)
            features.append(edge.features)
        edges[relation_key] = NIEdgeTensor(
            spec,
            torch.cat(indices, dim=1),
            torch.cat(features, dim=0),
        )

    action_counts = [sample.actions.action_count for sample in samples]
    action_offsets = _prefix_sum(action_counts)
    target_indices = []
    target_action_indices = []
    for state_index, sample in enumerate(samples):
        target_indices.append(
            sample.actions.target_operation_indices + node_offsets["OP"][state_index]
        )
        target_action_indices.append(
            sample.actions.target_action_index + action_offsets[state_index]
        )

    def action_tensor(name: str) -> torch.Tensor:
        return torch.cat([getattr(sample.actions, name) for sample in samples])

    def action_metadata(name: str) -> tuple[str, ...]:
        return tuple(
            value for sample in samples for value in getattr(sample.actions, name)
        )

    utility = action_tensor("utility")
    rank_better, rank_worse = _informative_pairs(utility, action_offsets)
    return NIBatch(
        tensor_schema_hash=next(iter(schema_hashes)),
        state_ids=tuple(sample.graph.state_id for sample in samples),
        instance_ids=tuple(sample.graph.instance_id for sample in samples),
        graph_hashes=tuple(sample.graph.graph_hash for sample in samples),
        node_features=node_features,
        node_batch_index=node_batch_index,
        node_ptr=node_ptr,
        edges=edges,
        graph_numeric=torch.stack([sample.graph.graph_numeric for sample in samples]),
        graph_categorical=torch.stack(
            [sample.graph.graph_categorical for sample in samples]
        ),
        action_to_state=torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long),
            torch.tensor(action_counts, dtype=torch.long),
        ),
        action_ptr=torch.tensor(action_offsets, dtype=torch.long),
        target_operation_indices=torch.cat(target_indices),
        target_action_index=torch.cat(target_action_indices),
        rank_better_index=rank_better,
        rank_worse_index=rank_worse,
        utility=utility,
        positive=action_tensor("positive"),
        rank_within_state=action_tensor("rank_within_state"),
        rank_percentile=action_tensor("rank_percentile"),
        regret_to_best=action_tensor("regret_to_best"),
        top1=action_tensor("top1"),
        top3=action_tensor("top3"),
        target_set_ids=action_metadata("target_set_ids"),
        arm_family=action_metadata("arm_family"),
        origin_destroy_operator=action_metadata("origin_destroy_operator"),
        origin_rules=action_metadata("origin_rules"),
        origin_families=action_metadata("origin_families"),
        structural_metadata=tuple(sample.structural_metadata for sample in samples),
    )
