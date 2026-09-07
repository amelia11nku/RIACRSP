"""Phase 6K exact eager reconstruction; historical preprocessing is untouched."""
from copy import deepcopy
import math
from typing import Mapping

import torch
from torch import nn

from rcias_clgri.csg.schema import NODE_TYPE_ORDER
from .batching import NIBatch
from .encoder import CSGRelationLayer, CSGStateEncoder, replace_batch_edges
from .action_encoder import TargetSetEncoder, segment_max


def prepare_topology(packed):
    """Outcome-free CPU metadata, once per fresh batch; included in live cost."""
    batch = packed['batch']
    if any(x.device.type != 'cpu' for x in batch.node_features.values()):
        raise ValueError('prepare topology before transfer')
    batch.incoming_relation_counts = {k: v.new_zeros((len(v), 1)) for k, v in batch.node_features.items()}
    for edge in batch.edges.values():
        if edge.index.shape[1]:
            batch.incoming_relation_counts[edge.spec.target_type][torch.unique(edge.index[1])] += 1
    batch.node_counts = {k: torch.bincount(batch.node_batch_index[k], minlength=batch.state_count).to(v.dtype).unsqueeze(-1)
                         for k, v in batch.node_features.items()}
    batch.target_sizes = torch.bincount(batch.target_action_index, minlength=batch.action_count).float()
    ids = list(batch.target_set_ids)
    rank = {name: i for i, name in enumerate(sorted(ids))}
    packed['lexical_rank'] = torch.tensor([rank[x] for x in ids], dtype=torch.long)
    packed['support_tensor'] = torch.as_tensor(packed['supported'], dtype=torch.bool)


def transfer_topology(packed, device):
    batch = packed['batch']
    for name in ('incoming_relation_counts', 'node_counts'):
        setattr(batch, name, {k: v.to(device) for k, v in getattr(batch, name).items()})
    batch.target_sizes = batch.target_sizes.to(device)
    for name in ('lexical_rank', 'support_tensor'):
        packed[name] = packed[name].to(device)


def segment_mean(values, segments, segment_count, counts):
    output = values.new_zeros((segment_count, values.shape[-1]))
    output.index_add_(0, segments, values)
    return output / counts.clamp_min(1)


def segment_softmax(scores: torch.Tensor, segments: torch.Tensor, segment_count: int) -> torch.Tensor:
    if scores.shape[0] == 0:
        return torch.empty_like(scores)
    expanded = segments.view(-1, *([1] * (scores.ndim - 1))).expand_as(scores)
    maxima = torch.full(
        (segment_count, *scores.shape[1:]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    numerator = torch.exp(scores - maxima[segments])
    denominator = torch.zeros_like(maxima)
    denominator.index_add_(0, segments, numerator)
    return numerator / denominator[segments].clamp_min(torch.finfo(scores.dtype).tiny)


class RuntimeRelationLayer(CSGRelationLayer):
    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: NIBatch,
    ) -> dict[str, torch.Tensor]:
        aggregates = {
            node_type: torch.zeros_like(hidden[node_type]) for node_type in NODE_TYPE_ORDER
        }
        relation_counts = batch.incoming_relation_counts
        for relation_key, edge in batch.edges.items():
            if edge.index.shape[1] == 0:
                continue
            source_index, target_index = edge.index
            source = hidden[edge.spec.source_type][source_index]
            target = hidden[edge.spec.target_type][target_index]
            query = self.query[edge.spec.target_type](target)
            key = self.relation_key[relation_key](self.key[edge.spec.source_type](source))
            value = self.relation_value[relation_key](
                self.value[edge.spec.source_type](source)
            )
            if (
                self.use_edge_features
                and edge.spec.edge_feature_names
                and edge.features.shape[1] > 0
            ):
                key = key + self.edge_key[relation_key](edge.features)
                value = value + self.edge_value[relation_key](edge.features)
            query = query.view(-1, self.heads, self.head_dim)
            key = key.view(-1, self.heads, self.head_dim)
            value = value.view(-1, self.heads, self.head_dim)
            scores = (query * key).sum(dim=-1) / math.sqrt(self.head_dim)
            weights = segment_softmax(scores, target_index, hidden[edge.spec.target_type].shape[0])
            messages = (weights.unsqueeze(-1) * value).reshape(-1, self.dim)
            aggregates[edge.spec.target_type].index_add_(0, target_index, messages)

        output = {}
        for node_type in NODE_TYPE_ORDER:
            aggregate = aggregates[node_type] / relation_counts[node_type].clamp_min(1)
            attended = self.attention_norm[node_type](
                hidden[node_type] + self.dropout(self.output[node_type](aggregate))
            )
            output[node_type] = self.feedforward_norm[node_type](
                attended + self.dropout(self.feedforward[node_type](attended))
            )
        return output


class HoistedRuntimeRelationLayer(RuntimeRelationLayer):
    """Reuse node-type Q/K/V projections across all relations in one layer."""

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: NIBatch,
    ) -> dict[str, torch.Tensor]:
        aggregates = {
            node_type: torch.zeros_like(hidden[node_type]) for node_type in NODE_TYPE_ORDER
        }
        relation_counts = batch.incoming_relation_counts
        queries = {node_type: self.query[node_type](hidden[node_type])
                   for node_type in NODE_TYPE_ORDER}
        keys = {node_type: self.key[node_type](hidden[node_type])
                for node_type in NODE_TYPE_ORDER}
        values = {node_type: self.value[node_type](hidden[node_type])
                  for node_type in NODE_TYPE_ORDER}
        for relation_key, edge in batch.edges.items():
            if edge.index.shape[1] == 0:
                continue
            source_index, target_index = edge.index
            query = queries[edge.spec.target_type][target_index]
            key = self.relation_key[relation_key](keys[edge.spec.source_type][source_index])
            value = self.relation_value[relation_key](values[edge.spec.source_type][source_index])
            if (
                self.use_edge_features
                and edge.spec.edge_feature_names
                and edge.features.shape[1] > 0
            ):
                key = key + self.edge_key[relation_key](edge.features)
                value = value + self.edge_value[relation_key](edge.features)
            query = query.view(-1, self.heads, self.head_dim)
            key = key.view(-1, self.heads, self.head_dim)
            value = value.view(-1, self.heads, self.head_dim)
            scores = (query * key).sum(dim=-1) / math.sqrt(self.head_dim)
            weights = segment_softmax(scores, target_index, hidden[edge.spec.target_type].shape[0])
            messages = (weights.unsqueeze(-1) * value).reshape(-1, self.dim)
            aggregates[edge.spec.target_type].index_add_(0, target_index, messages)

        output = {}
        for node_type in NODE_TYPE_ORDER:
            aggregate = aggregates[node_type] / relation_counts[node_type].clamp_min(1)
            attended = self.attention_norm[node_type](
                hidden[node_type] + self.dropout(self.output[node_type](aggregate))
            )
            output[node_type] = self.feedforward_norm[node_type](
                attended + self.dropout(self.feedforward[node_type](attended))
            )
        return output


class RuntimeStateEncoder(CSGStateEncoder):
    def forward(self, batch: NIBatch) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden = {
            node_type: self.input_projection[node_type](batch.node_features[node_type])
            for node_type in NODE_TYPE_ORDER
        }
        if self.config.message_passing:
            filtered_batch = replace_batch_edges(batch, self.relation_keys)
            for layer in self.layers:
                hidden = layer(hidden, filtered_batch)
        else:
            for layer in self.flat_layers:
                hidden = {
                    node_type: hidden[node_type] + layer[node_type](hidden[node_type])
                    for node_type in NODE_TYPE_ORDER
                }
        pooled = [
            segment_mean(
                hidden[node_type], batch.node_batch_index[node_type], batch.state_count, batch.node_counts[node_type]
            )
            for node_type in NODE_TYPE_ORDER
        ]
        graph_numeric = torch.stack(
            (
                torch.log1p(batch.graph_numeric[:, 0].clamp_min(0)),
                batch.graph_numeric[:, 1],
            ),
            dim=1,
        )
        graph_embedding = self.graph_projection(torch.cat(
            [*pooled, graph_numeric, batch.graph_categorical], dim=1
        ))
        return hidden, graph_embedding


class RuntimeTargetSetEncoder(TargetSetEncoder):
    def forward(
        self,
        operation_hidden: torch.Tensor,
        graph_embedding: torch.Tensor,
        batch: NIBatch,
    ) -> torch.Tensor:
        selected = operation_hidden[batch.target_operation_indices]
        action_index = batch.target_action_index
        mean = segment_mean(selected, action_index, batch.action_count, batch.target_sizes.unsqueeze(-1))
        maximum = segment_max(selected, action_index, batch.action_count)
        query = self.attention_query(graph_embedding[batch.action_to_state])
        keys = self.attention_key(selected)
        logits = (
            keys * query[action_index]
        ).sum(dim=-1) / math.sqrt(self.hidden_dim)
        weights = segment_softmax(logits, action_index, batch.action_count)
        attention = selected.new_zeros((batch.action_count, self.hidden_dim))
        attention.index_add_(0, action_index, weights.unsqueeze(-1) * selected)
        target_sizes = batch.target_sizes
        op_counts = (batch.node_ptr["OP"][1:] - batch.node_ptr["OP"][:-1]).to(
            selected.dtype
        )
        normalized_size = (
            target_sizes / op_counts[batch.action_to_state].clamp_min(1)
        ).unsqueeze(-1)
        return self.projection(torch.cat(
            [mean, maximum, attention, graph_embedding[batch.action_to_state], normalized_size],
            dim=-1,
        ))


def eager_reconstruction(e0):
    result = deepcopy(e0)
    state = result.base.state_encoder
    if state.config.relation_mode != 'FULL_CSG' or not state.config.message_passing:
        raise ValueError('Phase 6K requires the frozen full-CSG J1 encoder')
    state.__class__ = RuntimeStateEncoder
    for layer in state.layers:
        layer.__class__ = RuntimeRelationLayer
    result.base.action_encoder.__class__ = RuntimeTargetSetEncoder
    return result.eval()


def hoisted_qkv_reconstruction(e2):
    """Create the single preregistered E4S candidate without changing weights."""
    result = deepcopy(e2)
    layers = result.base.state_encoder.layers
    if not layers or any(layer.__class__ is not RuntimeRelationLayer for layer in layers):
        raise ValueError('E4S requires the exact E1/E2 runtime relation layers')
    for layer in layers:
        layer.__class__ = HoistedRuntimeRelationLayer
    return result.eval()


class DeviceDecision(nn.Module):
    """Three-seed population moments and unchanged Platt/LCB gate on device."""
    REASONS = ('INTERVENE', 'PROBABILITY', 'LCB', 'SUPPORT', 'IMMEDIATE_HARM', 'FALLBACK_ALREADY_BEST', 'NONFINITE')

    def __init__(self, protocol):
        super().__init__()
        calibrator = protocol['calibrator']
        if calibrator['method'] != 'PLATT':
            raise ValueError('this frozen deployment requires Platt calibration')
        self.coefficient = float(calibrator['parameters']['coefficient'])
        self.intercept = float(calibrator['parameters']['intercept'])
        self.p_min = float(protocol['gate']['p_min'])
        self.lcb_lambda = float(protocol['gate']['lcb_lambda'])
        self.delta_min = float(protocol['gate']['delta_min'])
        self.harm_floor = float(protocol['immediate_harm_floor'])

    @staticmethod
    def mean_three(values):
        return (values[0] + values[1] + values[2]) / 3

    def forward(self, outputs, support, lexical_rank, fallback_indices):
        advantage, logits, immediate = outputs
        mean = self.mean_three(advantage)
        centered = advantage - mean.unsqueeze(0)
        std = self.mean_three(centered * centered).sqrt()
        # Phase 6J numpy calibration and scalar gate operate in float64.
        calibrated_logits = (self.coefficient * self.mean_three(logits).double() + self.intercept).clamp(-40, 40)
        probability = 1 / (1 + torch.exp(-calibrated_logits))
        immediate_mean = self.mean_three(immediate)
        winner = torch.where(mean == mean.max(), lexical_rank, torch.full_like(lexical_rank, len(mean))).argmin().reshape(1)
        fallback = fallback_indices.reshape(1)
        lcb = mean.double() - self.lcb_lambda * std.double()
        passes = torch.stack((probability.gather(0, winner) >= self.p_min,
                              lcb.gather(0, winner) > self.delta_min,
                              support.gather(0, winner),
                              immediate_mean.double().gather(0, winner) >= self.harm_floor)).reshape(4)
        reason_codes = torch.arange(1, 5, device=mean.device)
        first_failure = torch.where(passes, torch.full_like(reason_codes, 5), reason_codes).min().reshape(1)
        finite = torch.isfinite(advantage).all() & torch.isfinite(logits).all() & torch.isfinite(immediate).all()
        intervene = passes.all() & (winner != fallback) & finite
        reason = torch.where(intervene, 0, first_failure)
        reason = torch.where(finite, reason, 6)
        selected = torch.where(intervene, winner, fallback)
        record = torch.cat((selected, winner, fallback, intervene.long(), reason))
        return record, (mean, std, probability, immediate_mean, lcb)


def extract_decision(device_result, target_ids):
    """The sole host extraction boundary of the optimized device decision."""
    selected, winner, fallback, intervene, reason = device_result[0].detach().cpu().tolist()
    if reason == 6:
        raise RuntimeError('non-finite J1 prediction')
    return {'selected_target_set_id': target_ids[selected], 'neural_target_set_id': target_ids[winner],
            'fallback_target_set_id': target_ids[fallback], 'intervened': bool(intervene),
            'reason': DeviceDecision.REASONS[reason]}
