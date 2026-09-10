"""Enhanced compact relation-aware mean-aggregation CSG encoder."""
import torch
from torch import nn

from rcias_clgri.csg.schema import EDGE_TYPE_ORDER, NODE_TYPE_ORDER
from rcias_ngas.csg.revised_features import EDGE_FEATURE_DIM, NODE_DIM


class CompactRelationalEncoder(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.input = nn.Linear(NODE_DIM, hidden)
        self.node_type = nn.Embedding(len(NODE_TYPE_ORDER), hidden)
        self.relation = nn.Embedding(2 * len(EDGE_TYPE_ORDER), hidden)
        self.edge = nn.Linear(EDGE_FEATURE_DIM, hidden)
        self.message = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.update = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
            for _ in range(layers))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(
            self.input(batch['x']) + self.node_type(batch['types']))
        source, target = batch['edge_index']
        relation = self.relation(batch['relations']) + self.edge(batch['edge_features'])
        degree = hidden.new_zeros(len(hidden)).index_add_(
            0, target, hidden.new_ones(len(target))).clamp_min(1).unsqueeze(1)
        for message, update in zip(self.message, self.update):
            values = torch.nn.functional.gelu(message(hidden[source]) + relation)
            pooled = torch.zeros_like(hidden).index_add_(0, target, values) / degree
            hidden = hidden + update(torch.cat((hidden, pooled), dim=1))
        return hidden
