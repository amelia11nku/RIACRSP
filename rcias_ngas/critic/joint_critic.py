"""One compact CSG encoder and batched Q(G,k,D,R), no flat action classifier."""
import torch
from torch import nn
from rcias_ngas.csg.features import NODE_DIM, NODE_TYPE_ORDER, EDGE_TYPE_ORDER
from .action_encoder import JointActionEncoder


class JointCritic(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.node = nn.Linear(NODE_DIM, hidden)
        self.node_type = nn.Embedding(len(NODE_TYPE_ORDER), hidden)
        self.relation = nn.Embedding(2 * len(EDGE_TYPE_ORDER), hidden)
        self.edge = nn.Linear(2, hidden)
        self.message = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.update = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)) for _ in range(layers))
        self.actions = JointActionEncoder(hidden)
        self.value = nn.Linear(hidden, 1)
        self.beats_fallback = nn.Linear(hidden, 1)

    def forward(self, batch):
        h = torch.nn.functional.gelu(self.node(batch['x']) + self.node_type(batch['types']))
        source, target = batch['edge_index']
        relation = self.relation(batch['relations']) + self.edge(batch['edge_features'])
        degree = h.new_zeros(len(h)).index_add_(0, target, h.new_ones(len(target))).clamp_min(1).unsqueeze(1)
        for message, update in zip(self.message, self.update):
            values = torch.nn.functional.gelu(message(h[source]) + relation)
            pooled = torch.zeros_like(h).index_add_(0, target, values) / degree
            h = h + update(torch.cat((h, pooled), dim=1))
        action = self.actions(h.mean(0, keepdim=True), h[batch['operation_nodes']], batch)
        return {'advantage': self.value(action).squeeze(-1),
                'beats_fallback_logit': self.beats_fallback(action).squeeze(-1)}
