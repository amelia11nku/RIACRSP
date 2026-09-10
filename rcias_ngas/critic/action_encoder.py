"""Size/target/repair interactions before the joint value head."""
import torch
from torch import nn
from rcias_ngas.csg.features import PROVENANCE_DIM


class JointActionEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.size = nn.Embedding(3, hidden)
        self.repair = nn.Embedding(5, hidden)
        self.target = nn.Sequential(nn.Linear(hidden + PROVENANCE_DIM + 1, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(7 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, global_hidden, operation_hidden, batch):
        count = batch['membership'].sum(1, keepdim=True).clamp_min(1)
        pooled = batch['membership'] @ operation_hidden / count
        target = self.target(torch.cat((pooled, batch['provenance'], count / len(operation_hidden)), dim=1))
        size, repair = self.size(batch['sizes']), self.repair(batch['repairs'])
        state = global_hidden.expand(len(size), -1)
        return self.fusion(torch.cat((state, size, target, repair, size * target, target * repair, size * repair), dim=1))
