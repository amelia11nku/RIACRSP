"""Batched target/boundary encoder retaining explicit (k,D,R) interactions."""
import torch
from torch import nn

from rcias_ngas.csg.revised_features import BOUNDARY_DIM
from rcias_ngas.csg.revised_schema import PROVENANCE_DIM


class RevisedJointActionEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.size = nn.Embedding(3, hidden)
        self.repair = nn.Embedding(5, hidden)
        self.boundary_relation = nn.Sequential(nn.Linear(BOUNDARY_DIM, hidden), nn.GELU())
        self.target = nn.Sequential(
            nn.Linear(4 * hidden + PROVENANCE_DIM + 2, 2 * hidden),
            nn.GELU(), nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))
        self.fusion = nn.Sequential(
            nn.Linear(7 * hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, state_hidden: torch.Tensor, node_hidden: torch.Tensor,
                batch: dict[str, torch.Tensor]) -> torch.Tensor:
        operation_hidden = node_hidden[batch['operation_nodes']]
        membership = batch['membership']
        count = membership.sum(1, keepdim=True).clamp_min(1)
        target_mean = membership @ operation_hidden / count
        expanded = operation_hidden.unsqueeze(0).expand(len(membership), -1, -1)
        target_max = expanded.masked_fill(~membership.bool().unsqueeze(-1), -torch.inf).amax(1)
        boundary_count = batch['boundary_membership'].sum(1, keepdim=True).clamp_min(1)
        boundary_mean = batch['boundary_membership'] @ node_hidden / boundary_count
        boundary_relation = self.boundary_relation(batch['boundary_stats'])
        target = self.target(torch.cat((
            target_mean, target_max, boundary_mean, boundary_relation,
            batch['provenance'], count / len(operation_hidden),
            batch['target_critical_overlap'],
        ), dim=1))
        size, repair = self.size(batch['sizes']), self.repair(batch['repairs'])
        state = state_hidden.expand(len(size), -1)
        return self.fusion(torch.cat((
            state, size, target, repair,
            size * target, target * repair, size * repair,
        ), dim=1))
