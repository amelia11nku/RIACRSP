"""Type-balanced and critical-aware pooling for one complete CSG."""
import torch
from torch import nn

from rcias_clgri.csg.schema import NODE_TYPE_ORDER
from rcias_ngas.csg.revised_features import MAJOR_NODE_TYPES


class TypeCriticalPooling(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        # Per type: mean, max, and critical mean defaults.
        self.empty = nn.Parameter(torch.zeros(len(MAJOR_NODE_TYPES), 3, hidden))
        self.fusion = nn.Sequential(
            nn.Linear((3 * len(MAJOR_NODE_TYPES) + 1) * hidden, 2 * hidden),
            nn.GELU(), nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pools = []
        for slot, node_type in enumerate(MAJOR_NODE_TYPES):
            mask = batch['types'] == NODE_TYPE_ORDER.index(node_type)
            nodes = hidden[mask]
            if len(nodes):
                mean, maximum = nodes.mean(0), nodes.amax(0)
            else:
                mean, maximum = self.empty[slot, 0], self.empty[slot, 1]
            critical_nodes = hidden[mask & batch['critical_mask']]
            critical = critical_nodes.mean(0) if len(critical_nodes) else self.empty[slot, 2]
            pools.extend((mean, maximum, critical))
        global_pool = hidden.mean(0) if len(hidden) else hidden.new_zeros(hidden.shape[-1])
        return self.fusion(torch.cat((*pools, global_pool))).unsqueeze(0)
