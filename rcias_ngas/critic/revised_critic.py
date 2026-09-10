"""A1.3R configurable C1/RT-HGT joint critic."""
import torch
from torch import nn

from rcias_ngas.critic.encoders import CompactRelationalEncoder, RTHGTEncoder
from rcias_ngas.critic.pooling import TypeCriticalPooling
from rcias_ngas.critic.revised_action_encoder import RevisedJointActionEncoder


class RevisedJointCritic(nn.Module):
    def __init__(self, encoder_type: str, hidden: int = 64,
                 layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if encoder_type == 'compact_relational':
            self.encoder = CompactRelationalEncoder(hidden, layers)
        elif encoder_type == 'rt_hgt':
            self.encoder = RTHGTEncoder(hidden, layers, heads)
        else:
            raise ValueError(f'Unknown NGAS encoder: {encoder_type}')
        self.encoder_type = encoder_type
        self.pool = TypeCriticalPooling(hidden)
        self.actions = RevisedJointActionEncoder(hidden)
        self.value = nn.Linear(hidden, 1)
        self.beats_fallback = nn.Linear(hidden, 1)

    def encode_state(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encoder(batch)
        return nodes, self.pool(nodes, batch)

    def score_actions(self, nodes: torch.Tensor, state: torch.Tensor,
                      batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        action = self.actions(state, nodes, batch)
        return {
            'advantage': self.value(action).squeeze(-1),
            'beats_fallback_logit': self.beats_fallback(action).squeeze(-1),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nodes, state = self.encode_state(batch)
        return self.score_actions(nodes, state, batch)
