"""Conditional J3 interaction over the unchanged J2 encoder and heads."""

from __future__ import annotations

import torch
from torch import nn

from .phase6j_caur_model import CAURModel, CAUROutput


class FallbackRelativeInteraction(nn.Module):
    """Rank-eight interaction of state, candidate, fallback and origin."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.action = nn.Linear(embedding_dim, 8, bias=False)
        self.state = nn.Linear(embedding_dim, 8, bias=False)
        self.origin = nn.Linear(12, 8, bias=False)
        self.output = nn.Linear(8, embedding_dim, bias=False)

    def forward(self, action, fallback, state, origin):
        difference = torch.tanh(self.action(action - fallback))
        context = torch.tanh(self.state(state) + self.origin(origin))
        return self.output(difference * context)


class RelationalCAURModel(CAURModel):
    def __init__(self, base, categorical_sizes: tuple[int, int, int]) -> None:
        super().__init__(base, categorical_sizes, family="J2_CONT_LASTBLOCK")
        self.family = "J3_CONT_RELATIONAL"
        self.interaction = FallbackRelativeInteraction(base.config.hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.base.state_encoder.layers[-1].train()
            self.base.action_encoder.projection.train()
        self.interaction.train(mode)
        return self

    def forward(self, batch, *, fallback_action_indices, categorical, numeric):
        nodes, graph = self.base.state_encoder(batch)
        action = self.base.action_encoder(nodes["OP"], graph, batch)
        fallback = action[fallback_action_indices][batch.action_to_state]
        origin = torch.cat([
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.heads.categorical_embeddings)
        ], dim=1)
        action = action + self.interaction(
            action, fallback, graph[batch.action_to_state], origin
        )
        advantage, beats, immediate = self.heads(
            action, batch.action_to_state, fallback_action_indices, categorical, numeric
        )
        return CAUROutput(advantage, beats, immediate, action)
