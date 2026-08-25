"""Integrated RT-HGT encoder, autoregressive policy, and value model."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from rcias_clgri.env.insertion_decoder import Action

from .config import ModelConfig
from .policy import AutoregressivePolicy
from .rt_hgt import RTHGTEncoder
from .tensorizer import GraphTensor, GraphTensorizer
from .value import ValueHead


class RCIASNeuralModel(nn.Module):
    def __init__(self, tensorizer: GraphTensorizer, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = RTHGTEncoder(
            tensorizer.node_input_dims, tensorizer.relation_specs, self.config
        )
        self.policy = AutoregressivePolicy(
            self.config.embedding_dim, tensorizer.candidate_input_dims
        )
        self.value = ValueHead(self.config.embedding_dim)

    def encode(self, graph: GraphTensor) -> Mapping[str, torch.Tensor]:
        return self.encoder(graph)

    def action_losses(self, graph: GraphTensor, action: Action) -> dict[str, torch.Tensor]:
        return self.policy.action_losses(graph, self.encode(graph), action)

    def greedy_action(self, graph: GraphTensor) -> Action:
        return self.policy.greedy_action(graph, self.encode(graph))

    def forward(self, graph: GraphTensor) -> torch.Tensor:
        return self.value(self.encode(graph))
