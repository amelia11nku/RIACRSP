"""Exact shared frozen-encoder inference for the three-seed J1 ensemble."""

from __future__ import annotations

import torch
from torch import nn


class SharedFrozenCAUREnsemble(nn.Module):
    """Evaluate the identical J1 base once and retain all three learned heads."""

    def __init__(self, models):
        super().__init__()
        if len(models) != 3 or any(model.family != "J1_CONT_FROZEN" for model in models):
            raise ValueError("shared frozen inference requires exactly three J1 seeds")
        reference = models[0].base.state_dict()
        for model in models:
            if any(p.requires_grad for p in model.base.parameters()):
                raise ValueError("shared encoder must be frozen")
            state = model.base.state_dict()
            if state.keys() != reference.keys() or any(not torch.equal(state[k], reference[k]) for k in state):
                raise ValueError("J1 seed encoders differ")
        self.base = models[0].base
        self.heads = nn.ModuleList(model.heads for model in models)
        self.eval()

    def forward(self, batch, *, fallback_action_indices, categorical, numeric):
        nodes, graph = self.base.state_encoder(batch)
        action = self.base.action_encoder(nodes["OP"], graph, batch)
        outputs = [head(action, batch.action_to_state, fallback_action_indices, categorical, numeric)
                   for head in self.heads]
        return tuple(torch.stack([output[index] for output in outputs], dim=0) for index in range(3))
