"""One fixed E2 strategy: vmap the three independently frozen J1 heads."""
from copy import deepcopy

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


class VectorizedHeads(nn.Module):
    def __init__(self, heads):
        super().__init__()
        if len(heads) != 3 or any(head.training for head in heads):
            raise ValueError('exactly three frozen evaluation heads required')
        parameters, buffers = stack_module_state(list(heads))
        if buffers:
            raise ValueError('frozen J1 heads have no buffers')
        self.parameter_names = tuple(parameters)
        self.stacked_parameters = nn.ParameterList(nn.Parameter(parameters[name], requires_grad=False) for name in self.parameter_names)
        # The meta template describes operators only; stacked_parameters owns the weights.
        object.__setattr__(self, 'template', deepcopy(heads[0]).to('meta'))

    def forward(self, action, action_to_state, fallback, categorical, numeric):
        parameters = dict(zip(self.parameter_names, self.stacked_parameters))
        def one_head(state, action, action_to_state, fallback, categorical, numeric):
            return functional_call(self.template, state, (action, action_to_state, fallback, categorical, numeric))
        return vmap(one_head, in_dims=(0, None, None, None, None, None), randomness='error')(
            parameters, action, action_to_state, fallback, categorical, numeric)


class VectorizedEnsemble(nn.Module):
    def __init__(self, e1):
        super().__init__()
        self.base = deepcopy(e1.base)
        self.heads = VectorizedHeads(e1.heads)
        self.eval()

    def forward(self, batch, *, fallback_action_indices, categorical, numeric):
        nodes, graph = self.base.state_encoder(batch)
        action = self.base.action_encoder(nodes['OP'], graph, batch)
        return self.heads(action, batch.action_to_state, fallback_action_indices, categorical, numeric)
