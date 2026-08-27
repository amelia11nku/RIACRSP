"""Sequentially valid stagewise hybrid policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn import GraphTensorizer, RCIASNeuralModel


@dataclass(frozen=True)
class HybridResult:
    makespan: float
    feasible: bool
    actions: tuple[Action, ...]


def _choose(distribution):
    return distribution.argmax()


def hybrid_action(
    prefix_model: RCIASNeuralModel,
    suffix_model: RCIASNeuralModel,
    graph,
    *,
    prefix_stages: int,
) -> Action:
    """Choose O/M/W/F autoregressively, switching models after a prefix."""
    if prefix_stages not in range(5):
        raise ValueError("prefix_stages must be between zero and four")
    with torch.no_grad():
        prefix_hidden = prefix_model.encode(graph)
        suffix_hidden = suffix_model.encode(graph)
        op_model, op_hidden = (
            (prefix_model, prefix_hidden) if prefix_stages >= 1
            else (suffix_model, suffix_hidden)
        )
        operation = _choose(op_model.policy.operation_distribution(graph, op_hidden))
        island_model, island_hidden = (
            (prefix_model, prefix_hidden) if prefix_stages >= 2
            else (suffix_model, suffix_hidden)
        )
        island = _choose(island_model.policy.island_distribution(
            graph, island_hidden, operation
        ))
        w_model, w_hidden = (
            (prefix_model, prefix_hidden) if prefix_stages >= 3
            else (suffix_model, suffix_hidden)
        )
        w_agv = _choose(w_model.policy.w_distribution(
            graph, w_hidden, operation, island
        ))
        f_model, f_hidden = (
            (prefix_model, prefix_hidden) if prefix_stages >= 4
            else (suffix_model, suffix_hidden)
        )
        f_agv = _choose(f_model.policy.f_distribution(
            graph, f_hidden, operation, island, w_agv
        ))
    return Action(operation, island, w_agv, f_agv)


def collect_hybrid_episode(
    prefix_model: RCIASNeuralModel,
    suffix_model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    instance,
    *,
    prefix_stages: int,
    device: torch.device | str,
) -> HybridResult:
    env = RCIASConstructionEnv(instance)
    actions = []
    for _ in range(instance.num_operations):
        graph = tensorizer.tensorize(build_graph_state(instance, env.schedule)).to(device)
        action = hybrid_action(
            prefix_model, suffix_model, graph, prefix_stages=prefix_stages
        )
        env.step(action)
        actions.append(action)
    audit = check_schedule(instance, env.schedule)
    return HybridResult(
        makespan=env.objective().makespan,
        feasible=bool(audit["feasible"]),
        actions=tuple(actions),
    )
