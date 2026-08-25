"""Policy rollouts through the frozen deterministic construction environment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import torch

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.nn.model import RCIASNeuralModel
from rcias_clgri.nn.tensorizer import GraphTensorizer

from .buffer import RolloutTransition
from .reward import horizon_scale, telescoping_makespan_reward


@dataclass(frozen=True)
class RolloutEpisode:
    instance_id: str
    actions: tuple[Action, ...]
    transitions: tuple[RolloutTransition, ...]
    makespan: float
    normalized_return: float
    reward_sum: float
    reward_scale: float
    feasible: bool
    stage_entropy: Mapping[str, float]
    joint_entropy: float
    normalized_entropy: float
    action_statistics: Mapping[str, object]
    timing: Mapping[str, float]


def collect_episode(
    model: RCIASNeuralModel,
    tensorizer: GraphTensorizer,
    instance: Instance,
    *,
    device: torch.device | str,
    deterministic: bool,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    store_transitions: bool = True,
) -> RolloutEpisode:
    """Construct one complete, independently checked schedule."""

    model.eval()
    env = RCIASConstructionEnv(instance)
    scale = horizon_scale(instance)
    previous_makespan = 0.0
    actions: list[Action] = []
    transitions: list[RolloutTransition] = []
    rewards: list[float] = []
    entropies = {name: [] for name in ("operation", "island", "w", "f")}
    joint_entropies: list[float] = []
    normalized_entropies: list[float] = []
    ready_counts: list[int] = []
    island_counts: list[int] = []
    w_none = 0
    same_configuration = 0
    w_use: Counter[str] = Counter()
    f_use: Counter[str] = Counter()
    timing = {"graph_build": 0.0, "forward": 0.0, "policy_scoring": 0.0, "decoder": 0.0}
    for step_index in range(instance.num_operations):
        started = perf_counter()
        graph_state = build_graph_state(instance, env.schedule)
        graph = tensorizer.tensorize(graph_state)
        graph_device = graph.to(device)
        timing["graph_build"] += perf_counter() - started
        started = perf_counter()
        with torch.no_grad():
            hidden = model.encode(graph_device)
            value = model.value(hidden)
        timing["forward"] += perf_counter() - started
        started = perf_counter()
        with torch.no_grad():
            evaluation = model.policy.sample_action(
                graph_device,
                hidden,
                deterministic=deterministic,
                temperature=temperature,
                generator=generator,
            )
        timing["policy_scoring"] += perf_counter() - started
        action = evaluation.action
        ready_counts.append(len(graph.candidates.ready_operations))
        island_counts.append(evaluation.stage_candidate_counts["island"])
        previous_sequence = env.schedule.island_timelines[action.island_id]
        previous_config = (
            instance.island_data[action.island_id].initial_config
            if not previous_sequence
            else env.schedule.operation_schedules[previous_sequence[-1]].config_id
        )
        same_configuration += int(
            previous_config == instance.operation_data[action.operation_id].required_config
        )
        if action.w_agv_id is None:
            w_none += 1
        else:
            w_use[action.w_agv_id] += 1
        f_use[action.f_agv_id] += 1
        started = perf_counter()
        env.step(action)
        timing["decoder"] += perf_counter() - started
        next_makespan = env.objective().makespan
        reward = telescoping_makespan_reward(previous_makespan, next_makespan, scale)
        previous_makespan = next_makespan
        done = env.done
        rewards.append(reward)
        actions.append(action)
        for name, value_entropy in evaluation.stage_entropies.items():
            entropies[name].append(float(value_entropy.detach().cpu()))
        joint_entropies.append(float(evaluation.joint_entropy.detach().cpu()))
        normalized_entropies.append(
            float(evaluation.active_stage_normalized_entropy.detach().cpu())
        )
        if store_transitions:
            transitions.append(RolloutTransition(
                graph=graph,
                action=action,
                old_joint_log_prob=float(evaluation.joint_log_prob.detach().cpu()),
                old_stage_log_probs={
                    name: float(value_log.detach().cpu())
                    for name, value_log in evaluation.stage_log_probs.items()
                },
                old_value=float(value.detach().cpu()),
                reward=reward,
                done=done,
                instance_id=instance.instance_id,
                step_index=step_index,
            ))
    audit = check_schedule(instance, env.schedule)
    makespan = env.objective().makespan
    reward_sum = float(sum(rewards))
    expected = -makespan / scale
    if abs(reward_sum - expected) > 1e-9:
        raise RuntimeError("dense rewards do not telescope to the final makespan")
    if not audit["feasible"]:
        raise RuntimeError(f"policy produced an infeasible decoder schedule: {audit['violations']}")
    count = max(1, len(actions))
    return RolloutEpisode(
        instance_id=instance.instance_id,
        actions=tuple(actions),
        transitions=tuple(transitions),
        makespan=makespan,
        normalized_return=expected,
        reward_sum=reward_sum,
        reward_scale=scale,
        feasible=True,
        stage_entropy={
            name: sum(values) / max(1, len(values)) for name, values in entropies.items()
        },
        joint_entropy=sum(joint_entropies) / max(1, len(joint_entropies)),
        normalized_entropy=sum(normalized_entropies) / max(1, len(normalized_entropies)),
        action_statistics={
            "mean_ready_operations": sum(ready_counts) / count,
            "mean_eligible_islands": sum(island_counts) / count,
            "w_none_ratio": w_none / count,
            "w_utilization": dict(sorted(w_use.items())),
            "f_utilization": dict(sorted(f_use.items())),
            "same_configuration_selection_ratio": same_configuration / count,
            "reconfiguration_count": env.objective().reconfiguration_count,
        },
        timing=timing,
    )
