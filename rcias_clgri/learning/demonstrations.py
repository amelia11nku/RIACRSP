"""Expert replay and auditable graph-state demonstration records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.env.rcias_env import RCIASConstructionEnv
from rcias_clgri.graph.builder import GraphState, build_graph_state


@dataclass(frozen=True)
class DemonstrationStep:
    instance_id: str
    source: str
    step_index: int
    graph: GraphState
    action: Action


@dataclass(frozen=True)
class DemonstrationEpisode:
    instance_id: str
    source: str
    steps: tuple[DemonstrationStep, ...]
    makespan: float
    feasible: bool


def replay_demonstration(
    instance: Instance, source: str, actions: Iterable[Action],
) -> DemonstrationEpisode:
    """Record every pre-action graph/mask and replay through the decoder."""

    env = RCIASConstructionEnv(instance)
    steps: list[DemonstrationStep] = []
    for step_index, action in enumerate(actions):
        graph = build_graph_state(instance, env.schedule)
        if not graph.operation_mask.get(action.operation_id, False):
            raise ValueError(f"expert operation violates mask at step {step_index}: {action}")
        if not graph.island_masks[action.operation_id].get(action.island_id, False):
            raise ValueError(f"expert island violates mask at step {step_index}: {action}")
        if action.w_agv_id not in graph.w_masks[(action.operation_id, action.island_id)]:
            raise ValueError(f"expert W action violates mask at step {step_index}: {action}")
        if action.f_agv_id not in graph.f_masks[(action.operation_id, action.island_id)]:
            raise ValueError(f"expert F action violates mask at step {step_index}: {action}")
        steps.append(DemonstrationStep(instance.instance_id, source, step_index, graph, action))
        env.step(action)
    audit = check_schedule(instance, env.schedule)
    return DemonstrationEpisode(
        instance.instance_id, source, tuple(steps), env.objective().makespan, bool(audit["feasible"])
    )


def _key(parts: tuple[object, ...]) -> str:
    return "|".join("NONE" if item is None else str(item) for item in parts)


def graph_record(graph: GraphState) -> dict[str, object]:
    """Serialize the complete learning interface, including masks and candidates."""

    return {
        "node_features": graph.node_features,
        "edges": [
            {
                "source_type": edge.source_type,
                "relation": edge.relation,
                "target_type": edge.target_type,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "features": edge.features,
            }
            for edge in graph.edges
        ],
        "masks": {
            "operation": graph.operation_mask,
            "island": graph.island_masks,
            "w": {_key(key): list(value) for key, value in graph.w_masks.items()},
            "f": {_key(key): list(value) for key, value in graph.f_masks.items()},
        },
        "candidate_features": {
            "operation": graph.operation_candidates,
            "island": {_key(key): value for key, value in graph.island_candidates.items()},
            "w": {_key(key): value for key, value in graph.w_candidates.items()},
            "f": {_key(key): value for key, value in graph.f_candidates.items()},
        },
        "normalization": graph.normalization,
        "probe_stats": graph.probe_stats.to_dict(),
    }


def episode_record(episode: DemonstrationEpisode) -> dict[str, object]:
    return {
        "instance_id": episode.instance_id,
        "source": episode.source,
        "makespan": episode.makespan,
        "feasible": episode.feasible,
        "steps": [
            {
                "step_index": step.step_index,
                "action": asdict(step.action),
                "graph_state": graph_record(step.graph),
            }
            for step in episode.steps
        ],
    }
