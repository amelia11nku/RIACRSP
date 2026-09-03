"""Generalized RIACRSP event graph shared by DABC and LG_HGA.

The graph is an analysis/search view of a schedule produced by the frozen
decoder.  It never constructs or evaluates a schedule itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from types import MappingProxyType
from typing import Mapping

from rcias_clgri.data.instance import Instance

from .common import DecodedCandidate


TOLERANCE = 1e-8


@dataclass(frozen=True)
class EventNode:
    node_id: str
    kind: str
    start_time: float
    end_time: float
    operation_id: str | None = None
    resource_id: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class EventArc:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class GeneralizedCHDG:
    nodes: Mapping[str, EventNode]
    arcs: tuple[EventArc, ...]
    predecessors: Mapping[str, tuple[str, ...]]
    successors: Mapping[str, tuple[str, ...]]
    topological_order: tuple[str, ...]
    makespan: float


@dataclass(frozen=True)
class CriticalPath:
    node_ids: tuple[str, ...]
    duration: float


@dataclass(frozen=True)
class CriticalIslandBlock:
    island_id: str
    operation_ids: tuple[str, ...]
    reconfiguration_node_ids: tuple[str, ...]


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, EventNode] = {}
        self.arcs: list[EventArc] = []
        self._arc_keys: set[tuple[str, str, str]] = set()

    def node(self, node: EventNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate event node: {node.node_id}")
        if node.duration < -TOLERANCE:
            raise ValueError(f"negative event duration: {node.node_id}")
        self.nodes[node.node_id] = node

    def arc(self, source: str, target: str, relation: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"event arc references an unknown node: {source}->{target}")
        key = (source, target, relation)
        if key not in self._arc_keys:
            self._arc_keys.add(key)
            self.arcs.append(EventArc(source, target, relation))


def _operation_node(operation_id: str) -> str:
    return f"OP:{operation_id}"


def _reconfiguration_node(operation_id: str) -> str:
    return f"RECONFIG:{operation_id}"


def _w_empty_node(task_id: str) -> str:
    return f"W_EMPTY:{task_id}"


def _w_loaded_node(task_id: str) -> str:
    return f"W_LOADED:{task_id}"


def _f_outbound_node(task_id: str) -> str:
    return f"F_OUTBOUND:{task_id}"


def _f_return_node(task_id: str) -> str:
    return f"F_RETURN:{task_id}"


def _topological_order(
    nodes: Mapping[str, EventNode], arcs: tuple[EventArc, ...]
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    predecessor_sets = {node_id: set() for node_id in nodes}
    successor_sets = {node_id: set() for node_id in nodes}
    for arc in arcs:
        predecessor_sets[arc.target].add(arc.source)
        successor_sets[arc.source].add(arc.target)
    indegree = {node_id: len(values) for node_id, values in predecessor_sets.items()}
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for successor in sorted(successor_sets[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
        raise ValueError(f"generalized CHDG is cyclic: {cyclic[:10]}")
    predecessors = {key: tuple(sorted(values)) for key, values in predecessor_sets.items()}
    successors = {key: tuple(sorted(values)) for key, values in successor_sets.items()}
    return tuple(order), predecessors, successors


def _add_unexplained_idle_nodes(builder: _GraphBuilder) -> None:
    """Represent decoder insertion-slot idle only when no dependency explains it."""

    arcs = tuple(builder.arcs)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for arc in arcs:
        predecessors[arc.target].add(arc.source)
    for node_id, node in tuple(builder.nodes.items()):
        if node_id in {"S", "E"}:
            continue
        direct = predecessors[node_id]
        latest = max((builder.nodes[source].end_time for source in direct), default=0.0)
        gap = node.start_time - latest
        if gap <= TOLERANCE:
            continue
        idle_id = f"IDLE:{node_id}"
        builder.node(EventNode(idle_id, "UNEXPLAINED_IDLE", latest, node.start_time))
        anchors = sorted(
            source for source in direct
            if math.isclose(builder.nodes[source].end_time, latest, abs_tol=TOLERANCE)
        ) or ["S"]
        for source in anchors:
            builder.arc(source, idle_id, "REALIZED_IDLE_AFTER")
        builder.arc(idle_id, node_id, "REALIZED_IDLE_BEFORE")


def build_generalized_chdg(instance: Instance, decoded: DecodedCandidate) -> GeneralizedCHDG:
    """Build and validate the realized precedence/resource event DAG."""

    schedule = decoded.schedule
    if schedule.instance_id != instance.instance_id:
        raise ValueError("decoded schedule and instance identifiers differ")
    builder = _GraphBuilder()
    builder.node(EventNode("S", "SOURCE", 0.0, 0.0))
    builder.node(EventNode("E", "SINK", decoded.makespan, decoded.makespan))

    for operation_id, record in schedule.operation_schedules.items():
        builder.node(EventNode(
            _reconfiguration_node(operation_id),
            "RECONFIGURATION",
            record.reconfiguration_start,
            record.reconfiguration_end,
            operation_id,
            record.island_id,
        ))
        builder.node(EventNode(
            _operation_node(operation_id),
            "OPERATION",
            record.start_time,
            record.completion_time,
            operation_id,
            record.island_id,
        ))

    for vehicle_id, tasks in schedule.w_timelines.items():
        for task in tasks:
            builder.node(EventNode(
                _w_empty_node(task.task_id), "W_EMPTY", task.empty_start, task.empty_arrival,
                task.operation_id, vehicle_id,
            ))
            builder.node(EventNode(
                _w_loaded_node(task.task_id), "W_LOADED", task.loaded_start, task.arrival_time,
                task.operation_id, vehicle_id,
            ))

    for vehicle_id, tasks in schedule.f_timelines.items():
        for task in tasks:
            builder.node(EventNode(
                _f_outbound_node(task.task_id), "F_OUTBOUND", task.departure_wh,
                task.arrival_island, task.operation_id, vehicle_id,
            ))
            builder.node(EventNode(
                _f_return_node(task.task_id), "F_RETURN", task.arrival_island,
                task.return_wh, task.operation_id, vehicle_id,
            ))

    for island_id, sequence in schedule.island_timelines.items():
        previous: str | None = None
        for operation_id in sequence:
            reconfiguration = _reconfiguration_node(operation_id)
            builder.arc(
                _operation_node(previous) if previous is not None else "S",
                reconfiguration,
                "ISLAND_ORDER",
            )
            builder.arc(reconfiguration, _operation_node(operation_id), "CONFIGURATION_READY")
            previous = operation_id

    for operation_id, record in schedule.operation_schedules.items():
        operation = _operation_node(operation_id)
        builder.arc(operation, "E", "MAKESPAN_COMPLETION")
        predecessor = record.product_predecessor
        if predecessor is not None:
            builder.arc(_operation_node(predecessor), operation, "REALIZED_PRODUCT_CHAIN")
        for technological_predecessor in instance.predecessors[operation_id]:
            builder.arc(
                _operation_node(technological_predecessor), operation, "TECHNOLOGICAL_PRECEDENCE"
            )

    for vehicle_id, tasks in schedule.w_timelines.items():
        previous_loaded: str | None = None
        for task in tasks:
            empty = _w_empty_node(task.task_id)
            loaded = _w_loaded_node(task.task_id)
            builder.arc(previous_loaded or "S", empty, "W_RESOURCE_ORDER")
            builder.arc(empty, loaded, "W_EMPTY_BEFORE_LOADED")
            if task.predecessor_op is not None:
                builder.arc(
                    _operation_node(task.predecessor_op), loaded, "WORKPIECE_RELEASE"
                )
            else:
                builder.arc("S", loaded, "WAREHOUSE_RELEASE")
            builder.arc(loaded, _operation_node(task.operation_id), "W_ARRIVAL_READY")
            previous_loaded = loaded

    for vehicle_id, tasks in schedule.f_timelines.items():
        previous_return: str | None = None
        for task in tasks:
            outbound = _f_outbound_node(task.task_id)
            returned = _f_return_node(task.task_id)
            builder.arc(previous_return or "S", outbound, "F_RESOURCE_ORDER")
            builder.arc(outbound, returned, "F_OUTBOUND_BEFORE_RETURN")
            builder.arc(outbound, _operation_node(task.operation_id), "F_ARRIVAL_READY")
            previous_return = returned

    _add_unexplained_idle_nodes(builder)
    arcs = tuple(builder.arcs)
    order, predecessors, successors = _topological_order(builder.nodes, arcs)
    nodes = MappingProxyType(dict(builder.nodes))
    graph = GeneralizedCHDG(
        nodes,
        arcs,
        MappingProxyType(predecessors),
        MappingProxyType(successors),
        order,
        decoded.makespan,
    )
    _validate_realized_times(graph)
    path = critical_path(graph)
    if not math.isclose(path.duration, decoded.makespan, abs_tol=TOLERANCE):
        raise ValueError(
            f"generalized CHDG longest path {path.duration} != makespan {decoded.makespan}"
        )
    return graph


def _longest_distances(graph: GeneralizedCHDG) -> dict[str, float]:
    distances: dict[str, float] = {}
    for node_id in graph.topological_order:
        start = max((distances[source] for source in graph.predecessors[node_id]), default=0.0)
        distances[node_id] = start + max(0.0, graph.nodes[node_id].duration)
    return distances


def _validate_realized_times(graph: GeneralizedCHDG) -> None:
    for arc in graph.arcs:
        source = graph.nodes[arc.source]
        target = graph.nodes[arc.target]
        if source.end_time > target.start_time + TOLERANCE:
            raise ValueError(
                f"event dependency violates realized time: {arc.source}->{arc.target}"
            )
    distances = _longest_distances(graph)
    for node_id, node in graph.nodes.items():
        if not math.isclose(distances[node_id], node.end_time, abs_tol=TOLERANCE):
            raise ValueError(
                f"event path time {distances[node_id]} != realized end {node.end_time}: {node_id}"
            )


def critical_path(
    graph: GeneralizedCHDG,
    rng: random.Random | None = None,
) -> CriticalPath:
    """Select one longest S-to-E path, randomly resolving ties when requested."""

    distances = _longest_distances(graph)
    current = "E"
    reversed_path = [current]
    while current != "S":
        node = graph.nodes[current]
        target_start = distances[current] - max(0.0, node.duration)
        candidates = [
            source for source in graph.predecessors[current]
            if math.isclose(distances[source], target_start, abs_tol=TOLERANCE)
        ]
        if not candidates:
            raise ValueError(f"no critical predecessor for {current}")
        candidates.sort()
        current = rng.choice(candidates) if rng is not None else candidates[0]
        reversed_path.append(current)
    return CriticalPath(tuple(reversed(reversed_path)), distances["E"])


def critical_island_blocks(
    decoded: DecodedCandidate,
    path: CriticalPath,
) -> tuple[CriticalIslandBlock, ...]:
    """Return maximal realized island runs whose operations lie on ``path``."""

    critical_operations = {
        node_id.removeprefix("OP:") for node_id in path.node_ids if node_id.startswith("OP:")
    }
    path_position = {node_id: index for index, node_id in enumerate(path.node_ids)}
    blocks: list[CriticalIslandBlock] = []
    for island_id, sequence in decoded.schedule.island_timelines.items():
        run: list[str] = []
        for operation_id in (*sequence, None):
            if operation_id is not None and operation_id in critical_operations:
                run.append(operation_id)
                continue
            if run:
                blocks.append(CriticalIslandBlock(
                    island_id,
                    tuple(run),
                    tuple(_reconfiguration_node(item) for item in run),
                ))
                run = []
    return tuple(sorted(
        blocks,
        key=lambda block: min(path_position[_operation_node(op)] for op in block.operation_ids),
    ))
