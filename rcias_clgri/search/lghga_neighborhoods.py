"""Four critical-path neighborhoods for the LG_HGA RIACRSP adaptation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from rcias_clgri.data.instance import Instance

from .common import Candidate, DecodedCandidate
from .dabc_chdg import build_generalized_chdg, critical_path


NEIGHBORHOODS = ("N1_CTU", "N2_EST", "N3_TOPO", "N4_MMIT")


@dataclass(frozen=True)
class NeighborhoodProposal:
    neighborhood_id: str
    candidate: Candidate
    operation_ids: tuple[str, ...]
    detail: dict[str, object]


def _insert_earlier(
    candidate: Candidate,
    operation_id: str,
    rng: random.Random,
) -> tuple[Candidate, int, int]:
    order = list(candidate.operation_order)
    source = order.index(operation_id)
    if source == 0:
        return candidate, source, source
    target = rng.randrange(source)
    order.insert(target, order.pop(source))
    return Candidate(
        tuple(order), candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
    ), source, target


def _critical_context(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
):
    graph = build_generalized_chdg(instance, decoded)
    path = critical_path(graph, rng)
    operations = tuple(
        graph.nodes[node_id].operation_id
        for node_id in path.node_ids
        if graph.nodes[node_id].kind == "OPERATION"
    )
    return graph, path, tuple(operation for operation in operations if operation is not None)


def _n1_ctu(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
) -> NeighborhoodProposal:
    graph, path, operations = _critical_context(instance, decoded, rng)
    path_tail: dict[str, float] = {}
    tail = 0.0
    for node_id in reversed(path.node_ids):
        tail += max(0.0, graph.nodes[node_id].duration)
        operation_id = graph.nodes[node_id].operation_id
        if graph.nodes[node_id].kind == "OPERATION" and operation_id is not None:
            path_tail[operation_id] = tail
    movable = [
        operation for operation in operations
        if decoded.candidate.operation_order.index(operation) > 0
    ]
    if not movable:
        return NeighborhoodProposal("N1_CTU", decoded.candidate, (), {"changed": False})
    product_score = {
        product: max(path_tail[operation] for operation in movable if instance.product_of[operation] == product)
        for product in {instance.product_of[operation] for operation in movable}
    }
    best_score = max(product_score.values())
    products = sorted(product for product, score in product_score.items() if score == best_score)
    product = rng.choice(products)
    candidates = [operation for operation in movable if instance.product_of[operation] == product]
    best_tail = max(path_tail[operation] for operation in candidates)
    operation = rng.choice(sorted(op for op in candidates if path_tail[op] == best_tail))
    candidate, source, target = _insert_earlier(decoded.candidate, operation, rng)
    return NeighborhoodProposal("N1_CTU", candidate, (operation,), {
        "changed": candidate != decoded.candidate,
        "urgency_product": product,
        "critical_tail_duration": best_tail,
        "source_position": source,
        "target_position": target,
        "priority_feasibility": "READY_OPERATION_DECODER",
    })


def _n2_est(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
) -> NeighborhoodProposal:
    _, _, operations = _critical_context(instance, decoded, rng)
    movable = [
        operation for operation in operations
        if decoded.candidate.operation_order.index(operation) > 0
    ]
    if not movable:
        return NeighborhoodProposal("N2_EST", decoded.candidate, (), {"changed": False})
    earliest = min(decoded.schedule.operation_schedules[operation].start_time for operation in movable)
    tied = sorted(
        operation for operation in movable
        if decoded.schedule.operation_schedules[operation].start_time == earliest
    )
    operation = rng.choice(tied)
    candidate, source, target = _insert_earlier(decoded.candidate, operation, rng)
    return NeighborhoodProposal("N2_EST", candidate, (operation,), {
        "changed": candidate != decoded.candidate,
        "realized_time_field": "OperationSchedule.start_time",
        "realized_start_time": earliest,
        "source_position": source,
        "target_position": target,
        "priority_feasibility": "READY_OPERATION_DECODER",
    })


def randomized_product_topology(
    instance: Instance,
    product_id: str,
    current_order: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    """Generate a different random Kahn topological order when one exists."""

    operations = tuple(instance.product_data[product_id].operations)
    current = tuple(operation for operation in current_order if operation in set(operations))
    for _ in range(20):
        remaining = set(operations)
        generated: list[str] = []
        while remaining:
            ready = sorted(
                operation for operation in remaining
                if not (instance.predecessors[operation] & remaining)
            )
            operation = rng.choice(ready)
            generated.append(operation)
            remaining.remove(operation)
        result = tuple(generated)
        if result != current:
            return result
    return current


def inject_product_topology(
    candidate: Candidate,
    product_operations: tuple[str, ...],
    topology: tuple[str, ...],
) -> Candidate:
    selected = set(product_operations)
    positions = [
        index for index, operation in enumerate(candidate.operation_order)
        if operation in selected
    ]
    order = list(candidate.operation_order)
    for position, operation in zip(positions, topology):
        order[position] = operation
    return Candidate(
        tuple(order), candidate.island_assignment, candidate.w_assignment, candidate.f_assignment
    )


def _n3_topo(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
) -> NeighborhoodProposal:
    _, _, operations = _critical_context(instance, decoded, rng)
    products = sorted({instance.product_of[operation] for operation in operations})
    if not products:
        return NeighborhoodProposal("N3_TOPO", decoded.candidate, (), {"changed": False})
    product = rng.choice(products)
    product_operations = instance.product_data[product].operations
    topology = randomized_product_topology(
        instance, product, decoded.candidate.operation_order, rng
    )
    candidate = inject_product_topology(decoded.candidate, product_operations, topology)
    return NeighborhoodProposal("N3_TOPO", candidate, tuple(product_operations), {
        "changed": candidate != decoded.candidate,
        "product_id": product,
        "topological_order": topology,
    })


def _n4_mmit(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
) -> NeighborhoodProposal:
    _, _, operations = _critical_context(instance, decoded, rng)
    movable = [operation for operation in operations if instance.operation_data[operation].eligible_islands]
    if not movable:
        return NeighborhoodProposal("N4_MMIT", decoded.candidate, (), {"changed": False})
    # SOURCE_GAP_ASSUMPTION: use the minimum one critical operation per proposal.
    operation = rng.choice(sorted(movable))
    load = Counter(
        record.island_id for record in decoded.schedule.operation_schedules.values()
    )
    eligible = instance.operation_data[operation].eligible_islands
    minimum = min(load[island] for island in eligible)
    target = rng.choice(sorted(island for island in eligible if load[island] == minimum))
    index = instance.operations.index(operation)
    islands = list(decoded.candidate.island_assignment)
    source = islands[index]
    islands[index] = target
    candidate = Candidate(
        decoded.candidate.operation_order,
        tuple(islands),
        decoded.candidate.w_assignment,
        decoded.candidate.f_assignment,
    )
    return NeighborhoodProposal("N4_MMIT", candidate, (operation,), {
        "changed": candidate != decoded.candidate,
        "source_island": source,
        "target_island": target,
        "target_operation_count": minimum,
        "selection_count": 1,
    })


def propose_neighborhood(
    instance: Instance,
    decoded: DecodedCandidate,
    neighborhood_id: str,
    rng: random.Random,
) -> NeighborhoodProposal:
    if neighborhood_id == "N1_CTU":
        return _n1_ctu(instance, decoded, rng)
    if neighborhood_id == "N2_EST":
        return _n2_est(instance, decoded, rng)
    if neighborhood_id == "N3_TOPO":
        return _n3_topo(instance, decoded, rng)
    if neighborhood_id == "N4_MMIT":
        return _n4_mmit(instance, decoded, rng)
    raise ValueError(f"unknown LG_HGA neighborhood: {neighborhood_id}")
