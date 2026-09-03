"""Source-paper feasibility and clipping predicates for DABC graph moves.

The source theorems are defined for RMSSP operation/machine graphs. RIACRSP
adds a product DAG and two logistics fleets, so this module evaluates the
published predicates on an explicit operation/island projection. Search
feasibility is checked independently on the full realized product/resource
order, and clipping remains diagnostic-only in :mod:`rcias_clgri.search.dabc`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from rcias_clgri.data.instance import Instance

from .common import DecodedCandidate
from .dabc_chdg import CriticalIslandBlock


SOURCE_EXACT = "SOURCE_PAPER_EXACT_RIACRSP_PROJECTION"
NOT_APPLICABLE = "SOURCE_PAPER_NOT_APPLICABLE"


@dataclass(frozen=True)
class RuleResult:
    theorem: str
    status: str
    triggered: bool
    detail: str


@dataclass(frozen=True)
class MoveRuleAudit:
    move_kind: str
    source_feasibility_predicate: bool
    full_dag_reachability_feasible: bool
    feasibility_rules: tuple[RuleResult, ...]
    clipping_rules: tuple[RuleResult, ...]

    @property
    def source_clip_predicate(self) -> bool:
        return any(rule.triggered for rule in self.clipping_rules)


@dataclass(frozen=True)
class SourceProjectionMetrics:
    """F/R values and predecessor maps from the paper's RMSSP projection."""

    processing_time: Mapping[str, float]
    machine_predecessor: Mapping[str, str | None]
    machine_successor: Mapping[str, str | None]
    job_predecessor: Mapping[str, str | None]
    job_successor: Mapping[str, str | None]
    head: Mapping[str, float]
    tail: Mapping[str, float]
    makespan: float


def _operation_graph_order(
    operations: Sequence[str],
    product_sequences: Mapping[str, Sequence[str]],
    island_sequences: Mapping[str, Sequence[str]],
) -> tuple[str, ...] | None:
    successors = {operation: set() for operation in operations}
    indegree = {operation: 0 for operation in operations}
    for sequences in (product_sequences.values(), island_sequences.values()):
        for sequence in sequences:
            for source, target in zip(sequence, sequence[1:]):
                if target not in successors[source]:
                    successors[source].add(target)
                    indegree[target] += 1
    ready = sorted(operation for operation, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        operation = ready.pop(0)
        order.append(operation)
        for successor in sorted(successors[operation]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return tuple(order) if len(order) == len(operations) else None


def _adjacent_maps(
    operations: Sequence[str],
    sequences: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    predecessor = {operation: None for operation in operations}
    successor = {operation: None for operation in operations}
    for sequence in sequences.values():
        for left, right in zip(sequence, sequence[1:]):
            predecessor[right] = left
            successor[left] = right
    return predecessor, successor


def _rt(
    instance: Instance,
    decoded: DecodedCandidate,
    island_id: str,
    predecessor: str | None,
    successor: str | None,
) -> float:
    """Paper RT(u,v), including its virtual-start configuration boundary."""

    if successor is None:
        return 0.0
    source_config = (
        instance.island_data[island_id].initial_config
        if predecessor is None
        else decoded.schedule.operation_schedules[predecessor].config_id
    )
    target_config = decoded.schedule.operation_schedules[successor].config_id
    return float(instance.reconfiguration_time[(island_id, source_config, target_config)])


def source_projection_metrics(
    instance: Instance,
    decoded: DecodedCandidate,
) -> SourceProjectionMetrics:
    """Compute the paper's exact F/R recurrences on the RIACRSP projection."""

    schedule = decoded.schedule
    topological = _operation_graph_order(
        instance.operations, schedule.product_sequences, schedule.island_timelines
    )
    if topological is None:
        raise ValueError("decoded operation/island projection is cyclic")
    machine_predecessor, machine_successor = _adjacent_maps(
        instance.operations, schedule.island_timelines
    )
    job_predecessor, job_successor = _adjacent_maps(
        instance.operations, schedule.product_sequences
    )
    processing = {
        operation: float(schedule.operation_schedules[operation].processing_time)
        for operation in instance.operations
    }
    head: dict[str, float] = {}
    for operation in topological:
        record = schedule.operation_schedules[operation]
        mp = machine_predecessor[operation]
        jp = job_predecessor[operation]
        machine_head = (
            0.0 if mp is None else head[mp] + processing[mp]
        ) + _rt(instance, decoded, record.island_id, mp, operation)
        job_head = 0.0 if jp is None else head[jp] + processing[jp]
        head[operation] = max(machine_head, job_head)
    tail: dict[str, float] = {}
    for operation in reversed(topological):
        record = schedule.operation_schedules[operation]
        ms = machine_successor[operation]
        js = job_successor[operation]
        machine_tail = (
            0.0
            if ms is None
            else tail[ms]
            + processing[ms]
            + _rt(instance, decoded, record.island_id, operation, ms)
        )
        job_tail = 0.0 if js is None else tail[js] + processing[js]
        tail[operation] = max(machine_tail, job_tail)
    makespan = max(
        (head[operation] + processing[operation] for operation in instance.operations),
        default=0.0,
    )
    return SourceProjectionMetrics(
        MappingProxyType(processing),
        MappingProxyType(machine_predecessor),
        MappingProxyType(machine_successor),
        MappingProxyType(job_predecessor),
        MappingProxyType(job_successor),
        MappingProxyType(head),
        MappingProxyType(tail),
        makespan,
    )


def _full_order_feasible(
    instance: Instance,
    decoded: DecodedCandidate,
    island_sequences: Mapping[str, Sequence[str]],
) -> bool:
    flattened = [operation for sequence in island_sequences.values() for operation in sequence]
    if len(flattened) != len(instance.operations) or set(flattened) != set(instance.operations):
        return False
    return _operation_graph_order(
        instance.operations,
        decoded.schedule.product_sequences,
        island_sequences,
    ) is not None


def _rule(
    theorem: str,
    applicable: bool,
    predicate: bool,
    detail: str,
) -> RuleResult:
    return RuleResult(
        theorem,
        SOURCE_EXACT if applicable else NOT_APPLICABLE,
        applicable and predicate,
        detail,
    )


def _inactive_clipping(detail: str) -> tuple[RuleResult, ...]:
    return tuple(_rule(f"THEOREM_{number}", False, False, detail) for number in range(5, 11))


def audit_intramachine_move(
    instance: Instance,
    decoded: DecodedCandidate,
    blocks: tuple[CriticalIslandBlock, ...],
    block_index: int,
    moved_operation: str,
    target_index: int,
    requested_block_order: tuple[str, ...],
) -> MoveRuleAudit:
    """Evaluate Theorems 1/2 and 5--10 for one CNS1 insertion."""

    block = blocks[block_index]
    original = block.operation_ids
    if (
        moved_operation not in original
        or not 0 <= target_index < len(original)
        or set(requested_block_order) != set(original)
        or len(requested_block_order) != len(original)
    ):
        raise ValueError("invalid CNS1 block insertion audit")
    original_index = original.index(moved_operation)
    if target_index == original_index:
        raise ValueError("CNS1 audit requires a nonidentity insertion")

    metrics = source_projection_metrics(instance, decoded)
    if target_index > original_index:
        direction = "MOVE_U_AFTER_V"
        u = moved_operation
        v = original[target_index]
        js_u = metrics.job_successor[u]
        applicable_1 = instance.product_of[u] != instance.product_of[v]
        left_1 = metrics.tail[v] + metrics.processing_time[v]
        right_1 = 0.0 if js_u is None else metrics.tail[js_u]
        theorem_1 = _rule(
            "THEOREM_1",
            applicable_1,
            left_1 > right_1,
            f"R(v)+p[v]={left_1:g} > R(js[u])={right_1:g}",
        )
        theorem_2 = _rule("THEOREM_2", False, False, "opposite insertion direction")
    else:
        direction = "MOVE_V_BEFORE_U"
        u = original[target_index]
        v = moved_operation
        jp_v = metrics.job_predecessor[v]
        applicable_2 = instance.product_of[u] != instance.product_of[v]
        left_2 = metrics.head[u] + metrics.processing_time[u]
        right_2 = 0.0 if jp_v is None else metrics.head[jp_v]
        theorem_1 = _rule("THEOREM_1", False, False, "opposite insertion direction")
        theorem_2 = _rule(
            "THEOREM_2",
            applicable_2,
            left_2 > right_2,
            f"F(u)+p[u]={left_2:g} > F(jp[v])={right_2:g}",
        )

    island_timeline = decoded.schedule.island_timelines[block.island_id]
    selected = set(original)
    positions = [index for index, operation in enumerate(island_timeline) if operation in selected]
    proposed_timeline = list(island_timeline)
    for position, operation in zip(positions, requested_block_order):
        proposed_timeline[position] = operation
    proposed_islands = {
        island: tuple(sequence)
        for island, sequence in decoded.schedule.island_timelines.items()
    }
    proposed_islands[block.island_id] = tuple(proposed_timeline)
    full_feasible = _full_order_feasible(instance, decoded, proposed_islands)

    index_u = original.index(u)
    index_v = original.index(v)
    internal = (
        0 < index_u < len(original) - 1
        and 0 < index_v < len(original) - 1
    )
    first_boundary = (
        block_index == 0
        and u == original[0]
        and 0 < index_v < len(original) - 1
        and metrics.machine_predecessor[u] is None
    )
    last_boundary = (
        block_index == len(blocks) - 1
        and v == original[-1]
        and 0 < index_u < len(original) - 1
        and metrics.machine_successor[v] is None
    )
    mp_u = metrics.machine_predecessor[u]
    ms_u = metrics.machine_successor[u]
    mp_v = metrics.machine_predecessor[v]
    ms_v = metrics.machine_successor[v]
    rt = lambda left, right: _rt(instance, decoded, block.island_id, left, right)

    clipping: list[RuleResult] = []
    applicable_5 = direction == "MOVE_U_AFTER_V" and internal
    lhs_5 = rt(mp_u, u) + rt(u, ms_u) + rt(v, ms_v)
    rhs_5 = rt(mp_u, ms_u) + rt(v, u) + rt(u, ms_v)
    clipping.append(_rule(
        "THEOREM_5", applicable_5, lhs_5 <= rhs_5,
        f"{lhs_5:g} <= {rhs_5:g} (move u after v; internal)",
    ))

    applicable_6 = direction == "MOVE_V_BEFORE_U" and internal
    lhs_6 = rt(mp_u, u) + rt(mp_v, v) + rt(v, ms_v)
    rhs_6 = rt(mp_u, v) + rt(v, u) + rt(mp_v, ms_v)
    clipping.append(_rule(
        "THEOREM_6", applicable_6, lhs_6 <= rhs_6,
        f"{lhs_6:g} <= {rhs_6:g} (move v before u; internal)",
    ))

    applicable_7 = direction == "MOVE_U_AFTER_V" and first_boundary
    lhs_7 = rt(None, u) + rt(u, ms_u) + rt(v, ms_v)
    rhs_7 = rt(None, ms_u) + rt(v, u) + rt(u, ms_v)
    clipping.append(_rule(
        "THEOREM_7", applicable_7, lhs_7 <= rhs_7,
        f"{lhs_7:g} <= {rhs_7:g} (first block; move u after v)",
    ))

    applicable_8 = direction == "MOVE_V_BEFORE_U" and first_boundary
    lhs_8 = rt(None, u) + rt(u, ms_u) + rt(mp_v, v)
    rhs_8 = rt(None, v) + rt(v, u) + rt(u, ms_u)
    clipping.append(_rule(
        "THEOREM_8", applicable_8, lhs_8 <= rhs_8,
        f"{lhs_8:g} <= {rhs_8:g} (first block; move v before u)",
    ))

    applicable_9 = direction == "MOVE_U_AFTER_V" and last_boundary
    lhs_9 = rt(mp_u, u)
    rhs_9 = rt(v, u)
    clipping.append(_rule(
        "THEOREM_9", applicable_9, lhs_9 <= rhs_9,
        f"{lhs_9:g} <= {rhs_9:g} (last block; move u after v)",
    ))

    applicable_10 = direction == "MOVE_V_BEFORE_U" and last_boundary
    lhs_10 = rt(mp_u, u) + rt(mp_v, v)
    rhs_10 = rt(mp_u, v) + rt(v, u)
    clipping.append(_rule(
        "THEOREM_10", applicable_10, lhs_10 <= rhs_10,
        f"{lhs_10:g} <= {rhs_10:g} (last block; move v before u)",
    ))

    feasibility = (theorem_1, theorem_2)
    return MoveRuleAudit(
        f"CNS1_INTRAMACHINE_{direction}",
        any(rule.triggered for rule in feasibility),
        full_feasible,
        feasibility,
        tuple(clipping),
    )


def audit_intermachine_move(
    instance: Instance,
    decoded: DecodedCandidate,
    operation_id: str,
    target_island: str,
    requested_island_order: tuple[str, ...],
) -> MoveRuleAudit:
    """Evaluate Theorems 3/4 and full reachability for one CNS2 insertion."""

    if requested_island_order.count(operation_id) != 1:
        raise ValueError("CNS2 target order must contain the moved operation exactly once")
    eligible = target_island in instance.operation_data[operation_id].eligible_islands
    proposed_islands = {
        island: tuple(operation for operation in sequence if operation != operation_id)
        for island, sequence in decoded.schedule.island_timelines.items()
    }
    proposed_islands[target_island] = requested_island_order
    full_feasible = eligible and _full_order_feasible(instance, decoded, proposed_islands)

    metrics = source_projection_metrics(instance, decoded)
    position = requested_island_order.index(operation_id)
    u = requested_island_order[position - 1] if position > 0 else None
    v = requested_island_order[position + 1] if position + 1 < len(requested_island_order) else None
    jp_w = metrics.job_predecessor[operation_id]
    js_w = metrics.job_successor[operation_id]

    head_u_completion = 0.0 if u is None else metrics.head[u] + metrics.processing_time[u]
    head_v_completion = (
        metrics.makespan if v is None else metrics.head[v] + metrics.processing_time[v]
    )
    head_jp_completion = (
        0.0 if jp_w is None else metrics.head[jp_w] + metrics.processing_time[jp_w]
    )
    tail_v_completion = 0.0 if v is None else metrics.processing_time[v] + metrics.tail[v]
    tail_u_completion = (
        metrics.makespan if u is None else metrics.processing_time[u] + metrics.tail[u]
    )
    tail_js_completion = (
        0.0 if js_w is None else metrics.processing_time[js_w] + metrics.tail[js_w]
    )
    theorem_3_predicate = (
        head_u_completion > head_jp_completion
        and tail_v_completion > tail_js_completion
    )
    theorem_4_predicate = (
        head_v_completion < head_jp_completion
        and tail_u_completion < tail_js_completion
    )
    theorem_3 = _rule(
        "THEOREM_3",
        eligible,
        theorem_3_predicate,
        f"{head_u_completion:g}>{head_jp_completion:g} and {tail_v_completion:g}>{tail_js_completion:g}",
    )
    theorem_4 = _rule(
        "THEOREM_4",
        eligible,
        theorem_4_predicate,
        f"{head_v_completion:g}<{head_jp_completion:g} and {tail_u_completion:g}<{tail_js_completion:g}",
    )
    feasibility = (theorem_3, theorem_4)
    return MoveRuleAudit(
        "CNS2_INTERMACHINE",
        any(rule.triggered for rule in feasibility),
        full_feasible,
        feasibility,
        _inactive_clipping("Theorems 5--10 apply only to CNS1 intramachine moves"),
    )
