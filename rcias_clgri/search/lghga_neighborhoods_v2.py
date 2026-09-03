"""Versioned LG_HGA neighborhoods with a source-closer multi-stage N4."""

from __future__ import annotations

from collections import Counter
import random

from rcias_clgri.data.instance import Instance

from .common import Candidate, DecodedCandidate
from .lghga_neighborhoods import (
    NeighborhoodProposal,
    _critical_context,
    propose_neighborhood as propose_neighborhood_v1,
)


N4_SELECTION_TARGET = 2


def _n4_mmit_multi(
    instance: Instance,
    decoded: DecodedCandidate,
    rng: random.Random,
) -> NeighborhoodProposal:
    """Move two critical stages when feasible, the minimum plural reading of "several"."""

    _, _, critical_operations = _critical_context(instance, decoded, rng)
    operations = list(dict.fromkeys(critical_operations))
    rng.shuffle(operations)
    load = Counter(
        record.island_id for record in decoded.schedule.operation_schedules.values()
    )
    islands = list(decoded.candidate.island_assignment)
    changes: list[dict[str, object]] = []

    for operation in operations:
        index = instance.operations.index(operation)
        source = islands[index]
        eligible = instance.operation_data[operation].eligible_islands
        minimum = min(load[island] for island in eligible)
        targets = sorted(
            island for island in eligible
            if island != source and load[island] == minimum
        )
        if not targets:
            continue
        target = rng.choice(targets)
        changes.append({
            "operation_id": operation,
            "source_island": source,
            "target_island": target,
            "target_operation_count_before": minimum,
            "eligible_operation_counts_before": {
                island: load[island] for island in eligible
            },
        })
        islands[index] = target
        load[source] -= 1
        load[target] += 1
        if len(changes) == N4_SELECTION_TARGET:
            break

    if not changes:
        return NeighborhoodProposal("N4_MMIT", decoded.candidate, (), {
            "changed": False,
            "selection_count": 0,
            "selection_target": N4_SELECTION_TARGET,
            "selection_rule": "MINIMAL_PLURAL_TWO_EFFECTIVE_MOVES",
            "changes": [],
        })

    candidate = Candidate(
        decoded.candidate.operation_order,
        tuple(islands),
        decoded.candidate.w_assignment,
        decoded.candidate.f_assignment,
    )
    return NeighborhoodProposal(
        "N4_MMIT",
        candidate,
        tuple(str(change["operation_id"]) for change in changes),
        {
            "changed": True,
            "selection_count": len(changes),
            "selection_target": N4_SELECTION_TARGET,
            "selection_rule": "MINIMAL_PLURAL_TWO_EFFECTIVE_MOVES",
            "changes": changes,
        },
    )


def propose_neighborhood(
    instance: Instance,
    decoded: DecodedCandidate,
    neighborhood_id: str,
    rng: random.Random,
) -> NeighborhoodProposal:
    """Use the v2 N4 while preserving the frozen v1 N1--N3 definitions."""

    if neighborhood_id == "N4_MMIT":
        return _n4_mmit_multi(instance, decoded, rng)
    return propose_neighborhood_v1(instance, decoded, neighborhood_id, rng)
