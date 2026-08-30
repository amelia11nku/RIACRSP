"""Outcome-blind Phase 6C target-set design and three-seed labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
import statistics
from typing import Mapping, Sequence

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.data.instance import Instance

from .alns import DESTROY, _destroy
from .common import DecodedCandidate
from .counterfactual import stable_seed


@dataclass(frozen=True)
class ArmProposal:
    origin_rule: str
    arm_family: str
    origin_destroy_operator: str
    destroyed_operations: tuple[str, ...]
    reference_operations: tuple[str, ...] = ()
    removed_operations: tuple[str, ...] = ()
    added_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase6CTargetArm:
    target_set_id: str
    arm_family: str
    origin_destroy_operator: str
    origin_rules: tuple[str, ...]
    origin_families: tuple[str, ...]
    destroyed_operations: tuple[str, ...]


@dataclass(frozen=True)
class ArmGenerationResult:
    arms: tuple[Phase6CTargetArm, ...]
    proposals: tuple[ArmProposal, ...]
    canonical_related_target: tuple[str, ...]
    requested_arm_count: int
    unique_arm_count: int
    duplicate_arm_count: int


def target_set_id(state_id: str, operations: Sequence[str]) -> str:
    payload = json.dumps([state_id, sorted(operations)], separators=(",", ":")).encode()
    return "ts_" + hashlib.sha256(payload).hexdigest()[:20]


def _random_swap(
    reference: tuple[str, ...], operations: tuple[str, ...], replace_count: int, seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    outside = sorted(set(operations) - set(reference))
    count = min(replace_count, len(reference), len(outside))
    rng = random.Random(seed)
    removed = tuple(sorted(rng.sample(list(reference), count)))
    added = tuple(sorted(rng.sample(outside, count)))
    target = tuple(sorted((set(reference) - set(removed)) | set(added)))
    return target, removed, added


def _ranked_replacement(
    reference: tuple[str, ...], operations: tuple[str, ...], replace_count: int,
    score: Mapping[str, float], state_id: str, rule: str, namespace: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    outside = sorted(set(operations) - set(reference))
    count = min(replace_count, len(reference), len(outside))
    tie = lambda operation: stable_seed(state_id, rule, operation, namespace=namespace)
    removed = tuple(sorted(sorted(reference, key=lambda op: (score[op], tie(op)))[:count]))
    added = tuple(sorted(sorted(outside, key=lambda op: (-score[op], tie(op)))[:count]))
    target = tuple(sorted((set(reference) - set(removed)) | set(added)))
    return target, removed, added


def _near_neighbor_scores(
    instance: Instance, current_decoded: DecodedCandidate, reference: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    features = schedule_features(instance, current_decoded.schedule)
    product_counts = Counter(instance.product_of[operation] for operation in reference)
    island_counts = Counter(features[operation]["assigned_island"] for operation in reference)
    direct_neighbors = {
        neighbor
        for operation in reference
        for neighbor in (*instance.predecessors[operation], *instance.successors[operation])
    }
    transitive_neighbors = {
        neighbor
        for operation in reference
        for neighbor in (*instance.transitive_predecessors[operation], *instance.transitive_successors[operation])
    }
    return {
        "same_product": {
            operation: float(product_counts[instance.product_of[operation]])
            for operation in instance.operations
        },
        "precedence_neighbor": {
            operation: float(2 if operation in direct_neighbors else 1 if operation in transitive_neighbors else 0)
            for operation in instance.operations
        },
        "same_island_chain": {
            operation: float(island_counts[features[operation]["assigned_island"]])
            for operation in instance.operations
        },
        "high_W_delay": {
            operation: float(features[operation]["W_waiting_or_delay_contribution"])
            for operation in instance.operations
        },
        "high_F_delay": {
            operation: float(features[operation]["F_waiting_or_delay_contribution"])
            for operation in instance.operations
        },
        "low_slack": {
            operation: -float(features[operation]["operation_slack"])
            for operation in instance.operations
        },
    }


def generate_revised_target_arms(
    instance: Instance,
    current_decoded: DecodedCandidate,
    state_id: str,
    destroy_count: int,
    seed_namespace: int,
) -> ArmGenerationResult:
    """Generate the frozen Phase 6C 24-rule design and deduplicate target sets."""
    count = min(int(destroy_count), instance.num_operations)
    if count <= 0:
        raise ValueError("destroy_count must be positive")
    proposals: list[ArmProposal] = []
    operator_targets: dict[str, tuple[str, ...]] = {}
    for operator in DESTROY:
        rng = random.Random(stable_seed(state_id, "operator", operator, namespace=seed_namespace))
        target = tuple(sorted(_destroy(instance, current_decoded, operator, count, rng)))
        operator_targets[operator] = target
        proposals.append(ArmProposal(f"operator_{operator}", "ORIGINAL_OPERATOR", operator, target))

    for index in range(1, 5):
        rng = random.Random(stable_seed(state_id, "related_variant", index, namespace=seed_namespace))
        target = tuple(sorted(_destroy(instance, current_decoded, "related", count, rng)))
        proposals.append(ArmProposal(f"related_variant_{index}", "RELATED_VARIANT", "related", target))
    for index in range(1, 4):
        rng = random.Random(stable_seed(state_id, "matched_random", index, namespace=seed_namespace))
        target = tuple(sorted(_destroy(instance, current_decoded, "random", count, rng)))
        proposals.append(ArmProposal(f"matched_random_{index}", "MATCHED_RANDOM", "random", target))

    reference = operator_targets["related"]
    local_rules = (("one_operation_swap", 1), ("two_operation_swap", 2),
                   ("related_replace_25", max(1, round(count * .25))),
                   ("related_replace_50", max(1, round(count * .50))))
    for rule, replace_count in local_rules:
        target, removed, added = _random_swap(
            reference, instance.operations, replace_count,
            stable_seed(state_id, rule, namespace=seed_namespace),
        )
        proposals.append(ArmProposal(
            rule, "LOCAL_PERTURBATION", "related", target, reference, removed, added,
        ))

    scores = _near_neighbor_scores(instance, current_decoded, reference)
    replace_count = max(1, round(count * .25))
    for rule, score in scores.items():
        target, removed, added = _ranked_replacement(
            reference, instance.operations, replace_count, score, state_id, rule, seed_namespace,
        )
        proposals.append(ArmProposal(
            f"near_{rule}", "STRUCTURED_NEAR_NEIGHBOR", "related", target,
            reference, removed, added,
        ))

    by_target: dict[tuple[str, ...], list[ArmProposal]] = {}
    for proposal in proposals:
        by_target.setdefault(proposal.destroyed_operations, []).append(proposal)
    arms = []
    for operations, origins in by_target.items():
        first = origins[0]
        arms.append(Phase6CTargetArm(
            target_set_id(state_id, operations), first.arm_family, first.origin_destroy_operator,
            tuple(origin.origin_rule for origin in origins),
            tuple(dict.fromkeys(origin.arm_family for origin in origins)), operations,
        ))
    return ArmGenerationResult(
        tuple(arms), tuple(proposals), reference, len(proposals), len(arms), len(proposals) - len(arms),
    )


def aggregate_repair_outcomes(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int | bool]:
    """Aggregate exactly three deterministic repair outcomes for one target set."""
    if len(rows) != 3 or len({int(row["repair_seed"]) for row in rows}) != 3:
        raise ValueError("exactly three distinct repair seeds are required")
    makespans = [float(row["counterfactual_makespan"]) for row in rows]
    absolute = [float(row["absolute_improvement"]) for row in rows]
    relative = [float(row["relative_improvement"]) for row in rows]
    positives = sum(value > 0 for value in relative)
    return {
        "mean_counterfactual_makespan": statistics.fmean(makespans),
        "median_counterfactual_makespan": statistics.median(makespans),
        "mean_absolute_improvement": statistics.fmean(absolute),
        "mean_relative_improvement": statistics.fmean(relative),
        "median_relative_improvement": statistics.median(relative),
        "std_relative_improvement": statistics.pstdev(relative),
        "improvement_probability": positives / 3.0,
        "positive_seed_count": positives,
        "positive_under_1_of_3": positives >= 1,
        "positive_under_2_of_3": positives >= 2,
        "positive_under_3_of_3": positives == 3,
    }


def pairwise_preference(gain_difference: float) -> int:
    return 1 if gain_difference > 0 else -1 if gain_difference < 0 else 0
