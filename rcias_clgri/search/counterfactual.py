"""Pure counterfactual destroy-target evaluation for Phase 6B diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import time

from rcias_clgri.data.instance import Instance

from .alns import DESTROY, _destroy, _neighbor
from .common import Candidate, DecodedCandidate, decode_candidate


@dataclass(frozen=True)
class CounterfactualResult:
    counterfactual: DecodedCandidate
    absolute_improvement: float
    relative_improvement: float
    improved: bool
    decoder_evaluations: int
    runtime: float


@dataclass(frozen=True)
class TargetArm:
    arm_id: str
    arm_family: str
    origin_destroy_operator: str
    destroyed_operations: tuple[str, ...]
    duplicate_origin_labels: tuple[str, ...] = ()


def stable_seed(*identifiers: object, namespace: int = 0) -> int:
    payload = "\x1f".join(map(str, (namespace, *identifiers))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def evaluate_counterfactual(
    instance: Instance,
    current_candidate: Candidate,
    current_decoded: DecodedCandidate,
    destroyed_operations: tuple[str, ...] | set[str],
    repair_operator: str,
    repair_seed: int,
    candidate_trials: int,
) -> CounterfactualResult:
    """Evaluate one target set without touching live search state or RNG."""
    if current_decoded.candidate != current_candidate:
        raise ValueError("current_candidate and current_decoded must describe the same state")
    if repair_operator != "transport_aware":
        raise ValueError("Phase 6B primary counterfactual evaluation fixes transport_aware repair")
    removed = frozenset(destroyed_operations)
    if not removed or not removed <= set(instance.operations):
        raise ValueError("destroyed_operations must be a non-empty subset of instance operations")
    if candidate_trials <= 0:
        raise ValueError("candidate_trials must be positive")
    rng = random.Random(repair_seed)
    started = time.perf_counter()
    candidates = [
        decode_candidate(instance, _neighbor(instance, current_candidate, set(removed), repair_operator, rng))
        for _ in range(candidate_trials)
    ]
    selected = min(candidates, key=lambda item: item.makespan)
    improvement = current_decoded.makespan - selected.makespan
    return CounterfactualResult(
        selected, improvement, improvement / current_decoded.makespan,
        improvement > 0, candidate_trials, time.perf_counter() - started,
    )


def generate_target_arms(
    instance: Instance,
    current_decoded: DecodedCandidate,
    state_id: str,
    destroy_count: int,
    seed_namespace: int,
) -> tuple[TargetArm, ...]:
    """Generate 14 outcome-blind target arms, deduplicating identical sets."""
    proposed: list[TargetArm] = []
    operator_targets = {}
    for operator in DESTROY:
        rng = random.Random(stable_seed(state_id, "operator", operator, namespace=seed_namespace))
        targets = tuple(sorted(_destroy(instance, current_decoded, operator, destroy_count, rng)))
        operator_targets[operator] = targets
        proposed.append(TargetArm(f"operator_{operator}", "ORIGINAL_OPERATOR", operator, targets))
    for index in range(1, 4):
        rng = random.Random(stable_seed(state_id, "related_variant", index, namespace=seed_namespace))
        targets = tuple(sorted(_destroy(instance, current_decoded, "related", destroy_count, rng)))
        proposed.append(TargetArm(f"related_variant_{index}", "RELATED_VARIANT", "related", targets))
    for index in range(1, 3):
        rng = random.Random(stable_seed(state_id, "matched_random", index, namespace=seed_namespace))
        targets = tuple(sorted(_destroy(instance, current_decoded, "random", destroy_count, rng)))
        proposed.append(TargetArm(f"matched_random_{index}", "MATCHED_RANDOM", "random", targets))
    reference = set(operator_targets["related"])
    outside = sorted(set(instance.operations) - reference)
    for fraction in (0.25, 0.50):
        rng = random.Random(stable_seed(state_id, "perturbation", fraction, namespace=seed_namespace))
        replace_count = min(len(outside), max(1, round(destroy_count * fraction)))
        perturbed = (reference - set(rng.sample(sorted(reference), replace_count))) | set(rng.sample(outside, replace_count))
        proposed.append(TargetArm(f"related_replace_{int(fraction * 100)}", "TARGET_PERTURBATION", "related", tuple(sorted(perturbed))))
    unique: dict[tuple[str, ...], TargetArm] = {}
    duplicates: dict[tuple[str, ...], list[str]] = {}
    for arm in proposed:
        if arm.destroyed_operations not in unique:
            unique[arm.destroyed_operations] = arm; duplicates[arm.destroyed_operations] = []
        else:
            duplicates[arm.destroyed_operations].append(arm.arm_id)
    return tuple(TargetArm(arm.arm_id, arm.arm_family, arm.origin_destroy_operator,
                           arm.destroyed_operations, tuple(duplicates[targets]))
                 for targets, arm in unique.items())


def swap_target(reference: tuple[str, ...], removed_in: str, added_out: str) -> tuple[str, ...]:
    if removed_in not in reference or added_out in reference:
        raise ValueError("marginal swap must remove an in-target and add an out-of-target operation")
    return tuple(sorted((set(reference) - {removed_in}) | {added_out}))
