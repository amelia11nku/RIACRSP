"""Outcome-blind candidate selection and post-decoder labels for Phase 6I-MR."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
import time
from typing import Sequence

import numpy as np
import torch

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.instance import Instance
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.live_inference import FrozenLiveInference
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.search.common import Candidate, DecodedCandidate, decode_candidate
from rcias_clgri.search.counterfactual import stable_seed
from rcias_clgri.search.alns import (
    ALNSConfig,
    DESTROY,
    REPAIR,
    _destroy,
    _neighbor,
    _roulette,
)


@dataclass(frozen=True)
class FrozenArmPrediction:
    target_set_id: str
    arm_family: str
    origin_destroy_operator: str
    origin_rules: tuple[str, ...]
    destroyed_operations: tuple[str, ...]
    raw_score: float
    raw_probability: float
    raw_utility: float
    calibrated_probability: float
    calibrated_utility: float


@dataclass(frozen=True)
class FrozenBankPrediction:
    arms: tuple[FrozenArmPrediction, ...]
    graph: object
    requested_arm_count: int
    unique_arm_count: int
    duplicate_arm_count: int
    state_feature_summary: dict[str, float]
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class ForcedRoleSelection:
    role: str
    arm: FrozenArmPrediction
    replacement: bool


@dataclass(frozen=True)
class ForcedDecodeResult:
    candidate: DecodedCandidate
    trial_makespans: tuple[float, ...]
    repair_seed: int
    decoder_evaluations: int
    runtime_ms: float


@dataclass(frozen=True)
class ContinuationResult:
    candidate: DecodedCandidate
    start_makespan: float
    best_makespan: float
    continuation_value: float
    continuation_seed: int
    derived_seed: int
    iterations: int
    decoder_evaluations: int
    accepted_moves: int
    improving_moves: int
    operator_selections: dict[str, int]
    runtime_ms: float


def _state_feature_summary(graph) -> dict[str, float]:
    operations = graph.nodes["OP"]
    makespan = max(float(graph.graph_features["current_makespan"]), 1.0)
    return {
        "mean_slack_ratio": float(np.mean([
            node.features["operation_slack"] / makespan for node in operations
        ])),
        "mean_w_delay_ratio": float(np.mean([
            node.features["w_delay"] / makespan for node in operations
        ])),
        "mean_f_delay_ratio": float(np.mean([
            node.features["f_delay"] / makespan for node in operations
        ])),
        "mean_island_relative_load": float(np.mean([
            node.features["island_relative_load"] for node in operations
        ])),
        "mean_local_reconfiguration_ratio": float(np.mean([
            node.features["local_reconfiguration"] / makespan for node in operations
        ])),
        "search_progress": float(graph.graph_features["search_progress"]),
    }


def score_frozen_candidate_bank(
    policy: FrozenLiveInference,
    instance: Instance,
    current: DecodedCandidate,
    *,
    state_id: str,
    destroy_count: int,
    search_progress: float,
    search_stage: str,
) -> FrozenBankPrediction:
    """Score every frozen proposal without consulting any decoded outcome."""
    started = time.perf_counter()
    graph = build_csg_from_schedule(
        instance,
        current.schedule,
        state_id=state_id,
        search_progress=search_progress,
        search_stage=search_stage,
    )
    state_features = _state_feature_summary(graph)
    csg_ms = (time.perf_counter() - started) * 1000.0

    proposal_started = time.perf_counter()
    generated, records = build_live_proposal_bank(
        instance,
        current,
        state_id=state_id,
        destroy_count=destroy_count,
        seed_namespace=policy.proposal_seed_namespace,
    )
    proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

    tensor_started = time.perf_counter()
    sample = NIStateSample(
        policy.tensorizer.tensorize(graph),
        tensorize_action_records(graph, records),
        {"search_stage": search_stage},
    )
    batch = batch_state_samples([sample]).to(policy.device)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    tensor_ms = (time.perf_counter() - tensor_started) * 1000.0
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if policy.device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        inference_started = time.perf_counter()
        output = policy.model(batch)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    calibration_started = time.perf_counter()
    raw_scores = output.scores.detach().float().cpu().numpy()
    raw_utilities = output.utility_predictions.detach().float().cpu().numpy()
    probabilities = policy.probability.predict(raw_scores)
    utilities = policy.utility.predict(raw_utilities)
    calibration_ms = (time.perf_counter() - calibration_started) * 1000.0
    predictions = tuple(
        FrozenArmPrediction(
            target_set_id=arm.target_set_id,
            arm_family=arm.arm_family,
            origin_destroy_operator=arm.origin_destroy_operator,
            origin_rules=arm.origin_rules,
            destroyed_operations=arm.destroyed_operations,
            raw_score=float(raw_scores[index]),
            raw_probability=float(
                1.0 / (1.0 + np.exp(-np.clip(raw_scores[index], -40.0, 40.0)))
            ),
            raw_utility=float(raw_utilities[index]),
            calibrated_probability=float(probabilities[index]),
            calibrated_utility=float(utilities[index]),
        )
        for index, arm in enumerate(generated.arms)
    )
    return FrozenBankPrediction(
        arms=predictions,
        graph=graph,
        requested_arm_count=generated.requested_arm_count,
        unique_arm_count=generated.unique_arm_count,
        duplicate_arm_count=generated.duplicate_arm_count,
        state_feature_summary=state_features,
        timings_ms={
            "csg_build": csg_ms,
            "proposal_bank": proposal_ms,
            "tensorization_and_transfer": tensor_ms,
            "model_inference_and_action_scoring": inference_ms,
            "calibration": calibration_ms,
            "total": (time.perf_counter() - started) * 1000.0,
        },
    )


def _score_order(arm: FrozenArmPrediction) -> tuple[float, str]:
    return (-arm.raw_score, arm.target_set_id)


def _fallback_priority(arm: FrozenArmPrediction) -> int:
    priorities = {
        "operator_related": 0,
        "related_variant_1": 1,
        "related_variant_2": 2,
        "related_variant_3": 3,
        "related_variant_4": 4,
    }
    direct = [priorities[rule] for rule in arm.origin_rules if rule in priorities]
    if direct:
        return min(direct)
    if (
        arm.origin_destroy_operator == "related"
        or arm.arm_family in {"LOCAL_PERTURBATION", "STRUCTURED_NEAR_NEIGHBOR"}
    ):
        return 5
    return 6


def _jaccard_distance(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 0.0 if not union else 1.0 - len(left_set & right_set) / len(union)


def select_forced_candidate_roles(
    arms: Sequence[FrozenArmPrediction],
) -> tuple[ForcedRoleSelection, ...]:
    """Select four unique roles using predictions and provenance only."""
    ranked = sorted(arms, key=_score_order)
    unique: list[FrozenArmPrediction] = []
    seen_targets: set[tuple[str, ...]] = set()
    for arm in ranked:
        target = tuple(sorted(arm.destroyed_operations))
        if target not in seen_targets:
            unique.append(arm)
            seen_targets.add(target)

    selected: list[ForcedRoleSelection] = []
    used: set[tuple[str, ...]] = set()

    def add(role: str, candidates: Sequence[FrozenArmPrediction]) -> bool:
        for index, arm in enumerate(candidates):
            target = tuple(sorted(arm.destroyed_operations))
            if target not in used:
                selected.append(ForcedRoleSelection(role, arm, index > 0))
                used.add(target)
                return True
        return False

    add("FROZEN_NEURAL_TOP1", unique[:1])
    add("FROZEN_NEURAL_TOP2", unique[1:])
    fallback_order = sorted(unique, key=lambda arm: (_fallback_priority(arm), *_score_order(arm)))
    add("ALNS_RELATED_FALLBACK", fallback_order)

    remaining = [
        arm for arm in unique if tuple(sorted(arm.destroyed_operations)) not in used
    ]
    if remaining:
        diverse_order = sorted(
            remaining,
            key=lambda arm: (
                -min(
                    _jaccard_distance(
                        arm.destroyed_operations, choice.arm.destroyed_operations
                    )
                    for choice in selected
                ),
                *_score_order(arm),
            ),
        )
        add("DETERMINISTIC_DIVERSE", diverse_order)

    requested_roles = (
        "FROZEN_NEURAL_TOP1",
        "FROZEN_NEURAL_TOP2",
        "ALNS_RELATED_FALLBACK",
        "DETERMINISTIC_DIVERSE",
    )
    present = {selection.role for selection in selected}
    for role in requested_roles:
        if role not in present:
            add(role, unique)
    role_position = {role: index for index, role in enumerate(requested_roles)}
    return tuple(sorted(selected, key=lambda row: role_position[row.role]))


def select_top_eight_audit_candidates(
    arms: Sequence[FrozenArmPrediction],
    broad_roles: Sequence[ForcedRoleSelection],
) -> tuple[FrozenArmPrediction, ...]:
    """Retain the four broad roles, then fill to eight by frozen score only."""
    selected: list[FrozenArmPrediction] = []
    seen: set[tuple[str, ...]] = set()
    for arm in [
        *(selection.arm for selection in broad_roles),
        *sorted(arms, key=_score_order),
    ]:
        target = tuple(sorted(arm.destroyed_operations))
        if target in seen:
            continue
        selected.append(arm)
        seen.add(target)
        if len(selected) == 8:
            break
    return tuple(selected)


def _transport_neighbor(
    instance: Instance,
    base: Candidate,
    removed_operations: Sequence[str],
    rng: random.Random,
) -> Candidate:
    """Frozen transport-aware repair with an explicit operation-ID tie break."""
    removed = set(removed_operations)
    retained = [operation for operation in base.operation_order if operation not in removed]
    removed_order = sorted(
        removed,
        key=lambda operation: (
            len(instance.operation_data[operation].eligible_islands), operation
        ),
    )
    for operation in removed_order:
        retained.insert(rng.randrange(len(retained) + 1), operation)
    islands = list(base.island_assignment)
    w_agvs = list(base.w_assignment)
    f_agvs = list(base.f_assignment)
    position = {operation: index for index, operation in enumerate(instance.operations)}
    for operation in removed_order:
        index = position[operation]
        eligible = instance.operation_data[operation].eligible_islands
        islands[index] = min(
            eligible,
            key=lambda island: sum(
                instance.f_outbound_time[(vehicle, island)]
                for vehicle in instance.agvs_f
            ),
        )
        w_agvs[index] = rng.choice(instance.agvs_w)
        f_agvs[index] = rng.choice(instance.agvs_f)
    return Candidate(
        tuple(retained), tuple(islands), tuple(w_agvs), tuple(f_agvs)
    )


def decode_forced_candidate(
    instance: Instance,
    current: DecodedCandidate,
    arm: FrozenArmPrediction,
    *,
    state_id: str,
    repair_seed_namespace: int,
    candidate_trials: int,
) -> ForcedDecodeResult:
    """Generate a post-selection label with isolated deterministic repair RNG."""
    if candidate_trials <= 0:
        raise ValueError("candidate_trials must be positive")
    repair_seed = stable_seed(
        state_id, "forced_transport_repair", namespace=repair_seed_namespace
    )
    rng = random.Random(repair_seed)
    started = time.perf_counter()
    candidates = tuple(
        decode_candidate(
            instance,
            _transport_neighbor(
                instance, current.candidate, arm.destroyed_operations, rng
            ),
        )
        for _ in range(candidate_trials)
    )
    best = min(candidates, key=lambda candidate: candidate.makespan)
    return ForcedDecodeResult(
        candidate=best,
        trial_makespans=tuple(candidate.makespan for candidate in candidates),
        repair_seed=repair_seed,
        decoder_evaluations=len(candidates),
        runtime_ms=(time.perf_counter() - started) * 1000.0,
    )


def continue_frozen_alns_from_candidate(
    instance: Instance,
    start: DecodedCandidate,
    *,
    state_id: str,
    continuation_seed: int,
    seed_namespace: int,
    iterations: int,
    config: ALNSConfig,
) -> ContinuationResult:
    """Run a fixed frozen-ALNS horizon from a supplied decoded candidate."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.candidate_trials <= 0:
        raise ValueError("candidate_trials must be positive")
    derived_seed = stable_seed(
        state_id,
        str(continuation_seed),
        namespace=seed_namespace,
    )
    rng = random.Random(derived_seed)
    started = time.perf_counter()
    current = start
    best = start
    weights = {name: 1.0 for name in (*DESTROY, *REPAIR)}
    selections: Counter[str] = Counter()
    temperature = config.initial_temperature * max(1.0, current.makespan)
    evaluations = 0
    accepted_moves = 0
    improving_moves = 0

    for _ in range(iterations):
        destroy = _roulette(DESTROY, weights, rng)
        repair = _roulette(REPAIR, weights, rng)
        selections.update((destroy, repair))
        count = min(
            max(2, round(instance.num_operations * config.destroy_fraction)),
            instance.num_operations,
        )
        removed = _destroy(instance, current, destroy, count, rng)
        candidates = tuple(
            decode_candidate(
                instance,
                _neighbor(instance, current.candidate, removed, repair, rng),
            )
            for _ in range(config.candidate_trials)
        )
        evaluations += len(candidates)
        candidate = min(candidates, key=lambda item: item.makespan)
        delta = candidate.makespan - current.makespan
        accepted = delta <= 0 or rng.random() < math.exp(
            -delta / max(temperature, 1e-12)
        )
        score = 0.0
        if accepted:
            current = candidate
            accepted_moves += 1
            score = 1.0
        if candidate.makespan < best.makespan:
            best = candidate
            improving_moves += 1
            score = 5.0
        for operator in (destroy, repair):
            weights[operator] = (
                (1 - config.reaction_factor) * weights[operator]
                + config.reaction_factor * max(score, 0.1)
            )
        temperature *= config.cooling_rate

    return ContinuationResult(
        candidate=best,
        start_makespan=float(start.makespan),
        best_makespan=float(best.makespan),
        continuation_value=float(
            (start.makespan - best.makespan) / start.makespan
        ),
        continuation_seed=int(continuation_seed),
        derived_seed=int(derived_seed),
        iterations=iterations,
        decoder_evaluations=evaluations,
        accepted_moves=accepted_moves,
        improving_moves=improving_moves,
        operator_selections=dict(selections),
        runtime_ms=(time.perf_counter() - started) * 1000.0,
    )
