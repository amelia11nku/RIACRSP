"""Conventional adaptive large-neighborhood search seeded by frozen H1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.heuristic.dispatching import solve_dispatching

from .common import Candidate, DecodedCandidate, SearchResult, TracePoint, candidate_from_actions, decode_candidate


DESTROY = ("random", "critical", "overloaded_island", "high_reconfiguration", "w_bottleneck", "f_bottleneck", "related")
REPAIR = ("greedy", "regret2", "regret3", "reconfiguration_aware", "transport_aware")


@dataclass(frozen=True)
class ALNSConfig:
    initial_temperature: float = 0.05
    cooling_rate: float = 0.995
    destroy_fraction: float = 0.15
    reaction_factor: float = 0.2
    candidate_trials: int = 8
    iteration_limit: int | None = None


ALNSObserver = Callable[[Mapping[str, object]], None]


def _roulette(names, weights, rng):
    return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]


def _destroy(instance, decoded, name, count, rng):
    schedule = decoded.schedule
    operations = list(instance.operations)
    if name == "random":
        ranked = rng.sample(operations, len(operations))
    elif name == "critical":
        ranked = sorted(operations, key=lambda op: schedule.operation_schedules[op].completion_time, reverse=True)
    elif name == "overloaded_island":
        load = Counter(record.island_id for record in schedule.operation_schedules.values())
        ranked = sorted(operations, key=lambda op: load[schedule.operation_schedules[op].island_id], reverse=True)
    elif name == "high_reconfiguration":
        ranked = sorted(operations, key=lambda op: schedule.operation_schedules[op].reconfiguration_end - schedule.operation_schedules[op].reconfiguration_start, reverse=True)
    elif name == "w_bottleneck":
        vehicle = max(schedule.w_timelines, key=lambda key: len(schedule.w_timelines[key]))
        preferred = {task.operation_id for task in schedule.w_timelines[vehicle]}
        ranked = sorted(operations, key=lambda op: op not in preferred)
    elif name == "f_bottleneck":
        vehicle = max(schedule.f_timelines, key=lambda key: len(schedule.f_timelines[key]))
        preferred = {task.operation_id for task in schedule.f_timelines[vehicle]}
        ranked = sorted(operations, key=lambda op: op not in preferred)
    else:
        pivot = rng.choice(operations)
        product = instance.product_of[pivot]
        ranked = sorted(operations, key=lambda op: (instance.product_of[op] != product, abs(operations.index(op) - operations.index(pivot))))
    return set(ranked[:count])


def _neighbor(instance: Instance, base: Candidate, removed: set[str], repair: str, rng: random.Random) -> Candidate:
    retained = [op for op in base.operation_order if op not in removed]
    removed_order = list(removed)
    if repair == "regret2":
        removed_order.sort(key=lambda op: len(instance.operation_data[op].eligible_islands))
    elif repair == "regret3":
        removed_order.sort(key=lambda op: len(instance.transitive_successors[op]), reverse=True)
    elif repair == "reconfiguration_aware":
        removed_order.sort(key=lambda op: instance.operation_data[op].required_config)
    elif repair == "transport_aware":
        removed_order.sort(key=lambda op: len(instance.operation_data[op].eligible_islands))
    else:
        rng.shuffle(removed_order)
    for operation in removed_order:
        retained.insert(rng.randrange(len(retained) + 1), operation)
    islands = list(base.island_assignment)
    w_agvs = list(base.w_assignment)
    f_agvs = list(base.f_assignment)
    position = {op: i for i, op in enumerate(instance.operations)}
    for operation in removed:
        index = position[operation]
        eligible = instance.operation_data[operation].eligible_islands
        if repair == "reconfiguration_aware":
            config = instance.operation_data[operation].required_config
            islands[index] = min(eligible, key=lambda island: instance.reconfiguration_time[(island, instance.island_data[island].initial_config, config)])
        elif repair == "transport_aware":
            islands[index] = min(eligible, key=lambda island: sum(instance.f_outbound_time[(f, island)] for f in instance.agvs_f))
        else:
            islands[index] = rng.choice(eligible)
        w_agvs[index] = rng.choice(instance.agvs_w)
        f_agvs[index] = rng.choice(instance.agvs_f)
    return Candidate(tuple(retained), tuple(islands), tuple(w_agvs), tuple(f_agvs))


def solve_alns(
    instance: Instance,
    time_limit: float,
    seed: int,
    config: ALNSConfig = ALNSConfig(),
    observer: ALNSObserver | None = None,
) -> SearchResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    best = current
    evaluations = 1
    best_time = time.perf_counter() - started
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    weights = {name: 1.0 for name in (*DESTROY, *REPAIR)}
    selections, successes, improvements = Counter(), Counter(), Counter()
    temperature = config.initial_temperature * max(1.0, current.makespan)
    iterations = 0
    while (
        time.perf_counter() - started < time_limit
        and (config.iteration_limit is None or iterations < config.iteration_limit)
    ):
        iteration_started = time.perf_counter()
        current_before = current
        best_before = best
        evaluations_before = evaluations
        weights_before = dict(weights)
        destroy = _roulette(DESTROY, weights, rng)
        repair = _roulette(REPAIR, weights, rng)
        selections.update((destroy, repair))
        count = max(2, round(instance.num_operations * config.destroy_fraction))
        removed = _destroy(instance, current, destroy, min(count, instance.num_operations), rng)
        candidates = []
        repair_started = time.perf_counter()
        for _ in range(config.candidate_trials):
            if time.perf_counter() - started >= time_limit and candidates:
                break
            candidates.append(decode_candidate(instance, _neighbor(instance, current.candidate, removed, repair, rng)))
            evaluations += 1
        candidate = min(candidates, key=lambda item: item.makespan)
        repair_runtime = time.perf_counter() - repair_started
        delta = candidate.makespan - current.makespan
        accepted = delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12))
        score = 0.0
        if accepted:
            current = candidate
            successes.update((destroy, repair))
            score = 1.0
        if candidate.makespan < best.makespan:
            best = candidate
            best_time = time.perf_counter() - started
            improvements.update((destroy, repair))
            trace.append(TracePoint(best_time, evaluations, best.makespan))
            score = 5.0
        for operator in (destroy, repair):
            weights[operator] = (1 - config.reaction_factor) * weights[operator] + config.reaction_factor * max(score, 0.1)
        temperature *= config.cooling_rate
        if observer is not None:
            observer({
                "iteration": iterations,
                "elapsed_time": time.perf_counter() - started,
                "iteration_runtime": time.perf_counter() - iteration_started,
                "decoder_evaluations": evaluations,
                "current_before": current_before,
                "best_before": best_before,
                "candidate": candidate,
                "current_after": current,
                "best_after": best,
                "destroy_operator": destroy,
                "repair_operator": repair,
                "destroy_fraction": config.destroy_fraction,
                "destroyed_operation_ids": tuple(sorted(removed)),
                "accepted": accepted,
                "new_global_best": candidate.makespan < best_before.makespan,
                "operator_weights_before": weights_before,
                "operator_weights_after": dict(weights),
                "repair_decoder_evaluations": evaluations - evaluations_before,
                "repair_runtime": repair_runtime,
                "candidate_trials_completed": len(candidates),
                "temperature_before": temperature / config.cooling_rate,
            })
        iterations += 1
    return SearchResult(
        "ALNS-H1", best, best_time, time.perf_counter() - started, evaluations, iterations, None, tuple(trace),
        {"operator_selections": dict(selections), "operator_successes": dict(successes), "operator_improvements": dict(improvements), "final_adaptive_weights": weights},
    )
