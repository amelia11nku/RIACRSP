"""Conventional problem-adapted genetic algorithm baseline."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

from rcias_clgri.data.instance import Instance

from .common import Candidate, SearchResult, TracePoint, decode_candidate, random_candidate


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    elite_fraction: float = 0.05
    tournament_size: int = 3


def _order_crossover(left: tuple[str, ...], right: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    first, last = sorted(rng.sample(range(len(left) + 1), 2))
    fixed = set(left[first:last])
    remainder = iter(item for item in right if item not in fixed)
    return tuple(next(remainder) if not (first <= i < last) else left[i] for i in range(len(left)))


def _uniform(left: tuple[str, ...], right: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    return tuple(a if rng.random() < 0.5 else b for a, b in zip(left, right))


def _crossover(left: Candidate, right: Candidate, rng: random.Random) -> Candidate:
    return Candidate(
        _order_crossover(left.operation_order, right.operation_order, rng),
        _uniform(left.island_assignment, right.island_assignment, rng),
        _uniform(left.w_assignment, right.w_assignment, rng),
        _uniform(left.f_assignment, right.f_assignment, rng),
    )


def _mutate(instance: Instance, candidate: Candidate, rng: random.Random) -> Candidate:
    order = list(candidate.operation_order)
    source, target = rng.sample(range(len(order)), 2)
    order.insert(target, order.pop(source))
    islands = list(candidate.island_assignment)
    w_agvs = list(candidate.w_assignment)
    f_agvs = list(candidate.f_assignment)
    index = rng.randrange(instance.num_operations)
    operation = instance.operations[index]
    layer = rng.randrange(3)
    if layer == 0:
        islands[index] = rng.choice(instance.operation_data[operation].eligible_islands)
    elif layer == 1:
        w_agvs[index] = rng.choice(instance.agvs_w)
    else:
        f_agvs[index] = rng.choice(instance.agvs_f)
    return Candidate(tuple(order), tuple(islands), tuple(w_agvs), tuple(f_agvs))


def solve_ga(instance: Instance, time_limit: float, seed: int, config: GAConfig = GAConfig()) -> SearchResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    population = [random_candidate(instance, rng) for _ in range(config.population_size)]
    evaluated = [decode_candidate(instance, item) for item in population]
    evaluations = len(evaluated)
    best = min(evaluated, key=lambda item: item.makespan)
    best_time = time.perf_counter() - started
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    generation = 0
    elite_count = max(1, round(config.population_size * config.elite_fraction))

    def tournament():
        return min(rng.sample(evaluated, min(config.tournament_size, len(evaluated))), key=lambda x: x.makespan)

    while time.perf_counter() - started < time_limit:
        evaluated.sort(key=lambda item: item.makespan)
        next_population = [item.candidate for item in evaluated[:elite_count]]
        while len(next_population) < config.population_size:
            parent_a, parent_b = tournament().candidate, tournament().candidate
            child = _crossover(parent_a, parent_b, rng) if rng.random() < config.crossover_probability else parent_a
            if rng.random() < config.mutation_probability:
                child = _mutate(instance, child, rng)
            next_population.append(child)
        evaluated = []
        for candidate in next_population:
            if time.perf_counter() - started >= time_limit and evaluated:
                break
            decoded = decode_candidate(instance, candidate)
            evaluations += 1
            evaluated.append(decoded)
            if decoded.makespan < best.makespan:
                best = decoded
                best_time = time.perf_counter() - started
                trace.append(TracePoint(best_time, evaluations, best.makespan))
        generation += 1
    runtime = time.perf_counter() - started
    return SearchResult("GA", best, best_time, runtime, evaluations, generation, generation, tuple(trace), {})
