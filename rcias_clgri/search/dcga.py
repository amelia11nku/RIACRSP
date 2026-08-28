"""Faithful RIACRSP adaptation of Han et al.'s 2024 DCGA."""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from rcias_clgri.data.instance import Instance
from .common import Candidate, DecodedCandidate, SearchResult, TracePoint, decode_candidate, random_candidate


@dataclass(frozen=True)
class DCGAConfig:
    population_size_each: int = 500
    crossover_probability: float = 0.9
    mutation_probability: float = 0.15
    diversity_check_interval: int = 300
    similarity_threshold: float = 0.8
    elite_count: int = 2


def _pox(instance: Instance, left: tuple[str, ...], right: tuple[str, ...], rng: random.Random):
    products = list(instance.products)
    rng.shuffle(products)
    split = rng.randrange(1, len(products)) if len(products) > 1 else 1
    group1, group2 = set(products[:split]), set(products[split:])

    def child(fixed_parent, fill_parent, fixed_products):
        result: list[str | None] = [None] * len(fixed_parent)
        for index, operation in enumerate(fixed_parent):
            if instance.product_of[operation] in fixed_products:
                result[index] = operation
        fill = iter(op for op in fill_parent if instance.product_of[op] not in fixed_products)
        return tuple(next(fill) if operation is None else operation for operation in result)

    return child(left, right, group1), child(right, left, group2)


def _uniform(left, right, rng):
    mask = [rng.random() < 0.5 for _ in left]
    return (tuple(a if m else b for a, b, m in zip(left, right, mask)),
            tuple(b if m else a for a, b, m in zip(left, right, mask)))


def _crossover(instance, left, right, rng):
    order1, order2 = _pox(instance, left.operation_order, right.operation_order, rng)
    island1, island2 = _uniform(left.island_assignment, right.island_assignment, rng)
    return (Candidate(order1, island1, left.w_assignment, left.f_assignment),
            Candidate(order2, island2, right.w_assignment, right.f_assignment))


def _mutate(instance, candidate, rng):
    order = list(candidate.operation_order)
    left, right = sorted(rng.sample(range(len(order)), 2))
    operator = rng.choice(("swap", "insert", "inversion"))
    if operator == "swap":
        order[left], order[right] = order[right], order[left]
    elif operator == "insert":
        order.insert(left, order.pop(right))
    else:
        order[left:right + 1] = reversed(order[left:right + 1])
    islands = list(candidate.island_assignment)
    index = rng.randrange(instance.num_operations)
    islands[index] = rng.choice(instance.operation_data[instance.operations[index]].eligible_islands)
    return Candidate(tuple(order), tuple(islands), candidate.w_assignment, candidate.f_assignment)


def _tournament(population, rng):
    return min(rng.sample(population, 2), key=lambda item: item.makespan)


def _decode(instance, candidate, population_index):
    return decode_candidate(instance, candidate, "fixed" if population_index == 0 else "cumulative")


def _diversity_check(instance, population, population_index, rng, threshold):
    regenerated = 0
    population.sort(key=lambda item: item.makespan)
    for right in range(1, len(population)):
        for left in range(right):
            if population[left].makespan != population[right].makespan:
                continue
            similarity = sum(a == b for a, b in zip(
                population[left].candidate.island_assignment,
                population[right].candidate.island_assignment)) / instance.num_operations
            if similarity >= threshold:
                population[right] = _decode(instance, random_candidate(instance, rng), population_index)
                regenerated += 1
                break
    return regenerated


def solve_dcga(instance: Instance, time_limit: float, seed: int, config: DCGAConfig = DCGAConfig()) -> SearchResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    populations = [[_decode(instance, random_candidate(instance, rng), index)
                    for _ in range(config.population_size_each)] for index in range(2)]
    evaluations = 2 * config.population_size_each
    best = min((*populations[0], *populations[1]), key=lambda item: item.makespan)
    best_time = time.perf_counter() - started
    trace = [TracePoint(best_time, evaluations, best.makespan)]
    generations = collaborations = regenerated = 0

    def record(item):
        nonlocal best, best_time
        if item.makespan < best.makespan:
            best, best_time = item, time.perf_counter() - started
            trace.append(TracePoint(best_time, evaluations, best.makespan))

    while time.perf_counter() - started < time_limit:
        next_populations = []
        for population_index, population in enumerate(populations):
            population.sort(key=lambda item: item.makespan)
            offspring = list(population[:config.elite_count])
            while len(offspring) < config.population_size_each:
                parent1, parent2 = _tournament(population, rng), _tournament(population, rng)
                children = (_crossover(instance, parent1.candidate, parent2.candidate, rng)
                            if rng.random() < config.crossover_probability else (parent1.candidate, parent2.candidate))
                for child in children:
                    if rng.random() < config.mutation_probability:
                        child = _mutate(instance, child, rng)
                    decoded = _decode(instance, child, population_index)
                    evaluations += 1
                    offspring.append(decoded)
                    record(decoded)
                    if len(offspring) == config.population_size_each:
                        break
            next_populations.append(offspring)
        populations = next_populations
        generations += 1
        if generations % config.diversity_check_interval == 0:
            for index in range(2):
                count = _diversity_check(instance, populations[index], index, rng, config.similarity_threshold)
                regenerated += count
                evaluations += count
        for _ in range(config.population_size_each):
            parent1, parent2 = _tournament(populations[0], rng), _tournament(populations[1], rng)
            order1, order2 = _pox(instance, parent1.candidate.operation_order, parent2.candidate.operation_order, rng)
            child1 = Candidate(order1, parent1.candidate.island_assignment, parent1.candidate.w_assignment, parent1.candidate.f_assignment)
            child2 = Candidate(order2, parent2.candidate.island_assignment, parent2.candidate.w_assignment, parent2.candidate.f_assignment)
            decoded1, decoded2 = _decode(instance, child1, 0), _decode(instance, child2, 1)
            evaluations += 2
            if decoded1.makespan < parent1.makespan:
                populations[0][populations[0].index(parent1)] = decoded1
                record(decoded1)
            if decoded2.makespan < parent2.makespan:
                populations[1][populations[1].index(parent2)] = decoded2
                record(decoded2)
            collaborations += 1
            if time.perf_counter() - started >= time_limit:
                break
    pathway_best = {"Decoding1_fixed": min(x.makespan for x in populations[0]),
                    "Decoding2_cumulative": min(x.makespan for x in populations[1])}
    return SearchResult("Adapted DCGA", best, best_time, time.perf_counter() - started, evaluations,
                        generations, generations, tuple(trace), {
                            "fidelity": "FAITHFUL_ADAPTATION", "population_pathway_best": pathway_best,
                            "collaboration_trials": collaborations, "diversity_regenerations": regenerated})


solve_dcga_inspired = solve_dcga
DCGAInspiredConfig = DCGAConfig
