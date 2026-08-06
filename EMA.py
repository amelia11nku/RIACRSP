"""Explicit-resource EMA for AFAISP.

This implementation searches the same four chromosome layers as BLCME/LMEO:

    (MS, TW, TF, OS)

AGV assignments are explicit decision variables and every candidate is
evaluated by simulate.Calculate.
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from data.loader import Instance, load_instance
from logger_config import setup_file_logger
from simulate import Calculate
from utils import (
    export_gantt_csv,
    gantt_dataframe_from_schedule_results,
    plot_gantt_three_swimlanes,
    save_best_solution,
    save_schedule_results,
)


logger = setup_file_logger("ema")

EPS = 1e-9

Individual = Tuple[int, float, List[int], List[int], List[int], List[int]]
Genes = Tuple[List[int], List[int], List[int], List[int]]


class EMA:
    """Memetic algorithm with explicit machine, AGV_W, AGV_F, and OS layers."""

    def __init__(
        self,
        instance: Instance,
        population_size: int = 150,
        max_iterations: int = 1000,
        early_stop_patience: int = 500,
        run_time_ratio: float = 2.0,
        pc: float = 0.8,
        pm: float = 0.25,
        local_search_rate: float = 0.40,
    ) -> None:
        self.instance = instance
        self.num_jobs = instance.num_jobs
        self.num_operations = instance.num_operations
        self.num_machines = instance.num_machines
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.job_operations = instance.job_operations
        self.processing_times = instance.processing_times
        self.priority_dict = instance.priority_dict

        self.population_size = max(2, int(population_size))
        self.max_iterations = int(max_iterations)
        self.early_stop_patience = int(early_stop_patience)
        self.max_run_time = float(instance.num_operations) * float(run_time_ratio)
        self.pc = float(pc)
        self.pm = float(pm)
        self.local_search_rate = float(local_search_rate)

        self.operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]
        self.op_id_to_index = {op_id: i + 1 for i, op_id in enumerate(self.operations_list)}
        self.op_index_to_id = {i + 1: op_id for i, op_id in enumerate(self.operations_list)}
        self.op_index_to_job = {
            i + 1: int(op_id.split("_", 1)[0][1:])
            for i, op_id in enumerate(self.operations_list)
        }
        self.available_machines = [
            [int(machine) for machine in self.processing_times[op_id].keys()]
            for op_id in self.operations_list
        ]

        self._successors: Dict[str, List[str]] = {op_id: [] for op_id in self.operations_list}
        self._in_degree0: Dict[str, int] = {op_id: 0 for op_id in self.operations_list}
        for op_id, predecessors in self.priority_dict.items():
            for predecessor in predecessors:
                self._successors[predecessor].append(op_id)
                self._in_degree0[op_id] += 1

        self.calculate = Calculate(instance)
        self.population: List[Individual] = []
        self.best_solution: Optional[Individual] = None
        self.best_generation = 0
        self._next_id = 1

    def _new_individual(
        self,
        machine: List[int],
        agv_w: List[int],
        agv_f: List[int],
        schedule: List[int],
    ) -> Individual:
        makespan = float(self.calculate.simulate(machine, agv_w, agv_f, schedule))
        individual = (
            self._next_id,
            makespan,
            machine.copy(),
            agv_w.copy(),
            agv_f.copy(),
            schedule.copy(),
        )
        self._next_id += 1
        return individual

    @staticmethod
    def _clone_individual(individual: Individual) -> Individual:
        return (
            individual[0],
            individual[1],
            individual[2].copy(),
            individual[3].copy(),
            individual[4].copy(),
            individual[5].copy(),
        )

    @staticmethod
    def _individual_key(individual: Individual) -> tuple:
        return (
            tuple(individual[2]),
            tuple(individual[3]),
            tuple(individual[4]),
            tuple(individual[5]),
        )

    def _random_topological_schedule(
        self, priorities: Optional[Dict[str, float]] = None
    ) -> List[int]:
        in_degree = dict(self._in_degree0)
        available = [op_id for op_id, degree in in_degree.items() if degree == 0]
        schedule: List[int] = []

        while available:
            if priorities is None:
                chosen = random.choice(available)
            else:
                best_value = min(priorities.get(op_id, 0.0) for op_id in available)
                tied = [
                    op_id
                    for op_id in available
                    if abs(priorities.get(op_id, 0.0) - best_value) <= EPS
                ]
                chosen = random.choice(tied)
            available.remove(chosen)
            schedule.append(self.op_id_to_index[chosen])
            for successor in self._successors[chosen]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    available.append(successor)

        if len(schedule) != self.num_operations:
            raise ValueError("The operation precedence graph contains a cycle.")
        return schedule

    def _repair_schedule(self, schedule: Sequence[int]) -> List[int]:
        rank = {int(op_index): pos for pos, op_index in enumerate(schedule)}
        priorities = {
            op_id: float(rank.get(self.op_id_to_index[op_id], self.num_operations))
            for op_id in self.operations_list
        }
        return self._random_topological_schedule(priorities)

    def _load_balanced_machine_string(self, schedule: Sequence[int]) -> List[int]:
        machine_string = [0] * self.num_operations
        loads = [0.0] * (self.num_machines + 1)
        for op_index in schedule:
            op_id = self.op_index_to_id[int(op_index)]
            machine = min(
                self.available_machines[int(op_index) - 1],
                key=lambda m: loads[m] + float(self.processing_times[op_id][m]),
            )
            machine_string[int(op_index) - 1] = int(machine)
            loads[machine] += float(self.processing_times[op_id][machine])
        return machine_string

    def initialize_population(self) -> None:
        self.population = []
        self.best_solution = None
        seen = set()
        attempts = 0

        while len(self.population) < self.population_size:
            schedule = self._random_topological_schedule()
            machine = self._load_balanced_machine_string(schedule)
            agv_w = [random.randint(1, self.num_agvs_w) for _ in range(self.num_operations)]
            agv_f = [random.randint(1, self.num_agvs_f) for _ in range(self.num_operations)]
            individual = self._new_individual(machine, agv_w, agv_f, schedule)
            key = self._individual_key(individual)
            attempts += 1
            if key in seen and attempts < self.population_size * 20:
                continue
            seen.add(key)
            self.population.append(individual)

        self.population.sort(key=lambda individual: individual[1])
        self.best_solution = self._clone_individual(self.population[0])
        logger.info(f"The initial best makespan is: {self.best_solution[1]:.2f}")

    def tournament_selection(self, population: Sequence[Individual], k: int = 2) -> Individual:
        candidates = random.sample(list(population), min(k, len(population)))
        return min(candidates, key=lambda individual: individual[1])

    def _resource_jcm(
        self, parent1: Sequence[int], parent2: Sequence[int], selected_job: int
    ) -> Tuple[List[int], List[int]]:
        child1 = [int(gene) for gene in parent1]
        child2 = [int(gene) for gene in parent2]
        for op_index in range(1, self.num_operations + 1):
            if self.op_index_to_job[op_index] == selected_job:
                pos = op_index - 1
                child1[pos], child2[pos] = child2[pos], child1[pos]
        return child1, child2

    def _pox_crossover(
        self, parent1: Sequence[int], parent2: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        if self.num_jobs < 2:
            return list(parent1), list(parent2)
        selected = {random.randint(1, self.num_jobs)}

        def build(keep: Sequence[int], fill: Sequence[int], keep_selected: bool) -> List[int]:
            child: List[Optional[int]] = [None] * self.num_operations
            for pos, op_index in enumerate(keep):
                belongs = self.op_index_to_job[int(op_index)] in selected
                if belongs == keep_selected:
                    child[pos] = int(op_index)
            remaining = [
                int(op_index)
                for op_index in fill
                if (self.op_index_to_job[int(op_index)] in selected) != keep_selected
            ]
            iterator = iter(remaining)
            for pos in range(self.num_operations):
                if child[pos] is None:
                    child[pos] = next(iterator)
            return self._repair_schedule([int(op_index) for op_index in child])

        return build(parent1, parent2, True), build(parent2, parent1, False)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Genes, Genes]:
        machine1, machine2 = parent1[2].copy(), parent2[2].copy()
        agv_w1, agv_w2 = parent1[3].copy(), parent2[3].copy()
        agv_f1, agv_f2 = parent1[4].copy(), parent2[4].copy()
        schedule1, schedule2 = parent1[5].copy(), parent2[5].copy()

        if random.random() < 0.5:
            schedule1, schedule2 = self._pox_crossover(parent1[5], parent2[5])
        else:
            selected_job = random.randint(1, self.num_jobs)
            machine1, machine2 = self._resource_jcm(parent1[2], parent2[2], selected_job)
            agv_w1, agv_w2 = self._resource_jcm(parent1[3], parent2[3], selected_job)
            agv_f1, agv_f2 = self._resource_jcm(parent1[4], parent2[4], selected_job)

        return (machine1, agv_w1, agv_f1, schedule1), (machine2, agv_w2, agv_f2, schedule2)

    def mutate(self, genes: Genes) -> Genes:
        machine, agv_w, agv_f, schedule = (layer.copy() for layer in genes)

        if self.num_operations > 1 and random.random() < 0.5:
            pos1, pos2 = random.sample(range(self.num_operations), 2)
            schedule[pos1], schedule[pos2] = schedule[pos2], schedule[pos1]
            schedule = self._repair_schedule(schedule)
        else:
            for pos in random.sample(range(self.num_operations), min(2, self.num_operations)):
                alternatives = [m for m in self.available_machines[pos] if m != machine[pos]]
                if alternatives:
                    machine[pos] = random.choice(alternatives)
                if self.num_agvs_w > 1:
                    agv_w[pos] = random.choice(
                        [a for a in range(1, self.num_agvs_w + 1) if a != agv_w[pos]]
                    )
                if self.num_agvs_f > 1:
                    agv_f[pos] = random.choice(
                        [a for a in range(1, self.num_agvs_f + 1) if a != agv_f[pos]]
                    )
        return machine, agv_w, agv_f, schedule

    def _critical_operations(self, individual: Individual) -> List[int]:
        self.calculate.simulate(*individual[2:])
        path = self.calculate.find_critical_path()
        return [self.op_id_to_index[operation.op_id] for operation in path]

    def _sequence_local_search(
        self, individual: Individual, critical_path: Sequence[int]
    ) -> Individual:
        if not critical_path:
            return individual
        candidate_schedule = individual[5].copy()
        selected = random.choice(list(critical_path))
        first = candidate_schedule.index(selected)
        second = random.randrange(self.num_operations)
        candidate_schedule[first], candidate_schedule[second] = (
            candidate_schedule[second],
            candidate_schedule[first],
        )
        candidate_schedule = self._repair_schedule(candidate_schedule)
        candidate = self._new_individual(
            individual[2], individual[3], individual[4], candidate_schedule
        )
        return candidate if candidate[1] + EPS < individual[1] else individual

    def _resource_local_search(
        self, individual: Individual, critical_path: Sequence[int]
    ) -> Individual:
        if not critical_path:
            return individual
        op_index = random.choice(list(critical_path))
        pos = op_index - 1
        op_id = self.op_index_to_id[op_index]

        alternatives = [
            machine
            for machine in self.available_machines[pos]
            if machine != individual[2][pos]
            and self.processing_times[op_id][machine]
            <= self.processing_times[op_id][individual[2][pos]]
        ]
        alternatives.sort(key=lambda machine: self.processing_times[op_id][machine])
        for machine in alternatives:
            candidate_machine = individual[2].copy()
            candidate_machine[pos] = machine
            candidate = self._new_individual(
                candidate_machine, individual[3], individual[4], individual[5]
            )
            if candidate[1] + EPS < individual[1]:
                return candidate

        for layer_index, upper_bound in ((3, self.num_agvs_w), (4, self.num_agvs_f)):
            current = individual[layer_index][pos]
            alternatives_agv = [a for a in range(1, upper_bound + 1) if a != current]
            random.shuffle(alternatives_agv)
            for agv in alternatives_agv:
                layers = [
                    individual[2].copy(),
                    individual[3].copy(),
                    individual[4].copy(),
                ]
                layers[layer_index - 2][pos] = agv
                candidate = self._new_individual(*layers, individual[5])
                if candidate[1] + EPS < individual[1]:
                    return candidate
        return individual

    def local_search(self, individual: Individual) -> Individual:
        critical_path = self._critical_operations(individual)
        improved = self._sequence_local_search(individual, critical_path)
        if improved[1] + EPS < individual[1]:
            return improved
        return self._resource_local_search(individual, critical_path)

    def _apply_elso(self, combined: Sequence[Individual]) -> List[Individual]:
        improved = [self._clone_individual(individual) for individual in combined]
        count = min(len(improved), max(1, int(len(improved) * self.local_search_rate)))
        for index in random.sample(range(len(improved)), count):
            improved[index] = self.local_search(improved[index])
        return improved

    def _population_update(self, combined: Sequence[Individual]) -> List[Individual]:
        pool = sorted(combined, key=lambda individual: individual[1])
        survivors: List[Individual] = []
        seen = set()
        for individual in pool:
            key = self._individual_key(individual)
            if key in seen:
                continue
            survivors.append(self._clone_individual(individual))
            seen.add(key)
            if len(survivors) >= self.population_size:
                break

        attempts = 0
        max_attempts = self.population_size * 50
        while len(survivors) < self.population_size and attempts < max_attempts:
            schedule = self._random_topological_schedule()
            machine = self._load_balanced_machine_string(schedule)
            agv_w = [random.randint(1, self.num_agvs_w) for _ in range(self.num_operations)]
            agv_f = [random.randint(1, self.num_agvs_f) for _ in range(self.num_operations)]
            immigrant = self._new_individual(machine, agv_w, agv_f, schedule)
            key = self._individual_key(immigrant)
            attempts += 1
            if key in seen:
                continue
            survivors.append(immigrant)
            seen.add(key)

        if len(survivors) < self.population_size:
            raise RuntimeError("Unable to replenish EMA with unique solutions.")
        return survivors

    def evolve(self) -> Tuple[Individual, List[float]]:
        self.initialize_population()
        assert self.best_solution is not None

        history = [self.best_solution[1]]
        early_stop_counter = 0
        start_time = time.time()

        for generation in range(self.max_iterations):
            if early_stop_counter >= self.early_stop_patience:
                logger.info(
                    f"Early stopping: no improvement for {self.early_stop_patience} generations."
                )
                break
            if time.time() - start_time > self.max_run_time:
                logger.info(f"Early stopping: runtime exceeded {self.max_run_time:.2f} seconds.")
                break

            offspring: List[Individual] = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                if random.random() < self.pc:
                    children = self.crossover(parent1, parent2)
                else:
                    children = (parent1[2:], parent2[2:])

                for genes in children:
                    if len(offspring) >= self.population_size:
                        break
                    cloned = tuple(layer.copy() for layer in genes)
                    if random.random() < self.pm:
                        cloned = self.mutate(cloned)
                    offspring.append(self._new_individual(*cloned))

            combined = self._apply_elso(self.population + offspring)
            self.population = self._population_update(combined)
            generation_best = min(self.population, key=lambda individual: individual[1])
            if generation_best[1] + EPS < self.best_solution[1]:
                self.best_solution = self._clone_individual(generation_best)
                self.best_generation = generation + 1
                early_stop_counter = 0
                logger.info(
                    f"In generation {generation + 1}, New best makespan: "
                    f"{self.best_solution[1]:.2f}"
                )
            else:
                early_stop_counter += 1
            history.append(self.best_solution[1])

        return self.best_solution, history


if __name__ == "__main__":
    instance = load_instance("AFAISP-M05")
    start_time = time.time()
    ema = EMA(instance=instance)
    best_solution, best_makespans = ema.evolve()

    makespan, schedule_results = ema.calculate.simulate(*best_solution[2:], return_schedule=True)
    output_dir = os.path.join("output", "ema_output")
    os.makedirs(output_dir, exist_ok=True)

    save_schedule_results(
        schedule_results,
        os.path.join(output_dir, "agv_w_schedule.csv"),
        os.path.join(output_dir, "agv_f_schedule.csv"),
        os.path.join(output_dir, "machine_schedule.csv"),
    )
    order = [ema.op_index_to_id[index] for index in best_solution[5]]
    save_best_solution(
        best_solution,
        order,
        os.path.join(output_dir, "best_solution.csv"),
        ema.best_generation,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_makespans)), best_makespans)
    plt.title("EMA Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ema_convergence.png"))
    plt.close()

    gantt_df = gantt_dataframe_from_schedule_results(schedule_results)
    export_gantt_csv(gantt_df, os.path.join(output_dir, "gantt_data.csv"))
    plot_gantt_three_swimlanes(
        gantt_df,
        title=f"EMA Schedule Gantt Chart, Cmax={makespan:.2f}",
        figsize=(14, 7),
        save_path=os.path.join(output_dir, "gantt.png"),
        dpi=300,
        show=False,
    )
    logger.info(
        f"Best makespan: {best_solution[1]:.2f}; generation: {ema.best_generation}; "
        f"time: {time.time() - start_time:.2f}s"
    )
