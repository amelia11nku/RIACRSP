"""
基于学习的双种群协同模因进化算法
- 基于启发式规则的机器分配和调度序列初始种群
- 开发种群: 基于Q学习的局部搜索算法
- 探索种群: 基于Q学习的扰动算子选择
- Q学习参数自适应
"""
from __future__ import annotations
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from data.loader import Instance, load_instance
from simulate import Calculate
from simulate_dcga import Decoder
from utils import (save_schedule_results, save_best_solution, 
                gantt_dataframe_from_schedule_results, export_gantt_csv, 
                plot_gantt_three_swimlanes)
from Exploiter_LS import ExploiterLocalSearch, QLearningSelector
from Explorer_PB import ExplorerPerturbation, ExplorerQLearning
from logger_config import setup_file_logger

logger = setup_file_logger('BLCMA')

class BLCMA:
    def __init__(self, instance: Instance | None = None,
                 population_size: int = 150, max_iterations: int = 1000, early_stop_patience: int = 500,
                 run_time_ratio: float = 2.0, explorer_ratio: float = 0.3, pc_e: float = 0.95, pm_e: float = 0.30, pc_x: float = 0.95, pm_x: float = 0.30, 
                 elite_ratio: float = 0.05, migration_interval: int = 10, migration_rate: float = 0.10,
                 q_alpha_e: float = 0.2, q_epsilon_e: float = 0.1, q_alpha_x: float = 0.2, q_epsilon_x: float = 0.1, gamma_e: float = 0.3, gamma_x: float = 0.3,
                 ls_budget: int = 12):
        if instance is None:
            instance = load_instance("AFAISP-S01")

        self.num_jobs = instance.num_jobs
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.num_operations = instance.num_operations
        self.num_machines = instance.num_machines
        self.job_operations = instance.job_operations
        self.processing_times = instance.processing_times
        self.tt_w = instance.agv_w_transport_times
        self.tt_f = instance.agv_f_transport_times
        self.priority_dict = instance.priority_dict
        
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.early_stop_patience = early_stop_patience
        self.max_run_time = self.num_operations * run_time_ratio
        # self.max_run_time = 3600
        self.explorer_size = max(1, min(population_size - 1, int(population_size * explorer_ratio)))
        self.exploiter_size = population_size - self.explorer_size
        self.explorer_pc, self.explorer_pm = pc_e, pm_e
        self.exploiter_pc, self.exploiter_pm = pc_x, pm_x
        self.elite_ratio = elite_ratio
        self.explorer_elite_count = max(1, int(self.explorer_size * elite_ratio))
        self.exploiter_elite_count = max(1, int(self.exploiter_size * elite_ratio))
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        
        self.explorer_population = []
        self.exploiter_population = []
        self.best_solution = None
        self.best_generation = 0
        
        self.operations_list = [f"o{j}_{s}" for j in range(1, self.num_jobs + 1) for s in range(1, self.job_operations[j] + 1)]
        self.op_index_to_id = {i + 1: op for i, op in enumerate(self.operations_list)}
        self.op_id_to_flat0 = {op: i for i, op in enumerate(self.operations_list)}
        self.op_id_to_index = {op: idx for idx, op in self.op_index_to_id.items()}
        
        self.calculate = Calculate(instance)
        self.decoder = Decoder(instance)

        self.explorer_pert = ExplorerPerturbation(instance, self.op_index_to_id, self.op_id_to_index)
        self.explorer_q = ExplorerQLearning(n_actions=12, alpha=q_alpha_e, gamma=gamma_e, epsilon=q_epsilon_e)

        self.exploiter_ls = ExploiterLocalSearch(instance, ls_budget=ls_budget)
        self.exploiter_q = QLearningSelector(n_actions=12, alpha=q_alpha_x, gamma=gamma_x, epsilon=q_epsilon_x)
        
        

    def initialize_populations(self):
        explorer_random_size = int(self.explorer_size * 0.7)
        explorer_heuristic_size = self.explorer_size - explorer_random_size
        exploiter_heuristic_size = int(self.exploiter_size * 0.7)
        exploiter_random_size = self.exploiter_size - exploiter_heuristic_size

        explorer_candidates = (
            self.initialize_population_random(explorer_random_size)
            + self.initialize_population_heuristic(explorer_heuristic_size)
        )
        exploiter_candidates = (
            self.initialize_population_heuristic(exploiter_heuristic_size)
            + self.initialize_population_random(exploiter_random_size)
        )

        self.explorer_population = self._deduplicate_and_fill_population(
            explorer_candidates, self.explorer_size, "explorer"
        )
        self.exploiter_population = self._deduplicate_and_fill_population(
            exploiter_candidates, self.exploiter_size, "exploiter"
        )
        self.explorer_population.sort(key=lambda x: x[1])
        self.exploiter_population.sort(key=lambda x: x[1])
        population = self.explorer_population + self.exploiter_population
        population.sort(key=lambda x: x[1])
        self.best_solution = population[0]

    def _deduplicate_and_fill_population(self, population, target_size, population_name):
        """Keep unique complete chromosomes and fill any shortage with random solutions."""
        unique_population = []
        seen = set()

        for individual in sorted(population, key=lambda x: x[1]):
            key = self._individual_key(individual)
            if key in seen:
                continue
            seen.add(key)
            unique_population.append(individual)
            if len(unique_population) >= target_size:
                break

        duplicate_count = len(population) - len(unique_population)
        attempts = 0
        max_attempts = max(100, target_size * 100)
        while len(unique_population) < target_size and attempts < max_attempts:
            attempts += 1
            candidates = self.initialize_population_random(1)
            if not candidates:
                continue
            candidate = candidates[0]
            key = self._individual_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_population.append(candidate)

        if len(unique_population) < target_size:
            raise RuntimeError(
                f"Could not initialize {target_size} unique {population_name} individuals "
                f"after {max_attempts} random refill attempts."
            )

        logger.debug(
            f"Initialized {population_name} population with {target_size} unique individuals; "
            f"removed {duplicate_count} duplicates and added "
            f"{max(0, target_size - (len(population) - duplicate_count))} random replacements."
        )
        return [
            (index + 1, individual[1], individual[2], individual[3], individual[4], individual[5])
            for index, individual in enumerate(unique_population)
        ]

    def initialize_population_heuristic(self, h_size) -> list:
        pop_h = []
        for i in range(h_size):
            OS = self._generate_schedule_by_heuristic()
            MS = self._generate_machine_by_location_aware(OS)
            # 动态选择最早可用AGV（若有多个可用则选择能最早到达运送目的地的）
            mk, TW, TF = self.decoder.simulate(MS, OS, return_agv_strings=True, strategy=2)
            if mk is not None: pop_h.append((i + 1, mk, MS, TW, TF, OS))
        return pop_h

    def initialize_population_random(self, r_size) -> list:
        pop_r = []
        for i in range(r_size):
            OS = self._generate_random_schedule()
            MS = self._generate_random_machine()
            TF = [random.randint(1, self.num_agvs_f) for _ in range(self.num_operations)]
            TW = [random.randint(1, self.num_agvs_w) for _ in range(self.num_operations)]
            mk = self.calculate.simulate(MS, TW, TF, OS)
            if mk is not None: pop_r.append((i + 1, mk, MS, TW, TF, OS))
        return pop_r

    def _generate_random_schedule(self) -> list:
        graph = {op: set() for op in self.operations_list}
        in_degree = {op: 0 for op in self.operations_list}
        for op, preds in self.priority_dict.items():
            for pred in preds:
                graph[pred].add(op)
                in_degree[op] += 1
        OS = []
        queue = [op for op, deg in in_degree.items() if deg == 0]
        while queue:
            random.shuffle(queue)
            op = queue.pop(0)
            OS.append(self.op_id_to_index[op])
            for succ in sorted(graph[op]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0: queue.append(succ)
        if len(OS) != self.num_operations:
            raise ValueError("Priority graph contains a cycle or references missing operations.")
        return OS

    def _generate_random_machine(self) -> list:
        machines = []
        for op in self.operations_list:
            alternatives = list(self.processing_times.get(op, {}).keys())
            if not alternatives:
                raise ValueError(f"Operation {op} has no eligible machine.")
            machines.append(random.choice(alternatives))
        return machines

    def _generate_schedule_by_heuristic(self) -> list:
        graph = {op: set() for op in self.operations_list}
        in_degree = {op: 0 for op in self.operations_list}
        for op, preds in self.priority_dict.items():
            for pred in preds:
                graph[pred].add(op); in_degree[op] += 1
        OS, scheduled_ops = [], set()
        queue = [op for op, deg in in_degree.items() if deg == 0]
        while queue:
            job_remain = {}
            for op in self.operations_list:
                if op not in scheduled_ops:
                    job_id = int(op.split('_')[0][1:])
                    job_remain[job_id] = job_remain.get(job_id, 0) + 1
            job_ops = {}
            for op in queue: job_ops.setdefault(int(op.split('_')[0][1:]), []).append(op)
            
            c_job = max((j for j in job_ops if j in job_remain), key=lambda j: job_remain[j], default=None)
            if c_job:
                ops = job_ops[c_job]; random.shuffle(ops)
                op = ops[0]
                queue.remove(op)
                OS.append(self.op_id_to_index[op])
                scheduled_ops.add(op)
                for succ in sorted(graph[op]):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0: queue.append(succ)
        if len(OS) != self.num_operations:
            raise ValueError("Priority graph contains a cycle or references missing operations.")
        return OS

    def _generate_machine_by_location_aware(self, OS) -> list:
        MS = [0] * self.num_operations
        job_last_machine = {}
        for op in [self.op_index_to_id[idx] for idx in OS]:
            job_id = int(op.split("_")[0][1:])
            idx = self.op_id_to_index[op] - 1
            alts = self.processing_times.get(op, {})
            if not alts:
                raise ValueError(f"Operation {op} has no eligible machine.")
            
            prev = job_last_machine.get(job_id)
            prev_key = f"M{prev}" if prev else None
            
            best_m, min_cost = None, float("inf")
            for m_id, pt in alts.items():
                m_key = f"M{m_id}"
                tc = max(self.tt_w[prev_key][m_key], self.tt_f["Warehouse"][m_key]) if prev else \
                     max(self.tt_w["Warehouse"][m_key], self.tt_f["Warehouse"][m_key])
                if pt + tc < min_cost:
                    min_cost, best_m = pt + tc, m_id
            MS[idx] = job_last_machine[job_id] = best_m
        return MS

    def tournament_selection(self, population):
        cands = random.sample(range(len(population)), min(2, len(population)))
        return min(cands, key=lambda i: population[i][1])

    def crossover_assignment_strings(self, p1, p2, is_machine=False):
        child = []
        for i, op in enumerate(self.operations_list):
            if is_machine:
                alts = list(self.processing_times[op].keys())
                child.append(p1[i] if random.random()<0.5 and p1[i] in alts else (p2[i] if p2[i] in alts else int(random.choice(alts))))
            else: child.append(p1[i] if random.random()<0.5 else p2[i])
        return child

    def crossover_OS(self, p1, p2):
        if self.num_jobs <= 1:
            return p1.copy()
        jp1 = set(random.sample(range(1, self.num_jobs + 1), random.randint(1, self.num_jobs - 1)))
        child, empty = [-1]*len(p1), []
        for i, op in enumerate(p1):
            if int(self.op_index_to_id[op].split('_')[0][1:]) in jp1: child[i] = op
            else: empty.append(i)
        p2_ops = [op for op in p2 if int(self.op_index_to_id[op].split('_')[0][1:]) not in jp1]
        for i, op in zip(empty, p2_ops): child[i] = op
        return child if self.explorer_pert._is_valid_OS(child) else p1.copy()

    def mutate_assignment_string(self, s, max_val, is_machine=False):
        mut = s.copy()
        i = random.randrange(len(mut))
        if is_machine: mut[i] = int(random.choice(list(self.processing_times[self.operations_list[i]].keys())))
        else: mut[i] = random.randint(1, max_val)
        return mut

    def mutate_OS(self, OS):
        for _ in range(3):
            mut = OS.copy()
            i, j = random.sample(range(len(mut)), 2)
            mut[i], mut[j] = mut[j], mut[i]
            try:
                self.exploiter_ls._check_precedence_ops([self.op_index_to_id[x] for x in mut], i, j)
                return mut
            except ValueError: pass
        return OS.copy()

    def _individual_key(self, ind):
        return (tuple(ind[2]), tuple(ind[3]), tuple(ind[4]), tuple(ind[5]))

    def _average_layer_distance(self, x, refs):
        if not refs:
            return 0.0
        dist_sum = 0.0
        for ref in refs:
            dist_sum += 0.25 * sum(self.explorer_q.calculate_layer_distances(x, ref, self.num_operations))
        return dist_sum / len(refs)

    def _build_explorer_elites(self, explorer_pop, gbest=None):
        k = self.explorer_elite_count
        global_k = max(1, k // 2)
        explorer_k = k - global_k

        combined = sorted(self.explorer_population + self.exploiter_population, key=lambda x: x[1])
        elites, seen = [], set()
        if gbest is not None:
            elites.append(gbest)
            seen.add(self._individual_key(gbest))

        for ind in combined:
            key = self._individual_key(ind)
            if key in seen:
                continue
            elites.append(ind)
            seen.add(key)
            if len(elites) >= global_k:
                break

        if explorer_k <= 0:
            return elites[:k]

        explorer_candidates = sorted(explorer_pop, key=lambda x: x[1])
        cand_mks = [ind[1] for ind in explorer_candidates]
        min_mk = min(cand_mks) if cand_mks else 0.0
        max_mk = max(cand_mks) if cand_mks else min_mk
        selected_explorer = []

        while len(selected_explorer) < explorer_k and explorer_candidates:
            best_pos, best_score = None, -float("inf")
            refs = elites + selected_explorer
            for pos, ind in enumerate(explorer_candidates):
                key = self._individual_key(ind)
                if key in seen:
                    continue
                quality = 1.0 - (ind[1] - min_mk) / (max_mk - min_mk + 1e-9)
                diversity = self._average_layer_distance(ind[2:], refs)
                score = 0.5 * quality + 0.5 * diversity
                if score > best_score:
                    best_pos, best_score = pos, score
            if best_pos is None:
                break
            ind = explorer_candidates.pop(best_pos)
            selected_explorer.append(ind)
            seen.add(self._individual_key(ind))

        return (elites + selected_explorer)[:k]

    def _explorer_reward_weights(self, generation):
        elapsed = time.time() - getattr(self, "_run_start_time", time.time())
        progress = min(1.0, max(0.0, elapsed / max(self.max_run_time, 1e-9)))
        novelty_w = 0.70 - 0.50 * progress
        quality_w = 1.0 - novelty_w
        return quality_w, novelty_w

    def _normalized_explorer_reward(self, q_gain, nov_gain, generation):
        quality_w, novelty_w = self._explorer_reward_weights(generation)
        q_norm = float(np.clip(q_gain / 0.05, -1.0, 1.0))
        nov_norm = float(np.clip(nov_gain / 0.10, -1.0, 1.0))
        return quality_w * q_norm + novelty_w * nov_norm

    def _accept_explorer_candidate(self, mk, nmk, reward, generation):
        if nmk < mk:
            return True
        _, novelty_w = self._explorer_reward_weights(generation)
        max_worsening = 0.05 * novelty_w / 0.70
        return reward > 0.0 and nmk <= mk * (1.0 + max_worsening)

    def _get_local_search_operator(self, action):
        if action in [0, 1, 2]:
            return getattr(self.exploiter_ls, f"machine_local_search{action + 1}")
        if action in [3, 4, 5]:
            return getattr(self.exploiter_ls, f"agv_w_local_search{action - 2}")
        if action in [6, 7, 8]:
            return getattr(self.exploiter_ls, f"agv_f_local_search{action - 5}")
        return getattr(self.exploiter_ls, f"schedule_local_search{action - 8}")

    def _replace_population_individual(self, old_ind, new_ind):
        old_key = self._individual_key(old_ind)
        for pop in (self.explorer_population, self.exploiter_population):
            for i, ind in enumerate(pop):
                if ind is old_ind or (ind[1] == old_ind[1] and self._individual_key(ind) == old_key):
                    pop[i] = new_ind
                    return True
        return False

    def intensify_global_best(self, generation):
        if self.best_solution is None:
            return False

        old_best = self.best_solution
        idx, mk, MS, TW, TF, OS = old_best
        MS, TW, TF, OS = MS.copy(), TW.copy(), TF.copy(), OS.copy()
        phases = [
            ("machine", random.randint(1, 3)),
            ("agv_w", random.randint(1, 3)),
            ("agv_f", random.randint(1, 3)),
            ("schedule", random.randint(1, 3)),
        ]
        random.shuffle(phases)

        improved = False
        self.exploiter_ls._clear_ls_ctx()
        for phase, ls_id in phases:
            func = getattr(self.exploiter_ls, f"{phase}_local_search{ls_id}")
            nMS, nTW, nTF, nOS, nmk = func(MS, TW, TF, OS, mk)
            if nmk is not None and nmk < mk:
                MS, TW, TF, OS, mk = nMS, nTW, nTF, nOS, nmk
                improved = True
                self.exploiter_ls._clear_ls_ctx()

        if not improved:
            return False

        new_best = (idx, mk, MS, TW, TF, OS)
        self.best_solution = new_best
        self.best_generation = generation
        self._replace_population_individual(old_best, new_best)
        return True

    def exploiter_evolution(self, population):
        new_pop = population[:self.exploiter_elite_count]
        while len(new_pop) < self.exploiter_size:
            p1 = population[self.tournament_selection(population)]
            p2 = population[self.tournament_selection(population)]
            MS, TW, TF, OS = p1[2].copy(), p1[3].copy(), p1[4].copy(), p1[5].copy()
            if random.random() < self.exploiter_pc:
                MS = self.crossover_assignment_strings(p1[2], p2[2], is_machine=True)
                TW = self.crossover_assignment_strings(p1[3], p2[3])
                TF = self.crossover_assignment_strings(p1[4], p2[4])
                OS = self.crossover_OS(p1[5], p2[5])
            if random.random() < self.exploiter_pm:
                MS, TW, TF, OS = self.mutate_assignment_string(MS, self.num_machines, True), \
                                 self.mutate_assignment_string(TW, self.num_agvs_w), \
                                 self.mutate_assignment_string(TF, self.num_agvs_f), \
                                 self.mutate_OS(OS)
            mk = self.calculate.simulate(MS, TW, TF, OS)
            if mk is not None: new_pop.append((len(new_pop), mk, MS, TW, TF, OS))

        # --- 提前计算 State 划分边界 ---
        q1, q2 = np.percentile([ind[1] for ind in new_pop], [33.33, 66.67])
        
        self.exploiter_ls._clear_ls_ctx()
        
        # for i in range(len(new_pop)):
        for i in range(int(len(new_pop))):  
            idx, mk, MS, TW, TF, OS = new_pop[i]

            # 获取当前状态 (0, 1, 或 2)
            ctx = self.exploiter_ls._get_ls_ctx(MS, TW, TF, OS, mk)
            delay_stats = ctx.get("delay_stats", (0.0, 0.0, 0.0, 0.0))
            prev_state = self.exploiter_q.get_state(mk, q1, q2, delay_stats)
            action = self.exploiter_q.choose_action(prev_state)
            func = self._get_local_search_operator(action)
            
            nMS, nTW, nTF, nOS, nmk = func(MS, TW, TF, OS, mk)
            
            reward = 0.0
            improvement = (mk - nmk) / max(mk, 1e-9)
            if nmk is not None and nmk < mk:
                MS, TW, TF, OS, mk = nMS, nTW, nTF, nOS, nmk
                self.exploiter_ls._clear_ls_ctx()
                reward = float(np.clip(improvement / 0.01, 0.0, 1.0))
                # reward = 1.0 if improvement >= 0.01 else 0.5
            
            # Q-learning 
            ctx = self.exploiter_ls._get_ls_ctx(MS, TW, TF, OS, mk)
            delay_stats = ctx.get("delay_stats", (0.0, 0.0, 0.0, 0.0))
            next_state = self.exploiter_q.get_state(mk, q1, q2, delay_stats)
            self.exploiter_q.update(prev_state, action, reward, next_state)
                
            new_pop[i] = (idx, mk, MS, TW, TF, OS)
        return new_pop

    def explorer_evolution(self, population: list, gbest: tuple, generation: int = 0) -> list:
        
        new_pop = population[:self.explorer_size]
        
        # --- 计算 State 划分边界 ---
        mks = [ind[1] for ind in new_pop]
        q1, q2 = np.percentile(mks, [33.33, 66.67])
        
        # 获取当前 Explorer 种群的 Top-K 代表个体
        elites = self._build_explorer_elites(new_pop, gbest)
        
        # 获取详细的混合距离构成 (相对 gbest 与 elites 的差异)
        dist_details = [self.explorer_q.calculate_mixed_layer_distances(ind[2:], elites, self.num_operations) for ind in new_pop]
        
        # 计算综合总距离 (0.25权重的均值)
        dists = [0.25 * sum(d) for d in dist_details]
        d1, d2 = np.percentile(dists, [33.33, 66.67])

        # --- 对所有个体执行 Q-learning 扰动 ---
        for i in range(0, len(new_pop)):
            idx, mk, MS, TW, TF, OS = new_pop[i]
            d_MS, d_TW, d_TF, d_OS = dist_details[i]
            dist = dists[i]
            
            # 【一步获取当前状态：自动映射为 36 种状态之一】
            state = self.explorer_q.get_state(mk, q1, q2, dist, d1, d2, d_MS, d_TW, d_TF, d_OS)
            
            action = self.explorer_q.choose_action(state)
            if action in [2, 5, 8, 11]:
                cur_mk = self.calculate.simulate(MS, TW, TF, OS)
                cp = self.calculate.find_critical_path() if cur_mk is not None else None
            else:
                cp = None
            nMS, nTW, nTF, nOS = self.explorer_pert.perturb(action, MS, TW, TF, OS, cp)
            nmk = self.calculate.simulate(nMS, nTW, nTF, nOS)
            if nmk is None: continue
            
            # 重新计算新解的细节距离
            n_d_MS, n_d_TW, n_d_TF, n_d_OS = self.explorer_q.calculate_mixed_layer_distances((nMS, nTW, nTF, nOS), elites, self.num_operations)
            ndist = 0.25 * (n_d_MS + n_d_TW + n_d_TF + n_d_OS)
            
            Qgain = (mk - nmk) / max(mk, 1e-9)
            NovGain = ndist - dist
            reward = self._normalized_explorer_reward(Qgain, NovGain, generation)
            
            if self._accept_explorer_candidate(mk, nmk, reward, generation):
                new_pop[i] = (idx, nmk, nMS, nTW, nTF, nOS)
                
                # 若被接受，获取新状态
                next_state = self.explorer_q.get_state(nmk, q1, q2, ndist, d1, d2, n_d_MS, n_d_TW, n_d_TF, n_d_OS)
            else: 
                next_state = state
                
            self.explorer_q.update(state, action, reward, next_state)

        return new_pop
    
    def migrate_individuals(self):
        exp_mig_cnt = max(1, int(self.explorer_size * self.migration_rate))
        expl_mig_cnt = max(1, int(self.exploiter_size * self.migration_rate))
        
        cands_exp = self.explorer_population[:len(self.explorer_population)//2]
        # cands_exp.sort(key=lambda x: self.explorer_q.calculate_layer_distances(x[2:], self.best_solution, self.num_operations), reverse=True)
        cands_exp.sort(
            key=lambda x: sum(self.explorer_q.calculate_layer_distances(
                x[2:], self.best_solution, self.num_operations
            )),
            reverse=True,
        )
        exp_migrants = cands_exp[:exp_mig_cnt]
        
        top_expl = self.exploiter_population[:expl_mig_cnt]
        
        expl_migrants = []
        for ind in top_expl:
            idx, mk, MS, TW, TF, OS = ind
            nMS, nTW, nTF, nOS = self.explorer_pert.perturb(1, MS, TW, TF, OS)
            nMS, nTW, nTF, nOS = self.explorer_pert.perturb(4, nMS, nTW, nTF, nOS)
            nMS, nTW, nTF, nOS = self.explorer_pert.perturb(7, nMS, nTW, nTF, nOS)
            nMS, nTW, nTF, nOS = self.explorer_pert.perturb(10, nMS, nTW, nTF, nOS)

            nmk = self.calculate.simulate(nMS, nTW, nTF, nOS)
            expl_migrants.append((idx, nmk, nMS, nTW, nTF, nOS) if nmk else ind)
        
        self.exploiter_population[-exp_mig_cnt:] = exp_migrants
        self.explorer_population[-expl_mig_cnt:] = expl_migrants

    def evolve(self):
        self.initialize_populations()
        best_makespans = [self.best_solution[1]]
        logger.info(f"The initial best makespan is: {self.best_solution[1]:.2f}")
        early_stop = 0
        start_time = time.time()
        self._run_start_time = start_time
        
        for generation in range(self.max_iterations):
            if early_stop >= self.early_stop_patience or (time.time() - start_time) > self.max_run_time:
                break
            
            self.explorer_population = self.explorer_evolution(self.explorer_population, self.best_solution, generation)
            self.exploiter_population = self.exploiter_evolution(self.exploiter_population)
            self.explorer_population.sort(key=lambda x: x[1])
            self.exploiter_population.sort(key=lambda x: x[1])
            
            if (generation + 1) % self.migration_interval == 0:
                self.migrate_individuals()
            
            e_best = min(self.explorer_population, key=lambda t: t[1])
            x_best = min(self.exploiter_population, key=lambda t: t[1])
            cur_best0 = e_best if e_best[1] < x_best[1] else x_best

            cur_best = cur_best0
            if cur_best[1] < self.best_solution[1]:
                self.best_solution, self.best_generation, early_stop = cur_best, generation + 1, 0
                logger.info(f"Gen {generation+1}, New best: {self.best_solution[1]:.2f}")
            else: early_stop += 1

            if self.intensify_global_best(generation + 1):
                early_stop = 0
                self.explorer_population.sort(key=lambda x: x[1])
                self.exploiter_population.sort(key=lambda x: x[1])
                logger.info(f"Gen {generation+1}, LS-improved best: {self.best_solution[1]:.2f}")

            best_makespans.append(self.best_solution[1])
            
        return self.best_solution, best_makespans

if __name__ == "__main__":
    output_dir = os.path.join("output", "BLCMA_output")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'BLCMA_log_{int(time.time())}.log')
    logger = setup_file_logger('BLCMA', log_file_path=log_file)
    instance = load_instance("AFAISP-S15")
    
    BLCMA = BLCMA(instance=instance, population_size=150, max_iterations=1000)
    best_solution, best_makespans = BLCMA.evolve()
    
    calculate = Calculate(instance)
    makespan, schedule_results = calculate.simulate(*best_solution[2:], return_schedule=True)
    save_schedule_results(schedule_results, os.path.join(output_dir, "agv_w.csv"), os.path.join(output_dir, "agv_f.csv"), os.path.join(output_dir, "machine.csv"))
    
    # 获取工序处理顺序
    order = [BLCMA.op_index_to_id[idx] for idx in best_solution[5]]
    # 保存最优解
    save_best_solution(
        best_solution,
        order,
        os.path.join(output_dir, "best_solution.csv"),
        BLCMA.best_generation
    )
    
    # 绘制迭代曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
    plt.title('BLCMA Algorithm Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'BLCMA_convergence.png'))
    plt.close()

    # 绘制甘特图
    df = gantt_dataframe_from_schedule_results(schedule_results)
    export_gantt_csv(df, os.path.join(output_dir, "gantt_data.csv"))
    plot_gantt_three_swimlanes(
        df,
        title="Schedule Gantt (Machine / AGV_W / AGV_F)",
        figsize=(14, 7),
        save_path=os.path.join(output_dir, 'gantt.png'),
        dpi=300,
        show=False,
    )

    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'BLCMA_convergence.png')}")
    logger.info(f"Detailed log saved to: {log_file}")
    logger.info(f"Gantt chart saved to:  {os.path.join(output_dir, 'gantt.png')}")

