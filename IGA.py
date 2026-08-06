"""
改进遗传算法模块[原论文复现]
[Yan, J.; Liu, Z.F.; Zhang, C.X.; Zhang, T.;  Zhang, Y.Z.; Yang, C.B.
Research on flexible job shop scheduling under finite transportation conditions for digital twin workshop.
Robot. Comput. Int. Manuf., vol. 72, pp. 102198, 2021. https://doi.org/10.1109/TEVC.2022.3219238.]

该模块负责实现遗传算法，包括：
- 随机+启发式生成初始种群
- 解码染色体为具体的调度方案
- 选择交叉变异+前向插入
- 迭代优化
"""

import os
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any

# 导入计算模块
from data.loader import Instance, load_instance
from decode_iga import decoder
from utils import (
    save_schedule_results, save_best_solution,
    gantt_dataframe_from_schedule_results, export_gantt_csv,
    plot_gantt_three_swimlanes
)

# 导入日志配置
from logger_config import setup_file_logger

logger = setup_file_logger('iga')

# (id, makespan, MS, TS_W, TS_F, OS)
Individual = Tuple[int, float, List[int], List[int], List[int], List[int]]


class IGA:
    """改进遗传算法类（迁移到 AFAISP，OS 为 precedence-feasible 的工序级序列）"""

    def __init__(
        self,
        instance: Instance,
        population_size: int = 80,
        max_iterations: int = 1000,
        early_stop_patience: int = 500,
        run_time_ratio: float = 2.0,
        pc: float = 0.70,
        pm: float = 0.30,
    ) -> None:
        self.num_jobs = instance.num_jobs
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.num_machines = instance.num_machines
        self.num_operations = instance.num_operations
        self.job_operations = instance.job_operations
        self.processing_times = instance.processing_times
        self.priority_dict = instance.priority_dict

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.pc = float(pc)
        self.pm = float(pm)

        self.elite_ratio = 0.10
        self.init_random_ratio = 0.20

        self.early_stop_patience = int(early_stop_patience)
        self.max_run_time = float(instance.num_operations) * float(run_time_ratio)

        self._next_id = 0
        self.population: List[Individual] = []
        self.best_solution: Optional[Individual] = None
        self.best_generation = 0

        # 解码器
        self.decoder = decoder(instance)

        # 所有工序列表（按 job / op_seq 构造，索引从 1..No）
        self.operations_list: List[str] = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]

        # 缓存映射：避免频繁构造
        self._op_id_to_index: Dict[str, int] = {op_id: i + 1 for i, op_id in enumerate(self.operations_list)}
        self._op_index_to_id: Dict[int, str] = {i + 1: op_id for i, op_id in enumerate(self.operations_list)}

    # =========================
    # Initialization
    # =========================
    def initialize_population(self) -> None:
        """初始化种群（随机 + IDPSM 解码补全）"""
        target_random = int(self.population_size * self.init_random_ratio)
        random_count = 0

        self.population.clear()
        self.best_solution = None
        self.best_generation = 0

        while len(self.population) < self.population_size:
            if random_count < target_random:
                makespan, MS, TS_W, TS_F, OS = self.generate_random_strings()
                random_count += 1
            else:
                makespan, MS, TS_W, TS_F, OS = self.generate_idpsm_strings()

            ind: Individual = (self._next_id, float(makespan), MS, TS_W, TS_F, OS)
            self._next_id += 1
            self.population.append(ind)

            if self.best_solution is None or ind[1] < self.best_solution[1]:
                self.best_solution = ind

        if not self.population:
            raise RuntimeError("初始化失败：未生成任何可行个体。")

    def _random_topological_os(self) -> List[int]:
        """生成 precedence-feasible OS（返回 op_index 序列，长度=No）"""
        graph: Dict[str, set] = {op_id: set() for op_id in self.operations_list}
        in_degree: Dict[str, int] = {op_id: 0 for op_id in self.operations_list}

        # priority_dict: op_id -> predecessors
        for op_id, predecessors in self.priority_dict.items():
            for pred in predecessors:
                graph[pred].add(op_id)
                in_degree[op_id] += 1

        OS: List[int] = []
        queue = [op_id for op_id, deg in in_degree.items() if deg == 0]

        while queue:
            np.random.shuffle(queue)
            op_id = queue.pop(0)
            OS.append(self._op_id_to_index[op_id])

            for succ in graph[op_id]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(OS) != self.num_operations:
            raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")
        if len(set(OS)) != self.num_operations:
            raise RuntimeError("生成的调度字符串中存在重复工序")
        return OS

    def generate_random_strings(self) -> Tuple[float, List[int], List[int], List[int], List[int]]:
        """随机生成编码字符串（OS precedence-feasible + MS/TS 随机）"""
        OS = self._random_topological_os()

        MS: List[int] = []
        TS_W: List[int] = []
        TS_F: List[int] = []

        for op_idx in OS:
            op_id = self._op_index_to_id[op_idx]

            avail = list(self.processing_times.get(op_id, {}).keys())
            if not avail:
                raise ValueError(f"没有为工序 {op_id} 指定可行机器")
            choice_m = random.choice(avail)
            try:
                MS.append(int(choice_m))
            except Exception:
                MS.append(int(str(choice_m)))

            TS_W.append(random.randint(1, self.num_agvs_w))
            TS_F.append(random.randint(1, self.num_agvs_f))

        dec = self.decoder.decode(MS, TS_W, TS_F, OS, return_schedule=False)
        makespan = float(dec["makespan"])

        # 基本健壮性检查（不额外消耗解码次数）
        if len(MS) != self.num_operations or len(TS_W) != self.num_operations or len(TS_F) != self.num_operations:
            raise RuntimeError("生成的编码长度与工序数量不匹配")
        return makespan, MS, TS_W, TS_F, OS

    def generate_idpsm_strings(self) -> Tuple[float, List[int], List[int], List[int], List[int]]:
        """
        IDPSM 初始化：OS precedence-feasible；MS/TS 置 None 让 decoder 用 IDPSM 补全。
        """
        OS = self._random_topological_os()

        MS: List[Optional[int]] = [None] * self.num_operations
        TS_W: List[Optional[int]] = [None] * self.num_operations
        TS_F: List[Optional[int]] = [None] * self.num_operations

        dec = self.decoder.decode(MS, TS_W, TS_F, OS, return_schedule=False)

        filled_MS = dec["filled_MS"]
        filled_TS_F = dec["filled_TS_F"]
        TS_W_raw = dec.get("filled_TS_W", [])

        filled_TS_W: List[int] = []
        for tw in TS_W_raw:
            try:
                v = int(tw)
            except Exception:
                v = 1
            if v == 0:  # same-machine 情形 decoder 可能置 0，这里按你原逻辑修正到 1（保持 GA 编码合法域）
                v = 1
            filled_TS_W.append(v)

        makespan = float(dec["makespan"])

        if len(filled_MS) != self.num_operations or len(filled_TS_W) != self.num_operations or len(filled_TS_F) != self.num_operations:
            raise RuntimeError("生成的编码长度与工序数量不匹配")
        return makespan, filled_MS, filled_TS_W, filled_TS_F, OS

    # =========================
    # Selection
    # =========================
    @staticmethod
    def tournament_selection(pop: List[Individual], k: int = 2) -> int:
        """锦标赛选择：返回 pop 中被选中个体的索引"""
        if not pop:
            raise ValueError("种群为空，无法进行选择")
        k = min(int(k), len(pop))
        cand_idx = random.sample(range(len(pop)), k)
        best = cand_idx[0]
        best_fit = pop[best][1]
        for idx in cand_idx[1:]:
            fit = pop[idx][1]
            if fit < best_fit:
                best, best_fit = idx, fit
        return best

    # =========================
    # Helpers: op mapping
    # =========================
    def _op_index_to_id_local(self, idx: int) -> str:
        return self.operations_list[idx - 1]

    def _op_id_to_job(self, op_id: str) -> int:
        return int(op_id[1:].split("_")[0])

    # =========================
    # Genetic operators
    # =========================
    def crossover(
        self,
        p1: Tuple[List[int], List[int], List[int], List[int]],  # (MS, TS_W, TS_F, OS)
        p2: Tuple[List[int], List[int], List[int], List[int]],
    ) -> Tuple[
        Tuple[List[int], List[int], List[int], List[int]],
        Tuple[List[int], List[int], List[int], List[int]]
    ]:
        """
        交叉（精简版）：
        1) OS：作业分组交叉（JOX）
        2) MS/TS：按工序 ID 映射重建；再用少量掩码位从另一父代拷贝对应工序的 (MS,W,F)
        """
        MS1, TS_W1, TS_F1, OS1 = p1
        MS2, TS_W2, TS_F2, OS2 = p2
        n = len(OS1)
        assert n and n == len(OS2) == len(MS1) == len(TS_W1) == len(TS_F1) == len(MS2) == len(TS_W2) == len(TS_F2)

        jobs = list(range(1, self.num_jobs + 1))
        random.shuffle(jobs)
        cut = random.randint(1, max(1, self.num_jobs - 1))
        JP1 = set(jobs[:cut])

        def build_child_os(keep_OS: List[int], fill_OS: List[int], keep_jobs: set) -> List[int]:
            child: List[Optional[int]] = [None] * n
            for i, op_idx in enumerate(keep_OS):
                if self._op_id_to_job(self._op_index_to_id_local(op_idx)) in keep_jobs:
                    child[i] = op_idx

            fill_iter = (
                op for op in fill_OS
                if self._op_id_to_job(self._op_index_to_id_local(op)) not in keep_jobs
            )
            for i in range(n):
                if child[i] is None:
                    child[i] = next(fill_iter)
            return [int(x) for x in child]  # type: ignore

        child1_OS = build_child_os(OS1, OS2, JP1)
        child2_OS = build_child_os(OS2, OS1, JP1)

        def op_maps(MS: List[int], W: List[int], F: List[int], OS: List[int]) -> Dict[str, Tuple[int, int, int]]:
            return {
                self._op_index_to_id_local(op_idx): (int(ms), int(w), int(f))
                for ms, w, f, op_idx in zip(MS, W, F, OS)
            }

        p1_map = op_maps(MS1, TS_W1, TS_F1, OS1)
        p2_map = op_maps(MS2, TS_W2, TS_F2, OS2)

        def materialize(parent_map: Dict[str, Tuple[int, int, int]], child_OS: List[int]) -> Tuple[List[int], List[int], List[int]]:
            ms_list: List[int] = []
            w_list: List[int] = []
            f_list: List[int] = []
            for op_idx in child_OS:
                op_id = self._op_index_to_id_local(op_idx)
                ms, w, f = parent_map.get(op_id, (1, 1, 1))
                ms_list.append(int(ms))
                w_list.append(int(w))
                f_list.append(int(f))
            return ms_list, w_list, f_list

        c1_MS, c1_W, c1_F = materialize(p1_map, child1_OS)
        c2_MS, c2_W, c2_F = materialize(p2_map, child2_OS)

        k = max(1, min(n, 3))
        for i in random.sample(range(n), k):
            op1_id = self._op_index_to_id_local(child1_OS[i])
            op2_id = self._op_index_to_id_local(child2_OS[i])
            if op1_id in p2_map:
                c1_MS[i], c1_W[i], c1_F[i] = p2_map[op1_id]
            if op2_id in p1_map:
                c2_MS[i], c2_W[i], c2_F[i] = p1_map[op2_id]

        # 轻度合法性修正：机器不可行则回退到“主父代”映射值
        for i, op_idx in enumerate(child1_OS):
            op_id = self._op_index_to_id_local(op_idx)
            if c1_MS[i] not in self.processing_times.get(op_id, {}):
                c1_MS[i] = p1_map.get(op_id, (c1_MS[i], 1, 1))[0]
        for i, op_idx in enumerate(child2_OS):
            op_id = self._op_index_to_id_local(op_idx)
            if c2_MS[i] not in self.processing_times.get(op_id, {}):
                c2_MS[i] = p2_map.get(op_id, (c2_MS[i], 1, 1))[0]

        return (c1_MS, c1_W, c1_F, child1_OS), (c2_MS, c2_W, c2_F, child2_OS)

    def mutate(
        self,
        individual: Tuple[List[int], List[int], List[int], List[int]],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        变异：仅对 MS/TS_W/TS_F 做点变异（OS 不变），并确保不会原地污染父代。
        """
        MS, TS_W, TS_F, OS = individual
        n = len(OS)
        if n == 0:
            return (MS, TS_W, TS_F, OS)

        # 防止就地修改导致父代被污染
        MS2 = MS.copy()
        W2 = TS_W.copy()
        F2 = TS_F.copy()
        OS2 = OS  # OS 在本算子中不改；若你后续扩展 OS 变异，再 copy

        i = random.randrange(n)
        op_id = self._op_index_to_id_local(OS2[i])

        avail_machs = list(self.processing_times.get(op_id, {}).keys())
        if not avail_machs:
            return (MS2, W2, F2, OS2)

        try:
            avail_machs_i = [int(m) for m in avail_machs]
        except Exception:
            avail_machs_i = [int(str(m)) for m in avail_machs]

        if len(avail_machs_i) > 1:
            cur = int(MS2[i])
            cand = [m for m in avail_machs_i if m != cur]
            if cand:
                MS2[i] = random.choice(cand)

        # W-AGV
        if self.num_agvs_w >= 1:
            cur_w = int(W2[i]) if W2[i] is not None else 0
            if self.num_agvs_w > 1:
                choices = [x for x in range(1, self.num_agvs_w + 1) if x != cur_w]
                W2[i] = random.choice(choices) if choices else max(1, cur_w)
            else:
                W2[i] = 1

        # F-AGV
        if self.num_agvs_f >= 1:
            cur_f = int(F2[i]) if F2[i] is not None else 0
            if self.num_agvs_f > 1:
                choices = [x for x in range(1, self.num_agvs_f + 1) if x != cur_f]
                F2[i] = random.choice(choices) if choices else max(1, cur_f)
            else:
                F2[i] = 1

        return (MS2, W2, F2, OS2)

    @staticmethod
    def _clone_genes(genes: Tuple[List[int], List[int], List[int], List[int]]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """用于在不交叉时复制父代基因，避免后续变异污染父代"""
        MS, W, F, OS = genes
        return (MS.copy(), W.copy(), F.copy(), OS.copy())

    # =========================
    # Evolution
    # =========================
    def evolve(self) -> Tuple[Individual, List[float]]:
        """进化过程：返回（最优个体，历史最优 makespan 列表）"""
        self.initialize_population()
        assert self.best_solution is not None

        best_makespans: List[float] = []
        elite_count = max(1, int(self.elite_ratio * self.population_size))

        early_stop_counter = 0
        start_time = time.time()
        prev_best = float(self.best_solution[1])

        for generation in range(self.max_iterations):
            elapsed = time.time() - start_time
            if early_stop_counter >= self.early_stop_patience:
                logger.info(f"Early stopping: no improvement for {self.early_stop_patience} generations.")
                break
            if elapsed > self.max_run_time:
                logger.info(f"Early stopping: runtime exceeded {self.max_run_time:.2f} seconds.")
                break

            # 精英保留
            elites = sorted(self.population, key=lambda x: x[1])[:elite_count]
            new_population: List[Individual] = list(elites)

            # 生成剩余个体（每次产生 1 或 2 个子代，避免越界）
            while len(new_population) < self.population_size:
                # 父代选择（直接从原种群锦标赛选，不构造额外池，避免额外开销）
                p1 = self.population[self.tournament_selection(self.population, k=2)]
                p2 = self.population[self.tournament_selection(self.population, k=2)]
                if p2[0] == p1[0] and len(self.population) > 1:
                    # 再抽一次尽量避免同一个体
                    p2 = self.population[self.tournament_selection(self.population, k=2)]

                # 交叉
                if random.random() < self.pc:
                    c1_genes, c2_genes = self.crossover(p1[2:], p2[2:])
                else:
                    c1_genes = self._clone_genes(p1[2:])
                    c2_genes = self._clone_genes(p2[2:])

                # 变异（只对 MS/TS）
                if random.random() < self.pm:
                    c1_genes = self.mutate(c1_genes)
                if random.random() < self.pm and len(new_population) + 1 < self.population_size:
                    c2_genes = self.mutate(c2_genes)

                # 解码评估（不增加额外次数）
                dec1 = self.decoder.decode(c1_genes[0], c1_genes[1], c1_genes[2], c1_genes[3], return_schedule=False)
                m1 = dec1.get("makespan", None)

                if m1 is not None:
                    child1: Individual = (self._next_id, float(m1), c1_genes[0], c1_genes[1], c1_genes[2], c1_genes[3])
                    self._next_id += 1
                    new_population.append(child1)

                    if float(m1) < self.best_solution[1]:
                        self.best_solution = child1
                        self.best_generation = generation + 1

                if len(new_population) >= self.population_size:
                    break

                # 第二个子代（若还有空间）
                if len(new_population) < self.population_size:
                    dec2 = self.decoder.decode(c2_genes[0], c2_genes[1], c2_genes[2], c2_genes[3], return_schedule=False)
                    m2 = dec2.get("makespan", None)
                    if m2 is not None:
                        child2: Individual = (self._next_id, float(m2), c2_genes[0], c2_genes[1], c2_genes[2], c2_genes[3])
                        self._next_id += 1
                        new_population.append(child2)

                        if float(m2) < self.best_solution[1]:
                            self.best_solution = child2
                            self.best_generation = generation + 1

            self.population = new_population

            cur_best = float(self.best_solution[1])
            best_makespans.append(cur_best)

            if cur_best + 1e-12 < prev_best:
                logger.info(f"In generation {generation + 1}, New best makespan: {cur_best:.2f}")
                prev_best = cur_best
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        assert self.best_solution is not None
        return self.best_solution, best_makespans


if __name__ == "__main__":
    debug = True

    start_time = time.time()
    instance = load_instance('MK03')

    iga = IGA(
        instance=instance,
        population_size=80,
        max_iterations=1000,
        early_stop_patience=500,
        run_time_ratio=2.0,
        pc=0.7,
        pm=0.3,
    )

    logger.info("Starting evolution...")
    best_solution, best_makespans = iga.evolve()

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Best solution found with makespan: {best_solution[1]:.2f}s at generation {iga.best_generation}")

    # 用最优解解码并保存
    dec = decoder(instance)
    schedule_results = dec.decode(*best_solution[2:], return_schedule=True, warehouse_node="Warehouse")

    output_dir = os.path.join("output", "iga_output")
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f'iga_log_{int(time.time())}.log')

    save_schedule_results(
        schedule_results["schedule_results"],
        os.path.join(output_dir, "agv_w_schedule.csv"),
        os.path.join(output_dir, "agv_f_schedule.csv"),
        os.path.join(output_dir, "machine_schedule.csv"),
    )

    op_index_to_id = {i + 1: op_id for i, op_id in enumerate(iga.operations_list)}
    order = [op_index_to_id[idx] for idx in best_solution[5]]  # OS 在 best_solution[5]

    save_best_solution(
        best_solution,
        order,
        os.path.join(output_dir, "best_solution.csv"),
        iga.best_generation,
    )

    # 收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
    plt.title('IGA Algorithm Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'iga_convergence.png'))
    plt.close()

    # 甘特图
    df = gantt_dataframe_from_schedule_results(schedule_results["schedule_results"])
    export_gantt_csv(df, os.path.join(output_dir, "gantt_data.csv"))
    plot_gantt_three_swimlanes(
        df,
        title=f"Schedule Gantt (Machine / AGV_W / AGV_F)  —  Cmax={schedule_results['makespan']:.1f}",
        figsize=(14, 7),
        save_path=os.path.join(output_dir, 'gantt.png'),
        dpi=300,
        show=False,
    )

    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'iga_convergence.png')}")
    logger.info(f"Detailed log saved to: {log_file}")
    logger.info(f"Gantt chart saved to:  {os.path.join(output_dir, 'gantt.png')}")
