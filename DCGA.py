"""
双种群协同遗传算法模块[原论文复现]
[Han, X.; Cheng, W.; Meng, L.; Zhang, B.; Gao, K.; Zhang, C.; Duan, P.
A Dual Population Collaborative Genetic Algorithm for Solving Flexible Job Shop Scheduling Problem with AGV.
Swarm and Evolutionary Computation 2024, 86, 101538. https://doi.org/10.1016/j.swevo.2024.101538.]

该模块负责实现双种群协同遗传算法，包括：
- 使用两个种群: P1(agv解码方法1)和P2(agv解码方法2)
- 周期性种群间协作机制
- 基于Q学习的自适应选择和局部搜索
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 导入计算模块
from data.loader import Instance, load_instance
from simulate_dcga import Decoder
from utils import (save_schedule_results, save_best_solution, 
                gantt_dataframe_from_schedule_results, export_gantt_csv, 
                plot_gantt_three_swimlanes)

# 导入日志配置
from logger_config import setup_file_logger
logger = setup_file_logger('dcea')

class DCGA:
    """双种群遗传算法类"""
    def __init__(self, 
                instance: Instance,
                population_size: int = 100,
                max_iterations: int = 100,
                early_stop_patience: int = 500,
                run_time_ratio: float = 2.0,
                pc: float = 0.9,  # 交叉概率
                pm: float = 0.15): # 变异概率
        
        self.num_jobs = instance.num_jobs
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.num_operations = instance.num_operations
        self.num_machines = instance.num_machines
        self.job_operations = instance.job_operations
        self.processing_times = instance.processing_times
        self.priority_dict = instance.priority_dict
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.max_run_time = instance.num_operations * run_time_ratio # 超时阈值，单位为秒
        self.early_stop_patience = early_stop_patience
        self.pc = pc
        self.pm = pm

        # 初始化所有工序的顺序列表
        self.operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]

        # 初始化种群
        self.population = []
        self.best_solution = None
        self.best_generation = 0  # 记录找到最优解的代数
        
        # 初始化计算器
        self.decode = Decoder(instance)

    def initialize_populations(self):
        """初始化两个子种群"""
        self.population1 = self.initialize_population_random(self.population_size//2, strategy = 1)
        self.population2 = self.initialize_population_random(self.population_size//2, strategy = 2)
        
        # 更新整体最优解
        population = self.population1 + self.population2
        population.sort(key=lambda x: x[1])
        self.best_solution = population[0]

    def initialize_population_random(self, pop_size, strategy: int = 1) -> list:
        """初始化种群"""
        population_e = []
        
        for i in range(pop_size):
            schedule_string = self._generate_random_schedule()
            machine_string = self._generate_random_machine()
            
            # 评估解的质量
            # 确保不返回调度结果，只返回完工时间数值
            makespan = self.decode.simulate(machine_string, schedule_string, False, strategy)
            
            if makespan is not None:
                solution = (i + 1, makespan, machine_string, schedule_string)
                population_e.append(solution)
        return population_e

    def _generate_random_schedule(self) -> list:
        """随机生成符合工序优先级约束的工序调度编码"""
        
        # 创建工序ID到索引的映射（1到工序数量）
        op_id_to_index = {op_id: i+1 for i, op_id in enumerate(self.operations_list)}
        
        # 构建工序依赖图（前置工序 -> 后继工序）
        graph = {op_id: set() for op_id in self.operations_list}
        in_degree = {op_id: 0 for op_id in self.operations_list}
        
        for op_id, predecessors in self.priority_dict.items():
            for pred in predecessors:
                graph[pred].add(op_id)  # pred是op_id的前置工序，所以op_id是pred的后继工序
                in_degree[op_id] += 1  # op_id的入度加1
        
        # 使用拓扑排序生成有效的工序序列
        schedule_string = []
        
        # 初始化队列为所有入度为0的工序（没有前置工序的工序）
        queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            # 随机选择一个没有前置工序的工序
            np.random.shuffle(queue)
            op_id = queue.pop(0)
            
            # 将该工序的索引添加到调度序列中
            schedule_string.append(op_id_to_index[op_id])
            
            # 更新其后继工序的入度
            for successor in graph[op_id]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # 检查是否所有工序都被排序（检测循环依赖）
        if len(schedule_string) != self.num_operations:
            raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")
        
        return schedule_string
    
    def _generate_random_machine(self) -> list:
        """随机生成机器分配序列"""
        machine_string = []
        
        # 为每个工序随机选择可用机器
        for op_key in self.operations_list:
            if op_key in self.processing_times:
                available_machines = list(self.processing_times[op_key].keys())
                if available_machines:
                    machine = np.random.choice(available_machines)
                else:
                    logger.warning(f"工序 {op_key} 没有可用机器")
                    # 如果没有可用机器，设置为1号机器
                    machine = 1
            else:
                logger.warning(f"工序 {op_key} 不在处理时间字典中")
                # 如果工序不在处理时间字典中，设置为1号机器
                machine = 1
            
            machine_string.append(int(machine))
        
        return machine_string

    def tournament_selection(self, population) -> int:
        """锦标赛选择"""
        # 随机选择k个个体，但不超过种群大小
        k = min(2, len(population))  # 锦标赛大小，确保不超过种群大小
        
        # 如果种群为空，抛出更明确的错误
        if len(population) == 0:
            raise ValueError("种群为空，无法进行选择")
    
        candidates = random.sample(range(len(population)), k)
        
        # 选择最好的个体
        best_idx = candidates[0]
        best_makespan = population[best_idx][1]
        
        for idx in candidates[1:]:
            makespan = population[idx][1]
            if makespan < best_makespan:
                best_idx = idx
                best_makespan = makespan
    
        return best_idx

    def crossover_assignment_strings(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """交叉分配字符串(机器分配)"""
        mask = np.random.randint(0, 2, size=len(parent1))  # 1 -> child1 takes p1, child2 takes p2
        child1 = [p1 if m == 1 else p2 for p1, p2, m in zip(parent1, parent2, mask)]
        child2 = [p2 if m == 1 else p1 for p1, p2, m in zip(parent1, parent2, mask)]
        return child1, child2
    
    def crossover_schedule_string(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """使用POX交叉算子进行调度序列交叉"""
        # 预处理：建立工序索引到作业ID的映射
        if not hasattr(self, '_op_index_to_job_cache'):
            self._op_index_to_job_cache = {}
            for i, op_id in enumerate(self.operations_list):
                job_id = int(op_id.split('_')[0][1:])  # 从'o1_2'提取'1'
                self._op_index_to_job_cache[i+1] = job_id
        
        op_index_to_job = self._op_index_to_job_cache
        
        # 随机将作业分为两组
        all_jobs = list(range(1, self.num_jobs + 1))
        num_jobs_in_jset1 = random.randint(1, self.num_jobs - 1)
        jset1 = set(random.sample(all_jobs, num_jobs_in_jset1))

        # 初始化子代和空位置列表
        child1 = [-1] * len(parent1)
        empty_pos_1 = []
        
        # 从父代1复制JP1中的作业对应的工序
        for i, op_idx in enumerate(parent1):
            if op_index_to_job[op_idx] in jset1:
                child1[i] = op_idx
            else:
                empty_pos_1.append(i)
        
        # 从父代2中收集JP2作业对应的工序
        p2_jset2_ops = [op for op in parent2 if op_index_to_job[op] not in jset1]
        for k, pos in enumerate(empty_pos_1):
            child1[pos] = p2_jset2_ops[k]
            
        # Child2: symmetric
        child2 = [-1] * len(parent2)
        empty_pos_2 = []
        for i, op_idx in enumerate(parent2):
            if op_index_to_job[op_idx] not in jset1:  # jobs in Jset2
                child2[i] = op_idx
            else:
                empty_pos_2.append(i)

        p1_jset1_ops = [op for op in parent1 if op_index_to_job[op] in jset1]
        for k, pos in enumerate(empty_pos_2):
            child2[pos] = p1_jset1_ops[k]

        return child1, child2
    
    def mutate_assignment_string(self, assignment_string: List[int]) -> List[int]:
        """变异分配字符串(机器分配)"""
        result = assignment_string.copy()
        
        # 随机选择变异的位置
        mutation_pos = random.randint(0, len(result) - 1)
        # 机器分配变异
        op_id = self.operations_list[mutation_pos]
        available_machines = list(self.processing_times[op_id].keys())
        if available_machines:
            # 选择与当前不同的机器
            current_machine = result[mutation_pos]
            alternative_machines = [int(m) for m in available_machines if int(m) != current_machine]
            if alternative_machines:
                result[mutation_pos] = random.choice(alternative_machines)
        
        return result
    
    def mutate_schedule_string(self, schedule_string: List[int]) -> List[int]:
        """调度序列变异, 支持 Swap / Insert / Inversion, 变异后验证合法性"""
        result = schedule_string.copy()

        # 获取工序索引到工序ID的映射
        op_index_to_id = {i+1: op_id for i, op_id in enumerate(self.operations_list)}

        op_count = len(result)
        mutation_types = ['swap', 'insert', 'inversion']

        for _ in range(3):  # 最多尝试3次合法变异
            mutated = result.copy()
            mutation_type = random.choice(mutation_types)

            pos1 = random.randint(0, op_count - 1)
            pos2 = random.randint(0, op_count - 1)
            if pos1 == pos2:
                continue
            idx1, idx2 = sorted([pos1, pos2])

            if mutation_type == 'swap':
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

            elif mutation_type == 'insert':
                op = mutated.pop(idx2)
                mutated.insert(idx1, op)

            elif mutation_type == 'inversion' and idx2 - idx1 > 1:
                segment = mutated[idx1 + 1:idx2]
                segment.reverse()
                mutated[idx1 + 1:idx2] = segment

            # 合法性验证（不抛异常即合法）
            try:
                self.decode._validate_schedule_string(mutated, op_index_to_id, self.priority_dict)
                return mutated  # 验证通过，返回变异结果
            except:
                continue  # 验证失败，重试

        return result  # 若3次都失败，则返回原始序列
    
    def population_diversity_check(self):
        """对子种群 population1 和 population2 分别进行多样性检查"""
        for i in range(len(self.population1)):
            for j in range(i + 1, len(self.population1)):
                ind1, ind2 = self.population1[i], self.population1[j]
                if ind1[1] == ind2[1]:  # makespan 相等
                    sim = sum(a == b for a, b in zip(ind1[2], ind2[2])) / self.num_operations  # machine_string 相似度
                    if sim >= 0.8:
                        schedule_string = self._generate_random_schedule()
                        machine_string = self._generate_random_machine()
                        # 评估解的质量
                        makespan = self.decode.simulate(machine_string, schedule_string, False, strategy=1)
                        if makespan is not None:
                            solution = (j+1, makespan, machine_string, schedule_string)
                            self.population1[j] = solution
        for i in range(len(self.population2)):
                for j in range(i + 1, len(self.population2)):
                    ind1, ind2 = self.population2[i], self.population2[j]
                    if ind1[1] == ind2[1]:  # makespan 相等
                        sim = sum(a == b for a, b in zip(ind1[2], ind2[2])) / self.num_operations  # machine_string 相似度
                        if sim >= 0.8:
                            schedule_string = self._generate_random_schedule()
                            machine_string = self._generate_random_machine()
                            # 评估解的质量
                            makespan = self.decode.simulate(machine_string, schedule_string, False, strategy=2)
                            if makespan is not None:
                                solution = (j+1, makespan, machine_string, schedule_string)
                                self.population2[j] = solution

    def population_collaboration(self):
        """子种群间进行合作"""
        parent1_idx = self.tournament_selection(self.population1)
        parent1 = self.population1[parent1_idx]
        parent2_idx = self.tournament_selection(self.population2)
        parent2 = self.population2[parent2_idx]
        child_s1, child_s2 = self.crossover_schedule_string(parent1[3], parent2[3])
        child1_makespan = self.decode.simulate(parent1[2], child_s1, False, strategy=1)
        child2_makespan = self.decode.simulate(parent2[2], child_s2, False, strategy=2)
        
        if child1_makespan < parent1[1]:
            self.population1[parent1_idx] = (parent1[0], child1_makespan, parent1[2], child_s1)
            
        if child2_makespan < parent2[1]:
            self.population2[parent2_idx] = (parent2[0], child2_makespan, parent2[2], child_s2)

    def evolve(self):
        """实现双种群遗传进化"""
        # 初始化两个种群
        self.initialize_populations()
        
        # 记录每代的最优解完工时间
        best_makespans = []
        best_makespans.append(self.best_solution[1])
        logger.info(f"The initial best makespan is: {self.best_solution[1]:.2f}")
        
        # 初始化早停相关变量
        early_stop_counter = 0
        start_time = time.time()
        
        # 主进化循环
        for iteration in range(self.max_iterations):
            
            # 检查是否满足早停条件
            elapsed_time = time.time() - start_time
            if early_stop_counter >= self.early_stop_patience:
                logger.info(f"Early stopping triggered: No improvement for {self.early_stop_patience} consecutive generations.")
                break
            if elapsed_time > self.max_run_time:
                logger.info(f"Early stopping triggered: Runtime exceeded {self.max_run_time:.2f} seconds.")
                break

            # 并行进化
            self.evolve_population(self.population1, strategy = 1)
            self.evolve_population(self.population2, strategy = 2)
            # 多样性检查
            if iteration % 300 == 0:
                self.population_diversity_check()
            # 种群合作
            for _ in range(self.population_size // 2):
                self.population_collaboration()
                                    
            # 更新全局最优解
            solution_best1 = min(self.population1, key=lambda x: x[1])
            solution_best2 = min(self.population2, key=lambda x: x[1])
            
            current_best = solution_best1 if solution_best1[1] < solution_best2[1] else solution_best2
            
            if current_best[1] < self.best_solution[1]:
                self.best_solution = current_best
                self.best_generation = iteration + 1
                logger.info(f"In generation {iteration + 1}, New best makespan: {self.best_solution[1]:.2f}")
                # 找到更好的解，重置早停计数器
                early_stop_counter = 0
            else:
                # 没有找到更好的解，增加早停计数器
                early_stop_counter += 1
            # 记录当前代的最优解
            best_makespans.append(self.best_solution[1])
            
        return self.best_solution, best_makespans
    
    def evolve_population(self, population, strategy: int = 1):
        """单个种群的进化步骤"""
        pop_size = len(population)
        # 保存精英个体
        elites = sorted(population, key=lambda x: x[1])[:1]
        new_population = []
        new_population.extend(elites)
        
        # 生成下一代种群
        while len(new_population) < pop_size:
            # 父代选择
            p1 = population[self.tournament_selection(population)]
            p2 = population[self.tournament_selection(population)]
            
            # 交叉
            if random.random() < self.pc:
                c1_m, c2_m = self.crossover_assignment_strings(p1[2], p2[2])
                c1_s, c2_s = self.crossover_schedule_string(p1[3], p2[3])
            else:
                c1_m, c1_s = p1[2].copy(), p1[3].copy()
                c2_m, c2_s = p2[2].copy(), p2[3].copy()

            children = [(c1_m, c1_s), (c2_m, c2_s)]

            for child_machine, child_schedule in children:
                if len(new_population) >= pop_size:
                    break

                # Independent mutation probabilities
                if random.random() < self.pm:
                    child_machine = self.mutate_assignment_string(child_machine)
                    child_schedule = self.mutate_schedule_string(child_schedule)

                # Evaluate
                child_makespan = self.decode.simulate(child_machine, child_schedule, False, strategy)
                if child_makespan is None:
                    continue

                child_solution = (len(new_population), child_makespan, child_machine, child_schedule)
                new_population.append(child_solution)
        
        # 更新种群
        population.clear()
        population.extend(new_population)

# 主程序执行部分
if __name__ == "__main__":
    # 加载实例数据
    instance = load_instance('MK03')
    # 记录开始时间
    start_time = time.time()
    # 初始化算法参数
    dcga = DCGA(
            instance=instance,
            population_size=1000,
            max_iterations=1000,
            early_stop_patience=500,
            run_time_ratio=2.0,
            pc= 0.9,
            pm= 0.15)
    # 运行进化算法
    logger.info("Starting DCGA evolution...")
    best_solution, best_makespans = dcga.evolve()
    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f}s")
    
    # 获取最优解
    logger.info(f"Best solution found with makespan: {best_solution[1]:.2f}s at generation {dcga.best_generation}")
    
    # 使用最优解进行调度并保存结果
    decode = Decoder(instance)
    _, schedule_results = decode.simulate(*best_solution[2:], return_schedule=True, strategy = 1)
    
    # 定义输出文件路径
    output_dir = os.path.join("output", "dcga_output")
    log_file = os.path.join(output_dir, f'dcga_log_{int(time.time())}.log')
    
    # 保存调度结果
    save_schedule_results(
        schedule_results,
        os.path.join(output_dir, "agv_w_schedule.csv"),
        os.path.join(output_dir, "agv_f_schedule.csv"),
        os.path.join(output_dir, "machine_schedule.csv")
    )
    
    # 获取工序处理顺序
    op_index_to_id = {i+1: op_id for i, op_id in enumerate(dcga.operations_list)}
    order = [op_index_to_id[idx] for idx in best_solution[3]]
    # 保存最优解
    save_best_solution(
        best_solution,
        order,
        os.path.join(output_dir, "best_solution.csv"),
        dcga.best_generation,
        Is_dcga = True,
    )
    
    # 绘制迭代曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
    plt.title('DCGA Algorithm Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dcga_convergence.png'))
    plt.close()

    
    # 绘制甘特图
    df = gantt_dataframe_from_schedule_results(schedule_results)
    export_gantt_csv(df, os.path.join(output_dir, "gantt_data.csv"))
    plot_gantt_three_swimlanes(
        df,
        title="Schedule Gantt Chart for AFAISP",
        figsize=(14, 7),
        save_path=os.path.join(output_dir, 'gantt.png'),
        dpi=300,
        show=False,
    )

    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'dcga_convergence.png')}")
    logger.info(f"Detailed log saved to: {log_file}")
    logger.info(f"Gantt chart saved to:  {os.path.join(output_dir, 'gantt.png')}")