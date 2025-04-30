"""
进化学习模块

该模块负责实现基于离散编码的进化学习优化算法，包括：
- 生成随机的机器分配、AGV分配和调度序列初始种群
- 基于Q学习的选择算子改进
- 基于问题性质局部搜索算法
"""

import os
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from collections import OrderedDict

# 导入计算模块
from calculate import Calculate

# 导入配置
from MK10 import (
    NUM_JOBS as num_jobs,
    NUM_MACHINES as num_machines,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    JOB_OPERATIONS as job_operations,
    PROCESSING_TIMES as processing_times,
    PRIORITY_DICT as priority_dict
)

# 导入日志配置
from logger_config import setup_file_logger

# 获取日志记录器
logger = setup_file_logger('el')

# ===== 类型别名定义 =====
SolutionType = Tuple[int, float, List[int], List[int], List[int], List[int]]  # (solution_id, makespan, machine_string, agv_w_string, agv_f_string, schedule_string)
ResultType = List[Dict[str, object]]  # 调度结果类型

def save_schedule_results(calculate: Calculate, agv_w_file: str, agv_f_file: str, machine_file: str) -> None:
    """将调度结果保存到文件
    
    Args:
        calculate: 工厂对象
        agv_w_file: AGV_W调度结果文件路径
        agv_f_file: AGV_F调度结果文件路径
        machine_file: 机器调度结果文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(agv_w_file), exist_ok=True)
    
    # 获取调度数据
    agv_w_headers, agv_w_data, agv_f_headers, agv_f_data, machine_headers, machine_data = calculate._collect_schedule_data()
    
    # 保存AGV_W数据，按AGV编号和运输开始时间排序
    with open(agv_w_file, 'w', encoding='utf-8') as f:
        f.write(",".join(agv_w_headers) + "\n")
        # 按AGV编号和开始运输时间排序
        sorted_agv_w_data = sorted(agv_w_data, key=lambda x: (x[0][0], x[0][1]))  # x[0][0]是AGV编号，x[0][1]是开始运输时间
        for row_sort, row_display in sorted_agv_w_data:
            f.write(",".join(str(item) for item in row_display) + "\n")
    
    # 保存AGV_F数据，按AGV编号和运输开始时间排序
    with open(agv_f_file, 'w', encoding='utf-8') as f:
        f.write(",".join(agv_f_headers) + "\n")
        # 按AGV编号和开始运输时间排序
        sorted_agv_f_data = sorted(agv_f_data, key=lambda x: (x[0][0], x[0][1]))  # x[0][0]是AGV编号，x[0][1]是开始运输时间
        for row_sort, row_display in sorted_agv_f_data:
            f.write(",".join(str(item) for item in row_display) + "\n")
    
    # 保存机器数据，按机器编号和加工开始时间排序
    with open(machine_file, 'w', encoding='utf-8') as f:
        f.write(",".join(machine_headers) + "\n")
        # 按机器编号和开始时间排序
        sorted_machine_data = sorted(machine_data, key=lambda x: (x[0][0], x[0][1]))  # x[0][0]是机器编号，x[0][1]是开始时间
        for row_sort, row_display in sorted_machine_data:
            f.write(",".join(str(item) for item in row_display) + "\n")


def save_best_solution(best_solution: SolutionType, order: List[str], output_file: str, best_generation: int) -> None:
    """保存最优解到CSV文件
    
    Args:
        best_solution: 最优解
        order: 工序处理顺序
        output_file: 输出文件路径
        best_generation: 找到最优解的代数
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 解包最优解
    solution_id, makespan, machine_string, agv_w_string, agv_f_string, schedule_string = best_solution
    
    # 创建有序结果字典
    result = OrderedDict([
        ("迭代次数", best_generation),
        ("最优解编号", solution_id),
        ("完工时间", makespan),
        ("机器分配", machine_string),
        ("AGV_W分配", agv_w_string),
        ("AGV_F分配", agv_f_string),
        ("调度顺序", schedule_string),
        ("工序处理顺序", order)
    ])
    
    # 保存到CSV文件
    df = pd.DataFrame([result])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')


class EvolutionaryLearning:
    """进化学习算法类"""
    def __init__(self, 
                num_jobs: int,
                num_agvs_w: int,
                num_agvs_f: int,
                job_operations: Dict[int, int],
                processing_times: Dict[str, Dict[int, float]],
                population_size: int = 100,
                max_iterations: int = 100,
                pc: float = 0.8,  # 交叉概率
                pm: float = 0.15): # 变异概率

        self.num_jobs = num_jobs
        self.num_agvs_w = num_agvs_w
        self.num_agvs_f = num_agvs_f
        self.job_operations = job_operations
        self.processing_times = processing_times
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.pc = pc
        self.pm = pm
        
        # 初始化种群
        self.population = []
        self.best_solution = None
        self.best_generation = 0  # 记录找到最优解的代数
        
        # 初始化Calculate类
        self.calculate = Calculate()
    
    def initialize_population(self) -> None:
        """初始化种群"""
        attempts = 0
        max_attempts = self.population_size * 10  # 最大尝试次数
        
        while len(self.population) < self.population_size and attempts < max_attempts:
            solution = self.generate_random_strings(
                self.num_jobs,
                self.num_agvs_w,
                self.num_agvs_f,
                self.job_operations,
                self.processing_times
            )
            calculate = Calculate()
            makespan = calculate.simulate(*solution)
            
            # 确保makespan不为None
            if makespan is not None:
                solution_tuple = (len(self.population) + 1, makespan, *solution)
                self.population.append(solution_tuple)
                
                if self.best_solution is None or makespan < self.best_solution[1]:
                    self.best_solution = solution_tuple
                
            attempts += 1
            
        if len(self.population) < self.population_size:
            raise ValueError(f"无法生成足够的有效解。当前仅生成了 {len(self.population)} 个解，需要 {self.population_size} 个。")
    
    def generate_random_strings(
        self,
        num_jobs: int,
        num_agvs_w: int,
        num_agvs_f: int,
        job_operations: Dict[int, int],
        processing_times: Dict[str, Dict[int, float]]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """随机生成编码字符串"""
        # 获取所有工序的顺序列表
        operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, num_jobs + 1)
            for op_seq in range(1, job_operations[job_id] + 1)
        ]
        
        # 生成机器分配字符串
        machine_string = []
        for op_id in operations_list:
            # 从可用机器中随机选择一个
            available_machines = list(processing_times[op_id].keys())
            machine_string.append(int(random.choice(available_machines)))
        
        # 生成AGV分配字符串
        agv_w_string = [random.randint(1, num_agvs_w) for _ in range(len(operations_list))]
        agv_f_string = [random.randint(1, num_agvs_f) for _ in range(len(operations_list))]

        # 生成调度字符串
        schedule_string = []
        remaining_ops = []  # 剩余未分配的工序
        for job_id in range(1, num_jobs + 1):
            for _ in range(job_operations[job_id]):
                remaining_ops.append(job_id)
        random.shuffle(remaining_ops)
        schedule_string = remaining_ops

        return machine_string, agv_w_string, agv_f_string, schedule_string

    def tournament_selection(self) -> int:
        """锦标赛选择
        
        Returns:
            int: 选中个体的索引
        """
        # 随机选择k个个体，但不超过种群大小
        k = min(2, len(self.population))  # 锦标赛大小，确保不超过种群大小
        
        # 如果种群为空
        if len(self.population) == 0:
            raise ValueError("种群为空，无法进行选择")
    
        candidates = random.sample(range(len(self.population)), k)
        
        # 选择最好的个体
        best_idx = candidates[0]
        best_makespan = self.population[best_idx][1]
        
        for idx in candidates[1:]:
            makespan = self.population[idx][1]
            if makespan < best_makespan:
                best_idx = idx
                best_makespan = makespan
    
        return best_idx
    
    def crossover_assignment_strings(self, parent1: List[int], parent2: List[int], is_machine: bool = False) -> List[int]:
        """交叉分配字符串（机器分配和AGV分配）"""
        if random.random() > self.pc:
            return parent1.copy()
            
        child = []
        operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]
        
        for i, op_id in enumerate(operations_list):
            if is_machine:
                # 对于机器分配，确保选择的机器可以加工该工序
                available_machines = list(self.processing_times[op_id].keys())
                if parent1[i] in available_machines and parent2[i] in available_machines:
                    # 如果两个父代的机器都是可用的，随机选择一个
                    child.append(parent1[i] if random.random() <= 0.5 else parent2[i])
                elif parent1[i] in available_machines:
                    child.append(parent1[i])
                elif parent2[i] in available_machines:
                    child.append(parent2[i])
                else:
                    # 如果两个父代的机器都不可用，随机选择一个可用机器
                    child.append(int(random.choice(available_machines)))
            else:
                # 对于AGV分配，直接随机选择父代的值
                child.append(parent1[i] if random.random() <= 0.5 else parent2[i])
        return child
    
    def crossover_schedule_string(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """POX交叉操作
        
        Args:
            parent1: 父代1的调度字符串
            parent2: 父代2的调度字符串
            
        Returns:
            List[int]: 子代的调度字符串
        """
        if random.random() > self.pc:
            return parent1.copy()
            
        # 初始化子代
        child = [-1] * len(parent1)
        
        # 随机将作业分为两组
        all_jobs = list(range(1, self.num_jobs + 1))
        num_jobs_in_jp1 = random.randint(1, self.num_jobs - 1)
        jp1 = set(random.sample(all_jobs, num_jobs_in_jp1))
        jp2 = set(all_jobs) - jp1
        
        # 从parent1复制JP1中的作业
        p1_positions = []
        for i, job in enumerate(parent1):
            if job in jp1:  # 直接检查作业号是否在jp1中
                child[i] = job
                p1_positions.append(i)
                
        # 从parent2获取JP2中的作业
        p2_elements = [job for job in parent2 if job in jp2]  # 获取parent2中属于jp2的作业
        empty_positions = [i for i in range(len(child)) if child[i] == -1]
        
        for pos, elem in zip(empty_positions, p2_elements):
            child[pos] = elem
            
        return child
    
    def mutate_assignment_string(self, string: List[int], max_value: int, is_machine: bool = False) -> List[int]:
        """位变异算子"""
        mutated = string.copy()
        operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]
        
        for i, op_id in enumerate(mutated):
            if random.random() < self.pm:
                if is_machine:
                    # 对于机器分配，随机选择一个可用机器
                    available_machines = list(self.processing_times[operations_list[i]].keys())
                    mutated[i] = int(random.choice(available_machines))
                else:
                    # 对于AGV分配，随机选择一个AGV
                    mutated[i] = random.randint(1, max_value)
        return mutated
    
    def mutate_schedule_string(self, schedule: List[int]) -> List[int]:
        """交换变异算子"""
        mutated = schedule.copy()
        if random.random() < self.pm:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def agv_local_search(self, child_machine: List[int], child_agv_w: List[int],
                        child_agv_f: List[int], child_schedule: List[int], current_makespan: float) -> Tuple[List[int], float]:
        """AGV分配局部搜索
        
        Args:
            child_machine: 机器分配序列
            child_agv_w: 工件AGV分配序列
            child_agv_f: 物料AGV分配序列
            child_schedule: 调度序列
            current_makespan: 当前完工时间
            
        Returns:
            Tuple[List[int], float]: 改进后的AGV分配序列和完工时间
        """
        # 随机选择两个不同的AGV_W
        agv1 = random.randint(1, self.num_agvs_w)
        agv2 = random.randint(1, self.num_agvs_w)
        while agv2 == agv1:
            agv2 = random.randint(1, self.num_agvs_w)
        
        # 获取这两个AGV在child_agv_w中出现的索引位置
        agv1_indices = [i for i, agv in enumerate(child_agv_w) if agv == agv1]
        agv2_indices = [i for i, agv in enumerate(child_agv_w) if agv == agv2]
        
        # 如果任一AGV没有任务，则无法交换
        if not agv1_indices or not agv2_indices:
            return child_agv_w, current_makespan
        
        # 从每个AGV的索引列表中随机选择一个起始索引
        start_idx1 = random.choice(agv1_indices)
        start_idx2 = random.choice(agv2_indices)
        
        # 创建新的AGV_W分配序列
        new_agv_w = child_agv_w.copy()
        
        # 交换AGV任务：从选定的索引开始，将所有AGV1的任务分配给AGV2，将所有AGV2的任务分配给AGV1
        for i in range(len(new_agv_w)):
            if i >= start_idx1 and new_agv_w[i] == agv1:
                new_agv_w[i] = agv2
            elif i >= start_idx2 and new_agv_w[i] == agv2:
                new_agv_w[i] = agv1
        
        # 计算新字符串的完工时间
        calculate = Calculate()
        new_makespan = calculate.simulate(child_machine, new_agv_w, child_agv_f, child_schedule)
        
        # 如果新的完工时间更短，则返回新的AGV_W分配序列
        if new_makespan is not None and new_makespan < current_makespan:
            return new_agv_w, new_makespan
        
        return child_agv_w, current_makespan

    def local_search(self, child_machine: List[int], child_agv_w: List[int],
                    child_agv_f: List[int], child_schedule: List[int], current_makespan: float) -> Tuple[List[int], List[int], List[int], List[int], float, str]:
        """局部搜索
        
        Args:
            child_machine: 机器分配序列
            child_agv_w: 工件AGV分配序列
            child_agv_f: 物料AGV分配序列
            child_schedule: 调度序列
            current_makespan: 当前完工时间
            
        Returns:
            Tuple: (改进后的 AGV_W分配序列, AGV_F分配序列, 调度序列, 完工时间, 改进信息)
        """
        # AGV分配局部搜索
        logger.debug(f"执行AGV分配局部搜索，当前完工时间: {current_makespan:.2f}")
        new_agv_w, new_makespan = self.agv_local_search(
            child_machine, child_agv_w, child_agv_f, child_schedule, current_makespan
        )
        
        if new_makespan < current_makespan:
            improvement = current_makespan - new_makespan
            improvement_percentage = (improvement / current_makespan) * 100
            logger.debug(f"AGV分配局部搜索成功！完工时间从 {current_makespan:.2f} 降低到 {new_makespan:.2f}，改进: {improvement:.2f} ({improvement_percentage:.2f}%)")
            return child_machine, new_agv_w, child_agv_f, child_schedule, new_makespan, f"AGV_Local_Search_Success：{improvement:.2f} ({improvement_percentage:.2f}%)"
        
        logger.debug("局部搜索未能改进当前解")
        return child_machine, child_agv_w, child_agv_f, child_schedule, current_makespan, "No_Improvement"

    def evolve(self, start_generation: int = 0) -> SolutionType:
        """进化过程
        
        Args:
            start_generation: 起始代数，用于继续统计学习的迭代计数
        
        Returns:
            SolutionType: 最优解
        """
        # 添加局部搜索统计信息
        local_search_stats = {
            "Number_of_attempts": 0,
            "AGV_Local_Search_Success_Times": 0,
            "AGV_Local_Search_Improvement": 0.0
        }
        
        best_makespans = []
        
        for generation in range(self.max_iterations):
            current_generation = start_generation + generation
            new_population = []
            
            # 计算当前种群的完工时间
            makespans = [solution[1] for solution in self.population]
            
            # 保留最优精英解
            elite_count = 1  
            elite_solutions = sorted(self.population, key=lambda x: x[1])[:elite_count]
            
            # 将精英解直接加入新种群
            new_population.extend(elite_solutions)
            
            # 生成新一代种群的剩余部分
            while len(new_population) < self.population_size:
                # 选择父代
                parent1_idx = self.tournament_selection()
                parent2_idx = self.tournament_selection()
                
                # 确保两个父代不同
                # 如果种群大小大于1，则尽量选择不同的父代
                max_attempts = 3  # 最大尝试次数
                attempts = 0
                
                while parent1_idx == parent2_idx and attempts < max_attempts and len(self.population) > 1:
                    parent2_idx = self.tournament_selection()
                    attempts += 1
                
                # 获取父代个体
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # 初始化子代（默认继承parent1的基因）
                child_machine = parent1[2]
                child_agv_w = parent1[3]
                child_agv_f = parent1[4]
                child_schedule = parent1[5]

                # 交叉
                if random.random() < self.pc:
                    child_machine = self.crossover_assignment_strings(parent1[2], parent2[2], is_machine=True)
                    child_agv_w = self.crossover_assignment_strings(parent1[3], parent2[3])
                    child_agv_f = self.crossover_assignment_strings(parent1[4], parent2[4])
                    child_schedule = self.crossover_schedule_string(parent1[5], parent2[5])
                
                # 变异
                if random.random() < self.pm:
                    child_machine = self.mutate_assignment_string(child_machine, num_machines, is_machine=True)
                    child_agv_w = self.mutate_assignment_string(child_agv_w, num_agvs_w)
                    child_agv_f = self.mutate_assignment_string(child_agv_f, num_agvs_f)
                    child_schedule = self.mutate_schedule_string(child_schedule)
                            
                # 评估适应度
                calculate = Calculate()
                makespan = calculate.simulate(child_machine, child_agv_w, child_agv_f, child_schedule)
                
                if makespan is not None:
                    # 局部搜索
                    local_search_stats["Number_of_attempts"] += 1
                    original_makespan = makespan
                    child_machine, child_agv_w, child_agv_f, child_schedule, new_makespan, improvement_info = self.local_search(
                        child_machine, child_agv_w, child_agv_f, child_schedule, makespan
                    )
                    
                    # 记录局部搜索结果
                    if "AGV_Local_Search_Success" in improvement_info:
                        local_search_stats["AGV_Local_Search_Success_Times"] += 1
                        local_search_stats["AGV_Local_Search_Improvement"] += (original_makespan - new_makespan)
                    
                    # 更新子代
                    child_solution = (len(new_population), new_makespan, child_machine, child_agv_w, child_agv_f, child_schedule)
                    new_population.append(child_solution)
                                                    
                    # 更新最优解
                    if new_makespan < self.best_solution[1]:
                        self.best_solution = child_solution
                        self.best_generation = current_generation + 1  # 更新当前代数
        
            self.population = new_population
            
            # 记录当前代的最优解
            best_makespans.append(self.best_solution[1])
            
            # 如果当前代的最优解有改进，则输出当前最优解
            if len(best_makespans) > 1 and best_makespans[-1] < best_makespans[-2]:
                logger.info(f"In generation {current_generation + 1}，New best makespan: {self.best_solution[1]:.2f}")
                
    
        # 输出最终局部搜索统计信息
        if logger.level == logging.DEBUG:
            success_rate = (local_search_stats["AGV_Local_Search_Success_Times"]) / max(1, local_search_stats["Number_of_attempts"]) * 100
            logger.debug("=" * 50)
            logger.debug("Efficiency Statistics of Local Search :")
            logger.debug(f"Number of attempts: {local_search_stats['Number_of_attempts']}")
            logger.debug(f"Average Success Rate: {success_rate:.2f}%")
            
            # 避免除以零错误
            agv_avg_improvement = 0
            if local_search_stats['AGV_Local_Search_Success_Times'] > 0:
                agv_avg_improvement = local_search_stats['AGV_Local_Search_Improvement'] / local_search_stats['AGV_Local_Search_Success_Times']
                
            logger.debug(f"AGV Local Search Success: {local_search_stats['AGV_Local_Search_Success_Times']}，Average Improvement: {agv_avg_improvement:.2f}")
        else:
            # 输出调度统计信息
            agv_avg_improvement = 0
            if local_search_stats['AGV_Local_Search_Success_Times'] > 0:
                agv_avg_improvement = local_search_stats['AGV_Local_Search_Improvement'] / local_search_stats['AGV_Local_Search_Success_Times']
        
        return self.best_solution, best_makespans

if __name__ == "__main__":
    # 使用遗传算法优化调度方案
    debug = True
    
    # 记录开始时间
    start_time = time.time()
    
    el = EvolutionaryLearning(
        num_jobs=num_jobs,
        num_agvs_w=num_agvs_w,
        num_agvs_f=num_agvs_f,
        job_operations=job_operations,
        processing_times=processing_times,
        population_size=100,
        max_iterations=100,
        pc=0.8,
        pm=0.15
    )
    
    # 初始化种群
    logger.info("Initializing population...")
    el.initialize_population()
    
    # 进化
    logger.info("Starting evolution...")
    best_solution, best_makespans = el.evolve()

    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f}s")

    # 获取最优解
    logger.info(f"Best solution found with makespan: {best_solution[1]:.2f}s at generation {el.best_generation}")
    
    # 使用最优解进行调度并保存结果
    calculate = Calculate()
    schedule_results = calculate.simulate(*best_solution[2:], return_schedule=True)
    
    # 定义输出文件路径
    output_dir = "el_output"
    log_file = os.path.join(output_dir, f'el_log_{int(time.time())}.log')
    
    # 保存调度结果
    save_schedule_results(
        calculate,
        os.path.join(output_dir, "agv_w_schedule.csv"),
        os.path.join(output_dir, "agv_f_schedule.csv"),
        os.path.join(output_dir, "machine_schedule.csv")
    )
    
    # 获取工序处理顺序
    order = calculate.decode_schedule(best_solution[5], priority_dict)
    
    # 保存最优解
    save_best_solution(
        best_solution,
        order,
        os.path.join(output_dir, "best_solution.csv"),
        el.best_generation
    )
    
    # 绘制迭代曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
    plt.title('Evolutionary Learning Algorithm Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'el_convergence.png'))
    plt.close()

    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'el_convergence.png')}")
    logger.info(f"Detailed log saved to: {log_file}")
