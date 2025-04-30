"""
多种群进化学习模块

该模块负责实现多种群进化学习算法，包括：
- 基于启发式规则的机器分配、AGV分配和调度序列初始种群
- 基于Q学习的选择算子改进
- 基于问题性质的局部搜索算法: AGV+机器局部搜索
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

# 导入日志配置
from logger_config import setup_file_logger

# 获取日志记录器
logger = setup_file_logger('lmeo')

# 导入实例
from MK10 import (
    NUM_JOBS as num_jobs,
    NUM_MACHINES as num_machines,
    NUM_OPERATIONS as num_operations,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    AGV_W_TRANSPORT_TIMES as agv_w_transport_times,
    AGV_F_TRANSPORT_TIMES as agv_f_transport_times,
    JOB_OPERATIONS as job_operations,
    PROCESSING_TIMES as processing_times,
    PRIORITY_DICT as priority_dict
)

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


class SLMEO:
    """多种群进化学习算法类"""
    def __init__(self, 
                num_jobs: int,
                num_agvs_w: int,
                num_agvs_f: int,
                job_operations: Dict[int, int],
                processing_times: Dict[str, Dict[int, float]],
                population_size: int = 100,
                max_iterations: int = 100,
                pc: float = 0.9,  # 交叉概率
                pm: float = 0.05): # 变异概率

        self.num_jobs = num_jobs
        self.num_agvs_w = num_agvs_w
        self.num_agvs_f = num_agvs_f
        self.num_operations = num_operations
        self.num_machines = num_machines
        self.job_operations = job_operations
        self.processing_times = processing_times
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.pc = pc
        self.pm = pm
        
        # 初始化操作列表
        self.operations_list = []
        for job in range(1, self.num_jobs + 1):
            for op in range(1, self.job_operations[job] + 1):
                self.operations_list.append(f'o{job}_{op}')
                
        # 初始化种群
        self.population = []
        self.best_solution = None
        self.best_generation = 0  # 记录找到最优解的代数

        # 初始化Q-learning配对选择器
        self.q_mating = QLearningMating()
        
        # 初始化计算器
        self.calculate = Calculate()
    
    def initialize_population(self) -> None:
        """初始化种群
        
        包含三个子种群：
        1. 精英子种群(P1)：使用启发式方法生成
        2. 开发子种群(P2)：从概率矩阵采样生成
        3. 探索子种群(P3)：从概率矩阵采样生成
        """
        
        # 计算各子种群大小
        self.elite_size = max(int(self.population_size * 0.5), 50)  #  精英子种群占50%
        self.develop_size = max(int(self.population_size * 0.25), 25)  # 开发子种群占25%
        self.explore_size = self.population_size - self.elite_size - self.develop_size  # 探索子种群占剩余25%
        
        # 初始化精英子种群
        self.elite_population = self._initialize_elite_population()
        
        # 初始化概率矩阵
        self.machine_prob, self.agv_w_prob, self.agv_f_prob, self.schedule_prob = self._initialize_probability_matrices()
        
        # 初始化开发子种群和探索子种群
        self.develop_population = self._samplepop_from_probability(self.machine_prob, self.agv_w_prob, self.agv_f_prob, self.schedule_prob, self.develop_size)
        self.explore_population = self._samplepop_from_probability(self.machine_prob, self.agv_w_prob, self.agv_f_prob, self.schedule_prob, self.explore_size)
        
        # 先更新子种群中个体的ID，然后再合并
        for i in range(len(self.develop_population)):
            # 开发子种群ID从精英子种群长度开始
            idx = len(self.elite_population) + i
            self.develop_population[i] = (idx + 1, self.develop_population[i][1], self.develop_population[i][2], 
                                        self.develop_population[i][3], self.develop_population[i][4], self.develop_population[i][5])
            
        for i in range(len(self.explore_population)):
            # 探索子种群ID从精英+开发子种群长度开始
            idx = len(self.elite_population) + len(self.develop_population) + i
            self.explore_population[i] = (idx + 1, self.explore_population[i][1], self.explore_population[i][2], 
                                        self.explore_population[i][3], self.explore_population[i][4], self.explore_population[i][5])
        
        # 合并所有子种群
        self.population = self.elite_population + self.develop_population + self.explore_population
        
        # 更新全局最优解
        self.population.sort(key=lambda x: x[1])  # 按完工时间排序
        self.best_solution = self.population[0]
        # 更新子种群排序位置之和
        # 使用降序排列，越好的解排序位置越高
        # 先将种群按完工时间降序排列
        sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)
        rank_map = {ind[0]: idx+1 for idx, ind in enumerate(sorted_population)}
        self.elite_rank_sum = sum([rank_map[ind[0]] for ind in self.elite_population])
        self.develop_rank_sum = sum([rank_map[ind[0]] for ind in self.develop_population])
        self.explore_rank_sum = sum([rank_map[ind[0]] for ind in self.explore_population])
        logger.info(f"In generation 0/{self.max_iterations}, Subpopulation size - P1: {len(self.elite_population)} , P2: {len(self.develop_population)} , P3: {len(self.explore_population)} ")
        logger.info(f"The initial best makespan is: {self.best_solution[1]:.2f}")
    
    def _initialize_elite_population(self) -> list:
        """初始化精英子种群
        
        使用多种启发式方法生成高质量初始解
        
        Returns:
            list: 精英子种群
        """
        population = []
        
        for i in range(self.elite_size):
            # 随机生成调度字符串
            schedule_string = self._generate_random_schedule()
            
            # 生成三个随机数
            alpha1 = np.random.random()
            alpha2 = np.random.random()
            alpha3 = np.random.random()
            
            # 根据alpha1选择机器分配方法
            if alpha1 <= 0.25:
                machine_string = self._generate_machine_by_heuristic('_calc_completion_time1')
            elif alpha1 <= 0.5:
                machine_string = self._generate_machine_by_heuristic('_calc_completion_time2')
            elif alpha1 <= 0.75:
                machine_string = self._generate_machine_by_heuristic('_calc_completion_time3')
            else:
                machine_string = self._generate_random_machine()
            
            # 根据alpha2选择AGV_F分配方法
            if alpha2 <= 0.5:
                agv_f_string = self._generate_agv_by_heuristic4(self.num_agvs_f)
            else:
                agv_f_string = self._generate_random_agv(self.num_agvs_f)
            
            # 根据alpha3选择AGV_W分配方法
            if alpha3 <= 0.5:
                agv_w_string = self._generate_agv_by_heuristic4(self.num_agvs_w)
            else:
                agv_w_string = self._generate_random_agv(self.num_agvs_w)
            
            # 评估解的质量
            makespan = self.calculate.simulate(machine_string, agv_w_string, agv_f_string, schedule_string)
            
            if makespan is not None:
                solution = (i + 1, makespan, machine_string, agv_w_string, agv_f_string, schedule_string)
                population.append(solution)
        return population
    
    def _initialize_probability_matrices(self) -> tuple:
        """初始化概率矩阵
        
        Returns:
            tuple: (machine_prob, agv_w_prob, agv_f_prob, schedule_prob)
        """
        # 初始化机器分配概率矩阵
        machine_prob = np.zeros((self.num_operations, self.num_machines))
        
        # 获取每个工序的可用机器
        op_idx = 0
        for job in range(1, self.num_jobs + 1):
            for op in range(1, self.job_operations[job] + 1):
                op_key = f'o{job}_{op}'
                if op_key in self.processing_times:
                    available_machines = [m for m, time in self.processing_times[op_key].items() if time > 0]
                    prob_value = 1.0 / len(available_machines) if available_machines else 0
                    for m in available_machines:
                        machine_prob[op_idx, m-1] = prob_value  # 机器编号从1开始，索引从0开始
                op_idx += 1
        
        # 初始化AGV分配概率矩阵
        agv_w_prob = np.full((self.num_operations, self.num_agvs_w), 1/self.num_agvs_w)
        agv_f_prob = np.full((self.num_operations, self.num_agvs_f), 1/self.num_agvs_f)
        
        # 初始化调度概率矩阵
        schedule_prob = np.full((1, self.num_operations), 1/self.num_operations)
        
        return machine_prob, agv_w_prob, agv_f_prob, schedule_prob
    
    def _samplepop_from_probability(self, machine_prob, agv_w_prob, agv_f_prob, schedule_prob, size) -> list:
        """从概率矩阵采样生成子种群
        
        Args:
            machine_prob: 机器分配概率矩阵
            agv_w_prob: AGV_W分配概率矩阵
            agv_f_prob: AGV_F分配概率矩阵
            schedule_prob: 调度概率矩阵
            size: 子种群大小
            
        Returns:
            list: 子种群
        """
        population = []
        
        for i in range(size):
            # 从概率矩阵采样
            machine_string, agv_w_string, agv_f_string, schedule_string = self._sample_from_probability(
                machine_prob.copy(), agv_w_prob.copy(), agv_f_prob.copy(), schedule_prob.copy()
            )
            
            # 评估解的质量
            makespan = self.calculate.simulate(machine_string, agv_w_string, agv_f_string, schedule_string)
            
            if makespan is not None:
                solution = (i+1, makespan, machine_string, agv_w_string, agv_f_string, schedule_string)
                population.append(solution)
        
        # 按完工时间排序
        population.sort(key=lambda x: x[1])
        return population
    
    def _sample_from_probability(self, machine_prob, agv_w_prob, agv_f_prob, schedule_prob) -> tuple:
        """从概率矩阵采样生成个体
        
        Args:
            machine_prob: 机器分配概率矩阵
            agv_w_prob: AGV_W分配概率矩阵
            agv_f_prob: AGV_F分配概率矩阵
            schedule_prob: 调度概率矩阵
            
        Returns:
            tuple: (machine_string, agv_w_string, agv_f_string, schedule_string)
        """
        # 初始化结果列表
        machine_string = []
        agv_w_string = []
        agv_f_string = []
        schedule_string = []
        
        # 记录每个作业已采样的工序数
        job_sampled_count = {j: 0 for j in range(1, self.num_jobs + 1)}
        
        # 预先计算每个作业在schedule_prob中的位置范围
        job_positions = {}
        current_pos = 0
        for j in range(1, self.num_jobs + 1):
            job_positions[j] = (current_pos, current_pos + self.job_operations[j])
            current_pos += self.job_operations[j]

        for i in range(self.num_operations):
            # 1. 采样机器分配
            current_op_id = self.operations_list[i]
            
            # 获取当前工序的可用机器
            available_machines = list(self.processing_times[current_op_id].keys())
            available_machines_indices = [int(m) - 1 for m in available_machines]  # 转为0-based索引
            
            # 从非零概率的可用机器中选择
            valid_indices = [idx for idx in available_machines_indices if idx < len(machine_prob[i]) and machine_prob[i][idx] > 0]
            
            if valid_indices:
                # 从有效的非零概率机器中选择
                probs = np.array([machine_prob[i][idx] for idx in valid_indices])
                selected_idx = valid_indices[np.random.choice(len(valid_indices), p=probs/np.sum(probs))]
                machine = selected_idx + 1  # 转回1-based索引
            else:
                # 如果没有有效的非零概率机器，随机选择一个可用机器
                machine = int(random.choice(available_machines))
            
            machine_string.append(machine)
            
            # 2. 采样AGV分配
            agv_w_string.append(int(np.random.choice(self.num_agvs_w, p=agv_w_prob[i]/np.sum(agv_w_prob[i])) + 1))
            agv_f_string.append(int(np.random.choice(self.num_agvs_f, p=agv_f_prob[i]/np.sum(agv_f_prob[i])) + 1))
        
        # 3. 采样调度序列
        # 构建工序依赖图（前置工序 -> 后继工序）
        graph = {op_id: set() for op_id in self.operations_list}
        in_degree = {op_id: 0 for op_id in self.operations_list}
        for op_id, predecessors in priority_dict.items():
            for pred in predecessors:
                graph[pred].add(op_id)  # pred是op_id的前置工序，所以op_id是pred的后继工序
                in_degree[op_id] += 1  # op_id的入度加1
        # 1) 创建工序ID到索引的映射（1到工序数量）
        op_id_to_index = {op_id: i+1 for i, op_id in enumerate(self.operations_list)}
        # 2) 使用拓扑排序生成有效的工序序列
        schedule_string = []
        queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
        while queue:
            # 根据schedule_prob中的概率选择一个没有前置工序的工序
            queue_indices = [op_id_to_index[op] - 1 for op in queue]  # 得到队列中工序在schedule_prob中的索引(0-based)
            # 获取这些工序在schedule_prob中的概率值
            probs = np.array([schedule_prob[0][idx] for idx in queue_indices])
            # 如果所有概率都为0，则均等随机选择
            if np.sum(probs) <= 0:
                selected_idx = np.random.randint(0, len(queue))
                op_id = queue.pop(selected_idx)
            else:
                # 根据概率选择工序
                selected_idx = np.random.choice(len(queue), p=probs/np.sum(probs))
                op_id = queue.pop(selected_idx)
            # 将该工序的索引添加到调度序列中
            schedule_string.append(op_id_to_index[op_id])
            # 更新其后继工序的入度
            for successor in graph[op_id]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        # 检查是否所有工序都被排序（检测循环依赖）
        if len(schedule_string) != num_operations:
            raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")
        
        return machine_string, agv_w_string, agv_f_string, schedule_string
    
    def _generate_random_schedule(self) -> list:
        """生成符合工序优先级约束的工序调度编码
        
        生成一个1到工序数量的排列, 确保前置工序在后继工序之前
        
        Returns:
            list: 工序编码序列
        """
        
        # 创建工序ID到索引的映射（1到工序数量）
        op_id_to_index = {op_id: i+1 for i, op_id in enumerate(self.operations_list)}
        
        # 构建工序依赖图（前置工序 -> 后继工序）
        graph = {op_id: set() for op_id in self.operations_list}
        in_degree = {op_id: 0 for op_id in self.operations_list}
        
        for op_id, predecessors in priority_dict.items():
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
        if len(schedule_string) != num_operations:
            raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")
        
        return schedule_string
    
    def _generate_random_machine(self) -> list:
        """随机生成机器分配
        
        Returns:
            list: 机器分配序列
        """
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
            
            machine_string.append(machine)
        
        return machine_string
    
    def _generate_random_agv(self, num_agvs) -> list:
        """随机生成AGV分配
        
        Args:
            num_agvs: AGV数量
            
        Returns:
            list: AGV分配序列
        """
        agv_string = []
        
        # 确保AGV编号在有效范围内
        for _ in range(self.num_operations):
            # 随机选择一个1到num_agvs之间的AGV编号
            agv = np.random.randint(1, num_agvs + 1)
            agv_string.append(agv)
        
        return agv_string
    
    # 不同的完工时间计算策略
    def _calc_completion_time1(self, op_key, m, last_machine_location, machine_available_time):
        """计算完工时间策略1：运输时间 + 加工时间 + 机器最早可用时间"""
        current_loc = f"M{m}"
        trans_time = max(
            agv_f_transport_times[last_machine_location][current_loc],
            agv_w_transport_times[last_machine_location][current_loc]
        )
        return trans_time + self.processing_times[op_key][m] + machine_available_time[m]
    
    def _calc_completion_time2(self, op_key, m, last_machine_location, machine_available_time):
        """计算完工时间策略2：运输时间 + 加工时间"""
        current_loc = f"M{m}"
        trans_time = max(
            agv_f_transport_times[last_machine_location][current_loc],
            agv_w_transport_times[last_machine_location][current_loc]
        )
        return trans_time + self.processing_times[op_key][m]
    
    def _calc_completion_time3(self, op_key, m, last_machine_location, machine_available_time):
        """计算完工时间策略3：加工时间 + 机器最早可用时间"""
        return self.processing_times[op_key][m] + machine_available_time[m]
    
    def _generate_machine_by_heuristic(self, completion_time_method) -> list:
        """通用的启发式机器分配方法
        
        基于作业优先级的机器分配方法，按照作业顺序依次为每个工序分配机器，
        优先为没有前置工序的工序分配机器
        
        Args:
            completion_time_method: 计算完工时间的方法名称
        
        Returns:
            list: 机器分配序列
        """
        
        # 初始化机器分配字符串和机器可用时间
        machine_string = [0] * num_operations
        machine_available_time = [0] * (self.num_machines + 1)  # 索引0不使用
        
        # 记录已分配的工序和上一个机器位置
        assigned_operations = set()
        last_machine_location = "Warehouse"
        
        # 随机打乱作业顺序
        job_ids = list(range(1, self.num_jobs + 1))
        np.random.shuffle(job_ids)
        
        # 依次处理每个作业
        for job_id in job_ids:
            # 获取该作业的有效工序
            job_operations = [f'o{job_id}_{op}' for op in range(1, self.job_operations[job_id] + 1)]
            valid_operations = [op for op in job_operations if op in self.processing_times]
            
            # 使用拓扑排序构建工序优先级队列
            ready_operations = []
            remaining = valid_operations.copy()
            priority_copy = {op: set(priority_dict.get(op, set())) for op in remaining}
            
            # 拓扑排序
            while remaining:
                # 找出没有前置工序的工序
                no_pred = [op for op in remaining if not priority_copy[op]]
                
                if not no_pred:
                    # 处理循环依赖
                    ready_operations.extend(remaining)
                    break
                
                ready_operations.extend(no_pred)
                
                # 更新剩余工序和优先级
                for op in no_pred:
                    remaining.remove(op)
                
                for op in remaining:
                    priority_copy[op] -= set(no_pred)
            
            # 为工序分配机器
            for op_key in ready_operations:
                available_machines = list(self.processing_times[op_key].keys())
                
                if available_machines:
                    # 计算每台机器的完工时间
                    best_machine = None
                    min_completion_time = float('inf')
                    
                    for m in available_machines:
                        # 使用传入的方法计算完工时间
                        completion_time = getattr(self, completion_time_method)(
                            op_key, m, last_machine_location, machine_available_time
                        )
                        
                        # 更新最佳机器
                        if completion_time < min_completion_time:
                            min_completion_time = completion_time
                            best_machine = m
                    
                    # 更新机器分配和状态
                    std_idx = self.operations_list.index(op_key)
                    machine_string[std_idx] = best_machine
                    machine_available_time[best_machine] = min_completion_time
                    last_machine_location = f"M{best_machine}"
                else:
                    # 处理没有可用机器的情况
                    logger.warning(f"工序 {op_key} 没有可用机器")
                    std_idx = self.operations_list.index(op_key)
                    machine_string[std_idx] = 1
                
                # 标记为已分配
                assigned_operations.add(op_key)
        
        return machine_string
    
    def _generate_agv_by_heuristic4(self, num_agvs) -> list:
        """使用启发式方法4生成AGV分配
        
        平均分配工序，余数随机分配，然后随机打乱
        
        Args:
            num_agvs: AGV数量
            
        Returns:
            list: AGV分配序列
        """
        # 创建基本分配列表
        base_tasks = self.num_operations // num_agvs
        agv_string = [agv for agv in range(1, num_agvs + 1) for _ in range(base_tasks)]
        
        # 分配余数
        remainder = self.num_operations % num_agvs
        if remainder > 0:
            # 使用tolist()将NumPy数组转换为Python列表，其中元素会自动转换为原生整数
            agv_string.extend(np.random.choice(range(1, num_agvs + 1), size=remainder, replace=False).tolist())
        
        # 随机打乱
        np.random.shuffle(agv_string)
        
        return agv_string

    def convert_assignments_to_probability(self, assignments, assignment_type=None):
        """将分配方案转换为概率矩阵
        
        Args:
            assignments: 分配方案列表，每个元素是一个分配ID
            assignment_type: 分配类型，可以是'machine', 'agv_w', 'agv_f'或None
            
        Returns:
            概率矩阵，选定的分配ID的概率为1.0，其他为0.0
        """
        
        # 根据分配类型确定资源数量
        if assignment_type == 'machine':
            num_resources = num_machines
        elif assignment_type == 'agv_w':
            num_resources = num_agvs_w
        elif assignment_type == 'agv_f':
            num_resources = num_agvs_f
        else:
            # 如果没有指定类型，尝试自动判断
            if len(assignments) == num_operations:
                # 先检查最大值来判断类型
                max_id = max(assignments) if assignments else 0
                if max_id <= num_machines:
                    num_resources = num_machines
                elif max_id <= num_agvs_w:
                    num_resources = num_agvs_w
                elif max_id <= num_agvs_f:
                    num_resources = num_agvs_f
                else:
                    # 如果无法确定，则使用最大值
                    num_resources = max_id
            else:
                # 如果无法确定，则使用最大值
                num_resources = max(assignments) if assignments else 0
        
        # 创建与当前模型相同大小的矩阵
        if assignment_type == 'machine':
            prob_matrix = np.zeros((num_operations, num_machines))
        elif assignment_type == 'agv_w':
            prob_matrix = np.zeros((num_operations, num_agvs_w))
        elif assignment_type == 'agv_f':
            prob_matrix = np.zeros((num_operations, num_agvs_f))
        else:
            # 如果无法确定，则创建新矩阵
            prob_matrix = np.zeros((len(assignments), num_resources))
        
        # 将选定的分配ID的概率设置为1.0
        for i, resource_id in enumerate(assignments):
            if 1 <= resource_id <= prob_matrix.shape[1]:  # 确保资源ID在有效范围内
                prob_matrix[i, resource_id-1] = 1.0
        
        return prob_matrix
    
    def convert_schedule_to_probability(self, schedule_string):
        """将调度方案转换为概率矩阵
        
        Args:
            schedule_string: 调度方案列表，每个元素是一个工序的索引
            
        Returns:
            概率矩阵, 每个工序的优先级按照在schedule_string中的顺序排列
        """
        # 初始化概率矩阵
        schedule_prob = np.zeros((1, num_operations))
        
        # 根据工序在schedule_string中的顺序赋予优先级
        for i, op_index in enumerate(schedule_string):
            # 确保索引在有效范围内
            if 1 <= op_index <= num_operations:
                # 赋予优先级，从num_operations开始递减
                schedule_prob[0, op_index-1] = num_operations - i
        
        # 归一化处理
        if np.sum(schedule_prob) > 0:  # 避免除以零
            # 归一化使所有值之和为1
            schedule_prob = schedule_prob / np.sum(schedule_prob)
        else:
            # 如果所有值都为0，设置为均匀分布
            schedule_prob[:] = 1.0 / num_operations
        
        return schedule_prob

    def tournament_selection(self) -> int:
        """锦标赛选择
        
        Returns:
            int: 选中个体的索引
        """
        # 随机选择k个个体，但不超过种群大小
        k = min(2, len(self.population))  # 锦标赛大小，确保不超过种群大小
        
        # 如果种群为空，抛出更明确的错误
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
        
    def select_parent2(self, parent1_idx: int, population: list, makespans: list) -> tuple:
        """使用Q-learning选择第二个父代
        
        Args:
            parent1_idx: 第一个父代的索引
            population: 当前种群
            makespans: 当前种群的完工时间列表
            
        Returns:
            tuple: (选中的第二个父代索引, 状态, 动作)
        """
        # 确定parent1的状态
        state = self.q_mating.get_state(parent1_idx, population, self.elite_population, self.develop_population, self.explore_population)
        
        # 选择动作
        action = self.q_mating.choose_action(state)
        
        # 根据动作选择parent2
        population_size = len(population)
        third_size = population_size // 3
        
        # 按完工时间升序排序的种群索引
        sorted_indices = sorted(range(population_size), key=lambda i: makespans[i])
        
        # 根据动作选择不同质量的个体
        if action == 0:  # 动作1: 从前1/3中选择
            candidates = sorted_indices[:third_size]
        elif action == 1:  # 动作2: 从中间1/3中选择
            candidates = sorted_indices[third_size:2*third_size]
        else:  # 动作3: 从后1/3中选择
            candidates = sorted_indices[2*third_size:]
        
        # 确保候选列表不为空
        if not candidates:
            candidates = sorted_indices  # 如果候选列表为空，使用所有个体
        
        # 计算适应度值的倒数作为权重
        fitness_values = [1.0 / makespans[idx] for idx in candidates]
        total_fitness = sum(fitness_values)
        
        # 轮盘赋选择
        if total_fitness > 0:
            # 计算选择概率
            probabilities = [fitness / total_fitness for fitness in fitness_values]
            # 轮盘赋选择
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            parent2_idx = candidates[selected_idx]
        else:
            # 如果所有适应度值都为0，随机选择
            parent2_idx = random.choice(candidates)
        
        return parent2_idx, state, action
        
    def update_q_learning(self, state: int, action: int, parent1_makespan: float, parent2_makespan: float, child_makespan: float):
        """更新Q-learning
        
        Args:
            state: 当前状态
            action: 执行的动作
            parent1_makespan: 第一个父代的完工时间
            parent2_makespan: 第二个父代的完工时间
            child_makespan: 子代的完工时间
        """
        # 确定子代的质量类型
        # 先按完工时间升序排序
        sorted_population = sorted(self.population, key=lambda x: x[1])
        population_size = len(self.population)
        third_size = population_size // 3
        
        # 确定子代在排序后的位置
        child_position = None
        for i in range(population_size):
            if child_makespan <= sorted_population[i][1]:
                child_position = i
                break
        
        if child_position is None:
            child_position = population_size
        
        # 确定子代的质量类型
        if child_position <= third_size:
            child_quality = 0  # good
        elif child_position <= 2 * third_size:
            child_quality = 1  # medium
        else:
            child_quality = 2  # poor
        
        # 确定子代的用途
        child_usage = []
        
        # 如果子代属于前30%，则同时用于精英子种群P1\更新开发子种群P2和探索子种群P3的概率矩阵
        if child_position <= third_size:
            child_usage.append(0)  # P1
            child_usage.append(1)  # P2
            child_usage.append(2)  # P3
        # 如果子代属于前30%之后的部分，则仅用于更新探索子种群P3的概率矩阵
        else:
            child_usage.append(2)  # P3
        
        # 计算奖励
        if child_makespan < parent1_makespan:
            reward = 1.0  # 子代比parent1好
        elif child_makespan == parent1_makespan:
            reward = 0.2  # 子代与parent1相同
        else:
            reward = 0.0  # 子代比parent1差
        
        # 更新Q表
        self.q_mating.update(state, action, reward, child_quality, child_usage)
    
    def crossover_assignment_strings(self, parent1: List[int], parent2: List[int], is_machine: bool = False) -> List[int]:
        """交叉分配字符串（机器分配和AGV分配）"""
        if random.random() > self.pc:
            return parent1.copy()
            
        child = []
        
        for i, op_id in enumerate(self.operations_list):
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
        
        # 预处理：建立工序索引到作业ID的映射
        if not hasattr(self, '_op_index_to_job_cache'):
            self._op_index_to_job_cache = {}
            for i, op_id in enumerate(self.operations_list):
                job_id = int(op_id.split('_')[0][1:])  # 从'o1_2'提取'1'
                self._op_index_to_job_cache[i+1] = job_id
        
        op_index_to_job = self._op_index_to_job_cache
        
        # 随机将作业分为两组
        all_jobs = list(range(1, self.num_jobs + 1))
        num_jobs_in_jp1 = random.randint(1, self.num_jobs - 1)
        jp1 = set(random.sample(all_jobs, num_jobs_in_jp1))
        
        # 初始化子代和空位置列表
        child = [-1] * len(parent1)
        empty_positions = []
        
        # 从父代1复制JP1中的作业对应的工序
        for i, op_index in enumerate(parent1):
            if op_index_to_job[op_index] in jp1:
                child[i] = op_index
            else:
                empty_positions.append(i)
        
        # 从父代2中收集JP2作业对应的工序
        p2_jp2_ops = [op for op in parent2 if op_index_to_job[op] not in jp1]
        
        # 将父代2中属于JP2的工序填入子代的空位置
        for i, op_index in zip(empty_positions, p2_jp2_ops):
            child[i] = op_index
            
        return child
    
    def mutate_assignment_string(self, string: List[int], max_value: int, is_machine: bool = False) -> List[int]:
        """位变异算子"""
        mutated = string.copy()
        
        for i, op_id in enumerate(mutated):
            if random.random() < self.pm:
                if is_machine:
                    # 对于机器分配，随机选择一个可用机器
                    available_machines = list(self.processing_times[self.operations_list[i]].keys())
                    mutated[i] = int(random.choice(available_machines))
                else:
                    # 对于AGV分配，随机选择一个AGV
                    mutated[i] = random.randint(1, max_value)
        return mutated
    
    def mutate_schedule_string(self, schedule: List[int]) -> List[int]:
        """交换变异算子（保证优先级约束）"""
        # 如果不变异，直接返回原始调度
        if random.random() >= self.pm:
            return schedule.copy()
        
        # 获取工序索引到工序ID的映射
        op_index_to_id = {i+1: op_id for i, op_id in enumerate(self.operations_list)}
        
        # 尝试多次变异，直到找到合法的交换或达到最大尝试次数
        max_attempts = 3
        for _ in range(max_attempts):
            mutated = schedule.copy()
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
            # 验证交换后的调度是否合法
            try:
                self.calculate._validate_schedule_string(mutated, op_index_to_id, priority_dict)
                return mutated  # 找到合法交换，返回
            except ValueError:
                continue  # 交换导致非法调度，尝试下一次
        
        # 如果所有尝试都失败，返回原始调度
        return schedule.copy()

    def machine_local_search(self, child_machine: List[int], child_agv_w: List[int], 
                        child_agv_f: List[int], child_schedule: List[int], current_makespan: float) -> Tuple[List[int], float]:
        """机器分配局部搜索
        
        Args:
            child_machine: 机器分配序列
            child_agv_w: 工件AGV分配序列
            child_agv_f: 物料AGV分配序列
            child_schedule: 调度序列
            current_makespan: 当前完工时间
            
        Returns:
            Tuple[List[int], float]: 改进后的机器分配序列和完工时间
        """
        # 创建Calculate对象并执行调度方案
        calculate = Calculate()
        
        # 随机选择一个工序进行机器替换
        if self.operations_list:
            op_idx = random.randint(0, len(self.operations_list) - 1)
            op_id = self.operations_list[op_idx]
            
            # 获取当前分配的机器
            current_machine = child_machine[op_idx]
            
            # 获取该工序的可用机器及其处理时间
            available_processing_times = processing_times[op_id]
            
            # 找到处理时间最短的机器
            best_machine = min(available_processing_times, key=available_processing_times.get)
            
            # 如果当前机器不是最优的，尝试替换
            if best_machine != current_machine:
                new_machine = child_machine.copy()
                new_machine[op_idx] = best_machine
                new_makespan = calculate.simulate(new_machine, child_agv_w, child_agv_f, child_schedule)
                if new_makespan is not None and new_makespan < current_makespan:
                    return new_machine, new_makespan
    
        return child_machine, current_makespan

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
            Tuple: (改进后的机器分配序列, AGV_W分配序列, AGV_F分配序列, 调度序列, 完工时间, 改进信息)
        """
        # 1. 机器分配局部搜索
        logger.debug(f"执行机器分配局部搜索，当前完工时间: {current_makespan:.2f}")
        new_machine, new_makespan = self.machine_local_search(
            child_machine, child_agv_w, child_agv_f, child_schedule, current_makespan
        )
        
        if new_makespan < current_makespan:
            improvement = current_makespan - new_makespan
            improvement_percentage = (improvement / current_makespan) * 100
            logger.debug(f"机器分配局部搜索成功！完工时间从 {current_makespan:.2f} 降低到 {new_makespan:.2f}，改进: {improvement:.2f} ({improvement_percentage:.2f}%)")
            return new_machine, child_agv_w, child_agv_f, child_schedule, new_makespan, f"Machine_Local_Search_Success：{improvement:.2f} ({improvement_percentage:.2f}%)"
        
        # 2. AGV分配局部搜索
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

    def evolve(self) -> tuple:
        """进化过程
        
        根据evolutionary_learning.py中的evolve函数对initialize_population得到的初始种群population
        进行选择交叉变异和局部搜索，计算种群中每个个体的解，记录当前最优解，并分别更新下一代的三个子种群
        
        Args:
            start_generation: 起始代数，用于继续迭代的计数
            
        Returns:
            tuple: (最优解, 每代最优解的完工时间列表)
        """
        # 记录每代的最优解完工时间
        best_makespans = []
        # 初始化种群
        self.initialize_population()
        # 记录当前最优解
        best_makespans.append(self.best_solution[1])

        # 初始化开发概率矩阵
        develop_machine_prob = self.machine_prob
        develop_agv_w_prob = self.agv_w_prob
        develop_agv_f_prob = self.agv_f_prob
        develop_schedule_prob = self.schedule_prob
        # 初始化探索概率矩阵
        explore_machine_prob = self.machine_prob
        explore_agv_w_prob = self.agv_w_prob
        explore_agv_f_prob = self.agv_f_prob
        explore_schedule_prob = self.schedule_prob
        
        # 设置学习率
        alpha = 0.25  # 学习率参数
        # 添加局部搜索统计信息
        local_search_stats = {
            "Number_of_attempts": 0,
            "Machine_Local_Search_Success_Times": 0,
            "AGV_Local_Search_Success_Times": 0,
            "Machine_Local_Search_Improvement": 0.0,
            "AGV_Local_Search_Improvement": 0.0
        }
        # 迭代优化
        for generation in range(self.max_iterations):
            logger.debug(f"Start Generation {generation+1}/{self.max_iterations}...")
            
            # 1. 选择、交叉、变异和局部搜索
            
            # 记录父代种群的完工时间
            makespans = [ind[1] for ind in self.population]
            
            # 创建下一代种群
            offspring = []
            
            # 生成新一代种群（交叉变异操作）
            for _ in range(self.population_size):
                # 选择父代
                parent1_idx = self.tournament_selection()
                parent2_idx, state, action = self.select_parent2(parent1_idx, self.population, makespans)
                # 确保两个父代不同
                # 如果种群大小大于1，则尽量选择不同的父代
                max_attempts = 3  # 最大尝试次数
                attempts = 0
                
                while parent1_idx == parent2_idx and attempts < max_attempts and len(self.population) > 1:
                    parent2_idx, state, action = self.select_parent2(parent1_idx, self.population, makespans)
                    attempts += 1
                    
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
                    # 更新Q-learning
                    self.update_q_learning(state, action, parent1[1], parent2[1], makespan)

                    # 局部搜索
                    local_search_stats["Number_of_attempts"] += 1
                    original_makespan = makespan
                    child_machine, child_agv_w, child_agv_f, child_schedule, new_makespan, improvement_info = self.local_search(
                        child_machine, child_agv_w, child_agv_f, child_schedule, makespan
                    )
                    
                    # 记录局部搜索结果
                    if "Machine_Local_Search_Success" in improvement_info:
                        local_search_stats["Machine_Local_Search_Success_Times"] += 1
                        local_search_stats["Machine_Local_Search_Improvement"] += (original_makespan - new_makespan)
                    elif "AGV_Local_Search_Success" in improvement_info:
                        local_search_stats["AGV_Local_Search_Success_Times"] += 1
                        local_search_stats["AGV_Local_Search_Improvement"] += (original_makespan - new_makespan)
                    
                    # 更新子代
                    child_solution = (len(offspring), new_makespan, child_machine, child_agv_w, child_agv_f, child_schedule)
                    offspring.append(child_solution)
                                                    
                    # 更新最优解
                    if new_makespan < self.best_solution[1]:
                        self.best_solution = child_solution
                        self.best_generation = generation + 1  # 更新当前代数
            
            # 更新精英子种群P1
            offspring.sort(key=lambda x: x[1])
            elite_size = int(self.population_size * 0.3)  # 精英子种群规模为30%
            self.elite_population = offspring[:elite_size]
            
            # 2. 更新开发概率矩阵（基于精英子种群）
            # 将精英子种群的解转换为概率矩阵
            beta1 = np.zeros_like(develop_machine_prob)
            beta2 = np.zeros_like(develop_agv_w_prob)
            beta3 = np.zeros_like(develop_agv_f_prob)
            beta4 = np.zeros_like(develop_schedule_prob)
            
            # 从精英子种群中提取概率矩阵
            for solution in self.elite_population:
                # 提取解的各个组成部分
                _, _, machine_string, agv_w_string, agv_f_string, schedule_string = solution
                # 转换为概率矩阵（使用SL算法中的函数）
                machine_prob = self.convert_assignments_to_probability(machine_string, 'machine')
                agv_w_prob = self.convert_assignments_to_probability(agv_w_string, 'agv_w')
                agv_f_prob = self.convert_assignments_to_probability(agv_f_string, 'agv_f')
                schedule_prob = self.convert_schedule_to_probability(schedule_string)
                
                # 累加到beta矩阵
                beta1 += machine_prob
                beta2 += agv_w_prob
                beta3 += agv_f_prob
                beta4 += schedule_prob
            
            # 计算开发概率矩阵
            # new_machine_prob(ij)=(1-alpha)*machine_prob(ij)+[alpha/(0.3*N)]*β1(ij)
            develop_machine_prob = (1 - alpha) * develop_machine_prob + (alpha / (0.3 * self.population_size)) * beta1
            develop_agv_w_prob = (1 - alpha) * develop_agv_w_prob + (alpha / (0.3 * self.population_size)) * beta2
            develop_agv_f_prob = (1 - alpha) * develop_agv_f_prob + (alpha / (0.3 * self.population_size)) * beta3
            develop_schedule_prob = (1 - alpha) * develop_schedule_prob + (alpha / (0.3 * self.population_size)) * beta4
            
            # 3. 更新探索概率矩阵（基于所有种群的个体）
            # 将所有offspring个体的解转换为概率矩阵
            theta1 = np.zeros_like(explore_machine_prob)
            theta2 = np.zeros_like(explore_agv_w_prob)
            theta3 = np.zeros_like(explore_agv_f_prob)
            theta4 = np.zeros_like(explore_schedule_prob)
            
            # 从所有个体中提取概率矩阵
            for solution in self.population:
                # 提取解的各个组成部分
                _, _, machine_string, agv_w_string, agv_f_string, schedule_string = solution
                
                # 转换为概率矩阵
                machine_prob = self.convert_assignments_to_probability(machine_string, 'machine')
                agv_w_prob = self.convert_assignments_to_probability(agv_w_string, 'agv_w')
                agv_f_prob = self.convert_assignments_to_probability(agv_f_string, 'agv_f')
                schedule_prob = self.convert_schedule_to_probability(schedule_string)
                
                # 计算(1-该元素)的值并累加到theta矩阵
                theta1 += (1 - machine_prob)
                theta2 += (1 - agv_w_prob)
                theta3 += (1 - agv_f_prob)
                theta4 += (1 - schedule_prob)
            
            # 获取每个工序的可用机器数量
            available_machines_count = []
            
            for op_id in self.operations_list:
                # 获取可用机器数量
                available_machines = list(self.processing_times[op_id].keys())
                available_machines_count.append(len(available_machines))
            
            # 计算探索概率矩阵
            for i in range(self.num_operations):
                # 机器概率矩阵
                machine_factor = 1.0 / ((available_machines_count[i] - 1) * self.population_size) if available_machines_count[i] > 1 else 0
                explore_machine_prob[i] = (1 - alpha) * explore_machine_prob[i] + alpha * machine_factor * theta1[i]
                
            # AGV_W概率矩阵
            explore_agv_w_prob = (1 - alpha) * explore_agv_w_prob + alpha / ((self.num_agvs_w-1) * self.population_size) * theta2
            # AGV_F概率矩阵
            explore_agv_f_prob = (1 - alpha) * explore_agv_f_prob + alpha / ((self.num_agvs_f-1) * self.population_size) * theta3
            # 调度概率矩阵
            explore_schedule_prob = (1 - alpha) * explore_schedule_prob + alpha / (self.num_jobs * self.population_size) * theta4
            
            # 4. 计算子种群P2和P3的规模
            
            # 计算Ω值
            omega = self.develop_rank_sum / (self.develop_rank_sum + self.explore_rank_sum) if (self.develop_rank_sum + self.explore_rank_sum) > 0 else 0.5
            
            # 计算新的子种群规模
            new_develop_size = int((self.population_size - elite_size) * min(0.95, omega))
            new_explore_size = self.population_size - elite_size - new_develop_size
            
            # 5. 生成新一代子种群P2和P3
            # 生成新的开发子种群P2
            new_develop_population = self._samplepop_from_probability(
                develop_machine_prob, develop_agv_w_prob, develop_agv_f_prob, develop_schedule_prob, new_develop_size
            )
            # 生成新的探索子种群P3
            new_explore_population = self._samplepop_from_probability(
                explore_machine_prob, explore_agv_w_prob, explore_agv_f_prob, explore_schedule_prob, new_explore_size
            )
            
            # 6. 更新全局种群和全局最优解
            # 更新开发子种群和探索子种群ID
            for i in range(len(new_develop_population)):
                idx = len(self.elite_population) + i
                new_develop_population[i] = (idx + 1, new_develop_population[i][1], new_develop_population[i][2], 
                                        new_develop_population[i][3], new_develop_population[i][4], new_develop_population[i][5])
            
            for i in range(len(new_explore_population)):
                idx = len(self.elite_population) + len(new_develop_population) + i
                new_explore_population[i] = (idx + 1, new_explore_population[i][1], new_explore_population[i][2], 
                                        new_explore_population[i][3], new_explore_population[i][4], new_explore_population[i][5])
            
            self.develop_population = new_develop_population
            self.explore_population = new_explore_population
            
            # 合并所有子种群
            self.population = self.elite_population + self.develop_population + self.explore_population
            
            # 更新全局最优解
            self.population.sort(key=lambda x: x[1])  # 按完工时间排序
            if self.population[0][1] < self.best_solution[1]:
                self.best_solution = self.population[0]
                self.best_generation = generation + 1  # 更新当前代数
            
            # 更新子种群排序位置之和
            # 使用降序排列，越好的解排序位置越高
            # 先将种群按完工时间降序排列
            sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)
            rank_map = {ind[0]: idx+1 for idx, ind in enumerate(sorted_population)}
            self.elite_rank_sum = sum([rank_map[ind[0]] for ind in self.elite_population])
            self.develop_rank_sum = sum([rank_map[ind[0]] for ind in self.develop_population])
            self.explore_rank_sum = sum([rank_map[ind[0]] for ind in self.explore_population])
            
            # 记录当前代的最优解
            best_makespans.append(self.best_solution[1])
            # 如果当前代的最优解有改进，则输出当前最优解
            if len(best_makespans) > 1 and best_makespans[-1] < best_makespans[-2]:
                logger.info(f"In generation {generation + 1}，New best makespan: {self.best_solution[1]:.2f}")
            # 输出当前代的信息
            logger.info(f"In generation {generation + 1}/{self.max_iterations}, Subpopulation size - P1: {len(self.elite_population)}, P2: {len(self.develop_population)}, P3: {len(self.explore_population)}")
            logger.info(f"Subpopulation Quality Rank - P1: {self.elite_rank_sum}, P2: {self.develop_rank_sum}, P3: {self.explore_rank_sum}")
            
        return self.best_solution, best_makespans

class QLearningMating:
    """Q-learning用于指导配对选择"""
    
    def __init__(self, n_actions: int = 3, learning_rate: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        """初始化Q-learning配对选择器
        
        Args:
            n_actions: 动作空间大小（选择策略数量）
            learning_rate: Q学习率
            gamma: 折扣因子
            epsilon: ε-贪婪策略的探索概率
        """
        # 状态设计: 9种状态，组合了子种群和解质量
        # (P1,good)=0, (P1,medium)=1, (P1,poor)=2
        # (P2,good)=3, (P2,medium)=4, (P2,poor)=5
        # (P3,good)=6, (P3,medium)=7, (P3,poor)=8
        self.n_states = 9
        self.n_actions = n_actions  # 3种动作：从前1/3、中1/3、后1/3中选择
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化Q表
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 记录历史数据
        self.history = []
    
    def get_state(self, parent1_idx: int, population: list, elite_population: list, develop_population: list, explore_population: list) -> int:
        """确定parent1的状态
        
        Args:
            parent1_idx: parent1在种群中的索引
            population: 当前种群
            elite_population: 精英子种群
            develop_population: 开发子种群
            explore_population: 探索子种群
        
        Returns:
            int: 状态索引 (0-8)
        """
        parent1 = population[parent1_idx]
        parent1_id = parent1[0]
        
        # 确定parent1所属的子种群
        subpop_type = 0  # 默认为精英子种群
        for ind in elite_population:
            if ind[0] == parent1_id:
                subpop_type = 0  # P1
                break
        else:
            for ind in develop_population:
                if ind[0] == parent1_id:
                    subpop_type = 1  # P2
                    break
            else:
                subpop_type = 2  # P3
        
        # 确定parent1的解质量
        # 先按完工时间升序排序
        sorted_population = sorted(population, key=lambda x: x[1])
        population_size = len(population)
        third_size = population_size // 3
        
        # 找到parent1在排序后的位置
        for i, ind in enumerate(sorted_population):
            if ind[0] == parent1_id:
                if i < third_size:
                    quality_type = 0  # good
                elif i < 2 * third_size:
                    quality_type = 1  # medium
                else:
                    quality_type = 2  # poor
                break
        
        # 计算状态索引
        state = subpop_type * 3 + quality_type
        return state
    
    def choose_action(self, state: int) -> int:
        """选择动作（配对策略）
        
        Args:
            state: 当前状态 (0-8)
        
        Returns:
            int: 选择的动作 (0-2)
        """
        if random.random() < self.epsilon:  # 探索
            return random.randint(0, self.n_actions - 1)
        else:  # 利用
            return np.argmax(self.q_table[state])
    
    def get_next_state(self, child_quality: int, child_usage: list) -> int:
        """根据子代的质量和用途确定下一个状态
        
        Args:
            child_quality: 子代的质量类型 (0=good, 1=medium, 2=poor)
            child_usage: 子代用于构成的子种群列表 (包含0=P1, 1=P2, 2=P3)
        
        Returns:
            int: 下一个状态
        """
        # 初始化过渡概率矩阵
        transition_probs = np.zeros(self.n_states)
        
        # 确定可能的状态
        possible_states = []
        for subpop in child_usage:
            state_idx = subpop * 3 + child_quality
            possible_states.append(state_idx)
        
        # 设置过渡概率
        for state_idx in possible_states:
            transition_probs[state_idx] = 1.0 / len(possible_states)
        
        # 从过渡概率中采样下一个状态
        next_state = np.random.choice(self.n_states, p=transition_probs)
        return next_state
    
    def update(self, state: int, action: int, reward: float, child_quality: int, child_usage: list):
        """更新Q表
        
        Args:
            state: 当前状态 (0-8)
            action: 执行的动作 (0-2)
            reward: 获得的奖励
            child_quality: 子代的质量类型 (0=good, 1=medium, 2=poor)
            child_usage: 子代用于构成的子种群列表 (包含0=P1, 1=P2, 2=P3)
        """
        # 初始化过渡概率矩阵
        transition_probs = np.zeros(self.n_states)
        
        # 确定可能的状态
        possible_states = []
        for subpop in child_usage:
            state_idx = subpop * 3 + child_quality
            possible_states.append(state_idx)
        
        # 设置过渡概率
        for state_idx in possible_states:
            transition_probs[state_idx] = 1.0 / len(possible_states)
        
        # 计算next_max
        next_max = 0
        for next_state in range(self.n_states):
            if transition_probs[next_state] > 0:
                next_max += transition_probs[next_state] * np.max(self.q_table[next_state])
        
        # 更新Q值
        old_value = self.q_table[state, action]
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
        
        # 记录更新历史
        self.history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'child_quality': child_quality,
            'child_usage': child_usage,
            'q_value': new_value
        })

if __name__ == "__main__":
    # 使用遗传算法优化调度方案
    debug = True
    
    # 记录开始时间
    start_time = time.time()
    
    lmeo = SLMEO(
        num_jobs=num_jobs,
        num_agvs_w=num_agvs_w,
        num_agvs_f=num_agvs_f,
        job_operations=job_operations,
        processing_times=processing_times,
        population_size=100,
        max_iterations=100,
        pc=0.9,
        pm=0.05
    )

    # 开始进化
    logger.info("Starting evolution...")
    best_solution, best_makespans = lmeo.evolve()

    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f}s")

    # 获取最优解
    logger.info(f"Best solution found with makespan: {best_solution[1]:.2f}s at generation {lmeo.best_generation}")
    
    # 使用最优解进行调度并保存结果
    calculate = Calculate()
    schedule_results = calculate.simulate(*best_solution[2:], return_schedule=True)
    
    # 定义输出文件路径
    output_dir = "lmeo_output"
    log_file = os.path.join(output_dir, f'lmeo_log_{int(time.time())}.log')
    
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
        lmeo.best_generation
    )
    
    # 绘制迭代曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
    plt.title('LMEO Algorithm Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Makespan')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lmeo_convergence.png'))
    plt.close()

    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'lmeo_convergence.png')}")
    logger.info(f"Detailed log saved to: {log_file}")
