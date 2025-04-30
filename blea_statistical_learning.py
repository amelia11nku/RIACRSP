"""
统计学习模块[BLEA论文复现,学习率为固定值0.8]

该模块负责实现基于概率模型的连续编码的统计学习优化算法，包括：
- 生成概率矩阵
- 解码概率矩阵为具体的调度方案
- 迭代优化概率模型
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

# 导入计算模块
from calculate import Calculate
from blea_evolutionary_learning import save_schedule_results, save_best_solution

# 导入配置
from MK10 import (
    NUM_OPERATIONS as num_operations,
    NUM_MACHINES as num_machines,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    JOB_OPERATIONS as job_operations,
    PROCESSING_TIMES as processing_times,
    PRIORITY_DICT as priority_dict,
)

# 导入统一日志配置
from logger_config import setup_file_logger

# 获取日志记录器
logger = setup_file_logger('sl')

# ===== 类型别名定义 =====
SolutionType = Tuple[int, float, List[int], List[int], List[int], List[int]]  # (solution_id, makespan, machine_string, agv_w_string, agv_f_string, schedule_string)
ResultType = List[Dict[str, object]]  # 调度结果类型

class StatisticalLearning:

    def __init__(self, job_operations, num_operations, num_machines, num_agvs_f, num_agvs_w,
                processing_times,
                population_size=100,
                max_iterations=100,
                learning_rate=0.8):
        """
        初始化连续编码类
        
        Args:
            job_operations (dict): 每个作业的工序数量
            num_operations (int): 总工序数量
            num_machines (int): 机器数量
            num_agvs_f (int): AGV_F数量
            num_agvs_w (int): AGV_W数量
            processing_times (dict): 工序在各机器上的加工时间
            population_size (int): 种群大小
            max_iterations (int): 最大迭代次数
            learning_rate (float): 学习率α
        """
        
        self.job_operations = job_operations
        self.num_machines = num_machines
        self.num_agvs_f = num_agvs_f
        self.num_agvs_w = num_agvs_w
        self.processing_times = processing_times
        self.num_operations = num_operations
        
        # 算法参数
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
        # 创建实例用于计算完工时间
        self.calculate = Calculate()

    def clip_to_range(self, array, min_val=0.0, max_val=1.0):
        """将数组的值限制在指定范围内"""
        return np.clip(array, min_val, max_val)

    def initialize_population(self):
        """
        初始化种群，使用0到1之间的均匀分布随机生成
        
        Returns:
            list: 初始种群列表，每个元素是一个个体(machine_prob, agv_w_prob, agv_f_prob, schedule_prob)
        """
        population = []
        
        # 使用0到1之间的均匀分布随机生成所有个体
        for _ in range(self.population_size):
            # 使用均匀分布随机生成概率矩阵
            machine_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_machines))
            agv_w_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_agvs_w))
            agv_f_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_agvs_f))
            schedule_prob = self.generate_random_schedule_prob(np.full((1, self.num_operations), 1/self.num_operations))
            population.append((machine_prob, agv_w_prob, agv_f_prob, schedule_prob))
        
        return population

    def initialize_probability_model(self, population, makespans):
        """
        初始化概率模型
        
        Args:
            population (list): 初始种群
            makespans (list): 对应的完工时间列表
        """
        # 初始化均值和方差矩阵
        self.mean_machine = np.zeros((self.num_operations, self.num_machines))
        self.var_machine = np.zeros((self.num_operations, self.num_machines))
        self.mean_agv_f = np.zeros((self.num_operations, self.num_agvs_f))
        self.var_agv_f = np.zeros((self.num_operations, self.num_agvs_f))
        self.mean_agv_w = np.zeros((self.num_operations, self.num_agvs_w))
        self.var_agv_w = np.zeros((self.num_operations, self.num_agvs_w))
        self.mean_schedule = np.zeros((1, self.num_operations))
        self.var_schedule = np.zeros((1, self.num_operations))
        
        # 计算均值
        machine_means = np.zeros((self.num_operations, self.num_machines))
        agv_f_means = np.zeros((self.num_operations, self.num_agvs_f))
        agv_w_means = np.zeros((self.num_operations, self.num_agvs_w))
        schedule_means = np.zeros((1, self.num_operations))
        
        # 计算方差
        machine_vars = np.zeros((self.num_operations, self.num_machines))
        agv_f_vars = np.zeros((self.num_operations, self.num_agvs_f))
        agv_w_vars = np.zeros((self.num_operations, self.num_agvs_w))
        schedule_vars = np.zeros((1, self.num_operations))
        
        # 计算均值和方差
        for individual, makespan in zip(population, makespans):
            machine_prob, agv_w_prob, agv_f_prob, schedule_prob = individual
            
            # 计算均值
            machine_means += machine_prob
            agv_f_means += agv_f_prob
            agv_w_means += agv_w_prob
            schedule_means += schedule_prob
            
            # 计算方差
            machine_vars += (machine_prob - machine_means / len(population)) ** 2
            agv_f_vars += (agv_f_prob - agv_f_means / len(population)) ** 2
            agv_w_vars += (agv_w_prob - agv_w_means / len(population)) ** 2
            schedule_vars += (schedule_prob - schedule_means / len(population)) ** 2
        
        # 计算均值
        population_size = len(population)
        if population_size > 0:
            machine_means /= population_size
            agv_f_means /= population_size
            agv_w_means /= population_size
            schedule_means /= population_size
        
        # 计算方差
        if population_size > 1:  # 防止除以0
            machine_vars /= (population_size - 1)
            agv_f_vars /= (population_size - 1)
            agv_w_vars /= (population_size - 1)
            schedule_vars /= (population_size - 1)
        
        # 初始化概率模型
        self.mean_machine = machine_means
        self.var_machine = machine_vars
        self.mean_agv_f = agv_f_means
        self.var_agv_f = agv_f_vars
        self.mean_agv_w = agv_w_means
        self.var_agv_w = agv_w_vars
        self.mean_schedule = schedule_means
        self.var_schedule = schedule_vars

    def generate_population(self, exploration_factor=0.2):
        """
        根据概率模型生成一个新种群，并确保生成的schedule_prob符合优先级约束
        
        Returns:
            list: 种群列表，每个元素是一个个体(machine_prob, agv_w_prob, agv_f_prob, schedule_prob)
        """
        population = []

        for _ in range(self.population_size):
            # 获取工序索引到工序ID的映射
            operations_list = self.calculate.get_operations_list()
            op_index_to_id = {i+1: op_id for i, op_id in enumerate(operations_list)}
            
            # 使用概率模型生成新个体
            if random.random() < exploration_factor:
                # 增加探索：生成更多随机解
                machine_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_machines))
                agv_w_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_agvs_w))
                agv_f_prob = np.random.uniform(0, 1, size=(self.num_operations, self.num_agvs_f))
                # 直接使用拓扑排序生成合法的调度概率
                schedule_prob = self.generate_random_schedule_prob(self.mean_schedule)
            else:
                # 基于当前概率模型生成
                machine_prob = np.random.normal(self.mean_machine, np.sqrt(self.var_machine))
                agv_w_prob = np.random.normal(self.mean_agv_w, np.sqrt(self.var_agv_w))
                agv_f_prob = np.random.normal(self.mean_agv_f, np.sqrt(self.var_agv_f))
                
                # 确保 schedule_prob 符合优先级约束
                valid_schedule = False
                max_attempts = 100
                
                for _ in range(max_attempts):
                    try:
                        # 生成新的schedule_prob
                        schedule_prob = np.random.normal(self.mean_schedule, np.sqrt(self.var_schedule))
                        # 确保非负并归一化
                        schedule_prob = np.maximum(0, schedule_prob)
                        if np.sum(schedule_prob) > 0:
                            schedule_prob = schedule_prob / np.sum(schedule_prob)
                        
                        # 将schedule_prob转换为调度字符串
                        schedule_string = self.convert_to_schedule_string(schedule_prob)
                        
                        # 验证调度字符串是否合法
                        self.calculate._validate_schedule_string(schedule_string, op_index_to_id, priority_dict)
                        
                        # 如果合法，标记并跳出循环
                        valid_schedule = True
                        break
                        
                    except ValueError:
                        # 如果不合法，继续尝试
                        continue
                
                # 如果达到最大尝试次数仍然无法生成合法解，使用拓扑排序生成
                if not valid_schedule:
                    schedule_prob = self.generate_random_schedule_prob(self.mean_schedule)
            
            # 防止概率值为负数
            machine_prob = np.maximum(0, machine_prob)
            agv_w_prob = np.maximum(0, agv_w_prob)
            agv_f_prob = np.maximum(0, agv_f_prob)
            schedule_prob = np.maximum(0, schedule_prob)
            
            population.append((machine_prob, agv_w_prob, agv_f_prob, schedule_prob))
    
        return population[:self.population_size]

    def generate_random_schedule_prob(self, mean_schedule):
        """
        使用拓扑排序生成一个有效的调度概率矩阵
        概率值基于mean_schedule中的值，并规范到0~1之间
        
        Returns:
            np.ndarray: 调度概率矩阵，形状为(1, self.num_operations)
        """
        # 初始化概率矩阵
        schedule_prob = np.zeros((1, self.num_operations))
        
        # 获取所有工序的顺序列表
        operations_list = self.calculate.get_operations_list()
        # 1) 创建工序ID到索引的映射（1到工序数量）
        op_id_to_index = {op_id: i+1 for i, op_id in enumerate(operations_list)}
        # 2) 构建工序依赖图（前置工序 -> 后继工序）
        graph = {op_id: set() for op_id in operations_list}
        in_degree = {op_id: 0 for op_id in operations_list}
        for op_id, predecessors in priority_dict.items():
            for pred in predecessors:
                graph[pred].add(op_id)  # pred是op_id的前置工序，所以op_id是pred的后继工序
                in_degree[op_id] += 1  # op_id的入度加1
        
        # 3) 使用拓扑排序生成有效的工序序列
        schedule_string = []  # 存储拓扑排序结果
        queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            # 根据mean_schedule中的概率选择一个没有前置工序的工序
            queue_indices = [op_id_to_index[op] - 1 for op in queue]  # 得到队列中工序在mean_schedule中的索引(0-based)
            # 获取这些工序在mean_schedule中的概率值
            probs = np.array([mean_schedule[0][idx] for idx in queue_indices])
            # 如果所有概率都为0，则均等随机选择
            if np.sum(probs) <= 0:
                selected_idx = np.random.randint(0, len(queue))
                op_id = queue.pop(selected_idx)
            else:
                # 根据概率选择工序
                selected_idx = np.random.choice(len(queue), p=probs/np.sum(probs))
                op_id = queue.pop(selected_idx)
            
            # 将该工序的索引添加到调度序列中
            op_index = op_id_to_index[op_id] - 1  # 转为0-based索引
            schedule_string.append(op_index)
            
            # 更新其后继工序的入度
            for successor in graph[op_id]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # 检查是否所有工序都被排序（检测循环依赖）
        if len(schedule_string) != self.num_operations:
            raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")
        
        # 根据拓扑排序结果生成概率矩阵
        # 位置越靠前的工序概率值越高
        for i, op_index in enumerate(schedule_string):
            # 赋予概率值，位置越靠前概率越高
            schedule_prob[0, op_index] = self.num_operations - i
        
        # 归一化处理，确保概率值在0~1之间且和为1
        if np.sum(schedule_prob) > 0:
            schedule_prob = schedule_prob / np.sum(schedule_prob)
        else:
            # 如果所有概率都为0，则使用均匀分布
            schedule_prob = np.ones((1, self.num_operations)) / self.num_operations
        
        return schedule_prob

    def get_operation_name(self, op_id):
        """获取工序名称"""
        current_op = op_id
        for job_id, num_ops in self.job_operations.items():
            if current_op < num_ops:
                return f'o{job_id}_{current_op + 1}'
            current_op -= num_ops
        return None

    def convert_to_assignments_string(self, machine_prob, agv_w_prob, agv_f_prob):
        """
        将分配概率矩阵转换为离散的分配字符串
        
        Args:
            machine_prob (np.ndarray): 机器分配概率矩阵 [工序数量 x 机器数量]
            agv_f_prob (np.ndarray): AGV_F分配概率矩阵 [工序数量 x AGV_F数量]
            agv_w_prob (np.ndarray): AGV_W分配概率矩阵 [工序数量 x AGV_W数量]
            
        Returns:
            Tuple[List[int], List[int], List[int]]: (机器分配列表, AGV_F分配列表, AGV_W分配列表)
        """
        # 转换机器分配概率矩阵
        machine_assignments = []
        op_id = 0
        for job_id in self.job_operations:
            for op_idx in range(self.job_operations[job_id]):
                # 获取工序名称
                op_name = self.get_operation_name(op_id)
                # 获取可用机器列表（转换为0-based索引）
                available_machines = [m-1 for m in self.processing_times[op_name].keys()]
                
                # 只考虑可用机器的概率值
                prob_values = machine_prob[op_id, available_machines]
                # 找到最大概率值对应的索引
                max_idx = available_machines[np.argmax(prob_values)]
                # 将机器序号（1-based）添加到分配列表
                machine_assignments.append(max_idx + 1)
                op_id += 1
        
        # 转换AGV_F分配概率矩阵
        agv_f_assignments = []
        for op_idx in range(agv_f_prob.shape[0]):
            # 找到最大概率值对应的AGV序号
            max_idx = np.argmax(agv_f_prob[op_idx])
            # 将AGV序号（1-based）添加到分配列表，并转换为普通int类型
            agv_f_assignments.append(int(max_idx + 1))
        
        # 转换AGV_W分配概率矩阵
        agv_w_assignments = []
        for op_idx in range(agv_w_prob.shape[0]):
            # 找到最大概率值对应的AGV序号
            max_idx = np.argmax(agv_w_prob[op_idx])
            # 将AGV序号（1-based）添加到分配列表，并转换为普通int类型
            agv_w_assignments.append(int(max_idx + 1))
        
        return machine_assignments, agv_w_assignments, agv_f_assignments

    def convert_to_schedule_string(self, schedule_prob):
        """
        调度概率矩阵转换为离散字符串
        
        Args:
            schedule_prob (np.ndarray): 调度概率矩阵，每个工序的优先级
            
        Returns:
            schedule_string: 调度字符串, 表示工序的处理顺序
        """
        # 创建工序索引和优先级的对应关系
        op_priorities = []
        for i in range(num_operations):
            # 工序索引从1开始
            op_index = i + 1
            priority = schedule_prob[0, i]
            op_priorities.append((op_index, priority))
        
        # 根据优先级对工序进行排序（降序）
        sorted_ops = sorted(op_priorities, key=lambda x: x[1], reverse=True)
        
        # 生成调度字符串，只取工序索引
        schedule_string = [op[0] for op in sorted_ops]
        
        return schedule_string

    def update_probability_model(self, population, makespans):
        """
        更新概率模型
        
        Args:
            population (list): 当前种群
            makespans (list): 对应的完工时间列表
        """
        # 计算权重（完工时间越小权重越大，均值调整越倾向于当前的概率矩阵元素）
        max_makespan = max(makespans)
        min_makespan = min(makespans)
        avg_makespan = sum(makespans) / len(makespans)
        
        # 如果最大完工时间和最小完工时间相等，则所有权重设置为相同值
        if max_makespan == min_makespan:
            weights = [1.0 / len(makespans)] * len(makespans)
        else:
            weights = [(max_makespan - m) / (max_makespan - min_makespan) for m in makespans]
            total_weight = sum(weights)
            if total_weight > 0:  # 避免除以0
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(makespans)] * len(makespans)
        
        # 初始化加权和矩阵
        weighted_sum_machine = np.zeros_like(self.mean_machine)
        weighted_sum_agv_f = np.zeros_like(self.mean_agv_f)
        weighted_sum_agv_w = np.zeros_like(self.mean_agv_w)
        weighted_sum_schedule = np.zeros_like(self.mean_schedule)
        
        # 计算加权和
        valid_count = 0  # 有效个体计数
        for individual, weight in zip(population, weights):
            try:
                machine_prob, agv_w_prob, agv_f_prob, schedule_prob = individual
                
                # 检查维度是否匹配，如果不匹配则跳过该个体
                if machine_prob.shape != self.mean_machine.shape:
                    logger.warning(f"机器概率矩阵维度不匹配: {machine_prob.shape} vs {self.mean_machine.shape}")
                    continue
                if agv_w_prob.shape != self.mean_agv_w.shape:
                    logger.warning(f"AGV_W概率矩阵维度不匹配: {agv_w_prob.shape} vs {self.mean_agv_w.shape}")
                    continue
                if agv_f_prob.shape != self.mean_agv_f.shape:
                    logger.warning(f"AGV_F概率矩阵维度不匹配: {agv_f_prob.shape} vs {self.mean_agv_f.shape}")
                    continue
                if schedule_prob.shape != self.mean_schedule.shape:
                    logger.warning(f"调度概率矩阵维度不匹配: {schedule_prob.shape} vs {self.mean_schedule.shape}")
                    continue
                
                # 所有维度都匹配，更新加权和
                weighted_sum_machine += weight * np.array(machine_prob)
                weighted_sum_agv_f += weight * np.array(agv_f_prob)
                weighted_sum_agv_w += weight * np.array(agv_w_prob)
                weighted_sum_schedule += weight * np.array(schedule_prob)
                valid_count += 1
            except Exception as e:
                logger.error(f"更新概率模型时出错: {str(e)}")
                continue
        
        # 如果没有有效个体，则不更新概率模型
        if valid_count == 0:
            logger.warning("没有有效个体用于更新概率模型")
            return
        
        # 更新概率模型的均值
        self.mean_machine = self.learning_rate * weighted_sum_machine + (1 - self.learning_rate) * self.mean_machine
        self.mean_agv_f = self.learning_rate * weighted_sum_agv_f + (1 - self.learning_rate) * self.mean_agv_f
        self.mean_agv_w = self.learning_rate * weighted_sum_agv_w + (1 - self.learning_rate) * self.mean_agv_w
        self.mean_schedule = self.learning_rate * weighted_sum_schedule + (1 - self.learning_rate) * self.mean_schedule
    
    def optimize(self):
        """
        执行优化过程
        
        Returns:
            dict: 包含最优解信息的字典
        """
        # 初始化最优解记录
        best_solution = None
        best_generation = 0
        best_makespans = []  # 记录每代的最优解
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            
            # 第一次迭代时初始化种群，之后根据概率模型生成新种群
            if iteration == 0:
                # 初始化种群
                population = self.initialize_population()
            else:
                # 根据当前概率模型抽样生成子代
                population = self.generate_population()
            
            # 评估种群
            makespans = []
            solutions = []
            
            for individual in population:
                machine_prob, agv_w_prob, agv_f_prob, schedule_prob = individual
                
                # 生成机器分配、AGV_W分配、AGV_F分配和调度字符串
                machine_assignments, agv_w_assignments, agv_f_assignments = self.convert_to_assignments_string(
                                                                                machine_prob, agv_w_prob, agv_f_prob)
                schedule_string = self.convert_to_schedule_string(schedule_prob)

                # 计算完工时间
                makespan = self.calculate.simulate(
                    machine_assignments, 
                    agv_w_assignments, 
                    agv_f_assignments, 
                    schedule_string
                )
                
                if makespan is not None:
                    makespans.append(makespan)
                    solution_tuple = (
                        len(makespans) + 1, 
                        makespan, 
                        machine_assignments, 
                        agv_w_assignments, 
                        agv_f_assignments, 
                        schedule_string
                    )
                    solutions.append(solution_tuple)
            
            # 如果没有有效解，跳过当前迭代
            if not makespans:
                logger.warning(f"第{iteration}代没有有效解，跳过当前迭代")
                continue
            
            # 第一次迭代时初始化概率模型，之后更新概率模型
            if iteration == 0:
                # 初始化概率模型
                self.initialize_probability_model(population, makespans)
                logger.info("Initialization probability model completed.")
            else:
                # 更新概率模型
                self.update_probability_model(population, makespans)
                logger.debug(f"第{iteration}代概率模型更新完成")
            
            # 更新最优解
            current_min_makespan = min(makespans)
            current_min_index = makespans.index(current_min_makespan)
            current_best_solution = solutions[current_min_index]
            
            if best_solution is None or current_min_makespan < best_solution[1]:
                best_solution = current_best_solution
                best_generation = iteration
                logger.info(f"In generation {iteration + 1}，New best makespan: {current_min_makespan:.2f}")
            
            # 记录当前代的最优解
            best_makespans.append(best_solution[1])
            
            # 打印当前迭代信息
            logger.debug(f"第{iteration}代完成，当前最优解完工时间: {current_min_makespan}，历史最优解完工时间: {best_solution[1]}")
        
        # 返回最优解和当前种群
        if best_solution:
            return {
                "best_solution": best_solution,
                "best_makespan": best_solution[1],
                "best_generation": best_generation,
                "best_makespans": best_makespans,
                "solutions": solutions  # 返回整个种群
            }
        else:
            raise ValueError("优化失败，未找到有效解")


if __name__ == "__main__":
    try:
        # 记录开始时间
        start_time = time.time()

        sl = StatisticalLearning(
            job_operations=job_operations,
            num_operations=num_operations,
            num_machines=num_machines,
            num_agvs_f=num_agvs_f,
            num_agvs_w=num_agvs_w,
            processing_times=processing_times,
            population_size=100,
            max_iterations=100,
            learning_rate=0.8
        )
        
        # 执行优化
        result = sl.optimize()
        
        # 计算总用时
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time: {total_time:.2f}s")

        # 获取最优解信息
        best_solution = result['best_solution']
        best_makespan = result['best_makespan']
        best_generation = result['best_generation']
        best_makespans = result['best_makespans']
        
        # 计算最优解的详细调度过程
        calculate = Calculate()
        machine_assignments, agv_w_assignments, agv_f_assignments, schedule_string = best_solution[2:]
        schedule_results = calculate.simulate(
            machine_assignments, 
            agv_w_assignments, 
            agv_f_assignments, 
            schedule_string,
            return_schedule=True
        )
        
        # 定义输出文件路径
        output_dir = "sl_output"
        os.makedirs(output_dir, exist_ok=True)
        
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
            best_generation
        )
        
        # 绘制迭代曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(best_makespans) + 1), best_makespans, marker='o', linestyle='-', color='b')
        plt.title('Statistical Learning Algorithm Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Makespan')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'sl_convergence.png'))
        plt.close()
        
        # 输出最优解信息
        logger.info("=" * 50)
        logger.info(f"Best solution found with makespan: {best_makespan:.2f}s at generation {best_generation}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Convergence curve saved to: {os.path.join(output_dir, 'sl_convergence.png')}")
        # 检查日志处理器是否有baseFilename属性
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                logger.info(f"Detailed log saved to: {handler.baseFilename}")
        
        # 输出存储位置信息
        print("\n" + "="*50)
        print(f"调度结果保存至: {output_dir}")
        print(f"迭代曲线保存至: {os.path.join(output_dir, 'sl_convergence.png')}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"优化过程中出错: {str(e)}")
