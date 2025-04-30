"""
Hybrid Learning Algorithm
Combines Statistical Learning (SL) and Evolutionary Learning (EL) for hybrid optimization
"""

import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from calculate import Calculate  
from logger_config import setup_file_logger

# Import existing algorithms
from statistical_learning import StatisticalLearning
from evolutionary_learning import EvolutionaryLearning, save_schedule_results

# Import instances
from MK10 import (
    NUM_JOBS as num_jobs,
    NUM_OPERATIONS as num_operations,
    NUM_MACHINES as num_machines,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    JOB_OPERATIONS as job_operations,
    PROCESSING_TIMES as processing_times,
    PRIORITY_DICT as priority_dict
)

# For type hinting
ResultType = dict

# 全局日志记录器
logger = setup_file_logger('hl')

class HybridLearningConfig:
    """Hybrid Learning Algorithm Configuration Parameters"""
    def __init__(self,
                population_size=100,
                total_max_iterations=100,
                pure_sl_min_iterations=20,        # Minimum iterations for pure SL
                stagnation_window_size=10,        # Stagnation detection window size
                stagnation_threshold=0.01,        # Stagnation threshold (improvement rate below this value is considered stagnation)
                force_switch_iterations=50,       # Force switch iterations
                initial_sl_ratio=0.9,             # Initial SL ratio
                initial_el_ratio=0.1,             # Initial EL ratio
                ratio_update_step=0.1,            # Ratio update step
                ratio_update_inertia=3,           # Ratio update inertia (number of consecutive generations to update)
                min_ratio=0.1,                    # Minimum ratio
                max_ratio=0.9                     # Maximum ratio
                ):
        self.population_size = population_size
        self.total_max_iterations = total_max_iterations
        self.pure_sl_min_iterations = pure_sl_min_iterations
        self.stagnation_window_size = stagnation_window_size
        self.stagnation_threshold = stagnation_threshold
        self.force_switch_iterations = force_switch_iterations
        self.initial_sl_ratio = initial_sl_ratio
        self.initial_el_ratio = initial_el_ratio
        self.ratio_update_step = ratio_update_step
        self.ratio_update_inertia = ratio_update_inertia
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio


class HybridLearning:
    """Hybrid Learning Algorithm"""
    
    def __init__(self, config=None):
        # Initialize configuration
        self.config = config if config else HybridLearningConfig()
        
        # Initialize SL algorithm
        self.sl = StatisticalLearning(
            job_operations=job_operations,
            num_operations=num_operations,
            num_machines=num_machines,
            num_agvs_w=num_agvs_w,
            num_agvs_f=num_agvs_f,
            processing_times=processing_times,
            population_size=self.config.population_size,
            max_iterations=1
        )
        
        # EL algorithm is initially None, created when needed
        self.el = None
        
        # Initialize other attributes
        self.best_solution = None
        self.best_makespan = float('inf')
        self.best_generation = 0
        self.best_makespans = []
        
        # Performance metrics recording
        self.performance_history = {
            'sl_improvements': [],      # SL improvement record
            'el_improvements': [],      # EL improvement record
            'sl_best_makespan': float('inf'),
            'el_best_makespan': float('inf'),
            'phase_history': [],        # Algorithm phase record
            'sl_ratio_history': [],     # SL ratio history
            'el_ratio_history': [],     # EL ratio history
            'sl_dominance_count': 0,    # SL consecutive dominance count
            'el_dominance_count': 0     # EL consecutive dominance count
        }
        
        # Set up logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger"""
        global logger
        if not logger:
            # 使用 logger_config 中的 setup_file_logger 函数配置日志
            from logger_config import setup_file_logger
            logger = setup_file_logger("HybridLearning")
    
    def _check_stagnation(self):
        """Check if the algorithm is stagnating
        
        Returns:
            bool: True if the algorithm is stagnating, False otherwise
        """
        if len(self.best_makespans) < self.config.stagnation_window_size:
            return False
            
        # Calculate average improvement rate within the window
        improvements = []
        for i in range(1, self.config.stagnation_window_size):
            imp = self.best_makespans[-i-1] - self.best_makespans[-i]
            improvements.append(imp if imp > 0 else 0)
        
        avg_improvement = sum(improvements) / len(improvements)
        return avg_improvement < self.config.stagnation_threshold
    
    def _init_sl_population(self, sl_count, iteration):
        """Initialize or generate SL population
        
        Args:
            sl_count: SL population size
            iteration: Current iteration
            
        Returns:
            list: SL population
        """
        if iteration == 0:
            # Initialize population
            return self.sl.initialize_population()[:sl_count]
        else:
            # Generate offspring based on current probability model
            return self.sl.generate_population()[:sl_count]
    
    def _evaluate_sl_individuals(self, sl_population):
        """Evaluate SL individuals
        
        Args:
            sl_population: SL population
            
        Returns:
            tuple: (solution list, makespan list)
        """
        sl_solutions = []
        sl_makespans = []
        
        for individual in sl_population:
            machine_prob, agv_w_prob, agv_f_prob, schedule_prob = individual
            
            # Generate machine assignments, AGV_W assignments, AGV_F assignments, and schedule string
            machine_assignments, agv_w_assignments, agv_f_assignments = self.sl.convert_to_assignments_string(
                                                                            machine_prob, agv_w_prob, agv_f_prob)
            schedule_string = self.sl.convert_to_schedule_string(schedule_prob)

            # Calculate makespan
            makespan = self.sl.calculate.simulate(
                machine_assignments, 
                agv_w_assignments, 
                agv_f_assignments, 
                schedule_string
            )
            
            if makespan is not None:
                sl_makespans.append(makespan)
                solution_dict = ({
                    'machine_assignments': machine_assignments,
                    'agv_w_assignments': agv_w_assignments,
                    'agv_f_assignments': agv_f_assignments,
                    'schedule_string': schedule_string,
                    'makespan': makespan,
                    'source': 'sl'  # Mark the solution source
                })
                sl_solutions.append(solution_dict)
        
        return sl_solutions, sl_makespans
    
    def _update_sl_model(self, sl_population, sl_makespans, iteration):
        """Update SL probability model
        
        Args:
            sl_population: SL population
            sl_makespans: Makespan list
            iteration: Current iteration
        """
        if iteration == 0:
            # Initialize probability model
            self.sl.initialize_probability_model(sl_population, sl_makespans)
            logger.debug("Initialized probability model")
        else:
            # Update probability model
            self.sl.update_probability_model(sl_population, sl_makespans)
            logger.debug(f"Updated probability model at iteration {iteration + 1}")
    
    def _update_best_solution(self, solution, iteration, source):
        """Update best solution
        
        Args:
            solution: Solution
            iteration: Current iteration
            source: Solution source ('sl' or 'el')
            
        Returns:
            bool: True if a new best solution is found, False otherwise
        """
        # Check if solution is a dictionary or a tuple
        if isinstance(solution, dict):
            makespan = solution['makespan']
        else:
            # If the solution is from EL (tuple format)
            makespan = solution[1]
        
        # Update best solution if a new best solution is found
        if self.best_solution is None or makespan < self.best_makespan:
            if isinstance(solution, dict):
                self.best_solution = solution
            else:
                # If the solution is from EL (tuple format), convert to dictionary format
                self.best_solution = {
                    'machine_assignments': solution[2],
                    'agv_w_assignments': solution[3],
                    'agv_f_assignments': solution[4],
                    'schedule_string': solution[5],
                    'makespan': solution[1],
                    'source': 'el'
                }
            
            # 获取完整的调度信息
            if 'schedule' not in self.best_solution:
                # 使用Calculate类模拟执行调度方案，获取调度结果
                calculate = Calculate()
                schedule_info = calculate.simulate(
                    machine_string=self.best_solution['machine_assignments'],
                    agv_w_string=self.best_solution['agv_w_assignments'],
                    agv_f_string=self.best_solution['agv_f_assignments'],
                    schedule_string=self.best_solution['schedule_string'],
                    return_schedule=True
                )[1]  # 返回值的第二项是调度结果
                
                # 将调度结果转换为按设备分组的字典格式
                schedule_dict = {}
                for item in schedule_info:
                    device = item['device']
                    if device not in schedule_dict:
                        schedule_dict[device] = []
                    schedule_dict[device].append(item)
                
                # 保存到最佳解中
                self.best_solution['schedule'] = schedule_dict
            
            self.best_generation = iteration
            self.best_makespan = makespan
            logger.info(f"In generation {iteration + 1}，New best makespan: {makespan:.2f}, source: {source}")
            return True
        
        return False
    
    def _init_or_update_el(self, el_count, sl_solutions, iteration):
        """Initialize or update EL algorithm
        
        Args:
            el_count: EL population size
            sl_solutions: SL solution list
            iteration: Current iteration
            
        Returns:
            tuple: EL best solution and makespan list
        """
        # If EL algorithm is not initialized, initialize it
        if self.el is None:
            self.el = EvolutionaryLearning(
                num_jobs=num_jobs,
                num_agvs_w=num_agvs_w,
                num_agvs_f=num_agvs_f,
                job_operations=job_operations,
                processing_times=processing_times,
                population_size=el_count,  # Use dynamic EL population size
                max_iterations=1,  # Perform one generation of evolution at a time
                pc=0.8,
                pm=0.15
            )
            
            # Initialize EL population
            # If there are SL solutions, use the best SL solution to initialize part of the EL population
            if sl_solutions:
                # Sort SL solutions by makespan
                sorted_sl_solutions = sorted(sl_solutions, key=lambda x: x['makespan'])
                
                # Select the top min(el_count, len(sorted_sl_solutions)) SL solutions and convert them to EL individuals
                for i in range(min(el_count, len(sorted_sl_solutions))):
                    solution = sorted_sl_solutions[i]
                    self.el.population.append((
                        i,
                        solution['makespan'],
                        solution['machine_assignments'],
                        solution['agv_w_assignments'],
                        solution['agv_f_assignments'],
                        solution['schedule_string']
                    ))
                    
            # If the EL population size is not enough, generate the remaining individuals randomly
            if len(self.el.population) < el_count:
                self.el.initialize_population()
                # Ensure the population size is correct
                self.el.population = self.el.population[:el_count]
        
        # Ensure EL's best_solution is initialized
        if self.el.best_solution is None and self.el.population:
            # Find the best individual in the current population as the initial best_solution
            best_individual = min(self.el.population, key=lambda x: x[1] if x[1] is not None else float('inf'))
            self.el.best_solution = best_individual
            self.el.best_generation = iteration
        
        # Update EL population size
        self.el.population_size = el_count
        # Perform one generation of evolution
        self.el.max_iterations = 1
        best_solution, makespans = self.el.evolve(start_generation=iteration)
        
        # 返回最优解和整个种群
        return best_solution, makespans, self.el.population

    def _knowledge_exchange(self, el_solutions, el_ratio, sl_solutions, sl_ratio, iteration, sl_population, sl_makespans):
        """Knowledge exchange: exchange knowledge between SL and EL
        
        Args:
            el_solutions: EL solution list
            el_ratio: EL ratio
            sl_solutions: SL solution list
            sl_ratio: SL ratio
            iteration: Current iteration
            sl_population: Current SL population
            sl_makespans: Current SL makespan list
            
        Returns:
            tuple: (updated SL population, updated SL makespan list)
        """
        # 初始化知识交换状态记录
        sl_to_el_improvement = 0  # SL到EL改进程度
        el_to_sl_count = 0  # EL到SL交换的解数量
        
        # Only perform knowledge exchange in the hybrid phase
        if sl_ratio > 0 and el_ratio > 0 and self.el is not None:
            # 1. Transfer knowledge from SL to EL
            if sl_solutions and self.el is not None and len(self.el.population) > 0:
                # Find the worst individual in EL
                worst_el_individual = max(self.el.population, key=lambda x: x[1] if x[1] is not None else float('inf'))
                
                # Find the best solution in SL
                best_sl_solution = min(sl_solutions, key=lambda x: x['makespan'] if x['makespan'] is not None else float('inf'))
                
                # If SL's best solution is better than EL's worst solution, replace it
                if best_sl_solution['makespan'] < worst_el_individual[1]:
                    # Convert SL solution to EL format
                    new_el_individual = (
                        worst_el_individual[0],  # worst_el_individual's ID
                        best_sl_solution['makespan'],         # Makespan
                        best_sl_solution['machine_assignments'],         # Machine assignments
                        best_sl_solution['agv_w_assignments'],         # AGV_W assignments
                        best_sl_solution['agv_f_assignments'],         # AGV_F assignments
                        best_sl_solution['schedule_string']          # Schedule string
                    )
                    
                    # Replace EL's worst individual
                    self.el.population[self.el.population.index(worst_el_individual)] = new_el_individual
                    sl_to_el_success = True
                    sl_to_el_improvement = worst_el_individual[1] - best_sl_solution['makespan']
                    logger.debug(f"  [SL->EL] 成功! 用SL最优解{best_sl_solution['makespan']:.2f}替换EL最差解{worst_el_individual[1]:.2f}, 改进了{sl_to_el_improvement:.2f}")
                else:
                    logger.debug(f"  [SL->EL] 失败! SL最优解{best_sl_solution['makespan']:.2f}不如EL最差解{worst_el_individual[1]:.2f}")
            else:
                logger.info(f"  [SL->EL] 失败! 缺少SL解或EL种群")
            
            # 2. Transfer knowledge from EL to SL: add EL's top solutions to SL's probability model
            if el_solutions and len(el_solutions) > 0:
                # 增加交换比例，从10%提高到20%
                top_el_num = max(2, int(len(el_solutions) * 0.2))  # 确保至少有2个解
                top_el_solutions = sorted(el_solutions, key=lambda x: x[1])[:top_el_num]  # 取前top_el_num个最优解
                
                for top_el_solution in top_el_solutions:
                    try:
                        # Convert EL solution to SL format (probability matrix)
                        machine_prob = self.convert_assignments_to_probability(top_el_solution[2], assignment_type='machine')
                        agv_w_prob = self.convert_assignments_to_probability(top_el_solution[3], assignment_type='agv_w')
                        agv_f_prob = self.convert_assignments_to_probability(top_el_solution[4], assignment_type='agv_f')
                        schedule_prob = self.convert_schedule_to_probability(top_el_solution[5])
                        
                        # Create a new SL individual
                        new_sl_individual = (machine_prob, agv_w_prob, agv_f_prob, schedule_prob)
                        # Add the new individual to the population
                        sl_population.append(new_sl_individual)
                        sl_makespans.append(top_el_solution[1])
                        el_to_sl_success = True
                        el_to_sl_count += 1
                    except Exception as e:
                        logger.error(f"    - 添加EL解到SL种群失败: {str(e)}")
                
                if el_to_sl_count > 0:
                    logger.debug(f"  [EL->SL] 成功添加{el_to_sl_count}个EL解到SL种群")
                else:
                    logger.debug(f"  [EL->SL] 失败! 没有成功添加EL解到SL种群")
            else:
                logger.info(f"  [EL->SL] 失败! 缺少EL解")
        else:
            # 非混合阶段不进行知识交换
            if iteration % 10 == 0:  # 每10次迭代记录一次
                logger.info(f"[知识交换] 第{iteration+1}次迭代不进行知识交换 (非混合阶段, SL比例={sl_ratio:.2f}, EL比例={el_ratio:.2f})")
        
        return sl_population, sl_makespans
    
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
        
    def _update_algorithm_ratio(self, performance_history, iteration, sl_ratio, el_ratio, algorithm_phase):
        """Dynamically update SL and EL ratios
        
        Args:
            performance_history: Performance history record
            iteration: Current iteration
            sl_ratio: Current SL ratio
            el_ratio: Current EL ratio
            algorithm_phase: Current algorithm phase
            
        Returns:
            tuple: (new SL ratio, new EL ratio, new algorithm phase)
        """
        # If still in the pure SL phase, check if it's time to switch to the hybrid phase
        if algorithm_phase == "statistical learning":
            # Check if the force switch condition is met
            if iteration >= self.config.force_switch_iterations:
                logger.info(f"\nReached force switch iteration {self.config.force_switch_iterations}, switching to hybrid phase...")
                return self.config.initial_sl_ratio, self.config.initial_el_ratio, "hybrid learning"
            
            # Check if the minimum SL iteration condition is met
            if iteration >= self.config.pure_sl_min_iterations:
                # Check if the algorithm is stagnating
                if self._check_stagnation():
                    logger.info(f"\nSL is stagnating, switching to hybrid phase...")
                    return self.config.initial_sl_ratio, self.config.initial_el_ratio, "hybrid learning"
            
            return sl_ratio, el_ratio, algorithm_phase
        
        # In the hybrid phase, dynamically adjust SL and EL ratios
        # Don't adjust ratios for the first 10 iterations
        if len(performance_history['el_improvements']) < 10:
            return sl_ratio, el_ratio, algorithm_phase
        
        # Calculate average improvement rates for SL and EL over the recent window
        recent_sl_improvements = performance_history['sl_improvements'][-self.config.stagnation_window_size:]
        sl_avg_improvement = sum(recent_sl_improvements) / len(recent_sl_improvements)
        
        recent_el_improvements = performance_history['el_improvements'][-self.config.stagnation_window_size:]
        el_avg_improvement = sum(recent_el_improvements) / len(recent_el_improvements)
        
        # Update consecutive dominance counts
        if sl_avg_improvement > el_avg_improvement:
            performance_history['sl_dominance_count'] += 1
            performance_history['el_dominance_count'] = 0
        elif el_avg_improvement > sl_avg_improvement:
            performance_history['el_dominance_count'] += 1
            performance_history['sl_dominance_count'] = 0
        else:
            # Performance is similar, reset counts
            performance_history['sl_dominance_count'] = 0
            performance_history['el_dominance_count'] = 0
        
        # Apply inertia mechanism: only adjust ratios if one algorithm has been consistently better for multiple generations
        new_sl_ratio = sl_ratio
        new_el_ratio = el_ratio
        
        if performance_history['sl_dominance_count'] >= self.config.ratio_update_inertia:
            # SL has been consistently better, increase SL ratio
            new_el_ratio = max(self.config.min_ratio, el_ratio - self.config.ratio_update_step)
            new_sl_ratio = 1.0 - new_el_ratio
            logger.info(f"SL has been consistently better for {performance_history['sl_dominance_count']} iterations, adjusting ratios: SL={new_sl_ratio:.2f}, EL={new_el_ratio:.2f}")
            # Reset count
            performance_history['sl_dominance_count'] = 0
            
        elif performance_history['el_dominance_count'] >= self.config.ratio_update_inertia:
            # EL has been consistently better, increase EL ratio
            new_sl_ratio = max(self.config.min_ratio, sl_ratio - self.config.ratio_update_step)
            new_el_ratio = 1.0 - new_sl_ratio
            logger.info(f"EL has been consistently better for {performance_history['el_dominance_count']} iterations, adjusting ratios: SL={new_sl_ratio:.2f}, EL={new_el_ratio:.2f}")
            # Reset count
            performance_history['el_dominance_count'] = 0
        
        return new_sl_ratio, new_el_ratio, algorithm_phase
    
    def _record_performance(self, sl_ratio, el_ratio, algorithm_phase):
        """Record performance metrics
        
        Args:
            sl_ratio: Current SL ratio
            el_ratio: Current EL ratio
            algorithm_phase: Current algorithm phase
        """
        self.performance_history['phase_history'].append(algorithm_phase)
        self.performance_history['sl_ratio_history'].append(sl_ratio)
        self.performance_history['el_ratio_history'].append(el_ratio)
    
    def optimize(self):
        """Perform hybrid optimization
        
        Returns:
            dict: Dictionary containing the best solution information
        """
        # Initialize
        iteration = 0
        sl_ratio = 1.0  # Initial pure SL phase
        el_ratio = 0.0
        algorithm_phase = "statistical learning"  # Current algorithm phase
        self.best_makespans = []
        
        # Initialize performance history
        self.performance_history = {
            'sl_improvements': [],      # SL improvement record
            'el_improvements': [],      # EL improvement record
            'sl_best_makespan': float('inf'),
            'el_best_makespan': float('inf'),
            'phase_history': [],
            'sl_ratio_history': [],
            'el_ratio_history': [],
            'sl_dominance_count': 0,
            'el_dominance_count': 0
        }
        
        logger.debug(f"Starting optimization, total iterations: {self.config.total_max_iterations}")
        logger.debug(f"Minimum iterations for pure SL: {self.config.pure_sl_min_iterations}")
        logger.debug(f"Stagnation detection window size: {self.config.stagnation_window_size}")
        logger.debug(f"Stagnation threshold: {self.config.stagnation_threshold}")
        logger.debug(f"Force switch iterations: {self.config.force_switch_iterations}")
        
        # Start iteration
        while iteration < self.config.total_max_iterations:
            # Record the best solution at the current iteration
            current_best_makespan = self.best_makespan if self.best_makespan != float('inf') else None
            
            # Calculate SL and EL population sizes
            sl_count = max(1, int(self.config.population_size * sl_ratio)) if sl_ratio > 0 else 0
            el_count = max(1, self.config.population_size - sl_count) if el_ratio > 0 else 0
            
            logger.debug(f"Iteration {iteration+1}/{self.config.total_max_iterations}: Phase={algorithm_phase}, SL ratio={sl_ratio:.2f}, EL ratio={el_ratio:.2f}")
            
            # ===== SL part =====
            sl_solutions = []
            sl_makespans = []
            
            if sl_count > 0:
                # Generate and evaluate SL individuals
                sl_population = self._init_sl_population(sl_count, iteration)
                
                # Evaluate each individual in the SL population
                sl_solutions, sl_makespans = self._evaluate_sl_individuals(sl_population)
                
                # 只在纯SL阶段更新概率模型，混合阶段在知识交换后更新
                if sl_makespans and el_ratio == 0:  # 只在纯SL阶段更新
                    self._update_sl_model(sl_population, sl_makespans, iteration)
                
                # Update best solution
                if sl_solutions:
                    sl_current_min_makespan = min(sl_makespans)
                    sl_current_min_index = sl_makespans.index(sl_current_min_makespan)
                    sl_current_best_solution = sl_solutions[sl_current_min_index]
                    self._update_best_solution(sl_current_best_solution, iteration, 'sl')
                    
                    # Record SL improvement
                    if current_best_makespan:
                        sl_improvement = current_best_makespan - sl_current_min_makespan if sl_current_min_makespan < current_best_makespan else 0
                    else:
                        sl_improvement = 0  # No improvement in the first iteration
                    self.performance_history['sl_improvements'].append(sl_improvement)
                    
                    # Update SL's best performance
                    self.performance_history['sl_best_makespan'] = min(
                        self.performance_history['sl_best_makespan'], 
                        sl_current_min_makespan
                    )
                    # 如果是纯SL阶段，为el_improvements添加0值
                    if el_ratio == 0:
                        self.performance_history['el_improvements'].append(0)
            
            # ===== EL part =====
            el_solutions = []
            el_makespans = []
            
            if el_count > 0:
                # Initialize or update EL algorithm
                el_best_solution, el_makespans, el_population = self._init_or_update_el(el_count, sl_solutions, iteration)
                
                if el_best_solution:
                    # 使用整个EL种群进行知识交换
                    el_solutions = el_population
                    
                    # Update best solution
                    solution_improved = self._update_best_solution(el_best_solution, iteration, 'el')
                    
                    # Record EL improvement
                    if current_best_makespan and solution_improved:
                        el_improvement = current_best_makespan - el_best_solution[1]
                    else:
                        el_improvement = 0
                    self.performance_history['el_improvements'].append(el_improvement)
                    
                    # Update EL's best performance
                    self.performance_history['el_best_makespan'] = min(
                        self.performance_history['el_best_makespan'], 
                        el_best_solution[1]
                    )
            
            # Record the best solution at the current iteration
            self.best_makespans.append(self.best_makespan)
            
            # Record performance metrics
            self._record_performance(sl_ratio, el_ratio, algorithm_phase)

            # ===== Knowledge exchange part =====
            if el_ratio > 0:  # 只在混合阶段进行知识交换和概率模型更新
                sl_population, sl_makespans = self._knowledge_exchange(el_solutions, el_ratio, sl_solutions, sl_ratio, iteration, sl_population, sl_makespans)
                # 只在知识交换后更新概率模型
                self._update_sl_model(sl_population, sl_makespans, iteration)
                
            # Dynamically adjust SL and EL ratios
            sl_ratio, el_ratio, algorithm_phase = self._update_algorithm_ratio(
                self.performance_history, iteration, sl_ratio, el_ratio, algorithm_phase
            )
            
            iteration += 1
        
        # Save results and visualize performance
        self._save_results()
        self._visualize_performance()
        
        logger.info(f"Best solution found with makespan: {self.best_makespan:.2f}s at generation {self.best_generation + 1}")
        
        return {
            'best_solution': self.best_solution,
            'best_makespan': self.best_makespan,
            'best_generation': self.best_generation,
            'best_makespans': self.best_makespans,
            'performance_history': self.performance_history
        }
    
    def _save_results(self):
        """Save the results of the hybrid learning algorithm"""
        if not self.best_solution:
            logger.warning("No best solution found")
            return
        
        # 创建输出目录
        os.makedirs("hl_output", exist_ok=True)
        
        # 保存最优解到CSV文件
        filename = "hl_output/best_solution.csv"
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["迭代次数", "最优解编号", "完工时间", "机器分配", "AGV_W分配", "AGV_F分配", "调度顺序", "工序处理顺序"])
            
            # 使用Calculate类解码调度字符串，获取工序处理顺序
            order = self.sl.calculate.decode_schedule(
                self.best_solution['schedule_string'],
                priority_dict
            )
            
            writer.writerow([
                self.best_generation,
                "",
                self.best_solution['makespan'],
                self.best_solution['machine_assignments'],
                self.best_solution['agv_w_assignments'],
                self.best_solution['agv_f_assignments'],
                self.best_solution['schedule_string'],
                order
            ])
        
        # 在保存调度结果之前，先使用最优解中的分配字符串模拟调度
        makespan, schedule_results = self.sl.calculate.simulate(
            self.best_solution['machine_assignments'],
            self.best_solution['agv_w_assignments'],
            self.best_solution['agv_f_assignments'],
            self.best_solution['schedule_string'],
            return_schedule=True
        )
        
        # 导入并调用 evolutionary_learning 中的 save_schedule_results 函数
        save_schedule_results(
            self.sl.calculate,
            "hl_output/agv_w_schedule.csv",
            "hl_output/agv_f_schedule.csv",
            "hl_output/machine_schedule.csv"
        )
        logger.info("=" * 50)
        logger.info("Results saved to: hl_output" )
    
    def _visualize_performance(self):
        """Visualize algorithm performance"""
        try:
            # Create plot
            plt.figure(figsize=(15, 5))
            
            # 2. SL and EL ratio changes
            plt.subplot(1, 3, 1)
            plt.plot(self.performance_history['sl_ratio_history'], 'r-', label='SL Ratio')
            plt.plot(self.performance_history['el_ratio_history'], 'g-', label='EL Ratio')
            plt.title('Algorithm Ratio Changes')
            plt.xlabel('Iteration')
            plt.ylabel('Ratio')
            plt.grid(True)
            plt.legend()
            
            # 2. SL and EL improvement
            plt.subplot(1, 3, 2)
            # Calculate cumulative improvements
            sl_cumulative_improvements = np.cumsum(self.performance_history['sl_improvements'])
            el_cumulative_improvements = np.cumsum(self.performance_history['el_improvements'])
            plt.plot(sl_cumulative_improvements, 'r-', label='SL Cumulative Improvement')
            plt.plot(el_cumulative_improvements, 'g-', label='EL Cumulative Improvement')
            plt.title('Algorithm Cumulative Contribution')
            plt.xlabel('Iteration')
            plt.ylabel('Cumulative Improvement')
            plt.grid(True)
            plt.legend()
            
            # 3. Algorithm phase transitions
            plt.subplot(1, 3, 3)
            # Mark phase transition points
            phase_changes = [i for i in range(1, len(self.performance_history['phase_history'])) 
                            if self.performance_history['phase_history'][i] != self.performance_history['phase_history'][i-1]]
            plt.plot(self.best_makespans, 'b-')
            for change in phase_changes:
                plt.axvline(x=change, color='r', linestyle='--')
                plt.text(change, min(self.best_makespans), self.performance_history['phase_history'][change], rotation=90)
            plt.title('Algorithm Phase Transitions')
            plt.xlabel('Iteration')
            plt.ylabel('Makespan')
            plt.grid(True)
            
            # Save the chart
            os.makedirs("hl_output", exist_ok=True)
            plt.tight_layout()
            plt.savefig("hl_output/hl_convergence.png")
            logger.info("Convergence curve saved to: hl_output/hl_convergence.png")

        except Exception as e:
            logger.error(f"Error generating performance chart: {e}")


# Example usage
if __name__ == "__main__":
    
    # Create configuration
    config = HybridLearningConfig(
        population_size=100,
        total_max_iterations=100, 
        pure_sl_min_iterations=20, # 20%*total_max_iterations
        stagnation_window_size=20, # 随迭代次数适当增加
        stagnation_threshold=0.01,
        force_switch_iterations=50, # 50%*total_max_iterations
        initial_sl_ratio=0.9,
        initial_el_ratio=0.1,
        ratio_update_step=0.1,
        ratio_update_inertia=5, # 惯性窗口
        min_ratio=0.1,
        max_ratio=0.9
    )
    
    # 记录开始时间
    start_time = time.time()
    # Create Hybrid Learning algorithm instance
    hl = HybridLearning(config)
    
    # Perform optimization
    result = hl.optimize()
    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f}s")