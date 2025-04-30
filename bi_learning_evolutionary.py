import logging
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from calculate import Calculate  
from statistical_learning import StatisticalLearning
from evolutionary_learning import EvolutionaryLearning
from logger_config import get_logger

# Get logger
logger = get_logger('iblea')

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

class BiLearningEvolutionaryConfig:
    """Configuration for Two-Phase Hybrid Learning Algorithm"""
    
    def __init__(self, 
                population_size=100,
                max_iterations=100,
                delta=0.05,  # 控制SL到EL的切换参数
                pure_sl_min_iterations=20):  
        
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.delta = delta
        self.pure_sl_min_iterations = pure_sl_min_iterations

class BiLearningEvolutionary:
    """Two-Phase Hybrid Learning Algorithm
    
    先进行SL阶段迭代，当满足切换条件时，切换到纯EL阶段直至结束
    """
    
    def __init__(self, config=None):
        """初始化两阶段混合学习算法
        
        Args:
            config: 算法配置，如果为None则使用默认配置
        """
        # 初始化配置
        self.config = config if config else BiLearningEvolutionaryConfig()
        
        # 初始化SL算法
        self.sl = StatisticalLearning(
            job_operations=job_operations,
            num_operations=num_operations,
            num_machines=num_machines,
            num_agvs_w=num_agvs_w,
            num_agvs_f=num_agvs_f,
            processing_times=processing_times,
            population_size=self.config.population_size,
            max_iterations=self.config.max_iterations  # 使用完整的最大迭代次数
        )
        
        # 初始化SL的状态变量
        self.sl_iteration = 0  # SL的迭代计数器
        self.sl_population = None  # SL的当前种群
        self.sl_probability_initialized = False  # 概率模型是否已初始化
        
        # EL算法初始为None，需要时创建
        self.el = None
        
        # 初始化计算器
        self.calculator = Calculate()
        
        # 算法状态变量
        self.current_phase = "statistical learning"  # 当前阶段：SL或EL
        self.best_solution = None  # 当前最佳解
        self.best_makespan = float('inf')  # 当前最佳完工时间
        self.pointer = 0  # 计数变量，用于判断是否切换到EL阶段
        
        # 性能记录
        self.makespan_history = []  # 记录每次迭代的最佳完工时间
        self.time_history = []  # 记录每次迭代的累计时间
    
    def optimize(self, max_iterations=None):
        """运行两阶段混合学习算法
        
        Args:
            max_iterations: 最大迭代次数，如果为None则使用配置中的值
            
        Returns:
            tuple: (最佳解, 最佳完工时间, 性能历史)
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        start_time = time.time()
        logger.debug(f"开始两阶段混合学习算法，最大迭代次数: {max_iterations}")
        
        # 初始化SL阶段
        self.current_phase = "statistical learning"
        self.pointer = 0
        
        # 迭代主循环
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            
            # 根据当前阶段执行相应的算法
            if self.current_phase == "statistical learning":
                self._run_sl_iteration(iteration, max_iterations)
            else:  # evolutionary learning
                best_el_solution, best_el_makespan = self._run_el_iteration(iteration)
                # 更新全局最佳解
                if best_el_makespan < self.best_makespan:
                    self.best_makespan = best_el_makespan
                    self.best_solution = best_el_solution
                    logger.info(f"In generation {iteration + 1}, EL found new best solution with makespan: {self.best_makespan:.2f}")
            
            # 记录本次迭代的性能
            self.makespan_history.append(self.best_makespan)
            self.time_history.append(time.time() - start_time)
            
            # 输出当前迭代信息
            iter_time = time.time() - iter_start_time
            total_time = time.time() - start_time
            logger.debug(f"迭代 {iteration+1}/{max_iterations} | 阶段: {self.current_phase} | "
                    f"最佳完工时间: {self.best_makespan:.2f} | 迭代耗时: {iter_time:.2f}s | ")
        
        # 算法结束
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Best solution found with makespan: {self.best_makespan:.2f}s at generation: {iteration+1}")
        logger.info(f"Total time: {total_time:.2f}s")
        
        return self.best_solution, self.best_makespan, {
            'makespan_history': self.makespan_history,
            'time_history': self.time_history
        }
    
    def _run_sl_iteration(self, iteration, max_iterations):
        """运行一次SL迭代
        
        Args:
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数
        """
        # 手动控制SL的迭代过程，而不是使用sl.optimize()
        # 第一次迭代时初始化种群，之后根据概率模型生成新种群
        if self.sl_iteration == 0:
            # 初始化种群
            population = self.sl.initialize_population()
            self.sl_population = population
            logger.debug(f"迭代 {iteration + 1}：SL初始化种群，大小: {len(population)}")
        else:
            # 根据当前概率模型抽样生成子代
            population = self.sl.generate_population()
            self.sl_population = population
            logger.debug(f"迭代 {iteration + 1}：SL生成新种群，大小: {len(population)}")
        
        # 评估种群
        makespans = []
        solutions = []
        
        for individual in population:
            machine_prob, agv_w_prob, agv_f_prob, schedule_prob = individual
            
            # 生成机器分配、AGV_W分配、AGV_F分配和调度字符串
            machine_assignments, agv_w_assignments, agv_f_assignments = self.sl.convert_to_assignments_string(
                                                                            machine_prob, agv_w_prob, agv_f_prob)
            schedule_string = self.sl.convert_to_schedule_string(schedule_prob)

            # 计算完工时间
            makespan = self.sl.calculate.simulate(
                machine_assignments, 
                agv_w_assignments, 
                agv_f_assignments, 
                schedule_string
            )
            
            if makespan is not None:
                makespans.append(makespan)
                solution_tuple = (
                    len(makespans), 
                    makespan, 
                    machine_assignments, 
                    agv_w_assignments, 
                    agv_f_assignments, 
                    schedule_string
                )
                solutions.append(solution_tuple)
        
        # 如果没有有效解，跳过当前迭代
        if not makespans:
            logger.warning(f"迭代 {iteration + 1}：SL没有有效解，跳过当前迭代")
            self.pointer += 1
            return
        
        # 第一次迭代时初始化概率模型，之后更新概率模型
        if not self.sl_probability_initialized:
            # 初始化概率模型
            self.sl.initialize_probability_model(population, makespans)
            self.sl_probability_initialized = True
            logger.info(f"Initialization probability model completed.")
        else:
            # 更新概率模型
            self.sl.update_probability_model(population, makespans)
            logger.debug(f"第{iteration + 1}代概率模型更新完成")
        
        # 更新最优解
        current_min_makespan = min(makespans)
        current_min_index = makespans.index(current_min_makespan)
        current_best_solution = solutions[current_min_index]
        
        # 获取SL的最佳解
        best_sl_solution = current_best_solution
        best_sl_makespan = current_min_makespan
        
        # 更新全局最佳解
        if best_sl_makespan < self.best_makespan:
            self.best_makespan = best_sl_makespan
            self.best_solution = best_sl_solution
            # 找到新的最优解，重置pointer
            self.pointer = 0
            logger.info(f"In generation {iteration + 1}：SL found new best solution with makespan: {self.best_makespan:.2f}，pointer=0")
        else:
            # 没有找到更好的解，pointer加1
            self.pointer += 1
            logger.info(f"In generation {iteration + 1}：SL not found better solution，pointer={self.pointer}")
        
        # 增加SL迭代计数
        self.sl_iteration += 1
        
        # 检查是否需要切换到EL阶段
        # 条件：pointer > δ*(max_iterations - iteration) / population_size
        remaining_iterations = max_iterations - iteration
        threshold = self.config.delta * remaining_iterations / self.sl.population_size
        
        if self.pointer > threshold and iteration >= self.config.pure_sl_min_iterations:
            logger.info(f"SL is stagnating at generation {iteration + 1}, switching to EL phase...")
            logger.info(f"pointer={self.pointer}, threshold={threshold:.2f}")
            
            # 切换到EL阶段，传递SL的整个种群
            self._switch_to_el(solutions)
    
    def _run_el_iteration(self, iteration):
        """运行一次EL迭代
        
        Args:
            iteration: 当前迭代次数
        """
        # 运行一次EL迭代
        el_solution, best_makespans = self.el.evolve()
        
        # 如果EL没有返回解，则跳过
        if not el_solution:
            logger.warning(f"迭代 {iteration + 1}：EL没有返回有效解")
            return None, float('inf')
        
        # 评估EL解的质量
        # 获取EL的最佳解
        best_el_solution = el_solution
        best_el_makespan = float(best_el_solution[1]) if isinstance(best_el_solution[1], str) else best_el_solution[1]
        return best_el_solution, best_el_makespan
    
    def _switch_to_el(self, sl_solutions):
        """从SL阶段切换到EL阶段"""
        self.current_phase = "evolutionary learning"
        
        # 初始化EL算法
        self.el = EvolutionaryLearning(
            num_jobs=num_jobs,
            num_agvs_w=num_agvs_w,
            num_agvs_f=num_agvs_f,
            job_operations=job_operations,
            processing_times=processing_times,
            population_size=self.config.population_size,
            max_iterations=1,
            pc=0.8,
            pm=0.15
        )
        
        # 使用SL的种群初始化EL的种群
        if sl_solutions and isinstance(sl_solutions, list) and len(sl_solutions) > 0:
            # 初始化EL的种群
            self.el.population = []
            
            # 将SL的种群转换为EL的种群格式并添加到EL的种群中
            for i, solution in enumerate(sl_solutions):
                self.el.population.append(solution)
            
            # 更新EL的最佳解
            for solution in sl_solutions:
                if self.el.best_solution is None or solution[1] < self.el.best_solution[1]:
                    self.el.best_solution = solution    
        
            # 如果SL种群数量小于EL所需种群大小，则随机生成其余部分
            if len(self.el.population) < self.el.population_size:
                remaining = self.el.population_size - len(self.el.population)
                logger.debug(f"SL种群大小({len(self.el.population)})小于EL所需种群大小({self.el.population_size})，随机生成剩余{remaining}个个体")
                
                # 记录当前种群大小
                current_size = len(self.el.population)
                
                # 生成剩余个体
                attempts = 0
                max_attempts = remaining * 10
                
                while len(self.el.population) < self.el.population_size and attempts < max_attempts:
                    solution = self.el.generate_random_strings(
                        self.el.num_jobs,
                        self.el.num_agvs_w,
                        self.el.num_agvs_f,
                        self.el.job_operations,
                        self.el.processing_times
                    )
                    calculate = Calculate()
                    makespan = calculate.simulate(*solution)
                    
                    if makespan is not None:
                        solution_tuple = (len(self.el.population) + 1, makespan, *solution)
                        self.el.population.append(solution_tuple)
                        
                        if self.el.best_solution is None or makespan < self.el.best_solution[1]:
                            self.el.best_solution = solution_tuple
                        
                    attempts += 1
            
            logger.debug(f"切换到EL阶段，使用SL种群初始化EL种群，种群大小: {len(self.el.population)}")
        else:
            # 如果没有SL种群，则随机初始化
            self.el.initialize_population()
            logger.debug(f"切换到EL阶段，使用随机初始化EL种群，种群大小: {len(self.el.population)}")

# 测试代码
if __name__ == "__main__":
    # 创建算法实例
    config = BiLearningEvolutionaryConfig(
        population_size=100,
        max_iterations=100,
        delta=5,
        pure_sl_min_iterations=0
    )
    
    iblea = BiLearningEvolutionary(config)
    
    # 运行算法
    best_solution, best_makespan, history = iblea.optimize()
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['makespan_history'])
    plt.title('Bi-Learning Evolutionary Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.savefig('blea_convergence.png')
    
    logger.info("Convergence curve saved to: blea_convergence.png")
