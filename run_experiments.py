"""
实验运行脚本

该脚本用于运行HL、SL和EL算法各50次，记录每次运行的最优解和求解时间
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime

# 导入算法模块
from statistical_learning import StatisticalLearning
from evolutionary_learning import EvolutionaryLearning
from hybrid_learning import HybridLearning, HybridLearningConfig
from blea import BLEA, BLEAConfig
from bi_learning_evolutionary import BiLearningEvolutionary, BiLearningEvolutionaryConfig

# 导入配置
from MK10 import (
    NUM_JOBS as num_jobs,
    NUM_OPERATIONS as num_operations,
    NUM_MACHINES as num_machines,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    JOB_OPERATIONS as job_operations,
    PROCESSING_TIMES as processing_times
)

# 导入统一的日志配置
from logger_config import setup_file_logger

# 获取日志记录器
logger = setup_file_logger('experiments')

# 创建结果目录
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)

# 结果CSV文件路径
results_file = os.path.join(results_dir, f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# 设置算法参数
MAX_ITERATIONS = 1000  # 每个算法的最大迭代次数
POPULATION_SIZE = 100  # 种群大小
NUM_RUNS = 10  # 每个算法运行的次数
LEARNING_RATE = 0.8  # SL算法学习率
DELTA = 0.5  # 控制BLEA算法中SL到EL的切换参数


def run_sl(run_id):
    """运行统计学习算法
    
    Args:
        run_id: 运行ID
        
    Returns:
        tuple: (最优解, 求解时间, 最优解代数)
    """
    logger.info(f"运行SL算法 - 第{run_id}次")
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化SL算法
    sl = StatisticalLearning(
        job_operations=job_operations,
        num_operations=num_operations,
        num_machines=num_machines,
        num_agvs_f=num_agvs_f,
        num_agvs_w=num_agvs_w,
        processing_times=processing_times,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        learning_rate=LEARNING_RATE
    )
    
    # 执行优化
    result = sl.optimize()
    
    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取最优解信息
    best_makespan = result['best_makespan']
    best_generation = result['best_generation']
    
    logger.info(f"SL算法第{run_id}次运行完成，最优解：{best_makespan:.2f}，最优解代数：{best_generation}，用时：{total_time:.2f}s")
    
    return best_makespan, total_time, best_generation


def run_el(run_id):
    """运行进化学习算法
    
    Args:
        run_id: 运行ID
        
    Returns:
        tuple: (最优解, 求解时间, 最优解代数)
    """
    logger.info(f"运行EL算法 - 第{run_id}次")
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化EL算法
    el = EvolutionaryLearning(
        num_jobs=num_jobs,
        num_agvs_w=num_agvs_w,
        num_agvs_f=num_agvs_f,
        job_operations=job_operations,
        processing_times=processing_times,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        pc=0.8,
        pm=0.15
    )
    
    # 初始化种群
    el.initialize_population()
    
    # 进化
    best_solution, best_makespans = el.evolve()
    
    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取最优解信息
    best_makespan = best_solution[1] if best_solution else float('inf')
    best_generation = el.best_generation  # 直接从el对象获取best_generation
    
    logger.info(f"EL算法第{run_id}次运行完成，最优解：{best_makespan:.2f}，最优解代数：{best_generation}，用时：{total_time:.2f}s")
    
    return best_makespan, total_time, best_generation


def run_hl(run_id):
    """运行混合学习算法
    
    Args:
        run_id: 运行ID
        
    Returns:
        tuple: (最优解, 求解时间, 最优解代数)
    """
    logger.info(f"运行HL算法 - 第{run_id}次")
    # Create configuration
    config = HybridLearningConfig(
        population_size=POPULATION_SIZE,
        total_max_iterations=MAX_ITERATIONS, 
        pure_sl_min_iterations=MAX_ITERATIONS*0.2, # 20%*total_max_iterations
        stagnation_window_size=20, # 随迭代次数适当增加
        stagnation_threshold=0.01,
        force_switch_iterations=MAX_ITERATIONS*0.5, # 50%*total_max_iterations
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
    result = hl.optimize()
    # 计算总用时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取最优解信息
    best_makespan = result['best_makespan']
    best_generation = result['best_generation']
    
    logger.info(f"HL算法第{run_id}次运行完成，最优解：{best_makespan:.2f}，最优解代数：{best_generation}，用时：{total_time:.2f}s")
    
    return best_makespan, total_time, best_generation


def run_blea(run_id):
    """运行 BLEA 算法
    
    Args:
        run_id: 运行ID
        
    Returns:
        tuple: (最优解, 求解时间, 最优解代数)
    """
    logger.info(f"运行 BLEA 算法 - 第{run_id}次")
    
    # 创建算法实例
    config = BLEAConfig(
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        delta=DELTA,
        pure_sl_min_iterations=0
    )
    
    # 记录开始时间
    start_time = time.time()

    blea = BLEA(config)
    
    # 运行算法
    best_solution, best_makespan, history = blea.optimize()
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 找到最优解的代数
    best_iteration = 0
    for i, makespan in enumerate(history['makespan_history']):
        if makespan == best_makespan:
            best_iteration = i + 1
            break
    
    logger.info(f"BLEA 算法第{run_id}次运行完成，最优解：{best_makespan:.2f}，最优解代数：{best_iteration}，用时：{total_time:.2f}s")
    
    return best_makespan, total_time, best_iteration


def run_iblea(run_id):
    """运行 BiLearningEvolutionary 算法
    
    Args:
        run_id: 运行ID
        
    Returns:
        tuple: (最优解, 求解时间, 最优解代数)
    """
    logger.info(f"运行 BLE 算法 - 第{run_id}次")
    
    # 创建算法实例
    config = BiLearningEvolutionaryConfig(
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        delta=DELTA,
        pure_sl_min_iterations=0
    )
    
    # 记录开始时间
    start_time = time.time()

    iblea = BiLearningEvolutionary(config)
    
    # 运行算法
    best_solution, best_makespan, history = iblea.optimize()
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 找到最优解的代数
    best_iteration = 0
    for i, makespan in enumerate(history['makespan_history']):
        if makespan == best_makespan:
            best_iteration = i + 1
            break
    
    logger.info(f"BLE 算法第{run_id}次运行完成，最优解：{best_makespan:.2f}，最优解代数：{best_iteration}，用时：{total_time:.2f}s")
    
    return best_makespan, total_time, best_iteration


def save_results(results):
    """保存实验结果到CSV文件
    
    Args:
        results: 实验结果列表，每个元素为(算法名称, 运行次数, 最优解, 求解时间, 最优解代数)
    """
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['算法名称', '运行次数', '最优解', '求解时间(s)', '最优解代数'])
        # 写入结果
        for result in results:
            writer.writerow(result)
    logger.info("\n" + "=" * 50)
    logger.info(f"实验结果已保存至：{results_file}")


def main():
    """主函数，运行所有实验并记录结果"""
    results = []
    
    # 运行SL算法NUM_RUNS次
    for i in range(1, NUM_RUNS + 1):
        best_makespan, total_time, best_iteration = run_sl(i)
        results.append(('SL', i, best_makespan, total_time, best_iteration))
    
    # 运行EL算法NUM_RUNS次
    for i in range(1, NUM_RUNS + 1):
        best_makespan, total_time, best_iteration = run_el(i)
        results.append(('EL', i, best_makespan, total_time, best_iteration))
    
    # 运行HL算法NUM_RUNS次
    for i in range(1, NUM_RUNS + 1):
        best_makespan, total_time, best_iteration = run_hl(i)
        results.append(('HL', i, best_makespan, total_time, best_iteration))
    
    # 运行BLEA算法NUM_RUNS次
    for i in range(1, NUM_RUNS + 1):
        best_makespan, total_time, best_iteration = run_blea(i)
        results.append(('BLEA', i, best_makespan, total_time, best_iteration))
    
    # 运行BLE算法NUM_RUNS次
    for i in range(1, NUM_RUNS + 1):
        best_makespan, total_time, best_iteration = run_iblea(i)
        results.append(('BLE', i, best_makespan, total_time, best_iteration))
    
    # 保存结果
    save_results(results)
    
    # 计算统计信息
    sl_results = [r for r in results if r[0] == 'SL']
    el_results = [r for r in results if r[0] == 'EL']
    hl_results = [r for r in results if r[0] == 'HL']
    blea_results = [r for r in results if r[0] == 'BLEA']
    iblea_results = [r for r in results if r[0] == 'IBLEA']
    
    sl_makespans = [r[2] for r in sl_results]
    el_makespans = [r[2] for r in el_results]
    hl_makespans = [r[2] for r in hl_results]
    blea_makespans = [r[2] for r in blea_results]
    iblea_makespans = [r[2] for r in iblea_results]
    
    sl_times = [r[3] for r in sl_results]
    el_times = [r[3] for r in el_results]
    hl_times = [r[3] for r in hl_results]
    blea_times = [r[3] for r in blea_results]
    iblea_times = [r[3] for r in iblea_results]
    
    sl_iterations = [r[4] for r in sl_results]
    el_iterations = [r[4] for r in el_results]
    hl_iterations = [r[4] for r in hl_results]
    blea_iterations = [r[4] for r in blea_results]
    iblea_iterations = [r[4] for r in iblea_results]
    
    # 输出统计信息
    logger.info("实验统计结果：")
    logger.info(f"SL算法：平均最优解 = {np.mean(sl_makespans):.2f}，最佳解 = {min(sl_makespans):.2f}，平均用时 = {np.mean(sl_times):.2f}s，平均最优解代数 = {np.mean(sl_iterations):.2f}")
    logger.info(f"EL算法：平均最优解 = {np.mean(el_makespans):.2f}，最佳解 = {min(el_makespans):.2f}，平均用时 = {np.mean(el_times):.2f}s，平均最优解代数 = {np.mean(el_iterations):.2f}")
    logger.info(f"HL算法：平均最优解 = {np.mean(hl_makespans):.2f}，最佳解 = {min(hl_makespans):.2f}，平均用时 = {np.mean(hl_times):.2f}s，平均最优解代数 = {np.mean(hl_iterations):.2f}")
    logger.info(f"BLEA算法：平均最优解 = {np.mean(blea_makespans):.2f}，最佳解 = {min(blea_makespans):.2f}，平均用时 = {np.mean(blea_times):.2f}s，平均最优解代数 = {np.mean(blea_iterations):.2f}")
    logger.info(f"IBLEA算法：平均最优解 = {np.mean(iblea_makespans):.2f}，最佳解 = {min(iblea_makespans):.2f}，平均用时 = {np.mean(iblea_times):.2f}s，平均最优解代数 = {np.mean(iblea_iterations):.2f}")
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        raise