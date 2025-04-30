
import os
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from collections import OrderedDict

# 导入计算模块
from calculate import Calculate
calc = Calculate()
from models import Operation, Machine, AGV
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
"""生成符合工序优先级约束的工序调度编码
        
生成一个1到工序数量的排列, 确保前置工序在后继工序之前

Returns:
    list: 工序编码序列
"""
# 获取所有工序ID列表
operations = {
            f"o{job_id}_{op_seq}": Operation(
                op_id=f"o{job_id}_{op_seq}",
                job_id=job_id,
                seq_in_job=op_seq
            )
            for job_id in range(1, num_jobs + 1)
            for op_seq in range(1, job_operations[job_id] + 1)
        }
operations_list = list(operations.keys())


schedule = []

for i in range(num_operations):

    schedule.append(i+1)


# 检查是否所有工序都被排序（检测循环依赖）
if len(schedule) != num_operations:
    raise ValueError("工序依赖图中存在循环依赖，无法生成有效的调度序列")

op_index_to_id = {i+1: op_id for i, op_id in enumerate(operations_list)}

# 验证调度字符串是否合法
calc._validate_schedule_string(schedule, op_index_to_id, priority_dict)

# 直接将调度字符串转换为工序处理顺序
order = [op_index_to_id[op_index] for op_index in schedule]
print(order)