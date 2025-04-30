"""
完工时间计算类定义

该模块用于模拟作业车间调度过程并计算完工时间，包括：
- 字符串解码
- 工序的模拟执行和调度
- 总完工时间（运输时间&加工时间）的计算
"""

# 标准库导入
from typing import Dict, List, Optional, Set, Tuple, Union

# 第三方库导入
import numpy as np

# 本地模块导入
from models import Operation, Machine, AGV
from MK10 import (
    NUM_JOBS as num_jobs,
    NUM_OPERATIONS as num_operations,
    NUM_MACHINES as num_machines,
    NUM_AGVS_W as num_agvs_w,
    NUM_AGVS_F as num_agvs_f,
    JOB_OPERATIONS as job_operations,
    AGV_W_TRANSPORT_TIMES as agv_w_transport_times,
    AGV_F_TRANSPORT_TIMES as agv_f_transport_times,
    PROCESSING_TIMES as processing_times,
    PRIORITY_DICT as priority_dict,
)

# 常量定义
DEFAULT_PICKUP_LOCATION = 'Warehouse'  # 默认AGV\工件初始位置

class Calculate:
    """计算完工时间，负责管理和执行作业车间调度过程"""
    
    def __init__(self):
        """初始化设备和工序对象"""
        # 初始化机器
        self.machines = [Machine(i+1) for i in range(num_machines)]
        # 初始化AGV
        self.agvs_w = [AGV(i+1) for i in range(num_agvs_w)]
        self.agvs_f = [AGV(i+1) for i in range(num_agvs_f)]
        # 初始化工序字典，键为工序ID（如'o1_1'），值为Operation对象
        self.operations = {
            f"o{job_id}_{op_seq}": Operation(
                op_id=f"o{job_id}_{op_seq}",
                job_id=job_id,
                seq_in_job=op_seq
            )
            for job_id in range(1, num_jobs + 1)
            for op_seq in range(1, job_operations[job_id] + 1)
        }

    def get_operations_list(self) -> List[str]:
        """获取所有工序ID列表
        
        Returns:
            List[str]: 工序ID列表，格式为 ['o11', 'o12', ...]
        """
        return list(self.operations.keys())

    def set_machine_assignments(self, M: np.ndarray) -> None:
        """设置机器分配
        
        Args:
            M: 机器分配矩阵 (工序数量 x 机器数量)
            
        Raises:
            ValueError: 当机器分配矩阵维度不正确或分配无效时
        """
        # 检查矩阵维度
        if M.shape != (num_operations, num_machines):
            raise ValueError(f"机器分配矩阵维度不正确，应为 ({num_operations}, {num_machines})")
            
        # 为每个工序分配机器
        for i, op_id in enumerate(self.get_operations_list()):
            machine_idx = np.where(M[i, :] == 1)[0]
            if len(machine_idx) != 1:
                raise ValueError(f"工序 {op_id} 的机器分配无效")
                
            machine_id = machine_idx[0] + 1
            self.operations[op_id].assigned_machine = self.machines[machine_id - 1]
            self.machines[machine_id - 1].operation_sequence.append(self.operations[op_id])

    def set_agv_assignments(self, A: np.ndarray, F: np.ndarray) -> None:
        """设置AGV分配
        
        Args:
            A: AGV_w分配矩阵 (工序数量 x AGV_W数量)
            F: AGV_F分配矩阵 (工序数量 x AGV_F数量)
            
        Raises:
            ValueError: 当AGV分配矩阵维度不正确或分配无效时
        """
        # 检查矩阵维度
        if A.shape != (num_operations, num_agvs_w):
            raise ValueError(f"AGV分配矩阵维度不正确，应为 ({num_operations}, {num_agvs_w})")
        if F.shape != (num_operations, num_agvs_f):
            raise ValueError(f"AGV分配矩阵维度不正确，应为 ({num_operations}, {num_agvs_f})")

        # 为每个工序分配AGV_W
        for i, op_id in enumerate(self.get_operations_list()):
            agv_idx = np.where(A[i, :] == 1)[0]
            if len(agv_idx) != 1:
                raise ValueError(f"工序 {op_id} 的AGV_W分配无效")
                
            agv_id = agv_idx[0] + 1
            self.operations[op_id].assigned_agv_w = self.agvs_w[agv_id - 1]
            self.agvs_w[agv_id - 1].operation_sequence.append(self.operations[op_id])

        # 为每个工序分配AGV_F
        for i, op_id in enumerate(self.get_operations_list()):
            agv_idx = np.where(F[i, :] == 1)[0]
            if len(agv_idx) != 1:
                raise ValueError(f"工序 {op_id} 的AGV_F分配无效")
                
            agv_id = agv_idx[0] + 1
            self.operations[op_id].assigned_agv_f = self.agvs_f[agv_id - 1]
            self.agvs_f[agv_id - 1].operation_sequence.append(self.operations[op_id])


    def get_operation_location(self, operation: Operation) -> str:
        """获取工序的取件位置
        
        Args:
            operation: 工序对象
            
        Returns:
            str: 取件位置
        """
        if operation.prev_operation:
            return operation.prev_operation.assigned_machine.location
        return DEFAULT_PICKUP_LOCATION  # 如果是第一道工序，从仓库取料

    def _collect_schedule_data(self) -> Tuple[List[str], List[Tuple[List, List]], List[str], List[Tuple[List, List]], List[str], List[Tuple[List, List]]]:
        """
        收集调度数据
        :return: (agv_w_headers, agv_w_data, agv_f_headers, agv_f_data, machine_headers, machine_data)
        """
        # AGV_W数据
        agv_w_headers = ["AGV", "工序", "出发位置", "取件位置", "目标机器", 
                    "开始运输", "到达取件点", "离开取件点", "到达机器", "完成配送"]
        agv_w_data = []
        
        for agv in self.agvs_w:
            for op in self.operations.values():
                if op.assigned_agv_w == agv:
                    # 获取位置信息
                    pickup_loc = self.get_operation_location(op)
                    delivery_loc = op.assigned_machine.location
                    from_loc = agv.get_location_at_operation(op.op_id + '_start')
                    
                    # 获取运输时间点
                    times = [op.transport_times[key] for key in Operation.AGV_W_TIME_KEYS]
                    times_numeric = [float(t) if t is not None else float('inf') for t in times]
                    times_display = [str(t) if t is not None else "" for t in times]
                    
                    # 将AGV编号和开始运输时间作为排序关键字
                    row_for_sort = [
                        agv.agv_id,  # 使用数字而不是字符串，便于排序
                        times_numeric[0],  # 开始运输时间
                        op.op_id,
                        from_loc,
                        pickup_loc,
                        delivery_loc,
                        *times_numeric[1:]  # 其他时间点
                    ]
                    row_for_display = [
                        f"AGV{agv.agv_id}",
                        op.op_id,
                        from_loc,
                        pickup_loc,
                        delivery_loc,
                        *times_display  # 用于显示的字符串
                    ]
                    agv_w_data.append((row_for_sort, row_for_display))
        
        # AGV_F数据
        agv_f_headers = ["AGV", "工序", "出发位置", "目标机器", "返回位置", 
                        "开始运输", "到达机器", "返回仓库"]
        agv_f_data = []
        
        for agv in self.agvs_f:
            for op in self.operations.values():
                if op.assigned_agv_f == agv:
                    # 获取位置信息
                    delivery_to = op.assigned_machine.location
                    
                    # 获取运输时间点
                    times = [op.transport_times[key] for key in Operation.AGV_F_TIME_KEYS]
                    times_numeric = [float(t) if t is not None else float('inf') for t in times]
                    times_display = [str(t) if t is not None else "" for t in times]
                    
                    # 将AGV编号和开始运输时间作为排序关键字
                    row_for_sort = [
                        agv.agv_id,  # 使用数字而不是字符串，便于排序
                        times_numeric[0],  # 开始运输时间
                        op.op_id,
                        'Warehouse',
                        delivery_to,
                        'Warehouse',
                        *times_numeric[1:]  # 其他时间点
                    ]
                    row_for_display = [
                        f"AGV{agv.agv_id}",
                        op.op_id,
                        'Warehouse',
                        delivery_to,
                        'Warehouse',
                        *times_display  # 用于显示的字符串
                    ]
                    agv_f_data.append((row_for_sort, row_for_display))
        
        # 机器数据
        machine_headers = ["机器", "工序", "开始加工", "完成加工", "加工时间"]
        machine_data = []
        
        for machine in self.machines:
            for op in self.operations.values():
                if op.assigned_machine == machine and op.process_start_time is not None:
                    process_time = op.process_end_time - op.process_start_time
                    
                    times_numeric = [
                        float(op.process_start_time),
                        float(op.process_end_time),
                        float(process_time)
                    ]
                    times_display = [
                        str(op.process_start_time),
                        str(op.process_end_time),
                        str(process_time)
                    ]
                    
                    # 将机器编号和开始加工时间作为排序关键字
                    row_for_sort = [
                        machine.machine_id,  # 使用数字而不是字符串，便于排序
                        float(op.process_start_time),  # 开始加工时间
                        op.op_id,
                        *times_numeric[1:]  # 其他时间点
                    ]
                    row_for_display = [
                        f"M{machine.machine_id}",
                        op.op_id,
                        *times_display  # 用于显示的字符串
                    ]
                    machine_data.append((row_for_sort, row_for_display))
        
        return agv_w_headers, agv_w_data, agv_f_headers, agv_f_data, machine_headers, machine_data

    def simulate(self, machine_string: Optional[List[int]] = None, 
                agv_w_string: Optional[List[int]] = None, 
                agv_f_string: Optional[List[int]] = None, 
                schedule_string: Optional[List[int]] = None,
                return_schedule: bool = False) -> Union[float, Tuple[float, List[Dict[str, object]]]]:
        """
        模拟执行调度方案
        :param machine_string: 机器分配字符串
        :param agv_w_string: AGV_W分配字符串
        :param agv_f_string: AGV_F分配字符串
        :param schedule_string: 调度字符串
        :param return_schedule: 是否返回调度结果
        :return: 如果return_schedule为False，返回makespan；否则返回(makespan, schedule_results)
        """
        # 重置工厂状态
        self.reset()
        
        # 如果没有提供参数，则直接执行当前的调度方案
        if machine_string is not None and agv_w_string is not None and agv_f_string is not None and schedule_string is not None:
            # 解码调度方案
            order = self.decode_schedule(schedule_string, priority_dict)
            # 解码机器分配和AGV分配
            M, A, F = self.decode_assignments(machine_string, agv_w_string, agv_f_string)
            
            # 设置机器分配和AGV分配
            self.set_machine_assignments(M)
            self.set_agv_assignments(A, F)
            
            # 更新工序序列
            self.update_sequences(order)
        
        # 模拟执行调度方案
        self._simulate_schedule()
        
        # 计算makespan
        max_machine_time = max((op.process_end_time or 0) 
                            for op in self.operations.values())
        max_agv_time = max((op.transport_times['delivery_complete'] or 0) 
                        for op in self.operations.values())
        makespan = max(max_machine_time, max_agv_time)
        
        if return_schedule:
            # 收集调度数据
            agv_w_headers, agv_w_data, agv_f_headers, agv_f_data, machine_headers, machine_data = self._collect_schedule_data()
            
            # 整理调度结果
            schedule_results = []
            
            # 添加AGV_W数据
            for _, row in agv_w_data:
                schedule_results.append({
                    'device': row[0],
                    'operation': row[1],
                    'from_location': row[2],
                    'pickup_location': row[3],
                    'delivery_location': row[4],
                    'start_time': row[5],
                    'pickup_arrive': row[6],
                    'pickup_leave': row[7],
                    'delivery_arrive': row[8],
                    'delivery_complete': row[9]
                })
            
            # 添加AGV_F数据
            for _, row in agv_f_data:
                schedule_results.append({
                    'device': row[0],
                    'operation': row[1],
                    'from_location': row[2],
                    'delivery_location': row[3],
                    'return_location': row[4],
                    'start_time': row[5],
                    'delivery_arrive': row[6],
                    'return_complete': row[7]
                })
            
            # 添加机器数据
            for _, row in machine_data:
                schedule_results.append({
                    'device': row[0],
                    'operation': row[1],
                    'start_time': row[2],
                    'end_time': row[3],
                    'process_time': row[4]
                })
            
            return makespan, schedule_results
        
        return makespan

    def decode_assignments(self, machine_string: List[int], agv_w_string: List[int], agv_f_string: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        解码机器分配和AGV分配字符串
        :param machine_string: 机器分配字符串
        :param agv_w_string: AGV_W分配字符串
        :param agv_f_string: AGV_F分配字符串
        :return: 机器分配矩阵M和AGV分配矩阵A
        """
        # 检查输入长度
        if len(machine_string) != num_operations or len(agv_w_string) != num_operations or len(agv_f_string) != num_operations:
            raise ValueError("分配字符串长度必须等于工序数量")
        
        # 创建机器分配矩阵 (工序数量 x 机器数量)
        M = np.zeros((num_operations, num_machines))
        # 创建AGV_W分配矩阵 (工序数量 x AGV_W数量)
        A = np.zeros((num_operations, num_agvs_w))
        # 创建AGV_F分配矩阵 (工序数量 x AGV_F数量)
        F = np.zeros((num_operations, num_agvs_f))
        
        # 解码机器分配
        for i, (op_id, machine_id) in enumerate(zip(self.get_operations_list(), machine_string)):
            # 检查机器分配是否合法
            if machine_id not in processing_times[op_id]:
                raise ValueError(f"工序 {op_id} 不能在机器 {machine_id} 上加工")
            M[i, machine_id - 1] = 1
            
        # 解码AGV_W分配
        for i, agv_id in enumerate(agv_w_string):
            if not 1 <= agv_id <= num_agvs_w:
                raise ValueError(f"AGV编号 {agv_id} 超出范围")
            A[i, agv_id - 1] = 1
        
        # 解码AGV_F分配
        for i, agv_id in enumerate(agv_f_string):
            if not 1 <= agv_id <= num_agvs_f:
                raise ValueError(f"AGV编号 {agv_id} 超出范围")
            F[i, agv_id - 1] = 1
            
        return M, A, F

    def decode_schedule(self, schedule_string: List[int], priority_dict: Dict[str, set]) -> List[str]:
        """
        解码调度字符串
        :param schedule_string: 调度字符串，直接表示工序的处理顺序
        :param priority_dict: 工序间优先级字典
        :return: 工序处理顺序列表
        """
        # 获取所有工序ID列表
        operations_list = self.get_operations_list()

        # 创建工序索引到工序ID的映射
        op_index_to_id = {i+1: op_id for i, op_id in enumerate(operations_list)}
        
        # 验证调度字符串是否合法
        self._validate_schedule_string(schedule_string, op_index_to_id, priority_dict)
        
        # 直接将调度字符串转换为工序处理顺序
        order = [op_index_to_id[op_index] for op_index in schedule_string]
        
        return order
        
    def _validate_schedule_string(self, schedule_string: List[int], op_index_to_id: Dict[int, str], priority_dict: Dict[str, set]) -> None:
        """
        验证调度字符串是否合法, 是否满足工序优先级约束
        
        Args:
            schedule_string: 调度字符串
            op_index_to_id: 工序索引到工序ID的映射
            priority_dict: 工序优先级字典
        
        Raises:
            ValueError: 如果调度字符串不满足工序优先级约束
        """
        # 验证调度字符串长度
        if len(schedule_string) != sum(job_operations.values()):
            raise ValueError(f"调度字符串长度({len(schedule_string)})必须等于工序数量({num_operations})")
        
        # 验证调度字符串中的值是否为1到工序数量的排列
        if sorted(schedule_string) != list(range(1, num_operations + 1)):
            raise ValueError("调度字符串必须是1到工序数量的排列")

        # 创建工序ID到其在调度字符串中位置的映射
        op_id_to_position = {}
        for i, op_index in enumerate(schedule_string):
            op_id = op_index_to_id[op_index]
            op_id_to_position[op_id] = i
        
        # 检查每个工序的前置工序是否在其之前
        for op_id, position in op_id_to_position.items():
            predecessors = priority_dict.get(op_id, set())
            for pred in predecessors:
                if pred in op_id_to_position and op_id_to_position[pred] >= position:
                    raise ValueError(
                        f"工序{op_id}的前置工序{pred}应该在其之前处理"
                    )

    def _find_assignable_op(self, op: str, job_id_operations: Dict[int, List[str]], 
                        order: List[str], priority_dict: Dict[str, set]) -> Optional[str]:
        """递归查找可以分配的工序
        
        Args:
            op: 当前检查的工序
            job_id_operations: 每个作业ID对应的未分配工序列表
            order: 当前的工序排序结果
            priority_dict: 工序间优先级字典
            
        Returns:
            Optional[str]: 可分配的工序，如果找不到返回None
        """
        visited = set()
        
        # 如果工序已经被访问过，说明存在循环依赖
        if op in visited:
            return None
        else: 
            visited.add(op) # 访问过的工序加入集合
        
        # 获取当前工序的前置工序
        predecessors = priority_dict.get(op, set())
        
        # 如果没有前置工序，直接返回当前工序
        if not predecessors:
            return op
        
        # 如果所有前置工序都不在order中，说明可以从前置工序中选择一个分配
        if all(pred not in order for pred in predecessors):
            # 递归检查每个前置工序
            for pred in predecessors:
                # 获取前置工序所属的作业ID
                pred_job = int(pred.split('_')[0][1:])  # 从'o{job_id}_{op_seq}'中提取作业ID中提取作业ID
                # 如果这个前置工序还在待分配列表中
                if pred in job_id_operations.get(pred_job, []):
                    assignable = self._find_assignable_op(
                        pred, job_id_operations, order, priority_dict
                    )
                    if assignable:
                        return assignable
        
        # 如果所有前置工序都已分配，返回当前工序
        if all(pred in order for pred in predecessors):
            return op
        
        return None

    def update_sequences(self, order: List[str]):
        """
        根据调度顺序更新机器和AGV的工序序列
        :param order: 工序处理顺序
        """
        try:
            # 清空现有序列
            self._clear_sequences()
            
            # 更新工序序号和依赖关系
            self._update_operation_dependencies(order)
            
            # 更新设备序列
            self._update_device_sequences(order)
            
        except Exception as e:
            raise ValueError(f"更新序列时出错: {str(e)}")

    def _clear_sequences(self):
        """清空所有设备的序列"""
        for machine in self.machines:
            machine.operation_sequence = []
        for agv in self.agvs_w:
            agv.operation_sequence = []
        for agv in self.agvs_f:
            agv.operation_sequence = []

    def _update_operation_dependencies(self, order: List[str]):
        """
        更新工序的序号和依赖关系
        """       
        # 记录每个作业的最后一道工序
        job_last_op = {}
        
        for op_id in order:
            op = self.operations[op_id]
            
            # 如果这个作业已经有工序了，设置依赖关系
            if op.job_id in job_last_op:
                prev_op = job_last_op[op.job_id]
                op.prev_operation = prev_op
            
            # 更新这个作业的最后一道工序
            job_last_op[op.job_id] = op
        
    def _update_device_sequences(self, order: List[str]):
        """
        更新机器和AGV的工序序列
        """
        for op_id in order:
            op = self.operations[op_id]
            # 更新机器序列
            if op.assigned_machine:
                op.assigned_machine.operation_sequence.append(op)
            # 更新AGV序列
            if op.assigned_agv_w:
                op.assigned_agv_w.operation_sequence.append(op)
            if op.assigned_agv_f:
                op.assigned_agv_f.operation_sequence.append(op)

    def _simulate_schedule(self) -> None:
        """
        模拟执行调度方案
        """
        # 持续模拟直到所有工序都完成
        while not self._all_operations_completed():
            # 更新AGV_W运输状态
            for agv in self.agvs_w:
                self._update_agv_w_status(agv)
                
            # 更新AGV_F运输状态
            for agv in self.agvs_f:
                self._update_agv_f_status(agv)
                
            # 更新机器加工状态
            for machine in self.machines:
                self._update_machine_status(machine)

    def _all_operations_completed(self) -> bool:
        """
        检查是否所有工序都已完成
        :return: 是否所有工序都已完成
        """
        return all(op.process_end_time is not None 
                for op in self.operations.values())

    def _update_agv_w_status(self, agv: AGV) -> None:
        """
        更新AGV_W状态
        :param agv: AGV对象
        """
        # 如果AGV没有任务，直接返回
        if not agv.operation_sequence:
            return
        
        current_op = agv.operation_sequence[0]
        
        # 如果是新任务，初始化运输时间
        if current_op.transport_times['pickup_start'] is None:
            # 检查前序工序是否完成
            if current_op.prev_operation and current_op.prev_operation.process_end_time is None:
                return  # 等待前序工序完成
            
            # 获取起点和终点
            from_loc = agv.current_location
            pickup_loc = self.get_operation_location(current_op)
            
            # 开始新的运输任务
            current_op.transport_times['pickup_start'] = agv.current_time
            pickup_time = agv_w_transport_times[from_loc][pickup_loc]  # 使用AGV_W的运输时间矩阵
            current_op.transport_times['pickup_arrive'] = agv.current_time + pickup_time
            
            # 记录运输起点和取件位置
            agv.update_location(current_op.op_id + '_start', from_loc)
            agv.update_location(current_op.op_id + '_pickup', pickup_loc)
            
        # 更新AGV位置和时间
        if current_op.transport_times['pickup_arrive'] is not None and current_op.transport_times['pickup_leave'] is None:
            # 到达取件点，等待前序工序完成（如果有）
            if current_op.prev_operation and current_op.prev_operation.process_end_time is None:
                return
            
            current_op.transport_times['pickup_leave'] = max(
                current_op.transport_times['pickup_arrive'],
                current_op.prev_operation.process_end_time if current_op.prev_operation else 0
            )
            
            # 计算到达目标机器的时间
            delivery_from = agv.current_location  # 当前位置就是取件位置
            delivery_to = current_op.assigned_machine.location
            delivery_time = agv_w_transport_times[delivery_from][delivery_to]  # 使用AGV_W的运输时间矩阵
            current_op.transport_times['delivery_arrive'] = current_op.transport_times['pickup_leave'] + delivery_time
            current_op.transport_times['delivery_complete'] = current_op.transport_times['delivery_arrive']
            
            # 更新AGV状态
            agv.update_location(current_op.op_id + '_delivery', delivery_to)
            agv.current_time = current_op.transport_times['delivery_complete']
            agv.operation_sequence.pop(0)  # 移除完成的任务

    def _update_agv_f_status(self, agv: AGV) -> None:
        """更新AGV_F状态
        
        Args:
            agv: AGV对象
        """
        # 如果AGV没有任务，直接返回
        if not agv.operation_sequence:
            return
        
        current_op = agv.operation_sequence[0]
        
        # 如果是新任务，初始化运输时间
        if current_op.transport_times['f_pickup_start'] is None:
            # 获取目标机器位置
            delivery_to = current_op.assigned_machine.location
            
            # 开始新的运输任务
            current_op.transport_times['f_pickup_start'] = agv.current_time
            delivery_time = agv_f_transport_times['Warehouse'][delivery_to]  # 使用AGV_F的运输时间矩阵
            current_op.transport_times['f_delivery_arrive'] = agv.current_time + delivery_time
            
            # 计算返回Warehouse的时间
            return_time = agv_f_transport_times[delivery_to]['Warehouse']  # 使用AGV_F的运输时间矩阵
            current_op.transport_times['f_delivery_complete'] = current_op.transport_times['f_delivery_arrive'] + return_time
            
            # 更新AGV状态
            agv.update_location(current_op.op_id + '_start', 'Warehouse')
            agv.update_location(current_op.op_id + '_delivery', delivery_to)
            agv.update_location(current_op.op_id + '_complete', 'Warehouse')
            agv.current_time = current_op.transport_times['f_delivery_complete']
            agv.operation_sequence.pop(0)  # 移除完成的任务

    def _update_machine_status(self, machine: Machine) -> None:
        """更新机器状态
        
        Args:
            machine: 机器对象
        """
        # 如果机器没有任务，直接返回
        if not machine.operation_sequence:
            return
        
        current_op = machine.operation_sequence[0]
        
        # 检查工件和物料是否都已经到达
        if (current_op.transport_times['delivery_complete'] is None or 
            current_op.transport_times['f_delivery_arrive'] is None):
            return
        
        # 如果工序还未开始加工
        if current_op.process_start_time is None:
            # 检查前序工序是否完成
            if current_op.prev_operation and current_op.prev_operation.process_end_time is None:
                return
            
            # 开始加工，需要同时满足AGV_W和AGV_F的到达时间约束
            current_op.process_start_time = max(
                machine.current_time,
                current_op.transport_times['delivery_complete'],
                current_op.transport_times['f_delivery_arrive']
            )
            process_time = processing_times[current_op.op_id][machine.machine_id]
            current_op.process_end_time = current_op.process_start_time + process_time
            
            # 更新机器状态
            machine.current_time = current_op.process_end_time
            machine.operation_sequence.pop(0)  # 移除完成的任务

    def reset(self):
        """重置工厂状态"""
        # 重置所有机器状态
        for machine in self.machines:
            machine.reset()
        
        # 重置所有AGV状态
        for agv in self.agvs_w:
            agv.reset()
        for agv in self.agvs_f:
            agv.reset()
        
        # 重置所有工序状态
        for operation in self.operations.values():
            operation.reset()