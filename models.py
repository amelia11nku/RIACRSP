"""
核心数据模型定义
"""
from typing import Optional, Dict, List

class Operation:
    AGV_W_TIME_KEYS = ['pickup_start', 'pickup_arrive', 'pickup_leave', 'delivery_arrive', 'delivery_complete']
    AGV_F_TIME_KEYS = ['f_pickup_start', 'f_delivery_arrive', 'f_delivery_complete']
    
    def __init__(self, op_id: str, job_id: int, seq_in_job: int):
        """
        初始化工序对象
        :param op_id: 工序ID，例如'o1_1'表示作业1的第1道工序
        :param job_id: 作业ID
        :param seq_in_job: 在作业中的序号
        """
        self.op_id: str = op_id
        self.job_id: int = job_id
        self.seq_in_job: int = seq_in_job
        self.prev_operation: Optional['Operation'] = None  # 前序工序
        self.assigned_machine: Optional['Machine'] = None  # 分配的机器
        self.assigned_agv_w: Optional['AGV'] = None  # 分配的AGV_W
        self.assigned_agv_f: Optional['AGV'] = None  # 分配的AGV_F
        
        # 加工时间相关
        self.process_start_time: Optional[float] = None  # 加工开始时间
        self.process_end_time: Optional[float] = None  # 加工结束时间
        
        # 运输时间记录
        self.transport_times: Dict[str, Optional[float]] = {
            **{key: None for key in self.AGV_W_TIME_KEYS},
            **{key: None for key in self.AGV_F_TIME_KEYS}
        }

    def reset(self):
        """重置工序状态"""
        self.prev_operation = None
        self.assigned_machine = None
        self.assigned_agv_w = None
        self.assigned_agv_f = None
        self.process_start_time = None
        self.process_end_time = None
        self.transport_times = {
            **{key: None for key in self.AGV_W_TIME_KEYS},
            **{key: None for key in self.AGV_F_TIME_KEYS}
        }

class Machine:
    def __init__(self, machine_id: int):
        """
        初始化机器对象
        :param machine_id: 机器ID
        """
        self.machine_id: int = machine_id
        self.operation_sequence: List['Operation'] = []  # 工序处理顺序
        self.current_time: float = 0  # 当前时间
        self.location: str = f"M{machine_id}"  # 机器位置

    def reset(self):
        """重置机器状态"""
        self.operation_sequence = []
        self.current_time = 0

class AGV:
    DEFAULT_LOCATION = 'Warehouse'
    
    def __init__(self, agv_id: int):
        """
        初始化AGV对象
        :param agv_id: AGV ID
        """
        self.agv_id: int = agv_id
        self.operation_sequence: List['Operation'] = []  # 运输任务序列
        self.locations: Dict[str, str] = {'initial': self.DEFAULT_LOCATION}  # 记录每个工序对应的位置
        self.current_time: float = 0  # 当前时间

    @property
    def current_location(self) -> str:
        """获取AGV当前位置"""
        return next(reversed(self.locations.values())) if self.locations else self.DEFAULT_LOCATION

    def update_location(self, op_id: str, location: str) -> None:
        """
        更新AGV在特定工序的位置
        :param op_id: 工序ID
        :param location: 新位置
        """
        self.locations[op_id] = location

    def get_location_at_operation(self, op_id: str) -> str:
        """
        获取AGV在特定工序时的位置
        :param op_id: 工序ID
        :return: 位置
        """
        return self.locations.get(op_id, self.current_location)

    def reset(self):
        """重置AGV状态"""
        self.operation_sequence = []
        self.locations = {'initial': self.DEFAULT_LOCATION}
        self.current_time = 0