# For LMEO and BLMCE
# -*- coding: utf-8 -*-
"""
Discrete Event Simulation (DES) Decoder for AFAISP
优化点：避免重复重建依赖图，优化关键路径提取效率
"""
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, Any
from data.loader import Instance

DEFAULT_LOCATION = "Warehouse"
EPSILON = 1e-6

class Operation:
    __slots__ = (
        "op_id", "job_id", "seq_in_job",
        "assigned_machine_id", "assigned_agv_w_id", "assigned_agv_f_id", "prev_operation",
        "w_pickup_start", "w_pickup_arrive", "w_pickup_leave",
        "w_delivery_arrive", "w_delivery_complete",
        "w_from_location", "w_pickup_location", "w_delivery_location",
        "f_start_time", "f_delivery_arrive", "f_return_complete",
        "f_from_location", "f_delivery_location", "f_return_location",
        "process_start", "process_end",
    )
    def __init__(self, op_id: str, job_id: int, seq_in_job: int):
        self.op_id, self.job_id, self.seq_in_job = op_id, job_id, seq_in_job
        self.reset()

    def reset(self) -> None:
        self.assigned_machine_id = self.assigned_agv_w_id = self.assigned_agv_f_id = self.prev_operation = None
        self.w_pickup_start = self.w_pickup_arrive = self.w_pickup_leave = None
        self.w_delivery_arrive = self.w_delivery_complete = None
        self.w_from_location = self.w_pickup_location = self.w_delivery_location = None
        self.f_start_time = self.f_delivery_arrive = self.f_return_complete = None
        self.f_from_location = self.f_delivery_location = self.f_return_location = None
        self.process_start = self.process_end = None

class Resource:
    __slots__ = ("id", "queue", "current_time", "location")
    def __init__(self, r_id: int, initial_location: str):
        self.id, self.location = r_id, initial_location
        self.queue, self.current_time = deque(), 0.0
    def reset(self) -> None:
        self.queue.clear()
        self.current_time = 0.0

class Machine(Resource):
    __slots__ = ()
    def __init__(self, m_id: int):
        super().__init__(m_id, f"M{m_id}")

class AGV(Resource):
    __slots__ = ("loc_history",)
    def __init__(self, a_id: int):
        super().__init__(a_id, DEFAULT_LOCATION)
        self.loc_history: Dict[str, str] = {}
    def reset(self) -> None:
        super().reset()
        self.location = DEFAULT_LOCATION
        self.loc_history.clear()
    def record_location(self, key: str, loc: str) -> None:
        self.loc_history[key] = loc
        self.location = loc

class Calculate:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.machines = [Machine(i + 1) for i in range(instance.num_machines)]
        self.agvs_w = [AGV(i + 1) for i in range(instance.num_agvs_w)]
        self.agvs_f = [AGV(i + 1) for i in range(instance.num_agvs_f)]

        self.operations: Dict[str, Operation] = {}
        self.op_list: List[Operation] = []
        self.op_id_to_idx: Dict[str, int] = {}

        idx = 0
        for job_id in range(1, instance.num_jobs + 1):
            for seq in range(1, instance.job_operations[job_id] + 1):
                op_id = f"o{job_id}_{seq}"
                op = Operation(op_id, job_id, seq)
                self.operations[op_id] = op
                self.op_list.append(op)
                self.op_id_to_idx[op_id] = idx
                idx += 1

        self.num_ops = len(self.op_list)
        self.proc_times = instance.processing_times
        self.tt_w = instance.agv_w_transport_times
        self.tt_f = instance.agv_f_transport_times
        self.idx_to_op = {i + 1: op for i, op in enumerate(self.op_list)}
        
        # 缓存关键路径，避免重复计算
        self._last_cp_ops: Optional[List[Operation]] = None

    def simulate(self, machine_seq: List[int], agv_w_seq: List[int], agv_f_seq: List[int],
                 os_seq: List[int], return_schedule: bool = False, return_pblock: bool = False):
        if not os_seq: return (0.0, []) if return_schedule else 0.0

        for m in self.machines: m.reset()
        for a in self.agvs_w: a.reset()
        for a in self.agvs_f: a.reset()
        for op in self.op_list: op.reset()
        self._last_cp_ops = None # 重置缓存

        for i, op in enumerate(self.op_list):
            op.assigned_machine_id = self.machines[machine_seq[i] - 1]
            op.assigned_agv_w_id = self.agvs_w[agv_w_seq[i] - 1]
            op.assigned_agv_f_id = self.agvs_f[agv_f_seq[i] - 1]

        job_last: Dict[int, Operation] = {}
        for os_idx in os_seq:
            op = self.idx_to_op[os_idx]
            op.prev_operation = job_last.get(op.job_id, None)
            job_last[op.job_id] = op
            op.assigned_machine_id.queue.append(op)
            if not (
                op.prev_operation is not None
                and op.prev_operation.assigned_machine_id == op.assigned_machine_id
            ):
                op.assigned_agv_w_id.queue.append(op)
            op.assigned_agv_f_id.queue.append(op)

        self._run_kernel(total_ops=self.num_ops)

        makespan = max((op.process_end for op in self.op_list if op.process_end is not None), default=0.0)

        if not return_schedule:
            return makespan
        if not return_pblock:
            return makespan, self._collect_schedule_data()
        occ_stats, delay_stats = self._block_stats()
        return makespan, self._collect_schedule_data(), occ_stats, delay_stats

    def _run_kernel(self, total_ops: int) -> None:
        active_machines = [m for m in self.machines if m.queue]
        active_agvs_w = [a for a in self.agvs_w if a.queue]
        active_agvs_f = [a for a in self.agvs_f if a.queue]
        completed_ops = 0

        while completed_ops < total_ops:
            # Phase 1: AGV_W
            for agv in list(active_agvs_w):
                if not agv.queue:
                    active_agvs_w.remove(agv); continue
                op = agv.queue[0]
                if op.prev_operation and op.prev_operation.process_end is None: continue
                prev_end = op.prev_operation.process_end if op.prev_operation else 0.0
                pickup_loc = op.prev_operation.assigned_machine_id.location if op.prev_operation else DEFAULT_LOCATION
                delivery_loc = op.assigned_machine_id.location

                if op.w_pickup_start is None:
                    op.w_from_location, op.w_pickup_location, op.w_delivery_location = agv.location, pickup_loc, delivery_loc
                    op.w_pickup_start = agv.current_time
                    op.w_pickup_arrive = op.w_pickup_start + self.tt_w[agv.location][pickup_loc]

                if op.w_pickup_arrive is not None and op.w_pickup_leave is None:
                    op.w_pickup_leave = max(op.w_pickup_arrive, float(prev_end))
                    op.w_delivery_arrive = op.w_delivery_complete = op.w_pickup_leave + self.tt_w[pickup_loc][delivery_loc]
                    agv.current_time = op.w_delivery_complete
                    agv.location = delivery_loc
                    agv.queue.popleft()

            # Phase 2: AGV_F
            for agv in list(active_agvs_f):
                if not agv.queue:
                    active_agvs_f.remove(agv); continue
                op = agv.queue[0]
                if op.f_start_time is None:
                    op.f_from_location = agv.location
                    op.f_return_location = DEFAULT_LOCATION
                    op.f_delivery_location = op.assigned_machine_id.location
                    op.f_start_time = agv.current_time
                    op.f_delivery_arrive = (
                        op.f_start_time
                        + self.tt_f[agv.location][DEFAULT_LOCATION]
                        + self.tt_f[DEFAULT_LOCATION][op.f_delivery_location]
                    )
                    op.f_return_complete = op.f_delivery_arrive + self.tt_f[op.f_delivery_location][DEFAULT_LOCATION]
                    agv.current_time = op.f_return_complete
                    agv.location = DEFAULT_LOCATION
                    agv.queue.popleft()

            # Phase 3: Machine
            for mach in list(active_machines):
                if not mach.queue:
                    active_machines.remove(mach); continue
                op = mach.queue[0]
                if (
                    op.w_delivery_complete is None
                    and op.prev_operation is not None
                    and op.prev_operation.assigned_machine_id == op.assigned_machine_id
                    and op.prev_operation.process_end is not None
                ):
                    t0 = float(op.prev_operation.process_end)
                    loc = op.assigned_machine_id.location
                    op.w_pickup_start = op.w_pickup_arrive = op.w_pickup_leave = t0
                    op.w_delivery_arrive = op.w_delivery_complete = t0
                    op.w_from_location = op.w_pickup_location = op.w_delivery_location = loc
                if op.w_delivery_complete is None or op.f_delivery_arrive is None:
                    continue
                if op.prev_operation and op.prev_operation.process_end is None:
                    continue
                start_time = max(mach.current_time, float(op.w_delivery_complete), float(op.f_delivery_arrive))

                if op.process_start is None:
                    op.process_start = start_time
                    op.process_end = start_time + self.proc_times[op.op_id][mach.id]
                    mach.current_time = op.process_end
                    mach.queue.remove(op)
                    completed_ops += 1

    # --------------------------
    # Partial simulation (prefix of OS)
    # --------------------------

    def simulate_partial(
        self,
        machine_seq: List[int],
        agv_w_seq: List[int],
        agv_f_seq: List[int],
        os_seq: List[int],
        prefix_len: Optional[int] = None,
        return_schedule: bool = False,
    ) -> Union[float, Tuple[float, List[Dict[str, Any]]]]:
        """
        Decode only the first `prefix_len` ops in OS (NEH-like partial makespan).
        Keeps complexity O(k) for op reset and assignment.

        Note:
        - Resources are reset fully (cheap, small count).
        - Only active ops in the prefix are reset & assigned.
        """
        if not os_seq:
            return (0.0, []) if return_schedule else 0.0

        k = len(os_seq) if prefix_len is None else max(0, min(prefix_len, len(os_seq)))
        if k == 0:
            return (0.0, []) if return_schedule else 0.0

        # Reset resources
        for m in self.machines:
            m.reset()
        for a in self.agvs_w:
            a.reset()
        for a in self.agvs_f:
            a.reset()

        # Collect active ops from OS prefix
        active_ops: List[Operation] = []
        for idx in os_seq[:k]:
            op = self.idx_to_op[idx]
            active_ops.append(op)

        # Reset active ops only
        for op in active_ops:
            op.reset()

        # Assign resources for active ops only
        for op in active_ops:
            i = self.op_id_to_idx[op.op_id]
            op.assigned_machine_id = self.machines[machine_seq[i] - 1]
            op.assigned_agv_w_id = self.agvs_w[agv_w_seq[i] - 1]
            op.assigned_agv_f_id = self.agvs_f[agv_f_seq[i] - 1]

        # Build precedence inside prefix + enqueue
        job_last: Dict[int, Operation] = {}
        for op in active_ops:
            op.prev_operation = job_last.get(op.job_id, None)
            job_last[op.job_id] = op

            op.assigned_machine_id.queue.append(op)
            if not (
                op.prev_operation is not None
                and op.prev_operation.assigned_machine_id == op.assigned_machine_id
            ):
                op.assigned_agv_w_id.queue.append(op)
            op.assigned_agv_f_id.queue.append(op)

        self._run_kernel(total_ops=k)

        makespan = 0.0
        for op in active_ops:
            if op.process_end is not None and op.process_end > makespan:
                makespan = op.process_end

        if not return_schedule:
            return makespan

        # Collect schedule only for active ops
        results: List[Dict[str, Any]] = []
        for op in active_ops:
            results.extend(self._op_to_schedule_items(op))
        return makespan, results

    # --------------------------
    # Schedule extraction
    # --------------------------
    def _block_stats(self):
        # 1) 建机器加工顺序（按 process_start 排）
        machine_ops = {}
        for op in self.op_list:
            m = op.assigned_machine_id.id
            machine_ops.setdefault(m, []).append(op)
        for m, ops in machine_ops.items():
            ops.sort(key=lambda x: x.process_start)

        # 2) 统计：occur 和 delay 两套
        occD = occB = occC = occA = 0
        sumD = sumB = sumC = sumA = 0.0

        for m, ops in machine_ops.items():
            prev_end = float(ops[0].process_end)
            for op in ops[1:]:
                # machine ready
                t_m = prev_end

                # 关键时刻
                t_w = float(op.w_delivery_complete)   # 工件到机
                t_f = float(op.f_delivery_arrive)     # 物料到机
                t0 = max(t_w, t_f)

                # A: machine wait after both ready
                a_block = max(0.0, float(op.process_start) - t0)

                # B/C: W/F relative to machine ready
                b_block = max(0.0, t_w - t_m)
                c_block = max(0.0, t_f - t_m)

                # D: precedence wait at pickup
                d_block = max(0.0, float(op.w_pickup_leave) - float(op.w_pickup_arrive))
                

                # 发生率（是否>0）
                if a_block > 0: occA += 1
                if b_block > 0: occB += 1
                if c_block > 0: occC += 1  
                if d_block > 0: occD += 1         

                # 延迟量
                sumA += a_block
                sumB += b_block
                sumC += c_block
                sumD += d_block

                prev_end = float(op.process_end)

        n = float(len(self.op_list)-1)
        # 发生率画像（不强制和为1）
        pA_occ, pB_occ, pC_occ, pD_occ = occA / n, occB / n, occC / n, occD / n

        # 延迟量画像（强制和为1，代表“等待时间构成”）
        total_delay = sumA + sumB + sumC + sumD + 1e-12
        pA_del, pB_del, pC_del, pD_del = sumA / total_delay, sumB / total_delay, sumC / total_delay, sumD / total_delay

        return (pA_occ, pB_occ, pC_occ, pD_occ), (pA_del, pB_del, pC_del, pD_del)

    def _op_to_schedule_items(self, op: Operation) -> List[Dict[str, Any]]:
        items = [{
            "device": f"M{op.assigned_machine_id.id}", "operation": op.op_id,
            "start_time": op.process_start, "end_time": op.process_end,
        }, {
            "device": f"AGV_W{op.assigned_agv_w_id.id}", "operation": op.op_id,
            "start_time": op.w_pickup_start, "pickup_arrive": op.w_pickup_arrive, "pickup_leave": op.w_pickup_leave,
            "delivery_arrive": op.w_delivery_arrive, "delivery_complete": op.w_delivery_complete,
            "from_location": op.w_from_location, "pickup_location": op.w_pickup_location, "delivery_location": op.w_delivery_location,
        }, {
            "device": f"AGV_F{op.assigned_agv_f_id.id}", "operation": op.op_id,
            "start_time": op.f_start_time, "delivery_arrive": op.f_delivery_arrive, "return_complete": op.f_return_complete,
            "from_location": op.f_from_location, "delivery_location": op.f_delivery_location, "return_location": op.f_return_location,
        }]
        return items

    def _collect_schedule_data(self) -> List[Dict[str, Any]]:
        return [item for op in self.op_list for item in self._op_to_schedule_items(op)]

    def _get_resource_predecessors(self):
        mach_prev, agvw_prev, agvf_prev = {}, {}, {}
        # 预分配空间提速
        mach_ops = {m.id: [] for m in self.machines}
        agvw_ops = {a.id: [] for a in self.agvs_w}
        agvf_ops = {a.id: [] for a in self.agvs_f}

        for op in self.op_list:
            if op.process_start is not None: mach_ops[op.assigned_machine_id.id].append(op)
            if op.w_pickup_start is not None and op.w_delivery_complete is not None:
                if float(op.w_delivery_complete) - float(op.w_pickup_start) > EPSILON:
                    agvw_ops[op.assigned_agv_w_id.id].append(op)
            if op.f_start_time is not None: agvf_ops[op.assigned_agv_f_id.id].append(op)

        for ops in mach_ops.values():
            ops.sort(key=lambda x: x.process_start)
            for i in range(1, len(ops)): mach_prev[ops[i].op_id] = ops[i-1]
        for ops in agvw_ops.values():
            ops.sort(key=lambda x: x.w_pickup_start)
            for i in range(1, len(ops)): agvw_prev[ops[i].op_id] = ops[i-1]
        for ops in agvf_ops.values():
            ops.sort(key=lambda x: x.f_start_time)
            for i in range(1, len(ops)): agvf_prev[ops[i].op_id] = ops[i-1]

        return mach_prev, agvw_prev, agvf_prev

    def find_critical_path(self) -> List[Operation]:
        if self._last_cp_ops is not None:
            return self._last_cp_ops

        if not self.op_list:
            return []
        last_op = max(
            (op for op in self.op_list if op.process_end is not None),
            key=lambda x: x.process_end,
            default=None,
        )
        if not last_op:
            return []

        mach_prev, agvw_prev, agvf_prev = self._get_resource_predecessors()
        critical_path, curr_op = [], last_op
        visited = set()

        def close(a: float | None, b: float | None) -> bool:
            if a is None or b is None:
                return False
            return abs(float(a) - float(b)) <= EPSILON

        while curr_op:
            if curr_op.op_id in visited:
                break
            visited.add(curr_op.op_id)
            critical_path.append(curr_op)
            t_start = float(curr_op.process_start)
            m_prev = mach_prev.get(curr_op.op_id)
            job_prev = curr_op.prev_operation
            w_prev = agvw_prev.get(curr_op.op_id)
            f_prev = agvf_prev.get(curr_op.op_id)

            if m_prev and close(t_start, m_prev.process_end):
                curr_op = m_prev
                continue

            if close(t_start, curr_op.w_delivery_complete):
                if job_prev and close(curr_op.w_pickup_leave, job_prev.process_end):
                    curr_op = job_prev
                    continue
                if w_prev and close(curr_op.w_pickup_start, w_prev.w_delivery_complete):
                    curr_op = w_prev
                    continue

            if close(t_start, curr_op.f_delivery_arrive):
                if f_prev and close(curr_op.f_start_time, f_prev.f_return_complete):
                    curr_op = f_prev
                    continue

            curr_op = None

        self._last_cp_ops = critical_path[::-1]
        return self._last_cp_ops
