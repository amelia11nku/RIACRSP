# -*- coding: utf-8 -*-
"""
Insertion-based Scheduler Decoder for AFAISP (Optimized Version)
Target: Flexible Job Shop Scheduling Problem with Operation Sequence Flexibility
        and Dual-Layer Transportation (Workpiece AGV_W + Material AGV_F).

优化说明 (Aligned with DES simulate.txt):
1. 数据结构：引入轻量级 Operation 和 Timeline 类并使用 __slots__ 预分配内存，杜绝 dict 的频繁创建。
2. 算力优化：Timeline 对象在 __init__ 中全局初始化并在 decode 中复用，_tt_w_cache 扁平化，支持 O(k) 复杂度的 decode_partial 局部评估。
3. 返回值对齐：decode 和 decode_partial 的返回值结构完全对齐 simulate 和 simulate_partial (支持 return_schedule, return_pblock 控制)。
4. 拓扑回溯：新增与 simulate.txt 等效的 _get_resource_predecessors、find_critical_path 和 _block_stats 函数。
5. 插入逻辑：依然保持机器端前向插入 (find_earliest_slot) 与 AGV 端尾部顺序追加 (SequentialTimeline) 的核心策略。
"""

import bisect
from typing import Dict, List, Optional, Tuple, Any, Union
from data.loader import Instance

DEFAULT_LOCATION = "Warehouse"
EPSILON = 1e-6

class TaskSlot:
    """Represents a busy interval on a resource timeline."""
    __slots__ = ("start", "end", "op_id", "type")
    def __init__(self, start: float, end: float, op_id: str, t_type: str = ""):
        self.start = start
        self.end = end
        self.op_id = op_id
        self.type = t_type

class Operation:
    """Stores all timestamps and states for a single operation."""
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
        self.assigned_machine_id = self.assigned_agv_w_id = self.assigned_agv_f_id = 0
        self.prev_operation = None
        self.w_pickup_start = self.w_pickup_arrive = self.w_pickup_leave = None
        self.w_delivery_arrive = self.w_delivery_complete = None
        self.w_from_location = self.w_pickup_location = self.w_delivery_location = None
        self.f_start_time = self.f_delivery_arrive = self.f_return_complete = None
        self.f_from_location = self.f_delivery_location = self.f_return_location = None
        self.process_start = self.process_end = None

class MachineTimeline:
    """Resource timeline supporting insertion-based scheduling for Machines."""
    __slots__ = ("id", "node", "busy")
    def __init__(self, m_id: int):
        self.id = m_id
        self.node = f"M{m_id}"
        self.busy: List[TaskSlot] = []

    def reset(self) -> None:
        self.busy.clear()

    def find_earliest_slot(self, release_time: float, duration: float) -> float:
        if not self.busy:
            return release_time
        t = release_time
        for slot in self.busy:
            if t + duration <= slot.start + EPSILON:
                return t
            t = max(t, slot.end)
        return t

    def add_task(self, release_time: float, duration: float, op_id: str) -> TaskSlot:
        start_time = self.find_earliest_slot(release_time, duration)
        end_time = start_time + duration
        new_slot = TaskSlot(start_time, end_time, op_id, "P")
        
        # Maintain sorted order
        starts = [s.start for s in self.busy]
        idx = bisect.bisect_left(starts, start_time)
        self.busy.insert(idx, new_slot)
        return new_slot

class SequentialTimeline:
    """For AGVs: tasks appended sequentially to preserve single-location state."""
    __slots__ = ("id", "busy", "location")
    def __init__(self, a_id: int):
        self.id = a_id
        self.busy: List[TaskSlot] = []
        self.location = DEFAULT_LOCATION

    def reset(self) -> None:
        self.busy.clear()
        self.location = DEFAULT_LOCATION

    @property
    def last_end_time(self) -> float:
        return self.busy[-1].end if self.busy else 0.0

    def add_task(self, release_time: float, duration: float, op_id: str, t_type: str) -> TaskSlot:
        s = max(float(release_time), self.last_end_time)
        e = s + float(duration)
        slot = TaskSlot(s, e, op_id, t_type)
        self.busy.append(slot)
        return slot

class decoder:
    """Insertion-based decoder aligned with AFAISP constraints."""

    def __init__(self, instance: Instance):
        self.num_jobs = instance.num_jobs
        self.num_machines = instance.num_machines
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.processing_times = instance.processing_times

        # Flatten transport matrices for O(1) lookups
        self._tt_w_cache = self._flatten_transport_matrix(instance.agv_w_transport_times)
        self._tt_f_cache = self._flatten_transport_matrix(instance.agv_f_transport_times)

        # Pre-allocate resources
        self.machines = [MachineTimeline(i + 1) for i in range(self.num_machines)]
        self.agvs_w = [SequentialTimeline(i + 1) for i in range(self.num_agvs_w)]
        self.agvs_f = [SequentialTimeline(i + 1) for i in range(self.num_agvs_f)]

        # Pre-allocate Operations
        self.operations: Dict[str, Operation] = {}
        self.op_list: List[Operation] = []
        self.op_id_to_idx: Dict[str, int] = {}
        self.idx_to_op: Dict[int, Operation] = {}

        idx = 0
        counter = 1
        for job_id in range(1, self.num_jobs + 1):
            num_ops = instance.job_operations[job_id]
            for seq in range(1, num_ops + 1):
                op_id = f"o{job_id}_{seq}"
                op = Operation(op_id, job_id, seq)
                self.operations[op_id] = op
                self.op_list.append(op)
                self.op_id_to_idx[op_id] = idx
                self.idx_to_op[counter] = op
                idx += 1
                counter += 1

        self.num_ops = len(self.op_list)
        self._last_cp_ops: Optional[List[Operation]] = None

    @staticmethod
    def _flatten_transport_matrix(matrix: Dict[str, Dict[str, float]]) -> Dict[Tuple[str, str], float]:
        flat: Dict[Tuple[str, str], float] = {}
        if not matrix: return flat
        for s, dests in matrix.items():
            for t, v in dests.items():
                flat[(s, t)] = float(v)
        return flat

    def _get_travel_time(self, agv_type: str, frm: str, to: str) -> float:
        if frm == to:
            return 0.0
        if agv_type == "W":
            cache = self._tt_w_cache
            label = "AGV_W"
        elif agv_type == "F":
            cache = self._tt_f_cache
            label = "AGV_F"
        else:
            raise ValueError(f"Unknown AGV type: {agv_type!r}")
        try:
            return cache[(frm, to)]
        except KeyError as exc:
            raise KeyError(f"Missing {label} transport time: {frm!r} -> {to!r}") from exc

    # --------------------------
    # Base Decode (Matches simulate return interface) 
    # --------------------------
    def decode(self, MS: List[int], TW: List[int], TF: List[int], OS: List[int],
               f_auto_return: bool = True, return_schedule: bool = False, return_pblock: bool = False) -> Union[float, Tuple[Any, ...]]:
        
        self._last_cp_ops = None
        for m in self.machines: m.reset()
        for a in self.agvs_w: a.reset()
        for a in self.agvs_f: a.reset()
        for op in self.op_list: op.reset()

        for i, op in enumerate(self.op_list):
            op.assigned_machine_id = int(MS[i])
            op.assigned_agv_w_id = int(TW[i])
            op.assigned_agv_f_id = int(TF[i])

        job_last: Dict[int, Operation] = {}

        for os_idx in OS:
            op = self.idx_to_op[os_idx]
            op.prev_operation = job_last.get(op.job_id, None)
            job_last[op.job_id] = op

            self._schedule_single_op(op, f_auto_return)

        makespan = max((op.process_end for op in self.op_list if op.process_end is not None), default=0.0)

        # Output interface aligns with simulate.txt 
        if not return_schedule:
            return makespan
        if not return_pblock:
            return makespan, self._collect_schedule_data()
            
        occ_stats, delay_stats = self._block_stats()
        return makespan, self._collect_schedule_data(), occ_stats, delay_stats

    # --------------------------
    # Partial Decode (O(k) Complexity) [cite: 98-106]
    # --------------------------
    def decode_partial(self, MS: List[int], TW: List[int], TF: List[int], OS: List[int],
                       prefix_len: Optional[int] = None, f_auto_return: bool = True, return_schedule: bool = False) -> Union[float, Tuple[float, List[Dict[str, Any]]]]:
        """Decode only the first `prefix_len` ops in OS."""
        if not OS: return (0.0, []) if return_schedule else 0.0

        k = len(OS) if prefix_len is None else max(0, min(prefix_len, len(OS)))
        if k == 0: return (0.0, []) if return_schedule else 0.0

        for m in self.machines: m.reset()
        for a in self.agvs_w: a.reset()
        for a in self.agvs_f: a.reset()

        active_ops: List[Operation] = []
        for idx in OS[:k]:
            op = self.idx_to_op[idx]
            active_ops.append(op)

        job_last: Dict[int, Operation] = {}
        
        for op in active_ops:
            op.reset()
            i = self.op_id_to_idx[op.op_id]
            op.assigned_machine_id = int(MS[i])
            op.assigned_agv_w_id = int(TW[i])
            op.assigned_agv_f_id = int(TF[i])
            
            op.prev_operation = job_last.get(op.job_id, None)
            job_last[op.job_id] = op

            self._schedule_single_op(op, f_auto_return)

        makespan = max((op.process_end for op in active_ops if op.process_end is not None), default=0.0)

        # Output interface aligns with simulate_partial 
        if not return_schedule:
            return makespan
        
        results = []
        for op in active_ops:
            results.extend(self._op_to_schedule_items(op))
        return makespan, results

    # --------------------------
    # Core Insertion Logic
    # --------------------------
    def _schedule_single_op(self, op: Operation, f_auto_return: bool) -> None:
        mach = self.machines[op.assigned_machine_id - 1]
        agv_w = self.agvs_w[op.assigned_agv_w_id - 1]
        agv_f = self.agvs_f[op.assigned_agv_f_id - 1]

        prev_finish = float(op.prev_operation.process_end) if op.prev_operation and op.prev_operation.process_end else 0.0
        prev_loc = f"M{op.prev_operation.assigned_machine_id}" if op.prev_operation else DEFAULT_LOCATION
        
        # 1. Phase W: Workpiece transport
        is_same_machine = (op.prev_operation is not None and op.prev_operation.assigned_machine_id == op.assigned_machine_id)

        if is_same_machine:
            op.w_pickup_start = op.w_pickup_arrive = op.w_pickup_leave = prev_finish
            op.w_delivery_arrive = op.w_delivery_complete = prev_finish
            op.w_from_location = op.w_pickup_location = prev_loc
            op.w_delivery_location = mach.node
        else:
            from_loc = agv_w.location
            pickup_loc = prev_loc
            delivery_loc = mach.node

            t_empty = self._get_travel_time("W", from_loc, pickup_loc)
            t_loaded = self._get_travel_time("W", pickup_loc, delivery_loc)

            empty_start = agv_w.last_end_time
            if t_empty > EPSILON:
                agv_w.add_task(empty_start, t_empty, op.op_id, "W_EMPTY")
            
            op.w_pickup_start = empty_start
            op.w_pickup_arrive = empty_start + t_empty
            op.w_pickup_leave = max(op.w_pickup_arrive, prev_finish)
            
            wait_dur = op.w_pickup_leave - op.w_pickup_arrive
            if wait_dur > EPSILON:
                agv_w.add_task(op.w_pickup_arrive, wait_dur, op.op_id, "W_WAIT")
            
            agv_w.add_task(op.w_pickup_leave, t_loaded, op.op_id, "W_LOADED")

            op.w_delivery_arrive = op.w_delivery_complete = op.w_pickup_leave + t_loaded
            op.w_from_location, op.w_pickup_location, op.w_delivery_location = from_loc, pickup_loc, delivery_loc
            agv_w.location = delivery_loc

        # 2. Phase F: Material feeding
        f_from = agv_f.location
        t_to_wh = self._get_travel_time("F", f_from, DEFAULT_LOCATION)
        t_wh_to_m = self._get_travel_time("F", DEFAULT_LOCATION, mach.node)
        t_return = self._get_travel_time("F", mach.node, DEFAULT_LOCATION) if f_auto_return else 0.0

        f_release = agv_f.last_end_time
        agv_f.add_task(f_release, t_to_wh + t_wh_to_m + t_return, op.op_id, "F_ALL")
        
        op.f_start_time = f_release
        op.f_delivery_arrive = f_release + t_to_wh + t_wh_to_m
        op.f_return_complete = f_release + t_to_wh + t_wh_to_m + t_return
        
        op.f_from_location = f_from
        op.f_delivery_location = mach.node
        op.f_return_location = DEFAULT_LOCATION if f_auto_return else mach.node
        
        agv_f.location = op.f_return_location

        # 3. Phase P: Processing (Insertion mechanism)
        proc_dur = float(self.processing_times[op.op_id][op.assigned_machine_id])
        earliest_start = max(prev_finish, op.w_delivery_arrive, op.f_delivery_arrive)

        p_slot = mach.add_task(earliest_start, proc_dur, op.op_id)
        
        op.process_start = p_slot.start
        op.process_end = p_slot.end

    # --------------------------
    # Output formatting & Metrics
    # --------------------------
    def _op_to_schedule_items(self, op: Operation) -> List[Dict[str, Any]]:
        items = [{
            "device": f"M{op.assigned_machine_id}", "operation": op.op_id,
            "start_time": op.process_start, "end_time": op.process_end,
        }]
        
        if op.w_pickup_start is not None and op.w_delivery_complete is not None:
            items.append({
                "device": f"AGV_W{op.assigned_agv_w_id}", "operation": op.op_id,
                "start_time": op.w_pickup_start, "pickup_arrive": op.w_pickup_arrive, "pickup_leave": op.w_pickup_leave,
                "delivery_arrive": op.w_delivery_arrive, "delivery_complete": op.w_delivery_complete,
                "from_location": op.w_from_location, "pickup_location": op.w_pickup_location, "delivery_location": op.w_delivery_location,
            })
            
        if op.f_start_time is not None:
            items.append({
                "device": f"AGV_F{op.assigned_agv_f_id}", "operation": op.op_id,
                "start_time": op.f_start_time, "delivery_arrive": op.f_delivery_arrive, "return_complete": op.f_return_complete,
                "from_location": op.f_from_location, "delivery_location": op.f_delivery_location, "return_location": op.f_return_location,
            })
        return items

    def _collect_schedule_data(self) -> List[Dict[str, Any]]:
        return [item for op in self.op_list for item in self._op_to_schedule_items(op)]

    def _block_stats(self):
        machine_ops = {}
        for op in self.op_list:
            m = op.assigned_machine_id
            machine_ops.setdefault(m, []).append(op)
        for m, ops in machine_ops.items():
            ops.sort(key=lambda x: x.process_start)

        occA = occB = occC = occD = 0
        sumA = sumB = sumC = sumD = 0.0

        for m, ops in machine_ops.items():
            prev_end = float(ops[0].process_end)
            for op in ops[1:]:
                t_m = prev_end
                t_w = float(op.w_delivery_complete)
                t_f = float(op.f_delivery_arrive)
                t0 = max(t_w, t_f)

                a_block = max(0.0, float(op.w_pickup_leave) - float(op.w_pickup_arrive))
                b_block = max(0.0, t_w - t_m)
                c_block = max(0.0, t_f - t_m)
                d_block = max(0.0, float(op.process_start) - t0)

                if a_block > 0: occA += 1
                if b_block > 0: occB += 1
                if c_block > 0: occC += 1
                if d_block > 0: occD += 1

                sumA += a_block
                sumB += b_block
                sumC += c_block
                sumD += d_block

                prev_end = float(op.process_end)

        n = float(max(1, len(self.op_list) - 1))
        pA_occ, pB_occ, pC_occ, pD_occ = occA / n, occB / n, occC / n, occD / n
        total_delay = sumA + sumB + sumC + sumD + 1e-12
        pA_del, pB_del, pC_del, pD_del = sumA / total_delay, sumB / total_delay, sumC / total_delay, sumD / total_delay

        return (pA_occ, pB_occ, pC_occ, pD_occ), (pA_del, pB_del, pC_del, pD_del)

    # --------------------------
    # Critical Path (Optimized Extraction)
    # --------------------------
    def _get_resource_predecessors(self):
        mach_prev, agvw_prev, agvf_prev = {}, {}, {}
        
        # O(N) extraction natively utilizing pre-sorted timeline structures
        for mach in self.machines:
            for i in range(1, len(mach.busy)):
                mach_prev[mach.busy[i].op_id] = self.operations[mach.busy[i-1].op_id]
                
        for agvw in self.agvs_w:
            # Deduplicate operations in AGV_W since 1 op splits into empty/wait/loaded slots
            ops_in_agv = []
            for slot in agvw.busy:
                if not ops_in_agv or ops_in_agv[-1].op_id != slot.op_id:
                    ops_in_agv.append(self.operations[slot.op_id])
            for i in range(1, len(ops_in_agv)):
                agvw_prev[ops_in_agv[i].op_id] = ops_in_agv[i-1]
                
        for agvf in self.agvs_f:
            for i in range(1, len(agvf.busy)):
                agvf_prev[agvf.busy[i].op_id] = self.operations[agvf.busy[i-1].op_id]

        return mach_prev, agvw_prev, agvf_prev

    def find_critical_path(self) -> List[Operation]:
        if self._last_cp_ops is not None:
            return self._last_cp_ops

        if not self.op_list: return []
        # 从最后完工的工序开始回溯
        last_op = max((op for op in self.op_list if op.process_end is not None), key=lambda x: x.process_end, default=None)
        if not last_op: return []

        mach_prev, agvw_prev, agvf_prev = self._get_resource_predecessors()
        critical_path = []
        curr_op = last_op
        visited = set() # 增加防循环绝对护城河

        while curr_op and curr_op.op_id not in visited:
            critical_path.append(curr_op)
            
            visited.add(curr_op.op_id)

            t_start = float(curr_op.process_start) if curr_op.process_start is not None else 0.0
            m_prev = mach_prev.get(curr_op.op_id)
            
            t_mach = float(m_prev.process_end) if m_prev and m_prev.process_end is not None else 0.0
            t_w = float(curr_op.w_delivery_complete) if curr_op.w_delivery_complete is not None else 0.0
            t_f = float(curr_op.f_delivery_arrive) if curr_op.f_delivery_arrive is not None else 0.0

            # 找出真正卡住开工时间的瓶颈源 (避免浮点数造成的判断错位)
            bottleneck_time = max(t_mach, t_w, t_f)

            # 如果没有约束条件 (追溯到了起点)
            if bottleneck_time < EPSILON:
                break

            # 瓶颈 1：同机器的前序工序
            if abs(bottleneck_time - t_mach) < EPSILON and m_prev:
                curr_op = m_prev

            # 瓶颈 2：工件 AGV_W
            elif abs(bottleneck_time - t_w) < EPSILON and t_w > 0:
                trace_op = curr_op
                next_cp_op = None
                
                # 沿着 AGV_W 运输链一直往前找，直到找到拖累 AGV 的机器加工节点
                while trace_op:
                    job_prev = trace_op.prev_operation
                    t_job = float(job_prev.process_end) if job_prev and job_prev.process_end is not None else 0.0
                    w_leave = float(trace_op.w_pickup_leave) if trace_op.w_pickup_leave is not None else 0.0

                    # 是因为前序工序加工太慢，导致 AGV 在原地等吗？
                    if job_prev and abs(w_leave - t_job) < EPSILON:
                        next_cp_op = job_prev
                        break

                    # 如果不是，说明是 AGV 上一趟任务来晚了，继续沿着 AGV 历史回溯
                    w_prev = agvw_prev.get(trace_op.op_id)
                    if w_prev:
                        if w_prev.op_id not in visited:
                            critical_path.append(w_prev)
                            visited.add(w_prev.op_id)
                        trace_op = w_prev
                    else:
                        break # 追溯到了 AGV 的初始发车点

                # 将关键路径无缝平滑切回到机器约束上
                curr_op = next_cp_op

            # 瓶颈 3：物料 AGV_F
            elif abs(bottleneck_time - t_f) < EPSILON and t_f > 0:
                trace_op = curr_op
                # AGV_F 不受机器加工的影响，纯粹是一趟接一趟的顺序任务
                # 因此只需将其历史记录全部拉进关键路径并结束回溯
                while True:
                    f_prev = agvf_prev.get(trace_op.op_id)
                    if f_prev and f_prev.op_id not in visited:
                        critical_path.append(f_prev)
                        visited.add(f_prev.op_id)
                        trace_op = f_prev
                    else:
                        break
                curr_op = None # AGV_F 
                
            else:
                curr_op = None
                

        self._last_cp_ops = critical_path[::-1]
        return self._last_cp_ops
    
    def find_critical_path1(self) -> List[Operation]:
        if self._last_cp_ops is not None:
            return self._last_cp_ops

        if not self.op_list: return []
        last_op = max((op for op in self.op_list if op.process_end is not None), key=lambda x: x.process_end, default=None)
        if not last_op: return []

        mach_prev, agvw_prev, agvf_prev = self._get_resource_predecessors()
        critical_path, curr_op = [], last_op

        while curr_op:
            critical_path.append(curr_op)
            t_start = float(curr_op.process_start)
            m_prev = mach_prev.get(curr_op.op_id)
            t_mach = float(m_prev.process_end) if m_prev else 0.0
            t_w = float(curr_op.w_delivery_complete) if curr_op.w_delivery_complete is not None else 0.0
            t_f = float(curr_op.f_delivery_arrive) if curr_op.f_delivery_arrive is not None else 0.0

            if abs(t_start - t_mach) < EPSILON and m_prev:
                curr_op = m_prev
            elif abs(t_start - t_w) < EPSILON and t_w > 0:
                job_prev = curr_op.prev_operation
                t_job = float(job_prev.process_end) if job_prev else 0.0
                w_prev = agvw_prev.get(curr_op.op_id)
                t_agvw = float(w_prev.w_delivery_complete) if w_prev else 0.0
                
                if job_prev and abs(float(curr_op.w_pickup_leave) - t_job) < EPSILON:
                    curr_op = job_prev
                elif w_prev and abs(float(curr_op.w_pickup_start) - t_agvw) < EPSILON:
                    curr_op = w_prev
                else: curr_op = None
            elif abs(t_start - t_f) < EPSILON and t_f > 0:
                f_prev = agvf_prev.get(curr_op.op_id)
                if f_prev and abs(float(curr_op.f_start_time) - float(f_prev.f_return_complete)) < EPSILON:
                    curr_op = f_prev
                else: curr_op = None
            else:
                curr_op = None
        self._last_cp_ops = critical_path[::-1]
        return self._last_cp_ops

    def check_critical_path(self, path: List['Operation']) -> Tuple[bool, List[str]]:
        """
        校验给定的关键路径是否连续无间隙 (slack ≈ 0)。
        """
        errors = []
        if not path:
            return False, ["Critical Path Check Failed: Path is empty."]

        eps = 1e-6

        # 1. 验证终点是否匹配 Makespan，起点是否为 0 
        makespan = max((float(op.process_end) for op in self.op_list if op.process_end is not None), default=0.0)
        if abs(float(path[-1].process_end) - makespan) > eps:
            errors.append(f"End Time Mismatch: Path ends at {path[-1].process_end}, but makespan is {makespan}")
        if abs(float(min(path[0].w_pickup_start, path[0].f_start_time)) - 0.0) > eps:
            errors.append(f"Start Time Mismatch: Path starts at {min(path[0].w_pickup_start, path[0].f_start_time)}, but should be 0.0")
        # 2. 验证路径上相邻节点的依赖紧凑性
        for i in range(len(path) - 1):
            op_prev = path[i]
            op_curr = path[i+1]
            
            is_valid_link = False
            
            # (A) 机器依赖：前工序在同机器完工，后工序立即开工
            if op_curr.assigned_machine_id == op_prev.assigned_machine_id:
                if abs(float(op_curr.process_start) - float(op_prev.process_end)) < eps:
                    is_valid_link = True
            
            # (B) 工件作业依赖：等同job的上道工序完工后，AGV_W立即取货 [cite: 46]
            if op_curr.prev_operation == op_prev:
                if abs(float(op_curr.w_pickup_leave) - float(op_prev.process_end)) < eps:
                    is_valid_link = True
            
            # (C) AGV_W依赖：等AGV_W前一个任务完成后，立即发车
            if op_curr.assigned_agv_w_id == op_prev.assigned_agv_w_id:
                if abs(float(op_curr.w_pickup_start) - float(op_prev.w_delivery_complete)) < eps:
                    is_valid_link = True
                    
            # (D) AGV_F依赖：等AGV_F回到仓库后，立即为当前工序发车 [cite: 52, 54]
            if op_curr.assigned_agv_f_id == op_prev.assigned_agv_f_id:
                if abs(float(op_curr.f_start_time) - float(op_prev.f_return_complete)) < eps:
                    is_valid_link = True
                    
            if not is_valid_link:
                errors.append(f"Slack Error: Disconnect between {op_prev.op_id} and {op_curr.op_id}")

        return (len(errors) == 0), errors
