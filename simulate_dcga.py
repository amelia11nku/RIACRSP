# -*- coding: utf-8 -*-
"""
Dynamic Discrete Event Simulation (Dynamic-DES) Decoder for AFAISP
Target: AFAISP (Flexible Job Shop + Dual-layer transport + job indivisibility)

Inputs:
- MS: machine assignment (aligned with natural operation order: op_index_1based - 1)
- OS: operation sequence (permutation of 1..num_operations), assumed precedence-feasible by caller
AGV assignments are decided online by heuristic (earliest delivery).

Aligned semantics (same as your verified insertion decoders):
- AGV_W: empty + wait + loaded, BUT if prev_mach == cur_mach -> NO W transport
- AGV_F: (from current loc -> WH) + (WH -> machine) + (machine -> WH if auto_return)
- Processing start = max(machine_free, prev_proc_end, W_delivery_arrive, F_delivery_arrive)
- AGVs are sequential resources with a single location state; no time-insertion allowed.

Feasibility checks aligned with finalized constraints:
(1) Resource non-overlap:
    - AGV_W: [start_time, delivery_complete]
    - AGV_F: [start_time, return_complete]
    - Machine: [start_time, end_time]
(2) Arrival-before-processing:
    - W_delivery_arrive <= M_start_time
    - F_delivery_arrive <= M_start_time
(3) Predecessor completion before W loaded departure:
    - prev_proc_end <= W_pickup_leave
(4) Job indivisibility:
    - processing segments within job do not overlap
    - loaded W segments [pickup_leave, delivery_arrive] within job do not overlap
    - processing segments and loaded W segments do not overlap
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any
from data.loader import Instance

DEFAULT_LOCATION = "Warehouse"
EPSILON = 1e-6


class Operation:
    """Lightweight operation data structure."""
    __slots__ = (
        "op_id", "job_id", "seq_in_job",
        "assigned_machine", "assigned_agv_w", "assigned_agv_f",
        "prev_operation",
        # W timestamps
        "w_pickup_start", "w_pickup_arrive", "w_pickup_leave",
        "w_delivery_arrive", "w_delivery_complete",
        "w_from_location", "w_pickup_location", "w_delivery_location",
        # F timestamps
        "f_start", "f_delivery_arrive", "f_return_complete",
        "f_from_location", "f_delivery_location", "f_return_location",
        # Processing timestamps
        "process_start", "process_end"
    )

    def __init__(self, op_id: str, job_id: int, seq_in_job: int):
        self.op_id = op_id
        self.job_id = job_id
        self.seq_in_job = seq_in_job
        self.reset()

    def reset(self) -> None:
        self.assigned_machine = None
        self.assigned_agv_w = None
        self.assigned_agv_f = None
        self.prev_operation = None

        self.w_pickup_start = None
        self.w_pickup_arrive = None
        self.w_pickup_leave = None
        self.w_delivery_arrive = None
        self.w_delivery_complete = None
        self.w_from_location = None
        self.w_pickup_location = None
        self.w_delivery_location = None

        self.f_start = None
        self.f_delivery_arrive = None
        self.f_return_complete = None
        self.f_from_location = None
        self.f_delivery_location = None
        self.f_return_location = None

        self.process_start = None
        self.process_end = None


class Resource:
    """Base resource class (sequential, single-location state)."""
    __slots__ = ("id", "current_time", "location")

    def __init__(self, r_id: int, initial_location: str):
        self.id = r_id
        self.location = initial_location
        self.current_time: float = 0.0

    def reset(self) -> None:
        self.current_time = 0.0


class Machine(Resource):
    __slots__ = ()

    def __init__(self, m_id: int):
        super().__init__(m_id, f"M{m_id}")

    def reset(self) -> None:
        super().reset()
        # location is static


class AGV(Resource):
    """
    AGV is a sequential resource with a single location state.
    We store travel-only stats if you need them, but they do not affect feasibility.
    """
    __slots__ = ("total_travel_time",)

    def __init__(self, a_id: int):
        super().__init__(a_id, DEFAULT_LOCATION)
        self.total_travel_time: float = 0.0

    def reset(self) -> None:
        super().reset()
        self.location = DEFAULT_LOCATION
        self.total_travel_time = 0.0


class Decoder:
    """Dynamic-DES decoder aligned with AFAISP."""

    def __init__(self, instance: Instance):
        self.instance = instance

        self.machines = [Machine(i + 1) for i in range(instance.num_machines)]
        self.agvs_w = [AGV(i + 1) for i in range(instance.num_agvs_w)]
        self.agvs_f = [AGV(i + 1) for i in range(instance.num_agvs_f)]

        # Pre-allocate operations in natural order: o{job}_{seq}
        self.op_list: List[Operation] = []
        self.idx_to_op: Dict[int, Operation] = {}  # 1-based index -> op

        idx = 1
        for job_id in range(1, instance.num_jobs + 1):
            num_ops = instance.job_operations[job_id]
            for seq in range(1, num_ops + 1):
                op_id = f"o{job_id}_{seq}"
                op = Operation(op_id, job_id, seq)
                self.op_list.append(op)
                self.idx_to_op[idx] = op
                idx += 1

        self.num_ops = len(self.op_list)

        # Cached data
        self.proc_times = instance.processing_times
        self.tt_w = instance.agv_w_transport_times
        self.tt_f = instance.agv_f_transport_times

    def reset(self) -> None:
        for m in self.machines:
            m.reset()
        for a in self.agvs_w:
            a.reset()
        for a in self.agvs_f:
            a.reset()
        for op in self.op_list:
            op.reset()

    # -------------------------
    # Safe travel time lookup
    # -------------------------
    def _tt(self, typ: str, frm: str, to: str) -> float:
        if frm == to:
            return 0.0
        if typ == "W":
            table = self.tt_w
            label = "AGV_W"
        elif typ == "F":
            table = self.tt_f
            label = "AGV_F"
        else:
            raise ValueError(f"Unknown transport type: {typ}")

        row = table.get(frm)
        if row is None:
            raise KeyError(f"Missing {label} transport-time row: {frm!r}")
        v = row.get(to)
        if v is None:
            raise KeyError(f"Missing {label} transport time: {frm!r} -> {to!r}")
        return float(v)

    def _validate_schedule_string(
        self,
        schedule_seq: List[int],
        op_index_to_id: Optional[Dict[int, str]] = None,
        priority_dict: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """
        Validate that OS is a complete permutation and respects precedence.

        Parameters are optional for compatibility with DCGA.py, which passes
        its own op-index map and priority dictionary during mutation.
        """
        if schedule_seq is None:
            raise ValueError("Schedule sequence is required.")
        if len(schedule_seq) != self.num_ops:
            raise ValueError(
                f"Schedule length mismatch: expected {self.num_ops}, got {len(schedule_seq)}."
            )

        try:
            schedule = [int(x) for x in schedule_seq]
        except (TypeError, ValueError) as exc:
            raise ValueError("Schedule sequence must contain integer operation indices.") from exc

        expected = set(range(1, self.num_ops + 1))
        counts = Counter(schedule)
        observed = set(counts)
        if observed != expected or any(v > 1 for v in counts.values()):
            missing = sorted(expected - observed)
            duplicate_or_invalid = sorted(
                x for x, count in counts.items() if count > 1 or x not in expected
            )
            raise ValueError(
                "Schedule must be a permutation of 1..num_operations. "
                f"Missing={missing}, duplicate_or_invalid={duplicate_or_invalid}"
            )

        idx_to_id = op_index_to_id or {i + 1: op.op_id for i, op in enumerate(self.op_list)}
        precedence = priority_dict if priority_dict is not None else self.instance.priority_dict
        position = {idx_to_id[int(op_idx)]: pos for pos, op_idx in enumerate(schedule)}

        for op_id, predecessors in precedence.items():
            if op_id not in position:
                raise ValueError(f"Operation {op_id!r} is absent from schedule.")
            for pred in predecessors:
                if pred not in position:
                    raise ValueError(f"Predecessor {pred!r} for {op_id!r} is absent from schedule.")
                if position[pred] > position[op_id]:
                    raise ValueError(f"Precedence violation: {pred!r} appears after {op_id!r}.")
        return True

    def _validate_machine_string(self, machine_seq: List[int]) -> bool:
        if machine_seq is None:
            raise ValueError("Machine sequence is required.")
        if len(machine_seq) != self.num_ops:
            raise ValueError(
                f"Machine sequence length mismatch: expected {self.num_ops}, got {len(machine_seq)}."
            )
        for i, (op, machine_id) in enumerate(zip(self.op_list, machine_seq), start=1):
            try:
                m_id = int(machine_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Machine id for operation index {i} must be an integer.") from exc
            if not (1 <= m_id <= len(self.machines)):
                raise ValueError(f"Machine id {m_id} for {op.op_id} is outside valid range.")
            if m_id not in self.proc_times.get(op.op_id, {}):
                raise ValueError(f"Machine M{m_id} cannot process operation {op.op_id}.")
        return True

    # -------------------------
    # Main simulation
    # -------------------------
    def simulate(
        self,
        machine_seq: List[int],   # MS (natural order)
        schedule_seq: List[int],  # OS (1..num_ops)
        return_schedule: bool = False,
        strategy: int = 1,
        return_agv_strings: bool = False,
    ) -> Union[
        float,
        Tuple[float, List[Dict[str, Any]]],
        Tuple[float, List[int], List[int]],
        Tuple[float, List[Dict[str, Any]], List[int], List[int]],
    ]:
        """
        One-pass decoding:
        - Traverse OS, link job predecessor by OS chain.
        - Decide AGV assignments online (earliest delivery).
        - Update resource states sequentially.

        strategy:
            1: if tie, pick fixed AGV id if available else min id
            2: if tie, pick min agv.current_time then min id
        """
        # self._validate_machine_string(machine_seq)
        # self._validate_schedule_string(schedule_seq)
        self.reset()

        # NEW: allocate assignment vectors only when needed (no extra overhead otherwise)
        agv_w_assign: Optional[List[int]] = [0] * self.num_ops if return_agv_strings else None
        agv_f_assign: Optional[List[int]] = [0] * self.num_ops if return_agv_strings else None

        # 1) Static machine assignment (MS aligned with natural op order)
        for i, op in enumerate(self.op_list):
            m_id = int(machine_seq[i])
            op.assigned_machine = self.machines[m_id - 1]

        # 2) Build job predecessor links in OS chain
        job_last_op: Dict[int, Operation] = {}

        fixed_agv_id_w = 1
        fixed_agv_id_f = 1

        for os_idx in schedule_seq:
            os_idx_int = int(os_idx)
            op = self.idx_to_op[os_idx_int]

            prev_op = job_last_op.get(op.job_id)
            op.prev_operation = prev_op
            job_last_op[op.job_id] = op

            mach = op.assigned_machine
            cur_mach_id = mach.id
            cur_mach_loc = mach.location

            # predecessor states (by OS chain)
            prev_proc_end = float(prev_op.process_end) if prev_op and prev_op.process_end is not None else 0.0
            prev_mach_id = prev_op.assigned_machine.id if prev_op else -1
            pickup_loc = prev_op.assigned_machine.location if prev_op else DEFAULT_LOCATION

            # -------------------------
            # Phase W (workpiece)
            # -------------------------
            if prev_mach_id == cur_mach_id:
                # Same machine => NO W transport; all W times = prev_proc_end
                op.assigned_agv_w = None
                op.w_pickup_start = prev_proc_end
                op.w_pickup_arrive = prev_proc_end
                op.w_pickup_leave = prev_proc_end
                op.w_delivery_arrive = prev_proc_end
                op.w_delivery_complete = prev_proc_end
                op.w_from_location = pickup_loc
                op.w_pickup_location = pickup_loc
                op.w_delivery_location = cur_mach_loc
                # IMPORTANT: do NOT update any AGV_W location/state
                if return_agv_strings:
                    agv_w_assign[os_idx_int - 1] = 1
            else:
                agv_w = self._select_agv_w_onepass(
                    pickup_loc=pickup_loc,
                    delivery_loc=cur_mach_loc,
                    job_ready_time=prev_proc_end,
                    strategy=strategy,
                    fixed_agv_id=fixed_agv_id_w,
                )
                op.assigned_agv_w = agv_w

                if return_agv_strings:
                    agv_w_assign[os_idx_int - 1] = int(agv_w.id)

                # empty segment starts when AGV is available (sequential)
                w_start = agv_w.current_time
                t_empty = self._tt("W", agv_w.location, pickup_loc)
                pickup_arrive = w_start + t_empty

                # wait at pickup until predecessor completes (workpiece ready)
                pickup_leave = max(pickup_arrive, prev_proc_end)

                # loaded segment
                t_loaded = self._tt("W", pickup_loc, cur_mach_loc)
                delivery_arrive = pickup_leave + t_loaded

                op.w_pickup_start = w_start
                op.w_pickup_arrive = pickup_arrive
                op.w_pickup_leave = pickup_leave
                op.w_delivery_arrive = delivery_arrive
                op.w_delivery_complete = delivery_arrive
                op.w_from_location = agv_w.location
                op.w_pickup_location = pickup_loc
                op.w_delivery_location = cur_mach_loc

                # update AGV_W state AFTER delivery
                agv_w.current_time = delivery_arrive
                agv_w.total_travel_time += (t_empty + t_loaded)
                agv_w.location = cur_mach_loc

            # -------------------------
            # Phase F (material)
            # -------------------------
            
            agv_f = self._select_agv_f_onepass(
                delivery_loc=cur_mach_loc,
                strategy=strategy,
                fixed_agv_id=fixed_agv_id_f,
            )
            op.assigned_agv_f = agv_f

            if return_agv_strings:
                agv_f_assign[os_idx_int - 1] = int(agv_f.id)

            f_start = agv_f.current_time

            f_from = agv_f.location
            t_to_wh = self._tt("F", f_from, DEFAULT_LOCATION)
            t_wh_to_m = self._tt("F", DEFAULT_LOCATION, cur_mach_loc)
            f_delivery_arrive = f_start + t_to_wh + t_wh_to_m

            t_ret = self._tt("F", cur_mach_loc, DEFAULT_LOCATION)
            f_return_complete = f_delivery_arrive + t_ret

            op.f_start = f_start
            op.f_delivery_arrive = f_delivery_arrive
            op.f_return_complete = f_return_complete
            op.f_from_location = f_from
            op.f_delivery_location = cur_mach_loc
            op.f_return_location = DEFAULT_LOCATION

            # update AGV_F at end
            agv_f.current_time = f_return_complete
            agv_f.total_travel_time += (t_to_wh + t_wh_to_m + t_ret)
            agv_f.location = DEFAULT_LOCATION

            # -------------------------
            # Phase P (processing)
            # -------------------------
            # start constrained by machine, precedence, and both arrivals
            p_start = max(
                mach.current_time,
                prev_proc_end,
                float(op.w_delivery_arrive) if op.w_delivery_arrive is not None else 0.0,
                float(op.f_delivery_arrive) if op.f_delivery_arrive is not None else 0.0,
            )

            duration = float(self.proc_times[op.op_id][cur_mach_id])
            op.process_start = p_start
            op.process_end = p_start + duration

            mach.current_time = op.process_end

        # Makespan (production completion only; if you want include AGV return, add them)
        makespan = 0.0
        for op in self.op_list:
            if op.process_end is not None and op.process_end > makespan:
                makespan = op.process_end

        if return_schedule:
            schedule = self._collect_schedule_data()
            if return_agv_strings:
                return makespan, schedule, agv_w_assign, agv_f_assign
            return makespan, schedule

        if return_agv_strings:
            return makespan, agv_w_assign, agv_f_assign

        return makespan

    # -------------------------
    # Schedule export
    # -------------------------
    def _collect_schedule_data(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for op in self.op_list:
            op_id = op.op_id

            # Machine
            results.append({
                "device": f"M{op.assigned_machine.id}",
                "operation": op_id,
                "start_time": op.process_start,
                "end_time": op.process_end,
            })

            # AGV_W (only if used)
            if op.assigned_agv_w is not None:
                results.append({
                    "device": f"AGV_W{op.assigned_agv_w.id}",
                    "operation": op_id,
                    "start_time": op.w_pickup_start,
                    "pickup_arrive": op.w_pickup_arrive,
                    "pickup_leave": op.w_pickup_leave,
                    "delivery_arrive": op.w_delivery_arrive,
                    "delivery_complete": op.w_delivery_complete,
                    "from_location": op.w_from_location,
                    "pickup_location": op.w_pickup_location,
                    "delivery_location": op.w_delivery_location,
                })

            # AGV_F (always used)
            if op.assigned_agv_f is not None:
                results.append({
                    "device": f"AGV_F{op.assigned_agv_f.id}",
                    "operation": op_id,
                    "start_time": op.f_start,
                    "delivery_arrive": op.f_delivery_arrive,
                    "return_complete": op.f_return_complete,
                    "from_location": op.f_from_location,
                    "delivery_location": op.f_delivery_location,
                    "return_location": op.f_return_location,
                })

        return results

    # -------------------------
    # AGV selection heuristics
    # -------------------------
    def _select_agv_w_onepass(
        self,
        pickup_loc: str,
        delivery_loc: str,
        job_ready_time: float,
        strategy: int,
        fixed_agv_id: int = 1,
    ) -> AGV:
        """
        Choose AGV_W with earliest feasible delivery arrival.
        Feasible delivery considers waiting for job_ready_time at pickup.
        """
        candidates: List[Tuple[float, float, int, AGV]] = []
        for agv in self.agvs_w:
            t_empty = self._tt("W", agv.location, pickup_loc)
            pickup_arrive = agv.current_time + t_empty
            pickup_leave = max(pickup_arrive, job_ready_time)
            t_loaded = self._tt("W", pickup_loc, delivery_loc)
            delivery_arrive = pickup_leave + t_loaded

            candidates.append((delivery_arrive, agv.current_time, agv.id, agv))

        candidates.sort(key=lambda x: x[0])
        best_time = candidates[0][0]
        tied = [c for c in candidates if abs(c[0] - best_time) <= EPSILON]

        if strategy == 1:
            for _, _, agv_id, agv in tied:
                if agv_id == fixed_agv_id:
                    return agv
            return min(tied, key=lambda x: x[2])[3]
        else:
            return min(tied, key=lambda x: (x[1], x[2]))[3]

    def _select_agv_f_onepass(
        self,
        delivery_loc: str,
        strategy: int,
        fixed_agv_id: int = 1,
    ) -> AGV:
        """
        Choose AGV_F with earliest material delivery arrival at machine.
        (Return does not affect processing feasibility, but affects AGV availability; we still simulate it.)
        """
        candidates: List[Tuple[float, float, int, AGV]] = []
        for agv in self.agvs_f:
            t_to_wh = self._tt("F", agv.location, DEFAULT_LOCATION)
            arrive_wh = agv.current_time + t_to_wh
            t_wh_to_m = self._tt("F", DEFAULT_LOCATION, delivery_loc)
            delivery_arrive = arrive_wh + t_wh_to_m

            candidates.append((delivery_arrive, agv.current_time, agv.id, agv))

        candidates.sort(key=lambda x: x[0])
        best_time = candidates[0][0]
        tied = [c for c in candidates if abs(c[0] - best_time) <= EPSILON]

        if strategy == 1:
            for _, _, agv_id, agv in tied:
                if agv_id == fixed_agv_id:
                    return agv
            return min(tied, key=lambda x: x[2])[3]
        else:
            return min(tied, key=lambda x: (x[1], x[2]))[3]

    # -------------------------
    # Feasibility check (aligned)
    # -------------------------
    def check_feasibility(self, schedule_results: List[Dict[str, Any]], OS: List[int]) -> Tuple[bool, List[str]]:
        if not OS:
            return False, ["OS required"]
        if not schedule_results:
            return False, ["schedule_results required"]

        eps = EPSILON
        errors: List[str] = []

        # Build OS predecessor map: op_id -> prev op_id in same job
        pred_map: Dict[str, Optional[str]] = {}
        last_by_job: Dict[int, str] = {}
        for idx in OS:
            op = self.idx_to_op[int(idx)]
            j = op.job_id
            pred_map[op.op_id] = last_by_job.get(j, None)
            last_by_job[j] = op.op_id

        # Resource intervals
        resource_intervals: Dict[str, List[Tuple[float, float, str]]] = {}

        # Per-op timings
        op_proc: Dict[str, Tuple[float, float]] = {}
        op_w: Dict[str, Dict[str, float]] = {}
        op_f: Dict[str, Dict[str, float]] = {}

        # Job indivisibility
        job_proc: Dict[int, List[Tuple[float, float, str]]] = {}
        job_w_loaded: Dict[int, List[Tuple[float, float, str]]] = {}

        def add_res(dev: str, s: float, e: float, op_id: str) -> None:
            resource_intervals.setdefault(dev, []).append((s, e, op_id))

        # Parse schedule
        for item in schedule_results:
            dev = item.get("device")
            op_id = item.get("operation")
            if not dev or not op_id:
                continue

            # job id from our op format o{job}_{seq}
            try:
                job = int(op_id.split("_", 1)[0][1:])
            except Exception:
                job = 0

            if dev.startswith("M"):
                s = float(item["start_time"])
                e = float(item["end_time"])
                op_proc[op_id] = (s, e)
                add_res(dev, s, e, op_id)
                job_proc.setdefault(job, []).append((s, e, op_id))

            elif dev.startswith("AGV_W"):
                s = float(item["start_time"])
                pickup_leave = float(item["pickup_leave"])
                delivery_arrive = float(item["delivery_arrive"])
                delivery_complete = float(item.get("delivery_complete", delivery_arrive))

                op_w[op_id] = {
                    "start_time": s,
                    "pickup_leave": pickup_leave,
                    "delivery_arrive": delivery_arrive,
                    "delivery_complete": delivery_complete,
                }
                # resource occupancy
                if delivery_complete - s >= eps:
                    add_res(dev, s, delivery_complete, op_id)

                # loaded segment for job indivisibility
                if delivery_arrive - pickup_leave >= eps:
                    job_w_loaded.setdefault(job, []).append((pickup_leave, delivery_arrive, op_id))

            elif dev.startswith("AGV_F"):
                s = float(item["start_time"])
                delivery_arrive = float(item["delivery_arrive"])
                return_complete = float(item["return_complete"])

                op_f[op_id] = {
                    "start_time": s,
                    "delivery_arrive": delivery_arrive,
                    "return_complete": return_complete,
                }
                add_res(dev, s, return_complete, op_id)

        # (1) Resource non-overlap
        for dev, ints in resource_intervals.items():
            ints.sort(key=lambda x: x[0])
            for i in range(len(ints) - 1):
                s1, e1, op1 = ints[i]
                s2, e2, op2 = ints[i + 1]
                if e1 > s2 + eps:
                    errors.append(f"[资源冲突] {dev}: {op1} overlaps {op2}")

        # (2) Arrival-before-processing
        for op_id, (ps, pe) in op_proc.items():
            if op_id in op_w and ps < op_w[op_id]["delivery_arrive"] - eps:
                errors.append(f"[到达约束] {op_id}: process starts before W delivery")
            if op_id in op_f and ps < op_f[op_id]["delivery_arrive"] - eps:
                errors.append(f"[到达约束] {op_id}: process starts before F delivery")

        # (3) Predecessor completion before W pickup_leave
        for op_id, prev_id in pred_map.items():
            if prev_id is None:
                continue
            if op_id not in op_w:
                # same-machine or no W record -> OK
                continue
            if prev_id not in op_proc:
                continue
            prev_end = op_proc[prev_id][1]
            if op_w[op_id]["pickup_leave"] < prev_end - eps:
                errors.append(f"[前序-运输] {op_id}: pickup_leave < {prev_id}.process_end")

        # (4) Job indivisibility
        def check_no_overlap(ints: List[Tuple[float, float, str]], label: str) -> None:
            ints.sort(key=lambda x: x[0])
            for i in range(len(ints) - 1):
                s1, e1, op1 = ints[i]
                s2, e2, op2 = ints[i + 1]
                if e1 > s2 + eps:
                    errors.append(f"[作业内冲突] {label}: {op1} overlaps {op2}")

        for j, ints in job_proc.items():
            check_no_overlap(ints, f"Job{j} Processing")
        for j, ints in job_w_loaded.items():
            check_no_overlap(ints, f"Job{j} Loaded-W")

        # proc vs loaded-W (two-pointer)
        for j in set(list(job_proc.keys()) + list(job_w_loaded.keys())):
            p_list = job_proc.get(j, [])
            w_list = job_w_loaded.get(j, [])
            if not p_list or not w_list:
                continue

            p_sorted = sorted(p_list, key=lambda x: x[0])
            w_sorted = sorted(w_list, key=lambda x: x[0])

            pi = 0
            wi = 0
            while pi < len(p_sorted) and wi < len(w_sorted):
                ps, pe, pop = p_sorted[pi]
                ws, we, wop = w_sorted[wi]

                if max(ps, ws) < min(pe, we) - eps:
                    errors.append(f"[作业内冲突] Job{j}: proc({pop}) overlaps loadedW({wop})")

                if pe <= we + eps:
                    pi += 1
                else:
                    wi += 1

        return (len(errors) == 0), errors
