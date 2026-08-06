# -*- coding: utf-8 -*-
"""
Insertion-based decoder (IDPSM resource selection) aligned with AFAISP.

Keep the SAME input encoding structure:
    decode(MS, TS_W, TS_F, OS, ...)

Key AFAISP alignment (same as our confirmed DES semantics):
- AGV_W: two segments
    (1) Empty:  AGV_W.loc -> pickup_node (prev op machine node or Warehouse)
    (2) Loaded: pickup_node -> dest_machine_node
  Empty can start when AGV is available; Loaded can start only after:
      - empty finished
      - predecessor processing finished (workpiece ready)
  If predecessor machine == current machine: NO W transport; all W timestamps = prev_finish.

- AGV_F: Warehouse -> Machine (delivery) -> Warehouse (return)
  Start when AGV_F is available. Early delivery is allowed.

- Machine start:
    start = max(machine_available, prev_finish, W_delivery_arrive, F_delivery_arrive)

Feasibility check aligned with finalized constraints:
(1) Resource non-overlap:
    - AGV_W: [start_time, delivery_complete]
    - AGV_F: [start_time, return_complete]
    - Machine: [start_time, end_time]
(2) Arrival-before-processing:
    - W_delivery_arrive <= M_start_time
    - F_delivery_arrive <= M_start_time
(3) Predecessor completion before W loaded departure:
    - M_pre_end <= W_pickup_leave
(4) Job indivisibility:
    - Processing segments within job do not overlap
    - Loaded W segments [pickup_leave, delivery_arrive] within job do not overlap
    - Processing segments and loaded W segments do not overlap

Performance / memory:
- No additional heavy structures.
- Job overlap checks are O(k log k) per job (sort) + O(k) two-pointer merges.
- Keeps your Timeline style and IDPSM selection; avoids AGV "time-travel reordering"
  by using release = timeline.last_end_time so AGV task order is consistent with position update.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import bisect
from data.loader import Instance

DEFAULT_LOCATION = "Warehouse"
EPSILON = 1e-6


@dataclass
class TaskSlot:
    """Represents a busy interval on a resource timeline."""
    __slots__ = ["start", "end", "meta"]
    start: float
    end: float
    meta: dict


class Timeline:
    """
    Resource timeline supporting insertion in gaps.
    """
    __slots__ = ["busy"]

    def __init__(self) -> None:
        self.busy: List[TaskSlot] = []

    def find_earliest_slot(self, release_time: float, duration: float) -> float:
        """
        Find earliest start time >= release_time where [start, start+duration] fits
        without overlapping existing busy slots.
        """
        if not self.busy:
            return release_time

        t = release_time
        for slot in self.busy:
            if t + duration <= slot.start:
                return t
            t = max(t, slot.end)
        return t

    def add_task(self, release_time: float, duration: float, meta: Dict[str, Any]) -> TaskSlot:
        """
        Insert a task at earliest feasible time >= release_time.
        """
        start_time = self.find_earliest_slot(release_time, duration)
        end_time = start_time + duration
        new_slot = TaskSlot(start_time, end_time, meta)

        # Maintain sorted by start time (small list typical)
        starts = [s.start for s in self.busy]
        idx = bisect.bisect_left(starts, start_time)
        self.busy.insert(idx, new_slot)
        return new_slot

    @property
    def last_end_time(self) -> float:
        return self.busy[-1].end if self.busy else 0.0

class SequentialTimeline(Timeline):
    """
    For AGVs: tasks must be appended in time order (no insertion),
    to preserve single-location state consistency.
    """
    __slots__ = ()

    def add_task(self, release_time: float, duration: float, meta: Dict[str, Any]) -> TaskSlot:
        s = max(float(release_time), self.last_end_time)
        e = s + float(duration)
        slot = TaskSlot(s, e, meta)
        self.busy.append(slot)  # always append
        return slot

class decoder:
    def __init__(self, instance: Instance):
        self.num_jobs = instance.num_jobs
        self.num_agvs_w = instance.num_agvs_w
        self.num_agvs_f = instance.num_agvs_f
        self.num_machines = instance.num_machines
        self.num_operations = instance.num_operations
        self.job_operations = instance.job_operations
        self.processing_times = instance.processing_times
        self.priority_dict = instance.priority_dict
        self.agv_w_transport_times = instance.agv_w_transport_times
        self.agv_f_transport_times = instance.agv_f_transport_times

        self.operations_list = [
            f"o{job_id}_{op_seq}"
            for job_id in range(1, self.num_jobs + 1)
            for op_seq in range(1, self.job_operations[job_id] + 1)
        ]

        # static job id cache
        self._op_static_info: Dict[str, int] = {}
        for op_id in self.operations_list:
            try:
                body = op_id[1:]
                job_s, _ = body.split("_", 1)
                job = int(job_s)
            except Exception:
                job = 0
            self._op_static_info[op_id] = job

        self.op_id_to_index_map = {op_id: i + 1 for i, op_id in enumerate(self.operations_list)}
        self.op_index_to_id_map = {i + 1: op_id for i, op_id in enumerate(self.operations_list)}

        self._machine_node_cache = {m: f"M{m}" for m in range(1, self.num_machines + 1)}

    # -------------------------
    # OS helper
    # -------------------------
    def _build_os_structures(self, OS: List[int]):
        op_index_to_id = self.op_index_to_id_map
        os_ops = [op_index_to_id[idx] for idx in OS]

        static_info = self._op_static_info
        op_to_job: Dict[str, int] = {}
        op_to_seq: Dict[str, int] = {}
        job_counters: Dict[int, int] = {}

        for op_id in os_ops:
            job = static_info.get(op_id, 0)
            op_to_job[op_id] = job
            job_counters[job] = job_counters.get(job, 0) + 1
            op_to_seq[op_id] = job_counters[job]

        last_pos_of_job: Dict[int, int] = {}
        prev_pos: List[Optional[int]] = [None] * len(OS)
        prev_op_os: Dict[str, Optional[str]] = {}

        for i, op_id in enumerate(os_ops):
            job = op_to_job[op_id]
            b = last_pos_of_job.get(job, None)
            prev_pos[i] = b
            prev_op_os[op_id] = os_ops[b] if b is not None else None
            last_pos_of_job[job] = i

        return op_to_job, op_to_seq, os_ops, prev_pos, prev_op_os

    def _node_of_machine(self, m: int) -> str:
        return self._machine_node_cache.get(int(m), f"M{int(m)}")

    def _travel_time_lookup(self, matrix: Dict[str, Dict[str, float]], frm: str, to: str) -> float:
        if frm == to:
            return 0.0
        if matrix is None:
            raise KeyError("Transport-time matrix is missing.")
        row = matrix.get(frm)
        if row is None:
            raise KeyError(f"Missing transport-time row: {frm!r}")
        val = row.get(to)
        if val is None:
            raise KeyError(f"Missing transport time: {frm!r} -> {to!r}")
        return float(val)

    def _travel_time(self, agv_type: str, frm: str, to: str) -> float:
        if agv_type == "W":
            try:
                return self._travel_time_lookup(self.agv_w_transport_times, frm, to)
            except KeyError as exc:
                raise KeyError(f"Missing AGV_W transport time: {frm!r} -> {to!r}") from exc
        if agv_type == "F":
            try:
                return self._travel_time_lookup(self.agv_f_transport_times, frm, to)
            except KeyError as exc:
                raise KeyError(f"Missing AGV_F transport time: {frm!r} -> {to!r}") from exc
        raise ValueError(f"Unknown AGV type: {agv_type!r}")

    def _pick_idpsm(self, timelines: Dict[int, Timeline], candidates: List[int]) -> int:
        """
        IDPSM:
        - pick first idle timeline (empty)
        - otherwise pick smallest last_end_time
        """
        if not candidates:
            raise ValueError("IDPSM: candidates is empty.")

        best = candidates[0]
        best_end = float("inf")
        for c in candidates:
            tl = timelines[c]
            if not tl.busy:
                return c
            e = tl.last_end_time
            if e < best_end:
                best_end = e
                best = c
        return best

    # -------------------------
    # Decode (aligned)
    # -------------------------
    def decode(
        self,
        MS: List[Optional[int]],
        TS_W: List[Optional[int]],
        TS_F: List[Optional[int]],
        OS: List[int],
        warehouse_node: str = DEFAULT_LOCATION,
        f_auto_return: bool = True,
        return_schedule: bool = True,
    ) -> Dict:
        """
        Do NOT change input structure: MS/TS_W/TS_F are aligned with OS positions (same length).
        """
        op_to_job, op_order_seq, os_ops, prev_pos, prev_op_os = self._build_os_structures(OS)

        n_ops = len(OS)
        if not (len(MS) == n_ops == len(TS_W) == len(TS_F)):
            raise ValueError("编码长度不一致：MS/TS_W/TS_F/OS 长度必须相同（按 OS 位置对齐）")

        MS_fill = list(MS)
        TS_W_fill = list(TS_W)
        TS_F_fill = list(TS_F)

        # Resource timelines
        mach_tl = {m: Timeline() for m in range(1, self.num_machines + 1)}
        agv_w_tl = {a: SequentialTimeline() for a in range(1, self.num_agvs_w + 1)}
        agv_f_tl = {a: SequentialTimeline() for a in range(1, self.num_agvs_f + 1)}

        # AGV positions (updated at the end of each physical segment)
        agv_w_pos = {a: warehouse_node for a in range(1, self.num_agvs_w + 1)}
        agv_f_pos = {a: warehouse_node for a in range(1, self.num_agvs_f + 1)}

        # Job states (by OS predecessor)
        prev_proc_finish = {j: 0.0 for j in range(1, self.num_jobs + 1)}
        prev_machine_node = {j: warehouse_node for j in range(1, self.num_jobs + 1)}

        # Keep last machine id per job *in OS chain* (needed for same-machine check)
        last_machine_id = {j: -1 for j in range(1, self.num_jobs + 1)}

        # Per-op ready times
        w_delivery_time: Dict[str, float] = {}
        w_pickup_leave_time: Dict[str, float] = {}
        f_delivery_time: Dict[str, float] = {}

        # outputs
        segments_proc: List[Dict[str, Any]] = []
        schedule_results: List[Dict[str, Any]] = []

        # candidates
        all_agvs_w = list(range(1, self.num_agvs_w + 1))
        all_agvs_f = list(range(1, self.num_agvs_f + 1))

        proc_times = self.processing_times
        _pick = self._pick_idpsm
        _tt = self._travel_time
        _node = self._node_of_machine

        for pos, op_index in enumerate(OS):
            op_id = os_ops[pos]
            job = op_to_job[op_id]
            seq_in_os = op_order_seq[op_id]

            # -------- Machine assignment --------
            mach = MS_fill[pos]
            if mach is None:
                avail = list(proc_times.get(op_id, {}).keys())
                avail = [int(m) for m in avail]
                mach = _pick(mach_tl, avail)
                MS_fill[pos] = mach
            else:
                mach = int(mach)
            dest_node = _node(mach)

            # Predecessor info (by OS chain)
            prev_finish = float(prev_proc_finish[job])
            pickup_node = warehouse_node if prev_op_os[op_id] is None else prev_machine_node[job]
            prev_mach = last_machine_id[job]

            # -------- Phase W: Workpiece transport (two segments) --------
            if prev_mach == mach:
                # same machine => NO W transport; all timestamps = prev_finish
                TS_W_fill[pos] = 0
                w_delivery_time[op_id] = prev_finish
                w_pickup_leave_time[op_id] = prev_finish
            else:
                agv_w = TS_W_fill[pos]
                if agv_w is None:
                    agv_w = _pick(agv_w_tl, all_agvs_w)
                    TS_W_fill[pos] = agv_w
                agv_w = int(agv_w)

                from_loc = agv_w_pos[agv_w]
                t_empty = _tt("W", from_loc, pickup_node)
                t_loaded = _tt("W", pickup_node, dest_node)

                # (1) empty segment: release when AGV is available (use last_end_time to keep position consistent)
                empty_release = agv_w_tl[agv_w].last_end_time
                empty_slot = agv_w_tl[agv_w].add_task(
                    empty_release,
                    t_empty,
                    {"type": "W_EMPTY", "op_id": op_id, "job": job, "seq": seq_in_os, "to": pickup_node},
                )
                w_start = empty_slot.start
                pickup_arrive = empty_slot.end

                # (2) wait-on-pickup segment (block the AGV while waiting for the workpiece)
                # AGV must stay at pickup until the predecessor finishes.
                pickup_leave = max(pickup_arrive, prev_finish)
                wait_dur = pickup_leave - pickup_arrive
                if wait_dur > EPSILON:
                    _ = agv_w_tl[agv_w].add_task(
                        pickup_arrive,          # release exactly at pickup_arrive
                        wait_dur,
                        {"type": "W_WAIT", "op_id": op_id, "job": job, "seq": seq_in_os, "at": pickup_node},
                    )

                # (3) loaded segment (must start exactly at pickup_leave)
                loaded_slot = agv_w_tl[agv_w].add_task(
                    pickup_leave,              # release at pickup_leave
                    t_loaded,
                    {"type": "W_LOADED", "op_id": op_id, "job": job, "seq": seq_in_os, "to": dest_node},
                )
                delivery_arrive = loaded_slot.end

                # update AGV physical position after loaded delivery
                agv_w_pos[agv_w] = dest_node

                w_delivery_time[op_id] = delivery_arrive
                w_pickup_leave_time[op_id] = pickup_leave

                if return_schedule:
                    schedule_results.append({
                        "device": f"AGV_W{agv_w}",
                        "operation": op_id,
                        "start_time": w_start,
                        "pickup_arrive": pickup_arrive,
                        "pickup_leave": pickup_leave,
                        "delivery_arrive": delivery_arrive,
                        "delivery_complete": delivery_arrive,
                        "from_location": from_loc,
                        "pickup_location": pickup_node,
                        "delivery_location": dest_node,
                    })

            # -------- Phase F: Material feeding --------
            agv_f = TS_F_fill[pos]
            if agv_f is None:
                agv_f = _pick(agv_f_tl, all_agvs_f)
                TS_F_fill[pos] = agv_f
            agv_f = int(agv_f)

            f_from = agv_f_pos[agv_f]
            t_to_wh = _tt("F", f_from, warehouse_node)
            t_wh_to_m = _tt("F", warehouse_node, dest_node)
            t_return = _tt("F", dest_node, warehouse_node) if f_auto_return else 0.0

            # start when AGV is available (use last_end_time to keep position consistent)
            f_release = agv_f_tl[agv_f].last_end_time
            f_slot = agv_f_tl[agv_f].add_task(
                f_release,
                t_to_wh + t_wh_to_m + t_return,
                {"type": "F", "op_id": op_id, "job": job, "seq": seq_in_os, "to": dest_node},
            )

            f_start = f_slot.start
            f_arrive = f_start + t_to_wh + t_wh_to_m
            f_ret = f_slot.end

            # update AGV_F position at end
            agv_f_pos[agv_f] = warehouse_node if f_auto_return else dest_node
            f_delivery_time[op_id] = f_arrive

            if return_schedule:
                schedule_results.append({
                    "device": f"AGV_F{agv_f}",
                    "operation": op_id,
                    "start_time": f_start,
                    "delivery_arrive": f_arrive,
                    "return_complete": f_ret,
                    "from_location": f_from,
                    "delivery_location": dest_node,
                    "return_location": warehouse_node if f_auto_return else dest_node,
                })

            # -------- Phase P: Processing --------
            try:
                ptime = float(proc_times[op_id][mach])
            except Exception:
                ptime = 0.0

            rel = prev_finish
            wd = float(w_delivery_time.get(op_id, prev_finish))
            fd = float(f_delivery_time.get(op_id, 0.0))
            if wd > rel:
                rel = wd
            if fd > rel:
                rel = fd

            p_slot = mach_tl[mach].add_task(
                rel,
                ptime,
                {"type": "P", "op_id": op_id, "job": job, "seq": seq_in_os, "mach": mach},
            )

            # update job states
            prev_proc_finish[job] = p_slot.end
            prev_machine_node[job] = dest_node
            last_machine_id[job] = mach

            segments_proc.append({"end": p_slot.end})

            if return_schedule:
                schedule_results.append({
                    "device": f"M{mach}",
                    "operation": op_id,
                    "start_time": p_slot.start,
                    "end_time": p_slot.end,
                })

        makespan = max((s["end"] for s in segments_proc), default=0.0)

        result = {
            "proc": segments_proc,
            "makespan": makespan,
            "filled_MS": MS_fill,
            "filled_TS_W": TS_W_fill,
            "filled_TS_F": TS_F_fill,
        }
        if return_schedule:
            result["schedule_results"] = schedule_results
        return result

    # -------------------------
    # Feasibility (aligned)
    # -------------------------
    def check_feasibility(self, result: Dict, OS: List[int]) -> Tuple[bool, List[str]]:
        if "schedule_results" not in result:
            return False, ["结果字典缺少 'schedule_results'，请在 decode 时设置 return_schedule=True"]

        data = result["schedule_results"]
        errors: List[str] = []
        eps = EPSILON

        # build OS predecessor map (op_id -> prev op_id in same job)
        try:
            _, _, os_ops, _, prev_op_os_map = self._build_os_structures(OS)
        except Exception as e:
            return False, [f"解析 OS 结构失败: {str(e)}"]

        # Resource intervals: dev -> [(s,e,op_id)]
        resource_intervals: Dict[str, List[Tuple[float, float, str]]] = {}

        # Per-op timings
        op_proc: Dict[str, Tuple[float, float]] = {}
        op_w: Dict[str, Dict[str, float]] = {}
        op_f: Dict[str, Dict[str, float]] = {}

        # Per-job intervals
        job_proc: Dict[int, List[Tuple[float, float, str]]] = {}
        job_w_loaded: Dict[int, List[Tuple[float, float, str]]] = {}

        def _add_interval(dev: str, s: float, e: float, op_id: str) -> None:
            if dev not in resource_intervals:
                resource_intervals[dev] = []
            resource_intervals[dev].append((s, e, op_id))

        # Parse schedule_results
        for item in data:
            dev = item.get("device")
            op_id = item.get("operation")
            if not dev or not op_id:
                continue

            job = self._op_static_info.get(op_id, 0)

            if dev.startswith("M"):
                s = float(item["start_time"])
                e = float(item["end_time"])
                op_proc[op_id] = (s, e)
                _add_interval(dev, s, e, op_id)
                job_proc.setdefault(job, []).append((s, e, op_id))

            elif dev.startswith("AGV_W"):
                s = float(item["start_time"])
                pickup_leave = float(item["pickup_leave"])
                delivery_arrive = float(item["delivery_arrive"])
                delivery_complete = float(item["delivery_complete"])

                op_w[op_id] = {
                    "start_time": s,
                    "pickup_leave": pickup_leave,
                    "delivery_arrive": delivery_arrive,
                    "delivery_complete": delivery_complete,
                }
                # resource occupancy
                if delivery_complete - s >= eps:
                    _add_interval(dev, s, delivery_complete, op_id)

                # job indivisibility uses LOADED segment only
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
                _add_interval(dev, s, return_complete, op_id)

        # (1) Resource overlap checks
        for dev, ints in resource_intervals.items():
            ints.sort(key=lambda x: x[0])
            for i in range(len(ints) - 1):
                s1, e1, op1 = ints[i]
                s2, e2, op2 = ints[i + 1]
                if e1 > s2 + eps:
                    errors.append(f"[资源冲突] {dev}: {op1} overlaps {op2}")

        # (2) Arrival-before-processing checks
        for op_id, (p_s, _p_e) in op_proc.items():
            if op_id in op_w and p_s < op_w[op_id]["delivery_arrive"] - eps:
                errors.append(f"[到达约束] {op_id}: process starts before W delivered")
            if op_id in op_f and p_s < op_f[op_id]["delivery_arrive"] - eps:
                errors.append(f"[到达约束] {op_id}: process starts before F delivered")

        # (3) Predecessor completion before W pickup_leave
        for op_id, prev_id in prev_op_os_map.items():
            if prev_id is None:
                continue
            if op_id not in op_w:
                # same-machine or no transport record -> OK
                continue
            if prev_id not in op_proc:
                continue
            prev_end = op_proc[prev_id][1]
            if op_w[op_id]["pickup_leave"] < prev_end - eps:
                errors.append(f"[前序-运输] {op_id}: pickup_leave < {prev_id}.process_end")

        # (4) Job indivisibility checks (efficient)
        def check_no_overlap(ints: List[Tuple[float, float, str]], label: str) -> None:
            ints.sort(key=lambda x: x[0])
            for i in range(len(ints) - 1):
                s1, e1, op1 = ints[i]
                s2, e2, op2 = ints[i + 1]
                if e1 > s2 + eps:
                    errors.append(f"[作业内冲突] {label}: {op1} overlaps {op2}")

        # (4-a) proc within job
        for j, ints in job_proc.items():
            check_no_overlap(ints, f"Job{j} Processing")

        # (4-b) loaded W within job
        for j, ints in job_w_loaded.items():
            check_no_overlap(ints, f"Job{j} Loaded-W")

        # (4-c) proc vs loaded W within job (two-pointer)
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

                # advance the interval that ends first
                if pe <= we + eps:
                    pi += 1
                else:
                    wi += 1

        return (len(errors) == 0), errors
