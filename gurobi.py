"""Gurobi MILP implementation for the AFAISP model in README.md.

The model follows the README notation as closely as possible:
    x, alpha, eta, y, thetaW, z, thetaF, xi, delta,
    P, S_P, S_W, S_F, T_W, E_W, T_F, C_max.

Implementation notes:
    - Continuous variables are given finite upper bounds derived from a simple
      conservative horizon to tighten the model.
    - The big-M value defaults to that horizon unless explicitly supplied.
    - The README defines E^W_{ii'} for all operation pairs. Here it is created
      only for i != i', which is the only case used by AGV sequencing.
"""

from __future__ import annotations

import csv
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from data.loader import Instance, load_instance


EPS = 1e-5
GANTT_COLUMNS = [
    "lane", "resource_id", "job", "op", "start", "finish", "kind",
    "w_empty_start", "w_empty_finish", "w_wait_start", "w_wait_finish",
    "w_loaded_start", "w_loaded_finish", "f_start", "f_arrival", "f_return",
    "wpos", "src", "dst",
]


# =============================================================================
# Batch solve config
# =============================================================================

GUROBI_INSTANCE_NAMES = [
    "kumar01", "kumar02", "kumar03", "kumar04", "kumar05",
    "kumar06", "kumar07", "kumar08", "kumar09", "kumar10",
    "MK01", "MK02", "MK03", "MK04", "MK05",
    # "Fattahi01", "Fattahi02", "Fattahi03", "Fattahi04", "Fattahi05",
    # "Fattahi06", "Fattahi07", "Fattahi08", "Fattahi09", "Fattahi10",
    # "Fattahi11", "Fattahi12", "Fattahi13", "Fattahi14", "Fattahi15",
    # "Fattahi16", "Fattahi17", "Fattahi18", "Fattahi19", "Fattahi20",
    # "Behnke01", "Behnke02", "Behnke03", "Behnke04", "Behnke05",
]

GUROBI_TIME_LIMIT = 3600
GUROBI_BIG_M = None
GUROBI_RETRY_FAILED = False
GUROBI_LOG_DIR = Path("logs_gurobi")
GUROBI_OUTPUT_DIR = Path("output") / "gurobi_output"
GUROBI_SUMMARY_CSV = GUROBI_OUTPUT_DIR / "gurobi_summary.csv"

SUMMARY_COLUMNS = [
    "instance", "status", "status_code", "sol_count", "best_objective",
    "best_bound", "mip_gap", "runtime_s", "wall_time_s", "node_count",
    "simplex_iterations", "num_vars", "num_constrs", "num_bin_vars",
    "num_int_vars", "time_limit", "big_M", "log_file", "gantt_file",
    "solution_file", "feasibility_file", "feasible_by_checker",
    "checker_issue_count", "error", "timestamp",
]


def parse_op(op_id: str) -> tuple[int, int]:
    job_str, seq_str = op_id[1:].split("_")
    return int(job_str), int(seq_str)


def loc_name(loc: int) -> str:
    return "Warehouse" if loc == 0 else f"M{loc}"


def op_job(op_id: str) -> int:
    return parse_op(op_id)[0]


def estimate_horizon(inst: Instance) -> float:
    max_proc_sum = sum(max(times.values()) for times in inst.processing_times.values())
    max_w = max(max(row.values()) for row in inst.agv_w_transport_times.values())
    max_f = max(max(row.values()) for row in inst.agv_f_transport_times.values())
    return float(max_proc_sum + inst.num_operations * (3 * max_w + 2 * max_f) + 1)


def build_afaisp_milp(inst: Instance, time_limit: float = 3600, big_M: float | None = None):
    ops = list(inst.processing_times.keys())
    op_pairs = [(ops[a], ops[b]) for a in range(len(ops)) for b in range(a + 1, len(ops))]
    ordered_pairs = [(i, ip) for i in ops for ip in ops if i != ip]
    machines = list(range(1, inst.num_machines + 1))
    agvs_w = list(range(1, inst.num_agvs_w + 1))
    agvs_f = list(range(1, inst.num_agvs_f + 1))
    jobs = sorted(inst.job_operations.keys())
    job_ops = {j: [op for op in ops if op_job(op) == j] for j in jobs}
    eligible = {op: sorted(int(k) for k in inst.processing_times[op].keys()) for op in ops}
    horizon = estimate_horizon(inst)
    M = float(big_M if big_M is not None else horizon)

    TT_W = {
        u: {v: float(inst.agv_w_transport_times[loc_name(u)][loc_name(v)])
            for v in range(0, inst.num_machines + 1)}
        for u in range(0, inst.num_machines + 1)
    }
    TT_F = {
        u: {v: float(inst.agv_f_transport_times[loc_name(u)][loc_name(v)])
            for v in range(0, inst.num_machines + 1)}
        for u in range(0, inst.num_machines + 1)
    }

    mdl = gp.Model("AFAISP_README")
    mdl.Params.TimeLimit = float(time_limit)

    C_max = mdl.addVar(lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="C_max")
    P = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="P")
    S_P = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="S_P")
    S_W = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="S_W")
    S_F = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="S_F")
    T_W = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="T_W")
    T_F = mdl.addVars(ops, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="T_F")
    E_W = mdl.addVars(ordered_pairs, lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name="E_W")

    x = mdl.addVars(((i, k) for i in ops for k in eligible[i]), vtype=GRB.BINARY, name="x")
    alpha = mdl.addVars(((i, ip, k) for i, ip in op_pairs for k in set(eligible[i]).intersection(eligible[ip])),
                        vtype=GRB.BINARY, name="alpha")
    eta = mdl.addVars(ops, vtype=GRB.BINARY, name="eta")
    y = mdl.addVars(((i, w) for i in ops for w in agvs_w), vtype=GRB.BINARY, name="y")
    z = mdl.addVars(((i, f) for i in ops for f in agvs_f), vtype=GRB.BINARY, name="z")
    thetaW = mdl.addVars(((i, ip, w) for i, ip in op_pairs for w in agvs_w), vtype=GRB.BINARY, name="thetaW")
    thetaF = mdl.addVars(((i, ip, f) for i, ip in op_pairs for f in agvs_f), vtype=GRB.BINARY, name="thetaF")
    delta = mdl.addVars(((j, i) for j in jobs for i in job_ops[j]), vtype=GRB.BINARY, name="delta")
    xi = mdl.addVars(((j, p, i) for j in jobs for p in job_ops[j] for i in job_ops[j] if p != i),
                     vtype=GRB.BINARY, name="xi")

    mdl.setObjective(C_max, GRB.MINIMIZE)

    # Objective and assignment.
    for i in ops:
        mdl.addConstr(C_max >= S_P[i] + P[i], name=f"cmax_{i}")
        mdl.addConstr(gp.quicksum(x[i, k] for k in eligible[i]) == 1, name=f"assign_{i}")
        mdl.addConstr(P[i] == gp.quicksum(float(inst.processing_times[i][k]) * x[i, k] for k in eligible[i]),
                      name=f"proc_{i}")

    # Assembly-unit capacity.
    for i, ip in op_pairs:
        common = set(eligible[i]).intersection(eligible[ip])
        for k in common:
            a = alpha[i, ip, k]
            mdl.addConstr(S_P[ip] >= S_P[i] + P[i] - M * (3 - x[i, k] - x[ip, k] - a),
                          name=f"unit_1_{i}_{ip}_{k}")
            mdl.addConstr(S_P[i] >= S_P[ip] + P[ip] - M * (2 - x[i, k] - x[ip, k] + a),
                          name=f"unit_2_{i}_{ip}_{k}")

    # Product chain selection with immediate predecessor variables.
    for j in jobs:
        mdl.addConstr(gp.quicksum(delta[j, i] for i in job_ops[j]) == 1, name=f"one_first_{j}")
        for p in job_ops[j]:
            mdl.addConstr(gp.quicksum(xi[j, p, i] for i in job_ops[j] if i != p) <= 1,
                          name=f"succ_at_most_one_{j}_{p}")
        for i in job_ops[j]:
            mdl.addConstr(delta[j, i] + gp.quicksum(xi[j, p, i] for p in job_ops[j] if p != i) == 1,
                          name=f"one_incoming_or_first_{j}_{i}")
        for p in job_ops[j]:
            for i in job_ops[j]:
                if p == i:
                    continue
                mdl.addConstr(S_P[i] >= S_P[p] + P[p] - M * (1 - xi[j, p, i]),
                              name=f"chain_time_{j}_{p}_{i}")

    # Technological precedence arcs from the instance.
    for i, preds in inst.priority_dict.items():
        for p in preds:
            mdl.addConstr(S_P[i] >= S_P[p] + P[p], name=f"prec_{p}_{i}")

    # Need for W-AGV loaded transfer.
    for j in jobs:
        for i in job_ops[j]:
            mdl.addConstr(eta[i] >= delta[j, i], name=f"needW_first_{j}_{i}")
            for p in job_ops[j]:
                if p == i:
                    continue
                common = set(eligible[p]).intersection(eligible[i])
                for k in common:
                    mdl.addConstr(eta[i] <= 3 - xi[j, p, i] - x[p, k] - x[i, k],
                                  name=f"needW_same_machine_{j}_{p}_{i}_{k}")
                for kp in eligible[p]:
                    for k in eligible[i]:
                        if kp == k:
                            continue
                        mdl.addConstr(eta[i] >= xi[j, p, i] + x[p, kp] + x[i, k] - 2,
                                      name=f"needW_diff_machine_{j}_{p}_{i}_{kp}_{k}")

    # AGV assignment.
    for i in ops:
        mdl.addConstr(gp.quicksum(y[i, w] for w in agvs_w) == eta[i], name=f"assign_W_{i}")
        mdl.addConstr(gp.quicksum(z[i, f] for f in agvs_f) == 1, name=f"assign_F_{i}")

    # Empty W travel E_W.
    for i, ip in ordered_pairs:
        jp = op_job(ip)
        for k in eligible[i]:
            mdl.addConstr(E_W[i, ip] <= TT_W[k][0] + M * (2 - x[i, k] - delta[jp, ip]),
                          name=f"E_first_up_{i}_{ip}_{k}")
            mdl.addConstr(E_W[i, ip] >= TT_W[k][0] - M * (2 - x[i, k] - delta[jp, ip]),
                          name=f"E_first_lo_{i}_{ip}_{k}")
        for p in job_ops[jp]:
            if p == ip or p == i:
                continue
            for k in eligible[i]:
                for kp in eligible[p]:
                    mdl.addConstr(E_W[i, ip] <= TT_W[k][kp] + M * (3 - x[i, k] - xi[jp, p, ip] - x[p, kp]),
                                  name=f"E_pred_up_{i}_{ip}_{p}_{k}_{kp}")
                    mdl.addConstr(E_W[i, ip] >= TT_W[k][kp] - M * (3 - x[i, k] - xi[jp, p, ip] - x[p, kp]),
                                  name=f"E_pred_lo_{i}_{ip}_{p}_{k}_{kp}")

    # Loaded travel times.
    for j in jobs:
        for i in job_ops[j]:
            mdl.addConstr(T_W[i] <= gp.quicksum(TT_W[0][k] * x[i, k] for k in eligible[i])
                          + M * (1 - delta[j, i]), name=f"TW_first_up_{j}_{i}")
            mdl.addConstr(T_W[i] >= gp.quicksum(TT_W[0][k] * x[i, k] for k in eligible[i])
                          - M * (1 - delta[j, i]), name=f"TW_first_lo_{j}_{i}")
            for p in job_ops[j]:
                if p == i:
                    continue
                for kp in eligible[p]:
                    mdl.addConstr(T_W[i] <= gp.quicksum(TT_W[kp][k] * x[i, k] for k in eligible[i])
                                  + M * (2 - xi[j, p, i] - x[p, kp]),
                                  name=f"TW_pred_up_{j}_{p}_{i}_{kp}")
                    mdl.addConstr(T_W[i] >= gp.quicksum(TT_W[kp][k] * x[i, k] for k in eligible[i])
                                  - M * (2 - xi[j, p, i] - x[p, kp]),
                                  name=f"TW_pred_lo_{j}_{p}_{i}_{kp}")

    for i in ops:
        mdl.addConstr(T_F[i] == gp.quicksum(TT_F[0][k] * x[i, k] for k in eligible[i]), name=f"TF_{i}")

    # AGV W capacity.
    for i, ip in op_pairs:
        for w in agvs_w:
            th = thetaW[i, ip, w]
            mdl.addConstr(S_W[ip] >= S_W[i] + T_W[i] + E_W[i, ip]
                          - M * (3 - y[i, w] - y[ip, w] - th),
                          name=f"Wcap_1_{i}_{ip}_{w}")
            mdl.addConstr(S_W[i] >= S_W[ip] + T_W[ip] + E_W[ip, i]
                          - M * (2 - y[i, w] - y[ip, w] + th),
                          name=f"Wcap_2_{i}_{ip}_{w}")

    # README initial-W constraint.
    for j in jobs:
        for p in job_ops[j]:
            for i in job_ops[j]:
                if p == i:
                    continue
                for kp in eligible[p]:
                    mdl.addConstr(S_W[i] >= TT_W[0][kp] - M * (3 - eta[i] - xi[j, p, i] - x[p, kp]),
                                  name=f"initial_W_{j}_{p}_{i}_{kp}")

    # AGV F capacity.
    ret_F = {
        i: gp.quicksum(TT_F[k][0] * x[i, k] for k in eligible[i])
        for i in ops
    }
    for i, ip in op_pairs:
        for f in agvs_f:
            th = thetaF[i, ip, f]
            mdl.addConstr(S_F[ip] >= S_F[i] + T_F[i] + ret_F[i]
                          - M * (3 - z[i, f] - z[ip, f] - th),
                          name=f"Fcap_1_{i}_{ip}_{f}")
            mdl.addConstr(S_F[i] >= S_F[ip] + T_F[ip] + ret_F[ip]
                          - M * (2 - z[i, f] - z[ip, f] + th),
                          name=f"Fcap_2_{i}_{ip}_{f}")

    # Release and synchronization.
    for j in jobs:
        for p in job_ops[j]:
            for i in job_ops[j]:
                if p == i:
                    continue
                mdl.addConstr(S_W[i] >= S_P[p] + P[p] - M * (2 - xi[j, p, i] - eta[i]),
                              name=f"release_W_{j}_{p}_{i}")
    for i in ops:
        mdl.addConstr(S_P[i] >= S_W[i] + T_W[i] - M * (1 - eta[i]), name=f"sync_W_{i}")
        mdl.addConstr(S_P[i] >= S_F[i] + T_F[i], name=f"sync_F_{i}")

    aux = dict(
        inst=inst,
        ops=ops, op_pairs=op_pairs, ordered_pairs=ordered_pairs, machines=machines,
        agvs_w=agvs_w, agvs_f=agvs_f, jobs=jobs, job_ops=job_ops,
        eligible=eligible, TT_W=TT_W, TT_F=TT_F, M=M, horizon=horizon,
        C_max=C_max, P=P, S_P=S_P, S_W=S_W, S_F=S_F, T_W=T_W,
        T_F=T_F, E_W=E_W, x=x, alpha=alpha, eta=eta, y=y, z=z,
        thetaW=thetaW, thetaF=thetaF, delta=delta, xi=xi,
    )
    return mdl, aux


# Backward-compatible aliases for older scripts that may import the previous
# builder name.
build_afaip_milp = build_afaisp_milp
build_sofjspt_milp = build_afaisp_milp


def val(v) -> float:
    return float(v.X)


def selected_machine(aux: dict, op: str) -> int:
    x = aux["x"]
    for k in aux["eligible"][op]:
        if val(x[op, k]) > 0.5:
            return int(k)
    raise ValueError(f"No selected machine for {op}")


def selected_agv(var, op: str, agvs: list[int]) -> int | None:
    for a in agvs:
        if val(var[op, a]) > 0.5:
            return int(a)
    return None


def selected_predecessor(aux: dict, op: str) -> str | None:
    j = op_job(op)
    if val(aux["delta"][j, op]) > 0.5:
        return None
    for p in aux["job_ops"][j]:
        if p != op and val(aux["xi"][j, p, op]) > 0.5:
            return p
    return None


def full_F_duration(aux: dict, op: str) -> float:
    k = selected_machine(aux, op)
    return val(aux["T_F"][op]) + aux["TT_F"][k][0]


def extract_solution_schedule(mdl, aux: dict) -> list[dict[str, Any]]:
    if mdl.SolCount <= 0:
        return []
    rows = []
    for op in aux["ops"]:
        job = op_job(op)
        k = selected_machine(aux, op)
        w = selected_agv(aux["y"], op, aux["agvs_w"])
        f = selected_agv(aux["z"], op, aux["agvs_f"])
        pred = selected_predecessor(aux, op)
        eta = val(aux["eta"][op]) > 0.5
        sp = val(aux["S_P"][op])
        pp = val(aux["P"][op])
        sf = val(aux["S_F"][op])
        tf = val(aux["T_F"][op])
        f_ret = sf + full_F_duration(aux, op)

        rows.append({
            "device": f"M{k}", "operation": op, "job": job,
            "start_time": sp, "end_time": sp + pp,
            "assigned_machine": k, "assigned_agv_w": w, "assigned_agv_f": f,
        })
        rows.append({
            "device": f"AGV_F{f}", "operation": op, "job": job,
            "start_time": sf, "delivery_arrive": sf + tf, "return_complete": f_ret,
            "from_location": "Warehouse", "delivery_location": f"M{k}",
        })
        if eta:
            sw = val(aux["S_W"][op])
            tw = val(aux["T_W"][op])
            src = "Warehouse" if pred is None else f"M{selected_machine(aux, pred)}"
            rows.append({
                "device": f"AGV_W{w}", "operation": op, "job": job,
                "start_time": sw, "pickup_arrive": sw, "pickup_leave": sw,
                "delivery_arrive": sw + tw, "delivery_complete": sw + tw,
                "from_location": src, "pickup_location": src, "delivery_location": f"M{k}",
            })
    return rows


def gantt_rows_from_solution(mdl, aux: dict) -> list[dict[str, Any]]:
    rows = []
    schedule = extract_solution_schedule(mdl, aux)
    w_prev_by_op = {}
    w_records_by_agv = {}
    for rec in schedule:
        if rec["device"].startswith("AGV_W"):
            w_records_by_agv.setdefault(int(rec["device"][5:]), []).append(rec)
    for records in w_records_by_agv.values():
        records.sort(key=lambda r: r["pickup_leave"])
        for prev, cur in zip(records, records[1:]):
            w_prev_by_op[cur["operation"]] = prev["operation"]

    for rec in schedule:
        dev = rec["device"]
        op = rec["operation"]
        job = op_job(op)
        if dev.startswith("M"):
            rows.append({
                "lane": "Machine", "resource_id": int(dev[1:]), "job": job, "op": op,
                "start": rec["start_time"], "finish": rec["end_time"], "kind": "proc",
                "w_empty_start": None, "w_empty_finish": None, "w_wait_start": None, "w_wait_finish": None,
                "w_loaded_start": None, "w_loaded_finish": None,
                "f_start": None, "f_arrival": None, "f_return": None,
                "wpos": None, "src": None, "dst": None,
            })
        elif dev.startswith("AGV_W"):
            prev_op = w_prev_by_op.get(op)
            if prev_op is None:
                empty_start = None
                empty_finish = None
                wait_start = rec["pickup_arrive"]
            else:
                prev_finish = val(aux["S_W"][prev_op]) + val(aux["T_W"][prev_op])
                empty_start = prev_finish
                empty_finish = prev_finish + val(aux["E_W"][prev_op, op])
                wait_start = empty_finish
            rows.append({
                "lane": "AGV_W", "resource_id": int(dev[5:]), "job": job, "op": op,
                "start": None, "finish": None, "kind": "transport",
                "w_empty_start": empty_start, "w_empty_finish": empty_finish,
                "w_wait_start": wait_start, "w_wait_finish": rec["pickup_leave"],
                "w_loaded_start": rec["pickup_leave"], "w_loaded_finish": rec["delivery_complete"],
                "f_start": None, "f_arrival": None, "f_return": None,
                "wpos": rec["from_location"], "src": rec["pickup_location"], "dst": rec["delivery_location"],
            })
        elif dev.startswith("AGV_F"):
            rows.append({
                "lane": "AGV_F", "resource_id": int(dev[5:]), "job": job, "op": op,
                "start": rec["start_time"], "finish": rec["return_complete"], "kind": "transport",
                "w_empty_start": None, "w_empty_finish": None, "w_wait_start": None, "w_wait_finish": None,
                "w_loaded_start": None, "w_loaded_finish": None,
                "f_start": rec["start_time"], "f_arrival": rec["delivery_arrive"], "f_return": rec["return_complete"],
                "wpos": None, "src": rec["from_location"], "dst": rec["delivery_location"],
            })
    return rows


def export_gantt_csv(mdl, aux: dict, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = gantt_rows_from_solution(mdl, aux)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GANTT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_solution_csv(mdl, aux: dict, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "operation", "job", "selected_machine", "processing_time",
        "process_start", "process_finish", "predecessor_in_chain",
        "needs_w_agv", "selected_agv_w", "w_start", "w_loaded_time",
        "w_delivery_finish", "selected_agv_f", "f_start", "f_loaded_time",
        "f_delivery_arrive", "f_return_finish",
    ]
    rows = []
    for op in aux["ops"]:
        k = selected_machine(aux, op)
        pred = selected_predecessor(aux, op)
        needs_w = val(aux["eta"][op]) > 0.5
        w = selected_agv(aux["y"], op, aux["agvs_w"]) if needs_w else None
        f = selected_agv(aux["z"], op, aux["agvs_f"])
        p_start = val(aux["S_P"][op])
        p_time = val(aux["P"][op])
        w_start = val(aux["S_W"][op]) if needs_w else None
        w_time = val(aux["T_W"][op]) if needs_w else None
        f_start = val(aux["S_F"][op])
        f_time = val(aux["T_F"][op])
        rows.append({
            "operation": op,
            "job": op_job(op),
            "selected_machine": k,
            "processing_time": p_time,
            "process_start": p_start,
            "process_finish": p_start + p_time,
            "predecessor_in_chain": pred or "",
            "needs_w_agv": int(needs_w),
            "selected_agv_w": w or "",
            "w_start": w_start if needs_w else "",
            "w_loaded_time": w_time if needs_w else "",
            "w_delivery_finish": (w_start + w_time) if needs_w else "",
            "selected_agv_f": f,
            "f_start": f_start,
            "f_loaded_time": f_time,
            "f_delivery_arrive": f_start + f_time,
            "f_return_finish": f_start + full_F_duration(aux, op),
        })
    rows.sort(key=lambda r: (r["process_start"], r["job"], r["operation"]))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _safe_model_attr(mdl, name: str, default=None):
    try:
        return getattr(mdl, name)
    except Exception:
        return default


def _gurobi_status_name(status_code: int | None) -> str:
    names = {
        getattr(GRB, "LOADED", None): "LOADED",
        getattr(GRB, "OPTIMAL", None): "OPTIMAL",
        getattr(GRB, "INFEASIBLE", None): "INFEASIBLE",
        getattr(GRB, "INF_OR_UNBD", None): "INF_OR_UNBD",
        getattr(GRB, "UNBOUNDED", None): "UNBOUNDED",
        getattr(GRB, "CUTOFF", None): "CUTOFF",
        getattr(GRB, "ITERATION_LIMIT", None): "ITERATION_LIMIT",
        getattr(GRB, "NODE_LIMIT", None): "NODE_LIMIT",
        getattr(GRB, "TIME_LIMIT", None): "TIME_LIMIT",
        getattr(GRB, "SOLUTION_LIMIT", None): "SOLUTION_LIMIT",
        getattr(GRB, "INTERRUPTED", None): "INTERRUPTED",
        getattr(GRB, "NUMERIC", None): "NUMERIC",
        getattr(GRB, "SUBOPTIMAL", None): "SUBOPTIMAL",
        getattr(GRB, "INPROGRESS", None): "INPROGRESS",
        getattr(GRB, "USER_OBJ_LIMIT", None): "USER_OBJ_LIMIT",
        getattr(GRB, "WORK_LIMIT", None): "WORK_LIMIT",
        getattr(GRB, "MEM_LIMIT", None): "MEM_LIMIT",
    }
    names.pop(None, None)
    return names.get(status_code, str(status_code))


def _append_summary_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _read_completed_instances(path: str | Path, retry_failed: bool = False) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            instance = row.get("instance")
            status = row.get("status")
            if not instance:
                continue
            if retry_failed and status == "FAILED":
                continue
            completed.add(instance)
    return completed


def collect_model_summary(
    instance_name: str,
    mdl,
    sol_exists: bool,
    log_path: str | Path,
    gantt_path: str | Path,
    solution_path: str | Path,
    check_path: str | Path,
    feasible_by_checker: bool | None,
    checker_issues: list[str] | None,
    wall_time_s: float,
    time_limit: float,
    big_M: float | None,
    error: str = "",
) -> dict:
    status_code = _safe_model_attr(mdl, "Status", None)
    sol_count = int(_safe_model_attr(mdl, "SolCount", 0) or 0)
    return {
        "instance": instance_name,
        "status": _gurobi_status_name(status_code),
        "status_code": status_code,
        "sol_count": sol_count,
        "best_objective": _safe_model_attr(mdl, "ObjVal", None) if sol_exists else None,
        "best_bound": _safe_model_attr(mdl, "ObjBound", None),
        "mip_gap": _safe_model_attr(mdl, "MIPGap", None) if sol_count > 0 else None,
        "runtime_s": _safe_model_attr(mdl, "Runtime", None),
        "wall_time_s": wall_time_s,
        "node_count": _safe_model_attr(mdl, "NodeCount", None),
        "simplex_iterations": _safe_model_attr(mdl, "IterCount", None),
        "num_vars": _safe_model_attr(mdl, "NumVars", None),
        "num_constrs": _safe_model_attr(mdl, "NumConstrs", None),
        "num_bin_vars": _safe_model_attr(mdl, "NumBinVars", None),
        "num_int_vars": _safe_model_attr(mdl, "NumIntVars", None),
        "time_limit": time_limit,
        "big_M": big_M,
        "log_file": str(log_path),
        "gantt_file": str(gantt_path) if sol_exists else "",
        "solution_file": str(solution_path) if sol_exists else "",
        "feasibility_file": str(check_path) if sol_exists else "",
        "feasible_by_checker": feasible_by_checker,
        "checker_issue_count": len(checker_issues or []),
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def check_solution_feasibility(mdl, aux: dict, tol: float = 1e-4) -> tuple[bool, list[str]]:
    if mdl.SolCount <= 0:
        return False, ["No solution available."]
    issues = []
    ops = aux["ops"]

    def le(lhs, rhs, name):
        if lhs > rhs + tol:
            issues.append(f"{name}: {lhs:.6f} > {rhs:.6f}")

    # Assignment and precedence checks.
    for op in ops:
        mach_sum = sum(1 for k in aux["eligible"][op] if val(aux["x"][op, k]) > 0.5)
        if mach_sum != 1:
            issues.append(f"{op}: selected machine count is {mach_sum}")
        f_sum = sum(1 for f in aux["agvs_f"] if val(aux["z"][op, f]) > 0.5)
        if f_sum != 1:
            issues.append(f"{op}: selected AGV_F count is {f_sum}")
        eta = val(aux["eta"][op]) > 0.5
        w_sum = sum(1 for w in aux["agvs_w"] if val(aux["y"][op, w]) > 0.5)
        if w_sum != int(eta):
            issues.append(f"{op}: selected AGV_W count {w_sum} != eta {int(eta)}")
        k = selected_machine(aux, op)
        p_calc = float(aux["P"][op].X)
        expected_p = float(aux["inst"].processing_times[op][k])
        if abs(p_calc - expected_p) > tol:
            issues.append(f"{op}: processing time {p_calc:.6f} != selected duration {expected_p:.6f}")
        if val(aux["S_P"][op]) + val(aux["P"][op]) > val(aux["C_max"]) + tol:
            issues.append(f"{op}: exceeds makespan.")

    # Technological precedence.
    for i, preds in aux["inst"].priority_dict.items():
        for p in preds:
            le(val(aux["S_P"][p]) + val(aux["P"][p]), val(aux["S_P"][i]), f"precedence {p}->{i}")

    # Use instance-derived precedence from selected xi and hard precedence constraints.
    for j in aux["jobs"]:
        first = [i for i in aux["job_ops"][j] if val(aux["delta"][j, i]) > 0.5]
        if len(first) != 1:
            issues.append(f"Job {j}: first operation count is {len(first)}")
        for i in aux["job_ops"][j]:
            incoming = [p for p in aux["job_ops"][j] if p != i and val(aux["xi"][j, p, i]) > 0.5]
            if len(incoming) + (1 if val(aux["delta"][j, i]) > 0.5 else 0) != 1:
                issues.append(f"{i}: invalid chain incoming/first relation")
            for p in incoming:
                le(val(aux["S_P"][p]) + val(aux["P"][p]), val(aux["S_P"][i]), f"chain {p}->{i}")

    # Resource overlaps.
    machine_tasks = {}
    w_tasks = {}
    f_tasks = {}
    for op in ops:
        k = selected_machine(aux, op)
        machine_tasks.setdefault(k, []).append((val(aux["S_P"][op]), val(aux["S_P"][op]) + val(aux["P"][op]), op))
        f = selected_agv(aux["z"], op, aux["agvs_f"])
        f_tasks.setdefault(f, []).append((val(aux["S_F"][op]), val(aux["S_F"][op]) + full_F_duration(aux, op), op))
        if val(aux["eta"][op]) > 0.5:
            w = selected_agv(aux["y"], op, aux["agvs_w"])
            w_tasks.setdefault(w, []).append((val(aux["S_W"][op]), val(aux["S_W"][op]) + val(aux["T_W"][op]), op))

    for label, tasks_by_res in [("Machine", machine_tasks), ("AGV_F", f_tasks)]:
        for res, tasks in tasks_by_res.items():
            tasks = sorted(tasks)
            for prev, cur in zip(tasks, tasks[1:]):
                if prev[1] > cur[0] + tol:
                    issues.append(f"{label} {res}: overlap {prev[2]} and {cur[2]}")
    for res, tasks in w_tasks.items():
        tasks = sorted(tasks)
        for prev, cur in zip(tasks, tasks[1:]):
            travel_ready = prev[1] + val(aux["E_W"][prev[2], cur[2]])
            if travel_ready > cur[0] + tol:
                issues.append(
                    f"AGV_W {res}: overlap/empty-travel violation {prev[2]}->{cur[2]} "
                    f"({travel_ready:.6f} > {cur[0]:.6f})"
                )

    return len(issues) == 0, issues


def print_detailed_schedule(mdl, aux, sol_exists, log_file_handle=None):
    if not sol_exists:
        print("No solution found.")
        return
    lines = [f"C_max = {mdl.ObjVal:.4f}"]
    for row in sorted(gantt_rows_from_solution(mdl, aux), key=lambda r: (str(r["lane"]), int(r["resource_id"]), r["start"] or r["w_loaded_start"] or 0)):
        if row["lane"] == "Machine":
            lines.append(f"Machine {row['resource_id']}: {row['op']} [{row['start']:.2f}, {row['finish']:.2f}]")
        elif row["lane"] == "AGV_W":
            lines.append(f"AGV_W {row['resource_id']}: {row['op']} [{row['w_loaded_start']:.2f}, {row['w_loaded_finish']:.2f}] {row['src']}->{row['dst']}")
        else:
            lines.append(f"AGV_F {row['resource_id']}: {row['op']} [{row['f_start']:.2f}, {row['f_return']:.2f}] {row['src']}->{row['dst']}")
    text = "\n".join(lines)
    print(text)
    if log_file_handle is not None:
        log_file_handle.write(text + "\n")


def solve_sofjspt_instance(
    inst: Instance,
    instance_name: str = "instance",
    time_limit: float = 3600,
    big_M: float | None = None,
    log_dir: str | Path = "./logs_gurobi",
    output_dir: str | Path = "./output/gurobi_output",
):
    mdl, aux = build_afaisp_milp(inst, time_limit=time_limit, big_M=big_M)
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"gurobi_{instance_name}_{timestamp}.log"
    gantt_path = output_dir / f"{instance_name}_gantt_data.csv"
    solution_path = output_dir / f"{instance_name}_best_solution.csv"
    check_path = output_dir / f"{instance_name}_feasibility_check.txt"

    mdl.Params.LogFile = str(log_path)
    start_time = time.time()
    mdl.optimize()
    wall_time_s = time.time() - start_time

    sol_exists = mdl.SolCount > 0
    feasible_by_checker = None
    checker_issues = []
    if sol_exists:
        feasible_by_checker, checker_issues = check_solution_feasibility(mdl, aux)
        export_gantt_csv(mdl, aux, gantt_path)
        export_solution_csv(mdl, aux, solution_path)
        with check_path.open("w", encoding="utf-8") as f:
            f.write(f"feasible_by_checker: {feasible_by_checker}\n")
            f.write(f"objective: {mdl.ObjVal:.6f}\n")
            f.write(f"status: {mdl.Status}\n")
            f.write(f"best_bound: {_safe_model_attr(mdl, 'ObjBound', '')}\n")
            f.write(f"mip_gap: {_safe_model_attr(mdl, 'MIPGap', '')}\n")
            f.write(f"runtime_s: {_safe_model_attr(mdl, 'Runtime', '')}\n")
            if checker_issues:
                f.write("\n".join(checker_issues))
        print(f"Feasibility checker: {feasible_by_checker}")
        if checker_issues:
            print("Checker issues:")
            for issue in checker_issues[:20]:
                print("  -", issue)
        print(f"Gantt data saved to: {gantt_path}")
        print(f"Best solution data saved to: {solution_path}")
        print(f"Feasibility report saved to: {check_path}")
    else:
        print("No feasible solution found.")

    mdl._blcme_summary_row = collect_model_summary(
        instance_name=instance_name,
        mdl=mdl,
        sol_exists=sol_exists,
        log_path=log_path,
        gantt_path=gantt_path,
        solution_path=solution_path,
        check_path=check_path,
        feasible_by_checker=feasible_by_checker,
        checker_issues=checker_issues,
        wall_time_s=wall_time_s,
        time_limit=time_limit,
        big_M=big_M,
    )

    return mdl, sol_exists, str(log_path)


def failed_summary_row(
    instance_name: str,
    error: str,
    wall_time_s: float,
    time_limit: float,
    big_M: float | None,
) -> dict:
    return {
        "instance": instance_name,
        "status": "FAILED",
        "status_code": "",
        "sol_count": 0,
        "best_objective": "",
        "best_bound": "",
        "mip_gap": "",
        "runtime_s": "",
        "wall_time_s": wall_time_s,
        "node_count": "",
        "simplex_iterations": "",
        "num_vars": "",
        "num_constrs": "",
        "num_bin_vars": "",
        "num_int_vars": "",
        "time_limit": time_limit,
        "big_M": big_M,
        "log_file": "",
        "gantt_file": "",
        "solution_file": "",
        "feasibility_file": "",
        "feasible_by_checker": "",
        "checker_issue_count": "",
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def solve_gurobi_instances(
    instance_names: list[str] | tuple[str, ...] = tuple(GUROBI_INSTANCE_NAMES),
    time_limit: float = GUROBI_TIME_LIMIT,
    big_M: float | None = GUROBI_BIG_M,
    log_dir: str | Path = GUROBI_LOG_DIR,
    output_dir: str | Path = GUROBI_OUTPUT_DIR,
    summary_csv: str | Path = GUROBI_SUMMARY_CSV,
    retry_failed: bool = GUROBI_RETRY_FAILED,
) -> None:
    completed = _read_completed_instances(summary_csv, retry_failed=retry_failed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance_name in instance_names:
        if instance_name in completed:
            print(f"[skip] {instance_name}: already recorded in {summary_csv}")
            continue

        print("=" * 72)
        print(f"[run] {instance_name}")
        print("=" * 72)
        start_time = time.time()
        try:
            inst = load_instance(instance_name)
            mdl, sol_exists, log_path = solve_sofjspt_instance(
                inst=inst,
                instance_name=instance_name,
                time_limit=time_limit,
                big_M=big_M,
                log_dir=log_dir,
                output_dir=output_dir,
            )
            row = getattr(mdl, "_blcme_summary_row")
            _append_summary_row(summary_csv, row)
            print(
                f"[done] {instance_name} | status={row['status']} | "
                f"obj={row['best_objective']} | bound={row['best_bound']} | "
                f"gap={row['mip_gap']} | time={row['wall_time_s']:.2f}s"
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            row = failed_summary_row(
                instance_name=instance_name,
                error=error + "\n" + traceback.format_exc(),
                wall_time_s=time.time() - start_time,
                time_limit=time_limit,
                big_M=big_M,
            )
            _append_summary_row(summary_csv, row)
            print(f"[failed] {instance_name}: {error}")


if __name__ == "__main__":
    solve_gurobi_instances(
        instance_names=GUROBI_INSTANCE_NAMES,
        time_limit=GUROBI_TIME_LIMIT,
        big_M=GUROBI_BIG_M,
        log_dir=GUROBI_LOG_DIR,
        output_dir=GUROBI_OUTPUT_DIR,
        summary_csv=GUROBI_SUMMARY_CSV,
        retry_failed=GUROBI_RETRY_FAILED,
    )
