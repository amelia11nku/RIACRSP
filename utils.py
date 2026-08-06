from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== Type aliases =====
SolutionType = Tuple[int, float, List[int], List[int], List[int], List[int]]
ResultType = List[Dict[str, object]]


# =========================
# I/O helpers
# =========================

def _ensure_parent_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    parent = os.path.dirname(path)
    os.makedirs(parent or ".", exist_ok=True)


def _job_id_from_op(op_id: Optional[str]) -> int:
    """Parse job id from operation id like 'o3_2' -> 3; returns -1 on failure."""
    if not op_id:
        return -1
    try:
        return int(str(op_id)[1:].split("_")[0])
    except Exception:
        return -1


def save_schedule_results(
    schedule_results: List[Dict],
    agv_w_file: str,
    agv_f_file: str,
    machine_file: str,
) -> None:
    """Save schedule_results into three CSV files (AGV_W / AGV_F / Machine)."""
    _ensure_parent_dir(agv_w_file)
    _ensure_parent_dir(agv_f_file)
    _ensure_parent_dir(machine_file)

    agv_w_headers = [
        "AGV_ID", "Operation", "Job", "Start",
        "Pickup_Arrive", "Pickup_Leave",
        "Delivery_Arrive", "Delivery_Complete",
        "From", "Pickup_Node", "Delivery_Node",
    ]
    agv_f_headers = [
        "AGV_ID", "Operation", "Job", "Start",
        "Delivery_Arrive", "Return_Complete",
        "From", "Delivery_Node",
    ]
    machine_headers = ["Machine", "Operation", "Job", "Start", "Finish"]

    agv_w_rows: List[Tuple[Tuple[int, float, float], List[object]]] = []
    agv_f_rows: List[Tuple[Tuple[int, float], List[object]]] = []
    machine_rows: List[Tuple[Tuple[int, float], List[object]]] = []

    for rec in schedule_results:
        dev = str(rec.get("device", "")).upper()
        op = rec.get("operation")
        job = _job_id_from_op(op)

        if dev.startswith("AGV_W"):
            agv_id = int(dev.replace("AGV_W", ""))
            st = float(rec.get("start_time", 0.0))
            pa = float(rec.get("pickup_arrive", st))
            pl = float(rec.get("pickup_leave", pa))
            da = float(rec.get("delivery_arrive", pl))
            dc = float(rec.get("delivery_complete", da))
            row = [
                agv_id, op, job,
                st, pa, pl, da, dc,
                rec.get("from_location"),
                rec.get("pickup_location"),
                rec.get("delivery_location"),
            ]
            agv_w_rows.append(((agv_id, dc, st), row))

        elif dev.startswith("AGV_F"):
            agv_id = int(dev.replace("AGV_F", ""))
            st = float(rec.get("start_time", 0.0))
            arr = float(rec.get("delivery_arrive", st))
            ret = float(rec.get("return_complete", arr))
            row = [
                agv_id, op, job,
                st, arr, ret,
                rec.get("from_location"),
                rec.get("delivery_location"),
            ]
            agv_f_rows.append(((agv_id, st), row))

        elif dev.startswith("M"):
            m_id = int(dev.replace("M", ""))
            st = float(rec.get("start_time", 0.0))
            fn = float(rec.get("end_time", st))
            row = [m_id, op, job, st, fn]
            machine_rows.append(((m_id, st), row))

    with open(agv_w_file, "w", encoding="utf-8") as f:
        f.write(",".join(agv_w_headers) + "\n")
        for _, row in sorted(agv_w_rows, key=lambda x: (x[0][0], x[0][1], x[0][2])):
            f.write(",".join(str(x) for x in row) + "\n")

    with open(agv_f_file, "w", encoding="utf-8") as f:
        f.write(",".join(agv_f_headers) + "\n")
        for _, row in sorted(agv_f_rows, key=lambda x: (x[0][0], x[0][1])):
            f.write(",".join(str(x) for x in row) + "\n")

    with open(machine_file, "w", encoding="utf-8") as f:
        f.write(",".join(machine_headers) + "\n")
        for _, row in sorted(machine_rows, key=lambda x: (x[0][0], x[0][1])):
            f.write(",".join(str(x) for x in row) + "\n")


def save_best_solution(
    best_solution: SolutionType,
    order: List[str],
    output_file: str,
    best_generation: int,
    Is_dcga: bool = False,
) -> None:
    """Save best solution summary to a single-row CSV."""
    _ensure_parent_dir(output_file)

    if Is_dcga:
        solution_id, makespan, machine_string, schedule_string = best_solution  # type: ignore[misc]
        result = OrderedDict([
            ("迭代次数", best_generation),
            ("最优解编号", solution_id),
            ("完工时间", makespan),
            ("机器分配", machine_string),
            ("调度顺序", schedule_string),
            ("工序处理顺序", order),
        ])
    else:
        solution_id, makespan, machine_string, agv_w_string, agv_f_string, schedule_string = best_solution
        result = OrderedDict([
            ("迭代次数", best_generation),
            ("最优解编号", solution_id),
            ("完工时间", makespan),
            ("机器分配", machine_string),
            ("AGV_W分配", agv_w_string),
            ("AGV_F分配", agv_f_string),
            ("调度顺序", schedule_string),
            ("工序处理顺序", order),
        ])

    pd.DataFrame([result]).to_csv(output_file, index=False, encoding="utf-8-sig")


# =========================
# Gantt helpers
# =========================

def _to_float(x) -> Optional[float]:
    """Convert blank/mixed values to float; returns None if not convertible."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_job(op: Optional[str]) -> Optional[int]:
    """Parse job id from operation id like 'o3_2' -> 3; returns None on failure."""
    if not op:
        return None
    try:
        s = str(op).strip()
        if s and s[0] in {"o", "O"}:
            return int(s[1:].split("_")[0])
    except Exception:
        pass
    return None


def _parse_device(device: str) -> Tuple[str, Optional[int]]:
    """Parse device string to (lane, resource_id)."""
    if not device:
        return ("Unknown", None)

    s = str(device).strip().upper()

    if s.startswith("M") and not s.startswith("MC"):
        try:
            return ("Machine", int(s[1:]))
        except Exception:
            return ("Machine", None)

    if s.startswith("AGV_W"):
        try:
            return ("AGV_W", int(s.replace("AGV_W", "")))
        except Exception:
            return ("AGV_W", None)

    if s.startswith("AGV_F"):
        try:
            return ("AGV_F", int(s.replace("AGV_F", "")))
        except Exception:
            return ("AGV_F", None)

    return ("Unknown", None)


def gantt_dataframe_from_schedule_results(schedule_results: List[Dict]) -> pd.DataFrame:
    """Convert schedule_results to a long-form DataFrame for plotting."""
    rows: List[Dict[str, object]] = []

    for rec in schedule_results:
        lane, rid = _parse_device(rec.get("device", ""))
        op = rec.get("operation")
        job = _parse_job(op)

        if lane == "Machine":
            rows.append({
                "lane": lane,
                "resource_id": rid,
                "job": job,
                "op": op,
                "start": _to_float(rec.get("start_time")),
                "finish": _to_float(rec.get("end_time")),
                "kind": "proc",
                "w_empty_start": None, "w_empty_finish": None,
                "w_wait_start": None,  "w_wait_finish": None,
                "w_loaded_start": None, "w_loaded_finish": None,
                "f_start": None, "f_arrival": None, "f_return": None,
                "wpos": None, "src": None, "dst": None,
            })

        elif lane == "AGV_W":
            s = _to_float(rec.get("start_time"))
            ap = _to_float(rec.get("pickup_arrive"))
            dp = _to_float(rec.get("pickup_leave"))
            arr = _to_float(rec.get("delivery_arrive"))
            done = _to_float(rec.get("delivery_complete")) or arr

            # w_empty_s, w_empty_e = (s, ap) if (s is not None and ap is not None and ap > s) else (None, None)
            # w_wait_s, w_wait_e = (ap, dp) if (ap is not None and dp is not None and dp > ap) else (None, None)
            # w_load_s, w_load_e = (dp, done) if (dp is not None and done is not None and done > dp) else (None, None)

            w_empty_s, w_empty_e = (s, ap) if (s is not None and ap is not None) else (None, None)
            w_wait_s, w_wait_e = (ap, dp) if (ap is not None and dp is not None) else (None, None)
            w_load_s, w_load_e = (dp, done) if (dp is not None and done is not None) else (None, None)
            
            rows.append({
                "lane": lane,
                "resource_id": rid,
                "job": job,
                "op": op,
                "start": None,
                "finish": None,
                "kind": "transport",
                "w_empty_start": w_empty_s, "w_empty_finish": w_empty_e,
                "w_wait_start": w_wait_s, "w_wait_finish": w_wait_e,
                "w_loaded_start": w_load_s, "w_loaded_finish": w_load_e,
                "f_start": None, "f_arrival": None, "f_return": None,
                "wpos": rec.get("from_location"),
                "src": rec.get("pickup_location"),
                "dst": rec.get("delivery_location"),
            })

        elif lane == "AGV_F":
            st = _to_float(rec.get("start_time"))
            arr = _to_float(rec.get("delivery_arrive"))
            ret = _to_float(rec.get("return_complete"))

            rows.append({
                "lane": lane,
                "resource_id": rid,
                "job": job,
                "op": op,
                "start": st,
                "finish": ret,
                "kind": "transport",
                "w_empty_start": None, "w_empty_finish": None,
                "w_wait_start": None,  "w_wait_finish": None,
                "w_loaded_start": None, "w_loaded_finish": None,
                "f_start": st, "f_arrival": arr, "f_return": ret,
                "wpos": None,
                "src": rec.get("from_location"),
                "dst": rec.get("delivery_location"),
            })

    df = pd.DataFrame(rows, columns=[
        "lane", "resource_id", "job", "op",
        "start", "finish", "kind",
        "w_empty_start", "w_empty_finish",
        "w_wait_start", "w_wait_finish",
        "w_loaded_start", "w_loaded_finish",
        "f_start", "f_arrival", "f_return",
        "wpos", "src", "dst",
    ])

    df.sort_values(
        ["lane", "resource_id", "start", "w_empty_start", "w_loaded_start", "job", "op"],
        inplace=True,
        ignore_index=True,
    )
    return df


def export_gantt_csv(df: pd.DataFrame, save_path: str) -> None:
    """Export the long-form gantt DataFrame to CSV."""
    _ensure_parent_dir(save_path)
    df.to_csv(save_path, index=False, encoding="utf-8")


def _lane_order_key(lane: str) -> int:
    """Lane ordering for the three swimlanes."""
    return {"Machine": 0, "AGV_W": 1, "AGV_F": 2}.get(lane, 99)


def plot_gantt_three_swimlanes(
    df: pd.DataFrame,
    title: str = "Schedule Gantt Chart for OSF-FJSPDT",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """Plot a three-swimlane Gantt chart from the long-form DataFrame."""
    if df.empty:
        raise ValueError("Empty DataFrame for Gantt plot.")
    # ===== 强制 Times New Roman（所有文本 & 数字）=====
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,                 # 统一字号（可调）
        "mathtext.fontset": "stix",       # 数学符号风格，和 TNR 最协调
        "axes.unicode_minus": False,      # 负号正常显示
    })
    lanes = sorted(df["lane"].unique(), key=_lane_order_key)
    jobs = sorted(df.loc[df["job"].notna(), "job"].unique())
    color_map = {job: plt.cm.tab20((int(job) - 1) % 20) for job in jobs}

    y_base = 0.0
    lane_gap = 1.2
    bar_h = 0.8

    fig, ax = plt.subplots(figsize=figsize)

    for lane in lanes:
        df_lane = df[df["lane"] == lane]
        resources = list(sorted(df_lane["resource_id"].dropna().unique()))

        for rid in resources:
            df_row = df_lane[df_lane["resource_id"] == rid]

            for _, r in df_row.iterrows():
                job = int(r["job"]) if pd.notna(r["job"]) else -1
                color = color_map.get(job, (0.3, 0.3, 0.3, 0.8))

                if lane == "Machine":
                    st, fn = r["start"], r["finish"]
                    if st is not None and fn is not None and fn > st:
                        ax.barh(y_base, fn - st, left=st, height=bar_h,
                                color=color, edgecolor="black", linewidth=0.4)
                        ax.text((st + fn) / 2, y_base, r["op"], ha="center", va="center", fontsize=7)

                elif lane == "AGV_W":
                    es, ee = r["w_empty_start"], r["w_empty_finish"]
                    if es is not None and ee is not None and ee > es:
                        ax.barh(y_base, ee - es, left=es, height=bar_h,
                                color=(0.7, 0.7, 0.7, 1.0), edgecolor="black", linewidth=0.4)

                    ws, we = r["w_wait_start"], r["w_wait_finish"]
                    if ws is not None and we is not None and we > ws:
                        ax.barh(y_base, we - ws, left=ws, height=bar_h,
                                color="white", edgecolor="black", linewidth=0.8, linestyle="--")

                    ls, le = r["w_loaded_start"], r["w_loaded_finish"]
                    if ls is not None and le is not None and le > ls:
                        ax.barh(y_base, le - ls, left=ls, height=bar_h,
                                color=color, edgecolor="black", linewidth=0.4)

                    segs = [(es, ee), (ws, we), (ls, le)]
                    segs = [(s, e) for (s, e) in segs if s is not None and e is not None and e > s]
                    if segs:
                        st = min(s for s, _ in segs)
                        fn = max(e for _, e in segs)
                        ax.text((st + fn) / 2, y_base, str(r["op"]), ha="center", va="center", fontsize=6)

                elif lane == "AGV_F":
                    fls, fle = r["f_start"], r["f_arrival"]
                    if fls is not None and fle is not None and fle > fls:
                        ax.barh(y_base, fle - fls, left=fls, height=bar_h,
                                color=color, edgecolor="black", linewidth=0.4)

                    fes, fee = r["f_arrival"], r["f_return"]
                    if fes is not None and fee is not None and fee > fes:
                        ax.barh(y_base, fee - fes, left=fes, height=bar_h,
                                color=(0.7, 0.7, 0.7, 1.0), edgecolor="black", linewidth=0.4)

                    segs = [(fls, fle), (fes, fee)]
                    segs = [(s, e) for (s, e) in segs if s is not None and e is not None and e > s]
                    if segs:
                        st = min(s for s, _ in segs)
                        fn = max(e for _, e in segs)
                        ax.text((st + fn) / 2, y_base, str(r["op"]), ha="center", va="center", fontsize=6)

            ax.text(
                -0.01, y_base, f"{lane}-{int(rid)}",
                ha="right", va="center", fontsize=8,
                transform=ax.get_yaxis_transform(),
            )
            y_base += 1.0

        y_base += lane_gap

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
