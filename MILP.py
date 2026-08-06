import sys
import os
import traceback
from data.loader import load_instance

# 你前面已经有 solve_sofjspt_instance 的定义，这里直接用即可
from gurobi import solve_sofjspt_instance  # 如果单独脚本就用这种

INSTANCES = [
    'MK01', 'MK02', 'MK03', 'MK04', 'MK05',
    'MK06', 'MK07', 'MK08', 'MK09', 'MK10',
    'MK11', 'MK12', 'MK13', 'MK14', 'MK15',
]


def run_batch_instances(
    instances,
    base_module_prefix="data.inst",
    time_limit=7200,
    big_M=10000,
    log_dir="./logs_gurobi_batch",
):
    """
    逐个实例用 Gurobi 求解，并为每个实例生成独立 log 文件。
    log 文件名格式：gurobi_<instance>_YYYYmmdd_HHMMSS.log
    （具体名字由 solve_sofjspt_instance 决定，这里只是统一 log_dir）
    """
    os.makedirs(log_dir, exist_ok=True)

    summary = []

    for name in instances:
        print("\n" + "=" * 70)
        print(f"开始求解实例: {name}")
        print("=" * 70)

        # 1) 组装模块路径，例如 data.instances.kumar01 / data.instances.MK01
        inst_name = name
        try:
            inst = load_instance(inst_name)
            print(f"  -> 成功加载实例模块: {inst_name}")
        except Exception as e:
            print(f"  -> 加载实例模块失败: {inst_name}")
            traceback.print_exc()
            summary.append((name, "LOAD_FAILED", None, None, None))
            continue

        try:
            # 2) 调用之前写好的 gurobi 求解函数
            mdl, sol_exists, log_path = solve_sofjspt_instance(
                inst,
                instance_name=name,
                time_limit=time_limit,
                big_M=big_M,
                log_dir=log_dir,
            )

            # 3) 汇总状态
            status_code = mdl.Status
            if status_code == 2:      # GRB.OPTIMAL
                status_str = "OPTIMAL"
            elif status_code == 9:    # GRB.TIME_LIMIT
                status_str = "TIME_LIMIT"
            elif status_code == 3:    # INFEASIBLE
                status_str = "INFEASIBLE"
            else:
                status_str = f"STATUS_{status_code}"

            obj_val = mdl.ObjVal if sol_exists else None
            best_bound = mdl.ObjBound if sol_exists else None
            mip_gap = mdl.MIPGap if sol_exists else None

            print(f"  -> 求解状态: {status_str}")
            if sol_exists:
                print(f"  -> 最好可行解: {obj_val:.4f}")
                print(f"  -> 最优下界:   {best_bound:.4f}")
                if mip_gap is not None:
                    print(f"  -> MIPGap:    {mip_gap*100:.2f}%")
            else:
                print("  -> 未找到可行解")

            print(f"  -> 日志文件: {log_path}")

            summary.append((name, status_str, obj_val, best_bound, log_path))

        except Exception as e:
            print(f"  -> 求解实例 {name} 时出错：")
            traceback.print_exc()
            summary.append((name, "SOLVE_ERROR", None, None, None))
            continue

    # 4) 打印汇总表
    print("\n" + "#" * 70)
    print("批量求解汇总：")
    print("#" * 70)
    for (name, status, obj, bound, log_path) in summary:
        if obj is not None and bound is not None:
            gap_str = "-"
            try:
                gap_str = f"{(obj - bound) / obj * 100:.2f}%"
            except Exception:
                pass
            print(
                f"{name:8s} | {status:12s} | Obj={obj!s:>8} | "
                f"Bound={bound!s:>8} | Gap≈{gap_str:>8} | log={log_path}"
            )
        else:
            print(
                f"{name:8s} | {status:12s} | Obj=   -    | "
                f"Bound=   -    | Gap=   -    | log={log_path}"
            )


if __name__ == "__main__":
    # 如果只想批量跑，直接调用批量函数即可
    run_batch_instances(
        INSTANCES,
        base_module_prefix="data.inst",
        time_limit=7200,
        big_M=10000,
        log_dir="./logs_gurobi_batch",
    )

    print("\n--- 批量求解完成 ---")
