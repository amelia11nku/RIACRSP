#!/usr/bin/env python3
"""Run and summarize the preregistered paper small-scale Gurobi campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.export import ResourceTimelineExporter  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.exact.general_gurobi import solve_general_gurobi  # noqa: E402
from scripts.plot_schedule_gantt import plot_resource_timeline  # noqa: E402


CONFIG_PATH = ROOT / "configs/paper_core_small_gurobi.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(finite_or_none(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def instance_characteristics(instance) -> dict[str, Any]:
    direct_edges = sum(
        len(instance.product_data[product].precedence)
        for product in instance.products
    )
    possible_edges = sum(
        len(instance.product_data[product].operations)
        * (len(instance.product_data[product].operations) - 1)
        / 2
        for product in instance.products
    )
    eligibilities = [
        len(instance.operation_data[op].eligible_islands)
        for op in instance.operations
    ]
    return {
        "n_products": len(instance.products),
        "n_operations": len(instance.operations),
        "n_islands": len(instance.islands),
        "n_w_agvs": len(instance.agvs_w),
        "n_f_agvs": len(instance.agvs_f),
        "n_configurations": len(instance.configurations),
        "precedence_edges": direct_edges,
        "precedence_density": 0.0 if possible_edges == 0 else direct_edges / possible_edges,
        "eligibility_mean": sum(eligibilities) / len(eligibilities),
        "eligibility_min": min(eligibilities),
        "eligibility_max": max(eligibilities),
        "eligibility_fraction_mean": sum(eligibilities)
        / (len(eligibilities) * len(instance.islands)),
    }


def verify_protocol(config: dict[str, Any]) -> None:
    if config.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise RuntimeError("paper Gurobi protocol is not frozen before execution")
    expected_ids = {
        f"CB1_CORE_S_CF{cf}_R{replicate:02d}"
        for cf in range(1, 4)
        for replicate in range(1, 6)
    }
    configured_ids = {case["instance_id"] for case in config["runs"]}
    if configured_ids != expected_ids or len(config["runs"]) != 15:
        raise RuntimeError("paper protocol must contain exactly the fixed 15 Core-S cases")
    for case in config["runs"]:
        path = ROOT / case["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"missing selected instance: {path}")
        instance = load_instance(path)
        if instance.instance_id != case["instance_id"]:
            raise RuntimeError(f"instance ID mismatch for {path}")
        expected_path = (
            Path("instances/controlled/RCIAS-CB1/core") / f"{instance.instance_id}.json"
        )
        if (
            case["suite"] != "core_small"
            or Path(case["relative_path"]) != expected_path
            or instance.metadata.get("suite") != "CORE"
            or instance.metadata.get("scale") != "S"
        ):
            raise RuntimeError(f"case is not a fixed CB1-Core Small instance: {instance.instance_id}")


def write_pre_run_inventory(config: dict[str, Any], output_root: Path) -> None:
    rows = []
    for case in config["runs"]:
        instance = load_instance(ROOT / case["relative_path"])
        rows.append({
            "instance_id": instance.instance_id,
            "suite": "CB1-Core",
            "size": "S",
            "cf_level": case["cf_level"],
            "replicate": case["replicate"],
            **instance_characteristics(instance),
            "variable_count": None,
            "constraint_count": None,
        })
    atomic_csv(rows, output_root / "pre_run_instance_inventory.csv")


def run_case(case: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    instance_path = ROOT / case["relative_path"]
    instance = load_instance(instance_path)
    run_dir = output_root / "runs" / case["suite"] / instance.instance_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"SKIP existing result {instance.instance_id}", flush=True)
        return load_json(result_path)

    log_path = output_root / "logs" / case["suite"] / f"{instance.instance_id}.log"
    solution_path = run_dir / "solution.json"
    feasibility_path = run_dir / "feasibility.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = config["gurobi"]
    characteristics = instance_characteristics(instance)
    base = {
        "schema": "paper-gurobi-run-v1",
        "commit_sha": git_head(),
        "protocol_sha256": sha256(CONFIG_PATH),
        "exact_model_sha256": sha256(ROOT / parameters["model"]),
        "instance_sha256": sha256(instance_path),
        "instance_id": instance.instance_id,
        "suite": case["suite"],
        "purpose": "formal paper exact comparison on fixed CB1-Core Small stratum",
        "size": "S",
        "cf_level": case["cf_level"],
        "replicate": case["replicate"],
        **characteristics,
        "gurobi_version": None,
        "seed": int(parameters["seed"]),
        "threads": int(parameters["threads"]),
        "time_limit_seconds": float(parameters["time_limit_seconds"]),
        "mip_gap_target": float(parameters["mip_gap"]),
        "h1_mip_start_used": bool(parameters["use_h1_mip_start"]),
        "status": "ENVIRONMENT_ERROR",
        "optimality_proven": False,
        "objective_makespan": None,
        "best_bound": None,
        "reported_gap": None,
        "solver_runtime_seconds": None,
        "total_runtime_seconds": None,
        "node_count": None,
        "variable_count": None,
        "constraint_count": None,
        "h1_makespan": None,
        "h1_gap_to_opt_percent": None,
        "replay_makespan": None,
        "replay_feasible": False,
        "native_replay_equal": False,
        "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "solution_path": str(solution_path.relative_to(ROOT)).replace("\\", "/"),
        "feasibility_path": str(feasibility_path.relative_to(ROOT)).replace("\\", "/"),
        "error": None,
    }
    print(
        "RUN "
        f"instance={instance.instance_id} purpose=formal Core-S exact comparison "
        f"time_limit={parameters['time_limit_seconds']}s; result tests exact status, "
        "native/replay makespan equality, and independent feasibility.",
        flush=True,
    )
    started = time.perf_counter()
    try:
        result = solve_general_gurobi(
            instance,
            time_limit_seconds=float(parameters["time_limit_seconds"]),
            seed=int(parameters["seed"]),
            threads=int(parameters["threads"]),
            mip_gap=float(parameters["mip_gap"]),
            log_file=str(log_path),
            use_h1_mip_start=bool(parameters["use_h1_mip_start"]),
        )
        audit = check_schedule(instance, result.schedule)
        native_replay_equal = math.isclose(
            result.solver_makespan, result.replay_makespan, rel_tol=0.0, abs_tol=1e-6
        )
        if not audit["feasible"] or not native_replay_equal:
            raise RuntimeError(
                f"post-solve verification failed: feasible={audit['feasible']} "
                f"native_replay_equal={native_replay_equal}"
            )
        h1_gap = (
            100.0 * (result.h1_upper_bound - result.solver_makespan) / result.solver_makespan
            if result.optimality_proven else None
        )
        base.update({
            "gurobi_version": result.solver_version,
            "status": result.status,
            "optimality_proven": result.optimality_proven,
            "objective_makespan": result.solver_makespan,
            "best_bound": finite_or_none(result.best_bound),
            "reported_gap": finite_or_none(result.gap),
            "solver_runtime_seconds": result.runtime_seconds,
            "total_runtime_seconds": result.total_runtime_seconds,
            "node_count": result.node_count,
            "variable_count": result.variable_count,
            "constraint_count": result.constraint_count,
            "h1_makespan": result.h1_upper_bound,
            "h1_gap_to_opt_percent": h1_gap,
            "replay_makespan": result.replay_makespan,
            "replay_feasible": bool(audit["feasible"]),
            "native_replay_equal": native_replay_equal,
        })
        atomic_json(result.to_dict(), solution_path)
        atomic_json(audit, feasibility_path)
        exporter = ResourceTimelineExporter(instance, result.schedule)
        exporter.export(run_dir / "schedule")
    except Exception as error:
        base.update({
            "status": "LICENSE_LIMIT" if "size-limited license" in str(error) else "ENVIRONMENT_ERROR",
            "total_runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        })
        if not log_path.exists():
            log_path.write_text(base["error"] + "\n", encoding="utf-8")
    atomic_json(base, result_path)
    print(
        f"DONE instance={instance.instance_id} status={base['status']} "
        f"objective={base['objective_makespan']} bound={base['best_bound']} "
        f"gap={base['reported_gap']} replay_feasible={base['replay_feasible']}",
        flush=True,
    )
    return base


def fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def plot_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if row["objective_makespan"] is not None]
    if valid:
        labels = [row["instance_id"].replace("CB1_CORE_", "") for row in valid]
        objectives = [row["objective_makespan"] for row in valid]
        bounds = [
            row["best_bound"] if row["best_bound"] is not None else 0.0
            for row in valid
        ]
        h1 = [row["h1_makespan"] for row in valid]
        x = list(range(len(valid)))
        width = 0.26
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.bar([value - width for value in x], bounds, width, label="Best bound")
        ax.bar(x, objectives, width, label="Gurobi incumbent/optimum")
        ax.bar([value + width for value in x], h1, width, label="H1")
        ax.set_ylabel("Makespan (lower is better)")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "solution_quality_vs_exact.png", dpi=220)
        fig.savefig(figures / "solution_quality_vs_exact.pdf")
        plt.close(fig)

    sized = [row for row in rows if row["variable_count"] is not None]
    if sized:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for row in sized:
            ax.scatter(row["n_operations"], row["solver_runtime_seconds"], s=55)
            ax.annotate(
                row["instance_id"].replace("CB1_CORE_", ""),
                (row["n_operations"], row["solver_runtime_seconds"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8,
            )
        ax.set_xlabel("Number of operations")
        ax.set_ylabel("Gurobi runtime (s)")
        if max(row["solver_runtime_seconds"] for row in sized) > 10:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "gurobi_computational_growth.png", dpi=220)
        fig.savefig(figures / "gurobi_computational_growth.pdf")
        plt.close(fig)


def write_report(config: dict[str, Any], rows: list[dict[str, Any]], output_root: Path) -> None:
    proven = sum(bool(row["optimality_proven"]) for row in rows)
    time_limited = sum(row["status"] == "TIME_LIMIT" for row in rows)
    all_replayed = bool(rows) and all(
        row["objective_makespan"] is not None
        and row["replay_feasible"]
        and row["native_replay_equal"]
        for row in rows
    )
    campaign_complete = len(rows) == len(config["runs"]) and all_replayed
    lines = [
        "# Paper Small-Scale Gurobi Validation",
        "",
        f"Base commit: `{config['source_commit']}`",
        "",
        f"Protocol SHA-256: `{sha256(CONFIG_PATH)}`",
        "",
        f"Selection rule: {config['selection_rule']}",
        "",
        "Historical Tiny anchors were reused without rerunning: "
        + ", ".join(f"`{key}={value:g}`" for key, value in config["historical_tiny_anchors"].items())
        + ".",
        "",
        "The current workstation smoke test separately proved `tiny_02=57` with Gurobi 13.0.3 and a feasible production replay; it is not part of the formal Core-S table.",
        "",
        "## Exact-run results",
        "",
        "| Instance | Size (J/O/M/W/F/C) | Status | Opt./inc. | Bound | Solver gap | Runtime (s) | H1 | H1 gap to optimum | Replay |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        size = "/".join(str(row[key]) for key in (
            "n_products", "n_operations", "n_islands", "n_w_agvs", "n_f_agvs", "n_configurations"
        ))
        gap = None if row["reported_gap"] is None else 100.0 * row["reported_gap"]
        lines.append(
            f"| `{row['instance_id']}` | {size} | {row['status']} | "
            f"{fmt(row['objective_makespan'])} | {fmt(row['best_bound'])} | "
            f"{fmt(gap)}% | {fmt(row['solver_runtime_seconds'])} | "
            f"{fmt(row['h1_makespan'])} | {fmt(row['h1_gap_to_opt_percent'])}% | "
            f"{'PASS' if row['replay_feasible'] and row['native_replay_equal'] else '--'} |"
        )
    lines.extend([
        "",
        "## Integrity summary",
        "",
        f"- Formal Core-S runs: {len(rows)}/{len(config['runs'])}; proven optima: {proven}; time-limit cases: {time_limited}.",
        f"- Every returned incumbent passed production replay and the independent checker: `{str(all_replayed).upper()}`.",
        "- H1 gap to optimum is reported only when Gurobi proved optimality.",
        "- Only the newly authorized formal Core-S stratum was accessed; Core-M/L, Sensitivity, Legacy, DEV-HOLDOUT, and CAL-HOLDOUT were not accessed.",
        "",
        "## Artifacts",
        "",
        f"- Machine-readable summary: `{(output_root / 'gurobi_results.csv').relative_to(ROOT).as_posix()}`",
        f"- Instance inventory: `{(output_root / 'instance_inventory.csv').relative_to(ROOT).as_posix()}`",
        f"- Pre-run selection inventory: `{(output_root / 'pre_run_instance_inventory.csv').relative_to(ROOT).as_posix()}`",
        f"- Solver logs: `{(output_root / 'logs').relative_to(ROOT).as_posix()}`",
        f"- Full solutions and feasibility reports: `{(output_root / 'runs').relative_to(ROOT).as_posix()}`",
        f"- Figures: `{(output_root / 'figures').relative_to(ROOT).as_posix()}`",
        "",
        "`PAPER_CORE_SMALL_GUROBI_COMPLETE = TRUE`" if campaign_complete else "`PAPER_CORE_SMALL_GUROBI_COMPLETE = FALSE`",
        "",
    ])
    report_path = ROOT / config["reporting"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def summarize(config: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for case in config["runs"]:
        path = output_root / "runs" / case["suite"] / case["instance_id"] / "result.json"
        if path.exists():
            rows.append(load_json(path))
    if not rows:
        raise RuntimeError("no paper Gurobi results are available to summarize")
    columns = [
        "commit_sha", "instance_id", "suite", "size", "cf_level", "replicate", "n_products", "n_operations",
        "n_islands", "n_w_agvs", "n_f_agvs", "n_configurations",
        "precedence_edges", "precedence_density", "eligibility_mean",
        "eligibility_min", "eligibility_max", "eligibility_fraction_mean",
        "gurobi_version", "seed", "threads", "time_limit_seconds",
        "mip_gap_target", "h1_mip_start_used", "status", "optimality_proven",
        "objective_makespan", "best_bound", "reported_gap",
        "solver_runtime_seconds", "total_runtime_seconds", "node_count",
        "variable_count", "constraint_count", "h1_makespan",
        "h1_gap_to_opt_percent", "replay_makespan", "replay_feasible",
        "native_replay_equal", "log_path", "solution_path", "feasibility_path", "error",
    ]
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    atomic_csv(normalized, output_root / "gurobi_results.csv")
    inventory_columns = [
        "instance_id", "suite", "size", "cf_level", "replicate", "n_products", "n_operations", "n_islands",
        "n_w_agvs", "n_f_agvs", "n_configurations", "precedence_edges",
        "precedence_density", "eligibility_mean", "eligibility_min",
        "eligibility_max", "eligibility_fraction_mean", "variable_count",
        "constraint_count",
    ]
    atomic_csv(
        [{key: row.get(key) for key in inventory_columns} for row in rows],
        output_root / "instance_inventory.csv",
    )
    plot_results(rows, output_root)
    write_report(config, rows, output_root)

    representative = next(
        (row for row in rows if row["replay_feasible"]),
        None,
    )
    if representative is not None:
        resource = (
            output_root / "runs" / representative["suite"] / representative["instance_id"]
            / "schedule" / "resource_timeline.csv"
        )
        if resource.exists():
            plot_resource_timeline(
                resource,
                output_root / "figures" / "representative_schedule_gantt.png",
                output_root / "figures" / "representative_schedule_gantt.pdf",
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    verify_protocol(config)
    output_root = ROOT / config["reporting"]["output_root"]
    write_pre_run_inventory(config, output_root)
    if args.inventory_only:
        print("PAPER_CORE_SMALL_PRE_RUN_INVENTORY_COMPLETE count=15", flush=True)
        return
    selected = set(args.case)
    if not args.summarize_only:
        for case in config["runs"]:
            if selected and case["instance_id"] not in selected:
                continue
            run_case(case, config, output_root)
    if args.no_summarize:
        return
    rows = summarize(config, output_root)
    errors = [row for row in rows if row["status"] in {"ENVIRONMENT_ERROR", "LICENSE_LIMIT"}]
    if errors:
        raise RuntimeError({row["instance_id"]: row["error"] for row in errors})
    complete = len(rows) == len(config["runs"])
    print(
        f"PAPER_CORE_SMALL_GUROBI_{'COMPLETE' if complete else 'PARTIAL'} runs={len(rows)}/{len(config['runs'])} "
        f"proven={sum(bool(row['optimality_proven']) for row in rows)} "
        f"time_limit={sum(row['status'] == 'TIME_LIMIT' for row in rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
