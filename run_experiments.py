"""Run benchmark comparisons with resumable per-instance result files.

Edit the CONFIG section to choose instances, algorithms, run counts, and
algorithm parameters. Each instance gets one source-data CSV containing all
algorithm/run records. The analysis step reads those CSV files and produces a
wide summary table similar to results_variant_0516.csv.
"""

from __future__ import annotations

import csv
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.loader import load_instance
from logger_config import setup_file_logger

from BLEA import BLEA
from DCGA import DCGA
from EMA import EMA
from IGA import IGA
from LMEO import LMEO
from BLCMA import BLCMA
from BLCMAnoi import BLCMA as BLCMA_noi
from BLCMAnoq import BLCMA as BLCMA_noq
from BLCMAnox import BLCMA as BLCMA_nox
from BLCMAnoe import BLCMA as BLCMA_noe
from simulate import Calculate

# =============================================================================
# CONFIG: edit this section for each experiment
# =============================================================================

RESULTS_DIR = Path("experiment_results") / "comparison"
SOURCE_DIR = RESULTS_DIR / "source_by_instance"
SUMMARY_CSV = RESULTS_DIR / "summary_comparison.csv"
SUMMARY_XLSX = RESULTS_DIR / "summary_comparison.xlsx"
LONG_STATS_CSV = RESULTS_DIR / "per_instance_stats.csv"
STAT_TESTS_CSV = RESULTS_DIR / "statistical_tests.csv"
LOG_FILE = RESULTS_DIR / "run_experiments.log"

INSTANCE_NAMES = [
    "AFAISP-S01", "AFAISP-S02", "AFAISP-S03", "AFAISP-S04", "AFAISP-S05",
    "AFAISP-S06", "AFAISP-S07", "AFAISP-S08", "AFAISP-S09", "AFAISP-S10", 
    "AFAISP-S11", "AFAISP-S12", "AFAISP-S13", "AFAISP-S14", "AFAISP-S15",
    "AFAISP-S16", "AFAISP-S17", "AFAISP-S18", "AFAISP-S19", "AFAISP-S20",
    "AFAISP-M01", "AFAISP-M02", "AFAISP-M03", "AFAISP-M04", "AFAISP-M05",
    "AFAISP-M06", "AFAISP-M07", "AFAISP-M08", "AFAISP-M09", "AFAISP-M10",
    "AFAISP-M11", "AFAISP-M12", "AFAISP-M13", "AFAISP-M14", "AFAISP-M15",
    "AFAISP-M16", "AFAISP-M17", "AFAISP-M18", "AFAISP-M19", "AFAISP-M20",
    
    "AFAISP-L01", "AFAISP-L02", "AFAISP-L03", "AFAISP-L04", "AFAISP-L05",
    "AFAISP-L06", "AFAISP-L07", "AFAISP-L08", "AFAISP-L09", "AFAISP-L10",
    "AFAISP-L11", "AFAISP-L12", "AFAISP-L13", "AFAISP-L14", "AFAISP-L15",
    "AFAISP-L16", "AFAISP-L17", "AFAISP-L18", "AFAISP-L19", "AFAISP-L20",
    
]

ALGORITHM_NAMES = [
    "BLCMA",
    "BLCMAnoi",
    "BLCMAnoq",
    "BLCMAnox",
    "BLCMAnoe",
    "BLEA",
    # "DCGA",
    "LMEO",
    "IGA",
    "EMA",
]

N_RUNS = 20
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 122, 133, 144, 155, 166, 177, 188, 199, 211, 222]
RETRY_FAILED = True
RUN_EXPERIMENTS = True
RUN_ANALYSIS = True

COMMON_PARAMS = dict(
    max_iterations=100000,
    early_stop_patience=5000,
    run_time_ratio=2.0,
)

BLCMA_PARAMS = dict(
    **COMMON_PARAMS,
    population_size=150,
    explorer_ratio=0.3,
    pc_e=0.95,
    pm_e=0.30,
    pc_x=0.95,
    pm_x=0.30,
    elite_ratio=0.05,
    migration_interval=10,
    migration_rate=0.10,
    q_alpha_e=0.2,
    q_epsilon_e=0.1,
    q_alpha_x=0.2,
    q_epsilon_x=0.1,
    gamma_e=0.3,
    gamma_x=0.3,
)

ALGORITHM_PARAMS = {
    "BLCMA": BLCMA_PARAMS,
    "BLCMAnoi": BLCMA_PARAMS,
    "BLCMAnoq": BLCMA_PARAMS,
    "BLCMAnox": BLCMA_PARAMS,
    "BLCMAnoe": BLCMA_PARAMS,
    "BLEA": dict(**COMMON_PARAMS, population_size=100, delta=0.05, sl_learning_rate=0.8, el_pc=0.8, el_pm=0.15),
    "DCGA": dict(**COMMON_PARAMS, population_size=100, pc=0.9, pm=0.15),
    "LMEO": {**COMMON_PARAMS, "population_size": 60, "pc": 0.9, "pm": 0.05},
    "IGA": {**COMMON_PARAMS, "population_size": 80, "pc": 0.70, "pm": 0.30},
    "EMA": dict(**COMMON_PARAMS, population_size=100, pc=0.8, pm=0.25, local_search_rate=0.40),
}


# =============================================================================
# Experiment engine
# =============================================================================

SOURCE_COLUMNS = [
    "instance",
    "algorithm",
    "run",
    "seed",
    "best_makespan",
    "best_generation",
    "solve_time_s",
    "status",
    "error",
    "timestamp",
]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    cls: type
    method: str


ALGORITHMS = {
    "BLCMA": AlgorithmSpec("BLCMA", BLCMA, "evolve"),
    "BLCMAnoi": AlgorithmSpec("BLCMAnoi", BLCMA_noi, "evolve"),
    "BLCMAnoq": AlgorithmSpec("BLCMAnoq", BLCMA_noq, "evolve"),
    "BLCMAnox": AlgorithmSpec("BLCMAnox", BLCMA_nox, "evolve"),
    "BLCMAnoe": AlgorithmSpec("BLCMAnoe", BLCMA_noe, "evolve"),
    "BLEA": AlgorithmSpec("BLEA", BLEA, "optimize"),
    "DCGA": AlgorithmSpec("DCGA", DCGA, "evolve"),
    "LMEO": AlgorithmSpec("LMEO", LMEO, "evolve"),
    "IGA": AlgorithmSpec("IGA", IGA, "evolve"),
    "EMA": AlgorithmSpec("EMA", EMA, "evolve"),
}


SOURCE_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_file_logger("experiments", log_file_path=str(LOG_FILE))


def source_path(instance_name: str) -> Path:
    return SOURCE_DIR / f"{instance_name}.csv"


def read_source(instance_name: str) -> pd.DataFrame:
    path = source_path(instance_name)
    if not path.exists():
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    return pd.read_csv(path)


def append_source_row(instance_name: str, row: dict) -> None:
    path = source_path(instance_name)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SOURCE_COLUMNS})


def is_completed(df: pd.DataFrame, algorithm: str, run_id: int) -> bool:
    if df.empty:
        return False
    mask = (df["algorithm"] == algorithm) & (df["run"].astype(int) == int(run_id))
    if not mask.any():
        return False
    if RETRY_FAILED:
        return bool((df.loc[mask, "status"] == "ok").any())
    return True


def seed_for_run(run_id: int) -> int:
    if SEEDS:
        return int(SEEDS[(run_id - 1) % len(SEEDS)])
    return int(run_id)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _extract_best_from_solution(best_solution) -> float:
    if isinstance(best_solution, (tuple, list)) and len(best_solution) > 1:
        return float(best_solution[1])
    return float(best_solution)


def _audit_evolutionary_solution(instance, best_solution, reported_makespan: float) -> None:
    if not isinstance(best_solution, (tuple, list)) or len(best_solution) < 6:
        raise ValueError("Evolutionary solver did not return a complete best-solution tuple.")

    MS, TW, TF, OS = best_solution[2:6]
    n_ops = int(instance.num_operations)
    expected_os = set(range(1, n_ops + 1))
    if not (len(MS) == len(TW) == len(TF) == len(OS) == n_ops):
        raise ValueError("Best solution has an incomplete chromosome.")
    if len(set(OS)) != n_ops or set(OS) != expected_os:
        raise ValueError("Best solution OS is not a permutation of all operations.")

    operations = [
        f"o{job}_{seq}"
        for job in range(1, instance.num_jobs + 1)
        for seq in range(1, instance.job_operations[job] + 1)
    ]
    op_index_to_id = {index + 1: op_id for index, op_id in enumerate(operations)}
    positions = {op_index_to_id[int(op_index)]: pos for pos, op_index in enumerate(OS)}

    for index, op_id in enumerate(operations):
        if int(MS[index]) not in instance.processing_times[op_id]:
            raise ValueError(f"Invalid machine M{MS[index]} for {op_id}.")
        if not 1 <= int(TW[index]) <= instance.num_agvs_w:
            raise ValueError(f"Invalid AGV_W assignment for {op_id}.")
        if not 1 <= int(TF[index]) <= instance.num_agvs_f:
            raise ValueError(f"Invalid AGV_F assignment for {op_id}.")

    for op_id, predecessors in instance.priority_dict.items():
        for predecessor in predecessors:
            if positions[predecessor] >= positions[op_id]:
                raise ValueError(f"OS precedence violation: {predecessor} appears after {op_id}.")

    recomputed_makespan, schedule = Calculate(instance).simulate(
        MS, TW, TF, OS, return_schedule=True
    )
    if not np.isclose(float(reported_makespan), float(recomputed_makespan)):
        raise ValueError(
            f"Reported makespan {reported_makespan} differs from recomputed "
            f"makespan {recomputed_makespan}."
        )

    process_records = {
        record["operation"]: record
        for record in schedule
        if str(record.get("device", "")).startswith("M")
    }
    if set(process_records) != set(operations):
        missing = sorted(set(operations) - set(process_records))
        raise ValueError(f"Decoded schedule is missing processing records: {missing}")
    for op_id, predecessors in instance.priority_dict.items():
        op_start = float(process_records[op_id]["start_time"])
        for predecessor in predecessors:
            pred_end = float(process_records[predecessor]["end_time"])
            if op_start < pred_end - 1e-9:
                raise ValueError(
                    f"Schedule precedence violation: {op_id} starts at {op_start} "
                    f"before {predecessor} ends at {pred_end}."
                )


def run_algorithm(instance, algorithm_name: str, seed: int) -> tuple[float, int, float]:
    spec = ALGORITHMS[algorithm_name]
    params = dict(ALGORITHM_PARAMS.get(algorithm_name, {}))
    set_seed(seed)

    solver = spec.cls(instance=instance, **params)
    start = time.time()

    if spec.method == "optimize":
        best_solution, best_makespan, _history = solver.optimize()
        best_makespan = float(best_makespan)
    else:
        best_solution, _curve = solver.evolve()
        best_makespan = _extract_best_from_solution(best_solution)
        # _audit_evolutionary_solution(instance, best_solution, best_makespan)

    solve_time_s = time.time() - start
    best_generation = int(getattr(solver, "best_generation", -1))
    return best_makespan, best_generation, solve_time_s


def run_all_experiments(
    instance_names: Iterable[str] = INSTANCE_NAMES,
    algorithm_names: Iterable[str] = ALGORITHM_NAMES,
    n_runs: int = N_RUNS,
) -> None:
    for instance_name in instance_names:
        logger.info("=" * 72)
        logger.info(f"Instance: {instance_name}")
        logger.info("=" * 72)

        instance = load_instance(instance_name)
        source_df = read_source(instance_name)

        for algorithm_name in algorithm_names:
            if algorithm_name not in ALGORITHMS:
                raise KeyError(f"Unknown algorithm: {algorithm_name}")

            for run_id in range(1, n_runs + 1):
                if is_completed(source_df, algorithm_name, run_id):
                    logger.info(f"[skip] {instance_name} | {algorithm_name} | run {run_id}")
                    continue

                seed = seed_for_run(run_id)
                logger.info(f"[run] {instance_name} | {algorithm_name} | run {run_id} | seed {seed}")
                row = {
                    "instance": instance_name,
                    "algorithm": algorithm_name,
                    "run": run_id,
                    "seed": seed,
                    "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                }

                try:
                    best_makespan, best_generation, solve_time_s = run_algorithm(instance, algorithm_name, seed)
                    row.update(
                        best_makespan=best_makespan,
                        best_generation=best_generation,
                        solve_time_s=solve_time_s,
                        status="ok",
                        error="",
                    )
                    logger.info(
                        f"[ok] {instance_name} | {algorithm_name} | run {run_id} | "
                        f"best={best_makespan:.4f} | gen={best_generation} | time={solve_time_s:.2f}s"
                    )
                except Exception as exc:
                    row.update(
                        best_makespan=np.nan,
                        best_generation=np.nan,
                        solve_time_s=np.nan,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    logger.error(f"[failed] {instance_name} | {algorithm_name} | run {run_id}: {exc}")
                    logger.debug(traceback.format_exc())

                append_source_row(instance_name, row)
                new_row = pd.DataFrame([row])
                source_df = new_row if source_df.empty else pd.concat([source_df, new_row], ignore_index=True)

# =============================================================================
# Analysis
# =============================================================================

def load_all_source_files(source_dir: Path = SOURCE_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(source_dir.glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _try_import_scipy_stats():
    try:
        from scipy import stats  # type: ignore
        return stats
    except Exception:
        return None


def _nemenyi_cd(num_algorithms: int, num_blocks: int, alpha: float = 0.05) -> float:
    if num_algorithms < 2 or num_blocks < 2:
        return np.nan
    try:
        from scipy.stats import studentized_range  # type: ignore

        q_alpha = float(studentized_range.ppf(1.0 - alpha, num_algorithms, np.inf) / np.sqrt(2.0))
        return q_alpha * np.sqrt(num_algorithms * (num_algorithms + 1) / (6.0 * num_blocks))
    except Exception:
        return np.nan


def build_long_stats(ok: pd.DataFrame, algorithm_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    ordered_algorithms = list(algorithm_names)
    for (instance_name, algorithm_name), part in ok.groupby(["instance", "algorithm"], sort=True):
        if algorithm_name not in ordered_algorithms:
            continue
        vals = part["best_makespan"].astype(float)
        gens = part["best_generation"].astype(float)
        times = part["solve_time_s"].astype(float)
        rows.append(
            {
                "instance": instance_name,
                "algorithm": algorithm_name,
                # "n_runs": int(vals.count()),
                "mean": vals.mean(),
                "best": vals.min(),
                "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                # "median": vals.median(),
                # "avg_generation": gens.mean(),
                # "avg_time_s": times.mean(),
            }
        )
    return pd.DataFrame(rows)


def build_statistical_tests(long_stats: pd.DataFrame, algorithm_names: Iterable[str]) -> pd.DataFrame:
    ordered_algorithms = list(algorithm_names)
    pivot = long_stats.pivot(index="instance", columns="algorithm", values="mean")
    available = [name for name in ordered_algorithms if name in pivot.columns]
    complete = pivot[available].dropna()
    rows = []

    if complete.shape[0] < 2 or complete.shape[1] < 2:
        return pd.DataFrame(
            [{
                "test": "insufficient_data",
                "detail": "Need at least two complete instances and two algorithms for paired statistical tests.",
            }]
        )

    stats = _try_import_scipy_stats()
    ranks = complete.rank(axis=1, method="average", ascending=True)
    avg_ranks = ranks.mean(axis=0)

    if stats is not None and complete.shape[1] >= 3:
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(
                *[complete[name].to_numpy(dtype=float) for name in available]
            )
            cd = _nemenyi_cd(len(available), len(complete))
            for name in available:
                rows.append(
                    {
                        "test": "friedman_rank",
                        "algorithm_a": name,
                        "algorithm_b": "",
                        "n_blocks": int(len(complete)),
                        "statistic": float(friedman_stat),
                        "p_value": float(friedman_p),
                        "avg_rank_a": float(avg_ranks[name]),
                        "avg_rank_b": np.nan,
                        "rank_diff": np.nan,
                        "nemenyi_cd_0.05": cd,
                        "significant_0.05": bool(friedman_p < 0.05),
                    }
                )
        except Exception as exc:
            rows.append({"test": "friedman_failed", "detail": f"{type(exc).__name__}: {exc}"})
    else:
        rows.append(
            {
                "test": "friedman_unavailable",
                "detail": "scipy is not installed or fewer than three algorithms are available.",
            }
        )

    if stats is not None:
        for alg_a, alg_b in combinations(available, 2):
            paired = complete[[alg_a, alg_b]].dropna()
            diff = paired[alg_a] - paired[alg_b]
            try:
                if np.allclose(diff.to_numpy(dtype=float), 0.0):
                    stat, p_value = 0.0, 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        stat, p_value = stats.wilcoxon(paired[alg_a], paired[alg_b], zero_method="wilcox")
                rows.append(
                    {
                        "test": "wilcoxon_signed_rank",
                        "algorithm_a": alg_a,
                        "algorithm_b": alg_b,
                        "n_blocks": int(len(paired)),
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "avg_rank_a": float(avg_ranks[alg_a]),
                        "avg_rank_b": float(avg_ranks[alg_b]),
                        "rank_diff": abs(float(avg_ranks[alg_a] - avg_ranks[alg_b])),
                        "nemenyi_cd_0.05": _nemenyi_cd(len(available), len(complete)),
                        "significant_0.05": bool(p_value < 0.05),
                        "mean_a_minus_b": float(diff.mean()),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "test": "wilcoxon_failed",
                        "algorithm_a": alg_a,
                        "algorithm_b": alg_b,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        rows.append({"test": "wilcoxon_unavailable", "detail": "scipy is not installed."})

    if np.isfinite(_nemenyi_cd(len(available), len(complete))):
        cd = _nemenyi_cd(len(available), len(complete))
        for alg_a, alg_b in combinations(available, 2):
            rank_diff = abs(float(avg_ranks[alg_a] - avg_ranks[alg_b]))
            rows.append(
                {
                    "test": "nemenyi_posthoc",
                    "algorithm_a": alg_a,
                    "algorithm_b": alg_b,
                    "n_blocks": int(len(complete)),
                    "avg_rank_a": float(avg_ranks[alg_a]),
                    "avg_rank_b": float(avg_ranks[alg_b]),
                    "rank_diff": rank_diff,
                    "nemenyi_cd_0.05": cd,
                    "significant_0.05": bool(rank_diff > cd),
                }
            )

    return pd.DataFrame(rows)


def summarize_results(
    source_dir: Path = SOURCE_DIR,
    algorithm_names: Iterable[str] = ALGORITHM_NAMES,
    summary_csv: Path = SUMMARY_CSV,
    summary_xlsx: Path = SUMMARY_XLSX,
    long_stats_csv: Path = LONG_STATS_CSV,
    stat_tests_csv: Path = STAT_TESTS_CSV,
) -> pd.DataFrame:
    raw = load_all_source_files(source_dir)
    if raw.empty:
        logger.warning("No source result files found for analysis.")
        return pd.DataFrame()

    ok = raw[raw["status"] == "ok"].copy()
    ok["best_makespan"] = pd.to_numeric(ok["best_makespan"], errors="coerce")
    ok["best_generation"] = pd.to_numeric(ok["best_generation"], errors="coerce")
    ok["solve_time_s"] = pd.to_numeric(ok["solve_time_s"], errors="coerce")
    ok = ok[np.isfinite(ok["best_makespan"])]

    rows = []
    for instance_name, part in ok.groupby("instance", sort=True):
        row = {"Instance": instance_name}
        aver_values = []
        best_values = []
        stdev_values = []

        for algorithm_name in algorithm_names:
            vals = part.loc[part["algorithm"] == algorithm_name, "best_makespan"].astype(float)
            gens = part.loc[part["algorithm"] == algorithm_name, "best_generation"].astype(float)
            times = part.loc[part["algorithm"] == algorithm_name, "solve_time_s"].astype(float)

            row[f"{algorithm_name}_AVER"] = vals.mean() if not vals.empty else np.nan
            row[f"{algorithm_name}_BEST"] = vals.min() if not vals.empty else np.nan
            row[f"{algorithm_name}_STDEV"] = vals.std(ddof=1) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            # row[f"{algorithm_name}_AVG_GEN"] = gens.mean() if not gens.empty else np.nan
            # row[f"{algorithm_name}_AVG_TIME"] = times.mean() if not times.empty else np.nan

            if not vals.empty:
                aver_values.append(row[f"{algorithm_name}_AVER"])
                best_values.append(row[f"{algorithm_name}_BEST"])
                stdev_values.append(row[f"{algorithm_name}_STDEV"])

        row["MIN_AVER"] = np.nanmin(aver_values) if aver_values else np.nan
        row["MIN_BEST"] = np.nanmin(best_values) if best_values else np.nan
        row["MIN_STDEV"] = np.nanmin(stdev_values) if stdev_values else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    long_stats = build_long_stats(ok, algorithm_names)
    stat_tests = build_statistical_tests(long_stats, algorithm_names)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    long_stats.to_csv(long_stats_csv, index=False, encoding="utf-8-sig")
    stat_tests.to_csv(stat_tests_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        long_stats.to_excel(writer, sheet_name="per_instance_stats", index=False)
        stat_tests.to_excel(writer, sheet_name="statistical_tests", index=False)
        ok.to_excel(writer, sheet_name="source_ok", index=False)

    logger.info(f"Summary CSV saved to: {summary_csv}")
    logger.info(f"Per-instance stats CSV saved to: {long_stats_csv}")
    logger.info(f"Statistical tests CSV saved to: {stat_tests_csv}")
    logger.info(f"Summary XLSX saved to: {summary_xlsx}")
    return summary


def main() -> None:
    if RUN_EXPERIMENTS:
        run_all_experiments()
    if RUN_ANALYSIS:
        summarize_results()


if __name__ == "__main__":
    main()
