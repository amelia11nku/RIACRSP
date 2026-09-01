#!/usr/bin/env python3
"""Resumable Phase 6H tiny and license-conditional small exact validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import Phase6HLiveObserver  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.exact.general_gurobi import solve_general_gurobi  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.dcga import DCGAConfig, solve_dcga  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/phase6h_exact_validation.json"
PHASE_CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
FROZEN_POLICY = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
FREEZE_RECORD = ROOT / "outputs/phase6h_calibration/frozen/freeze_record.json"
OUT = ROOT / "outputs/phase6h_exact_validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def frozen_config(name: str, cls):
    raw = load_json(ROOT / "configs" / name)
    return cls(**{
        key: value for key, value in raw.items() if key in cls.__dataclass_fields__
    })


def verify_freeze(protocol: dict, phase_config: dict) -> None:
    freeze = load_json(FREEZE_RECORD)
    if (
        freeze.get("status") != "FROZEN_BEFORE_CAL_HOLDOUT"
        or digest(FROZEN_POLICY) != protocol["selected_policy_sha256"]
        or freeze.get("policy_sha256") != protocol["selected_policy_sha256"]
        or phase_config["frozen_phase6f"]["checkpoint_sha256"]
        != load_json(FROZEN_POLICY)["checkpoint_sha256"]
    ):
        raise RuntimeError("Phase 6H exact validation policy freeze is invalid")


def save_gurobi(instance, instance_path: Path, suite: str, protocol: dict) -> dict:
    result_path = OUT / "gurobi/runs" / suite / f"{instance.instance_id}.json"
    if result_path.exists():
        return load_json(result_path)
    log_path = OUT / "gurobi/logs" / suite / f"{instance.instance_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = protocol["gurobi"]
    record = {
        "schema": "phase6h-gurobi-run-v1",
        "suite": suite,
        "instance_id": instance.instance_id,
        "instance_sha256": digest(instance_path),
        "solver": "gurobi-general-rcias-milp",
        "exact_model_path": parameters["model"],
        "exact_model_sha256": digest(ROOT / parameters["model"]),
        "protocol_sha256": digest(PROTOCOL_PATH),
        "solver_log": str(log_path.relative_to(ROOT)),
        "time_limit_seconds": parameters["time_limit_seconds"],
        "seed": parameters["seed"],
        "threads": parameters["threads"],
        "mip_gap_target": parameters["mip_gap"],
        "status": "ENVIRONMENT_ERROR",
        "runtime_seconds": None,
        "total_runtime_seconds": None,
        "incumbent": None,
        "lower_bound": None,
        "mip_gap": None,
        "optimality_proven": False,
        "replay_feasible": False,
    }
    started = time.perf_counter()
    try:
        result = solve_general_gurobi(
            instance,
            time_limit_seconds=float(parameters["time_limit_seconds"]),
            seed=int(parameters["seed"]),
            threads=int(parameters["threads"]),
            mip_gap=float(parameters["mip_gap"]),
            log_file=str(log_path),
        )
        record.update({
            "status": result.status,
            "solver_version": result.solver_version,
            "runtime_seconds": result.runtime_seconds,
            "total_runtime_seconds": result.total_runtime_seconds,
            "incumbent": result.replay_makespan,
            "native_incumbent": result.solver_makespan,
            "lower_bound": result.best_bound,
            "mip_gap": result.gap,
            "optimality_proven": result.optimality_proven,
            "replay_feasible": result.replay_feasible,
            "variable_count": result.variable_count,
            "constraint_count": result.constraint_count,
            "h1_upper_bound": result.h1_upper_bound,
        })
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        record.update({
            "status": "LICENSE_LIMIT" if "size-limited license" in message else "ENVIRONMENT_ERROR",
            "total_runtime_seconds": time.perf_counter() - started,
            "error": message,
        })
        if not log_path.exists():
            log_path.write_text(message + "\n", encoding="utf-8")
    record["solver_log_sha256"] = digest(log_path)
    atomic_json(record, result_path)
    print(
        f"GUROBI suite={suite} instance={instance.instance_id} "
        f"status={record['status']} incumbent={record['incumbent']} ",
        flush=True,
    )
    return record


def save_h1(instance, suite: str) -> None:
    path = OUT / "heuristic/runs" / suite / instance.instance_id / "H1/result.json"
    if path.exists():
        return
    result = solve_dispatching(instance, "H1")
    if not check_schedule(instance, result.schedule)["feasible"]:
        raise RuntimeError(f"H1 infeasible on {instance.instance_id}")
    atomic_json({
        "schema": "phase6h-exact-heuristic-run-v1",
        "suite": suite,
        "instance_id": instance.instance_id,
        "method": "H1",
        "seed": None,
        "runtime_seconds": result.runtime_seconds,
        "makespan": result.objective.makespan,
        "feasible": True,
    }, path)


def save_stochastic(
    instance,
    suite: str,
    method: str,
    seed: int,
    budget: float,
    phase_config: dict,
    policy: FrozenLiveInference | None,
) -> FrozenLiveInference | None:
    path = OUT / "heuristic/runs" / suite / instance.instance_id / method / f"seed_{seed}.json"
    if path.exists():
        return policy
    alns_config = frozen_config("phase5c_alns.json", ALNSConfig)
    observer = None
    if method == "ALNS":
        result = solve_alns(instance, budget, seed, alns_config)
    elif method == "GA":
        result = solve_ga(instance, budget, seed, frozen_config("phase5c_ga.json", GAConfig))
    elif method == "DCGA":
        result = solve_dcga(instance, budget, seed, frozen_config("phase5c_dcga.json", DCGAConfig))
    elif method == "PHASE6H_CSGNI":
        if policy is None:
            policy = FrozenLiveInference(
                ROOT / phase_config["frozen_phase6f"]["experiment_freeze"],
                device="cuda",
                proposal_seed_namespace=phase_config["rng_namespaces"]["proposal"],
                deployment_artifact=FROZEN_POLICY,
            )
        observer = Phase6HLiveObserver({
            "instance_id": instance.instance_id,
            "seed": seed,
            "scale": "TINY" if suite == "tiny" else "S",
            "CF_level": "NA",
            "calibration_split": "EXACT_VALIDATION",
            "method": method,
            "policy_name": policy.policy_name,
        })
        result = solve_csgni(
            instance,
            budget,
            seed,
            policy,
            alns_config=alns_config,
            csgni_config=CSGNIConfig(
                intervention_rate=int(phase_config["search"]["intervention_rate"]),
                proposal_seed_namespace=phase_config["rng_namespaces"]["proposal"],
                ni_repair_seed_namespace=phase_config["rng_namespaces"]["ni_repair"],
                acceptance_seed_namespace=phase_config["rng_namespaces"]["acceptance"],
                diagnostics_seed_namespace=phase_config["rng_namespaces"]["diagnostics"],
            ),
            observer=observer,
        )
    else:
        raise ValueError(f"unsupported exact-validation heuristic: {method}")
    audit = check_schedule(instance, result.best.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"{method} infeasible on {instance.instance_id}, seed {seed}")
    atomic_json({
        "schema": "phase6h-exact-heuristic-run-v1",
        "suite": suite,
        "instance_id": instance.instance_id,
        "method": method,
        "seed": seed,
        "time_limit_seconds": budget,
        "runtime_seconds": result.runtime,
        "makespan": result.best.makespan,
        "feasible": True,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "diagnostics": result.diagnostics,
        "policy_sha256": (
            digest(FROZEN_POLICY) if method == "PHASE6H_CSGNI" else None
        ),
    }, path)
    if observer is not None:
        log_path = OUT / "live_logs" / suite / instance.instance_id / f"seed_{seed}.parquet"
        atomic_parquet(pd.DataFrame(observer.rows), log_path)
    print(
        f"HEURISTIC suite={suite} instance={instance.instance_id} method={method} "
        f"seed={seed} makespan={result.best.makespan:g}",
        flush=True,
    )
    return policy


def summarize(protocol: dict) -> None:
    exact_rows = []
    for path in sorted((OUT / "gurobi/runs").rglob("*.json")):
        record = load_json(path)
        exact_rows.append({key: record.get(key) for key in (
            "suite", "instance_id", "solver", "status", "time_limit_seconds",
            "runtime_seconds", "total_runtime_seconds", "incumbent", "lower_bound",
            "mip_gap", "optimality_proven", "replay_feasible", "variable_count",
            "constraint_count", "error", "solver_log", "solver_log_sha256",
        )})
    atomic_csv(pd.DataFrame(exact_rows), OUT / "gurobi/gurobi_summary.csv")
    references = {
        row["instance_id"]: float(row["incumbent"])
        for row in exact_rows if row["optimality_proven"]
    }
    heuristic_rows = []
    for path in sorted((OUT / "heuristic/runs").rglob("*.json")):
        record = load_json(path)
        reference = references.get(record["instance_id"])
        heuristic_rows.append({
            "suite": record["suite"],
            "instance_id": record["instance_id"],
            "method": record["method"],
            "seed": record.get("seed"),
            "runtime_seconds": record["runtime_seconds"],
            "makespan": record["makespan"],
            "feasible": record["feasible"],
            "proven_optimum": reference,
            "gap_to_proven_optimum": (
                None if reference is None else (record["makespan"] - reference) / reference
            ),
        })
    frame = pd.DataFrame(heuristic_rows)
    atomic_csv(frame, OUT / "exact_heuristic_comparison.csv")
    tiny = frame[frame.suite == "tiny"]
    expected = len(protocol["tiny_instances"]) * len(protocol["tiny_stochastic_seeds"])
    csgni = tiny[tiny.method == "PHASE6H_CSGNI"]
    atomic_json({
        "schema": "phase6h-exact-validation-integrity-v1",
        "status": "PASS" if (
            len(csgni) == expected
            and csgni.feasible.all()
            and (csgni.gap_to_proven_optimum == 0.0).all()
        ) else "FAIL",
        "tiny_phase6h_run_count": len(csgni),
        "tiny_phase6h_expected_run_count": expected,
        "tiny_all_feasible": bool(csgni.feasible.all()),
        "tiny_all_recover_proven_optimum": bool(
            len(csgni) == expected and (csgni.gap_to_proven_optimum == 0.0).all()
        ),
        "gurobi_proven_optima_count": sum(
            bool(row["optimality_proven"]) for row in exact_rows
        ),
        "additional_small_license_limited_count": sum(
            row["suite"] == "cal_fit_small" and row["status"] == "LICENSE_LIMIT"
            for row in exact_rows
        ),
        "core_accessed": False,
    }, OUT / "audit/exact_validation_integrity.json")


def main() -> None:
    protocol = load_json(PROTOCOL_PATH)
    phase_config = load_json(PHASE_CONFIG_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_PHASE6H_EXACT_VALIDATION":
        raise RuntimeError("Phase 6H exact protocol is not frozen")
    verify_freeze(protocol, phase_config)
    print("PHASE6H_EXACT_VALIDATION_START", flush=True)
    policy = None
    for instance_id in protocol["tiny_instances"]:
        instance_path = ROOT / "instances/tiny" / f"{instance_id}.json"
        instance = load_instance(instance_path)
        save_h1(instance, "tiny")
        save_gurobi(instance, instance_path, "tiny", protocol)
        for seed in protocol["tiny_stochastic_seeds"]:
            policy = save_stochastic(
                instance,
                "tiny",
                "PHASE6H_CSGNI",
                int(seed),
                float(protocol["tiny_stochastic_budget_seconds"]),
                phase_config,
                policy,
            )
    cal_root = ROOT / phase_config["calibration_instances"]["root"]
    for case in protocol["additional_cal_fit_small_cases"]:
        instance_path = cal_root / case["relative_path"]
        instance = load_instance(instance_path)
        if instance.num_operations != int(case["number_of_operations"]):
            raise RuntimeError(f"operation count mismatch for {instance.instance_id}")
        exact = save_gurobi(instance, instance_path, "cal_fit_small", protocol)
        if exact["status"] in {"LICENSE_LIMIT", "ENVIRONMENT_ERROR"}:
            continue
        save_h1(instance, "cal_fit_small")
        budget = 2.0 * instance.num_operations
        for method in ("ALNS", "GA", "DCGA", "PHASE6H_CSGNI"):
            for seed in protocol["adequate_license_stochastic_seeds"]:
                policy = save_stochastic(
                    instance,
                    "cal_fit_small",
                    method,
                    int(seed),
                    budget,
                    phase_config,
                    policy,
                )
    summarize(protocol)
    integrity = load_json(OUT / "audit/exact_validation_integrity.json")
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Phase 6H exact validation failed: {integrity}")
    print("PHASE6H_EXACT_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
