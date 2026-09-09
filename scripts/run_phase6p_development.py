#!/usr/bin/env python3
"""Resumable matched-budget Phase 6P development comparison."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.ni.phase6p_live_inference import (  # noqa: E402
    FrozenPhase6NCriticEnsemble,
)
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.lghga import LGHGAConfig  # noqa: E402
from rcias_clgri.search.lghga_2o import solve_lghga_2o  # noqa: E402
from rcias_clgri.search.lghga_learning import load_dtr_bundle  # noqa: E402
from rcias_clgri.search.phase6p_adaptive import solve_phase6p  # noqa: E402


METHODS = (
    "P1_CSG_ADAPTIVE_PORTFOLIO",
    "PHASE6N_DETERMINISTIC_TOP1",
    "ALNS",
    "PHASE6H",
    "LG_HGA_2O",
)
CONFIG = ROOT / "configs/phase6p_adaptive_portfolio_v1.json"
PROTOCOL = ROOT / "outputs/phase6p_adaptive_portfolio_v1/development/protocol.json"
INSTANCE_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14"
PHASE6P_OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/development"
REGISTRY_ROOT = ROOT / "outputs/frozen_2o_baselines"
REGISTRY = REGISTRY_ROOT / "registry.json"
PHASE6H_CONFIG = ROOT / "configs/phase6h_live_calibration.json"
PHASE6H_POLICY = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
LGHGA_CONFIG = ROOT / "configs/lghga_2o_baseline.json"
LGHGA_MODELS = ROOT / "outputs/baselines/lghga_kb_v2/models"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def task_key(task: dict) -> str:
    return f"{task['method']}::{task['instance_id']}::{task['seed']}"


def build_tasks(protocol: dict) -> list[dict]:
    tasks = []
    for method in METHODS:
        for instance in protocol["instances"]:
            for seed in protocol["development_seeds"]:
                tasks.append({
                    "method": method,
                    "instance_id": instance["instance_id"],
                    "instance_relative_path": instance["relative_path"],
                    "instance_sha256": instance["sha256"],
                    "num_operations": instance["num_operations"],
                    "scale": instance["scale"],
                    "CF_level": instance["CF_level"],
                    "cell_replicate": instance["cell_replicate"],
                    "seed": int(seed),
                    "budget_seconds": 2.0 * int(instance["num_operations"]),
                    "result_source": "NEW_RUN",
                })
    if len(tasks) != 270 or len({task_key(task) for task in tasks}) != 270:
        raise RuntimeError("Phase 6P development requires exactly 270 unique runs")
    return tasks


def result_path(task: dict) -> Path:
    roots = {
        "P1_CSG_ADAPTIVE_PORTFOLIO": PHASE6P_OUT / "runs/P1_CSG_ADAPTIVE_PORTFOLIO",
        "PHASE6N_DETERMINISTIC_TOP1": REGISTRY_ROOT / "phase6n/runs",
        "ALNS": REGISTRY_ROOT / "alns/runs",
        "PHASE6H": REGISTRY_ROOT / "phase6h/runs",
        "LG_HGA_2O": REGISTRY_ROOT / "lg_hga_2o/runs",
    }
    return roots[task["method"]] / task["instance_id"] / f"seed_{task['seed']}.json"


def validate_protocol() -> dict:
    protocol = load_json(PROTOCOL)
    if (
        protocol.get("schema") != "phase6p-development-protocol-v1"
        or protocol.get("status") != "FROZEN_BEFORE_P3_SOLVER_OUTCOMES"
        or protocol.get("methods") != list(METHODS)
        or protocol.get("development_seeds") != [746101, 746102, 746103]
        or protocol.get("task_count") != 270
        or protocol.get("r13_accessed") is not False
        or protocol.get("r14_accessed") is not False
    ):
        raise RuntimeError("invalid Phase 6P development protocol")
    for relative, expected in protocol["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6P development source changed: {relative}")
    for relative, expected in protocol["artifact_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6P development artifact changed: {relative}")
    if digest(REGISTRY_ROOT / "instance_manifest.json") != protocol["instance_manifest_sha256"]:
        raise RuntimeError("canonical 2|O| instance manifest changed")
    if (
        (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists()
        or (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists()
    ):
        raise RuntimeError("R13/R14 must remain locked during Phase 6P development")
    audit = load_json(
        ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit/comparator_reuse_audit.json"
    )
    if (
        audit.get("status") != "COMPLETE_BEFORE_COMPARATOR_EXECUTION"
        or not audit.get("comparators_may_start")
        or {entry["action"] for entry in audit["entries"]} != {"RERUN_REQUIRED"}
    ):
        raise RuntimeError("Phase 6P comparator audit does not authorize new runs")
    return protocol


def valid_result(task: dict, protocol_hash: str) -> dict | None:
    path = result_path(task)
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
        if not all((
            payload.get("schema") == "phase6p-development-run-v1",
            payload.get("status") == "COMPLETE",
            payload.get("development_protocol_sha256") == protocol_hash,
            payload.get("method") == task["method"],
            payload.get("instance_id") == task["instance_id"],
            payload.get("instance_sha256") == task["instance_sha256"],
            payload.get("seed") == task["seed"],
            payload.get("time_limit_seconds") == task["budget_seconds"],
            payload.get("result_source") == "NEW_RUN",
            payload.get("feasible") is True,
            payload.get("r13_accessed") is False,
            payload.get("r14_accessed") is False,
        )):
            return None
        return payload
    except (OSError, json.JSONDecodeError, TypeError):
        return None


class PortfolioObserver:
    def __init__(self, budget: float) -> None:
        self.budget = budget
        self.iterations = 0
        self.accepted = 0
        self.new_best = 0
        self.current_improvement = 0
        self.neural_decisions = 0
        self.safe_fallbacks = 0
        self.canonical_fallback_selections = 0
        self.rank = Counter()
        self.rules = Counter()
        self.families = Counter()
        self.destroy = Counter()
        self.repair = Counter()
        self.weight_checkpoints: list[dict] = []
        self.next_fraction_index = 0
        self.fractions = tuple(index / 10 for index in range(11))
        self.last_weights = None
        self.runtime_sums = Counter()
        self.portfolio_l1: list[float] = []

    def __call__(self, event: Mapping[str, object]) -> None:
        self.iterations += 1
        self.accepted += int(bool(event["accepted"]))
        self.new_best += int(bool(event["new_global_best"]))
        self.current_improvement += int(
            event["candidate"].makespan < event["current_before"].makespan
        )
        self.repair.update((str(event["repair_operator"]),))
        self.destroy.update(str(value) for value in event["credited_destroy_operators"])
        self.runtime_sums.update({
            "iteration": float(event["iteration_runtime"]),
            "decoder": float(event["decoder_runtime"]),
            "repair_excluding_decoder": float(event["repair_excluding_decoder_runtime"]),
        })
        if event["neural_eligible"]:
            self.neural_decisions += 1
            self.safe_fallbacks += int(bool(event["safe_fallback"]))
            self.canonical_fallback_selections += int(
                event["selected_target_set_id"]
                == event["canonical_fallback_target_set_id"]
            )
            ranked = tuple(event["ranked_target_ids"])
            selected = str(event["selected_target_set_id"])
            if selected in ranked:
                self.rank.update((ranked.index(selected) + 1,))
            else:
                self.rank.update(("SAFE_FALLBACK_UNRANKED",))
            self.rules.update(str(value) for value in event["selected_origin_rules"])
            self.families.update(str(value) for value in event["selected_origin_families"])
            distribution = tuple(event["portfolio_distribution"])
            if distribution:
                pure_raw = [1.0 / rank for rank in range(1, len(distribution) + 1)]
                denominator = math.fsum(pure_raw)
                pure = [value / denominator for value in pure_raw]
                self.portfolio_l1.append(math.fsum(
                    abs(float(row[1]) - baseline)
                    for row, baseline in zip(distribution, pure)
                ))
            for name, value in dict(event["critic_timing_ms"]).items():
                self.runtime_sums[f"critic_ms_{name}"] += float(value)
        weights = dict(event["operator_weights_after"])
        self.last_weights = weights
        elapsed_fraction = min(1.0, float(event["elapsed_time"]) / self.budget)
        while (
            self.next_fraction_index < len(self.fractions)
            and elapsed_fraction >= self.fractions[self.next_fraction_index]
        ):
            self.weight_checkpoints.append({
                "budget_fraction": self.fractions[self.next_fraction_index],
                "elapsed_seconds": float(event["elapsed_time"]),
                "weights": weights,
            })
            self.next_fraction_index += 1

    def summary(self) -> dict:
        while self.last_weights is not None and self.next_fraction_index < len(self.fractions):
            self.weight_checkpoints.append({
                "budget_fraction": self.fractions[self.next_fraction_index],
                "elapsed_seconds": self.budget,
                "weights": self.last_weights,
            })
            self.next_fraction_index += 1
        return {
            "iterations": self.iterations,
            "neural_decisions": self.neural_decisions,
            "safe_fallbacks": self.safe_fallbacks,
            "canonical_fallback_selections": self.canonical_fallback_selections,
            "sampled_neural_rank_distribution": dict(self.rank),
            "selected_24_rule_origin_distribution": dict(self.rules),
            "selected_origin_family_distribution": dict(self.families),
            "selected_destroy_operator_distribution": dict(self.destroy),
            "selected_repair_distribution": dict(self.repair),
            "acceptance_rate": self.accepted / self.iterations if self.iterations else 0.0,
            "new_global_best_rate": self.new_best / self.iterations if self.iterations else 0.0,
            "current_improvement_rate": (
                self.current_improvement / self.iterations if self.iterations else 0.0
            ),
            "adaptive_weight_checkpoints": self.weight_checkpoints,
            "runtime_component_sums": dict(self.runtime_sums),
            "mean_portfolio_probability_l1_vs_rank_prior": (
                math.fsum(self.portfolio_l1) / len(self.portfolio_l1)
                if self.portfolio_l1 else 0.0
            ),
            "portfolio_probability_changed_by_destroy_weights_fraction": (
                sum(value > 1e-12 for value in self.portfolio_l1) / len(self.portfolio_l1)
                if self.portfolio_l1 else 0.0
            ),
        }


def compact_diagnostics(method: str, diagnostics: Mapping[str, object]) -> dict:
    result = dict(diagnostics)
    if method == "LG_HGA_2O":
        generations = result.pop("generation_records", [])
        result["generation_records_count"] = len(generations)
    return result


class RuntimeObjects:
    def __init__(self, device: str) -> None:
        self.device = device
        self.phase6n = None
        self.phase6n_load_seconds = None
        self.phase6h = None
        self.phase6h_load_seconds = None
        self.lghga = {}
        self.lghga_load_seconds = {}
        raw = load_json(ROOT / "configs/phase5c_alns.json")
        self.alns = ALNSConfig(**{
            key: value for key, value in raw.items()
            if key in ALNSConfig.__dataclass_fields__
        })
        phase6h = load_json(PHASE6H_CONFIG)
        self.phase6h_settings = phase6h
        raw_lg = load_json(LGHGA_CONFIG)
        self.lghga_config = LGHGAConfig(**{
            key: value for key, value in raw_lg.items()
            if key in LGHGAConfig.__dataclass_fields__
        })

    def phase6n_critic(self):
        if self.phase6n is None:
            started = time.perf_counter()
            self.phase6n = FrozenPhase6NCriticEnsemble(device=self.device)
            self.phase6n_load_seconds = time.perf_counter() - started
        return self.phase6n

    def phase6h_policy(self):
        if self.phase6h is None:
            started = time.perf_counter()
            config = self.phase6h_settings
            self.phase6h = FrozenLiveInference(
                ROOT / config["frozen_phase6f"]["experiment_freeze"],
                device=self.device,
                proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                deployment_artifact=PHASE6H_POLICY,
            )
            self.phase6h_load_seconds = time.perf_counter() - started
        return self.phase6h

    def lghga_bundle(self, scale: str, cf_level: str):
        cell = f"{scale}_{cf_level}"
        if cell not in self.lghga:
            started = time.perf_counter()
            self.lghga[cell] = load_dtr_bundle(LGHGA_MODELS / cell)
            self.lghga_load_seconds[cell] = time.perf_counter() - started
        return self.lghga[cell]


def execute(task: dict, runtime: RuntimeObjects):
    instance_path = INSTANCE_ROOT / task["instance_relative_path"]
    if digest(instance_path) != task["instance_sha256"]:
        raise RuntimeError(f"instance changed: {instance_path}")
    instance = load_instance(instance_path)
    if instance.num_operations != task["num_operations"]:
        raise RuntimeError("operation count changed")
    method = task["method"]
    budget = task["budget_seconds"]
    observer = None
    model_load_seconds = None
    outer_started = time.perf_counter()
    if method == "P1_CSG_ADAPTIVE_PORTFOLIO":
        torch.use_deterministic_algorithms(True)
        critic = runtime.phase6n_critic()
        model_load_seconds = runtime.phase6n_load_seconds
        observer = PortfolioObserver(budget)
        result = solve_phase6p(
            instance, budget, task["seed"], critic,
            target_mode="portfolio", alns_config=runtime.alns, observer=observer,
        )
    elif method == "PHASE6N_DETERMINISTIC_TOP1":
        torch.use_deterministic_algorithms(True)
        critic = runtime.phase6n_critic()
        model_load_seconds = runtime.phase6n_load_seconds
        observer = PortfolioObserver(budget)
        result = solve_phase6p(
            instance, budget, task["seed"], critic,
            target_mode="top1", alns_config=runtime.alns, observer=observer,
        )
    elif method == "ALNS":
        torch.use_deterministic_algorithms(False)
        result = solve_alns(instance, budget, task["seed"], runtime.alns)
    elif method == "PHASE6H":
        torch.use_deterministic_algorithms(False)
        policy = runtime.phase6h_policy()
        model_load_seconds = runtime.phase6h_load_seconds
        config = runtime.phase6h_settings
        result = solve_csgni(
            instance,
            budget,
            task["seed"],
            policy,
            alns_config=runtime.alns,
            csgni_config=CSGNIConfig(
                intervention_rate=int(config["search"]["intervention_rate"]),
                proposal_seed_namespace=int(config["rng_namespaces"]["proposal"]),
                ni_repair_seed_namespace=int(config["rng_namespaces"]["ni_repair"]),
                acceptance_seed_namespace=int(config["rng_namespaces"]["acceptance"]),
                diagnostics_seed_namespace=int(config["rng_namespaces"]["diagnostics"]),
            ),
        )
    else:
        torch.use_deterministic_algorithms(False)
        bundle = runtime.lghga_bundle(task["scale"], task["CF_level"])
        model_load_seconds = runtime.lghga_load_seconds[
            f"{task['scale']}_{task['CF_level']}"
        ]
        result = solve_lghga_2o(
            instance, budget, task["seed"], bundle, runtime.lghga_config
        )
    outer_runtime = time.perf_counter() - outer_started
    replay = check_schedule(instance, result.best.schedule)
    if not result.best.feasible or not replay["feasible"]:
        raise RuntimeError({"task": task, "violations": replay["violations"]})
    trace = [asdict(point) for point in result.convergence_trace]
    if (
        not trace
        or trace[-1]["current_best_makespan"] != result.best.makespan
        or any(
            earlier["elapsed_time"] > later["elapsed_time"]
            or earlier["decoder_evaluations"] > later["decoder_evaluations"]
            or earlier["current_best_makespan"] < later["current_best_makespan"]
            for earlier, later in zip(trace, trace[1:])
        )
    ):
        raise RuntimeError("invalid best-so-far trace")
    return result, trace, replay, observer, model_load_seconds, outer_runtime


def write_progress(
    tasks: list[dict], completed: set[str], current: dict | None, started: float
) -> None:
    remaining_budget = math.fsum(
        task["budget_seconds"] for task in tasks if task_key(task) not in completed
    )
    payload = {
        "schema": "phase6p-development-progress-v1",
        "status": "COMPLETE" if len(completed) == len(tasks) else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": len(completed),
        "total_runs": len(tasks),
        "completed_by_method": {
            method: sum(key.startswith(f"{method}::") for key in completed)
            for method in METHODS
        },
        "current_task": current,
        "process_elapsed_seconds": time.perf_counter() - started,
        "nominal_remaining_budget_seconds": remaining_budget,
        "nominal_remaining_hours": remaining_budget / 3600.0,
        "progress_path": str((PHASE6P_OUT / "progress.json").relative_to(ROOT)),
        "resume_command": [sys.executable, "-u", "scripts/run_phase6p_development.py"],
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(PHASE6P_OUT / "progress.json", payload)


def completed_rows(tasks: list[dict], protocol_hash: str) -> list[dict]:
    rows = []
    for task in tasks:
        payload = valid_result(task, protocol_hash)
        if payload is None:
            continue
        rows.append({
            **{key: task[key] for key in (
                "method", "instance_id", "scale", "CF_level", "cell_replicate",
                "seed", "budget_seconds", "result_source",
            )},
            "final_makespan": payload["final_makespan"],
            "runtime_seconds": payload["runtime_seconds"],
            "outer_runtime_seconds": payload["outer_runtime_seconds"],
            "best_found_seconds": payload["best_found_seconds"],
            "decoder_evaluations": payload["decoder_evaluations"],
            "iterations": payload["iterations"],
            "neural_decisions": payload["portfolio_diagnostics"].get(
                "neural_decisions", payload["search_diagnostics"].get(
                    "ni_eligible_iterations", 0
                )
            ),
            "fallback_count": payload["portfolio_diagnostics"].get(
                "safe_fallbacks", payload["search_diagnostics"].get("ni_fallbacks", 0)
            ),
            "feasible": payload["feasible"],
            "result_path": str(result_path(task).relative_to(ROOT)),
        })
    return rows


def freeze_completed_comparators(
    tasks: list[dict], protocol_hash: str, completed_at: str
) -> None:
    mapping = {
        "ALNS": ("ALNS", "alns"),
        "PHASE6H": ("PHASE6H", "phase6h"),
        "PHASE6N_DETERMINISTIC_TOP1": ("PHASE6N_TOP1", "phase6n"),
        "LG_HGA_2O": ("LG_HGA_2O", "lg_hga_2o"),
    }
    registry = load_json(REGISTRY)
    for method, (algorithm_id, directory) in mapping.items():
        entry = next(
            item for item in registry["entries"] if item["algorithm_id"] == algorithm_id
        )
        if entry["status"] == "FROZEN_CANONICAL":
            continue
        method_tasks = [task for task in tasks if task["method"] == method]
        payloads = [valid_result(task, protocol_hash) for task in method_tasks]
        if any(payload is None for payload in payloads):
            continue
        rows = [
            row for row in completed_rows(method_tasks, protocol_hash)
            if row["method"] == method
        ]
        summary = {
            "schema": "frozen-2o-method-summary-v1",
            "status": "FROZEN_CANONICAL",
            "algorithm_id": algorithm_id,
            "method": method,
            "run_count": len(rows),
            "instance_count": len({row["instance_id"] for row in rows}),
            "seeds": sorted({row["seed"] for row in rows}),
            "feasibility_rate": sum(row["feasible"] for row in rows) / len(rows),
            "mean_final_makespan": float(pd.DataFrame(rows).final_makespan.mean()),
            "median_final_makespan": float(pd.DataFrame(rows).final_makespan.median()),
            "development_protocol_sha256": protocol_hash,
            "completed_at_utc": completed_at,
        }
        directory_path = REGISTRY_ROOT / directory
        atomic_json(directory_path / "summary.json", summary)
        manifest = {
            "schema": "frozen-2o-run-manifest-v1",
            "status": "FROZEN_CANONICAL",
            "algorithm_id": algorithm_id,
            "runs": [
                {
                    "instance_id": task["instance_id"],
                    "seed": task["seed"],
                    "path": str(result_path(task).relative_to(ROOT)),
                    "sha256": digest(result_path(task)),
                }
                for task in method_tasks
            ],
        }
        atomic_json(directory_path / "manifest.json", manifest)
        entry.update({
            "status": "FROZEN_CANONICAL",
            "started_at_utc": min(payload["started_at_utc"] for payload in payloads),
            "completed_at_utc": completed_at,
            "feasibility_evidence": {
                "rate": summary["feasibility_rate"],
                "run_count": len(rows),
            },
            "aggregate_summary_path": str((directory_path / "summary.json").relative_to(ROOT)),
            "aggregate_summary_sha256": digest(directory_path / "summary.json"),
            "run_manifest_path": str((directory_path / "manifest.json").relative_to(ROOT)),
            "run_manifest_sha256": digest(directory_path / "manifest.json"),
            "regression_integrity_result": "PENDING_FINAL_DEVELOPMENT_AUDIT",
        })
    if all(entry["status"] == "FROZEN_CANONICAL" for entry in registry["entries"]):
        registry["status"] = "FROZEN_CANONICAL_THREE_SEED_DEVELOPMENT"
        registry["completed_at_utc"] = completed_at
    atomic_json(REGISTRY, registry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 6P development requires the canonical CUDA environment")
    protocol = validate_protocol()
    protocol_hash = digest(PROTOCOL)
    tasks = build_tasks(protocol)
    process_started = time.perf_counter()
    completed = {
        task_key(task) for task in tasks if valid_result(task, protocol_hash) is not None
    }
    if args.summarize_only:
        rows = completed_rows(tasks, protocol_hash)
        atomic_csv(PHASE6P_OUT / "run_summary.csv", pd.DataFrame(rows))
        write_progress(tasks, completed, None, process_started)
        print(json.dumps({"status": "SUMMARY_RETURNED", "completed": len(completed)}))
        return
    runtime = RuntimeObjects(args.device)
    pending = [task for task in tasks if task_key(task) not in completed]
    write_progress(tasks, completed, pending[0] if pending else None, process_started)
    print(json.dumps({
        "event": "phase6p_development_start",
        "pending_runs": len(pending),
        "completed_runs": len(completed),
        "nominal_pending_seconds": math.fsum(task["budget_seconds"] for task in pending),
    }), flush=True)
    for index, task in enumerate(pending, 1):
        started_at = datetime.now(timezone.utc).isoformat()
        result, trace, replay, observer, model_load_seconds, outer_runtime = execute(
            task, runtime
        )
        payload = {
            "schema": "phase6p-development-run-v1",
            "status": "COMPLETE",
            **task,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_protocol_sha256": protocol_hash,
            "time_limit_seconds": task["budget_seconds"],
            "runtime_seconds": result.runtime,
            "outer_runtime_seconds": outer_runtime,
            "best_found_seconds": result.best_found_time,
            "final_makespan": result.best.makespan,
            "decoder_evaluations": result.decoder_evaluations,
            "iterations": result.iterations,
            "generations_if_applicable": result.generations_if_applicable,
            "feasible": True,
            "feasibility_replay": replay,
            "one_time_model_load_seconds_outside_run_budget": model_load_seconds,
            "model_residency": "load once per process; instance-specific preparation remains inside budget",
            "incumbent_trace": trace,
            "search_diagnostics": compact_diagnostics(task["method"], result.diagnostics),
            "portfolio_diagnostics": observer.summary() if observer else {},
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
            "historical_score_calls": 0 if task["method"] in {
                "P1_CSG_ADAPTIVE_PORTFOLIO", "PHASE6N_DETERMINISTIC_TOP1"
            } else None,
            "phase6o_critic_calls": 0 if task["method"] in {
                "P1_CSG_ADAPTIVE_PORTFOLIO", "PHASE6N_DETERMINISTIC_TOP1"
            } else None,
            "gurobi_run": False,
            "r13_accessed": False,
            "r14_accessed": False,
        }
        atomic_json(result_path(task), payload)
        completed.add(task_key(task))
        completed_at = payload["completed_at_utc"]
        freeze_completed_comparators(tasks, protocol_hash, completed_at)
        atomic_csv(
            PHASE6P_OUT / "run_summary.csv",
            pd.DataFrame(completed_rows(tasks, protocol_hash)),
        )
        next_task = pending[index] if index < len(pending) else None
        write_progress(tasks, completed, next_task, process_started)
        print(json.dumps({
            "event": "phase6p_run_complete",
            "batch_index": index,
            "batch_total": len(pending),
            "completed_runs": len(completed),
            "method": task["method"],
            "instance_id": task["instance_id"],
            "seed": task["seed"],
            "budget_seconds": task["budget_seconds"],
            "runtime_seconds": result.runtime,
            "final_makespan": result.best.makespan,
            "decoder_evaluations": result.decoder_evaluations,
        }), flush=True)
    print(json.dumps({"event": "phase6p_development_complete", "runs": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
