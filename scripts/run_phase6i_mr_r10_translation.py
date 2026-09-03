#!/usr/bin/env python3
"""Run the one-time, resumable Phase 6I-MR R10 solver-translation check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import (  # noqa: E402
    Phase6HLiveObserver,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.phase6i_live_inference import (  # noqa: E402
    Phase6IMRLiveInference,
)
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
MANIFEST_PATH = (
    ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11/manifests"
    / "phase6i_instance_manifest.csv"
)
SELECTION_PATH = ROOT / "outputs/phase6i_mr/r10_selection/selection_decision.json"
POLICY_PATH = ROOT / "outputs/phase6i_mr/frozen/r10_translation_policy.json"
AUTHORIZATION_PATH = ROOT / "outputs/phase6i_mr/frozen/r10_translation_authorization.json"
OUT = ROOT / "outputs/phase6i_mr/r10_translation"
RUNS = OUT / "runs"
LIVE_LOGS = OUT / "live_logs"
METHODS = ("H1", "ALNS", "PHASE6I_MR_CSGNI")
CHECKPOINT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
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


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def validate_authorization() -> tuple[dict, dict, dict]:
    config = load_json(CONFIG_PATH)
    selection = load_json(SELECTION_PATH)
    policy = load_json(POLICY_PATH)
    authorization = load_json(AUTHORIZATION_PATH)
    checks = [
        selection.get("status") == "SELECTED_IMMUTABLY_FOR_R10_TRANSLATION",
        selection.get("r10_refit_performed") is False,
        selection.get("r11_accessed") is False,
        policy.get("status") == "FROZEN_FOR_R10_TRANSLATION_ONLY",
        policy.get("r11_accessed") is False,
        authorization.get("status") == "FROZEN_BEFORE_R10_TRANSLATION",
        authorization.get("translation_started") is False,
        authorization.get("r11_accessed") is False,
        authorization.get("selection_decision_sha256") == digest(SELECTION_PATH),
        authorization.get("translation_policy_sha256") == digest(POLICY_PATH),
        authorization.get("config_sha256") == digest(CONFIG_PATH),
        authorization.get("code_hashes", {}).get(
            "rcias_clgri/ni/phase6i_live_inference.py"
        ) == digest(ROOT / "rcias_clgri/ni/phase6i_live_inference.py"),
        authorization.get("code_hashes", {}).get(
            "scripts/run_phase6i_mr_r10_translation.py"
        ) == digest(Path(__file__)),
    ]
    if not all(checks):
        raise RuntimeError("R10 translation authorization or frozen hashes are invalid")
    return config, selection, authorization


def build_tasks(config: dict) -> list[dict]:
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[
        manifest.replicate.eq("R10") & manifest.cell_replicate.eq("C02")
    ].sort_values(["scale", "CF_level", "instance_id"], kind="stable")
    if len(selected) != 9 or selected.groupby(["scale", "CF_level"]).size().ne(1).any():
        raise RuntimeError("R10 translation must use exactly nine C02 structural cells")
    seeds = tuple(int(seed) for seed in config["r10_solver_translation"]["matched_seeds"])
    if seeds != (681501, 681502):
        raise RuntimeError("R10 translation seed protocol changed")
    tasks = []
    for row in selected.itertuples(index=False):
        common = {
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "instance_sha256": row.sha256,
            "scale": row.scale,
            "CF_level": row.CF_level,
        }
        tasks.append({**common, "method": "H1", "seed": None})
        for seed in seeds:
            tasks.append({**common, "method": "ALNS", "seed": seed})
            tasks.append({**common, "method": "PHASE6I_MR_CSGNI", "seed": seed})
    if len(tasks) != 45:
        raise RuntimeError("R10 translation task count must be 45")
    return tasks


def result_path(task: dict) -> Path:
    filename = "result.json" if task["seed"] is None else f"seed_{task['seed']}.json"
    return RUNS / task["method"] / task["instance_id"] / filename


def live_log_path(task: dict) -> Path:
    return LIVE_LOGS / task["instance_id"] / f"seed_{task['seed']}.parquet"


def valid_result(task: dict, authorization_hash: str, policy_hash: str) -> dict | None:
    path = result_path(task)
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
        if not all([
            payload.get("status") == "COMPLETE",
            payload.get("instance_id") == task["instance_id"],
            payload.get("method") == task["method"],
            payload.get("seed") == task["seed"],
            payload.get("instance_sha256") == task["instance_sha256"],
            payload.get("translation_authorization_sha256") == authorization_hash,
            payload.get("r10_accessed") is True,
            payload.get("r11_accessed") is False,
        ]):
            return None
        if task["method"] == "PHASE6I_MR_CSGNI":
            log = live_log_path(task)
            if not all([
                payload.get("translation_policy_sha256") == policy_hash,
                log.is_file(),
                payload.get("live_log_sha256") == digest(log),
                payload.get("replay_integrity") is True,
            ]):
                return None
        return payload
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return None


def _candidate_from_dict(payload: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(payload["operation_order"]),
        tuple(payload["island_assignment"]),
        tuple(payload["w_assignment"]),
        tuple(payload["f_assignment"]),
    )


def _search_stage(progress: float) -> str:
    bounds = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
    return bounds[min(4, int(max(0.0, min(progress, 0.999999)) * 5))]


def validate_policy_replay(policy: Phase6IMRLiveInference) -> dict:
    replay_root = (
        ROOT / "outputs/phase6i_mr/collection/r10/state_replays"
        / "CB1_LIVE_REV_S_CF1_RI2_TI2_R10_C02__seed681301"
    )
    replay_path = sorted(replay_root.glob("*.json"))[0]
    replay = load_json(replay_path)
    instance_path = (
        ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11"
        / replay["instance_relative_path"]
    )
    instance = load_phase6i_instance(instance_path)
    current = decode_candidate(instance, _candidate_from_dict(replay["current_candidate"]))
    policy.prepare_instance(instance, solve_dispatching(instance, "H1").schedule)
    destroy_count = min(
        max(2, round(instance.num_operations * read_alns_config().destroy_fraction)),
        instance.num_operations,
    )
    evaluation = policy.evaluate(
        instance,
        current,
        state_id=replay["state_id"],
        destroy_count=destroy_count,
        search_progress=float(replay["search_progress"]),
        search_stage=_search_stage(float(replay["search_progress"])),
    )
    offline = pd.read_parquet(
        ROOT / "outputs/phase6i_mr/r10_selection/predictions/U2_MIXED_OLD_NEW.parquet"
    )
    offline = offline[offline.state_id.eq(replay["state_id"])].copy()
    live = pd.DataFrame(evaluation.candidate_diagnostics)
    shared = offline.merge(live, on="target_set_id", suffixes=("_offline", "_live"))
    raw_context_errors = {
        name: abs(
            float(offline[f"raw_context__{name}"].iloc[0])
            - float(evaluation.raw_context[name])
        )
        for name in evaluation.raw_context
    }
    score_error = float(np.max(np.abs(
        shared.ensemble_raw_score.to_numpy(dtype=float)
        - shared.raw_score_live.to_numpy(dtype=float)
    )))
    utility_error = float(np.max(np.abs(
        shared.ensemble_raw_value.to_numpy(dtype=float)
        - shared.ensemble_raw_utility.to_numpy(dtype=float)
    )))
    replay_bank_ids = {
        str(record["target_set_id"])
        for record in replay["forced_selection_inputs"]
    }
    live_bank_ids = set(live.target_set_id.astype(str))
    checks = {
        "four_offline_roles_shared": len(shared) == 4,
        "full_bank_target_ids_exact": replay_bank_ids == live_bank_ids,
        "raw_context_exact": max(raw_context_errors.values()) <= 1e-9,
        "frozen_score_within_0_01": score_error <= 0.01,
        "ensemble_utility_within_0_01": utility_error <= 0.01,
        "support_decision_exact": bool(
            offline.supported.astype(bool).eq(evaluation.decision.support_in_range).all()
        ),
    }
    fixture = {
        "schema": "phase6i-mr-r10-live-policy-replay-v1",
        "status": "PASS" if all(checks.values()) else "FAILED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "state_id": replay["state_id"],
        "replay_path": str(replay_path.relative_to(ROOT)),
        "replay_sha256": digest(replay_path),
        "shared_action_count": len(shared),
        "full_bank_action_count": len(live),
        "maximum_raw_context_error": max(raw_context_errors.values()),
        "maximum_frozen_score_error": score_error,
        "maximum_ensemble_utility_error": utility_error,
        "decision": asdict(evaluation.decision),
        "r10_accessed": True,
        "r11_accessed": False,
    }
    atomic_json(fixture, OUT / "deterministic_replay_fixture.json")
    if fixture["status"] != "PASS":
        raise RuntimeError(f"Phase 6I live policy replay failed: {checks}")
    return fixture


def validate_live_log(frame: pd.DataFrame, result) -> bool:
    if len(frame) != result.iterations or frame.empty:
        return False
    interventions = frame[frame.ni_intervention.astype(bool)]
    fallbacks = frame[frame.fallback.astype(bool)]
    selected_matches = all(
        sorted(json.loads(row.selected_operation_ids))
        == sorted(json.loads(row.executed_operation_ids))
        for row in interventions.itertuples(index=False)
    )
    return bool(all([
        frame.ni_eligible.astype(bool).all(),
        frame.candidate_feasible.astype(bool).all(),
        frame.policy_name.eq("PHASE6I_MR_U2_MIXED_OLD_NEW_3SEED").all(),
        interventions.selected_target_set_id.notna().all(),
        fallbacks.fallback_reason.notna().all(),
        selected_matches,
        len(interventions) == int(result.diagnostics["ni_interventions"]),
        len(fallbacks) == int(result.diagnostics["ni_fallbacks"]),
        int(result.diagnostics["ni_eligible_iterations"]) == len(frame),
        frame.ni_overhead_ms.ge(0).all(),
        frame.repair_time_ms.ge(0).all(),
    ]))


def write_progress(tasks: list[dict], authorization_hash: str, policy_hash: str) -> None:
    complete = [
        task for task in tasks
        if valid_result(task, authorization_hash, policy_hash) is not None
    ]
    atomic_json({
        "schema": "phase6i-mr-r10-translation-progress-v1",
        "status": "COMPLETE" if len(complete) == len(tasks) else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": len(complete),
        "total_runs": len(tasks),
        "completed_by_method": {
            method: sum(task["method"] == method for task in complete)
            for method in METHODS
        },
        "r10_accessed": True,
        "r11_accessed": False,
    }, OUT / "progress.json")


def finalize(
    tasks: list[dict],
    authorization_hash: str,
    policy_hash: str,
    replay_fixture: dict,
) -> dict:
    records = [valid_result(task, authorization_hash, policy_hash) for task in tasks]
    if any(record is None for record in records):
        raise RuntimeError("cannot finalize an incomplete R10 translation run")
    rows = pd.DataFrame([{
        "method": record["method"],
        "instance_id": record["instance_id"],
        "scale": record["scale"],
        "CF_level": record["CF_level"],
        "seed": record["seed"],
        "final_best": record["final_best"],
        "total_runtime": record["total_runtime"],
        "total_decoder_evals": record["total_decoder_evals"],
        "feasible": record["feasible"],
        "interventions": record.get("diagnostics", {}).get("ni_interventions", 0),
        "fallbacks": record.get("diagnostics", {}).get("ni_fallbacks", 0),
    } for record in records])
    summary_path = OUT / "translation_run_summary.csv"
    atomic_csv(rows, summary_path)
    iterative = rows[rows.method.ne("H1")]
    paired = (
        iterative.pivot_table(
            index=["instance_id", "scale", "CF_level"],
            columns="method",
            values="final_best",
            aggfunc="mean",
        )
        .reset_index()
    )
    paired["revised_relative_to_alns"] = (
        paired.PHASE6I_MR_CSGNI - paired.ALNS
    ) / paired.ALNS
    relative_path = OUT / "translation_paired_metrics.csv"
    atomic_csv(paired, relative_path)
    aggregate = float(paired.revised_relative_to_alns.mean())
    scale_means = {
        str(scale): float(group.revised_relative_to_alns.mean())
        for scale, group in paired.groupby("scale", sort=True)
    }
    maximum_instance = float(paired.revised_relative_to_alns.max())
    revised_rows = rows[rows.method.eq("PHASE6I_MR_CSGNI")]
    intervention_total = int(revised_rows.interventions.sum())
    fallback_total = int(revised_rows.fallbacks.sum())
    checks = {
        "all_45_runs_complete": len(rows) == 45,
        "one_h1_per_instance": len(rows[rows.method.eq("H1")]) == 9,
        "two_matched_seeds_per_iterative_method": bool(
            rows[rows.method.ne("H1")]
            .groupby(["method", "instance_id"]).seed.nunique().eq(2).all()
        ),
        "feasibility_100_percent": bool(rows.feasible.astype(bool).all()),
        "deterministic_replay_integrity": replay_fixture["status"] == "PASS",
        "live_intervention_fallback_integrity": bool(all(
            record["replay_integrity"]
            for record in records
            if record["method"] == "PHASE6I_MR_CSGNI"
        )),
        "aggregate_relative_makespan_le_0_01": aggregate <= 0.01,
        "each_scale_relative_makespan_le_0_03": all(
            value <= 0.03 for value in scale_means.values()
        ),
        "each_instance_relative_makespan_le_0_075": maximum_instance <= 0.075,
    }
    status = (
        "PASS_AUTHORIZE_SELECTED_ARTIFACT_FREEZE"
        if all(checks.values()) else "MODEL_REVISION_R10_TRANSLATION_FAILED"
    )
    decision = {
        "schema": "phase6i-mr-r10-translation-decision-v1.2",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "aggregate_revised_relative_to_alns": aggregate,
        "directional_solver_improvement_observed": aggregate < 0.0,
        "scale_mean_revised_relative_to_alns": scale_means,
        "maximum_instance_revised_relative_to_alns": maximum_instance,
        "intervention_total": intervention_total,
        "fallback_total": fallback_total,
        "intervention_coverage": (
            intervention_total / (intervention_total + fallback_total)
            if intervention_total + fallback_total else 0.0
        ),
        "translation_policy_sha256": policy_hash,
        "translation_authorization_sha256": authorization_hash,
        "selection_decision_sha256": digest(SELECTION_PATH),
        "input_output_hashes": {
            "deterministic_replay_fixture": digest(
                OUT / "deterministic_replay_fixture.json"
            ),
            "translation_run_summary": digest(summary_path),
            "translation_paired_metrics": digest(relative_path),
        },
        "no_r10_retuning": True,
        "r10_accessed": True,
        "r11_accessed": False,
    }
    atomic_json(decision, OUT / "translation_decision.json")
    write_progress(tasks, authorization_hash, policy_hash)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    final_path = OUT / "translation_decision.json"
    if final_path.exists():
        final = load_json(final_path)
        print(json.dumps({
            "event": "r10_translation_already_finalized",
            "status": final.get("status"),
        }), flush=True)
        return
    config, selection, _ = validate_authorization()
    authorization_hash = digest(AUTHORIZATION_PATH)
    policy_hash = digest(POLICY_PATH)
    tasks = build_tasks(config)
    policy_started = time.perf_counter()
    policy = Phase6IMRLiveInference(
        ROOT,
        POLICY_PATH,
        device=args.device,
        required_status="FROZEN_FOR_R10_TRANSLATION_ONLY",
    )
    model_load_seconds = time.perf_counter() - policy_started
    replay_fixture = validate_policy_replay(policy)
    write_progress(tasks, authorization_hash, policy_hash)
    pending = [
        task for task in tasks
        if valid_result(task, authorization_hash, policy_hash) is None
    ]
    print(
        f"PHASE6I_MR_R10_TRANSLATION_START pending={len(pending)} "
        f"total={len(tasks)} selected={selection['selected']['candidate_id']}",
        flush=True,
    )
    instance_root = ROOT / config["instance_suite"]["root"]
    alns_config = read_alns_config()
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["frozen_live_proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["frozen_live_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["frozen_live_acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["frozen_live_diagnostics"]),
    )
    budget_factor = float(
        config["r10_solver_translation"]["wall_clock_seconds_per_operation"]
    )
    for index, task in enumerate(pending, 1):
        instance_path = instance_root / task["instance_relative_path"]
        if digest(instance_path) != task["instance_sha256"]:
            raise RuntimeError(f"R10 instance hash mismatch: {task['instance_id']}")
        instance = load_phase6i_instance(instance_path)
        method = task["method"]
        observer = None
        if method == "H1":
            result = solve_dispatching(instance, "H1")
            final_best = float(result.objective.makespan)
            trace = [{
                "elapsed_time": float(result.runtime_seconds),
                "decoder_evaluations": 1,
                "current_best_makespan": final_best,
            }]
            total_runtime = float(result.runtime_seconds)
            total_evals = 1
            iterations = 0
            initialization_seconds = total_runtime
            best_schedule = result.schedule
            best_actions = result.actions
            diagnostics = {}
            budget = None
            replay_integrity = True
            log_hash = None
        else:
            budget = budget_factor * instance.num_operations
            if method == "ALNS":
                result = solve_alns(instance, budget, task["seed"], alns_config)
            else:
                observer = Phase6HLiveObserver({
                    **task,
                    "live_revision_split": "LIVE_REV_SELECT",
                    "policy_name": policy.policy_name,
                })
                result = solve_csgni(
                    instance,
                    budget,
                    task["seed"],
                    policy,
                    alns_config=alns_config,
                    csgni_config=csgni_config,
                    observer=observer,
                )
            final_best = float(result.best.makespan)
            trace = validate_incumbent_trace(result.convergence_trace, final_best=final_best)
            total_runtime = float(result.runtime)
            total_evals = int(result.decoder_evaluations)
            iterations = int(result.iterations)
            initialization_seconds = float(
                result.diagnostics.get("initialization_seconds", trace[0]["elapsed_time"])
            )
            best_schedule = result.best.schedule
            best_actions = result.best.actions
            diagnostics = result.diagnostics
            if observer is not None:
                live = pd.DataFrame(observer.rows)
                replay_integrity = validate_live_log(live, result)
                log_path = live_log_path(task)
                atomic_parquet(live, log_path)
                log_hash = digest(log_path)
            else:
                replay_integrity = True
                log_hash = None
        feasibility = check_schedule(instance, best_schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        payload = {
            "schema": "phase6i-mr-r10-translation-run-v1.2",
            "status": "COMPLETE",
            **task,
            "time_limit_seconds": budget,
            "final_best": final_best,
            "time_to_best": float(trace[-1]["elapsed_time"]),
            "evals_to_best": int(trace[-1]["decoder_evaluations"]),
            "total_runtime": total_runtime,
            "total_decoder_evals": total_evals,
            "initialization_seconds": initialization_seconds,
            "iterations": iterations,
            "feasible": True,
            "model_load_seconds_excluded_from_budget": (
                model_load_seconds if method == "PHASE6I_MR_CSGNI" else None
            ),
            "translation_policy_sha256": (
                policy_hash if method == "PHASE6I_MR_CSGNI" else None
            ),
            "translation_authorization_sha256": authorization_hash,
            "live_log_sha256": log_hash,
            "replay_integrity": replay_integrity,
            "incumbent_trace": trace,
            "normalized_budget_checkpoints": (
                [] if budget is None else sample_incumbent_trace(
                    trace, budget=budget, fractions=CHECKPOINT_FRACTIONS
                )
            ),
            "diagnostics": diagnostics,
            "best_solution": best_schedule.to_dict(),
            "best_actions": [asdict(action) for action in best_actions],
            "r10_accessed": True,
            "r11_accessed": False,
        }
        atomic_json(payload, result_path(task))
        write_progress(tasks, authorization_hash, policy_hash)
        print(
            f"[{index}/{len(pending)}] {method} {task['instance_id']} "
            f"seed={task['seed']} makespan={final_best:g} evals={total_evals} "
            f"runtime={total_runtime:.2f}s",
            flush=True,
        )
    decision = finalize(
        tasks, authorization_hash, policy_hash, replay_fixture
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
