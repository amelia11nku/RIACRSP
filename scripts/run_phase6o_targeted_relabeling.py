#!/usr/bin/env python3
"""Generate the frozen Phase 6O targeted high-fidelity continuation labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    FrozenArmPrediction,
    decode_forced_candidate,
)
from rcias_clgri.analysis.phase6j_caur import (  # noqa: E402
    continue_frozen_alns_at_horizons,
    fallback_relative_advantage,
)
from rcias_clgri.analysis.phase6l_legacy_score import select_score_free_fallback  # noqa: E402
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402
from scripts.run_phase6j_caur_pilot import read_alns_config  # noqa: E402


CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
PHASE6J_CONFIG = ROOT / "configs/phase6j_caur.json"
PREREG = ROOT / "outputs/phase6o_neural_shortlist_v1/preregistration"
PREREGISTRATION = PREREG / "preregistration.json"
SOURCE_HASHES = PREREG / "source_hashes.json"
PLAN = PREREG / "targeted_relabeling_plan.json"
UNION = PREREG / "targeted_relabel_candidate_union.csv"
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling"
RAW_SHARDS = OUT / "raw_additional_seed_labels"
STATUS_SHARDS = OUT / "state_status"
IMPLEMENTATION = OUT / "implementation.json"
PROGRESS = OUT / "progress.json"
REPORT = ROOT / "docs/reports/phase6o_targeted_relabeling_report.md"
PHASE6N_GROUPED = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
PHASE6N_RAW = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_seed_labels.parquet"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def candidate_from_dict(value: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(value["operation_order"]),
        tuple(value["island_assignment"]),
        tuple(value["w_assignment"]),
        tuple(value["f_assignment"]),
    )


def prediction(arm) -> FrozenArmPrediction:
    return FrozenArmPrediction(
        target_set_id=arm.target_set_id,
        arm_family=arm.arm_family,
        origin_destroy_operator=arm.origin_destroy_operator,
        origin_rules=arm.origin_rules,
        destroyed_operations=arm.destroyed_operations,
        raw_score=0.0,
        raw_probability=0.0,
        raw_utility=0.0,
        calibrated_probability=0.0,
        calibrated_utility=0.0,
    )


def validate_boundary() -> tuple[dict, dict, dict, pd.DataFrame, str]:
    config = load_json(CONFIG)
    phase6j_config = load_json(PHASE6J_CONFIG)
    prereg = load_json(PREREGISTRATION)
    plan = load_json(PLAN)
    require(prereg["status"] == "FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP", "Phase 6O preregistration is not frozen")
    require(prereg["route"] == "PROCEED_TOP_UTILITY_RETRAIN", "Phase 6O route changed")
    require(prereg["config_sha256"] == digest(CONFIG), "Phase 6O config changed")
    require(prereg["source_hashes_sha256"] == digest(SOURCE_HASHES), "Phase 6O source hashes changed")
    require(prereg["targeted_relabeling_plan_sha256"] == digest(PLAN), "Phase 6O relabel plan changed")
    require(prereg["candidate_union_sha256"] == digest(UNION), "Phase 6O candidate union changed")
    source_hashes = load_json(SOURCE_HASHES)
    require(source_hashes["config_sha256"] == digest(CONFIG), "config/source hash mismatch")
    for relative, expected in source_hashes["sources"].items():
        require(digest(ROOT / relative) == expected, f"locked source changed: {relative}")
    require(plan["additional_crn_seeds"] == config["targeted_relabeling"]["additional_crn_seeds"], "additional CRN seeds changed")
    require(int(phase6j_config["rng"]["proposal_namespace"]) == 692000000, "inherited proposal namespace changed")
    require(int(plan["repair_seed_namespace"]) == int(phase6j_config["rng"]["repair_namespace"]) == 693000000, "inherited repair namespace changed")
    require(int(plan["continuation_seed_namespace"]) == int(phase6j_config["rng"]["continuation_namespace"]) == 694000000, "inherited continuation namespace changed")
    require(plan["actual_additional_continuation_rows"] == 14358, "relabel row count changed")
    union = pd.read_csv(UNION)
    require(len(union) == plan["actual_state_candidate_rows"] == 4786, "candidate union is incomplete")
    require(union.state_id.nunique() == 864, "candidate union state count changed")
    for relative in (
        "outputs/phase6j_caur/r13_selection/access_ledger.json",
        "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json",
        "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json",
    ):
        require(not (ROOT / relative).exists(), f"forbidden holdout access: {relative}")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", prereg["o0_evidence_commit"], "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode == 0,
        "O0 evidence commit is not an ancestor of HEAD",
    )
    return config, phase6j_config, plan, union, digest(PREREGISTRATION)


def freeze_implementation(preregistration_sha256: str) -> dict:
    code_paths = [
        Path(__file__),
        ROOT / "rcias_clgri/analysis/phase6i_mr.py",
        ROOT / "rcias_clgri/analysis/phase6j_caur.py",
        ROOT / "rcias_clgri/analysis/phase6l_legacy_score.py",
        ROOT / "rcias_clgri/data/phase6j_access.py",
        ROOT / "rcias_clgri/env/feasibility.py",
        ROOT / "rcias_clgri/search/alns.py",
        ROOT / "rcias_clgri/search/common.py",
        ROOT / "rcias_clgri/search/phase6c.py",
        ROOT / "scripts/run_phase6j_caur_pilot.py",
    ]
    code_sha256 = {
        str(path.relative_to(ROOT)): digest(path) for path in code_paths
    }
    runtime_dependency_sha256 = {
        str(PHASE6J_CONFIG.relative_to(ROOT)): digest(PHASE6J_CONFIG),
    }
    if IMPLEMENTATION.exists():
        frozen = load_json(IMPLEMENTATION)
        require(frozen["schema"] == "phase6o-targeted-relabel-implementation-v1", "relabel implementation schema changed")
        require(frozen["preregistration_sha256"] == preregistration_sha256, "relabel preregistration changed after outcomes")
        require(frozen["plan_sha256"] == digest(PLAN), "relabel plan changed after outcomes")
        require(frozen["candidate_union_sha256"] == digest(UNION), "relabel union changed after outcomes")
        require(frozen["code_sha256"] == code_sha256, "relabel implementation changed after outcomes")
        require(frozen["runtime_dependency_sha256"] == runtime_dependency_sha256, "relabel runtime dependency changed after outcomes")
        require(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", frozen["implementation_commit"], "HEAD"],
                cwd=ROOT,
                check=False,
            ).returncode == 0,
            "frozen relabel implementation commit is not an ancestor of HEAD",
        )
        return frozen
    payload = {
        "schema": "phase6o-targeted-relabel-implementation-v1",
        "status": "FROZEN_BEFORE_FIRST_ADDITIONAL_OUTCOME",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preregistration_sha256": preregistration_sha256,
        "plan_sha256": digest(PLAN),
        "candidate_union_sha256": digest(UNION),
        "code_sha256": code_sha256,
        "runtime_dependency_sha256": runtime_dependency_sha256,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(IMPLEMENTATION, payload)
    return payload


def replay_path(source: str, state_id: str) -> Path:
    if source == "ORIGINAL_PHASE6J_CAUR":
        root = ROOT / "outputs/phase6j_caur/r12_collection/state_replays"
    elif source == "NEW_ALNS_EXPANSION":
        root = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/new_r12_collection/state_replays"
    else:
        raise ValueError(f"unknown Phase 6N data origin: {source}")
    return root / f"{state_id}.json"


def state_paths(state_id: str) -> tuple[Path, Path]:
    return RAW_SHARDS / f"{state_id}.parquet", STATUS_SHARDS / f"{state_id}.json"


def valid_state(state_id: str, implementation: dict, expected_candidates: int, seeds: list[int]) -> dict | None:
    raw_path, status_path = state_paths(state_id)
    if not raw_path.exists() or not status_path.exists():
        return None
    try:
        status = load_json(status_path)
    except (OSError, json.JSONDecodeError):
        return None
    expected = (
        status.get("schema") == "phase6o-targeted-relabel-state-v1"
        and status.get("status") == "COMPLETE"
        and status.get("state_id") == state_id
        and status.get("implementation_commit") == implementation["implementation_commit"]
        and status.get("worker_sha256") == implementation["code_sha256"]["scripts/run_phase6o_targeted_relabeling.py"]
        and status.get("candidate_count") == expected_candidates
        and status.get("additional_rows") == expected_candidates * len(seeds)
        and status.get("additional_crn_seeds") == seeds
        and status.get("raw_sha256") == digest(raw_path)
        and status.get("historical_score_calls") == 0
        and status.get("r13_accessed") is False
        and status.get("r14_accessed") is False
    )
    return status if expected else None


def collect_state(
    state_union: pd.DataFrame,
    config: dict,
    plan: dict,
    implementation: dict,
    alns_config,
) -> dict:
    state_id = str(state_union.state_id.iloc[0])
    source = str(state_union.phase6n_data_origin.iloc[0])
    seeds = [int(value) for value in plan["additional_crn_seeds"]]
    existing = valid_state(state_id, implementation, len(state_union), seeds)
    if existing is not None:
        return existing
    started = time.perf_counter()
    replay_file = replay_path(source, state_id)
    replay = load_json(replay_file)
    require(replay["state_id"] == state_id, "state replay identity mismatch")
    instance_path = ROOT / config["locked_inputs"]["instance_root"] / replay["instance_relative_path"]
    require(digest(instance_path) == replay["instance_sha256"], "instance hash mismatch")
    instance = load_phase6j_instance(instance_path)
    current = decode_candidate(instance, candidate_from_dict(replay["snapshot"]["current_candidate"]))
    require(math.isclose(current.makespan, float(replay["current_replay_makespan"]), abs_tol=1e-9), "current-state replay mismatch")
    destroy_count = min(
        max(2, round(instance.num_operations * alns_config.destroy_fraction)),
        instance.num_operations,
    )
    generated = generate_revised_target_arms(
        instance,
        current,
        state_id,
        destroy_count,
        int(config["rng"]["proposal_namespace"]),
    )
    require(generated.requested_arm_count == 24, "full 24-rule bank was not generated")
    replay_ids = replay.get("full_bank_target_ids_in_generator_order", replay.get("full_bank_target_ids"))
    require([arm.target_set_id for arm in generated.arms] == replay_ids, "full-bank candidate identity/order changed")
    fallback = select_score_free_fallback(generated)
    require(fallback.target_set_id == replay["fallback_target_set_id"], "canonical fallback changed")
    selected_ids = set(state_union.target_set_id.astype(str))
    require(fallback.target_set_id in selected_ids, "targeted union omitted fallback")
    arms = [arm for arm in generated.arms if arm.target_set_id in selected_ids]
    require(len(arms) == len(selected_ids), "targeted union escaped generated full bank")

    decoded = {}
    forced_decode_seconds = 0.0
    for arm in arms:
        result = decode_forced_candidate(
            instance,
            current,
            prediction(arm),
            state_id=state_id,
            repair_seed_namespace=int(plan["repair_seed_namespace"]),
            candidate_trials=int(plan["candidate_trials_per_target"]),
        )
        require(result.candidate.feasible, "targeted candidate repair is infeasible")
        require(check_schedule(instance, result.candidate.schedule)["feasible"], "targeted candidate schedule is infeasible")
        decoded[arm.target_set_id] = result
        forced_decode_seconds += result.runtime_ms / 1000.0

    horizon = int(plan["continuation_horizon"])
    rows = []
    for seed in seeds:
        continuations = {
            target_id: continue_frozen_alns_at_horizons(
                instance,
                repaired.candidate,
                state_id=state_id,
                continuation_seed=seed,
                seed_namespace=int(plan["continuation_seed_namespace"]),
                horizons=[horizon],
                config=alns_config,
            )[horizon]
            for target_id, repaired in decoded.items()
        }
        fallback_prefix = continuations[fallback.target_set_id]
        for target_id, prefix in continuations.items():
            require(prefix.result.derived_seed == fallback_prefix.result.derived_seed, "candidate/fallback CRN seed mismatch")
            rows.append({
                "instance_id": replay["instance_id"],
                "scale": str(state_union.scale.iloc[0]),
                "CF_level": str(state_union.CF_level.iloc[0]),
                "state_id": state_id,
                "target_set_id": target_id,
                "fallback_target_set_id": fallback.target_set_id,
                "continuation_seed": seed,
                "horizon": horizon,
                "candidate_continuation_best_makespan": prefix.result.best_makespan,
                "fallback_continuation_best_makespan": fallback_prefix.result.best_makespan,
                "continuation_advantage": fallback_relative_advantage(
                    fallback_prefix.result.best_makespan,
                    prefix.result.best_makespan,
                    current.makespan,
                ),
                "paired_derived_seed": prefix.result.derived_seed,
                "is_fallback": target_id == fallback.target_set_id,
                "phase6n_data_origin": source,
                "candidate_continuation_decoder_seconds": prefix.decoder_seconds,
                "fallback_continuation_decoder_seconds": fallback_prefix.decoder_seconds,
            })
    raw = pd.DataFrame(rows).sort_values(
        ["target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(len(raw) == len(arms) * len(seeds), "additional label row count mismatch")
    require(raw.groupby("target_set_id").continuation_seed.nunique().eq(len(seeds)).all(), "additional seed coverage mismatch")
    require(np.isfinite(raw.continuation_advantage).all(), "non-finite additional label")
    raw_path, status_path = state_paths(state_id)
    atomic_parquet(raw_path, raw)
    status = {
        "schema": "phase6o-targeted-relabel-state-v1",
        "status": "COMPLETE",
        "state_id": state_id,
        "instance_id": replay["instance_id"],
        "phase6n_data_origin": source,
        "implementation_commit": implementation["implementation_commit"],
        "worker_sha256": implementation["code_sha256"]["scripts/run_phase6o_targeted_relabeling.py"],
        "candidate_count": len(arms),
        "additional_rows": len(raw),
        "additional_crn_seeds": seeds,
        "forced_decode_seconds": forced_decode_seconds,
        "continuation_decoder_seconds": float(raw.candidate_continuation_decoder_seconds.sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": digest(raw_path),
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(status_path, status)
    return status


def progress_payload(statuses: list[dict], started: float, status: str) -> dict:
    elapsed = sum(float(row["elapsed_seconds"]) for row in statuses)
    average = elapsed / len(statuses) if statuses else None
    return {
        "schema": "phase6o-targeted-relabel-progress-v1",
        "status": status,
        "states_complete": len(statuses),
        "states_expected": 864,
        "state_candidate_rows_complete": sum(int(row["candidate_count"]) for row in statuses),
        "state_candidate_rows_expected": 4786,
        "additional_rows_complete": sum(int(row["additional_rows"]) for row in statuses),
        "additional_rows_expected": 14358,
        "measured_state_seconds": elapsed,
        "measured_seconds_per_state": average,
        "measured_remaining_state_seconds": (864 - len(statuses)) * average if average is not None else None,
        "process_elapsed_seconds": time.perf_counter() - started,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def summarize(union: pd.DataFrame, statuses: list[dict], implementation: dict) -> dict:
    require(len(statuses) == 864, "cannot summarize incomplete targeted relabeling")
    additional = pd.concat(
        [pd.read_parquet(state_paths(str(state_id))[0]) for state_id in sorted(union.state_id.unique())],
        ignore_index=True,
    ).sort_values(["state_id", "target_set_id", "continuation_seed"], kind="stable").reset_index(drop=True)
    require(len(additional) == 14358, "additional label total changed")
    existing = pd.read_parquet(PHASE6N_RAW).copy()
    standard_columns = list(existing.columns)
    require(set(standard_columns).issubset(additional.columns), "additional label schema is incomplete")
    existing["label_provenance"] = "PHASE6N_EXISTING_TWO_CRN"
    selected = additional[standard_columns].copy()
    selected["label_provenance"] = "PHASE6O_TARGETED_ADDITIONAL_CRN"
    combined = pd.concat([existing, selected], ignore_index=True).sort_values(
        ["state_id", "target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(not combined[["state_id", "target_set_id", "continuation_seed"]].duplicated().any(), "duplicate combined CRN label")
    targeted_keys = set(zip(union.state_id.astype(str), union.target_set_id.astype(str)))
    seed_counts = combined.groupby(["state_id", "target_set_id"]).continuation_seed.nunique()
    require(set(seed_counts.unique()) == {2, 5}, "merged seed counts must be two or five")
    require(int(seed_counts.eq(5).sum()) == len(union), "targeted candidates do not have five seeds")
    metrics = combined.groupby(["state_id", "target_set_id"], sort=True).agg(
        continuation_seed_count=("continuation_seed", "nunique"),
        continuation_advantage_mean=("continuation_advantage", "mean"),
        continuation_advantage_std=("continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        beats_fallback=("continuation_advantage", lambda x: float(np.mean(np.asarray(x) > 0))),
        continuation_best_makespan=("candidate_continuation_best_makespan", "mean"),
        fallback_continuation_best_makespan=("fallback_continuation_best_makespan", "mean"),
    ).reset_index()
    grouped = pd.read_parquet(PHASE6N_GROUPED).copy()
    original_candidate_order = list(zip(grouped.state_id.astype(str), grouped.target_set_id.astype(str)))
    grouped["_phase6o_source_row"] = np.arange(len(grouped))
    replace = [
        "continuation_seed_count",
        "continuation_advantage_mean",
        "continuation_advantage_std",
        "beats_fallback",
        "continuation_best_makespan",
        "fallback_continuation_best_makespan",
    ]
    grouped = grouped.drop(columns=replace).merge(
        metrics, on=["state_id", "target_set_id"], validate="one_to_one"
    )
    grouped["phase6o_targeted_relabel"] = [
        (str(state), str(target)) in targeted_keys
        for state, target in zip(grouped.state_id, grouped.target_set_id)
    ]
    grouped = grouped.sort_values("_phase6o_source_row", kind="stable").drop(
        columns="_phase6o_source_row"
    ).reset_index(drop=True)
    require(
        list(zip(grouped.state_id.astype(str), grouped.target_set_id.astype(str)))
        == original_candidate_order,
        "Phase 6N candidate row order changed",
    )
    require(len(grouped) == 20441 and grouped.state_id.nunique() == 864, "training labels lost full-bank rows")
    require(grouped.groupby("state_id").is_fallback.sum().eq(1).all(), "fallback integrity changed")
    require(grouped.candidate_feasible.astype(bool).all(), "infeasible candidate entered labels")
    combined_path = OUT / "combined_seed_labels.parquet"
    training_path = OUT / "training_grouped_labels.parquet"
    atomic_parquet(combined_path, combined)
    atomic_parquet(training_path, grouped)
    audit = {
        "schema": "phase6o-targeted-relabel-integrity-v1",
        "status": "PASS",
        "implementation_commit": implementation["implementation_commit"],
        "states": 864,
        "instances": int(grouped.instance_id.nunique()),
        "full_bank_candidates": len(grouped),
        "targeted_candidates": len(union),
        "additional_seed_rows": len(additional),
        "combined_seed_rows": len(combined),
        "targeted_five_seed_candidates": int(seed_counts.eq(5).sum()),
        "non_targeted_two_seed_candidates": int(seed_counts.eq(2).sum()),
        "candidate_feasibility": True,
        "canonical_fallback_per_state": True,
        "candidate_identity_and_order_preserved": True,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "artifacts": {
            str(combined_path.relative_to(ROOT)): digest(combined_path),
            str(training_path.relative_to(ROOT)): digest(training_path),
        },
    }
    atomic_json(OUT / "integrity.json", audit)
    REPORT.write_text(f"""# Phase 6O Targeted Relabeling 报告

状态：**PASS**。

按 O1 冻结 union 完成 864 states、{len(union):,} targeted candidates 和 {len(additional):,} additional three-CRN rows。Targeted candidates 现使用五 seed 统计，其他 {audit['non_targeted_two_seed_candidates']:,} candidates 保持 Phase 6N 两 seed truth；没有覆盖任何 Phase 6L/6N 文件。

每个状态均先重建完整 24-rule bank并验证候选身份/顺序，再只对冻结 union 执行 deterministic repair 与 H=4 continuation。全部候选可行，每状态 canonical fallback 恰好一个。historical scorer calls 为 0；未运行 Gurobi；R13/R14 未访问。
""")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-states", type=int)
    args = parser.parse_args()
    config, phase6j_config, plan, union, preregistration_sha256 = validate_boundary()
    implementation = freeze_implementation(preregistration_sha256)
    alns_config = read_alns_config(phase6j_config)
    groups = [(state, group.reset_index(drop=True)) for state, group in union.groupby("state_id", sort=True)]
    started = time.perf_counter()
    statuses = []
    new_states = 0
    for state_id, group in groups:
        existing = valid_state(
            str(state_id), implementation, len(group),
            [int(value) for value in plan["additional_crn_seeds"]],
        )
        if existing is None:
            if args.max_new_states is not None and new_states >= args.max_new_states:
                continue
            existing = collect_state(group, config, plan, implementation, alns_config)
            new_states += 1
        statuses.append(existing)
        running = progress_payload(statuses, started, "RUNNING")
        atomic_json(PROGRESS, running)
        print(json.dumps({
            "state_id": str(state_id),
            "states_complete": running["states_complete"],
            "states_expected": running["states_expected"],
            "measured_remaining_state_seconds": running["measured_remaining_state_seconds"],
        }, sort_keys=True), flush=True)
    if len(statuses) != 864:
        payload = progress_payload(statuses, started, "PARTIAL")
        payload["new_states_this_run"] = new_states
        atomic_json(PROGRESS, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2)
    audit = summarize(union, statuses, implementation)
    progress = progress_payload(statuses, started, "COMPLETE")
    progress["new_states_this_run"] = new_states
    progress["integrity_path"] = str((OUT / "integrity.json").relative_to(ROOT))
    progress["integrity_sha256"] = digest(OUT / "integrity.json")
    atomic_json(PROGRESS, progress)
    print(json.dumps({"progress": progress, "integrity": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
