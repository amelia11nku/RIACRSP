#!/usr/bin/env python3
"""Run the preregistered R09-only Phase 6I-MR continuation diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    FrozenArmPrediction,
    continue_frozen_alns_from_candidate,
    decode_forced_candidate,
)
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
PILOT = ROOT / "outputs/phase6i_mr/pilot_v12"
OUT = ROOT / "outputs/phase6i_mr/continuation"
STATE_RUNS = OUT / "state_runs"
TARGET_PROGRESS = (0.15, 0.50, 0.85)


def configure_output_root(path: Path) -> None:
    global OUT, STATE_RUNS
    OUT = path
    STATE_RUNS = OUT / "state_runs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value
        for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def candidate_from_dict(payload: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(payload["operation_order"]),
        tuple(payload["island_assignment"]),
        tuple(payload["w_assignment"]),
        tuple(payload["f_assignment"]),
    )


def arm_from_row(row) -> FrozenArmPrediction:
    return FrozenArmPrediction(
        target_set_id=str(row.target_set_id),
        arm_family=str(row.origin_family),
        origin_destroy_operator=str(row.origin_destroy_operator),
        origin_rules=tuple(json.loads(row.origin_rules)),
        destroyed_operations=tuple(json.loads(row.target_operation_ids)),
        raw_score=float(row.raw_score),
        raw_probability=float(row.raw_probability),
        raw_utility=float(row.raw_utility),
        calibrated_probability=float(row.calibrated_probability),
        calibrated_utility=float(row.calibrated_utility),
    )


def select_tasks(actions: pd.DataFrame) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for instance_id, instance_rows in actions.groupby("instance_id", sort=True):
        states = instance_rows[[
            "state_id",
            "instance_relative_path",
            "scale",
            "CF_level",
            "target_progress",
            "search_progress",
            "search_stage",
        ]].drop_duplicates("state_id")
        used: set[str] = set()
        for target in TARGET_PROGRESS:
            choices = states[~states.state_id.isin(used)].copy()
            choices["distance"] = (choices.target_progress - target).abs()
            selected = choices.sort_values(
                ["distance", "target_progress", "state_id"]
            ).iloc[0]
            used.add(str(selected.state_id))
            tasks.append({
                "instance_id": str(instance_id),
                "instance_relative_path": str(selected.instance_relative_path),
                "scale": str(selected.scale),
                "CF_level": str(selected.CF_level),
                "state_id": str(selected.state_id),
                "target_continuation_progress": float(target),
                "pilot_target_progress": float(selected.target_progress),
                "search_progress": float(selected.search_progress),
                "search_stage": str(selected.search_stage),
            })
    if len(tasks) != 27 or len({str(task["state_id"]) for task in tasks}) != 27:
        raise RuntimeError("continuation diagnostic requires 27 unique R09 pilot states")
    return tasks


def replay_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted((PILOT / "state_replays").rglob("*.json")):
        state_id = path.stem
        if state_id in index:
            raise RuntimeError(f"duplicate replay for {state_id}")
        index[state_id] = path
    return index


def table_path(task: dict[str, object]) -> Path:
    return STATE_RUNS / f"{task['state_id']}.parquet"


def result_path(task: dict[str, object]) -> Path:
    return STATE_RUNS / f"{task['state_id']}.json"


def valid_result(
    task: dict[str, object],
    *,
    config_sha256: str,
    pilot_sha256: str,
) -> dict | None:
    result_file = result_path(task)
    table_file = table_path(task)
    if not result_file.is_file() or not table_file.is_file():
        return None
    try:
        result = load_json(result_file)
    except (OSError, json.JSONDecodeError):
        return None
    if not all([
        result.get("status") == "COMPLETE",
        result.get("state_id") == task["state_id"],
        result.get("config_sha256") == config_sha256,
        result.get("pilot_integrity_sha256") == pilot_sha256,
        result.get("table_sha256") == digest(table_file),
        result.get("rows") == 8,
        result.get("r10_accessed") is False,
        result.get("r11_accessed") is False,
    ]):
        return None
    return result


def finite_spearman(left: pd.Series, right: pd.Series) -> tuple[float, bool]:
    if left.nunique(dropna=False) < 2 or right.nunique(dropna=False) < 2:
        return 0.0, True
    result = spearmanr(left, right)
    value = float(result.statistic)
    if math.isfinite(value):
        return value, False
    return 0.0, True


def summarize(
    tasks: list[dict[str, object]],
    config: dict,
    *,
    config_sha256: str,
    pilot_sha256: str,
) -> None:
    frames = []
    for task in tasks:
        if valid_result(
            task,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        ) is not None:
            frames.append(pd.read_parquet(table_path(task)))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metric_rows = []
    if len(rows):
        for (state_id, seed), group in rows.groupby(
            ["state_id", "continuation_seed"], sort=True
        ):
            correlation, constant = finite_spearman(
                group.decoded_immediate_utility,
                group.continuation_value,
            )
            immediate_top = group.sort_values(
                ["decoded_immediate_utility", "target_set_id"],
                ascending=[False, True],
            ).iloc[0]
            continuation_top = group.sort_values(
                ["continuation_value", "target_set_id"],
                ascending=[False, True],
            ).iloc[0]
            metric_rows.append({
                "state_id": state_id,
                "instance_id": immediate_top.instance_id,
                "scale": immediate_top.scale,
                "CF_level": immediate_top.CF_level,
                "search_stage": immediate_top.search_stage,
                "continuation_seed": int(seed),
                "within_state_spearman": correlation,
                "constant_rank_input": constant,
                "immediate_top_target_set_id": immediate_top.target_set_id,
                "continuation_top_target_set_id": continuation_top.target_set_id,
                "top1_agreement": bool(
                    immediate_top.target_set_id
                    == continuation_top.target_set_id
                ),
            })
    metrics = pd.DataFrame(metric_rows)
    diagnostic = config["continuation_diagnostic"]
    median_spearman = (
        float(metrics.within_state_spearman.median())
        if len(metrics) else math.nan
    )
    top1_agreement = (
        float(metrics.top1_agreement.mean()) if len(metrics) else math.nan
    )
    complete = all([
        len(rows) == 216,
        rows.state_id.nunique() == 27 if len(rows) else False,
        len(metrics) == 54,
        bool(rows.start_replay_match.all()) if len(rows) else False,
        bool(rows.start_feasible.all()) if len(rows) else False,
        bool(rows.continuation_feasible.all()) if len(rows) else False,
        bool(rows.continuation_iterations.eq(12).all()) if len(rows) else False,
        bool(rows.continuation_decoder_evaluations.eq(96).all()) if len(rows) else False,
        bool(rows.split.eq("R09_PILOT_ONLY").all()) if len(rows) else False,
    ])
    branch = (
        "IMMEDIATE_TARGET_VALID"
        if complete and median_spearman >= 0.30 and top1_agreement >= 0.50
        else "TARGET_MISMATCH"
    )
    by_scale = (
        metrics.groupby("scale").agg(
            state_seed_groups=("state_id", "count"),
            median_spearman=("within_state_spearman", "median"),
            mean_spearman=("within_state_spearman", "mean"),
            top1_agreement=("top1_agreement", "mean"),
        ).reset_index()
        if len(metrics) else pd.DataFrame()
    )
    by_stage = (
        metrics.groupby("search_stage").agg(
            state_seed_groups=("state_id", "count"),
            median_spearman=("within_state_spearman", "median"),
            mean_spearman=("within_state_spearman", "mean"),
            top1_agreement=("top1_agreement", "mean"),
        ).reset_index()
        if len(metrics) else pd.DataFrame()
    )
    summary = {
        "schema": "phase6i-mr-continuation-branch-v1.2",
        "status": "PASS" if complete else "INCOMPLETE",
        "branch": branch if complete else None,
        "config_sha256": config_sha256,
        "pilot_integrity_sha256": pilot_sha256,
        "states": int(rows.state_id.nunique()) if len(rows) else 0,
        "state_seed_groups": len(metrics),
        "action_continuations": len(rows),
        "median_within_state_spearman": median_spearman,
        "top1_agreement": top1_agreement,
        "thresholds": {
            "median_within_state_spearman_min": 0.30,
            "top1_agreement_min": 0.50,
        },
        "constant_rank_groups_conservatively_scored_zero": int(
            metrics.constant_rank_input.sum()
        ) if len(metrics) else 0,
        "u2h_activated": branch == "TARGET_MISMATCH" if complete else None,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_parquet(rows, OUT / "continuation_action_table.parquet")
    atomic_csv(rows, OUT / "continuation_action_table.csv")
    atomic_csv(metrics, OUT / "continuation_state_seed_metrics.csv")
    atomic_csv(by_scale, OUT / "continuation_by_scale.csv")
    atomic_csv(by_stage, OUT / "continuation_by_search_stage.csv")
    atomic_json(summary, OUT / "continuation_branch_decision.json")
    report = f"""# Phase 6I-MR continuation diagnostic

```text
status = {summary['status']}
branch = {summary['branch']}
r10_accessed = false
r11_accessed = false
states = {summary['states']}
state_seed_groups = {summary['state_seed_groups']}
action_continuations = {summary['action_continuations']}
```

The preregistered median within-state Spearman is {median_spearman:+.4f}
(required >= 0.30), and immediate/continuation top-1 agreement is
{top1_agreement:.2%} (required >= 50%). The immutable branch is therefore
`{summary['branch']}`. Constant-input rank groups are conservatively assigned
zero association rather than being dropped.

Every action starts from the pilot's deterministically replayed forced decode,
uses one of the two common random seeds within its state, and receives exactly
12 frozen ALNS iterations with eight decoder trials per iteration. The branch
does not authorize candidate-bank changes. `TARGET_MISMATCH` activates the
separate U2-H continuation head while retaining the immediate-utility target.
"""
    (OUT / "phase6i_mr_continuation_report.md").write_text(
        report, encoding="utf-8"
    )
    required = [
        OUT / "continuation_action_table.parquet",
        OUT / "continuation_action_table.csv",
        OUT / "continuation_state_seed_metrics.csv",
        OUT / "continuation_by_scale.csv",
        OUT / "continuation_by_search_stage.csv",
        OUT / "continuation_branch_decision.json",
        OUT / "phase6i_mr_continuation_report.md",
    ]
    atomic_json({
        "schema": "phase6i-mr-continuation-integrity-v1.2",
        "status": "PASS" if complete else "INCOMPLETE",
        "checks": {
            "exactly_27_states": int(rows.state_id.nunique()) == 27 if len(rows) else False,
            "exactly_54_state_seed_groups": len(metrics) == 54,
            "exactly_216_action_continuations": len(rows) == 216,
            "four_roles_per_state_seed": bool(
                rows.groupby(["state_id", "continuation_seed"])
                .candidate_role.nunique().eq(4).all()
            ) if len(rows) else False,
            "all_replays_exact": bool(rows.start_replay_match.all()) if len(rows) else False,
            "all_feasible": bool(
                rows.start_feasible.all() and rows.continuation_feasible.all()
            ) if len(rows) else False,
            "fixed_12_by_8_budget": bool(
                rows.continuation_iterations.eq(12).all()
                and rows.continuation_decoder_evaluations.eq(96).all()
            ) if len(rows) else False,
            "r09_only": bool(rows.split.eq("R09_PILOT_ONLY").all()) if len(rows) else False,
        },
        "required_outputs": {path.name: digest(path) for path in required},
        "r10_accessed": False,
        "r11_accessed": False,
    }, OUT / "continuation_integrity.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def write_progress(
    tasks: list[dict[str, object]],
    *,
    started: float,
    config_sha256: str,
    pilot_sha256: str,
) -> None:
    complete = sum(
        valid_result(
            task,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        ) is not None
        for task in tasks
    )
    elapsed = time.perf_counter() - started
    rate = complete / elapsed if elapsed > 0 else 0.0
    remaining = (len(tasks) - complete) / rate if rate > 0 else None
    atomic_json({
        "schema": "phase6i-mr-continuation-progress-v1.2",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_states": complete,
        "total_states": len(tasks),
        "current_process_elapsed_seconds": elapsed,
        "current_process_naive_remaining_seconds": remaining,
        "status": "COMPLETE" if complete == len(tasks) else "RUNNING",
        "r10_accessed": False,
        "r11_accessed": False,
    }, OUT / "progress.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    del args.device
    if args.output_root is not None:
        configure_output_root(args.output_root.resolve())

    config = load_json(CONFIG_PATH)
    pilot_integrity_path = PILOT / "pilot_integrity.json"
    pilot_analysis_path = PILOT / "analysis/analysis_integrity.json"
    pilot_integrity = load_json(pilot_integrity_path)
    pilot_analysis = load_json(pilot_analysis_path)
    if not all([
        pilot_integrity.get("status") == "PASS",
        pilot_integrity.get("r10_accessed") is False,
        pilot_integrity.get("r11_accessed") is False,
        pilot_analysis.get("status") == "PASS",
        pilot_analysis.get("r10_accessed") is False,
        pilot_analysis.get("r11_accessed") is False,
    ]):
        raise RuntimeError("formal R09 pilot and analysis must pass first")
    config_sha256 = digest(CONFIG_PATH)
    pilot_sha256 = digest(pilot_integrity_path)
    actions = pd.read_parquet(PILOT / "forced_action_failure_table.parquet")
    if set(actions.split) != {"R09_PILOT"}:
        raise RuntimeError("continuation source is not R09-pilot-only")
    tasks = select_tasks(actions)
    replays = replay_index()
    if any(str(task["state_id"]) not in replays for task in tasks):
        raise RuntimeError("missing selected R09 pilot state replay")
    started = time.perf_counter()
    if args.summarize_only:
        summarize(
            tasks,
            config,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        )
        write_progress(
            tasks,
            started=started,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        )
        return

    pending = [
        task for task in tasks
        if valid_result(
            task,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        ) is None
    ]
    if args.limit_states is not None:
        pending = pending[:args.limit_states]
    print(
        f"PHASE6I_MR_CONTINUATION_START pending={len(pending)} total={len(tasks)}",
        flush=True,
    )
    alns_config = read_alns_config()
    diagnostic = config["continuation_diagnostic"]
    if alns_config.candidate_trials != int(
        diagnostic["repair_decoder_trials_per_target"]
    ):
        raise RuntimeError("continuation repair trials differ from frozen ALNS")
    instance_root = ROOT / config["instance_suite"]["root"]
    action_by_state = {
        state_id: group.sort_values("candidate_role")
        for state_id, group in actions.groupby("state_id")
    }
    cached_instance_id = None
    instance = None
    for index, task in enumerate(pending, 1):
        state_id = str(task["state_id"])
        if cached_instance_id != task["instance_id"]:
            instance = load_phase6i_instance(
                instance_root / str(task["instance_relative_path"])
            )
            cached_instance_id = task["instance_id"]
        replay = load_json(replays[state_id])
        current = decode_candidate(instance, candidate_from_dict(replay["current_candidate"]))
        if not math.isclose(
            current.makespan,
            float(replay["current_makespan"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"current replay mismatch: {state_id}")
        state_rows = []
        for action in action_by_state[state_id].itertuples(index=False):
            decoded = decode_forced_candidate(
                instance,
                current,
                arm_from_row(action),
                state_id=state_id,
                repair_seed_namespace=int(config["rng_namespaces"]["forced_repair"]),
                candidate_trials=int(diagnostic["repair_decoder_trials_per_target"]),
            )
            replay_match = math.isclose(
                decoded.candidate.makespan,
                float(action.decoded_candidate_makespan),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            if not replay_match:
                raise RuntimeError(
                    f"forced candidate replay mismatch: {state_id} {action.candidate_role}"
                )
            for continuation_seed in diagnostic["common_random_seeds"]:
                continued = continue_frozen_alns_from_candidate(
                    instance,
                    decoded.candidate,
                    state_id=state_id,
                    continuation_seed=int(continuation_seed),
                    seed_namespace=int(config["rng_namespaces"]["continuation"]),
                    iterations=int(diagnostic["fixed_horizon_alns_iterations"]),
                    config=alns_config,
                )
                feasibility = check_schedule(instance, continued.candidate.schedule)
                state_rows.append({
                    **task,
                    "split": "R09_PILOT_ONLY",
                    "candidate_role": action.candidate_role,
                    "target_set_id": action.target_set_id,
                    "decoded_immediate_utility": float(action.decoded_immediate_utility),
                    "candidate_start_makespan": continued.start_makespan,
                    "best_makespan_after_continuation": continued.best_makespan,
                    "continuation_value": continued.continuation_value,
                    "continuation_seed": continued.continuation_seed,
                    "derived_continuation_seed": continued.derived_seed,
                    "continuation_iterations": continued.iterations,
                    "continuation_decoder_evaluations": continued.decoder_evaluations,
                    "continuation_accepted_moves": continued.accepted_moves,
                    "continuation_improving_moves": continued.improving_moves,
                    "continuation_operator_selections": json.dumps(
                        continued.operator_selections, sort_keys=True
                    ),
                    "continuation_runtime_ms": continued.runtime_ms,
                    "start_replay_match": replay_match,
                    "start_feasible": bool(decoded.candidate.feasible),
                    "continuation_feasible": bool(
                        continued.candidate.feasible and feasibility["feasible"]
                    ),
                })
        state_frame = pd.DataFrame(state_rows)
        if len(state_frame) != 8:
            raise RuntimeError(f"unexpected continuation rows for {state_id}")
        atomic_parquet(state_frame, table_path(task))
        atomic_json({
            "schema": "phase6i-mr-continuation-state-result-v1.2",
            "status": "COMPLETE",
            "state_id": state_id,
            "config_sha256": config_sha256,
            "pilot_integrity_sha256": pilot_sha256,
            "rows": len(state_frame),
            "table_sha256": digest(table_path(task)),
            "r10_accessed": False,
            "r11_accessed": False,
        }, result_path(task))
        write_progress(
            tasks,
            started=started,
            config_sha256=config_sha256,
            pilot_sha256=pilot_sha256,
        )
        print(
            f"PHASE6I_MR_CONTINUATION_STATE {index}/{len(pending)} "
            f"state_id={state_id}",
            flush=True,
        )

    summarize(
        tasks,
        config,
        config_sha256=config_sha256,
        pilot_sha256=pilot_sha256,
    )
    write_progress(
        tasks,
        started=started,
        config_sha256=config_sha256,
        pilot_sha256=pilot_sha256,
    )
    print("PHASE6I_MR_CONTINUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
