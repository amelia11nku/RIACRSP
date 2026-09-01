#!/usr/bin/env python3
"""Stratified one-seed feature-capture audit for Phase 6G live-state drift."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6g import Phase6GLiveObserver  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


OUT = ROOT / "outputs/phase6g/drift_audit"
LOG_ROOT = ROOT / "outputs/phase6g/live_logs/drift_audit"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-instances", type=int)
    args = parser.parse_args()
    config = load_json(ROOT / "configs/phase6g_live_solver.json")
    freeze = load_json(ROOT / "outputs/phase6g/frequency_study/selected_rate_freeze.json")
    if freeze.get("status") != "FROZEN_BEFORE_DEV_HOLDOUT":
        raise RuntimeError("selected rate is not frozen")
    selected_rate = int(freeze["selected_rate"])
    seed = int(config["seeds"]["DEV_HOLDOUT"][0])
    split = pd.read_csv(ROOT / "outputs/phase6g/environment/dev_split.csv")
    split = split[split.split == "DEV_HOLDOUT"]
    if args.limit_instances is not None:
        split = split.iloc[:args.limit_instances]
    pending = [
        row for row in split.itertuples(index=False)
        if not (LOG_ROOT / row.instance_id / f"seed_{seed}.parquet").exists()
    ]
    print(
        f"PHASE6G_DRIFT_AUDIT_START pending={len(pending)} total={len(split)} "
        f"seed={seed} rate=R{selected_rate}", flush=True,
    )
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    alns_config = ALNSConfig(**{
        key: value for key, value in raw.items() if key in ALNSConfig.__dataclass_fields__
    })
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["frozen_phase6f"]["experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
    )
    model_load_seconds = time.perf_counter() - load_started
    for index, row in enumerate(pending, 1):
        instance = load_instance(ROOT / "instances/controlled/RCIAS-CB1" / row.relative_path)
        budget = config["search"]["wall_clock_seconds_per_operation"] * instance.num_operations
        observer = Phase6GLiveObserver({
            "instance_id": row.instance_id,
            "seed": seed,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "intervention_rate": selected_rate,
            "split": "DEV_HOLDOUT_DRIFT_AUDIT",
            "method": "CSGNI",
        })
        result = solve_csgni(
            instance, budget, seed, policy,
            alns_config=alns_config,
            csgni_config=CSGNIConfig(
                intervention_rate=selected_rate,
                proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                ni_repair_seed_namespace=config["rng_namespaces"]["ni_repair"],
                acceptance_seed_namespace=config["rng_namespaces"]["acceptance"],
                diagnostics_seed_namespace=config["rng_namespaces"]["diagnostics"],
            ),
            observer=observer,
        )
        feasibility = check_schedule(instance, result.best.schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"instance": row.instance_id, "violations": feasibility["violations"]})
        frame = pd.DataFrame(observer.rows)
        feature_columns = [
            "mean_slack_ratio", "mean_w_delay_ratio", "mean_f_delay_ratio",
            "mean_island_relative_load", "mean_local_reconfiguration_ratio",
            "search_progress",
        ]
        if frame[feature_columns].isna().any().any():
            raise RuntimeError(f"missing drift features for {row.instance_id}")
        atomic_parquet(frame, LOG_ROOT / row.instance_id / f"seed_{seed}.parquet")
        atomic_json({
            "schema": "phase6g-drift-audit-run-v1",
            "status": "COMPLETE",
            "instance_id": row.instance_id,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "seed": seed,
            "selected_rate": selected_rate,
            "time_limit_seconds": budget,
            "runtime": result.runtime,
            "best_makespan": result.best.makespan,
            "feasible": True,
            "iteration_rows": len(frame),
            "model_load_seconds": model_load_seconds,
        }, OUT / "runs" / row.instance_id / f"seed_{seed}.json")
        atomic_json({
            "schema": "phase6g-drift-audit-progress-v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_instances": len(list((OUT / "runs").rglob("*.json"))),
            "total_instances": 9,
            "status": "RUNNING",
        }, OUT / "progress.json")
        print(
            f"[{index}/{len(pending)}] {row.instance_id} rows={len(frame)} "
            f"makespan={result.best.makespan:g} runtime={result.runtime:.2f}s",
            flush=True,
        )
    completed = len(list((OUT / "runs").rglob("*.json")))
    atomic_json({
        "schema": "phase6g-drift-audit-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_instances": completed,
        "total_instances": 9,
        "status": "COMPLETE" if completed == 9 else "RUNNING",
    }, OUT / "progress.json")
    print("PHASE6G_DRIFT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
