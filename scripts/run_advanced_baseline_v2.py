#!/usr/bin/env python3
"""Resumable Core runner for the versioned LG_HGA v2 comparator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.search.lghga import LGHGAConfig
from rcias_clgri.search.lghga_learning import load_dtr_bundle
from rcias_clgri.search.lghga_v2 import METHOD, solve_lghga_v2
from scripts.run_advanced_baselines import (
    FORMAL_MANIFEST,
    _atomic_json,
    _display_path,
    _git_commit,
    _load_formal_rows,
    _read_dataclass_config,
    _sha256,
    _utc_now,
)


CONFIG_PATH = ROOT / "configs/baselines/lghga_v2_riacrsp.json"
KB_CONFIG_PATH = ROOT / "configs/baselines/lghga_v2_kb.json"


def _verify_implementation() -> str:
    kb = json.loads(KB_CONFIG_PATH.read_text())
    path = ROOT / str(kb["implementation_manifest"])
    manifest = json.loads(path.read_text())
    for record in manifest["files"]:
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"LG_HGA v2 implementation mismatch: {record['path']}")
    return _sha256(path)


def _validate_models(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    knowledge_path = model_dir.parent / "knowledge_manifest.json"
    if not knowledge_path.exists():
        raise FileNotFoundError("formal LG_HGA v2 requires its frozen knowledge manifest")
    knowledge = json.loads(knowledge_path.read_text())
    if knowledge.get("status") != "FROZEN_V2_NO_FORMAL_TEST_LEAKAGE":
        raise ValueError("LG_HGA v2 knowledge freeze has not passed")
    root_manifest_path = model_dir / "model_manifest.json"
    if _sha256(root_manifest_path) != knowledge.get("model_manifest_sha256"):
        raise ValueError("LG_HGA v2 root model manifest hash mismatch")
    root_manifest = json.loads(root_manifest_path.read_text())
    if root_manifest.get("knowledge_manifest_hash") != knowledge.get("knowledge_manifest_hash"):
        raise ValueError("LG_HGA v2 knowledge/model hashes disagree")
    return knowledge, root_manifest


def _output_path(output_root: Path, instance_id: str, seed: int, budget_scale: float) -> Path:
    budget = "formal" if budget_scale == 1.0 else f"pilot_{budget_scale:g}x"
    return output_root / budget / "lg_hga_riacrsp_v2_n4m" / instance_id / f"seed_{seed}.json"


def _run_task(
    task: tuple[dict[str, str], int, float, str, str, str]
) -> tuple[str, float, int]:
    row, seed, budget_scale, model_root_string, output_root_string, implementation_hash = task
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    instance_path = ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"]
    instance_hash = _sha256(instance_path)
    if instance_hash != row["expected_sha256"]:
        raise ValueError(f"formal v2 instance hash mismatch: {row['instance_id']}")
    instance = load_instance(instance_path)
    regime = f"{row['scale']}_{row['CF_level']}"
    model_root = Path(model_root_string)
    root_manifest = json.loads((model_root / "model_manifest.json").read_text())
    if regime not in root_manifest["regimes"]:
        raise KeyError(f"missing LG_HGA v2 model regime: {regime}")
    model_dir = model_root / regime
    if _sha256(model_dir / "model_manifest.json") != root_manifest["regimes"][regime]["model_manifest_sha256"]:
        raise ValueError(f"LG_HGA v2 regime manifest mismatch: {regime}")
    models = load_dtr_bundle(model_dir)
    config = _read_dataclass_config(CONFIG_PATH, LGHGAConfig)
    time_limit = 2.0 * instance.num_operations * budget_scale
    result = solve_lghga_v2(instance, time_limit, seed, models, config)
    audit = check_schedule(instance, result.best.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"independent v2 final check failed: {instance.instance_id}")
    output = _output_path(
        Path(output_root_string), instance.instance_id, seed, budget_scale
    )
    payload = {
        "schema": "advanced-baseline-run-v2",
        "method": result.method,
        "suite": "RCIAS-CB1 Core",
        "scale": row["scale"],
        "CF_level": row["CF_level"],
        "instance_id": instance.instance_id,
        "instance_path": str(instance_path.relative_to(ROOT)),
        "instance_sha256": instance_hash,
        "number_of_operations": instance.num_operations,
        "seed": seed,
        "started_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG_PATH),
        "effective_config": asdict(config),
        "formal_manifest_sha256": _sha256(FORMAL_MANIFEST),
        "implementation_manifest_sha256": implementation_hash,
        "time_limit_seconds": time_limit,
        "budget_scale": budget_scale,
        "best_makespan": result.best.makespan,
        "final_incumbent_makespan": result.best.makespan,
        "best_found_time": result.best_found_time,
        "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "generations_if_applicable": result.generations_if_applicable,
        "feasible": result.best.feasible,
        "independent_feasibility_audit": audit,
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "diagnostics": result.diagnostics,
        "lghga_models": {
            "regime": regime,
            "model_dir": _display_path(model_dir),
            "root_model_manifest_sha256": _sha256(model_root / "model_manifest.json"),
            "regime_model_manifest_sha256": _sha256(model_dir / "model_manifest.json"),
            "model_hashes": dict(models.model_hashes),
            "knowledge_manifest_hash": models.knowledge_manifest_hash,
        },
        "compute": {"cpu_threads": 1, "gpu_usage": False, "process_count": 1},
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
    }
    _atomic_json(output, payload)
    return str(output.relative_to(ROOT)), result.best.makespan, result.decoder_evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "outputs/baselines/lghga_kb_v2/models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/baselines/comparison_advanced_v2",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.budget_scale <= 0:
        raise ValueError("workers and budget scale must be positive")
    if args.limit_instances is not None and args.limit_instances < 1:
        raise ValueError("limit-instances must be positive")
    implementation_hash = _verify_implementation()
    formal, rows = _load_formal_rows()
    rows = rows[: args.limit_instances]
    seeds = args.seeds or json.loads((ROOT / str(formal["seed_manifest"])).read_text())["seeds"]
    if args.budget_scale == 1.0:
        _validate_models(args.model_dir)
    tasks = []
    for row in rows:
        for seed in seeds:
            output = _output_path(args.output_root, row["instance_id"], seed, args.budget_scale)
            if not output.exists():
                tasks.append((
                    row,
                    int(seed),
                    args.budget_scale,
                    str(args.model_dir.resolve()),
                    str(args.output_root.resolve()),
                    implementation_hash,
                ))
    print(
        f"LGHGA_V2_CORE_START pending={len(tasks)} workers={args.workers} "
        f"budget_scale={args.budget_scale:g}",
        flush=True,
    )
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            path, makespan, evaluations = _run_task(task)
            print(f"[{index}/{len(tasks)}] {path} makespan={makespan:g} evals={evaluations}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                path, makespan, evaluations = future.result()
                print(f"[{index}/{len(tasks)}] {path} makespan={makespan:g} evals={evaluations}", flush=True)
    print(f"LGHGA_V2_CORE_COMPLETE completed={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
