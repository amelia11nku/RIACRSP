#!/usr/bin/env python3
"""Resumable matched runner for DABC-RIACRSP and LG_HGA-RIACRSP."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.search.dabc import DABCConfig, solve_dabc
from rcias_clgri.search.lghga import LGHGAConfig, solve_lghga
from rcias_clgri.search.lghga_learning import load_dtr_bundle


METHODS = ("DABC-RIACRSP", "LG_HGA-RIACRSP")
FORMAL_MANIFEST = ROOT / "configs/baselines/advanced_formal_manifest.json"
IMPLEMENTATION_MANIFEST = ROOT / "configs/baselines/advanced_implementation_manifest.json"
CONFIG_PATHS = {
    "DABC-RIACRSP": ROOT / "configs/baselines/dabc_riacrsp.json",
    "LG_HGA-RIACRSP": ROOT / "configs/baselines/lghga_riacrsp.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return process.stdout.strip()


def _verify_implementation_manifest() -> str:
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"implementation freeze mismatch: {record['path']}")
    return _sha256(IMPLEMENTATION_MANIFEST)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read_dataclass_config(path: Path, cls):
    raw = json.loads(path.read_text())
    return cls(**{
        key: value for key, value in raw.items()
        if key in cls.__dataclass_fields__
    })


def _method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def _regime(budget_scale: float) -> str:
    return "formal" if budget_scale == 1.0 else f"pilot_{budget_scale:g}x"


def _output_path(
    output_root: Path,
    method: str,
    instance_id: str,
    seed: int,
    budget_scale: float,
) -> Path:
    return (
        output_root / _regime(budget_scale) / _method_slug(method)
        / instance_id / f"seed_{seed}.json"
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_task(task: tuple[str, dict[str, str], int, float, str, str]) -> tuple[str, float, int]:
    method, row, seed, budget_scale, model_dir_string, output_root_string = task
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    instance_path = ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"]
    instance_hash = _sha256(instance_path)
    if instance_hash != row["expected_sha256"]:
        raise ValueError(f"formal instance hash mismatch: {row['instance_id']}")
    instance = load_instance(instance_path)
    time_limit = 2.0 * instance.num_operations * budget_scale
    started_at = _utc_now()
    config_path = CONFIG_PATHS[method]
    if method == "DABC-RIACRSP":
        config = _read_dataclass_config(config_path, DABCConfig)
        result = solve_dabc(instance, time_limit, seed, config)
        model_metadata = None
    else:
        model_dir = Path(model_dir_string)
        models = load_dtr_bundle(model_dir)
        config = _read_dataclass_config(config_path, LGHGAConfig)
        result = solve_lghga(instance, time_limit, seed, models, config)
        model_metadata = {
            "model_dir": _display_path(model_dir),
            "model_manifest_sha256": _sha256(model_dir / "model_manifest.json"),
            "model_hashes": dict(models.model_hashes),
            "knowledge_manifest_hash": models.knowledge_manifest_hash,
        }
    independent_audit = check_schedule(instance, result.best.schedule)
    if not independent_audit["feasible"]:
        raise RuntimeError(f"independent final check failed: {instance.instance_id}")
    output = _output_path(
        Path(output_root_string), method, instance.instance_id, seed, budget_scale
    )
    payload = {
        "schema": "advanced-baseline-run-v1",
        "method": result.method,
        "suite": "RCIAS-CB1 Core",
        "scale": row["scale"],
        "CF_level": row["CF_level"],
        "instance_id": instance.instance_id,
        "instance_path": str(instance_path.relative_to(ROOT)),
        "instance_sha256": instance_hash,
        "number_of_operations": instance.num_operations,
        "seed": seed,
        "started_at_utc": started_at,
        "git_commit": _git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "effective_config": asdict(config),
        "formal_manifest_sha256": _sha256(FORMAL_MANIFEST),
        "implementation_manifest_sha256": _sha256(IMPLEMENTATION_MANIFEST),
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
        "independent_feasibility_audit": independent_audit,
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "diagnostics": result.diagnostics,
        "lghga_models": model_metadata,
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


def _load_formal_rows() -> tuple[dict[str, object], list[dict[str, str]]]:
    formal = json.loads(FORMAL_MANIFEST.read_text())
    instance_manifest = ROOT / str(formal["instance_manifest"])
    instance_checksums = ROOT / str(formal["instance_checksums"])
    seed_manifest = ROOT / str(formal["seed_manifest"])
    if _sha256(instance_manifest) != formal["instance_manifest_sha256"]:
        raise ValueError("formal instance manifest hash mismatch")
    if _sha256(seed_manifest) != formal["seed_manifest_sha256"]:
        raise ValueError("formal seed manifest hash mismatch")
    if _sha256(instance_checksums) != formal["instance_checksums_sha256"]:
        raise ValueError("formal instance checksum-file hash mismatch")
    expected_hashes = {}
    for line in instance_checksums.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        expected_hashes[relative_path] = digest
    with instance_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != formal["instance_count"]:
        raise ValueError("formal instance count mismatch")
    for row in rows:
        try:
            row["expected_sha256"] = expected_hashes[row["relative_path"]]
        except KeyError as error:
            raise ValueError(
                f"formal instance missing from checksum freeze: {row['relative_path']}"
            ) from error
    return formal, rows


def _validate_formal_models(model_dir: Path) -> None:
    """Refuse a formal run unless the complete leakage-audited freeze exists."""

    knowledge_path = model_dir.parent / "knowledge_manifest.json"
    if not knowledge_path.exists():
        raise FileNotFoundError(
            "formal LG_HGA requires the complete knowledge_manifest.json freeze"
        )
    knowledge = json.loads(knowledge_path.read_text())
    if knowledge.get("status") != "FROZEN_NO_FORMAL_TEST_LEAKAGE":
        raise ValueError("LG_HGA knowledge manifest has not passed its leakage freeze")
    model_manifest = model_dir / "model_manifest.json"
    if _sha256(model_manifest) != knowledge.get("model_manifest_sha256"):
        raise ValueError("formal LG_HGA model manifest does not match the knowledge freeze")
    persisted = json.loads(model_manifest.read_text())
    if persisted.get("knowledge_manifest_hash") != knowledge.get("knowledge_manifest_hash"):
        raise ValueError("formal LG_HGA knowledge/model hashes disagree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "outputs/baselines/lghga_kb/models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/baselines/comparison_advanced",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.budget_scale <= 0:
        raise ValueError("workers and budget scale must be positive")
    if args.limit_instances is not None and args.limit_instances < 1:
        raise ValueError("limit-instances must be positive")
    formal, rows = _load_formal_rows()
    _verify_implementation_manifest()
    rows = rows[: args.limit_instances]
    default_seeds = json.loads((ROOT / str(formal["seed_manifest"])).read_text())["seeds"]
    seeds = args.seeds or default_seeds
    if "LG_HGA-RIACRSP" in args.methods and not (
        args.model_dir / "model_manifest.json"
    ).exists():
        raise FileNotFoundError(
            "LG_HGA requires a frozen DTR bundle; run run_lghga_knowledge.py first"
        )
    if "LG_HGA-RIACRSP" in args.methods and args.budget_scale == 1.0:
        _validate_formal_models(args.model_dir)
    output_root = args.output_root.resolve()
    tasks = []
    for method in args.methods:
        for row in rows:
            for seed in seeds:
                output = _output_path(
                    output_root, method, row["instance_id"], seed, args.budget_scale
                )
                if not output.exists():
                    tasks.append((
                        method,
                        row,
                        seed,
                        args.budget_scale,
                        str(args.model_dir.resolve()),
                        str(output_root),
                    ))
    print(
        f"ADVANCED_BASELINES_START pending={len(tasks)} workers={args.workers} "
        f"budget_scale={args.budget_scale:g}",
        flush=True,
    )
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            path, makespan, evaluations = _run_task(task)
            print(
                f"[{index}/{len(tasks)}] {path} makespan={makespan:g} evals={evaluations}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                path, makespan, evaluations = future.result()
                print(
                    f"[{index}/{len(tasks)}] {path} makespan={makespan:g} evals={evaluations}",
                    flush=True,
                )
    print(f"ADVANCED_BASELINES_COMPLETE completed={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
