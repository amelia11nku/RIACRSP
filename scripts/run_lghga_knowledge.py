#!/usr/bin/env python3
"""Generate, audit, and freeze the disjoint LG_HGA knowledge base."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.lghga import LGHGAConfig, generate_knowledge_run
from rcias_clgri.search.lghga_learning import save_dtr_bundle, train_dtr_bundle
from rcias_clgri.search.lghga_neighborhoods import NEIGHBORHOODS


KB_CONFIG = ROOT / "configs/baselines/lghga_kb.json"
IMPLEMENTATION_MANIFEST = ROOT / "configs/baselines/advanced_implementation_manifest.json"
TRAIN_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
CORE_ROOT = ROOT / "instances/controlled/RCIAS-CB1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_implementation_manifest() -> str:
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"implementation freeze mismatch: {record['path']}")
    return _sha256(IMPLEMENTATION_MANIFEST)


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return process.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_configs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    kb = json.loads(KB_CONFIG.read_text())
    algorithm_path = ROOT / str(kb["algorithm_config"])
    training_path = ROOT / str(kb["training_manifest"])
    return kb, json.loads(algorithm_path.read_text()), json.loads(training_path.read_text())


def _algorithm_config(raw: dict[str, object], max_generations: int | None = None) -> LGHGAConfig:
    values = {
        key: value for key, value in raw.items()
        if key in LGHGAConfig.__dataclass_fields__
    }
    if max_generations is not None:
        values["max_generations"] = max_generations
    return LGHGAConfig(**values)


def _formal_run_path(output_root: Path, instance_id: str, seed: int) -> Path:
    return output_root / "runs" / instance_id / f"seed_{seed}.json"


def _knowledge_task(
    task: tuple[dict[str, object], int, dict[str, object], str, str, str, str]
) -> str:
    row, seed, algorithm_raw, output_string, kb_hash, algorithm_hash, implementation_hash = task
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    path = TRAIN_ROOT / str(row["relative_path"])
    actual_hash = _sha256(path)
    if actual_hash != row["sha256"]:
        raise ValueError(f"knowledge instance hash mismatch: {row['instance_id']}")
    instance = load_instance(path)
    started_at = _utc_now()
    config = _algorithm_config(algorithm_raw)
    result = generate_knowledge_run(instance, seed, config, time_limit=math.inf)
    output = Path(output_string)
    payload = {
        "schema": "lghga-kb-run-v1",
        "method": "LG_HGA-KB",
        "instance_id": instance.instance_id,
        "instance_path": str(path.relative_to(ROOT)),
        "instance_sha256": actual_hash,
        "scale": row["scale"],
        "CF_level": row["CF_level"],
        "seed": seed,
        "started_at_utc": started_at,
        "git_commit": _git_commit(),
        "kb_config_sha256": kb_hash,
        "algorithm_config_sha256": algorithm_hash,
        "implementation_manifest_sha256": implementation_hash,
        "effective_algorithm_config": asdict(config),
        "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations,
        "generations": result.generations,
        "best_makespan": result.best.makespan,
        "feasible": result.best.feasible,
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
        "knowledge_rows": list(result.rows),
        "compute": {"cpu_threads": 1, "gpu_usage": False, "process_count": 1},
    }
    _atomic_json(output, payload)
    return str(output.relative_to(ROOT))


def generate(args: argparse.Namespace) -> None:
    kb, algorithm_raw, training = _load_configs()
    rows = list(training["instances"])
    seeds = list(kb["run_seeds"])
    if args.limit_instances is not None:
        rows = rows[: args.limit_instances]
    if args.runs is not None:
        seeds = seeds[: args.runs]
    if args.smoke_generations is not None:
        algorithm_raw = dict(algorithm_raw)
        algorithm_raw["max_generations"] = args.smoke_generations
        output_root = ROOT / str(kb["output_root"]) / "smoke" / f"g{args.smoke_generations}"
    else:
        output_root = ROOT / str(kb["output_root"])
    kb_hash = _sha256(KB_CONFIG)
    implementation_hash = _verify_implementation_manifest()
    algorithm_path = ROOT / str(kb["algorithm_config"])
    algorithm_hash = _sha256(algorithm_path)
    tasks = []
    for row in rows:
        for seed in seeds:
            output = (
                output_root / str(row["instance_id"]) / f"seed_{seed}.json"
                if args.smoke_generations is not None
                else _formal_run_path(output_root, str(row["instance_id"]), seed)
            )
            if not output.exists():
                tasks.append((
                    row,
                    seed,
                    algorithm_raw,
                    str(output),
                    kb_hash,
                    algorithm_hash,
                    implementation_hash,
                ))
    print(f"LGHGA_KB_START pending={len(tasks)} workers={args.workers}", flush=True)
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {_knowledge_task(task)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_knowledge_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                print(f"[{index}/{len(tasks)}] {future.result()}", flush=True)
    print(f"LGHGA_KB_COMPLETE completed={len(tasks)}", flush=True)


def _core_hashes() -> tuple[set[str], set[str]]:
    import csv

    with (CORE_ROOT / "manifests/core_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = {row["instance_id"] for row in rows}
    hashes = {_sha256(CORE_ROOT / row["relative_path"]) for row in rows}
    return identifiers, hashes


def train(_: argparse.Namespace) -> None:
    kb, _, training = _load_configs()
    implementation_hash = _verify_implementation_manifest()
    output_root = ROOT / str(kb["output_root"])
    rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    expected_paths: list[Path] = []
    for instance_record in training["instances"]:
        for seed in kb["run_seeds"]:
            expected_paths.append(
                _formal_run_path(output_root, str(instance_record["instance_id"]), int(seed))
            )
    missing = [str(path.relative_to(ROOT)) for path in expected_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"cannot train frozen DTR bundle: {len(missing)} of {len(expected_paths)} runs missing"
        )
    for path in expected_paths:
        payload = json.loads(path.read_text())
        if not payload["feasible"]:
            raise RuntimeError(f"infeasible knowledge run: {path}")
        rows.extend(payload["knowledge_rows"])
        artifacts.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "runtime": payload["runtime"],
            "decoder_evaluations": payload["decoder_evaluations"],
        })

    training_ids = {str(row["instance_id"]) for row in training["instances"]}
    training_hashes = {str(row["sha256"]) for row in training["instances"]}
    core_ids, core_hashes = _core_hashes()
    overlap_ids = sorted(training_ids & core_ids)
    overlap_hashes = sorted(training_hashes & core_hashes)
    if overlap_ids or overlap_hashes:
        raise RuntimeError("knowledge/formal instance leakage detected")

    data_hash = _canonical_hash(rows)
    input_manifest = {
        "schema": "lghga-knowledge-input-freeze-v1",
        "kb_config_sha256": _sha256(KB_CONFIG),
        "algorithm_config_sha256": _sha256(ROOT / str(kb["algorithm_config"])),
        "training_manifest_sha256": _sha256(ROOT / str(kb["training_manifest"])),
        "implementation_manifest_sha256": implementation_hash,
        "knowledge_rows_sha256": data_hash,
        "run_artifacts": artifacts,
        "run_count": len(artifacts),
        "row_count": len(rows),
        "training_instance_ids": sorted(training_ids),
        "formal_instance_manifest": training["formal_evaluation_manifest"],
        "formal_instance_count": len(core_ids),
        "instance_id_overlap": overlap_ids,
        "instance_hash_overlap": overlap_hashes,
    }
    knowledge_manifest_hash = _canonical_hash(input_manifest)
    bundle = train_dtr_bundle(
        rows,
        random_state=int(kb["dtr_random_state"]),
        knowledge_manifest_hash=knowledge_manifest_hash,
    )
    rmse = {}
    for neighborhood in NEIGHBORHOODS:
        selected = [row for row in rows if row["neighborhood_id"] == neighborhood]
        x = np.asarray(
            [[float(row["normalized_generation_index"])] for row in selected], dtype=float
        )
        y = np.asarray([float(row["R_pct"]) for row in selected], dtype=float)
        predicted = bundle.models[neighborhood].predict(x)
        rmse[neighborhood] = float(np.sqrt(np.mean((predicted - y) ** 2)))

    model_dir = output_root / "models"
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen model directory: {model_dir}")
    model_manifest = save_dtr_bundle(bundle, model_dir)
    rows_path = output_root / "knowledge_rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = rows_path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(rows_path)
    manifest = {
        **input_manifest,
        "knowledge_manifest_hash": knowledge_manifest_hash,
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "model_manifest": str((model_dir / "model_manifest.json").relative_to(ROOT)),
        "model_manifest_sha256": _sha256(model_dir / "model_manifest.json"),
        "model_hashes": model_manifest["model_hashes"],
        "training_rmse_pct": rmse,
        "offline_runtime_seconds": sum(float(item["runtime"]) for item in artifacts),
        "offline_decoder_evaluations": sum(
            int(item["decoder_evaluations"]) for item in artifacts
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
        "status": "FROZEN_NO_FORMAL_TEST_LEAKAGE",
    }
    _atomic_json(output_root / "knowledge_manifest.json", manifest)
    print(
        f"LGHGA_DTR_FROZEN rows={len(rows)} models={len(model_manifest['model_hashes'])} "
        f"manifest={knowledge_manifest_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--workers", type=int, default=1)
    generate_parser.add_argument("--limit-instances", type=int)
    generate_parser.add_argument("--runs", type=int)
    generate_parser.add_argument("--smoke-generations", type=int)
    generate_parser.set_defaults(function=generate)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(function=train)
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be positive")
    if getattr(args, "runs", None) is not None and args.runs < 1:
        raise ValueError("runs must be positive")
    if getattr(args, "smoke_generations", None) is not None and args.smoke_generations < 1:
        raise ValueError("smoke generations must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
