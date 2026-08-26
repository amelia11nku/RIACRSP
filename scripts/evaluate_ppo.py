#!/usr/bin/env python3
"""Evaluate frozen BC/PPO checkpoints on synthetic and canonical held-out sets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.learning.evaluator import evaluate_baselines, evaluate_policy, load_checkpoint
from rcias_clgri.learning.experiment import (
    load_phase3_config,
    make_factory,
    resolve_device,
)

_WORKER_POLICIES = None


def _initialize_canonical_worker(checkpoint_specs):
    global _WORKER_POLICIES
    import torch
    torch.set_num_threads(1)
    _WORKER_POLICIES = []
    for method, seed, checkpoint in checkpoint_specs:
        model, tensorizer, _ = load_checkpoint(checkpoint, device="cpu")
        _WORKER_POLICIES.append((method, seed, model, tensorizer))


def _canonical_worker(path_string):
    instance = load_instance(path_string)
    return _evaluate_instance(
        instance, "canonical_public_test", _family(instance.instance_id),
        _WORKER_POLICIES, "cpu",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(instance_id: str) -> str:
    if instance_id.startswith("BR_"):
        return "Brandimarte"
    if instance_id.startswith("HU_E_"):
        return "Hurink E"
    if instance_id.startswith("HU_R_"):
        return "Hurink R"
    if instance_id.startswith("HU_V_"):
        return "Hurink V"
    return "Synthetic"


def _evaluate_instance(instance, split, group, policies, device):
    rows = []
    for result in evaluate_baselines(instance):
        rows.append({
            **result.to_dict(), "data_split": split, "group": group,
            "training_seed": "NA",
        })
    for method, seed, model, tensorizer in policies:
        result = evaluate_policy(
            model, tensorizer, instance, device=device, method=method
        )
        rows.append({
            **result.to_dict(), "data_split": split, "group": group,
            "training_seed": seed,
        })
    best = min(float(row["makespan"]) for row in rows)
    for row in rows:
        row["best_compared_makespan"] = best
        row["gap_percent"] = 100.0 * (float(row["makespan"]) - best) / max(best, 1e-12)
    return rows


def _aggregate(rows):
    def as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    groups = {}
    for row in rows:
        keys = (
            (row["data_split"], row["group"], row["method"]),
            (row["data_split"], "Overall", row["method"]),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    records = []
    for (split, group, method), values in sorted(groups.items()):
        makespans = [float(row["makespan"]) for row in values]
        gaps = [float(row["gap_percent"]) for row in values]
        records.append({
            "data_split": split,
            "group": group,
            "method": method,
            "runs": len(values),
            "mean_makespan": mean(makespans),
            "std_makespan": pstdev(makespans) if len(makespans) > 1 else 0.0,
            "best_makespan": min(makespans),
            "mean_gap_percent": mean(gaps),
            "std_gap_percent": pstdev(gaps) if len(gaps) > 1 else 0.0,
            "mean_runtime_seconds": mean(float(row["runtime_seconds"]) for row in values),
            "mean_inference_seconds": mean(float(row["inference_seconds"]) for row in values),
            "feasibility_rate": mean(float(as_bool(row["feasible"])) for row in values),
        })
    return records


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_specs(config, out_dir):
    bc_directory = (
        "bc_large"
        if "demonstration_instances" in config["bc_warm_start"]
        else "bc_pretrain"
    )
    return [
        ("BC_GREEDY", "BC", out_dir / bc_directory / "best.pt"),
        *[
            ("PPO_GREEDY", seed, out_dir / f"ppo_seed_{index}" / "best.pt")
            for index, seed in enumerate(
                config["training_seed_policy"]["independent_training_seeds"], start=1
            )
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--canonical-workers", type=int, default=6)
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    device = resolve_device(args.device)
    checkpoint_specs = _checkpoint_specs(config, args.out_dir)
    missing = [str(path) for _, _, path in checkpoint_specs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen evaluation checkpoints are missing: {missing}")
    policies = []
    frozen = []
    for method, seed, path in checkpoint_specs:
        model, tensorizer, metadata = load_checkpoint(
            path, device="cpu" if args.canonical_only else device
        )
        policies.append((method, seed, model, tensorizer))
        frozen.append({
            "method": method, "training_seed": seed,
            "checkpoint": path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": _sha256(path), "metadata": metadata,
        })
    freeze_record = {
        "frozen_before_canonical_evaluation": True,
        "selection_data": "fixed synthetic validation only",
        "canonical_training_or_tuning_instances": 0,
        "checkpoints": frozen,
        "evaluation_config": config["evaluation"],
    }
    write_json(freeze_record, args.out_dir / "frozen_evaluation_config.json")
    print(
        "This experiment compares H1/H2/H3, BC greedy, and three PPO greedy seeds on "
        "held-out synthetic data and, after checkpoint freeze, the canonical public suite."
    )
    synthetic_rows = []
    if not args.canonical_only:
        factory = make_factory(ROOT, config)
        for level, seeds in config["validation_seeds"].items():
            for seed in seeds:
                instance = factory.sample(int(seed), level)
                synthetic_rows.extend(_evaluate_instance(
                    instance, "synthetic_validation", f"Level {level}", policies, device
                ))
        scenario_seeds = (44001, 44002, 44003)
        for scenario in ("high_reconfiguration", "fleet_scarcity", "high_travel"):
            for seed in scenario_seeds:
                instance = factory.sample(seed, "M", scenario=scenario)
                synthetic_rows.extend(_evaluate_instance(
                    instance, "synthetic_structural_generalization", scenario,
                    policies, device,
                ))
        validation_dir = args.out_dir / "validation"
        synthetic_aggregate = _aggregate(synthetic_rows)
        _write_csv(validation_dir / "synthetic_results.csv", synthetic_rows)
        _write_csv(validation_dir / "synthetic_summary.csv", synthetic_aggregate)
        write_json({
            "records": synthetic_rows,
            "summary": synthetic_aggregate,
            "training_seed_overlap": False,
        }, validation_dir / "final_info.json")
        print(f"synthetic_evaluation_runs={len(synthetic_rows)}")

    canonical_rows = []
    canonical_runtime = 0.0
    if not args.synthetic_only:
        canonical_root = ROOT / "instances" / "canonical" / "RCIAS-2.0"
        paths = sorted(
            path for path in canonical_root.rglob("*.json")
            if path.name not in {"manifest.json", "generation_config.json"}
        )
        if len(paths) != 130:
            raise RuntimeError(f"expected 130 canonical instances, found {len(paths)}")
        started = perf_counter()
        worker_specs = [
            (method, seed, str(path.resolve()))
            for method, seed, path in checkpoint_specs
        ]
        with ProcessPoolExecutor(
            max_workers=args.canonical_workers,
            initializer=_initialize_canonical_worker,
            initargs=(worker_specs,),
        ) as executor:
            futures = {
                executor.submit(_canonical_worker, str(path.resolve())): path
                for path in paths
            }
            for index, future in enumerate(as_completed(futures), start=1):
                canonical_rows.extend(future.result())
                if index % 10 == 0 or index == len(paths):
                    print(f"canonical_evaluated={index}/{len(paths)}")
        canonical_rows.sort(key=lambda row: (
            row["instance_id"], row["method"], str(row["training_seed"])
        ))
        canonical_runtime = perf_counter() - started
        canonical_dir = args.out_dir / "canonical_evaluation"
        canonical_aggregate = _aggregate(canonical_rows)
        _write_csv(canonical_dir / "canonical_results.csv", canonical_rows)
        _write_csv(canonical_dir / "canonical_summary.csv", canonical_aggregate)
        write_json({
            "checkpoint_freeze": freeze_record,
            "instance_count": 130,
            "records": canonical_rows,
            "summary": canonical_aggregate,
            "runtime_seconds": canonical_runtime,
            "canonical_evaluation_complete": True,
        }, canonical_dir / "final_info.json")
    previous_evaluation = {}
    evaluation_path = args.out_dir / "evaluation_final_info.json"
    if evaluation_path.exists():
        with evaluation_path.open("r", encoding="utf-8") as handle:
            previous_evaluation = json.load(handle)
    write_json({
        "synthetic_runs": (
            previous_evaluation.get("synthetic_runs", 0)
            if args.canonical_only else len(synthetic_rows)
        ),
        "canonical_instances": 0 if args.synthetic_only else 130,
        "canonical_runs": len(canonical_rows),
        "canonical_runtime_seconds": canonical_runtime,
        "frozen_before_canonical": True,
        "evaluation_complete": not args.synthetic_only,
    }, args.out_dir / "evaluation_final_info.json")
    print(
        "FROZEN_EVALUATION_COMPLETE = TRUE | "
        f"synthetic_runs={len(synthetic_rows)} canonical_instances="
        f"{0 if args.synthetic_only else 130}"
    )


if __name__ == "__main__":
    main()
