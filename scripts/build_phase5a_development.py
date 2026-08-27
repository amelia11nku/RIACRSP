#!/usr/bin/env python3
"""Freeze the synthetic Phase 5A development benchmark and BC reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.learning.evaluator import evaluate_baselines, evaluate_policy, load_checkpoint
from rcias_clgri.learning.experiment import load_phase3_config, make_factory, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seed_boundaries(config: dict[str, object]) -> None:
    development = {
        int(seed) for seeds in config["development_seeds"].values() for seed in seeds
    }
    historical = {
        int(seed)
        for seeds in config["historical_validation_seeds"].values()
        for seed in seeds
    }
    training = {
        int(seed) for seed in config["training_seed_policy"]["independent_training_seeds"]
    }
    if len(development) != 30:
        raise ValueError("Phase 5A development set must contain 30 unique seeds")
    if development & historical or development & training:
        raise ValueError("development seeds overlap historical validation or training seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5a_training.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5a/development_reference"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_phase3_config(args.config)
    _validate_seed_boundaries(config)
    device = resolve_device(args.device)
    checkpoint = ROOT / config["teacher_checkpoint"]
    model, tensorizer, metadata = load_checkpoint(checkpoint, device=device)
    factory = make_factory(ROOT, config)
    rows = []
    for level, seeds in config["development_seeds"].items():
        for seed in seeds:
            instance = factory.sample(int(seed), level)
            baselines = {row.method: row for row in evaluate_baselines(instance)}
            bc = evaluate_policy(model, tensorizer, instance, device=device, method="BC_GREEDY")
            best = min(baselines["H1"].makespan, baselines["H2"].makespan, bc.makespan)
            rows.append({
                "instance": instance.instance_id,
                "seed": int(seed),
                "level": level,
                "h1_makespan": baselines["H1"].makespan,
                "h2_makespan": baselines["H2"].makespan,
                "bc_makespan": bc.makespan,
                "best_reference": best,
                "bc_feasible": bc.feasible,
            })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "development_reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json({
        "instance_count": len(rows),
        "canonical_instances": 0,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "levels": {level: sum(row["level"] == level for row in rows) for level in ("S", "M", "L")},
        "mean_makespan": {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in ("h1_makespan", "h2_makespan", "bc_makespan")
        },
        "all_bc_feasible": all(bool(row["bc_feasible"]) for row in rows),
        "records": rows,
    }, args.out_dir / "final_info.json")
    print(f"PHASE5A_DEVELOPMENT_FROZEN = TRUE | instances={len(rows)} canonical=0")


if __name__ == "__main__":
    main()
