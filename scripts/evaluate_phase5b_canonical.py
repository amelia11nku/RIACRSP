#!/usr/bin/env python3
"""Run the one-time frozen Phase 5B evaluation on all 130 canonical instances."""

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
from rcias_clgri.learning.evaluator import (
    evaluate_baselines,
    evaluate_policy,
    load_checkpoint,
    load_operation_anchored_checkpoint,
)

_WORKER_POLICIES = None


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
    return "Unknown"


def _initialize_worker(specs):
    global _WORKER_POLICIES
    import torch
    torch.set_num_threads(1)
    _WORKER_POLICIES = []
    for checkpoint_type, method, seed, checkpoint in specs:
        loader = (
            load_operation_anchored_checkpoint
            if checkpoint_type == "operation_anchored"
            else load_checkpoint
        )
        model, tensorizer, _ = loader(checkpoint, device="cpu")
        _WORKER_POLICIES.append((method, seed, model, tensorizer))


def _worker(path_string: str):
    instance = load_instance(path_string)
    rows = []
    for result in evaluate_baselines(instance):
        if result.method not in {"H1", "H2"}:
            continue
        rows.append({
            **result.to_dict(),
            "family": _family(instance.instance_id),
            "training_seed": "NA",
        })
    for method, seed, model, tensorizer in _WORKER_POLICIES:
        result = evaluate_policy(model, tensorizer, instance, device="cpu", method=method)
        rows.append({
            **result.to_dict(),
            "family": _family(instance.instance_id),
            "training_seed": seed,
        })
    bc = next(float(row["makespan"]) for row in rows if row["method"] == "BC_GREEDY")
    for row in rows:
        row["gap_to_bc_percent"] = 100.0 * (float(row["makespan"]) - bc) / bc
    return rows


def _aggregate(rows):
    records = []
    for family in sorted({row["family"] for row in rows}) + ["Overall"]:
        for method in sorted({row["method"] for row in rows}):
            selected = [
                row for row in rows
                if row["method"] == method and (family == "Overall" or row["family"] == family)
            ]
            gaps = [float(row["gap_to_bc_percent"]) for row in selected]
            records.append({
                "family": family,
                "method": method,
                "runs": len(selected),
                "mean_makespan": mean(float(row["makespan"]) for row in selected),
                "mean_gap_to_bc_percent": mean(gaps),
                "std_gap_to_bc_percent": pstdev(gaps) if len(gaps) > 1 else 0.0,
                "feasibility_rate": mean(float(row["feasible"]) for row in selected),
                "mean_runtime_seconds": mean(float(row["runtime_seconds"]) for row in selected),
            })
    return records


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase5b/canonical_evaluation"))
    args = parser.parse_args()
    output_dir = ROOT / args.out_dir
    final_path = output_dir / "final_info.json"
    if final_path.exists():
        raise RuntimeError("Phase 5B canonical evaluation already exists and must not be rerun")
    gate_path = ROOT / "outputs/phase5b/canonical_gate.json"
    gate = json.loads(gate_path.read_text())
    if not gate.get("canonical_evaluation_authorized"):
        raise RuntimeError("Phase 5B canonical gate has not authorized evaluation")
    specs = [
        ("regular", "BC_GREEDY", "BC", str((ROOT / "outputs/phase4/bc_large/best.pt").resolve())),
        *[
            (
                "operation_anchored",
                "PHASE5B_DOWNSTREAM_PPO",
                str(seed),
                str((ROOT / checkpoint["checkpoint"]).resolve()),
            )
            for seed, checkpoint in zip((520101, 520102, 520103), gate["frozen_checkpoints"])
        ],
    ]
    canonical_root = ROOT / "instances/canonical/RCIAS-2.0"
    paths = sorted(
        path for path in canonical_root.rglob("*.json")
        if path.name not in {"manifest.json", "generation_config.json"}
    )
    if len(paths) != 130:
        raise RuntimeError(f"expected 130 canonical instances, found {len(paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_record = {
        "frozen_before_canonical_evaluation": True,
        "canonical_gate": gate,
        "config_sha256": gate["config_sha256"],
        "checkpoints": gate["frozen_checkpoints"],
    }
    write_json(freeze_record, output_dir / "frozen_evaluation_config.json")
    rows = []
    started = perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(specs,),
    ) as executor:
        futures = {executor.submit(_worker, str(path.resolve())): path for path in paths}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 10 == 0 or index == len(paths):
                print(f"canonical_evaluated={index}/130")
    rows.sort(key=lambda row: (row["instance_id"], row["method"], str(row["training_seed"])))
    runtime = perf_counter() - started
    summary = _aggregate(rows)
    _write_csv(output_dir / "canonical_results.csv", rows)
    _write_csv(output_dir / "canonical_summary.csv", summary)
    write_json({
        "canonical_evaluation_complete": True,
        "canonical_instances": 130,
        "canonical_runs": len(rows),
        "runtime_seconds": runtime,
        "checkpoint_freeze": freeze_record,
        "all_feasible": all(row["feasible"] for row in rows),
        "summary": summary,
        "records": rows,
    }, final_path)
    print(f"PHASE5B_CANONICAL_EVALUATION_COMPLETE instances=130 runs={len(rows)}")


if __name__ == "__main__":
    main()
