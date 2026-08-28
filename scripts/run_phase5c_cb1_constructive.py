#!/usr/bin/env python3
"""Evaluate frozen constructive methods on the 45 CB1-Core instances."""
from __future__ import annotations
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.learning.evaluator import evaluate_policy, load_checkpoint, load_operation_anchored_checkpoint
from rcias_clgri.heuristic.dispatching import solve_dispatching

MODELS = None


def initialize(specs):
    global MODELS
    import torch
    torch.set_num_threads(1)
    MODELS = []
    for kind, method, seed, path in specs:
        loader = load_operation_anchored_checkpoint if kind == "operation" else load_checkpoint
        model, tensorizer, _ = loader(path, device="cpu")
        MODELS.append((method, seed, model, tensorizer))


def worker(row):
    instance = load_instance(ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"])
    records = []
    for method in ("H1", "H2"):
        result = solve_dispatching(instance, method)
        records.append({"instance_id": instance.instance_id, "suite": "CORE", "scale": row["scale"],
                        "CF_level": row["CF_level"], "method": method, "training_seed": "NA",
                        "makespan": result.objective.makespan, "runtime_seconds": result.runtime_seconds,
                        "inference_seconds": result.runtime_seconds, "feasible": True})
    for method, seed, model, tensorizer in MODELS:
        result = evaluate_policy(model, tensorizer, instance, device="cpu", method=method)
        records.append({**result.to_dict(), "suite": "CORE", "scale": row["scale"],
                        "CF_level": row["CF_level"], "training_seed": seed})
    h1 = next(item["makespan"] for item in records if item["method"] == "H1")
    for item in records: item["gap_to_h1_percent"] = 100 * (item["makespan"] - h1) / h1
    return records


def main():
    output = ROOT / "outputs/phase5c/comparisons/cb1_constructive_results.csv"
    if output.exists():
        print("CB1_CONSTRUCTIVE_ALREADY_COMPLETE"); return
    gate = json.loads((ROOT / "outputs/phase5c/controlled_benchmark_audit/stage_a_gate.json").read_text())
    if not gate["passed"]: raise RuntimeError("Stage-A gate is not passed")
    phase5b = json.loads((ROOT / "outputs/phase5b/canonical_gate.json").read_text())
    specs = [("regular", "BC", "BC", str(ROOT / "outputs/phase4/bc_large/best.pt"))]
    for seed, checkpoint in zip((520101, 520102, 520103), phase5b["frozen_checkpoints"]):
        specs.append(("operation", "PPO", str(seed), str(ROOT / checkpoint["checkpoint"])))
    with (ROOT / "instances/controlled/RCIAS-CB1/manifests/core_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    with ProcessPoolExecutor(max_workers=6, initializer=initialize, initargs=(specs,)) as executor:
        futures = [executor.submit(worker, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            records.extend(future.result()); print(f"core_constructive={index}/45", flush=True)
    records.sort(key=lambda item: (item["instance_id"], item["method"], str(item["training_seed"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(f"CB1_CONSTRUCTIVE_COMPLETE rows={len(records)} feasible={all(x['feasible'] for x in records)}")


if __name__ == "__main__": main()
