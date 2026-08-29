#!/usr/bin/env python3
"""Run frozen ALNS on the 81 R01 pilot instances and sample immutable states."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import Phase6AObserver
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns

TRAIN = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
OUT = ROOT / "outputs/phase6b/trajectory_reservoir"
STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")


def candidate_json(candidate):
    return json.dumps({"operation_order": candidate.operation_order,
                       "island_assignment": candidate.island_assignment,
                       "w_assignment": candidate.w_assignment,
                       "f_assignment": candidate.f_assignment}, separators=(",", ":"))


class ReservoirObserver:
    def __init__(self, instance, metadata, time_limit, state_seed):
        self.phase6a = Phase6AObserver(instance, metadata)
        self.time_limit = time_limit
        self.rng = random.Random(state_seed)
        self.seen = {stage: 0 for stage in STAGES}
        self.rows = {stage: [] for stage in STAGES}

    def __call__(self, event):
        self.phase6a(event)
        transition = self.phase6a.transitions.pop()
        self.phase6a.targets.clear()
        progress = min(float(event["elapsed_time"]) / self.time_limit, 0.999999)
        stage = STAGES[min(4, int(progress * 5))]
        current = event["current_before"]; best = event["best_before"]
        state_id = f"{self.phase6a.metadata['instance_id']}__it{int(event['iteration']):07d}"
        row = {
            **self.phase6a.metadata, "state_id": state_id, "search_stage": stage,
            "search_progress": progress, "elapsed_time": event["elapsed_time"],
            "iteration": event["iteration"], "current_makespan": current.makespan,
            "historical_best_makespan": best.makespan, "temperature_before": event["temperature_before"],
            "operator_weights_before": json.dumps(event["operator_weights_before"], sort_keys=True),
            "current_candidate": candidate_json(current.candidate), "historical_best_candidate": candidate_json(best.candidate),
            "bottleneck_proxy": transition["bottleneck_type"],
            "trajectory_destroy_operator": event["destroy_operator"],
            "trajectory_repair_operator": event["repair_operator"],
        }
        self.seen[stage] += 1
        bucket = self.rows[stage]
        if len(bucket) < 50:
            bucket.append(row)
        else:
            replacement = self.rng.randrange(self.seen[stage])
            if replacement < 50:
                bucket[replacement] = row

    def selected(self):
        target = 40 if self.phase6a.metadata["scale"] == "S" or self.phase6a.metadata["CF_level"] == "CF3" else 25
        rare = {"F_LOGISTICS": 0, "CROSS_RESOURCE_SYNCHRONIZATION": 0}
        selected = []
        for stage in STAGES:
            rows = self.rows[stage]
            rows.sort(key=lambda row: (rare.get(row["bottleneck_proxy"], 1), self.rng.random()))
            selected.extend(rows[:target])
        return selected


def config():
    raw = json.loads((ROOT / "configs/phase5c_alns.json").read_text())
    return ALNSConfig(**{key: value for key, value in raw.items() if key in ALNSConfig.__dataclass_fields__})


def run_one(record):
    instance = load_instance(TRAIN / record["relative_path"])
    time_limit = 2.0 * instance.num_operations
    metadata = {key: record[key] for key in ("instance_id", "training_split", "scale", "CF_level", "RI_level", "TI_level")}
    metadata.update({"run_id": f"{instance.instance_id}_trajectory", "suite": "TRAIN_ONLY", "seed": int(record["trajectory_seed"])})
    observer = ReservoirObserver(instance, metadata, time_limit, int(record["state_sampling_seed"]))
    result = solve_alns(instance, time_limit, int(record["trajectory_seed"]), config(), observer)
    shard = OUT / "shards" / instance.instance_id; shard.mkdir(parents=True, exist_ok=True)
    states = observer.selected()
    pd.DataFrame(states).to_parquet(shard / "states.parquet", index=False)
    summary = {**metadata, "time_limit": time_limit, "runtime": result.runtime,
               "iterations": result.iterations, "decoder_evaluations": result.decoder_evaluations,
               "best_makespan": result.best.makespan, "feasible": result.best.feasible,
               "state_count": len(states), "seen_by_stage": observer.seen}
    (shard / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return instance.instance_id, len(states), result.iterations


def combine():
    paths = sorted((OUT / "shards").glob("*/states.parquet"))
    summaries = [json.loads(path.read_text()) for path in sorted((OUT / "shards").glob("*/summary.json"))]
    if len(paths) != 81:
        raise RuntimeError(f"expected 81 pilot shards, found {len(paths)}")
    states = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if states.state_id.nunique() != len(states) or len(states) < 10000:
        raise RuntimeError(f"invalid state reservoir: rows={len(states)} unique={states.state_id.nunique()}")
    states.to_parquet(OUT / "pilot_state_manifest.parquet", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "pilot_trajectory_summary.csv", index=False)
    coverage = states.groupby(["scale", "CF_level", "RI_level", "TI_level", "search_stage"]).size().reset_index(name="state_count")
    coverage.to_csv(OUT / "pilot_state_coverage.csv", index=False)
    print(f"PILOT_STATE_RESERVOIR_CREATED states={len(states)}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--combine-only", action="store_true"); args = parser.parse_args()
    if args.combine_only: combine(); return
    manifest = pd.read_csv(TRAIN / "manifests/train_instance_manifest.csv")
    records = manifest[(manifest.replicate == "R01") & (manifest.training_split == "TRAIN")].to_dict("records")
    pending = [record for record in records if not (OUT / "shards" / record["instance_id"] / "summary.json").exists()]
    print(f"PHASE6B_RESERVOIR_START pending={len(pending)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, record) for record in pending]
        for index, future in enumerate(as_completed(futures), 1):
            instance_id, states, iterations = future.result()
            print(f"[{index}/{len(pending)}] {instance_id} states={states} iterations={iterations}", flush=True)
    combine()


if __name__ == "__main__": main()
