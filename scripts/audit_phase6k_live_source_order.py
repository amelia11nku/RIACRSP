#!/usr/bin/env python3
"""Resolve cached-vs-live FP16 provenance discrepancies on affected R12 states."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import score_frozen_candidate_bank
from rcias_clgri.data.phase6j_access import load_phase6j_instance
from rcias_clgri.search.common import decode_candidate
from scripts import build_phase6j_caur_tensor_cache as cache
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_source_precision import order
from scripts.audit_phase6k_start import OUT, digest, write_once
from scripts.run_phase6j_caur_pilot import load_policy


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = r.load_json(r.CONFIG_PATH)
    policy = load_policy(config, "cuda")
    previous = json.loads((OUT / "source_precision_audit.json").read_text())
    selected = [row["state_id"] for row in previous["rows"] if not row["fp16_archive_rank_equal"]]
    source = pd.read_parquet(r.SOURCE_PATH, columns=["state_id", "target_set_id", "instance_relative_path", "frozen_raw_score"])
    replays = cache.replay_index()
    rows = []
    for state_id in selected:
        frame = source[source.state_id == state_id].sort_values("target_set_id")
        snapshot = json.loads(replays[state_id].read_text())["snapshot"]
        path = ROOT / config["instance_suite"]["root"] / str(frame.instance_relative_path.iloc[0])
        instance = load_phase6j_instance(path)
        current = decode_candidate(instance, collection.candidate_from_dict(snapshot["current_candidate"]))
        progress = float(snapshot["search_progress"])
        alns = json.loads((ROOT / config["locked_inputs"]["frozen_alns_config"]).read_text())
        count = min(max(2, round(instance.num_operations * alns["destroy_fraction"])), instance.num_operations)
        bank = score_frozen_candidate_bank(policy, instance, current, state_id=state_id,
                                          destroy_count=count, search_progress=progress,
                                          search_stage=collection.search_stage(progress))
        scores = {arm.target_set_id: arm.raw_score for arm in bank.arms}
        actual = np.array([scores[x] for x in frame.target_set_id])
        archived = frame.frozen_raw_score.to_numpy()
        rows.append({"state_id": state_id, "raw_scores_exact": bool(np.array_equal(actual, archived)),
                     "max_absolute_error": float(np.max(np.abs(actual - archived))),
                     "rank_equal": order(actual, tuple(frame.target_set_id)) == order(archived, tuple(frame.target_set_id)),
                     "live_order_equals_sorted_cache_order": tuple(x.target_set_id for x in bank.arms) == tuple(frame.target_set_id)})
    report = {"status": "PASS" if all(row["raw_scores_exact"] for row in rows) else "UNRESOLVED",
              "scope": "original live FP16 candidate order on all cached/archive rank discrepancies",
              "rows": rows, "r13_accessed": False, "r14_accessed": False,
              "script_sha256": digest(Path(__file__)), "source_precision_audit_sha256": digest(OUT / "source_precision_audit.json")}
    write_once(OUT / "live_source_order_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
