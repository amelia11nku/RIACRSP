#!/usr/bin/env python3
"""Replay all R12 frozen-score features under the original collection environment."""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import score_frozen_candidate_bank, select_forced_candidate_roles
from rcias_clgri.analysis.phase6j_caur import build_candidate_source_features, critical_and_bottleneck_operations
from rcias_clgri.data.phase6j_access import load_phase6j_instance
from rcias_clgri.search.common import decode_candidate
from rcias_clgri.search.phase6c import generate_revised_target_arms
from scripts import build_phase6j_caur_tensor_cache as cache
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_start import OUT, digest, write_once
from scripts.run_phase6j_caur_pilot import load_policy


def main():
    if os.environ.get("PYTHONHASHSEED") != "0" or torch.get_num_threads() != 1:
        raise RuntimeError("use original collection environment: PYTHONHASHSEED=0 and all CPU thread limits=1")
    assert not torch.are_deterministic_algorithms_enabled()
    config = r.load_json(r.CONFIG_PATH)
    policy = load_policy(config, "cuda")
    alns = collection.read_alns_config(config)
    columns = list(dict.fromkeys(["state_id", "target_set_id", "frozen_raw_score", "scale", "CF_level",
                                  *r.CATEGORICAL_COLUMNS, *r.NUMERIC_COLUMNS]))
    source = pd.read_parquet(r.SOURCE_PATH, columns=columns)
    frames = r.state_frames(source)
    instances, rows = {}, []
    start = time.perf_counter()
    for state_id, replay_path in sorted(cache.replay_index().items()):
        replay = json.loads(replay_path.read_text())
        relative = replay["instance_relative_path"]
        if relative not in instances:
            path = ROOT / config["instance_suite"]["root"] / relative
            assert digest(path) == replay["instance_sha256"]
            instances[relative] = load_phase6j_instance(path)
        instance = instances[relative]
        snapshot = replay["snapshot"]
        current = decode_candidate(instance, collection.candidate_from_dict(snapshot["current_candidate"]))
        assert abs(current.makespan - snapshot["current_makespan"]) < 1e-9
        progress = float(snapshot["search_progress"])
        count = min(max(2, round(instance.num_operations * alns.destroy_fraction)), instance.num_operations)
        bank = score_frozen_candidate_bank(policy, instance, current, state_id=state_id,
                                          destroy_count=count, search_progress=progress,
                                          search_stage=collection.search_stage(progress))
        generated = generate_revised_target_arms(instance, current, state_id, count, int(config["rng"]["proposal_namespace"]))
        roles = select_forced_candidate_roles(bank.arms)
        fallback_id = next(x.arm.target_set_id for x in roles if x.role == "ALNS_RELATED_FALLBACK")
        critical, bottleneck, _ = critical_and_bottleneck_operations(instance, current)
        scores = {arm.target_set_id: arm.raw_score for arm in bank.arms}
        features = pd.DataFrame(build_candidate_source_features(
            generated, state_id=state_id, operation_count=instance.num_operations,
            fallback_target_set_id=fallback_id, frozen_scores=scores,
            critical_operations=critical, bottleneck_operations=bottleneck,
        )).sort_values("target_set_id").reset_index(drop=True)
        original = frames[state_id]
        original_ids = list(original.target_set_id)
        actual_scores = np.array([scores[x] for x in original_ids])
        archived_scores = original.frozen_raw_score.to_numpy()
        feature_equal = {column: bool(np.array_equal(features[column].to_numpy(), original[column].to_numpy()))
                         for column in (*r.CATEGORICAL_COLUMNS, *r.NUMERIC_COLUMNS)}
        rows.append({"state_id": state_id, "scale": str(original.scale.iloc[0]),
                     "CF_level": str(original.CF_level.iloc[0]), "candidate_rows": len(original),
                     "bank_identity_and_order_exact": [a.target_set_id for a in bank.arms] == replay["full_bank_target_ids"],
                     "score_values_exact": bool(np.array_equal(actual_scores, archived_scores)),
                     "score_max_absolute_error": float(np.max(np.abs(actual_scores - archived_scores))),
                     "features_exact": all(feature_equal.values()), "feature_checks": feature_equal,
                     "fallback_exact": fallback_id == replay["fallback_target_set_id"]})
        if len(rows) % 24 == 0:
            print(json.dumps({"completed": len(rows), "expected": 288, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    checks = {key: all(row[key] for row in rows) for key in
              ("bank_identity_and_order_exact", "score_values_exact", "features_exact", "fallback_exact")}
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "states": len(rows), "candidate_rows": sum(row["candidate_rows"] for row in rows),
              "elapsed_seconds": time.perf_counter() - start,
              "environment": {"PYTHONHASHSEED": "0", "torch_threads": 1, "deterministic_algorithms": False,
                              "dtype": "historical CUDA FP16 autocast then .detach().float().cpu().numpy()",
                              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
              "mismatch_states": {key: sum(not row[key] for row in rows) for key in checks},
              "r13_accessed": False, "r14_accessed": False,
              "script_sha256": digest(Path(__file__)), "source_sha256": digest(r.SOURCE_PATH), "rows": rows}
    write_once(OUT / "preprocessing_equivalence.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
