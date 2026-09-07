#!/usr/bin/env python3
"""Diagnose the inherited FP16 feature boundary using R12 only; no selection."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import FrozenArmPrediction, select_forced_candidate_roles
from rcias_clgri.ni.batching import batch_state_samples
from scripts import train_phase6j_caur as r
from scripts.run_phase6j_caur_pilot import load_policy
from scripts.audit_phase6k_start import OUT, digest, write_once


def order(scores, ids):
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))


def fallback(scores, frame):
    arms = tuple(FrozenArmPrediction(
        target_set_id=str(row.target_set_id), arm_family=str(row.origin_family),
        origin_destroy_operator=str(row.origin_destroy_operator),
        origin_rules=tuple(json.loads(row.origin_rules)),
        destroyed_operations=tuple(json.loads(row.target_operation_ids)),
        raw_score=float(scores[i]), raw_probability=0, raw_utility=0,
        calibrated_probability=0, calibrated_utility=0,
    ) for i, row in enumerate(frame.itertuples(index=False)))
    return next(x.arm.target_set_id for x in select_forced_candidate_roles(arms)
                if x.role == "ALNS_RELATED_FALLBACK")


def main():
    output = OUT / "source_precision_audit.json"
    if output.exists():
        raise RuntimeError("precision audit already exists; inspect instead of rerunning")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("precision audit requires host CUDA")
    policy = load_policy(r.load_json(r.CONFIG_PATH), "cuda")
    # Read only identities/provenance and previously frozen inference features.
    columns = ["state_id", "target_set_id", "scale", "CF_level", "frozen_raw_score",
               "origin_family", "origin_destroy_operator", "origin_rules", "target_operation_ids"]
    frames = r.state_frames(pd.read_parquet(r.SOURCE_PATH, columns=columns))
    samples = r.load_samples()
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, state_id in enumerate(sorted(frames)):
            frame = frames[state_id]
            batch = batch_state_samples([samples[state_id]]).to("cuda")
            assert tuple(frame.target_set_id) == batch.target_set_ids
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                old = policy.model(batch).scores.float().cpu().numpy()
            new = policy.model(batch).scores.float().cpu().numpy()
            archived = frame.frozen_raw_score.to_numpy()
            assert np.isfinite(old).all() and np.isfinite(new).all()
            ids = batch.target_set_ids
            old_order, new_order = order(old, ids), order(new, ids)
            old_ranks, new_ranks = np.argsort(old_order), np.argsort(new_order)
            archive_order = order(archived, ids)
            rows.append({"state_id": state_id, "scale": str(frame.scale.iloc[0]),
                         "CF_level": str(frame.CF_level.iloc[0]), "candidates": len(ids),
                         "fp16_archive_max_abs_error": float(np.max(np.abs(old - archived))),
                         "fp16_archive_rank_equal": old_order == archive_order,
                         "fp32_vs_fp16_max_abs_error": float(np.max(np.abs(new - old))),
                         "changed_rank_rows": int(np.count_nonzero(old_ranks != new_ranks)),
                         "top1_changed": old_order[0] != new_order[0],
                         "fallback_changed": fallback(old, frame) != fallback(new, frame)})
            if (index + 1) % 24 == 0:
                print(json.dumps({"completed": index + 1, "expected": len(frames),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    report = {"status": "DIAGNOSTIC_ONLY_PENDING_PRECISION_CONTRACT", "states": len(rows),
              "candidate_rows": sum(x["candidates"] for x in rows),
              "changed_rank_states": sum(x["changed_rank_rows"] > 0 for x in rows),
              "changed_rank_rows": sum(x["changed_rank_rows"] for x in rows),
              "top1_changed_states": sum(x["top1_changed"] for x in rows),
              "fallback_changed_states": sum(x["fallback_changed"] for x in rows),
              "fp16_archive_rank_mismatch_states": sum(not x["fp16_archive_rank_equal"] for x in rows),
              "elapsed_seconds": time.perf_counter() - started,
              "source_sha256": digest(r.SOURCE_PATH), "script_sha256": digest(Path(__file__)),
              "r13_accessed": False, "r14_accessed": False, "rows": rows}
    write_once(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
