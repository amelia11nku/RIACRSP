#!/usr/bin/env python3
"""Diagnose historical FP16 reduction settings on the two discrepant R12 states."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.batching import batch_state_samples
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_source_precision import order
from scripts.audit_phase6k_start import OUT, digest, write_once
from scripts.run_phase6j_caur_pilot import load_policy


def main():
    torch.set_num_threads(4)
    previous = json.loads((OUT / "source_precision_audit.json").read_text())
    selected = [row["state_id"] for row in previous["rows"] if not row["fp16_archive_rank_equal"]]
    source = pd.read_parquet(r.SOURCE_PATH, columns=["state_id", "target_set_id", "frozen_raw_score"])
    samples = r.load_samples()
    policy = load_policy(r.load_json(r.CONFIG_PATH), "cuda")
    rows = []
    with torch.inference_mode():
        for state_id in selected:
            frame = source[source.state_id == state_id].sort_values("target_set_id")
            archived = frame.frozen_raw_score.to_numpy()
            batch = batch_state_samples([samples[state_id]]).to("cuda")
            for deterministic in (False, True):
                torch.use_deterministic_algorithms(deterministic)
                for repetition in range(3):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        actual = policy.model(batch).scores.float().cpu().numpy()
                    rows.append({"state_id": state_id, "deterministic_algorithms": deterministic,
                                 "repetition": repetition, "raw_scores_exact": bool(np.array_equal(actual, archived)),
                                 "maximum_absolute_error": float(np.max(np.abs(actual - archived))),
                                 "rank_equal": order(actual, batch.target_set_ids) == order(archived, batch.target_set_ids)})
    report = {"scope": "DIAGNOSTIC_ONLY_NOT_RUNTIME_SELECTION", "rows": rows,
              "script_sha256": digest(Path(__file__)), "r13_accessed": False, "r14_accessed": False}
    write_once(OUT / "source_determinism_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
