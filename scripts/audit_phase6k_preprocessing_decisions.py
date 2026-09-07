#!/usr/bin/env python3
"""Measure input/decision consequences of historical reduction nondeterminism.

Both preprocessing modes are diagnostics, not deployable runtime candidates.
J1 heads, normalization, calibration and gate remain frozen throughout.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

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
from scripts import prepare_phase6j_caur_deployment as deploy
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_start import digest, write_once
from scripts.diagnose_phase6k_preprocessing import OUT
from scripts.run_phase6j_caur_pilot import load_policy


def inference_inputs(sample, frame, transform, device):
    """Single-state model inputs from membership/features only; no label fields."""
    graph, actions = sample.graph, sample.actions
    if tuple(frame.target_set_id) != actions.target_set_ids:
        raise ValueError("inference candidate ordering changed")
    categorical, numeric, supported = r.transform_features(frame, transform)
    fallback = np.flatnonzero(frame.is_fallback.to_numpy(dtype=bool))
    if len(fallback) != 1:
        raise ValueError("one fallback is required")
    batch = SimpleNamespace(
        tensor_schema_hash=graph.tensor_schema_hash, state_count=1, action_count=len(frame),
        target_set_ids=actions.target_set_ids, node_features={k: v.to(device) for k, v in graph.node_features.items()},
        node_batch_index={k: torch.zeros(len(v), dtype=torch.long, device=device) for k, v in graph.node_features.items()},
        node_ptr={k: torch.tensor([0, len(v)], device=device) for k, v in graph.node_features.items()},
        edges={k: v.to(device) for k, v in graph.edges.items()},
        graph_numeric=graph.graph_numeric.unsqueeze(0).to(device),
        graph_categorical=graph.graph_categorical.unsqueeze(0).to(device),
        action_to_state=torch.zeros(len(frame), dtype=torch.long, device=device),
        target_operation_indices=actions.target_operation_indices.to(device),
        target_action_index=actions.target_action_index.to(device),
    )
    return {"batch": batch, "categorical": torch.as_tensor(categorical, dtype=torch.long, device=device),
            "numeric": torch.as_tensor(numeric, dtype=torch.float32, device=device), "supported": supported,
            "fallback_indices": torch.as_tensor(fallback, dtype=torch.long, device=device), "frame": frame}


def main():
    if os.environ.get("PYTHONHASHSEED") != "0" or torch.get_num_threads() != 1:
        raise RuntimeError("original collection environment required")
    output_path = OUT / "decision_impact.json"
    if output_path.exists():
        raise RuntimeError("decision impact audit exists; do not rerun")
    config = r.load_json(r.CONFIG_PATH)
    protocol, parent = deploy.validate_protocol(), r.validate_protocol()
    checkpoint = torch.load(deploy.seed_paths(696101)[0], map_location="cpu", weights_only=False)
    frozen = checkpoint["feature_transform"]
    transform = r.FeatureTransform({k: tuple(v) for k, v in frozen["vocabularies"].items()}, frozen["medians"], frozen["iqrs"])
    models = [deploy.load_seed(seed, transform, protocol, parent, torch.device("cuda")) for seed in (696101, 696102, 696103)]
    ensemble = deploy.SharedFrozenCAUREnsemble(models)
    policy = load_policy(config, "cuda")
    settings = {"status": "FROZEN_BEFORE_DIAGNOSTIC", "script_sha256": digest(Path(__file__)),
                "states": 288, "preprocessing_modes": ["historical_default", "deterministic_diagnostic"],
                "j1_inference": "unchanged E0 FP32, deterministic algorithms True", "repetitions": 1,
                "selection_allowed": False, "source_sha256": digest(r.SOURCE_PATH),
                "deployment_protocol_sha256": digest(deploy.PROTOCOL_PATH),
                "seed_hashes": {str(seed): digest(deploy.seed_paths(seed)[0]) for seed in (696101, 696102, 696103)},
                "r13_accessed": False, "r14_accessed": False}
    write_once(OUT / "decision_protocol.json", settings)
    columns = list(dict.fromkeys(["state_id", "target_set_id", "scale", "CF_level", "frozen_raw_score",
                                  *r.CATEGORICAL_COLUMNS, *r.NUMERIC_COLUMNS]))
    frames = r.state_frames(pd.read_parquet(r.SOURCE_PATH, columns=columns))
    samples, instances, rows = r.load_samples(), {}, []
    alns = collection.read_alns_config(config)
    started = time.perf_counter()
    with torch.inference_mode():
        for state_id, replay_path in sorted(cache.replay_index().items()):
            original = frames[state_id]
            replay = r.load_json(replay_path)
            relative = replay["instance_relative_path"]
            if relative not in instances:
                instance_path = ROOT / config["instance_suite"]["root"] / relative
                assert digest(instance_path) == replay["instance_sha256"]
                instances[relative] = load_phase6j_instance(instance_path)
            instance, snapshot = instances[relative], replay["snapshot"]
            current = decode_candidate(instance, collection.candidate_from_dict(snapshot["current_candidate"]))
            assert abs(current.makespan - snapshot["current_makespan"]) < 1e-9
            count = min(max(2, round(instance.num_operations * alns.destroy_fraction)), instance.num_operations)
            critical, bottleneck, _ = critical_and_bottleneck_operations(instance, current)
            generated = generate_revised_target_arms(instance, current, state_id, count, int(config["rng"]["proposal_namespace"]))
            torch.use_deterministic_algorithms(True)
            packed = inference_inputs(samples[state_id], original, transform, "cuda")
            reference_output = ensemble(packed["batch"], **deploy.model_inputs(packed))
            reference = deploy.deployment_decision(reference_output, packed, protocol)
            for mode in settings["preprocessing_modes"]:
                torch.use_deterministic_algorithms(mode == "deterministic_diagnostic")
                bank = score_frozen_candidate_bank(policy, instance, current, state_id=state_id, destroy_count=count,
                        search_progress=snapshot["search_progress"], search_stage=collection.search_stage(snapshot["search_progress"]))
                fallback_id = next(role.arm.target_set_id for role in select_forced_candidate_roles(bank.arms)
                                   if role.role == "ALNS_RELATED_FALLBACK")
                scores = {arm.target_set_id: arm.raw_score for arm in bank.arms}
                features = pd.DataFrame(build_candidate_source_features(generated, state_id=state_id,
                    operation_count=instance.num_operations, fallback_target_set_id=fallback_id, frozen_scores=scores,
                    critical_operations=critical, bottleneck_operations=bottleneck)).sort_values("target_set_id").reset_index(drop=True)
                torch.use_deterministic_algorithms(True)
                changed = inference_inputs(samples[state_id], features, transform, "cuda")
                actual_output = ensemble(changed["batch"], **deploy.model_inputs(changed))
                actual = deploy.deployment_decision(actual_output, changed, protocol)
                rows.append({"state_id": state_id, "scale": str(original.scale.iloc[0]), "CF_level": str(original.CF_level.iloc[0]),
                    "mode": mode, "candidate_rows": len(original),
                    "bank_order_exact": [x.target_set_id for x in bank.arms] == replay["full_bank_target_ids"],
                    "scores_exact": np.array_equal([scores[x] for x in original.target_set_id], original.frozen_raw_score.to_numpy()),
                    "features_exact": all(np.array_equal(features[c].to_numpy(), original[c].to_numpy()) for c in (*r.CATEGORICAL_COLUMNS, *r.NUMERIC_COLUMNS)),
                    "support_exact": np.array_equal(packed["supported"], changed["supported"]),
                    "selected_action_exact": actual.selected_target_set_id == reference.selected_target_set_id,
                    "neural_winner_exact": actual.neural_target_set_id == reference.neural_target_set_id,
                    "fallback_exact": actual.fallback_target_set_id == reference.fallback_target_set_id,
                    "intervention_exact": actual.intervened == reference.intervened,
                    "reference": asdict(reference), "actual": asdict(actual),
                    "three_seed_max_absolute_difference": max(float((a - b).abs().max()) for a, b in zip(actual_output, reference_output))})
            if len(rows) % 48 == 0:
                print(json.dumps({"completed_states": len(rows) // 2, "expected_states": 288,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    keys = ("bank_order_exact", "scores_exact", "features_exact", "support_exact", "selected_action_exact",
            "neural_winner_exact", "fallback_exact", "intervention_exact")
    summary = {mode: {key: sum(not row[key] for row in rows if row["mode"] == mode) for key in keys}
               for mode in settings["preprocessing_modes"]}
    write_once(output_path, {"status": "DIAGNOSIS_COMPLETE_NOT_ELIGIBILITY", "mismatch_states": summary, "rows": rows,
                "elapsed_seconds": time.perf_counter() - started, "protocol_sha256": digest(OUT / "decision_protocol.json"),
                "r13_accessed": False, "r14_accessed": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
