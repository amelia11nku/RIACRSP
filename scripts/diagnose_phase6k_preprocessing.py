#!/usr/bin/env python3
"""Bounded R12-only diagnosis of historical frozen-score repeatability."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.phase6j_access import load_phase6j_instance
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.search.common import decode_candidate
from scripts import build_phase6j_caur_tensor_cache as cache
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_start import digest, write_once
from scripts.run_phase6j_caur_pilot import load_policy

OUT = ROOT / "outputs/phase6k_runtime_v1/diagnostics/preprocessing_v2"


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    raise TypeError(type(value))


def input_fingerprint(batch):
    return fingerprint({
        "node_features": dict(batch.node_features), "node_batch_index": dict(batch.node_batch_index),
        "node_ptr": dict(batch.node_ptr), "edges": {key: (edge.index, edge.features) for key, edge in batch.edges.items()},
        **{key: getattr(batch, key) for key in ("graph_numeric", "graph_categorical", "action_to_state",
                                                "target_operation_indices", "target_action_index")},
    })


def live_batch(policy, instance, current, snapshot, count):
    state_id, progress = snapshot["state_id"], snapshot["search_progress"]
    graph = build_csg_from_schedule(instance, current.schedule, state_id=state_id,
                                    search_progress=progress, search_stage=collection.search_stage(progress))
    generated, records = build_live_proposal_bank(instance, current, state_id=state_id, destroy_count=count,
                                                 seed_namespace=policy.proposal_seed_namespace)
    sample = NIStateSample(policy.tensorizer.tensorize(graph), tensorize_action_records(graph, records), {})
    return batch_state_samples([sample]), generated


@contextmanager
def module_fingerprints(model, captured):
    modules = {
        **{f"input.{name}": module for name, module in model.state_encoder.input_projection.items()},
        **{f"layer.{i}": layer for i, layer in enumerate(model.state_encoder.layers)},
        "graph_projection": model.state_encoder.graph_projection,
        "action_encoder": model.action_encoder, "score_head": model.score_head,
    }
    handles = [module.register_forward_hook(
        lambda module, args, output, name=name: captured.__setitem__(name, fingerprint(output)))
        for name, module in modules.items()]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def main():
    if os.environ.get("PYTHONHASHSEED") != "0" or torch.get_num_threads() != 1:
        raise RuntimeError("historical hash seed and thread limits required")
    if (OUT / "repeatability.json").exists():
        raise RuntimeError("diagnostic complete; inspect existing evidence")
    config = r.load_json(r.CONFIG_PATH)
    source = pd.read_parquet(r.SOURCE_PATH, columns=["state_id", "target_set_id", "scale", "frozen_raw_score"])
    parent = r.load_json(ROOT / "outputs/phase6k_runtime_v1/audit/preprocessing_equivalence.json")
    selected = sorted({row["state_id"] for row in parent["rows"] if not row["features_exact"]}
                      | {source[source.scale == scale].state_id.min() for scale in ("S", "M", "L")})
    protocol = {"status": "FROZEN_BEFORE_DIAGNOSTIC", "state_ids": selected,
                "repetitions": 30, "traced_repetitions": 10,
                "modes": ["fixed_gpu_input", "traced_fixed_gpu_input"],
                "purpose": "localize reproducibility failure, never select runtime or model",
                "script_sha256": digest(Path(__file__)), "source_sha256": digest(r.SOURCE_PATH),
                "r13_accessed": False, "r14_accessed": False}
    write_once(OUT / "protocol.json", protocol)
    policy = load_policy(config, "cuda")
    alns = collection.read_alns_config(config)
    replays, rows = cache.replay_index(), []
    for state_id in selected:
        replay = r.load_json(replays[state_id])
        instance = load_phase6j_instance(ROOT / config["instance_suite"]["root"] / replay["instance_relative_path"])
        snapshot = replay["snapshot"]
        current = decode_candidate(instance, collection.candidate_from_dict(snapshot["current_candidate"]))
        count = min(max(2, round(instance.num_operations * alns.destroy_fraction)), instance.num_operations)
        first, _ = live_batch(policy, instance, current, snapshot, count)
        rebuilt, _ = live_batch(policy, instance, current, snapshot, count)
        input_hash = input_fingerprint(first)
        assert input_hash == input_fingerprint(rebuilt)
        batch = first.to("cuda")
        expected = source[source.state_id == state_id].set_index("target_set_id").loc[list(batch.target_set_ids)].frozen_raw_score.to_numpy()
        outputs, traced = [], []
        for repetition in range(protocol["repetitions"]):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = policy.model(batch)
            outputs.append(output.scores.detach().float().cpu().numpy())
        for repetition in range(protocol["traced_repetitions"]):
            captured = {}
            with module_fingerprints(policy.model, captured), torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = policy.model(batch)
            traced.append(captured)
        variants = {name: len({json.dumps(item[name], sort_keys=True) for item in traced}) for name in traced[0]}
        score_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in outputs]
        rows.append({"state_id": state_id, "input_fingerprint": input_hash,
                     "reconstructed_inputs_equal": True, "fixed_input_unchanged": input_hash == input_fingerprint(batch),
                     "score_variants": len(set(score_hashes)), "score_sha256_by_repetition": score_hashes,
                     "archive_exact_repetitions": sum(np.array_equal(x, expected) for x in outputs),
                     "max_repeat_difference": max(float(np.max(np.abs(x - outputs[0]))) for x in outputs),
                     "module_output_variants": variants, "traced_fingerprints": traced,
                     "scores_by_repetition": [x.tolist() for x in outputs]})
        print(json.dumps({k: rows[-1][k] for k in ("state_id", "score_variants", "archive_exact_repetitions", "max_repeat_difference", "module_output_variants")}), flush=True)
    write_once(OUT / "repeatability.json", {"status": "DIAGNOSIS_COMPLETE", "rows": rows,
                "protocol_sha256": digest(OUT / "protocol.json"), "r13_accessed": False, "r14_accessed": False})


if __name__ == "__main__":
    main()
