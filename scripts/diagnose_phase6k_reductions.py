#!/usr/bin/env python3
"""Localize historical score nondeterminism using fixed captured CUDA operands."""

import json
import os
from pathlib import Path
import sys
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6j_access import load_phase6j_instance
from rcias_clgri.search.common import decode_candidate
from scripts import build_phase6j_caur_tensor_cache as cache
from scripts import run_phase6j_caur_collection as collection
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_start import digest, write_once
from scripts.diagnose_phase6k_preprocessing import OUT, fingerprint, live_batch
from scripts.run_phase6j_caur_pilot import load_policy


class CaptureIndexAdds(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.index_add_.default:
            target, dim, index, source = args
            locations = [f"{Path(item.filename).relative_to(ROOT)}:{item.lineno}"
                         for item in traceback.extract_stack()
                         if str(ROOT / "rcias_clgri/ni") in item.filename]
            self.records.append({"target": target.clone(), "dim": dim, "index": index.clone(),
                                 "source": source.clone(), "kwargs": kwargs, "locations": locations})
        return func(*args, **kwargs)


def replay_operation(record, repetitions):
    outputs = []
    for _ in range(repetitions):
        target = record["target"].clone()
        target.index_add_(record["dim"], record["index"], record["source"], **record["kwargs"])
        outputs.append(target.detach().cpu())
    return {"variants": len({fingerprint(x)["sha256"] for x in outputs}),
            "max_absolute_difference": max(float((x.float() - outputs[0].float()).abs().max()) for x in outputs),
            "first_sha256": fingerprint(outputs[0])["sha256"]}


def main():
    if os.environ.get("PYTHONHASHSEED") != "0" or torch.get_num_threads() != 1:
        raise RuntimeError("original collection environment required")
    output = OUT / "reduction_localization.json"
    if output.exists():
        raise RuntimeError("operator diagnostic exists; do not overwrite")
    parent = r.load_json(OUT / "protocol.json")
    protocol = {"status": "FROZEN_BEFORE_OPERATOR_DIAGNOSTIC", "state_ids": parent["state_ids"],
                "repetitions_per_operation": 50, "determinism_modes": [False, True],
                "script_sha256": digest(Path(__file__)), "parent_sha256": digest(OUT / "repeatability.json"),
                "scope": "fixed captured operands, diagnostic only; no deployable code change",
                "r13_accessed": False, "r14_accessed": False}
    write_once(OUT / "reduction_protocol.json", protocol)
    config = r.load_json(r.CONFIG_PATH)
    policy = load_policy(config, "cuda")
    replays, rows = cache.replay_index(), []
    alns = collection.read_alns_config(config)
    with torch.inference_mode():
        for state_id in protocol["state_ids"]:
            replay = r.load_json(replays[state_id])
            instance = load_phase6j_instance(ROOT / config["instance_suite"]["root"] / replay["instance_relative_path"])
            snapshot = replay["snapshot"]
            current = decode_candidate(instance, collection.candidate_from_dict(snapshot["current_candidate"]))
            count = min(max(2, round(instance.num_operations * alns.destroy_fraction)), instance.num_operations)
            batch, _ = live_batch(policy, instance, current, snapshot, count)
            capture = CaptureIndexAdds()
            torch.use_deterministic_algorithms(False)
            with capture, torch.autocast("cuda", dtype=torch.float16):
                policy.model(batch.to("cuda"))
            operations = []
            for ordinal, record in enumerate(capture.records):
                before = fingerprint({key: record[key] for key in ("target", "index", "source")})
                runs = {}
                for deterministic in protocol["determinism_modes"]:
                    torch.use_deterministic_algorithms(deterministic)
                    runs[str(deterministic)] = replay_operation(record, protocol["repetitions_per_operation"])
                assert before == fingerprint({key: record[key] for key in ("target", "index", "source")})
                operations.append({"ordinal": ordinal, "locations": record["locations"],
                                   "operand_fingerprints": before, "runs": runs})
            rows.append({"state_id": state_id, "operations": operations})
            print(json.dumps({"state_id": state_id, "operations": len(operations),
                              "variable_default": sum(x["runs"]["False"]["variants"] > 1 for x in operations),
                              "variable_deterministic": sum(x["runs"]["True"]["variants"] > 1 for x in operations)}), flush=True)
    write_once(output, {"status": "DIAGNOSIS_COMPLETE", "rows": rows,
                        "protocol_sha256": digest(OUT / "reduction_protocol.json"),
                        "r13_accessed": False, "r14_accessed": False})


if __name__ == "__main__":
    main()
