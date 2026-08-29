#!/usr/bin/env python3
"""Freeze the generated RCIAS-CB1-TRAIN distribution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; TRAIN=ROOT/"instances/controlled/RCIAS-CB1-TRAIN"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files=sorted(path for path in TRAIN.rglob("*.json") if path.parent.name != "manifests")
    if len(files)!=405: raise RuntimeError(f"expected 405 instances, found {len(files)}")
    entries={str(path.relative_to(TRAIN)):digest(path) for path in files}
    canonical=json.dumps(entries,sort_keys=True,separators=(",",":")).encode()
    record={"schema":"rcias-cb1-train-freeze-v1","instance_count":405,
            "instance_collection_sha256":hashlib.sha256(canonical).hexdigest(),
            "manifest_sha256":digest(TRAIN/"manifests/train_instance_manifest.csv"),
            "checksums_sha256":digest(TRAIN/"manifests/checksums.sha256"),
            "generation_spec_sha256":digest(TRAIN/"manifests/generation_spec.json"),
            "frozen_evaluation_hash_overlap":False,"training_only":True}
    record["freeze_hash"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    path=ROOT/"outputs/phase6b/audit/train_distribution_freeze.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n"); print("RCIAS_CB1_TRAIN_FROZEN",record["freeze_hash"])


if __name__=="__main__":main()
