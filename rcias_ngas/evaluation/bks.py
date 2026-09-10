"""Immutable BKS manifests; reference evidence is independent of algorithms."""
import hashlib
import json
import math


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_manifest(raw_runs, version, previous=None):
    if version < 1 or (previous and version != previous["version"] + 1):
        raise ValueError("BKS versions must advance by one")
    entries = {key: dict(value) for key, value in (previous or {}).get("instances", {}).items()}
    for row in raw_runs:
        if not row["feasible"]:
            continue
        value = row["final_makespan"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid feasible makespan")
        key = row["instance_id"]
        old = entries.get(key)
        if old and old["instance_sha256"] != row["instance_sha256"]:
            raise ValueError("Instance hash drift")
        if old is None or value < old["makespan"]:
            entries[key] = {
                "makespan": value, "instance_sha256": row["instance_sha256"],
                "raw_result_path": row["raw_result_path"], "raw_sha256": row["raw_sha256"],
            }
    return {"schema": "ngas-bks-v1", "version": version,
            "previous_manifest_sha256": content_hash(previous) if previous else None,
            "instances": entries}


def write_immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
