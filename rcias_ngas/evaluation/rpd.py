"""Derived RPD always names and hashes its BKS manifest."""
from .bks import content_hash


def derive_rpd(raw_runs, manifest):
    rows = []
    for raw in raw_runs:
        reference = manifest["instances"][raw["instance_id"]]
        if reference["instance_sha256"] != raw["instance_sha256"]:
            raise ValueError("Instance hash mismatch")
        rows.append({
            "raw_result_path": raw["raw_result_path"], "raw_sha256": raw["raw_sha256"],
            "algorithm_id": raw["algorithm_id"], "instance_id": raw["instance_id"],
            "seed": raw["seed"],
            "rpd_percent": 100.0 * (raw["final_makespan"] - reference["makespan"]) / reference["makespan"] if raw["feasible"] else None,
        })
    return {"schema": "ngas-derived-metrics-v1", "bks_version": manifest["version"],
            "bks_content_sha256": content_hash(manifest), "rows": rows}
