#!/usr/bin/env python3
"""Freeze the Phase 6E latency bottleneck diagnosis used by Phase 6F."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "phase6e" / "profiling" / "inference_profile.csv"
SOURCE_SUMMARY = (
    ROOT / "outputs" / "phase6e" / "profiling" / "inference_profile_summary.json"
)
OUTPUT = (
    ROOT / "outputs" / "phase6f" / "profiling" / "phase6e_latency_bottleneck_audit.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    profile = pd.read_csv(SOURCE)
    phase6e_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    expected = {"S", "M", "L"}
    if set(profile["scale"]) != expected or phase6e_summary.get("profile_split") != "TRAIN_VALIDATION":
        raise ValueError("Phase 6E latency evidence does not match the frozen protocol")

    rows = []
    component_columns = {
        "state_reconstruction": "p90_reconstruction_ms",
        "csg_build": "p90_csg_build_ms",
        "tensorization": "p90_tensorization_ms",
        "action_projection": "p90_action_projection_ms",
        "host_to_device": "p90_gpu_transfer_ms",
        "csg_encoding": "p90_gpu_graph_encoding_ms",
        "target_scoring": "p90_gpu_action_scoring_ms",
    }
    for record in profile.to_dict("records"):
        components = {
            name: float(record[column]) for name, column in component_columns.items()
        }
        ordered = sorted(components, key=components.get, reverse=True)
        rows.append({
            "scale": record["scale"],
            "p90_single_model_end_to_end_ms": float(
                record["p90_total_single_model_decision_ms"]
            ),
            "p90_phase6e_ensemble_end_to_end_ms": float(
                record["p90_projected_ensemble_decision_ms"]
            ),
            "p90_model_forward_ms": float(record["p90_gpu_shared_forward_ms"]),
            "components_p90_ms": components,
            "largest_component": ordered[0],
            "second_largest_component": ordered[1],
        })

    l_scale = next(row for row in rows if row["scale"] == "L")
    checks = {
        "source_is_train_validation_only": True,
        "all_scales_present": len(rows) == 3,
        "phase6e_ensemble_fails_150ms": max(
            row["p90_phase6e_ensemble_end_to_end_ms"] for row in rows
        ) > 150.0,
        "single_model_forward_below_150ms": max(
            row["p90_model_forward_ms"] for row in rows
        ) <= 150.0,
        "l_preprocessing_is_primary_bottleneck": (
            l_scale["largest_component"] in {"csg_build", "state_reconstruction"}
        ),
    }
    payload = {
        "schema": "phase6f-phase6e-latency-bottleneck-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_profile_sha256": sha256_file(SOURCE),
        "source_summary_sha256": sha256_file(SOURCE_SUMMARY),
        "hardware": phase6e_summary["device"],
        "profile_split": "TRAIN_VALIDATION",
        "rows": rows,
        "diagnosis": {
            "primary_end_to_end_bottlenecks": ["csg_build", "state_reconstruction"],
            "phase6e_deployment_cost": "three sequential model forwards",
            "phase6f_actions": [
                "deploy one compact model",
                "preserve one-state-one-encoding all-action scoring",
                "profile model decision and end-to-end latency separately",
                "do not weaken CSG-1.0 semantics for speed",
            ],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)
    print("PHASE6F_LATENCY_BASELINE_AUDIT_PASS")


if __name__ == "__main__":
    main()
