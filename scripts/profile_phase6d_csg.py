#!/usr/bin/env python3
"""Profile CSG construction, validation, serialization, and memory on S/M/L states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg.builder import build_csg
from rcias_clgri.csg.serialize import canonical_json
from rcias_clgri.csg.validate import validate_csg
from rcias_clgri.data.phase6c import reconstruct_state
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json


MANIFEST = ROOT / "outputs/phase6c/manifests/state_manifest.csv"
TRAIN_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
DEFAULT_OUTPUT = ROOT / "outputs/phase6d/profiling"


def select_states(frame: pd.DataFrame, per_scale: int) -> pd.DataFrame:
    pool = frame[frame.training_split == "TRAIN_VALIDATION"].sort_values("state_id")
    selected = pd.concat([
        pool[pool.scale == scale].head(per_scale) for scale in ("S", "M", "L")
    ], ignore_index=True)
    if len(selected) != per_scale * 3:
        raise RuntimeError("insufficient S/M/L states for profiling")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-scale", type=int, default=10)
    args = parser.parse_args()
    selected = select_states(pd.read_csv(args.manifest), args.per_scale)
    rows = []
    for index, record in enumerate(selected.to_dict("records"), start=1):
        reconstruct_started = time.perf_counter()
        reconstructed = reconstruct_state(record, args.train_root)
        reconstruction_seconds = time.perf_counter() - reconstruct_started
        instance = reconstructed.instance
        precedence_count = sum(len(instance.predecessors[operation]) for operation in instance.operations)
        eligibility_count = sum(len(instance.operation_data[operation].eligible_islands) for operation in instance.operations)
        capability_count = sum(len(instance.island_data[island].supported_configs) for island in instance.islands)
        tracemalloc.start()
        build_started = time.perf_counter()
        graph = build_csg(reconstructed, record)
        build_seconds = time.perf_counter() - build_started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        validation_started = time.perf_counter()
        validation = validate_csg(graph, instance, reconstructed.decoded.schedule)
        validation_seconds = time.perf_counter() - validation_started
        serialization_started = time.perf_counter()
        serialized_bytes = len(canonical_json(graph).encode())
        serialization_seconds = time.perf_counter() - serialization_started
        rows.append({
            "state_id": graph.state_id,
            "instance_id": graph.instance_id,
            "scale": record["scale"],
            "operation_count": len(instance.operations),
            "precedence_edge_count": precedence_count,
            "eligibility_edge_count": eligibility_count,
            "capability_edge_count": capability_count,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "reconstruction_seconds": reconstruction_seconds,
            "construction_seconds": build_seconds,
            "validation_seconds": validation_seconds,
            "serialization_seconds": serialization_seconds,
            "serialized_bytes": serialized_bytes,
            "peak_traced_bytes": peak_bytes,
            "validation_passed": validation.passed,
        })
        print(f"profiled={index}/{len(selected)} scale={record['scale']}", flush=True)
    detail = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(detail, args.output / "profiling_per_state.csv")
    summary = detail.groupby("scale", sort=False).agg(
        state_count=("state_id", "count"),
        mean_operation_count=("operation_count", "mean"),
        mean_node_count=("node_count", "mean"),
        mean_edge_count=("edge_count", "mean"),
        mean_construction_seconds=("construction_seconds", "mean"),
        maximum_construction_seconds=("construction_seconds", "max"),
        mean_validation_seconds=("validation_seconds", "mean"),
        mean_serialized_bytes=("serialized_bytes", "mean"),
        maximum_peak_traced_bytes=("peak_traced_bytes", "max"),
        validation_pass_rate=("validation_passed", "mean"),
    ).reset_index()
    atomic_write_csv(summary, args.output / "profiling_summary.csv")

    design = np.column_stack([
        np.ones(len(detail)),
        detail.operation_count,
        detail.precedence_edge_count,
        detail.eligibility_edge_count,
        detail.capability_edge_count,
    ])
    coefficients, _, _, _ = np.linalg.lstsq(design, detail.construction_seconds, rcond=None)
    prediction = design @ coefficients
    residual = np.square(detail.construction_seconds.to_numpy() - prediction).sum()
    total = np.square(detail.construction_seconds.to_numpy() - detail.construction_seconds.mean()).sum()
    model = {
        "schema": "phase6d-csg-complexity-profile-v1",
        "expected_complexity": "O(|V_op| + |E_precedence| + |E_eligibility| + |E_capability| + |E_realized_events| + |E_resource_chain|)",
        "implementation_scan_note": "Product/island/event positions and configuration counts are pre-indexed; construction traverses canonical nodes and relations without operation-pair scans.",
        "linear_model_coefficients_seconds": {
            "intercept": float(coefficients[0]),
            "operation_count": float(coefficients[1]),
            "precedence_edge_count": float(coefficients[2]),
            "eligibility_edge_count": float(coefficients[3]),
            "capability_edge_count": float(coefficients[4]),
        },
        "linear_model_r_squared": float(1.0 - residual / total) if total > 0 else 1.0,
        "all_validation_passed": bool(detail.validation_passed.all()),
        "profiled_state_count": len(detail),
    }
    atomic_write_json(model, args.output / "complexity_model.json")
    if not model["all_validation_passed"]:
        raise SystemExit(1)
    print("PHASE6D_CSG_PROFILING_COMPLETE")


if __name__ == "__main__":
    main()
