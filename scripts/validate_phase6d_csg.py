#!/usr/bin/env python3
"""Balanced, resumable structural validation of CSG-1.0 on Phase 6C states."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg.actions import project_target_set
from rcias_clgri.csg.builder import build_csg
from rcias_clgri.csg.schema import EDGE_TYPE_ORDER, NODE_TYPE_ORDER
from rcias_clgri.csg.serialize import canonical_json
from rcias_clgri.csg.validate import validate_csg
from rcias_clgri.data.phase6c import reconstruct_state
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json


DEFAULT_MANIFEST = ROOT / "outputs/phase6c/manifests/state_manifest.csv"
DEFAULT_TRAIN_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
DEFAULT_DATASET = ROOT / "outputs/phase6c/dataset"
DEFAULT_OUTPUT = ROOT / "outputs/phase6d/validation"
BALANCE_COLUMNS = [
    "training_split", "scale", "CF_level", "RI_level", "TI_level", "search_stage",
]
ALLOWED_SPLITS = {"TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"}
SPLIT_DIRECTORIES = {
    "TRAIN": "train",
    "TRAIN_VALIDATION": "validation",
    "TRAIN_INTERNAL_HOLDOUT": "internal_holdout",
}


def _priority(seed: int, state_id: str) -> str:
    return hashlib.sha256(f"phase6d|{seed}|{state_id}".encode()).hexdigest()


def select_balanced_states(states: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Round-robin joint cells so every frozen balance factor is represented."""

    pool = states[states.training_split.isin(ALLOWED_SPLITS)].copy()
    if sample_size <= 0 or sample_size > len(pool):
        raise ValueError(f"sample_size must be in [1, {len(pool)}]")
    if pool[BALANCE_COLUMNS + ["state_id"]].isna().any().any():
        raise ValueError("balance columns contain missing values")
    pool["_priority"] = [
        _priority(seed, state_id) for state_id in pool.state_id.astype(str)
    ]
    pool["_cell_priority"] = [
        _priority(seed + 1, "|".join(str(row[column]) for column in BALANCE_COLUMNS))
        for row in pool.to_dict("records")
    ]
    pool = pool.sort_values(BALANCE_COLUMNS + ["_priority", "state_id"])
    pool["_cell_rank"] = pool.groupby(BALANCE_COLUMNS, sort=True).cumcount()
    selected = pool.sort_values(["_cell_rank", "_cell_priority", "_priority"]).head(sample_size)
    return selected.drop(columns=["_priority", "_cell_priority", "_cell_rank"]).reset_index(drop=True)


def _target_sets_by_state(
    selected: pd.DataFrame,
    dataset_root: Path,
) -> dict[str, list[tuple[str, list[str], str]]]:
    result: dict[str, list[tuple[str, list[str], str]]] = {}
    columns = ["state_id", "target_set_id", "destroyed_operation_ids", "arm_family"]
    for (split, instance_id), group in selected.groupby(["training_split", "instance_id"]):
        path = dataset_root / SPLIT_DIRECTORIES[str(split)] / str(instance_id) / "target_set_aggregates.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        state_ids = set(group.state_id.astype(str))
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame.state_id.astype(str).isin(state_ids)]
        for row in frame.itertuples(index=False):
            result.setdefault(str(row.state_id), []).append((
                str(row.target_set_id),
                list(json.loads(str(row.destroyed_operation_ids))),
                str(row.arm_family),
            ))
    missing = sorted(set(selected.state_id.astype(str)) - set(result))
    if missing:
        raise RuntimeError(f"selected states lack Phase6C target sets: {missing[:5]}")
    for records in result.values():
        records.sort(key=lambda item: item[0])
    return result


def _validate_one(payload: tuple[dict[str, Any], str, list[tuple[str, list[str], str]]]) -> dict[str, Any]:
    record, train_root_value, target_sets = payload
    started = time.perf_counter()
    state_id = str(record["state_id"])
    base = {
        key: record[key] for key in (
            "state_id", "instance_id", "training_split", "scale",
            "CF_level", "RI_level", "TI_level", "search_stage",
        )
    }
    try:
        reconstruction_started = time.perf_counter()
        reconstructed = reconstruct_state(record, Path(train_root_value))
        reconstruction_seconds = time.perf_counter() - reconstruction_started
        build_started = time.perf_counter()
        graph = build_csg(reconstructed, record)
        build_seconds = time.perf_counter() - build_started
        repeat_started = time.perf_counter()
        repeat = build_csg(reconstructed, record)
        repeat_build_seconds = time.perf_counter() - repeat_started
        deterministic = graph.graph_hash == repeat.graph_hash and canonical_json(graph) == canonical_json(repeat)

        action_started = time.perf_counter()
        action_hashes = []
        mapped_operations = 0
        for target_set_id, operations, arm_family in target_sets:
            view = project_target_set(
                graph,
                target_set_id,
                operations,
                {"arm_family": arm_family},
            )
            action_hashes.append(view.graph_hash)
            mapped_operations += len(view.target_operation_node_indices)
        action_invariant = bool(action_hashes) and set(action_hashes) == {graph.graph_hash}
        action_seconds = time.perf_counter() - action_started

        validation_started = time.perf_counter()
        validation = validate_csg(graph, reconstructed.instance, reconstructed.decoded.schedule)
        validation_seconds = time.perf_counter() - validation_started
        temporal = {}
        for edge_type, edges in graph.edges.items():
            gaps = [float(edge.features["temporal_gap"]) for edge in edges if "temporal_gap" in edge.features]
            temporal[edge_type] = {
                "count": len(gaps),
                "binding_count": sum(
                    float(edge.features.get("binding_indicator", 0.0)) == 1.0 for edge in edges
                ),
                "negative_count": sum(gap < -1e-9 for gap in gaps),
                "minimum_gap": min(gaps, default=0.0),
                "maximum_gap": max(gaps, default=0.0),
            }
        serialized_bytes = len(canonical_json(graph).encode())
        passed = validation.passed and deterministic and action_invariant
        return {
            **base,
            "passed": passed,
            "error": "",
            "violations": list(validation.violations),
            "checks": dict(validation.checks),
            "graph_hash": graph.graph_hash,
            "node_counts": {key: len(graph.nodes[key]) for key in NODE_TYPE_ORDER},
            "edge_counts": {key: len(graph.edges[key]) for key in EDGE_TYPE_ORDER},
            "temporal": temporal,
            "causal_depth": validation.metrics["causal_depth"],
            "deterministic_build": deterministic,
            "same_state_action_graph_invariant": action_invariant,
            "target_set_count": len(target_sets),
            "mapped_target_operation_count": mapped_operations,
            "serialized_bytes": serialized_bytes,
            "reconstruction_seconds": reconstruction_seconds,
            "build_seconds": build_seconds,
            "repeat_build_seconds": repeat_build_seconds,
            "action_projection_seconds": action_seconds,
            "validation_seconds": validation_seconds,
            "total_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            **base,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "violations": [],
            "checks": {},
            "graph_hash": "",
            "node_counts": {},
            "edge_counts": {},
            "temporal": {},
            "causal_depth": -1,
            "deterministic_build": False,
            "same_state_action_graph_invariant": False,
            "target_set_count": len(target_sets),
            "mapped_target_operation_count": 0,
            "serialized_bytes": 0,
            "reconstruction_seconds": 0.0,
            "build_seconds": 0.0,
            "repeat_build_seconds": 0.0,
            "action_projection_seconds": 0.0,
            "validation_seconds": 0.0,
            "total_seconds": time.perf_counter() - started,
        }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summary_rows(results: list[dict[str, Any]], key: str, names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        values = [int(row[key].get(name, 0)) for row in results]
        rows.append({
            "type": name,
            "state_count": len(values),
            "total_count": sum(values),
            "mean_count": sum(values) / max(len(values), 1),
            "minimum_count": min(values, default=0),
            "maximum_count": max(values, default=0),
        })
    return pd.DataFrame(rows)


def write_summaries(
    results: list[dict[str, Any]],
    selected: pd.DataFrame,
    output: Path,
    elapsed_seconds: float,
    seed: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(_summary_rows(results, "node_counts", NODE_TYPE_ORDER).rename(columns={"type": "node_type"}), output / "node_type_summary.csv")
    atomic_write_csv(_summary_rows(results, "edge_counts", EDGE_TYPE_ORDER).rename(columns={"type": "edge_type"}), output / "edge_type_summary.csv")

    temporal_rows = []
    for relation in EDGE_TYPE_ORDER:
        records = [row["temporal"].get(relation, {}) for row in results]
        temporal_rows.append({
            "edge_type": relation,
            "edge_count": sum(record.get("count", 0) for record in records),
            "binding_count": sum(record.get("binding_count", 0) for record in records),
            "negative_gap_count": sum(record.get("negative_count", 0) for record in records),
            "minimum_gap": min((record.get("minimum_gap", 0.0) for record in records), default=0.0),
            "maximum_gap": max((record.get("maximum_gap", 0.0) for record in records), default=0.0),
        })
    atomic_write_csv(pd.DataFrame(temporal_rows), output / "temporal_consistency_summary.csv")

    resource_rows = []
    for scale in ("S", "M", "L"):
        part = [row for row in results if row["scale"] == scale]
        resource_rows.append({
            "scale": scale,
            "state_count": len(part),
            "product_chain_pass_count": sum(row["checks"].get("product_chain_exact", False) for row in part),
            "island_chain_pass_count": sum(row["checks"].get("island_chain_exact", False) for row in part),
            "w_chain_pass_count": sum(row["checks"].get("w_chain_exact", False) for row in part),
            "f_chain_pass_count": sum(row["checks"].get("f_chain_exact", False) for row in part),
            "synchronization_pass_count": sum(
                row["checks"].get("w_synchronization_exact", False)
                and row["checks"].get("f_synchronization_exact", False)
                and row["checks"].get("reconfiguration_synchronization_exact", False)
                for row in part
            ),
        })
    atomic_write_csv(pd.DataFrame(resource_rows), output / "resource_chain_summary.csv")

    per_state = pd.DataFrame([{
        **{key: row[key] for key in (
            "state_id", "instance_id", "training_split", "scale", "CF_level",
            "RI_level", "TI_level", "search_stage", "passed", "error", "graph_hash",
            "causal_depth", "deterministic_build", "same_state_action_graph_invariant",
            "target_set_count", "mapped_target_operation_count", "serialized_bytes",
            "reconstruction_seconds", "build_seconds", "repeat_build_seconds",
            "action_projection_seconds", "validation_seconds", "total_seconds",
        )},
        "violations": json.dumps(row["violations"], separators=(",", ":")),
    } for row in results])
    atomic_write_csv(per_state, output / "per_state_validation.csv")

    check_names = sorted({name for row in results for name in row["checks"]})
    check_summary = pd.DataFrame([{
        "check": name,
        "pass_count": sum(row["checks"].get(name, False) for row in results),
        "state_count": len(results),
        "passed": all(row["checks"].get(name, False) for row in results),
    } for name in check_names])
    atomic_write_csv(check_summary, output / "structural_check_summary.csv")

    balance = {
        column: {str(key): int(value) for key, value in selected[column].value_counts().sort_index().items()}
        for column in BALANCE_COLUMNS
    }
    state_count = len(results)
    summary = {
        "schema": "phase6d-csg-validation-summary-v1",
        "csg_schema_version": "CSG-1.0",
        "selection_seed": seed,
        "requested_state_count": len(selected),
        "completed_state_count": state_count,
        "passed_state_count": sum(bool(row["passed"]) for row in results),
        "failed_state_count": sum(not bool(row["passed"]) for row in results),
        "balanced_factors": balance,
        "target_set_count": sum(int(row["target_set_count"]) for row in results),
        "mapped_target_operation_count": sum(int(row["mapped_target_operation_count"]) for row in results),
        "serialized_bytes": sum(int(row["serialized_bytes"]) for row in results),
        "elapsed_seconds": elapsed_seconds,
        "states_per_second": state_count / max(elapsed_seconds, 1e-12),
        "all_structural_checks_passed": bool(state_count) and all(bool(row["passed"]) for row in results),
        "deterministic_build_passed": bool(state_count) and all(bool(row["deterministic_build"]) for row in results),
        "same_state_action_graph_invariance_passed": bool(state_count) and all(bool(row["same_state_action_graph_invariant"]) for row in results),
        "phase6c_state_reconstruction_reused": True,
    }
    atomic_write_json(summary, output / "csg_validation_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=667_006_004)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    states = pd.read_csv(args.manifest)
    selected = select_balanced_states(states, args.sample_size, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(selected, args.output / "selected_state_manifest.csv")
    target_sets = _target_sets_by_state(selected, args.dataset_root)
    progress_path = args.output / "validation_progress.jsonl"
    previous = _read_jsonl(progress_path) if args.resume else []
    completed_ids = {str(row["state_id"]) for row in previous}
    pending = [
        (
            {key: value.item() if hasattr(value, "item") else value for key, value in row.items()},
            str(args.train_root),
            target_sets[str(row["state_id"])],
        )
        for row in selected.to_dict("records")
        if str(row["state_id"]) not in completed_ids
    ]
    if not args.resume:
        _write_jsonl(progress_path, (), "w")
    started = time.perf_counter()
    new_results: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = map(_validate_one, pending)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_validate_one, pending, chunksize=1)
    try:
        for index, result in enumerate(iterator, start=1):
            new_results.append(result)
            _write_jsonl(progress_path, (result,), "a")
            completed = len(previous) + index
            if completed % args.progress_every == 0 or completed == len(selected):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-12)
                print(
                    f"completed={completed}/{len(selected)} new_rate={rate:.3f} states/s "
                    f"failed={sum(not row['passed'] for row in new_results)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    results_by_id = {str(row["state_id"]): row for row in [*previous, *new_results]}
    results = [results_by_id[str(state_id)] for state_id in selected.state_id]
    elapsed = sum(float(row["total_seconds"]) for row in results) / max(args.workers, 1)
    summary = write_summaries(results, selected, args.output, elapsed, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_structural_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
