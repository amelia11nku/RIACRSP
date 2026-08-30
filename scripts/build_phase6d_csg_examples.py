#!/usr/bin/env python3
"""Export seven focused, human-auditable CSG-1.0 examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg.builder import build_csg
from rcias_clgri.csg.diagnostics import export_csg_neighborhood, graph_diagnostics
from rcias_clgri.csg.serialize import export_csg_tables
from rcias_clgri.csg.validate import validate_csg
from rcias_clgri.data.phase6c import reconstruct_state
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json


MANIFEST = ROOT / "outputs/phase6c/manifests/state_manifest.csv"
TRAIN_ROOT = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
DEFAULT_OUTPUT = ROOT / "outputs/phase6d/examples"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def select_examples(states: pd.DataFrame) -> list[tuple[str, dict]]:
    pool = states[states.training_split.isin({"TRAIN_VALIDATION", "TRAIN_INTERNAL_HOLDOUT"})]
    specifications = (
        ("small_state", "scale", "S"),
        ("medium_state", "scale", "M"),
        ("large_state", "scale", "L"),
        ("reconfiguration_pressure", "bottleneck_proxy", "RECONFIGURATION"),
        ("w_logistics_heavy", "bottleneck_proxy", "W_LOGISTICS"),
        ("f_logistics_heavy", "bottleneck_proxy", "F_LOGISTICS"),
        ("cross_resource_synchronization", "bottleneck_proxy", "CROSS_RESOURCE_SYNCHRONIZATION"),
    )
    selected: list[tuple[str, dict]] = []
    used: set[str] = set()
    for role, column, value in specifications:
        eligible = pool[(pool[column] == value) & ~pool.state_id.astype(str).isin(used)]
        if eligible.empty:
            raise RuntimeError(f"no unused example for {role}")
        row = eligible.sort_values(["state_id"]).iloc[0].to_dict()
        used.add(str(row["state_id"]))
        selected.append((role, row))
    return selected


def _timeline_rows(reconstructed) -> list[dict]:
    schedule = reconstructed.decoded.schedule
    rows = []
    for operation, record in schedule.operation_schedules.items():
        rows.append({
            "event_type": "OP", "event_key": operation, "operation_id": operation,
            "resource_type": "ISLAND", "resource_key": record.island_id,
            "start_time": record.start_time, "end_time": record.completion_time,
            "product_id": record.product_id, "configuration": record.config_id,
        })
        if record.reconfiguration_end - record.reconfiguration_start > 1e-9:
            rows.append({
                "event_type": "RECONF_EVENT", "event_key": f"R:{operation}", "operation_id": operation,
                "resource_type": "ISLAND", "resource_key": record.island_id,
                "start_time": record.reconfiguration_start, "end_time": record.reconfiguration_end,
                "product_id": record.product_id, "configuration": record.config_id,
            })
    for resource, tasks in schedule.w_timelines.items():
        rows.extend({
            "event_type": "W_EVENT", "event_key": task.task_id, "operation_id": task.operation_id,
            "resource_type": "W_AGV", "resource_key": resource,
            "start_time": task.empty_start, "end_time": task.arrival_time,
            "product_id": task.product_id, "configuration": "",
        } for task in tasks)
    for resource, tasks in schedule.f_timelines.items():
        rows.extend({
            "event_type": "F_EVENT", "event_key": task.task_id, "operation_id": task.operation_id,
            "resource_type": "F_AGV", "resource_key": resource,
            "start_time": task.departure_wh, "end_time": task.return_wh,
            "product_id": "", "configuration": "",
        } for task in tasks)
    return sorted(rows, key=lambda row: (row["start_time"], row["event_type"], row["event_key"]))


def _relation_rows(graph) -> list[dict]:
    rows = []
    for edge_type, edges in graph.edges.items():
        gaps = [float(edge.features["temporal_gap"]) for edge in edges if "temporal_gap" in edge.features]
        rows.append({
            "edge_type": edge_type,
            "edge_class": next((edge.edge_class for edge in edges), "SCHEMA_DEFINED_EMPTY"),
            "edge_count": len(edges),
            "binding_count": sum(float(edge.features.get("binding_indicator", 0.0)) == 1.0 for edge in edges),
            "minimum_gap": min(gaps, default=0.0),
            "maximum_gap": max(gaps, default=0.0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()
    states = pd.read_csv(args.manifest)
    manifest_rows = []
    for role, record in select_examples(states):
        reconstructed = reconstruct_state(record, args.train_root)
        graph = build_csg(reconstructed, record)
        validation = validate_csg(graph, reconstructed.instance, reconstructed.decoded.schedule)
        if not validation.passed:
            raise RuntimeError(f"{role}: {validation.violations}")
        directory = args.output / _slug(role)
        export_csg_tables(graph, directory / "tables")
        atomic_write_csv(pd.DataFrame(_timeline_rows(reconstructed)), directory / "timeline_reference.csv")
        atomic_write_csv(pd.DataFrame(_relation_rows(graph)), directory / "relation_summary.csv")
        focus = min(
            graph.nodes["OP"],
            key=lambda node: (node.features["operation_slack"], -node.features["completion_time"], node.key),
        ).key
        export_csg_neighborhood(graph, focus, directory / "critical_operation_neighborhood.dot", hops=args.hops)
        diagnostics = graph_diagnostics(graph)
        atomic_write_json(diagnostics, directory / "graph_diagnostics.json")
        manifest_rows.append({
            "role": role,
            "state_id": graph.state_id,
            "instance_id": graph.instance_id,
            "training_split": record["training_split"],
            "scale": record["scale"],
            "CF_level": record["CF_level"],
            "RI_level": record["RI_level"],
            "TI_level": record["TI_level"],
            "search_stage": record["search_stage"],
            "bottleneck_proxy": graph.graph_categories["bottleneck_proxy"],
            "focus_operation": focus,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "graph_hash": graph.graph_hash,
            "validation_passed": validation.passed,
            "directory": str(directory.relative_to(ROOT)),
        })
    atomic_write_csv(pd.DataFrame(manifest_rows), args.output / "example_manifest.csv")
    atomic_write_json({
        "schema": "phase6d-csg-examples-v1",
        "example_count": len(manifest_rows),
        "all_validation_passed": all(row["validation_passed"] for row in manifest_rows),
        "roles": [row["role"] for row in manifest_rows],
    }, args.output / "examples_summary.json")
    print("PHASE6D_CSG_EXAMPLES_COMPLETE")


if __name__ == "__main__":
    main()
