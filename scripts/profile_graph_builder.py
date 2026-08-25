#!/usr/bin/env python3
"""Profile graph construction and verify linear hierarchical probing."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import write_json
from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state
from rcias_clgri.graph.candidates import (
    CandidateFeatureExtractor,
    get_f_candidate_features,
    get_island_candidate_features,
    get_operation_candidate_features,
    get_w_candidate_features,
)


INSTANCE_PATHS = (
    Path("instances/tiny/tiny_01.json"),
    Path("instances/canonical/RCIAS-2.0/brandimarte/BR_Mk01.json"),
    Path("instances/canonical/RCIAS-2.0/brandimarte/BR_Mk05.json"),
    Path("instances/canonical/RCIAS-2.0/brandimarte/BR_Mk10.json"),
    Path("instances/canonical/RCIAS-2.0/hurink/edata/HU_E_la01.json"),
    Path("instances/canonical/RCIAS-2.0/hurink/rdata/HU_R_la01.json"),
    Path("instances/canonical/RCIAS-2.0/hurink/vdata/HU_V_la01.json"),
)


def profile(path: Path) -> dict[str, object]:
    instance = load_instance(path)
    schedule = InsertionDecoder(instance).empty_schedule()
    extractor = CandidateFeatureExtractor(instance, schedule)
    candidate_start = perf_counter()
    operation_features = get_operation_candidate_features(extractor)
    island_count = 0
    w_count = 0
    f_count = 0
    for op_id in instance.operations:
        islands = get_island_candidate_features(extractor, op_id)
        if op_id not in extractor.ready_operations:
            continue
        island_count += len(islands)
        for island_id in islands:
            w_count += len(get_w_candidate_features(extractor, op_id, island_id))
            f_count += len(get_f_candidate_features(extractor, op_id, island_id))
    candidate_seconds = perf_counter() - candidate_start

    graph_start = perf_counter()
    graph = build_graph_state(instance, schedule)
    graph_seconds = perf_counter() - graph_start
    ready_pairs = sum(
        len(instance.operation_data[op_id].eligible_islands)
        for op_id in extractor.ready_operations
    )
    linear_bound = ready_pairs * (len(instance.agvs_w) + len(instance.agvs_f) + 1)
    stats = extractor.stats.to_dict()
    if stats["total_probes"] > linear_bound:
        raise RuntimeError(f"probe bound violated for {instance.instance_id}")
    return {
        "instance_id": instance.instance_id,
        "path": path.as_posix(),
        "nodes": sum(len(nodes) for nodes in graph.node_features.values()),
        "edges": len(graph.edges),
        "ready_operations": len(extractor.ready_operations),
        "candidate_actions": len(extractor.ready_operations) + island_count + w_count + f_count,
        "operation_feature_rows": len(operation_features),
        "island_candidate_rows": island_count,
        "w_candidate_rows": w_count,
        "f_candidate_rows": f_count,
        "graph_build_seconds": graph_seconds,
        "candidate_feature_seconds": candidate_seconds,
        "probe_counts": stats,
        "linear_probe_bound": linear_bound,
        "legacy_cartesian_transport_combinations": (
            ready_pairs * len(instance.agvs_w) * len(instance.agvs_f)
        ),
        "hierarchical_transport_probes": stats["w_probes"] + stats["f_probes"],
        "uses_cartesian_w_f_probing": False,
        "linear_probe_bound_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/profiling/graph_builder_profile.json"),
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = tuple(args.paths) or INSTANCE_PATHS
    print("This run measures graph/candidate cost and verifies the O(|W|+|F|) probe bound.")
    records = []
    for path in paths:
        record = profile(path)
        records.append(record)
        print(
            f"{record['instance_id']}: nodes={record['nodes']} edges={record['edges']} "
            f"ready={record['ready_operations']} candidates={record['candidate_actions']} "
            f"graph={record['graph_build_seconds']:.4f}s "
            f"candidate={record['candidate_feature_seconds']:.4f}s "
            f"probes={record['probe_counts']['total_probes']}/{record['linear_probe_bound']}"
        )
    payload = {
        "profile_version": "RCIAS-graph-profile-1.0",
        "semantics": "ready-only dynamic features; hierarchical W plus F probing",
        "all_linear_probe_bounds_pass": all(item["linear_probe_bound_pass"] for item in records),
        "instances": records,
    }
    write_json(payload, args.output)
    print(f"GRAPH_PROFILE_VALID = TRUE | wrote {args.output}")


if __name__ == "__main__":
    main()
