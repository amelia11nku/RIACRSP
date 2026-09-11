#!/usr/bin/env python3
"""Validate compact-runtime capacity bounds on all frozen A1.6R instances."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.search.common import candidate_from_actions, decode_candidate  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import digest, load_json  # noqa: E402
from rcias_ngas.rng import RNGStreams  # noqa: E402
from rcias_ngas.runtime import CompactStateBuilder  # noqa: E402


CONFIG = ROOT / 'configs/ngas_a16r_integrity_diagnostic_v1.json'
OUT = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1/preregistration/capacity_audit.json'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main() -> None:
    config = load_json(CONFIG)
    manifest_path = ROOT / config['scope']['instance_manifest_path']
    manifest = load_json(manifest_path)
    rows = []
    for item in manifest['instances']:
        instance_path = ROOT / config['scope']['instance_root'] / item['relative_path']
        if digest(instance_path) != item['sha256']:
            raise RuntimeError(f"instance changed: {item['instance_id']}")
        instance = load_instance(instance_path)
        h1 = solve_dispatching(instance, 'H1')
        current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
        builder = CompactStateBuilder(instance)
        built = builder.build(
            current, f"{item['instance_id']}:a16r-capacity",
            RNGStreams(item['instance_id'], 746101))
        audit = builder.capacity_audit()
        observed = {
            'neural_nodes': built.node_count,
            'forward_edges': built.forward_edge_count,
            'targets': built.target_count,
            'actions': len(built.actions),
        }
        if any(observed[name] > audit['allocated_capacities'][name]
               for name in observed):
            raise RuntimeError(f"capacity failure: {item['instance_id']}")
        rows.append({
            'instance_id': item['instance_id'],
            'scale': item['scale'],
            'observed_h1': observed,
            'capacity': audit,
        })
    checks = {
        'scope_exact_18': len(rows) == 18,
        'all_bounds_equal_allocations': all(
            row['capacity']['required_upper_bounds']
            == row['capacity']['allocated_capacities'] for row in rows),
        'all_h1_observations_within_bounds': all(
            all(value <= row['capacity']['allocated_capacities'][name]
                for name, value in row['observed_h1'].items()) for row in rows),
        'no_resize_events': all(row['capacity']['resize_events'] == 0 for row in rows),
        'no_silent_truncation': all(
            not row['capacity']['silent_truncation_allowed'] for row in rows),
    }
    payload = {
        'schema': 'ngas-a16r-capacity-preflight-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'instance_manifest_sha256': digest(manifest_path),
        'checks': checks,
        'instances': rows,
        'scope_boundary': 'current frozen 18-instance R12 development scope only',
        'r13_r14_authorized': False,
    }
    atomic_json(OUT, payload)
    print(json.dumps({'status': payload['status'], 'instances': len(rows),
                      'path': str(OUT.relative_to(ROOT))}, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
