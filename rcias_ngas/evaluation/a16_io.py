"""Atomic immutable output helpers for NGAS A1.6."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_new_json(path: Path, payload: object) -> None:
    """Atomically create a JSON file and refuse to replace any existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_raw_contract(payload: dict, task: dict, protocol_sha256: str) -> None:
    expected = {
        'schema': 'ngas-a16-formal-run-v1',
        'status': 'COMPLETE',
        'protocol_sha256': protocol_sha256,
        'algorithm_id': 'NGAS_A1_6',
        'instance_id': task['instance_id'],
        'instance_sha256': task['instance_sha256'],
        'seed': task['seed'],
        'budget_seconds': task['budget_seconds'],
        'budget_accounting': 'A16_INSTANCE_TOTAL',
        'feasible': True,
        'r13_accessed': False,
        'r14_accessed': False,
        'gurobi_run': False,
    }
    mismatches = {
        key: {'expected': value, 'actual': payload.get(key)}
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError({'invalid_a16_raw_contract': mismatches})
