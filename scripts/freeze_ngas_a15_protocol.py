#!/usr/bin/env python3
"""Freeze representative states and the outcome-blind NGAS A1.5 protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import Candidate, candidate_from_actions, decode_candidate
from rcias_ngas.evaluation.bks import content_hash

CONFIG = ROOT / 'configs/ngas_a1_latency_qualification_v1.json'
A14_AUDIT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1/audit/completion_audit.json'
SMOKE = ROOT / 'outputs/ngas_a1/latency_qualification_v1/smoke/pre_freeze_gpu_smoke.json'
OUT = ROOT / 'outputs/ngas_a1/latency_qualification_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
STATES = OUT / 'preregistration/representative_states.json'
PROGRESS = OUT / 'progress.json'
SOURCE_PATHS = (
    'rcias_clgri/csg/builder.py',
    'rcias_ngas/csg/critical_sync.py',
    'rcias_ngas/csg/revised_features.py',
    'rcias_ngas/latency/live_refresh.py',
    'scripts/run_ngas_a15_latency.py',
    'scripts/finalize_ngas_a15_latency.py',
)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def candidate_payload(candidate: Candidate) -> dict:
    return {
        'operation_order': list(candidate.operation_order),
        'island_assignment': list(candidate.island_assignment),
        'w_assignment': list(candidate.w_assignment),
        'f_assignment': list(candidate.f_assignment),
    }


def main() -> None:
    if PROTOCOL.exists() or STATES.exists():
        raise RuntimeError('A1.5 protocol boundary already exists')
    config = json.loads(CONFIG.read_text())
    audit = json.loads(A14_AUDIT.read_text())
    smoke = json.loads(SMOKE.read_text())
    if audit.get('terminal_decision') != 'NGAS_A1_4_PASS_C1_FIXED_REFRESH':
        raise RuntimeError('A1.4 completion audit does not unlock A1.5')
    if smoke.get('status') != 'PASS':
        raise RuntimeError('Pre-freeze A1.5 GPU smoke did not pass')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if smoke.get('implementation_commit') != head:
        raise RuntimeError('GPU smoke does not identify the current implementation commit')
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f'Missing A1.5 implementation sources: {missing}')

    states = []
    for spec in config['representative_states']:
        instance_path = ROOT / config['instance_root'] / spec['relative_path']
        if digest(instance_path) != spec['instance_sha256']:
            raise RuntimeError(f"Instance hash mismatch: {spec['instance_id']}")
        instance = load_instance(instance_path)
        if instance.instance_id != spec['instance_id'] or instance.num_operations != spec['num_operations']:
            raise RuntimeError('Representative instance identity mismatch')
        if spec['candidate_source'] == 'DETERMINISTIC_H1':
            solution = solve_dispatching(instance, 'H1')
            candidate = candidate_from_actions(instance, solution.actions)
            source_hash = None
        else:
            source_path = ROOT / spec['candidate_source']
            source = json.loads(source_path.read_text())
            raw = source[spec['candidate_field']]
            candidate = Candidate(*(tuple(raw[name]) for name in (
                'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))
            source_hash = digest(source_path)
        decoded = decode_candidate(instance, candidate)
        if not decoded.feasible:
            raise RuntimeError(f"Representative state is infeasible: {spec['label']}")
        frozen_candidate = candidate_payload(candidate)
        states.append({
            **spec,
            'candidate_source_sha256': source_hash,
            'candidate': frozen_candidate,
            'candidate_sha256': content_hash(frozen_candidate),
            'replayed_makespan': decoded.makespan,
            'replay_feasible': decoded.feasible,
        })
    state_manifest = {
        'schema': 'ngas-a15-representative-states-v1',
        'created_before_formal_timing': True,
        'states': states,
    }
    atomic_json(STATES, state_manifest)

    for item in [config['production'], *config['development_ensemble']]:
        if digest(ROOT / item['checkpoint_path' if 'checkpoint_path' in item else 'path']) \
                != item['checkpoint_sha256' if 'checkpoint_sha256' in item else 'sha256']:
            raise RuntimeError('A1.5 checkpoint hash mismatch')
    protocol = {
        'schema': 'ngas-a15-latency-protocol-v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': head,
        'config_path': str(CONFIG.relative_to(ROOT)),
        'config_sha256': digest(CONFIG),
        'A1_4_completion_audit_path': str(A14_AUDIT.relative_to(ROOT)),
        'A1_4_completion_audit_sha256': digest(A14_AUDIT),
        'pre_freeze_smoke_path': str(SMOKE.relative_to(ROOT)),
        'pre_freeze_smoke_sha256': digest(SMOKE),
        'representative_states_path': str(STATES.relative_to(ROOT)),
        'representative_states_sha256': digest(STATES),
        'source_hashes': {path: digest(ROOT / path) for path in SOURCE_PATHS},
        'production': config['production'],
        'development_ensemble': config['development_ensemble'],
        'measurement': config['measurement'],
        'equivalence_gate': config['equivalence_gate'],
        'latency_gate': config['latency_gate'],
        'formal_result_scope': {'states': 4, 'modes': ['SINGLE_C1', 'TEACHER_ENSEMBLE'], 'files': 8},
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda': smoke['device']['cuda'],
            'device_name': smoke['device']['name'],
        },
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
        'failure_decision': 'NGAS_A1_REVISE_RUNTIME',
        'a1_6_status_before_audit': 'LOCKED',
    }
    atomic_json(PROTOCOL, protocol)
    protocol_sha256 = digest(PROTOCOL)
    atomic_json(PROGRESS, {
        'schema': 'ngas-a15-latency-progress-v1',
        'status': 'PROTOCOL_FROZEN',
        'completed_units': 0,
        'expected_units': 8,
        'protocol_sha256': protocol_sha256,
        'decision': 'PENDING_FORMAL_MEASUREMENT',
        'next_gate': 'A1_5_FORMAL_LATENCY_MEASUREMENT',
        'A1_6': 'LOCKED', 'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    })
    print(json.dumps({'protocol': str(PROTOCOL.relative_to(ROOT)),
                      'protocol_sha256': protocol_sha256,
                      'representative_states_sha256': digest(STATES)}, indent=2))


if __name__ == '__main__':
    main()
