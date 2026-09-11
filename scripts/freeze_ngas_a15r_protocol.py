#!/usr/bin/env python3
"""Freeze the A1.5R formal boundary before latency outcomes exist."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'
CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
PREFORMAL = OUT / 'audit/preformal_audit.json'
SOURCES = (
    'rcias_ngas/runtime/compact_state.py',
    'rcias_ngas/runtime/production_refresh.py',
    'rcias_ngas/bank/ngas_bank_v1.py',
    'rcias_ngas/search/ngas_solver.py',
    'rcias_ngas/critic/inference.py',
    'rcias_ngas/critic/revised_critic.py',
    'rcias_ngas/critic/encoders/rt_hgt.py',
    'rcias_ngas/csg/revised_features.py',
    'rcias_ngas/search/persistent_prior.py',
    'scripts/audit_ngas_a15r_equivalence.py',
    'scripts/audit_ngas_a15r_search_integration.py',
    'scripts/run_ngas_a15r_formal.py',
    'scripts/run_ngas_a15r_transition_trace.py',
    'tests/ngas/test_a15r_runtime.py',
    'tests/ngas/test_a14_search_integration.py',
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if PROTOCOL.exists():
        raise RuntimeError('A1.5R formal protocol already exists; refusing to rewrite it')
    if list((OUT / 'formal/raw').glob('*.json')):
        raise RuntimeError('Formal outcomes exist before protocol freeze')
    config = json.loads(CONFIG.read_text())
    preformal = json.loads(PREFORMAL.read_text())
    if preformal['status'] != 'PASS' or preformal['config_sha256'] != digest(CONFIG):
        raise RuntimeError('Preformal A1.5R audit is absent, failed, or stale')
    states = ROOT / config['historical_A1_5']['representative_states']
    checkpoint = ROOT / config['production_checkpoint']['path']
    historical = ROOT / config['historical_A1_5']['completion_audit']
    payload = {
        'schema': 'ngas-a15r-formal-protocol-v1',
        'status': 'FROZEN_BEFORE_FORMAL_RESULTS',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_base_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config_path': str(CONFIG.relative_to(ROOT)),
        'config_sha256': digest(CONFIG),
        'representative_states_path': str(states.relative_to(ROOT)),
        'representative_states_sha256': digest(states),
        'checkpoint_path': str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256': digest(checkpoint),
        'historical_A1_5_completion_audit_path': str(historical.relative_to(ROOT)),
        'historical_A1_5_completion_audit_sha256': digest(historical),
        'preformal_audit_path': str(PREFORMAL.relative_to(ROOT)),
        'preformal_audit_sha256': digest(PREFORMAL),
        'source_hashes': {path: digest(ROOT / path) for path in SOURCES},
        'measurement': {
            'device': 'cuda:0', 'cuda_device_class': 'NVIDIA GeForce RTX 4060 Ti',
            'normal_gc': True, 'warmup_complete_refreshes': 30,
            'complete_refresh_repetitions_per_state': 200,
            'sample_seed': 746101, 'percentile_method': 'linear',
            'authoritative_metric': 'complete_refresh_wall_ms',
        },
        'primary_criterion': {
            'threshold_ms': 30.,
            'per_state': ['S', 'M', 'L', 'L_MAX'],
            'require_each_state_p90': True, 'require_pooled_p90': True,
        },
        'equivalence_tolerance': 1e-7,
        'transition_trace': {'seed': 746103, 'distinct_states_per_scale': 20},
        'decision_states': {
            'pass': 'NGAS_A15R_RUNTIME_PASS',
            'latency_fail_with_semantics_intact': 'NGAS_A15R_REVISE_RUNTIME',
            'semantic_failure': 'NGAS_A15R_SEMANTIC_HARD_FAILURE',
        },
        'historical_A1_5_immutable': True,
        'locks': {'A1_6': 'LOCKED_UNTIL_A15R_PASS',
                  'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi': False},
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROTOCOL.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(PROTOCOL)
    print(json.dumps({
        'protocol': str(PROTOCOL.relative_to(ROOT)),
        'protocol_sha256': digest(PROTOCOL),
        'status': payload['status'],
    }, indent=2))


if __name__ == '__main__':
    main()
