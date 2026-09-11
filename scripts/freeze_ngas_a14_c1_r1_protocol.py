#!/usr/bin/env python3
"""Freeze the predeclared matched C1/R1 explanatory protocol."""
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
PRIMARY = ROOT / 'outputs/ngas_a1/search_integration_v1'
OUT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1'
SMOKE = OUT / 'smoke/smoke.json'
PROTOCOL = OUT / 'preregistration/protocol.json'
SOURCES = (
    'scripts/run_ngas_a14_c1_r1_ablation.py',
    'scripts/smoke_ngas_a14_c1_r1.py',
    'scripts/freeze_ngas_a14_c1_r1_protocol.py',
    'scripts/launch_ngas_a14_c1_r1.py',
)
CHECKPOINTS = {
    'C1': {
        '746101': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/C1/seed_746101/fold_0/model.pt', '3cd419ada2287d377c194e24adb0b8bf29226ab9e7195e5d1d6704cba3ec591a'),
        '746102': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/C1/seed_746102/fold_0/model.pt', '119347f8db7945c40c282dde094903b79ab12c57a7f300dc5c1b606c66824a5e'),
        '746103': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/C1/seed_746103/fold_0/model.pt', 'be7499fdf8727ec11432005d4a51196bfb998f1f31512872b57b05f1429b63a5'),
    },
    'R1': {
        '746101': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/R1/seed_746101/fold_0/model.pt', '532f9aa448d26c7ffd4e238eaa20daa57ae1eab50b20a405dbc21bbb936dee4f'),
        '746102': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/R1/seed_746102/fold_0/model.pt', '66167e7b0e9f3a510337562d9c1a64e4cec9aeb4ded09d8cd68e96e5fce2aaed'),
        '746103': ('outputs/ngas_a1/critic_training_rthgt_v2/variants/R1/seed_746103/fold_0/model.pt', '8cfefcbe9f93878c9c30d7804486f8b79fb554062e6d06d3efcc2121fcec59aa'),
    },
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(f'C1/R1 protocol already exists: {PROTOCOL}')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('C1/R1 freeze requires a clean committed worktree')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    smoke = json.loads(SMOKE.read_text())
    if smoke['status'] != 'PASS' or smoke['implementation_commit'] != head:
        raise RuntimeError('C1/R1 CUDA smoke does not match the implementation commit')
    primary_protocol_path = PRIMARY / 'preregistration/protocol.json'
    primary_decision_path = PRIMARY / 'mechanism_decision.json'
    primary_audit_path = PRIMARY / 'audit/completion_audit.json'
    primary_protocol = json.loads(primary_protocol_path.read_text())
    primary_decision = json.loads(primary_decision_path.read_text())
    primary_audit = json.loads(primary_audit_path.read_text())
    if (primary_decision['selected_mode'] != 'PERSISTENT_FIXED_REFRESH'
            or primary_audit['status'] != 'PASS'):
        raise RuntimeError('Primary A1.4 mechanism boundary is not eligible')
    checkpoints = {}
    for variant, by_seed in CHECKPOINTS.items():
        checkpoints[variant] = {}
        for seed, (relative, expected) in by_seed.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'C1/R1 checkpoint drift: {relative}')
            checkpoints[variant][seed] = {
                'path': relative, 'sha256': expected, 'held_fold': 0,
                'role': 'PRODUCTION' if variant == 'C1' else 'EXPLANATORY_ONLY',
            }
    protocol = {
        'schema': 'ngas-a14-c1-r1-protocol-v1',
        'status': 'FROZEN_BEFORE_OUTCOMES',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': head,
        'primary_protocol_path': str(primary_protocol_path.relative_to(ROOT)),
        'primary_protocol_sha256': digest(primary_protocol_path),
        'primary_decision_path': str(primary_decision_path.relative_to(ROOT)),
        'primary_decision_sha256': digest(primary_decision_path),
        'primary_completion_audit_path': str(primary_audit_path.relative_to(ROOT)),
        'primary_completion_audit_sha256': digest(primary_audit_path),
        'smoke_path': str(SMOKE.relative_to(ROOT)), 'smoke_sha256': digest(SMOKE),
        'ablation_source_hashes': {relative: digest(ROOT / relative) for relative in SOURCES},
        'selected_mechanism': 'PERSISTENT_FIXED_REFRESH',
        'variants': ['C1', 'R1'], 'seeds': primary_protocol['seeds'],
        'instances': list(primary_protocol['instance_hashes']),
        'run_order': 'seed_then_instance_then_C1_R1',
        'expected_runs': 18,
        'budget': primary_protocol['budget'],
        'search_config_sha256': primary_protocol['config_sha256'],
        'checkpoints': checkpoints,
        'interpretation_contract': {
            'single_variable': 'representation checkpoint variant C1 versus R1',
            'production_variant': 'C1',
            'R1_role': 'EXPLANATORY_ONLY',
            'R1_cannot_be_promoted': True,
            'outcome_cannot_change_selected_mechanism': True,
        },
        'A1_5_unlock_rule': 'all 18 runs, immutable hashes, exact replay and completion audit pass',
        'environment': {
            'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'locks': {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False},
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    with PROTOCOL.open('x') as stream:
        stream.write(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
