#!/usr/bin/env python3
"""Freeze the outcome-blind A1.4 small-R12 development protocol."""
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
CONFIG = ROOT / 'configs/ngas_a1_search_integration_v1.json'
SMOKE = ROOT / 'outputs/ngas_a1/search_integration_v1/smoke/smoke.json'
PROTOCOL = ROOT / 'outputs/ngas_a1/search_integration_v1/preregistration/protocol.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_paths() -> list[Path]:
    paths = [path for path in (ROOT / 'rcias_ngas').rglob('*.py')]
    paths += [
        CONFIG,
        ROOT / 'scripts/smoke_ngas_a14.py',
        ROOT / 'scripts/freeze_ngas_a14_protocol.py',
        ROOT / 'scripts/run_ngas_a14_integration.py',
        ROOT / 'scripts/launch_ngas_a14_integration.py',
        ROOT / 'rcias_clgri/search/common.py',
        ROOT / 'rcias_clgri/search/alns.py',
        ROOT / 'rcias_clgri/heuristic/dispatching.py',
        ROOT / 'rcias_clgri/data/loader.py',
        ROOT / 'rcias_clgri/env/feasibility.py',
        ROOT / 'rcias_clgri/env/insertion_decoder.py',
    ]
    return sorted(set(paths))


def main() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(f'Frozen protocol already exists: {PROTOCOL}')
    worktree = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if worktree:
        raise RuntimeError('Freeze requires a clean committed implementation worktree')
    config = json.loads(CONFIG.read_text())
    smoke = json.loads(SMOKE.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if smoke['status'] != 'PASS' or smoke['implementation_commit'] != head:
        raise RuntimeError('CUDA smoke is absent, failed, or belongs to another commit')
    instance_root = ROOT / config['instance_source']['root']
    for row in config['development_subset']:
        if digest(instance_root / row['relative_path']) != row['sha256']:
            raise RuntimeError(f"Frozen instance drift: {row['instance_id']}")
    checkpoint = ROOT / config['representation_boundary']['checkpoint_path']
    if digest(checkpoint) != config['representation_boundary']['checkpoint_sha256']:
        raise RuntimeError('Frozen C1 checkpoint drift')
    sources = source_paths()
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError(f'Missing formal source files: {missing}')
    protocol = {
        'schema': 'ngas-a14-search-integration-protocol-v1',
        'status': 'FROZEN_BEFORE_OUTCOMES',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': head,
        'config_path': str(CONFIG.relative_to(ROOT)),
        'config_sha256': digest(CONFIG),
        'smoke_path': str(SMOKE.relative_to(ROOT)),
        'smoke_sha256': digest(SMOKE),
        'starting_state_audit_path': 'outputs/ngas_a1/search_integration_v1/audit/starting_state.json',
        'starting_state_audit_sha256': digest(
            ROOT / 'outputs/ngas_a1/search_integration_v1/audit/starting_state.json'),
        'source_hashes': {
            str(path.relative_to(ROOT)): digest(path) for path in sources},
        'instance_hashes': {
            row['instance_id']: row['sha256'] for row in config['development_subset']},
        'checkpoint': config['representation_boundary'],
        'modes': config['ablation_modes'],
        'seeds': config['development_seeds'],
        'run_order': 'instance_then_seed_then_ablation_mode',
        'expected_primary_runs': (
            len(config['development_subset']) * len(config['development_seeds'])
            * len(config['ablation_modes'])),
        'budget': config['budget'],
        'selection_gate': config['selection_gate'],
        'post_selection_C1_R1_template': {
            'mechanism': 'deterministic selected mechanism from selection_gate',
            'instances': [row['instance_id'] for row in config['development_subset']],
            'seeds': config['development_seeds'],
            'budget': config['budget'],
            'held_fold': 0,
            'checkpoint_template': 'outputs/ngas_a1/critic_training_rthgt_v2/variants/{variant}/seed_{seed}/fold_0/model.pt',
            'variants': ['C1', 'R1'],
            'role': 'explanatory only; R1 is ineligible for production regardless of solver outcome',
        },
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
