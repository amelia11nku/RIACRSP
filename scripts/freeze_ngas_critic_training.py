#!/usr/bin/env python3
"""Freeze A1.3 code, data, folds, environment and fixed-epoch training contract."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.train import digest, load_cache
from rcias_ngas.evaluation.bks import write_immutable
from scripts.build_ngas_training_cache import CACHE, MANIFEST, AUDIT, DEVELOPMENT

OUT = ROOT / 'outputs/ngas_a1/critic_training_gpu_v1'
PROTOCOL = OUT / 'training_protocol.json'
CONFIG = ROOT / 'configs/ngas_a1_development_v1.json'
GPU_CONFIG = ROOT / 'configs/ngas_a1_critic_training_gpu_v1.json'
GPU_SMOKE = OUT / 'gpu_feasibility.json'
RETIREMENT = ROOT / 'outputs/ngas_a1/critic_training_v1/retirement/retirement_record.json'


def main() -> None:
    if PROTOCOL.exists():
        raise FileExistsError('Critic training protocol already frozen')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Freeze requires a clean committed worktree')
    config = json.loads(CONFIG.read_text())
    gpu_config = json.loads(GPU_CONFIG.read_text())
    smoke = json.loads(GPU_SMOKE.read_text())
    retirement = json.loads(RETIREMENT.read_text())
    gate = json.loads((DEVELOPMENT / 'data_gate.json').read_text())
    audit = json.loads(AUDIT.read_text())
    records = load_cache(CACHE, MANIFEST)
    if gate['decision'] != 'READY_FOR_JOINT_CRITIC_TRAINING' or audit['status'] != 'PASS':
        raise RuntimeError('Training input gate has not passed')
    if smoke['status'] != 'PASS' or smoke['formal_optimizer_steps'] != 0:
        raise RuntimeError('Outcome-blind GPU feasibility gate has not passed')
    if retirement['status'] != 'SUPERSEDED_INCOMPLETE' or retirement['outcome_metrics_inspected']:
        raise RuntimeError('Old CPU protocol retirement boundary failed')
    if (gpu_config['formal_device'] != 'cuda:0' or gpu_config['cpu_fallback']
            or not gpu_config['restart_from_zero'] or gpu_config['reuse_superseded_cpu_models']):
        raise RuntimeError('GPU restart-from-zero execution contract failed')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != gpu_config['cublas_workspace_config']:
        raise RuntimeError('Frozen CUBLAS workspace environment is unavailable')
    if not torch.cuda.is_available() or torch.cuda.device_count() != gpu_config['required_cuda_device_count']:
        raise RuntimeError('Frozen CUDA device contract is unavailable')
    if any((OUT / name).exists() for name in ('oof', 'oof_gate.json', 'production', 'final_decision.json')):
        raise RuntimeError('Training outputs exist before protocol freeze')
    fold_audit = []
    for fold in range(3):
        held = {row['instance_id'] for row in records if row['fold'] == fold}
        fitted = {row['instance_id'] for row in records if row['fold'] != fold}
        fold_audit.append({'held_fold': fold, 'held_instances': sorted(held),
                           'fit_instances': sorted(fitted), 'instance_overlap': sorted(held & fitted),
                           'held_states': sum(row['fold'] == fold for row in records),
                           'fit_states': sum(row['fold'] != fold for row in records)})
    if any(row['instance_overlap'] or row['held_states'] != 24 or row['fit_states'] != 48
           for row in fold_audit):
        raise RuntimeError('Grouped OOF fold contract failed')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'regression.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml', str(OUT / 'regression.xml')],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    suite = ET.parse(OUT / 'regression.xml').getroot().find('testsuite')
    sources = [
        ROOT / 'rcias_ngas/csg/features.py', ROOT / 'rcias_ngas/critic/action_encoder.py',
        ROOT / 'rcias_ngas/critic/joint_critic.py', ROOT / 'rcias_ngas/critic/losses.py',
        ROOT / 'rcias_ngas/critic/train.py', ROOT / 'rcias_ngas/evaluation/ranking.py',
        ROOT / 'scripts/train_ngas_joint_critic.py', ROOT / 'scripts/audit_ngas_critic_training.py',
        ROOT / 'scripts/launch_ngas_joint_critic.py', CONFIG,
        GPU_CONFIG, ROOT / 'scripts/smoke_ngas_joint_critic_gpu.py',
        ROOT / 'scripts/retire_ngas_cpu_training.py',
    ]
    inputs = [CACHE, MANIFEST, AUDIT, GPU_SMOKE, RETIREMENT, DEVELOPMENT / 'protocol.json',
              DEVELOPMENT / 'data_gate.json', DEVELOPMENT / 'result_hash_manifest.json']
    properties = torch.cuda.get_device_properties(0)
    payload = {
        'schema': 'ngas-a13-gpu-critic-training-protocol-v1',
        'status': 'FROZEN_BEFORE_GPU_FORMAL_OPTIMIZER_STEP',
        'formal_optimizer_steps_started': False,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config, 'gpu_execution': gpu_config, 'fold_audit': fold_audit,
        'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'input_hashes': {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        'training_scope': {'states': 72, 'instances': 18, 'actions': 6465,
                           'seeds': config['training']['seeds'], 'folds': 3,
                           'fixed_epochs': config['training']['epochs']},
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
                        'cuda_version': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'device_name': properties.name, 'device_count': torch.cuda.device_count(),
                        'total_memory_bytes': properties.total_memory,
                        'compute_capability': f'{properties.major}.{properties.minor}'},
        'full_regression_tests': int(suite.get('tests')),
        'regression_xml_sha256': digest(OUT / 'regression.xml'),
        'checkpoint_selection': 'fixed final epoch; OOF labels never select epochs or checkpoints',
        'production_condition': 'fit fixed seed on all R12 development states only if at least two OOF seeds pass',
        'superseded_cpu_models_reused': False,
        'formal_run_starts_from_zero': True,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    write_immutable(PROTOCOL, payload)
    print(json.dumps({'status': payload['status'], 'tests': payload['full_regression_tests'],
                      'protocol_sha256': digest(PROTOCOL), 'implementation_commit': payload['implementation_commit']}, indent=2))


if __name__ == '__main__':
    main()
