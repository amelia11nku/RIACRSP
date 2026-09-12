#!/usr/bin/env python3
"""Run the frozen A1.7A-S supplemental full-bank audit."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, append_exposure, sha256_file,
)
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from scripts.run_ngas_a17ar_full_bank_audit import evaluate_state  # noqa: E402


PROTOCOL = ROOT / 'artifacts/ngas_a17as/supplemental_protocol_manifest.json'
CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
REGISTRY = ROOT / 'configs/dataset_role_registry.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17as_v1'
RAW = OUT / 'raw/full_bank'
FORMAL_OWNER = 'NGAS_A1_7AS_FULL_BANK_OWNER_V1'
FORMAL_PHASE = 'NGAS_A1_7A_S_FULL_BANK'
FORMAL_COMMAND = 'scripts/run_ngas_a17as_full_bank.py'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def raw_path(state: dict) -> Path:
    name = hashlib.sha256(state['state_key'].encode()).hexdigest()[:24]
    return RAW / f'{name}.json'


def load_boundary() -> tuple[dict, dict, str]:
    protocol = json.loads(PROTOCOL.read_text())
    config = json.loads(CONFIG.read_text())
    protocol_sha = sha256_file(PROTOCOL)
    if protocol.get('status') != 'FROZEN_BEFORE_SUPPLEMENTAL_OUTCOMES':
        raise RuntimeError('A1.7A-S supplemental protocol is not frozen')
    for path, expected in protocol['source_hashes'].items():
        if sha256_file(ROOT / path) != expected:
            raise RuntimeError(f'frozen A1.7A-S source changed: {path}')
    if sha256_file(ROOT / protocol['checkpoint_path']) != protocol['checkpoint_sha256']:
        raise RuntimeError('frozen production checkpoint changed')
    return protocol, config, protocol_sha


def validate_worktree(*, formal_resume: bool) -> None:
    allowed_prefixes = (
        'artifacts/ngas_a17as/',
        'configs/ngas_a17as_',
        'outputs/ngas_a1/trajectory_utility_a17as_v1/',
        'reports/ngas_a17as_',
        'scripts/audit_ngas_a17as_',
        'scripts/freeze_ngas_a17as_',
        'scripts/run_ngas_a17as_',
        'rcias_ngas/evaluation/a17as.py',
        'tests/ngas/test_a17as_',
    )
    unexpected = []
    for line in subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines():
        path = line[3:]
        if path == relative(LEDGER) and line[:2] == ' M':
            continue
        if any(path.startswith(prefix) for prefix in allowed_prefixes):
            continue
        unexpected.append(line)
    if unexpected:
        raise RuntimeError(f'A1.7A-S has unexpected worktree changes: {unexpected}')
    if formal_resume and not PROTOCOL.is_file():
        raise RuntimeError('formal run requires frozen A1.7A-S protocol')


def valid_raw(path: Path, state: dict, protocol_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
        actions = row.get('actions', [])
        return (
            row.get('schema') == 'ngas-a17as-full-bank-state-v1'
            and row.get('status') == 'COMPLETE'
            and row.get('protocol_sha256') == protocol_sha
            and row.get('state_key') == state['state_key']
            and row.get('state_payload_sha256') == state['state_payload_sha256']
            and row.get('evaluated_actions') == row.get('full_unique_bank_actions')
            and len(actions) == row.get('evaluated_actions')
            and len({action.get('action_id') for action in actions}) == len(actions)
            and row.get('matched_trials_per_action') == 8
            and all(
                [trial.get('trial') for trial in action.get('direct_trials', [])]
                    == list(range(1, 9))
                and action.get('continuation', {}).get(
                    'additional_decoder_evaluations') == 16
                and len(action.get('continuation', {}).get('trace', [])) == 2
                and all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in action.get('utility', {}).values())
                for action in actions))
    except (KeyError, json.JSONDecodeError):
        return False


def write_progress(protocol: dict, protocol_sha: str, completed: list[dict],
                   session: ExperimentSessionLock, started: float,
                   current: dict | None) -> None:
    complete = {row['state_key'] for row in completed}
    pending = [row for row in protocol['states'] if row['state_key'] not in complete]
    elapsed = time.perf_counter() - started
    mean_seconds = elapsed / len(completed) if completed else None
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a17as-full-bank-progress-v1',
        'status': 'COMPLETE' if not pending else 'RUNNING',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha,
        'completed_states': len(completed),
        'expected_states': protocol['state_count'],
        'completed_actions': sum(row['evaluated_actions'] for row in completed),
        'completed_direct_trials': sum(
            row['evaluated_actions'] * 8 for row in completed),
        'current_state': current,
        'elapsed_seconds': elapsed,
        'estimated_remaining_seconds': (
            mean_seconds * len(pending) if mean_seconds is not None else None),
        'session_id': session.session_id,
        'pid': session.pid,
        'hostname': session.hostname,
        'resume_command': (
            f'{sys.executable} -u {FORMAL_COMMAND} --device cuda:0'),
        'boundaries': protocol['boundaries'],
    })


def evaluated_state(state: dict, protocol: dict, protocol_sha: str,
                    config: dict, runtime: ProductionRefreshRuntime,
                    action_limit: int | None = None) -> dict:
    row = evaluate_state(
        state, protocol, protocol_sha, config, runtime, action_limit=action_limit)
    row['schema'] = 'ngas-a17as-full-bank-state-v1'
    row['state_payload_sha256'] = state['state_payload_sha256']
    row['stage'] = state['stage']
    row['search_stage'] = state['stage']
    row['audit_only'] = True
    row['eligible_for_future_training'] = False
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocol, config, protocol_sha = load_boundary()
    validate_worktree(formal_resume=not args.smoke)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('A1.7A-S full-bank audit requires CUDA')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    critic = FrozenJointCritic(
        ROOT / protocol['checkpoint_path'], args.device,
        protocol['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(
        critic,
        prior_advantage_scale=config['production_solver']['search']['prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])
    registry = DatasetRegistry(REGISTRY, ROOT)
    for state in protocol['states']:
        registry.authorize({
            'instance_id': state['instance_id'],
            'content_sha256': state['instance_sha256'],
            'dataset_role': state['dataset_role'],
        }, 'diagnostic')

    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.smoke:
        state = protocol['states'][0]
        append_exposure(LEDGER, {
            'git_sha': commit,
            'command': f'{FORMAL_COMMAND} --smoke',
            'phase': 'NGAS_A1_7A_S_FULL_BANK_SMOKE',
            'instance_id': state['instance_id'],
            'instance_content_sha256': state['instance_sha256'],
            'requested_purpose': 'diagnostic',
            'dataset_role': state['dataset_role'],
            'checkpoint_identifier': protocol['checkpoint_sha256'],
            'permitted': True,
            'access_scope': 'one-action CUDA smoke; excluded from formal metrics',
        })
        row = evaluated_state(
            state, protocol, protocol_sha, config, runtime, action_limit=1)
        row['schema'] = 'ngas-a17as-full-bank-smoke-v1'
        row['formal_scope'] = False
        atomic_json(OUT / 'smoke/cuda_full_bank_smoke.json', row)
        print(json.dumps({
            'status': 'PASS', 'device': args.device,
            'state_key': state['state_key'], 'actions': row['evaluated_actions'],
            'path': relative(OUT / 'smoke/cuda_full_bank_smoke.json'),
        }, indent=2))
        return

    existing = any(valid_raw(raw_path(state), state, protocol_sha)
                   for state in protocol['states'])
    if existing:
        validate_worktree(formal_resume=True)
    lock = ExperimentSessionLock(
        OUT / 'integrity/full_bank.lock',
        OUT / 'integrity/full_bank_sessions',
        experiment='NGAS_A1_7A_S_FULL_BANK',
        stage='A1.7A-S_SUPPLEMENTAL_FULL_BANK',
        implementation_commit=commit,
        command=[sys.executable, *sys.argv],
        formal_owner_id=FORMAL_OWNER,
        heartbeat_interval_seconds=30.)
    started = time.perf_counter()
    with lock as session:
        completed = []
        for state in protocol['states']:
            path = raw_path(state)
            if valid_raw(path, state, protocol_sha):
                completed.append(json.loads(path.read_text()))
                continue
            write_progress(protocol, protocol_sha, completed, session, started, state)
            append_exposure(LEDGER, {
                'git_sha': commit,
                'command': FORMAL_COMMAND,
                'phase': FORMAL_PHASE,
                'instance_id': state['instance_id'],
                'instance_content_sha256': state['instance_sha256'],
                'requested_purpose': 'diagnostic',
                'dataset_role': state['dataset_role'],
                'checkpoint_identifier': protocol['checkpoint_sha256'],
                'permitted': True,
                'access_scope': (
                    'frozen clean supplemental full-bank U0-U3 audit; labels audit-only'),
            })
            session.record('full_bank_state_started', state_key=state['state_key'])
            row = evaluated_state(state, protocol, protocol_sha, config, runtime)
            atomic_json(path, row)
            session.record(
                'full_bank_state_completed',
                state_key=state['state_key'],
                actions=row['evaluated_actions'],
                raw_path=relative(path),
                raw_sha256=sha256_file(path))
            completed.append(row)
            write_progress(protocol, protocol_sha, completed, session, started, None)
            print(json.dumps({
                'event': 'full_bank_state_complete',
                'completed': len(completed),
                'state_key': state['state_key'],
                'actions': row['evaluated_actions'],
                'elapsed_seconds': time.perf_counter() - started,
            }), flush=True)
        manifest = {
            'schema': 'ngas-a17as-full-bank-raw-manifest-v1',
            'protocol_sha256': protocol_sha,
            'completed_states': len(completed),
            'completed_actions': sum(row['evaluated_actions'] for row in completed),
            'completed_direct_trials': sum(
                row['evaluated_actions'] * 8 for row in completed),
            'completed_continuation_decoder_evaluations': sum(
                row['evaluated_actions'] * 16 for row in completed),
            'files': {relative(raw_path(state)): sha256_file(raw_path(state))
                      for state in protocol['states']},
        }
        atomic_json(OUT / 'raw/full_bank_manifest.json', manifest)
        session.record(
            'full_bank_audit_complete', states=len(completed),
            actions=manifest['completed_actions'],
            direct_trials=manifest['completed_direct_trials'])
    print(json.dumps({
        'event': 'full_bank_audit_complete',
        'states': len(completed),
        'actions': sum(row['evaluated_actions'] for row in completed),
    }))


if __name__ == '__main__':
    main()
