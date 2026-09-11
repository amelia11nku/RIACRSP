#!/usr/bin/env python3
"""Run the integrity-revalidated 54-run NGAS A1.6R comparison."""
from __future__ import annotations

from dataclasses import asdict
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

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import (  # noqa: E402
    audit_frozen_inputs, digest, load_json,
)
from rcias_ngas.evaluation.a16_io import (  # noqa: E402
    validate_raw_contract, write_new_json,
)
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.search.ngas_solver import (  # noqa: E402
    NGASDiagnosticConfig, search_config_from_dict, solve_ngas,
)


OUT = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1'
CONFIG = ROOT / 'configs/ngas_a16r_integrity_diagnostic_v1.json'
PROTOCOL = OUT / 'preregistration/protocol.json'
PROGRESS = OUT / 'progress.json'
FORMAL_OWNER_ID = 'NGAS_A1_6R_FORMAL_OWNER_V1'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def candidate_payload(candidate: Candidate) -> dict:
    return {
        'operation_order': list(candidate.operation_order),
        'island_assignment': list(candidate.island_assignment),
        'w_assignment': list(candidate.w_assignment),
        'f_assignment': list(candidate.f_assignment),
    }


def candidate_from_payload(payload: dict) -> Candidate:
    return Candidate(*(tuple(payload[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return False


def raw_path(task: dict) -> Path:
    return OUT / 'raw' / task['instance_id'] / f"seed_{task['seed']}.json"


def build_tasks(config: dict) -> list[dict]:
    manifest = load_json(ROOT / config['scope']['instance_manifest_path'])
    tasks = []
    for instance in manifest['instances']:
        for seed in config['scope']['seeds']:
            tasks.append({
                'instance_id': instance['instance_id'],
                'instance_relative_path': instance['relative_path'],
                'instance_sha256': instance['sha256'],
                'num_operations': int(instance['num_operations']),
                'scale': instance['scale'],
                'CF_level': instance['CF_level'],
                'cell_replicate': instance['cell_replicate'],
                'variant': config['production_solver']['variant'],
                'seed': int(seed),
                'budget_seconds': config['budget']['multiplier'] * int(instance['num_operations']),
            })
    if len(tasks) != 54 or len({(row['instance_id'], row['seed']) for row in tasks}) != 54:
        raise RuntimeError('A1.6R requires exactly 54 unique instance-seed runs')
    return tasks


def forbidden_holdout_access_exists() -> bool:
    return any((ROOT / path).exists() for path in (
        'outputs/phase6j_caur/r13_selection/access_ledger.json',
        'outputs/phase6j_caur/r14_holdout/access_ledger.json',
    ))


def load_boundary(require_clean: bool = True) -> tuple[dict, dict, str]:
    protocol = load_json(PROTOCOL)
    config = load_json(CONFIG)
    if protocol.get('schema') != 'ngas-a16r-formal-protocol-v1' \
            or protocol.get('status') != 'FROZEN_BEFORE_FORMAL_RESULTS':
        raise RuntimeError('A1.6R formal protocol is not frozen')
    if protocol.get('formal_owner_id') != FORMAL_OWNER_ID:
        raise RuntimeError('A1.6R formal owner identity changed')
    if digest(CONFIG) != protocol['config_sha256']:
        raise RuntimeError('A1.6R config changed after freeze')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'A1.6R frozen source changed: {relative}')
    for relative, expected in protocol['artifact_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'A1.6R frozen artifact changed: {relative}')
    preformal_path = ROOT / protocol['preformal_audit_path']
    preformal = load_json(preformal_path)
    if digest(preformal_path) != protocol['preformal_audit_sha256'] \
            or preformal.get('status') != 'PASS':
        raise RuntimeError('A1.6R preformal audit is absent, changed, or failed')
    live_audit = audit_frozen_inputs(ROOT, config)
    if live_audit['status'] != 'PASS' \
            or live_audit['registry_sha256'] != protocol['registry_sha256'] \
            or live_audit['bks_sha256'] != protocol['bks_sha256']:
        raise RuntimeError('A1.6R frozen comparison inputs changed')
    if forbidden_holdout_access_exists():
        raise RuntimeError('R13/R14 access is prohibited during A1.6R')
    if require_clean and subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('A1.6R formal execution requires a clean committed worktree')
    return protocol, config, digest(PROTOCOL)


def validate_existing_raw(path: Path, task: dict, protocol_sha256: str,
                          config: dict) -> dict:
    payload = load_json(path)
    validate_raw_contract(
        payload, task, protocol_sha256,
        schema='ngas-a16r-formal-run-v1', algorithm_id='NGAS_A1_6R')
    if payload.get('config_sha256') != digest(CONFIG) \
            or payload.get('checkpoint_sha256') != config['production_solver']['checkpoint_sha256'] \
            or payload.get('scale') != task['scale'] \
            or payload.get('CF_level') != task['CF_level'] \
            or payload.get('cell_replicate') != task['cell_replicate'] \
            or payload.get('formal_owner_id') != FORMAL_OWNER_ID \
            or not all_finite(payload):
        raise RuntimeError(f'Existing A1.6R raw payload failed metadata validation: {path}')
    instance_path = ROOT / config['scope']['instance_root'] / task['instance_relative_path']
    if digest(instance_path) != task['instance_sha256']:
        raise RuntimeError(f'A1.6R instance changed: {instance_path}')
    instance = load_instance(instance_path)
    replay = decode_candidate(instance, candidate_from_payload(payload['best_candidate']))
    replay_audit = check_schedule(instance, replay.schedule)
    replay_record = {
        'instance_sha256': task['instance_sha256'],
        'best_candidate': payload['best_candidate'],
        'best_makespan': replay.makespan,
        'best_schedule': replay.schedule.to_dict(),
        'feasible': replay.feasible and replay_audit['feasible'],
    }
    if (not replay_record['feasible']
            or replay.makespan != payload['final_makespan']
            or canonical_hash(replay_record) != payload['replay_integrity_sha256']
            or payload['atomic_budget_audit']['started_after_deadline_count'] != 0):
        raise RuntimeError(f'Existing A1.6R raw payload failed replay/integrity: {path}')
    return payload


def write_progress(tasks: list[dict], completed: set[tuple[str, int]],
                   current: dict | None, process_started: float,
                   protocol_sha256: str, session: ExperimentSessionLock) -> None:
    pending_budget = math.fsum(
        task['budget_seconds'] for task in tasks
        if (task['instance_id'], task['seed']) not in completed
    )
    atomic_json(PROGRESS, {
        'schema': 'ngas-a16r-progress-v1',
        'status': 'FORMAL_RAW_COMPLETE' if len(completed) == 54 else 'RUNNING',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'completed_runs': len(completed),
        'expected_runs': 54,
        'current_task': current,
        'process_elapsed_seconds': time.perf_counter() - process_started,
        'nominal_remaining_budget_seconds': pending_budget,
        'nominal_remaining_hours': pending_budget / 3600.,
        'protocol_sha256': protocol_sha256,
        'formal_owner_id': session.formal_owner_id,
        'session_id': session.session_id,
        'session_pid': session.pid,
        'session_hostname': session.hostname,
        'raw_path': str((OUT / 'raw').relative_to(ROOT)),
        'resume_command': f'{sys.executable} -u scripts/run_ngas_a16r_solver_comparison.py --device cuda:0',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    })


def refresh_latency_payload(diagnostics: dict) -> dict:
    values = [
        float(row['components_ms']['complete_refresh'])
        for row in diagnostics['refreshes'] if row.get('components_ms')
    ]
    return {
        'complete_refresh_ms': values,
        'count': len(values),
        'count_gt_30_ms': sum(value > 30. for value in values),
        'count_gt_50_ms': sum(value > 50. for value in values),
        'count_gt_100_ms': sum(value > 100. for value in values),
        'maximum_ms': max(values) if values else None,
    }


def execute(task: dict, protocol_sha256: str, config: dict, critic,
            runtime: ProductionRefreshRuntime, search_config,
            diagnostic_config: NGASDiagnosticConfig,
            session: ExperimentSessionLock) -> dict:
    instance_path = ROOT / config['scope']['instance_root'] / task['instance_relative_path']
    if digest(instance_path) != task['instance_sha256']:
        raise RuntimeError(f"A1.6R instance hash changed: {task['instance_id']}")
    instance = load_instance(instance_path)
    bks_manifest = load_json(ROOT / config['bks']['path'])
    bks = float(bks_manifest['instances'][task['instance_id']]['makespan'])
    started_at = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    result = solve_ngas(
        instance, task['budget_seconds'], task['seed'],
        config['production_solver']['mode'], critic, search_config,
        refresh_runtime=runtime, budget_accounting=config['budget']['accounting_mode'],
        diagnostic_config=diagnostic_config)
    replay = decode_candidate(instance, result.best.candidate)
    replay_audit = check_schedule(instance, replay.schedule)
    end_to_end_elapsed = time.perf_counter() - end_to_end_started
    if not replay.feasible or not replay_audit['feasible'] \
            or replay.makespan != result.best.makespan:
        raise RuntimeError(f"A1.6R final replay failed: {task['instance_id']} seed {task['seed']}")
    trace = [asdict(point) for point in result.convergence_trace]
    if (not trace or trace[-1]['current_best_makespan'] != result.best.makespan
            or any(a['elapsed_time'] > b['elapsed_time']
                   or a['decoder_evaluations'] > b['decoder_evaluations']
                   or a['current_best_makespan'] < b['current_best_makespan']
                   for a, b in zip(trace, trace[1:]))):
        raise RuntimeError('A1.6R incumbent trace failed monotonicity validation')
    best_candidate = candidate_payload(result.best.candidate)
    replay_record = {
        'instance_sha256': task['instance_sha256'],
        'best_candidate': best_candidate,
        'best_makespan': replay.makespan,
        'best_schedule': replay.schedule.to_dict(),
        'feasible': True,
    }
    diagnostics = dict(result.diagnostics)
    termination = diagnostics['telemetry']['termination']
    runtime_components = diagnostics['runtime_components']
    payload = {
        'schema': 'ngas-a16r-formal-run-v1', 'status': 'COMPLETE',
        'algorithm_id': 'NGAS_A1_6R',
        **task,
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_sha256': protocol_sha256,
        'formal_owner_id': session.formal_owner_id,
        'session_id': session.session_id,
        'session_pid': session.pid,
        'session_hostname': session.hostname,
        'session_process_start_ticks': session.process_start_ticks,
        'config_sha256': digest(CONFIG),
        'checkpoint_sha256': critic.sha256,
        'bks_manifest_sha256': digest(ROOT / config['bks']['path']),
        'comparator_registry_sha256': digest(ROOT / config['comparators']['registry_path']),
        'started_at_utc': started_at,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'budget_accounting': config['budget']['accounting_mode'],
        'solver_budget_elapsed_seconds': result.runtime,
        'end_to_end_run_elapsed_seconds': end_to_end_elapsed,
        'runtime_preparation_seconds': runtime_components['runtime_preparation_seconds'],
        'budget_overshoot_seconds': diagnostics['budget_overshoot_seconds'],
        'termination_reason': diagnostics['termination_reason'],
        'h1_makespan': trace[0]['current_best_makespan'],
        'final_makespan': result.best.makespan,
        'bks_v001_makespan': bks,
        'rpd_v001_percent': 100. * (result.best.makespan - bks) / bks,
        'best_found_seconds': result.best_found_time,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations,
        'accepted_moves': termination['accepted_moves'],
        'improving_moves': termination['improving_moves'],
        'new_best_moves': termination['new_best_moves'],
        'critic_calls': termination['neural_calls'],
        'refresh_count': len(diagnostics['refreshes']),
        'guided_iterations': diagnostics['guided_iterations'],
        'guided_iterations_per_critic_call': diagnostics['guided_iterations_per_critic_call'],
        'incumbent_trace': trace,
        'best_candidate': best_candidate,
        'best_schedule': result.best.schedule.to_dict(),
        'best_actions': [asdict(action) for action in result.best.actions],
        'runtime_components': runtime_components,
        'refresh_latency': refresh_latency_payload(diagnostics),
        'selected_actions_by_size_repair': diagnostics['selection_counts_size_repair'],
        'online_portfolio': diagnostics['online_portfolio'],
        'search_diagnostics': diagnostics,
        'atomic_budget_audit': diagnostics['atomic_budget_audit'],
        'feasible': True,
        'feasibility_replay': replay_audit,
        'replay_integrity_sha256': canonical_hash(replay_record),
        'r13_accessed': False, 'r14_accessed': False, 'gurobi_run': False,
    }
    if not all_finite(payload):
        raise RuntimeError('A1.6R raw payload contains non-finite values')
    validate_raw_contract(
        payload, task, protocol_sha256,
        schema='ngas-a16r-formal-run-v1', algorithm_id='NGAS_A1_6R')
    return payload


def _run_locked(args, session: ExperimentSessionLock) -> None:
    compute_processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip().splitlines()
    if compute_processes:
        raise RuntimeError(
            f'competing GPU compute process exists before A1.6R CUDA init: '
            f'{compute_processes}')
    session.record('gpu_resource_preflight', competing_compute_processes=[])
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Formal A1.6R requires the qualified CUDA environment')
    _, config, protocol_sha256 = load_boundary(require_clean=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.empty(1, device=args.device).fill_(1.)
    torch.cuda.synchronize()
    model_started = time.perf_counter()
    critic = FrozenJointCritic(
        ROOT / config['production_solver']['checkpoint_path'], args.device,
        config['production_solver']['checkpoint_sha256'],
        config['production_solver']['variant'])
    runtime = ProductionRefreshRuntime(
        critic,
        prior_advantage_scale=config['production_solver']['search']['prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])
    model_load_seconds = time.perf_counter() - model_started
    search_config = search_config_from_dict(config['production_solver']['search'])
    tasks = build_tasks(config)
    process_started = time.perf_counter()
    existing_manifest_path = OUT / 'raw_manifest.json'
    existing_manifest = load_json(existing_manifest_path) if existing_manifest_path.exists() else {
        'schema': 'ngas-a16r-raw-manifest-v1', 'files': {}
    }
    for relative, expected in existing_manifest.get('files', {}).items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'A1.6R raw changed after manifesting: {relative}')
    completed = set()
    for task in tasks:
        path = raw_path(task)
        if path.exists():
            validate_existing_raw(path, task, protocol_sha256, config)
            completed.add((task['instance_id'], task['seed']))
    pending = [task for task in tasks
               if (task['instance_id'], task['seed']) not in completed]
    write_progress(tasks, completed, pending[0] if pending else None,
                   process_started, protocol_sha256, session)
    session.record(
        'formal_queue_validated', pending_runs=len(pending),
        completed_runs=len(completed), protocol_sha256=protocol_sha256)
    print(json.dumps({
        'event': 'a16r_formal_start', 'pending_runs': len(pending),
        'completed_runs': len(completed),
        'nominal_pending_budget_seconds': math.fsum(row['budget_seconds'] for row in pending),
        'model_load_seconds_outside_run_budget': model_load_seconds,
        'protocol_sha256': protocol_sha256,
    }), flush=True)
    for index, task in enumerate(pending, 1):
        print(json.dumps({
            'event': 'a16r_run_start', 'pending_index': index,
            'pending_total': len(pending), 'instance_id': task['instance_id'],
            'seed': task['seed'], 'budget_seconds': task['budget_seconds'],
        }), flush=True)
        capture_tasks = {
            (row['instance_id'], int(row['seed']))
            for row in config['diagnostics']['offline_state_capture']['tasks']
        }
        capture = tuple(config['diagnostics']['offline_state_capture']['fractions']) \
            if (task['instance_id'], task['seed']) in capture_tasks else ()
        diagnostic_config = NGASDiagnosticConfig(True, capture)
        session.record(
            'formal_run_started', instance_id=task['instance_id'], seed=task['seed'],
            budget_seconds=task['budget_seconds'])
        payload = execute(
            task, protocol_sha256, config, critic, runtime, search_config,
            diagnostic_config, session)
        path = raw_path(task)
        write_new_json(path, payload)
        validate_existing_raw(path, task, protocol_sha256, config)
        completed.add((task['instance_id'], task['seed']))
        manifest = {
            'schema': 'ngas-a16r-raw-manifest-v1',
            'protocol_sha256': protocol_sha256,
            'completed_runs': len(completed),
            'files': {
                str(raw_path(row).relative_to(ROOT)): digest(raw_path(row))
                for row in tasks if raw_path(row).exists()
            },
        }
        atomic_json(existing_manifest_path, manifest)
        next_task = pending[index] if index < len(pending) else None
        write_progress(
            tasks, completed, next_task, process_started, protocol_sha256, session)
        session.record(
            'formal_run_completed', instance_id=task['instance_id'], seed=task['seed'],
            raw_path=str(path.relative_to(ROOT)), raw_sha256=digest(path))
        print(json.dumps({
            'event': 'a16r_run_complete', 'completed_runs': len(completed),
            'instance_id': task['instance_id'], 'seed': task['seed'],
            'solver_budget_elapsed_seconds': payload['solver_budget_elapsed_seconds'],
            'budget_overshoot_seconds': payload['budget_overshoot_seconds'],
            'final_makespan': payload['final_makespan'],
            'rpd_v001_percent': payload['rpd_v001_percent'],
        }), flush=True)
    print(json.dumps({
        'event': 'a16r_formal_raw_complete', 'completed_runs': len(completed),
        'protocol_sha256': protocol_sha256,
    }), flush=True)
    session.record('formal_raw_complete', completed_runs=len(completed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    lock = ExperimentSessionLock(
        OUT / 'integrity/formal.lock', OUT / 'integrity/sessions',
        experiment='NGAS_A1_6R', stage='A1.6R', implementation_commit=commit,
        command=[sys.executable, *sys.argv], formal_owner_id=FORMAL_OWNER_ID,
        heartbeat_interval_seconds=30.)
    with lock as session:
        _run_locked(args, session)


if __name__ == '__main__':
    main()
