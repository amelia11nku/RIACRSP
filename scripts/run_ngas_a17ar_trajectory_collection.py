#!/usr/bin/env python3
"""Collect the frozen clean non-R12 A1.7A-R production trajectories."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, append_exposure, sha256_file, validate_manifest,
)
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.runtime.compact_state import CompactStateBuilder  # noqa: E402
from rcias_ngas.rng import RNGStreams  # noqa: E402
from rcias_ngas.search.ngas_solver import (  # noqa: E402
    NGASDiagnosticConfig, search_config_from_dict, solve_ngas,
)


CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
PROTOCOL = ROOT / 'artifacts/ngas_a17ar/trajectory_protocol_manifest.json'
REGISTRY_PATH = ROOT / 'configs/dataset_role_registry.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
FORMAL_OWNER = 'NGAS_A1_7AR_TRAJECTORY_OWNER_V1'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def raw_path(task: dict) -> Path:
    return OUT / 'raw' / task['dataset_role'].lower() / f"{task['instance_id']}.json"


def load_boundary(require_clean: bool = True) -> tuple[dict, dict, str]:
    config = json.loads(CONFIG.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = sha256_file(PROTOCOL)
    if protocol.get('status') != 'FROZEN_BEFORE_FORMAL_COLLECTION':
        raise RuntimeError('A1.7A-R trajectory protocol is not frozen')
    if sha256_file(CONFIG) != protocol['config_sha256'] \
            or sha256_file(REGISTRY_PATH) != protocol['registry_sha256']:
        raise RuntimeError('A1.7A-R config or registry changed after protocol freeze')
    for name, expected in protocol['source_hashes'].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f'A1.7A-R frozen source changed: {name}')
    if sha256_file(ROOT / protocol['checkpoint_path']) != protocol['checkpoint_sha256']:
        raise RuntimeError('A1.7A-R frozen C1 checkpoint changed')
    if require_clean and subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Formal A1.7A-R collection requires a clean committed worktree')
    for path in (CONFIG, PROTOCOL):
        subprocess.run(['git', 'ls-files', '--error-unmatch', relative(path)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    registry = DatasetRegistry(REGISTRY_PATH, ROOT)
    for role, purpose in (('TRAIN', 'training'), ('VALIDATION', 'model_selection')):
        rows = [{
            'instance_id': row['instance_id'],
            'content_sha256': row['content_sha256'],
            'dataset_role': role,
        } for row in protocol['instances'] if row['dataset_role'] == role]
        validate_manifest(rows, registry, purpose)
    selected_ids = {row['instance_id'] for row in protocol['instances']}
    selected_hashes = {row['content_sha256'] for row in protocol['instances']}
    reserved = [row for row in registry.payload['instances']
                if row['family_id'] in config['reserved_families']]
    if selected_ids & {row['instance_id'] for row in reserved} \
            or selected_hashes & {row['content_sha256'] for row in reserved}:
        raise RuntimeError('Formal A1.7A-R split overlaps a reserved dataset')
    return config, protocol, protocol_sha


def _history_at(iterations: list[dict], iteration: int) -> tuple[list[dict], float | None, int]:
    history = [row for row in iterations if row['iteration'] <= iteration]
    recent = history[-20:]
    best = recent[-1]['best_after'] if recent else None
    best_iterations = [row['iteration'] for row in history
                       if row['outcome_class'] == 'new_global_best']
    stagnation = iteration - (best_iterations[-1] if best_iterations else 0)
    compact = [{
        'iteration': row['iteration'], 'best_after': row['best_after'],
        'current_after': row['current_after'], 'accepted': row['accepted'],
        'outcome_class': row['outcome_class'],
        'relative_current_improvement': row['relative_current_improvement'],
    } for row in recent]
    return compact, best, stagnation


def _compact_features(instance, snapshot: dict, trajectory_seed: int,
                      builder: CompactStateBuilder) -> dict:
    raw = snapshot['current_candidate']
    candidate = Candidate(*(
        tuple(raw[name]) for name in (
            'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))
    current = decode_candidate(instance, candidate)
    if not current.feasible or current.makespan != snapshot['current_makespan']:
        raise RuntimeError(f'{snapshot["state_id"]} compact-feature replay failed')
    compact = builder.build(
        current, snapshot['state_id'], RNGStreams(instance.instance_id, trajectory_seed))
    if [action.action_id for action in compact.actions] != snapshot['action_ids']:
        raise RuntimeError(f'{snapshot["state_id"]} candidate bank changed during replay')
    batch = compact.cpu_batch
    features = {
        'schema': 'ngas-compact-relational-state-features-v1',
        'node_features': batch['x'].tolist(),
        'node_types': batch['types'].tolist(),
        'edge_index': batch['edge_index'].tolist(),
        'edge_types': batch['relations'].tolist(),
        'edge_features': batch['edge_features'].tolist(),
        'operation_nodes': batch['operation_nodes'].tolist(),
        'critical_mask': batch['critical_mask'].tolist(),
    }
    graph_identity = {
        name: features[name] for name in (
            'node_types', 'edge_index', 'edge_types', 'operation_nodes')}
    features['graph_hash'] = canonical_hash(graph_identity)
    features['state_feature_hash'] = canonical_hash(features)
    return features


def _state_records(instance, task: dict, diagnostics: dict, protocol: dict,
                   collector_commit: str) -> list[dict]:
    records = []
    builder = CompactStateBuilder(instance)
    iterations = diagnostics['iterations']
    refresh_by_iteration = {row['iteration']: row for row in diagnostics['refreshes']}
    selected_by_start = {row['iteration'] - 1: row for row in iterations}
    for snapshot in diagnostics['a16r_replayable_states']:
        iteration = int(snapshot['iteration'])
        selected = selected_by_start.get(iteration)
        history, best, stagnation = _history_at(iterations, iteration)
        refresh = refresh_by_iteration[iteration]
        observation = selected.get('a16r_observation', {}) if selected else {}
        compact_features = _compact_features(
            instance, snapshot, task['trajectory_seed'], builder)
        tags = []
        if any(row['relative_current_improvement'] > 0 for row in history[-5:]):
            tags.append('improving')
        else:
            tags.append('plateau_or_stagnating')
        if history and history[-1]['relative_current_improvement'] > 0:
            tags.append('immediately_post_improvement')
        changed = refresh.get('changed_since_previous_refresh') or {}
        if changed.get('critical_signature') or changed.get('dominant_bottleneck'):
            tags.append('bottleneck_or_critical_transition')
        record = {
            'schema': 'ngas-a17ar-clean-trajectory-state-v1',
            'state_id': snapshot['state_id'],
            'dataset_role': task['dataset_role'], 'audit_only': False,
            'eligible_for_future_training': task['dataset_role'] == 'TRAIN',
            'instance_id': task['instance_id'],
            'instance_content_sha256': task['content_sha256'],
            'generation_provenance': {
                'base_structure': task['base_structure'],
                'base_generation_seed': task['base_generation_seed'],
                'final_generation_seed': task['final_generation_seed'],
            },
            'trajectory_seed': task['trajectory_seed'],
            'search_iteration': iteration,
            'elapsed_wall_clock_seconds': snapshot['observed_budget_fraction'] * task['budget_seconds'],
            'normalized_budget_fraction': snapshot['observed_budget_fraction'],
            'target_capture_fraction': snapshot['capture_fraction'],
            'incumbent_objective': snapshot['current_makespan'],
            'best_so_far_objective': best,
            'recent_improvement_history': history,
            'stagnation_length': stagnation,
            'bottleneck_descriptors': {
                'critical_signature': snapshot['critical_signature'],
                'dominant_bottleneck': snapshot['dominant_bottleneck'],
                'changed_since_previous_refresh': changed,
            },
            'compact_relational_features': compact_features,
            'candidate_bank_identifier': 'NGAS_BANK_V1_FULL_UNIQUE',
            'candidate_bank_hash': protocol['candidate_bank_hash'],
            'candidate_action_ids': snapshot['action_ids'],
            'critic_checkpoint_hash': protocol['checkpoint_sha256'],
            'critic_scores': {
                'advantage': snapshot['advantage'],
                'beats_fallback_probability': snapshot['beats_fallback_probability'],
                'neural_prior': snapshot['prior'],
            },
            'portfolio_adjusted_scores': {
                'scope': 'selected_action_and_top1_identity_only',
                'selected_combined_probability': (
                    selected.get('combined_probability') if selected else None),
                'combined_top1_action_id': observation.get('combined_top1_action_id'),
                'portfolio_factor': observation.get('portfolio_factor'),
            },
            'selected_joint_action': {
                'action_id': selected.get('action_id') if selected else None,
                'neighborhood_size': selected.get('action_size') if selected else None,
                'destroy_target_operations': (
                    selected.get('target_operations') if selected else None),
                'repair_strategy': selected.get('repair') if selected else None,
            },
            'stochastic_trial_outcomes': observation.get('candidate_trials', []),
            'search_condition_tags': tags,
            'replay_metadata': {
                'current_candidate': snapshot['current_candidate'],
                'candidate_fingerprint': canonical_hash(snapshot['current_candidate']),
                'source_state_schema': snapshot['schema'],
            },
            'collector_git_sha': collector_commit,
        }
        records.append(record)
    if len(records) != len(protocol['capture_fractions']):
        raise RuntimeError(
            f'{task["instance_id"]} did not produce every frozen capture fraction')
    entropies = []
    for record in records:
        values = record['critic_scores']['neural_prior']
        entropies.append(-math.fsum(value * math.log(max(value, 1e-300)) for value in values))
    median = statistics.median(entropies)
    for record, entropy in zip(records, entropies):
        record['critic_scores']['neural_prior_entropy'] = entropy
        record['search_condition_tags'].append(
            'high_action_entropy' if entropy >= median else 'low_action_entropy')
        action_count = len(record['candidate_action_ids'])
        critic = record['critic_scores']
        if (not record['compact_relational_features']['state_feature_hash']
                or not record['compact_relational_features']['graph_hash']):
            raise RuntimeError(f'{record["state_id"]} is missing compact-state identity')
        if any(len(critic[name]) != action_count for name in (
                'advantage', 'beats_fallback_probability', 'neural_prior')):
            raise RuntimeError(f'{record["state_id"]} has inconsistent critic output shapes')
        numerical = [
            value for name in ('advantage', 'beats_fallback_probability', 'neural_prior')
            for value in critic[name]
        ]
        if not all(math.isfinite(value) for value in numerical + [entropy]):
            raise RuntimeError(f'{record["state_id"]} has non-finite critic output')
        if not math.isclose(math.fsum(critic['neural_prior']), 1., rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f'{record["state_id"]} has an invalid neural prior')
    return records


def valid_raw(path: Path, task: dict, protocol_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        return (
            payload.get('schema') == 'ngas-a17ar-clean-trajectory-run-v1'
            and payload.get('status') == 'COMPLETE'
            and payload.get('protocol_sha256') == protocol_sha
            and payload.get('instance_id') == task['instance_id']
            and payload.get('instance_content_sha256') == task['content_sha256']
            and payload.get('trajectory_seed') == task['trajectory_seed']
            and len(payload.get('states', ())) == 5
            and all(row.get('dataset_role') == task['dataset_role']
                    and row.get('audit_only') is False
                    and row.get('instance_content_sha256') == task['content_sha256']
                    and row.get('compact_relational_features', {}).get('state_feature_hash')
                    and row.get('compact_relational_features', {}).get('graph_hash')
                    for row in payload['states'])
            and payload['search_diagnostics']['final_replay']['feasible'] is True
        )
    except (KeyError, json.JSONDecodeError):
        return False


def execute(task: dict, config: dict, protocol: dict, protocol_sha: str,
            critic: FrozenJointCritic, runtime: ProductionRefreshRuntime,
            search_config, collector_commit: str, *,
            phase: str = 'NGAS_A1_7A_R_STAGE7',
            access_scope: str = 'clean non-R12 production-trajectory collection') -> dict:
    path = ROOT / task['relative_path']
    if sha256_file(path) != task['content_sha256']:
        raise RuntimeError(f'Instance hash changed: {task["instance_id"]}')
    append_exposure(LEDGER, {
        'git_sha': collector_commit,
        'command': 'scripts/run_ngas_a17ar_trajectory_collection.py',
        'phase': phase, 'instance_id': task['instance_id'],
        'instance_content_sha256': task['content_sha256'],
        'requested_purpose': 'diagnostic', 'dataset_role': task['dataset_role'],
        'checkpoint_identifier': protocol['checkpoint_sha256'], 'permitted': True,
        'access_scope': access_scope,
    })
    instance = load_instance(path)
    if instance.num_operations != task['num_operations']:
        raise RuntimeError(f'Operation count changed: {task["instance_id"]}')
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = solve_ngas(
        instance, task['budget_seconds'], task['trajectory_seed'],
        config['production_solver']['mode'], critic=critic, config=search_config,
        refresh_runtime=runtime, budget_accounting='A16_INSTANCE_TOTAL',
        diagnostic_config=NGASDiagnosticConfig(
            True, tuple(protocol['capture_fractions'])))
    end_to_end = time.perf_counter() - started
    diagnostics = result.diagnostics
    if diagnostics['a16r_instrumentation']['unreached_capture_fractions']:
        raise RuntimeError(f'Capture fractions not reached: {task["instance_id"]}')
    return {
        'schema': 'ngas-a17ar-clean-trajectory-run-v1', 'status': 'COMPLETE',
        'protocol_sha256': protocol_sha, 'collector_git_sha': collector_commit,
        'started_at_utc': started_at, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'instance_id': task['instance_id'], 'instance_relative_path': task['relative_path'],
        'instance_content_sha256': task['content_sha256'],
        'dataset_role': task['dataset_role'], 'audit_only': False,
        'trajectory_seed': task['trajectory_seed'], 'budget_seconds': task['budget_seconds'],
        'end_to_end_seconds': end_to_end, 'final_makespan': result.best.makespan,
        'best_found_seconds': result.best_found_time,
        'decoder_evaluations': result.decoder_evaluations,
        'iterations': result.iterations, 'states': _state_records(
            instance, task, diagnostics, protocol, collector_commit),
        'search_diagnostics': diagnostics,
        'R13': 'LOCKED_NO_ACCESS', 'R14': 'LOCKED_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXCLUDED', 'gurobi_run': False,
    }


def write_progress(tasks: list[dict], completed: list[dict], current: dict | None,
                   protocol_sha: str, session: ExperimentSessionLock,
                   process_started: float) -> None:
    pending = [row for row in tasks if row['instance_id'] not in {
        item['instance_id'] for item in completed}]
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a17ar-trajectory-progress-v1',
        'status': 'FORMAL_COLLECTION_COMPLETE' if not pending else 'RUNNING',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'completed_runs': len(completed), 'expected_runs': len(tasks),
        'completed_states': 5 * len(completed), 'expected_states': 540,
        'current_task': current, 'process_elapsed_seconds': time.perf_counter() - process_started,
        'nominal_remaining_budget_seconds': math.fsum(row['budget_seconds'] for row in pending),
        'protocol_sha256': protocol_sha, 'session_id': session.session_id,
        'pid': session.pid, 'hostname': session.hostname,
        'resume_command': f'{sys.executable} -u scripts/run_ngas_a17ar_trajectory_collection.py --device cuda:0',
        'log_path': 'outputs/ngas_a1/trajectory_utility_a17ar_v1/logs/formal.log',
        'R13': 'LOCKED_NO_ACCESS', 'R14': 'LOCKED_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXCLUDED',
    })


def run(args, session: ExperimentSessionLock) -> None:
    config, protocol, protocol_sha = load_boundary()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Formal A1.7A-R trajectory collection requires CUDA')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.empty(1, device=args.device).fill_(1.)
    torch.cuda.synchronize()
    critic = FrozenJointCritic(
        ROOT / protocol['checkpoint_path'], args.device,
        protocol['checkpoint_sha256'], 'C1')
    runtime = ProductionRefreshRuntime(
        critic,
        prior_advantage_scale=config['production_solver']['search']['prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])
    search_config = search_config_from_dict(config['production_solver']['search'])
    tasks = protocol['instances']
    collector_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    completed = []
    for task in tasks:
        path = raw_path(task)
        if path.exists() and not valid_raw(path, task, protocol_sha):
            raise RuntimeError(f'Existing trajectory raw is invalid: {relative(path)}')
        if path.exists():
            completed.append({'instance_id': task['instance_id'],
                              'path': relative(path), 'sha256': sha256_file(path)})
    process_started = time.perf_counter()
    pending = [task for task in tasks if not raw_path(task).exists()]
    write_progress(tasks, completed, pending[0] if pending else None,
                   protocol_sha, session, process_started)
    print(json.dumps({'event': 'a17ar_collection_start', 'completed': len(completed),
                      'pending': len(pending), 'protocol_sha256': protocol_sha}), flush=True)
    for index, task in enumerate(pending, 1):
        session.record('trajectory_run_started', instance_id=task['instance_id'],
                       dataset_role=task['dataset_role'], budget_seconds=task['budget_seconds'])
        print(json.dumps({'event': 'a17ar_run_start', 'pending_index': index,
                          'pending_total': len(pending), 'instance_id': task['instance_id'],
                          'role': task['dataset_role'], 'budget_seconds': task['budget_seconds']}),
              flush=True)
        payload = execute(task, config, protocol, protocol_sha, critic, runtime,
                          search_config, collector_commit)
        path = raw_path(task)
        if path.exists():
            raise RuntimeError(f'Refusing to overwrite formal raw: {relative(path)}')
        atomic_json(path, payload)
        if not valid_raw(path, task, protocol_sha):
            raise RuntimeError(f'New trajectory raw failed validation: {relative(path)}')
        completed.append({'instance_id': task['instance_id'], 'path': relative(path),
                          'sha256': sha256_file(path)})
        atomic_json(OUT / 'raw_manifest.json', {
            'schema': 'ngas-a17ar-trajectory-raw-manifest-v1',
            'protocol_sha256': protocol_sha, 'completed_runs': len(completed),
            'completed_states': 5 * len(completed),
            'files': {row['path']: row['sha256'] for row in completed},
        })
        next_task = pending[index] if index < len(pending) else None
        write_progress(tasks, completed, next_task, protocol_sha, session, process_started)
        session.record('trajectory_run_completed', instance_id=task['instance_id'],
                       raw_path=relative(path), raw_sha256=sha256_file(path))
        print(json.dumps({'event': 'a17ar_run_complete', 'completed': len(completed),
                          'instance_id': task['instance_id'],
                          'states': len(payload['states']),
                          'elapsed_seconds': payload['end_to_end_seconds']}), flush=True)
    session.record('formal_collection_complete', completed_runs=len(completed),
                   completed_states=5 * len(completed))
    print(json.dumps({'event': 'a17ar_collection_complete', 'runs': len(completed),
                      'states': 5 * len(completed)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    lock = ExperimentSessionLock(
        OUT / 'integrity/formal.lock', OUT / 'integrity/sessions',
        experiment='NGAS_A1_7A_R_TRAJECTORY', stage='A1.7A-R_STAGE7',
        implementation_commit=commit, command=[sys.executable, *sys.argv],
        formal_owner_id=FORMAL_OWNER, heartbeat_interval_seconds=30.)
    with lock as session:
        run(args, session)


if __name__ == '__main__':
    main()
