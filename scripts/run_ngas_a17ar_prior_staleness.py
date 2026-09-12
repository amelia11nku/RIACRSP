#!/usr/bin/env python3
"""Run the frozen A1.7A-R fixed-refresh prior-staleness audit."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_ngas.actions.repair import construct_neighbor  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.evaluation.a17ar import (  # noqa: E402
    canonical_hash, ranking_metrics, safe_spearman,
)
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, append_exposure, sha256_file,
)
from rcias_ngas.rng import RNGStreams  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402


PROTOCOL = ROOT / 'artifacts/ngas_a17ar/prior_staleness_protocol_manifest.json'
CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
REGISTRY = ROOT / 'configs/dataset_role_registry.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
RAW = OUT / 'diagnostics/prior_staleness_raw'
FORMAL_OWNER = 'NGAS_A1_7AR_STALENESS_OWNER_V1'


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def candidate_from_payload(payload: dict) -> Candidate:
    return Candidate(*(tuple(payload[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def raw_path(state: dict) -> Path:
    name = hashlib.sha256(state['state_key'].encode()).hexdigest()[:24]
    return RAW / f'{name}.json'


def semantic_key(action) -> tuple[str, tuple[str, ...], str]:
    return action.size, tuple(action.target.operations), action.repair


def semantic_text(action) -> str:
    size, operations, repair = semantic_key(action)
    return json.dumps([size, operations, repair], separators=(',', ':'))


def load_boundary() -> tuple[dict, dict, str]:
    protocol = json.loads(PROTOCOL.read_text())
    config = json.loads(CONFIG.read_text())
    protocol_sha = sha256_file(PROTOCOL)
    if protocol.get('status') != 'FROZEN_BEFORE_STALENESS_RESULTS':
        raise RuntimeError('A1.7A-R prior-staleness protocol is not frozen')
    for path, expected in protocol['source_hashes'].items():
        if sha256_file(ROOT / path) != expected:
            raise RuntimeError(f'frozen prior-staleness source changed: {path}')
    for path_key, hash_key in (
            ('checkpoint_path', 'checkpoint_sha256'),
            ('full_bank_protocol_path', 'full_bank_protocol_sha256'),
            ('full_bank_completion_audit_path', 'full_bank_completion_audit_sha256'),
            ('full_bank_raw_manifest_path', 'full_bank_raw_manifest_sha256')):
        if sha256_file(ROOT / protocol[path_key]) != protocol[hash_key]:
            raise RuntimeError(f'frozen prior-staleness dependency changed: {path_key}')
    return protocol, config, protocol_sha


def validate_worktree(*, allow_modified_ledger: bool) -> None:
    unexpected = []
    for line in subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines():
        if (allow_modified_ledger and line[:2] == ' M'
                and line[3:] == relative(LEDGER)):
            continue
        unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            f'prior-staleness audit has unexpected worktree changes: {unexpected}')


def load_archived_context(state: dict, runtime: ProductionRefreshRuntime) -> dict:
    formal_path = ROOT / state['formal_raw_path']
    if sha256_file(formal_path) != state['formal_raw_sha256']:
        raise RuntimeError(f'formal raw changed: {state["state_key"]}')
    formal = json.loads(formal_path.read_text())
    if state['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT':
        matches = [row for row in formal['states']
                   if canonical_hash(row) == state['state_sha256']]
        if len(matches) != 1:
            raise RuntimeError(f'clean snapshot identity failed: {state["state_key"]}')
        archived = matches[0]
        snapshot = {
            'state_id': archived['state_id'],
            'iteration': archived['search_iteration'],
            'current_candidate': archived['replay_metadata']['current_candidate'],
            'current_makespan': archived['incumbent_objective'],
            'action_ids': archived['candidate_action_ids'],
            'prior': archived['critic_scores']['neural_prior'],
            'advantage': archived['critic_scores']['advantage'],
        }
    else:
        matches = [row for row in formal['search_diagnostics']['a16r_replayable_states']
                   if canonical_hash(row) == state['state_sha256']]
        if len(matches) != 1:
            raise RuntimeError(f'R12 snapshot identity failed: {state["state_key"]}')
        snapshot = matches[0]

    instance_path = ROOT / state['instance_relative_path']
    if sha256_file(instance_path) != state['instance_sha256']:
        raise RuntimeError(f'instance changed: {state["state_key"]}')
    instance = load_instance(instance_path)
    current = decode_candidate(instance, candidate_from_payload(snapshot['current_candidate']))
    replay = check_schedule(instance, current.schedule)
    if (not current.feasible or not replay['feasible']
            or current.makespan != snapshot['current_makespan']):
        raise RuntimeError(f'captured candidate replay failed: {state["state_key"]}')
    streams = RNGStreams(instance.instance_id, int(state['seed']))
    refresh = runtime.refresh(
        instance, current, snapshot['state_id'], streams,
        sample_seed=streams.seed(
            'neural_prior', snapshot['state_id'], snapshot['iteration']))
    if ([action.action_id for action in refresh.actions] != snapshot['action_ids']
            or max(abs(a - b) for a, b in zip(refresh.prior, snapshot['prior'])) > 1e-7
            or max(abs(a - b) for a, b in zip(
                refresh.advantage, snapshot['advantage'])) > 1e-7):
        raise RuntimeError(f'base critic/bank replay failed: {state["state_key"]}')
    iteration_rows = {row['iteration']: row
                      for row in formal['search_diagnostics']['iterations']}
    return {
        'formal': formal, 'instance': instance, 'current': current,
        'snapshot': snapshot, 'streams': streams, 'base_refresh': refresh,
        'iteration_rows': iteration_rows,
    }


def replay_to_offset(context: dict, offset: int):
    instance = context['instance']
    current = context['current']
    snapshot = context['snapshot']
    actions = {action.action_id: action for action in context['base_refresh'].actions}
    base_iteration = int(snapshot['iteration'])
    for step in range(1, offset + 1):
        row = context['iteration_rows'].get(base_iteration + step)
        if row is None:
            raise RuntimeError(f'archived iteration missing at offset {step}')
        if row['action_id'] not in actions:
            raise RuntimeError('fixed refresh bank changed before offset 19')
        if current.makespan != row['current_before']:
            raise RuntimeError('archived current-before does not replay')
        action = actions[row['action_id']]
        candidates = []
        trials = row['a16r_observation']['candidate_trials']
        if len(trials) != 8:
            raise RuntimeError('archived selected action did not use eight trials')
        for archived in trials:
            neighbor = construct_neighbor(
                instance, current, action,
                random.Random(int(archived['repair_rng_seed'])))
            candidate = decode_candidate(instance, neighbor)
            if not candidate.feasible or candidate.makespan != archived['candidate_makespan']:
                raise RuntimeError('archived repair/decode trial replay failed')
            candidates.append(candidate)
        selected = min(candidates, key=lambda item: item.makespan)
        if selected.makespan != row['candidate_makespan']:
            raise RuntimeError('archived best-of-eight replay failed')
        if row['accepted']:
            current = selected
        if current.makespan != row['current_after']:
            raise RuntimeError('archived current-after does not replay')
    return current


def representation(runtime: ProductionRefreshRuntime, instance, current,
                   state_id: str, streams: RNGStreams) -> dict:
    builder = runtime._builders[instance.instance_id]
    compact = builder.build(current, state_id, streams)
    x = compact.cpu_batch['x'].numpy().copy()
    operation_nodes = compact.cpu_batch['operation_nodes'].numpy().copy()
    edges = compact.cpu_batch['edge_index'].numpy().copy()
    relations = compact.cpu_batch['relations'].numpy().copy()
    graph = set(zip(edges[0].tolist(), edges[1].tolist(), relations.tolist()))
    return {
        'operation_features': x[operation_nodes],
        'graph': graph,
        'critical_signature': compact.critical_signature,
        'dominant_bottleneck': compact.dominant_bottleneck,
        'operation_feature_sha256': hashlib.sha256(
            x[operation_nodes].tobytes()).hexdigest(),
        'graph_sha256': hashlib.sha256(json.dumps(
            sorted(graph), separators=(',', ':')).encode()).hexdigest(),
    }


def representation_drift(base: dict, current: dict) -> dict:
    left = base['operation_features']
    right = current['operation_features']
    if left.shape != right.shape:
        raise RuntimeError('operation feature shape changed inside one instance')
    difference = right - left
    union = base['graph'] | current['graph']
    intersection = base['graph'] & current['graph']
    return {
        'operation_feature_relative_l2': float(
            np.linalg.norm(difference) / max(np.linalg.norm(left), 1e-12)),
        'operation_feature_mean_absolute': float(np.mean(np.abs(difference))),
        'graph_edge_jaccard': len(intersection) / len(union) if union else 1.,
        'critical_signature_changed': (
            base['critical_signature'] != current['critical_signature']),
        'dominant_bottleneck_changed': (
            base['dominant_bottleneck'] != current['dominant_bottleneck']),
        'operation_feature_sha256': current['operation_feature_sha256'],
        'graph_sha256': current['graph_sha256'],
    }


def decode_action(instance, current, action, streams: RNGStreams,
                  trial_state_id: str) -> dict:
    rows = []
    for trial in range(8):
        seed = streams.seed('neighbor', trial_state_id, trial)
        candidate = decode_candidate(instance, construct_neighbor(
            instance, current, action, random.Random(seed)))
        if not candidate.feasible:
            raise RuntimeError('prior-staleness repair produced infeasible candidate')
        rows.append({
            'trial': trial + 1, 'repair_rng_seed': seed,
            'candidate_makespan': candidate.makespan,
            'signed_improvement': current.makespan - candidate.makespan,
        })
    best = min(row['candidate_makespan'] for row in rows)
    return {
        'action_id': action.action_id, 'semantic_key': semantic_text(action),
        'size': action.size, 'repair': action.repair,
        'target_operations': list(action.target.operations),
        'best_candidate_makespan': best,
        'U0_immediate_best_gain': max(0., current.makespan - best),
        'trials': rows,
    }


def score_drift(stale, fresh) -> dict:
    stale_by_semantic = {semantic_key(action): index
                         for index, action in enumerate(stale.actions)}
    fresh_by_semantic = {semantic_key(action): index
                         for index, action in enumerate(fresh.actions)}
    common = sorted(set(stale_by_semantic) & set(fresh_by_semantic))
    stale_rank = {index: rank for rank, index in enumerate(stale.ranking)}
    fresh_rank = {index: rank for rank, index in enumerate(fresh.ranking)}
    stale_percentile = [stale_rank[stale_by_semantic[key]] /
                        max(len(stale.actions) - 1, 1) for key in common]
    fresh_percentile = [fresh_rank[fresh_by_semantic[key]] /
                        max(len(fresh.actions) - 1, 1) for key in common]
    stale_top5 = {semantic_key(stale.actions[index]) for index in stale.ranking[:5]}
    fresh_top5 = {semantic_key(fresh.actions[index]) for index in fresh.ranking[:5]}
    stale_top10 = {semantic_key(stale.actions[index]) for index in stale.ranking[:10]}
    fresh_top10 = {semantic_key(fresh.actions[index]) for index in fresh.ranking[:10]}
    return {
        'common_semantic_actions': len(common),
        'semantic_union_actions': len(set(stale_by_semantic) | set(fresh_by_semantic)),
        'semantic_bank_jaccard': (
            len(common) / len(set(stale_by_semantic) | set(fresh_by_semantic))),
        'top1_same_semantics': (
            semantic_key(stale.actions[stale.ranking[0]])
            == semantic_key(fresh.actions[fresh.ranking[0]])),
        'top5_overlap': len(stale_top5 & fresh_top5),
        'top10_overlap': len(stale_top10 & fresh_top10),
        'mean_absolute_prior_drift_common': float(np.mean([
            abs(stale.prior[stale_by_semantic[key]]
                - fresh.prior[fresh_by_semantic[key]]) for key in common]))
            if common else None,
        'mean_absolute_advantage_drift_common': float(np.mean([
            abs(stale.advantage[stale_by_semantic[key]]
                - fresh.advantage[fresh_by_semantic[key]]) for key in common]))
            if common else None,
        'mean_absolute_percentile_rank_drift_common': float(np.mean(np.abs(
            np.asarray(stale_percentile) - np.asarray(fresh_percentile))))
            if common else None,
        'percentile_rank_spearman_common': safe_spearman(
            stale_percentile, fresh_percentile),
    }


def base_u0_rows(state: dict, action_limit: int | None) -> list[dict]:
    full_path = ROOT / state['full_bank_raw_path']
    if sha256_file(full_path) != state['full_bank_raw_sha256']:
        raise RuntimeError(f'full-bank raw changed: {state["state_key"]}')
    payload = json.loads(full_path.read_text())
    actions = payload['actions'][:action_limit] if action_limit else payload['actions']
    return [{
        'action_id': row['action_id'],
        'semantic_key': json.dumps(
            [row['size'], row['target_operations'], row['repair']], separators=(',', ':')),
        'size': row['size'], 'repair': row['repair'],
        'target_operations': row['target_operations'],
        'best_candidate_makespan': min(
            trial['candidate_makespan'] for trial in row['direct_trials']),
        'U0_immediate_best_gain': row['utility']['U0_immediate_best_gain'],
        'trials': [{
            'trial': trial['trial'], 'repair_rng_seed': trial['repair_rng_seed'],
            'candidate_makespan': trial['candidate_makespan'],
            'signed_improvement': trial['signed_improvement'],
        } for trial in row['direct_trials']],
    } for row in actions]


def evaluate_state(state: dict, protocol: dict, protocol_sha: str,
                   runtime: ProductionRefreshRuntime,
                   action_limit: int | None = None,
                   offsets: tuple[int, ...] | None = None) -> dict:
    context = load_archived_context(state, runtime)
    instance = context['instance']
    streams = context['streams']
    base_refresh = context['base_refresh']
    base_actions = base_refresh.actions[:action_limit] if action_limit else base_refresh.actions
    base_iteration = int(context['snapshot']['iteration'])
    base_representation = representation(
        runtime, instance, context['current'], context['snapshot']['state_id'], streams)
    offset_rows = []
    for offset in offsets or tuple(protocol['offsets']):
        current = replay_to_offset(context, offset)
        fresh_state_id = f'{instance.instance_id}:seed{state["seed"]}:iteration{base_iteration + offset}'
        fresh = runtime.refresh(
            instance, current, fresh_state_id, streams,
            sample_seed=streams.seed(
                'neural_prior', fresh_state_id, base_iteration + offset))
        fresh_representation = representation(
            runtime, instance, current, fresh_state_id, streams)
        drift = score_drift(base_refresh, fresh)
        trial_state_id = f'NGAS_A17AR_STALENESS|{state["state_key"]}|offset{offset}'
        if offset == 0:
            action_rows = base_u0_rows(state, action_limit)
        else:
            action_rows = [decode_action(
                instance, current, action, streams, trial_state_id)
                for action in base_actions]
        utility_by_id = {row['action_id']: row['U0_immediate_best_gain']
                         for row in action_rows}
        action_ids = [action.action_id for action in base_actions]
        utilities = [utility_by_id[action_id] for action_id in action_ids]
        stale_scores = list(base_refresh.prior[:len(base_actions)])
        stale_metrics = ranking_metrics(stale_scores, utilities, action_ids)
        stale_top = base_actions[sorted(
            range(len(base_actions)),
            key=lambda index: (-stale_scores[index], action_ids[index]))[0]]
        fresh_top = fresh.actions[fresh.ranking[0]]
        stale_by_semantic = {semantic_key(action): action
                             for action in base_actions}
        if semantic_key(fresh_top) in stale_by_semantic:
            fresh_top_utility = utility_by_id[
                stale_by_semantic[semantic_key(fresh_top)].action_id]
            fresh_top_trials = 'REUSED_STALE_BANK_MATCHED_TRIALS'
        else:
            fresh_evaluation = decode_action(
                instance, current, fresh_top, streams, trial_state_id)
            fresh_top_utility = fresh_evaluation['U0_immediate_best_gain']
            fresh_top_trials = fresh_evaluation
        selected_iteration = base_iteration + offset + 1
        selected_row = context['iteration_rows'].get(selected_iteration)
        if selected_row is None or selected_row['action_id'] not in utility_by_id:
            raise RuntimeError('archived selected action missing from persistent base bank')
        best_union = max(max(utilities), fresh_top_utility)
        offset_rows.append({
            'offset': offset, 'iteration': base_iteration + offset,
            'current_makespan': current.makespan,
            'persistent_stale_actions_evaluated': len(action_rows),
            'full_unique_stale_bank_actions': len(base_refresh.actions),
            'fresh_bank_actions': len(fresh.actions),
            'stale_ranking_quality_U0': stale_metrics,
            'stale_vs_fresh': drift,
            'representation_drift': representation_drift(
                base_representation, fresh_representation),
            'selected_action_quality': {
                'stale_neural_top1_action_id': stale_top.action_id,
                'stale_neural_top1_U0': utility_by_id[stale_top.action_id],
                'fresh_neural_top1_action_id': fresh_top.action_id,
                'fresh_neural_top1_semantic_key': semantic_text(fresh_top),
                'fresh_neural_top1_U0': fresh_top_utility,
                'fresh_top1_trial_evidence': fresh_top_trials,
                'archived_selected_action_id': selected_row['action_id'],
                'archived_selected_matched_U0': utility_by_id[selected_row['action_id']],
                'archived_selected_original_U0': selected_row[
                    'a16r_observation']['best_of_trials_improvement'],
                'best_observed_union_U0': best_union,
                'stale_top1_regret_union': (
                    best_union - utility_by_id[stale_top.action_id]),
                'fresh_top1_regret_union': best_union - fresh_top_utility,
                'archived_selected_regret_union': (
                    best_union - utility_by_id[selected_row['action_id']]),
            },
            'stale_action_evaluations': action_rows,
        })
    return {
        'schema': 'ngas-a17ar-prior-staleness-state-v1',
        'status': 'COMPLETE', 'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha, **state,
        'base_iteration': base_iteration,
        'refresh_interval': protocol['refresh_interval'],
        'offsets': list(offsets or tuple(protocol['offsets'])),
        'base_operation_feature_sha256': base_representation['operation_feature_sha256'],
        'base_graph_sha256': base_representation['graph_sha256'],
        'offset_results': offset_rows,
        'audit_only': state['audit_only'],
        'eligible_for_future_training': state['eligible_for_future_training'],
        'formal_solver_budget_inclusion': False,
        'boundaries': protocol['boundaries'],
    }


def valid_raw(path: Path, state: dict, protocol: dict, protocol_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        return (
            payload.get('schema') == 'ngas-a17ar-prior-staleness-state-v1'
            and payload.get('status') == 'COMPLETE'
            and payload.get('protocol_sha256') == protocol_sha
            and payload.get('state_key') == state['state_key']
            and payload.get('offsets') == protocol['offsets']
            and len(payload.get('offset_results', [])) == len(protocol['offsets'])
            and all(row.get('persistent_stale_actions_evaluated')
                    == row.get('full_unique_stale_bank_actions')
                    for row in payload['offset_results']))
    except (KeyError, json.JSONDecodeError):
        return False


def write_progress(protocol: dict, protocol_sha: str, completed: list[dict],
                   session: ExperimentSessionLock, started: float,
                   current: dict | None) -> None:
    complete = {row['state_key'] for row in completed}
    pending = [row for row in protocol['states'] if row['state_key'] not in complete]
    elapsed = time.perf_counter() - started
    mean = elapsed / len(completed) if completed else None
    atomic_json(OUT / 'diagnostics/prior_staleness_progress.json', {
        'schema': 'ngas-a17ar-prior-staleness-progress-v1',
        'status': 'COMPLETE' if not pending else 'RUNNING',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha,
        'completed_states': len(completed), 'expected_states': len(protocol['states']),
        'completed_offsets': sum(len(row['offset_results']) for row in completed),
        'expected_offsets': len(protocol['states']) * len(protocol['offsets']),
        'current_state': current, 'elapsed_seconds': elapsed,
        'estimated_remaining_seconds': mean * len(pending) if mean is not None else None,
        'session_id': session.session_id, 'pid': session.pid, 'hostname': session.hostname,
        'resume_command': (
            f'{sys.executable} -u scripts/run_ngas_a17ar_prior_staleness.py '
            '--device cuda:0'),
        'R13': 'LOCKED_NO_ACCESS', 'R14': 'LOCKED_NO_ACCESS',
        'RCIAS_CB1_CORE45': 'EXCLUDED', 'gurobi_run': False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocol, config, protocol_sha = load_boundary()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('A1.7A-R prior-staleness audit requires CUDA')
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
        prior_advantage_scale=config['production_solver']['search'][
            'prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])

    if args.smoke:
        validate_worktree(allow_modified_ledger=False)
        state = protocol['states'][0]
        append_exposure(LEDGER, {
            'git_sha': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'command': 'scripts/run_ngas_a17ar_prior_staleness.py --smoke',
            'phase': 'NGAS_A1_7A_R_STAGE7_STALENESS_SMOKE',
            'instance_id': state['instance_id'],
            'instance_content_sha256': state['instance_sha256'],
            'requested_purpose': 'diagnostic', 'dataset_role': state['dataset_role'],
            'checkpoint_identifier': protocol['checkpoint_sha256'], 'permitted': True,
            'access_scope': 'three-action offsets 0 and 5 smoke; excluded from formal metrics',
        })
        row = evaluate_state(
            state, protocol, protocol_sha, runtime, action_limit=3, offsets=(0, 5))
        row['schema'] = 'ngas-a17ar-prior-staleness-smoke-v1'
        row['formal_scope'] = False
        atomic_json(OUT / 'smoke/prior_staleness_smoke.json', row)
        print(json.dumps({
            'status': 'PASS', 'state_key': state['state_key'],
            'offsets': row['offsets'], 'actions_per_offset': 3,
            'path': relative(OUT / 'smoke/prior_staleness_smoke.json'),
        }, indent=2))
        return

    registry = DatasetRegistry(REGISTRY, ROOT)
    for state in protocol['states']:
        registry.authorize({
            'instance_id': state['instance_id'],
            'content_sha256': state['instance_sha256'],
            'dataset_role': state['dataset_role'],
        }, 'diagnostic')
    has_existing = any(valid_raw(raw_path(state), state, protocol, protocol_sha)
                       for state in protocol['states'])
    validate_worktree(allow_modified_ledger=has_existing)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    lock = ExperimentSessionLock(
        OUT / 'integrity/prior_staleness.lock',
        OUT / 'integrity/prior_staleness_sessions',
        experiment='NGAS_A1_7A_R_PRIOR_STALENESS',
        stage='A1.7A-R_STAGE7_PRIOR_STALENESS',
        implementation_commit=commit, command=[sys.executable, *sys.argv],
        formal_owner_id=FORMAL_OWNER, heartbeat_interval_seconds=30.)
    started = time.perf_counter()
    with lock as session:
        completed = []
        for state in protocol['states']:
            path = raw_path(state)
            if valid_raw(path, state, protocol, protocol_sha):
                completed.append(json.loads(path.read_text()))
                continue
            write_progress(protocol, protocol_sha, completed, session, started, state)
            append_exposure(LEDGER, {
                'git_sha': commit,
                'command': 'scripts/run_ngas_a17ar_prior_staleness.py',
                'phase': 'NGAS_A1_7A_R_STAGE7_STALENESS',
                'instance_id': state['instance_id'],
                'instance_content_sha256': state['instance_sha256'],
                'requested_purpose': 'diagnostic', 'dataset_role': state['dataset_role'],
                'checkpoint_identifier': protocol['checkpoint_sha256'], 'permitted': True,
                'access_scope': (
                    'fixed-refresh prior-staleness diagnostic; R12 rows audit-only'),
            })
            session.record('staleness_state_started', state_key=state['state_key'])
            row = evaluate_state(state, protocol, protocol_sha, runtime)
            atomic_json(path, row)
            session.record(
                'staleness_state_completed', state_key=state['state_key'],
                offsets=len(row['offset_results']), raw_path=relative(path),
                raw_sha256=sha256_file(path))
            completed.append(row)
            write_progress(protocol, protocol_sha, completed, session, started, None)
            print(json.dumps({
                'event': 'staleness_state_complete', 'completed': len(completed),
                'state_key': state['state_key'],
                'elapsed_seconds': time.perf_counter() - started,
            }), flush=True)
        manifest = {
            'schema': 'ngas-a17ar-prior-staleness-raw-manifest-v1',
            'protocol_sha256': protocol_sha,
            'completed_states': len(completed),
            'completed_offsets': sum(len(row['offset_results']) for row in completed),
            'stale_action_offset_evaluations': sum(
                offset['persistent_stale_actions_evaluated']
                for row in completed for offset in row['offset_results']),
            'files': {relative(raw_path(state)): sha256_file(raw_path(state))
                      for state in protocol['states']},
        }
        atomic_json(OUT / 'diagnostics/prior_staleness_raw_manifest.json', manifest)
        session.record(
            'staleness_audit_complete', states=len(completed),
            offsets=manifest['completed_offsets'],
            action_offset_evaluations=manifest['stale_action_offset_evaluations'])
    print(json.dumps({
        'event': 'staleness_audit_complete', 'states': len(completed),
        'offsets': sum(len(row['offset_results']) for row in completed),
    }))


if __name__ == '__main__':
    main()
