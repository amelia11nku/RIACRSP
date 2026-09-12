#!/usr/bin/env python3
"""Run the frozen A1.7A-R full-bank U0-U3 counterfactual audit."""
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
    canonical_hash, clone_portfolio, outcome_class, ranking_metrics,
    replay_portfolio, utility_agreement,
)
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    DatasetRegistry, append_exposure, sha256_file,
)
from rcias_ngas.rng import RNGStreams  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from rcias_ngas.search.ngas_solver import (  # noqa: E402
    _combine, _select, search_config_from_dict,
)


PROTOCOL = ROOT / 'artifacts/ngas_a17ar/full_bank_protocol_manifest.json'
CONFIG = ROOT / 'configs/ngas_a17ar_trajectory_collection_protocol.yaml'
REGISTRY = ROOT / 'configs/dataset_role_registry.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
RAW = OUT / 'diagnostics/full_bank_raw'
FORMAL_OWNER = 'NGAS_A1_7AR_FULL_BANK_OWNER_V1'


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


def candidate_from_payload(payload: dict) -> Candidate:
    return Candidate(*(tuple(payload[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def stage_label(fraction: float) -> str:
    labels = ('0-20%', '20-40%', '40-60%', '60-80%', '80-100%')
    return labels[min(int(float(fraction) * 5.), 4)]


def load_boundary() -> tuple[dict, dict, str]:
    protocol = json.loads(PROTOCOL.read_text())
    config = json.loads(CONFIG.read_text())
    protocol_sha = sha256_file(PROTOCOL)
    if protocol.get('status') != 'FROZEN_BEFORE_FULL_BANK_RESULTS':
        raise RuntimeError('A1.7A-R full-bank protocol is not frozen')
    for path, expected in protocol['source_hashes'].items():
        if sha256_file(ROOT / path) != expected:
            raise RuntimeError(f'frozen full-bank source changed: {path}')
    if sha256_file(ROOT / protocol['checkpoint_path']) != protocol['checkpoint_sha256']:
        raise RuntimeError('frozen C1 checkpoint changed')
    if sha256_file(ROOT / protocol['collection_audit_path']) \
            != protocol['collection_audit_sha256']:
        raise RuntimeError('clean collection audit changed')
    if sha256_file(ROOT / protocol['R12_state_protocol_path']) \
            != protocol['R12_state_protocol_sha256']:
        raise RuntimeError('R12 audit-state protocol changed')
    return protocol, config, protocol_sha


def validate_worktree(*, allow_modified_ledger: bool) -> None:
    lines = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    unexpected = []
    for line in lines:
        path = line[3:]
        if allow_modified_ledger and path == relative(LEDGER) and line[:2] == ' M':
            continue
        unexpected.append(line)
    if unexpected:
        raise RuntimeError(f'full-bank audit has unexpected worktree changes: {unexpected}')


def _state_conditions(formal: dict, snapshot: dict) -> list[str]:
    history = [row for row in formal['search_diagnostics']['iterations']
               if row['iteration'] <= snapshot['iteration']][-20:]
    tags = []
    if any(row['relative_current_improvement'] > 0. for row in history[-5:]):
        tags.append('improving')
    else:
        tags.append('plateau_or_stagnating')
    if history and history[-1]['relative_current_improvement'] > 0.:
        tags.append('immediately_post_improvement')
    refresh = next(row for row in formal['search_diagnostics']['refreshes']
                   if row['iteration'] == snapshot['iteration'])
    change = refresh.get('changed_since_previous_refresh') or {}
    if change.get('critical_signature') or change.get('dominant_bottleneck'):
        tags.append('bottleneck_or_critical_transition')
    return tags


def load_archived_context(state: dict, protocol: dict,
                          runtime: ProductionRefreshRuntime) -> dict:
    formal_path = ROOT / state['formal_raw_path']
    if sha256_file(formal_path) != state['formal_raw_sha256']:
        raise RuntimeError(f'formal raw changed: {state["state_key"]}')
    formal = json.loads(formal_path.read_text())
    if state['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT':
        matches = [row for row in formal['states']
                   if canonical_hash(row) == state['state_sha256']]
        if len(matches) != 1:
            raise RuntimeError(f'clean state identity failed: {state["state_key"]}')
        archived = matches[0]
        snapshot = {
            'state_id': archived['state_id'],
            'iteration': archived['search_iteration'],
            'current_candidate': archived['replay_metadata']['current_candidate'],
            'current_makespan': archived['incumbent_objective'],
            'action_ids': archived['candidate_action_ids'],
            'prior': archived['critic_scores']['neural_prior'],
            'advantage': archived['critic_scores']['advantage'],
            'beats_fallback_probability': archived['critic_scores'][
                'beats_fallback_probability'],
        }
        best_makespan = float(archived['best_so_far_objective'])
        conditions = list(archived['search_condition_tags'])
        expected_selected = archived['selected_joint_action']['action_id']
        expected_factor = archived['portfolio_adjusted_scores']['portfolio_factor']
    else:
        snapshots = formal['search_diagnostics']['a16r_replayable_states']
        matches = [row for row in snapshots if canonical_hash(row) == state['state_sha256']]
        if len(matches) != 1:
            raise RuntimeError(f'R12 state identity failed: {state["state_key"]}')
        snapshot = matches[0]
        history = [row for row in formal['search_diagnostics']['iterations']
                   if row['iteration'] <= snapshot['iteration']]
        best_makespan = float(
            history[-1]['best_after'] if history else snapshot['current_makespan'])
        conditions = _state_conditions(formal, snapshot)
        selected_row = next(
            row for row in formal['search_diagnostics']['iterations']
            if row['iteration'] == snapshot['iteration'] + 1)
        expected_selected = selected_row['action_id']
        expected_factor = selected_row['a16r_observation']['portfolio_factor']

    instance_path = ROOT / state['instance_relative_path']
    if sha256_file(instance_path) != state['instance_sha256']:
        raise RuntimeError(f'instance changed: {state["instance_id"]}')
    instance = load_instance(instance_path)
    current = decode_candidate(instance, candidate_from_payload(snapshot['current_candidate']))
    replay = check_schedule(instance, current.schedule)
    if not current.feasible or not replay['feasible'] \
            or current.makespan != snapshot['current_makespan']:
        raise RuntimeError(f'captured candidate replay failed: {state["state_key"]}')
    streams = RNGStreams(instance.instance_id, int(state['seed']))
    refresh = runtime.refresh(
        instance, current, snapshot['state_id'], streams,
        sample_seed=streams.seed(
            'neural_prior', snapshot['state_id'], snapshot['iteration']))
    if ([action.action_id for action in refresh.actions] != snapshot['action_ids']
            or max(abs(a - b) for a, b in zip(
                refresh.prior, snapshot['prior'])) > 1e-7
            or max(abs(a - b) for a, b in zip(
                refresh.advantage, snapshot['advantage'])) > 1e-7):
        raise RuntimeError(f'frozen critic/bank replay failed: {state["state_key"]}')
    settings = json.loads(CONFIG.read_text())['production_solver']['search']
    portfolio = replay_portfolio(
        formal['search_diagnostics']['iterations'], snapshot['iteration'],
        segment_length=settings['portfolio_segment_length'],
        reaction=settings['portfolio_reaction'], strength=settings['portfolio_strength'])
    selected_index = next(index for index, action in enumerate(refresh.actions)
                          if action.action_id == expected_selected)
    if not math.isclose(
            portfolio.factor(refresh.actions[selected_index]), expected_factor,
            rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f'portfolio replay failed: {state["state_key"]}')
    return {
        'formal': formal, 'instance': instance, 'current': current,
        'best_makespan': best_makespan, 'snapshot': snapshot,
        'conditions': conditions, 'production_selected_action_id': expected_selected,
        'streams': streams, 'refresh': refresh, 'portfolio': portfolio,
    }


def decode_trials(instance, current, action, streams: RNGStreams,
                  state_id: str, trials: int = 8) -> tuple[list, list[dict], float]:
    candidates, rows = [], []
    total_seconds = 0.
    for trial in range(trials):
        started = time.perf_counter()
        neighbor = construct_neighbor(
            instance, current, action,
            random.Random(streams.seed('neighbor', state_id, trial)))
        repair_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate = decode_candidate(instance, neighbor)
        decoder_seconds = time.perf_counter() - started
        if not candidate.feasible:
            raise RuntimeError('counterfactual repair produced an infeasible candidate')
        signed = float(current.makespan - candidate.makespan)
        elapsed = repair_seconds + decoder_seconds
        candidates.append(candidate)
        rows.append({
            'trial': trial + 1,
            'repair_rng_seed': streams.seed('neighbor', state_id, trial),
            'candidate_makespan': candidate.makespan,
            'signed_improvement': signed,
            'positive_improvement': max(0., signed),
            'repair_seconds': repair_seconds,
            'decoder_seconds': decoder_seconds,
            'trial_seconds': elapsed,
        })
        total_seconds += elapsed
    return candidates, rows, total_seconds


def continuation(instance, captured_current, forced_candidate, search_best: float,
                 forced_action, refresh, portfolio, streams: RNGStreams,
                 cache_state_id: str, base_iteration: int, settings) -> dict:
    active_portfolio = clone_portfolio(portfolio)
    forced_outcome, forced_relative = outcome_class(
        forced_candidate.makespan, captured_current.makespan, search_best, True)
    active_portfolio.observe(forced_action, forced_outcome, forced_relative)
    search_best = min(search_best, forced_candidate.makespan)
    local_best = min(captured_current.makespan, forced_candidate.makespan)
    current = forced_candidate
    total_seconds = 0.
    trace = []
    for step in range(1, 3):
        combined = _combine(refresh.prior, refresh.actions, active_portfolio)
        iteration = base_iteration + step
        index, explored, probability = _select(
            refresh.actions, combined, streams, cache_state_id,
            iteration, True, settings.online_exploration, False)
        selected = refresh.actions[index]
        trial_state_id = (
            f'{instance.instance_id}:seed{streams.run_seed}:iteration{iteration}')
        candidates, _, elapsed = decode_trials(
            instance, current, selected, streams, trial_state_id)
        total_seconds += elapsed
        candidate = min(candidates, key=lambda row: row.makespan)
        temperature = (settings.initial_temperature_fraction * max(1., current.makespan)
                       * settings.cooling_rate ** iteration)
        delta = candidate.makespan - current.makespan
        accepted = delta <= 0. or streams.stream(
            'acceptance', trial_state_id).random() < math.exp(
                -delta / max(temperature, 1e-12))
        outcome, relative = outcome_class(
            candidate.makespan, current.makespan, search_best, accepted)
        if accepted:
            current = candidate
        search_best = min(search_best, candidate.makespan)
        local_best = min(local_best, candidate.makespan)
        active_portfolio.observe(selected, outcome, relative)
        trace.append({
            'step': step, 'iteration': iteration,
            'selected_action_id': selected.action_id, 'explored': explored,
            'selection_probability': probability,
            'candidate_makespan': candidate.makespan, 'accepted': accepted,
            'current_after': current.makespan, 'local_best_after': local_best,
            'outcome_class': outcome,
        })
    return {
        'best_makespan': local_best,
        'additional_decoder_evaluations': 16,
        'additional_repair_decode_seconds': total_seconds,
        'trace': trace,
    }


def evaluate_state(state: dict, protocol: dict, protocol_sha: str,
                   config: dict, runtime: ProductionRefreshRuntime,
                   action_limit: int | None = None) -> dict:
    context = load_archived_context(state, protocol, runtime)
    instance, current = context['instance'], context['current']
    refresh, streams = context['refresh'], context['streams']
    settings = search_config_from_dict(config['production_solver']['search'])
    actions = refresh.actions[:action_limit] if action_limit else refresh.actions
    combined = _combine(refresh.prior, refresh.actions, context['portfolio'])
    action_rows = []
    for index, action in enumerate(actions):
        candidates, trials, direct_seconds = decode_trials(
            instance, current, action, streams, context['snapshot']['state_id'])
        best_candidate = min(candidates, key=lambda row: row.makespan)
        horizon = continuation(
            instance, current, best_candidate, context['best_makespan'], action,
            refresh, context['portfolio'], streams,
            context['snapshot']['state_id'], int(context['snapshot']['iteration']), settings)
        signed = [row['signed_improvement'] for row in trials]
        immediate = max(0., current.makespan - best_candidate.makespan)
        horizon_gain = max(0., current.makespan - horizon['best_makespan'])
        total_seconds = direct_seconds + horizon['additional_repair_decode_seconds']
        action_rows.append({
            'action_index': index, 'action_id': action.action_id,
            'size': action.size, 'repair': action.repair,
            'destroy_count': len(action.target.operations),
            'target_id': action.target.target_id,
            'target_operations': list(action.target.operations),
            'origin_rules': list(action.target.origin_rules),
            'origin_families': list(action.target.origin_families),
            'origin_operators': list(action.target.origin_operators),
            'critic_advantage': refresh.advantage[index],
            'critic_fallback_probability': refresh.beats_fallback_probability[index],
            'neural_prior': refresh.prior[index],
            'portfolio_factor': context['portfolio'].factor(action),
            'portfolio_adjusted_probability': combined[index],
            'direct_trials': trials,
            'continuation': horizon,
            'utility': {
                'U0_immediate_best_gain': immediate,
                'U1_short_horizon_best_gain': horizon_gain,
                'U2_immediate_gain_per_decoder': immediate / 8.,
                'U2_horizon_gain_per_decoder': horizon_gain / 24.,
                'U2_horizon_gain_per_second': (
                    horizon_gain / total_seconds if total_seconds else 0.),
                'U3_improvement_probability': float(np.mean(np.asarray(signed) > 0.)),
                'U3_mean_signed_improvement': float(np.mean(signed)),
                'U3_signed_improvement_variance': float(np.var(signed, ddof=0)),
                'U3_downside_probability': float(np.mean(np.asarray(signed) < 0.)),
            },
        })
    action_ids = [row['action_id'] for row in action_rows]
    scores = [row['critic_advantage'] for row in action_rows]
    utility_fields = {
        'U0_IMMEDIATE': 'U0_immediate_best_gain',
        'U1_SHORT_HORIZON': 'U1_short_horizon_best_gain',
        'U2_COST_NORMALIZED': 'U2_horizon_gain_per_second',
        'U3_STOCHASTIC_ROBUSTNESS': 'U3_mean_signed_improvement',
    }
    rankings = {
        name: ranking_metrics(
            scores, [row['utility'][field] for row in action_rows], action_ids)
        for name, field in utility_fields.items()}
    neural_order = sorted(
        range(len(action_rows)),
        key=lambda i: (-action_rows[i]['neural_prior'], action_rows[i]['action_id']))
    portfolio_order = sorted(
        range(len(action_rows)),
        key=lambda i: (-action_rows[i]['portfolio_adjusted_probability'],
                       action_rows[i]['action_id']))
    selected = context['production_selected_action_id']
    selected_index = action_ids.index(selected) if selected in action_ids else None
    portfolio = {
        'neural_top1_action_id': action_ids[neural_order[0]],
        'portfolio_top1_action_id': action_ids[portfolio_order[0]],
        'top1_changed': neural_order[0] != portfolio_order[0],
        'top5_overlap': len(set(neural_order[:5]) & set(portfolio_order[:5])),
        'production_selected_action_id': selected,
        'production_selected_in_evaluated_scope': selected_index is not None,
        'neural_top1_U0': action_rows[neural_order[0]]['utility'][
            'U0_immediate_best_gain'],
        'portfolio_top1_U0': action_rows[portfolio_order[0]]['utility'][
            'U0_immediate_best_gain'],
        'neural_top1_U1': action_rows[neural_order[0]]['utility'][
            'U1_short_horizon_best_gain'],
        'portfolio_top1_U1': action_rows[portfolio_order[0]]['utility'][
            'U1_short_horizon_best_gain'],
        'production_selected_U0': (
            action_rows[selected_index]['utility']['U0_immediate_best_gain']
            if selected_index is not None else None),
        'production_selected_U1': (
            action_rows[selected_index]['utility']['U1_short_horizon_best_gain']
            if selected_index is not None else None),
    }
    return {
        'schema': 'ngas-a17ar-full-bank-state-v1', 'status': 'COMPLETE',
        'protocol_sha256': protocol_sha, 'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        **state, 'search_iteration': context['snapshot']['iteration'],
        'search_stage': stage_label(state['observed_budget_fraction']),
        'search_condition_tags': context['conditions'],
        'captured_current_makespan': current.makespan,
        'captured_best_so_far_makespan': context['best_makespan'],
        'full_unique_bank_actions': len(refresh.actions),
        'evaluated_actions': len(action_rows),
        'matched_trials_per_action': 8,
        'actions': action_rows, 'critic_rankings': rankings,
        'utility_agreement': utility_agreement(action_rows),
        'portfolio_interaction': portfolio,
        'formal_solver_budget_inclusion': False,
        'audit_only': state['audit_only'],
        'eligible_for_future_training': state['eligible_for_future_training'],
        'boundaries': protocol['boundaries'],
    }


def valid_raw(path: Path, state: dict, protocol_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        actions = payload.get('actions', [])
        return (
            payload.get('schema') == 'ngas-a17ar-full-bank-state-v1'
            and payload.get('status') == 'COMPLETE'
            and payload.get('protocol_sha256') == protocol_sha
            and payload.get('state_key') == state['state_key']
            and payload.get('state_sha256') == state['state_sha256']
            and payload.get('evaluated_actions') == payload.get('full_unique_bank_actions')
            and payload.get('matched_trials_per_action') == 8
            and len(actions) == payload.get('evaluated_actions')
            and len({row.get('action_id') for row in actions}) == len(actions)
            and all(len(row.get('direct_trials', [])) == 8
                    and row.get('continuation', {}).get(
                        'additional_decoder_evaluations') == 16
                    and len(row.get('continuation', {}).get('trace', [])) == 2
                    and all(isinstance(value, (int, float)) and math.isfinite(value)
                            for value in row.get('utility', {}).values())
                    for row in actions))
    except (KeyError, json.JSONDecodeError):
        return False


def write_progress(protocol: dict, protocol_sha: str, completed: list[dict],
                   session: ExperimentSessionLock, started: float,
                   current: dict | None) -> None:
    complete_keys = {row['state_key'] for row in completed}
    pending = [row for row in protocol['states'] if row['state_key'] not in complete_keys]
    elapsed = time.perf_counter() - started
    mean = elapsed / len(completed) if completed else None
    atomic_json(OUT / 'diagnostics/full_bank_progress.json', {
        'schema': 'ngas-a17ar-full-bank-progress-v1',
        'status': 'COMPLETE' if not pending else 'RUNNING',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha,
        'completed_states': len(completed), 'expected_states': len(protocol['states']),
        'completed_actions': sum(row['evaluated_actions'] for row in completed),
        'current_state': current, 'elapsed_seconds': elapsed,
        'estimated_remaining_seconds': mean * len(pending) if mean is not None else None,
        'session_id': session.session_id, 'pid': session.pid, 'hostname': session.hostname,
        'resume_command': (
            f'{sys.executable} -u scripts/run_ngas_a17ar_full_bank_audit.py --device cuda:0'),
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
        raise RuntimeError('A1.7A-R full-bank audit requires CUDA')
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
    if args.smoke:
        validate_worktree(allow_modified_ledger=False)
        state = protocol['states'][0]
        append_exposure(LEDGER, {
            'git_sha': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'command': 'scripts/run_ngas_a17ar_full_bank_audit.py --smoke',
            'phase': 'NGAS_A1_7A_R_STAGE7_FULL_BANK_SMOKE',
            'instance_id': state['instance_id'],
            'instance_content_sha256': state['instance_sha256'],
            'requested_purpose': 'diagnostic', 'dataset_role': state['dataset_role'],
            'checkpoint_identifier': protocol['checkpoint_sha256'], 'permitted': True,
            'access_scope': 'two-action implementation smoke; excluded from formal metrics',
        })
        row = evaluate_state(state, protocol, protocol_sha, config, runtime, action_limit=2)
        row['schema'] = 'ngas-a17ar-full-bank-smoke-v1'
        row['formal_scope'] = False
        atomic_json(OUT / 'smoke/full_bank_smoke.json', row)
        print(json.dumps({
            'status': 'PASS', 'state_key': state['state_key'],
            'actions': row['evaluated_actions'],
            'path': relative(OUT / 'smoke/full_bank_smoke.json'),
        }, indent=2))
        return

    registry = DatasetRegistry(REGISTRY, ROOT)
    for state in protocol['states']:
        registry.authorize({
            'instance_id': state['instance_id'],
            'content_sha256': state['instance_sha256'],
            'dataset_role': state['dataset_role'],
        }, 'diagnostic')
    has_existing = any(valid_raw(raw_path(state), state, protocol_sha)
                       for state in protocol['states'])
    validate_worktree(allow_modified_ledger=has_existing)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    lock = ExperimentSessionLock(
        OUT / 'integrity/full_bank.lock', OUT / 'integrity/full_bank_sessions',
        experiment='NGAS_A1_7A_R_FULL_BANK', stage='A1.7A-R_STAGE7_FULL_BANK',
        implementation_commit=commit, command=[sys.executable, *sys.argv],
        formal_owner_id=FORMAL_OWNER, heartbeat_interval_seconds=30.)
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
                'git_sha': commit, 'command': 'scripts/run_ngas_a17ar_full_bank_audit.py',
                'phase': 'NGAS_A1_7A_R_STAGE7_FULL_BANK',
                'instance_id': state['instance_id'],
                'instance_content_sha256': state['instance_sha256'],
                'requested_purpose': 'diagnostic', 'dataset_role': state['dataset_role'],
                'checkpoint_identifier': protocol['checkpoint_sha256'], 'permitted': True,
                'access_scope': (
                    'frozen full-bank U0-U3 diagnostic; R12 rows audit-only'),
            })
            session.record('full_bank_state_started', state_key=state['state_key'])
            row = evaluate_state(state, protocol, protocol_sha, config, runtime)
            atomic_json(path, row)
            session.record(
                'full_bank_state_completed', state_key=state['state_key'],
                actions=row['evaluated_actions'], raw_path=relative(path),
                raw_sha256=sha256_file(path))
            completed.append(row)
            write_progress(protocol, protocol_sha, completed, session, started, None)
            print(json.dumps({
                'event': 'full_bank_state_complete', 'completed': len(completed),
                'state_key': state['state_key'], 'actions': row['evaluated_actions'],
                'elapsed_seconds': time.perf_counter() - started,
            }), flush=True)
        manifest = {
            'schema': 'ngas-a17ar-full-bank-raw-manifest-v1',
            'protocol_sha256': protocol_sha,
            'completed_states': len(completed),
            'completed_actions': sum(row['evaluated_actions'] for row in completed),
            'files': {relative(raw_path(state)): sha256_file(raw_path(state))
                      for state in protocol['states']},
        }
        atomic_json(OUT / 'diagnostics/full_bank_raw_manifest.json', manifest)
        session.record(
            'full_bank_audit_complete', states=len(completed),
            actions=manifest['completed_actions'])
    print(json.dumps({
        'event': 'full_bank_audit_complete', 'states': len(completed),
        'actions': sum(row['evaluated_actions'] for row in completed),
    }))


if __name__ == '__main__':
    main()
