#!/usr/bin/env python3
"""Validate and summarize the frozen A1.7A-R full-bank utility audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a17ar import ranking_metrics  # noqa: E402
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    sha256_file, verify_exposure_ledger,
)


OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
PROTOCOL = ROOT / 'artifacts/ngas_a17ar/full_bank_protocol_manifest.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
RAW = OUT / 'diagnostics/full_bank_raw'
RAW_MANIFEST = OUT / 'diagnostics/full_bank_raw_manifest.json'
PROGRESS = OUT / 'diagnostics/full_bank_progress.json'
SESSION_DIR = OUT / 'integrity/full_bank_sessions'
AUDIT = OUT / 'audit/full_bank_completion_audit.json'
DERIVED = OUT / 'derived'
FORMAL_OWNER = 'NGAS_A1_7AR_FULL_BANK_OWNER_V1'
FORMAL_PHASE = 'NGAS_A1_7A_R_STAGE7_FULL_BANK'
FORMAL_COMMAND = 'scripts/run_ngas_a17ar_full_bank_audit.py'
ORIGINS = ('CLEAN_NON_R12_DEVELOPMENT', 'R12_DEVELOPMENT_EXPOSED')
UTILITY_COLUMNS = {
    'U0_IMMEDIATE': 'U0_immediate_best_gain',
    'U1_SHORT_HORIZON': 'U1_short_horizon_best_gain',
    'U2_COST_NORMALIZED': 'U2_horizon_gain_per_second',
    'U3_STOCHASTIC_ROBUSTNESS': 'U3_mean_signed_improvement',
}
REPORTS = {
    'ranking': ROOT / 'reports/ngas_a17ar_critic_ranking_audit.md',
    'alignment': ROOT / 'reports/ngas_a17ar_utility_alignment.md',
    'portfolio': ROOT / 'reports/ngas_a17ar_portfolio_interaction.md',
    'trials': ROOT / 'reports/ngas_a17ar_candidate_trials_audit.md',
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'cannot write empty CSV: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def mean_or_none(values) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def median_or_none(values) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.median(clean) if clean else None


def instance_interval(rows: list[dict], column: str, *, seed_key: str,
                      resamples: int = 10_000) -> dict:
    """Unweighted instance-cluster percentile interval for an exploratory mean."""
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(column)
        if value is not None and math.isfinite(float(value)):
            grouped[row['instance_id']].append(float(value))
    cluster_values = [statistics.fmean(values) for values in grouped.values()]
    if not cluster_values:
        return {'mean': None, 'lower_95': None, 'upper_95': None,
                'instances': 0, 'resamples': resamples}
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(
        cluster_values[rng.randrange(len(cluster_values))]
        for _ in cluster_values) for _ in range(resamples))
    lower = estimates[int(.025 * resamples)]
    upper = estimates[min(int(.975 * resamples), resamples - 1)]
    return {
        'mean': statistics.fmean(cluster_values),
        'lower_95': lower,
        'upper_95': upper,
        'instances': len(cluster_values),
        'resamples': resamples,
    }


def session_audit(manifest: dict, expected_state_keys: set[str]) -> tuple[dict, list[Path]]:
    paths = sorted(SESSION_DIR.glob('*.jsonl'))
    sessions = []
    all_events = []
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        sessions.append((path, events))
        all_events.extend(events)
    acquired = [row for row in all_events if row.get('event') == 'lock_acquired']
    released = [row for row in all_events if row.get('event') == 'lock_released']
    starts = [row for row in all_events if row.get('event') == 'full_bank_state_started']
    completions = [row for row in all_events if row.get('event') == 'full_bank_state_completed']
    terminal = [row for row in all_events if row.get('event') == 'full_bank_audit_complete']
    completion_hashes = {
        row.get('raw_path'): row.get('raw_sha256') for row in completions
    }
    execution_commit = acquired[0].get('implementation_commit') if len(acquired) == 1 else None
    commit_exists = bool(execution_commit) and subprocess.run(
        ['git', 'cat-file', '-e', f'{execution_commit}^{{commit}}'], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    checks = {
        'full_bank_lock_released': not (OUT / 'integrity/full_bank.lock').exists(),
        'exactly_one_process_session': len(sessions) == 1,
        'event_sequences_contiguous': all(
            [row.get('sequence') for row in events] == list(range(1, len(events) + 1))
            for _, events in sessions),
        'one_logical_formal_owner': (
            {row.get('formal_owner_id') for row in all_events} == {FORMAL_OWNER}),
        'one_successful_acquire_release': (
            len(acquired) == len(released) == 1 and released[0].get('status') == 'SUCCESS'),
        'formal_starts_exact_once': (
            len(starts) == len(expected_state_keys)
            and Counter(row.get('state_key') for row in starts)
            == Counter({key: 1 for key in expected_state_keys})),
        'formal_completions_exact_once': (
            len(completions) == len(expected_state_keys)
            and Counter(row.get('state_key') for row in completions)
            == Counter({key: 1 for key in expected_state_keys})),
        'completion_raw_hashes_match_manifest': completion_hashes == manifest['files'],
        'one_terminal_completion_event': (
            len(terminal) == 1
            and terminal[0].get('states') == len(expected_state_keys)
            and terminal[0].get('actions') == manifest['completed_actions']),
        'execution_commit_exists': commit_exists,
    }
    return {
        'schema': 'ngas-a17ar-full-bank-session-integrity-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'execution_commit': execution_commit,
        'sessions': [{
            'path': relative(path),
            'session_id': events[0].get('session_id') if events else None,
            'event_count': len(events),
            'heartbeat_count': sum(row.get('event') == 'heartbeat' for row in events),
        } for path, events in sessions],
    }, paths


def validate_raw(protocol: dict, manifest: dict) -> tuple[list[dict], dict]:
    protocol_sha = sha256_file(PROTOCOL)
    expected = {row['state_key']: row for row in protocol['states']}
    actual_paths = sorted(RAW.glob('*.json'))
    raw_states = []
    errors = []
    actual_hashes = {}
    action_total = 0
    for path in actual_paths:
        actual_hashes[relative(path)] = sha256_file(path)
        state = json.loads(path.read_text())
        raw_states.append(state)
        frozen = expected.get(state.get('state_key'))
        state_errors = []
        if frozen is None:
            state_errors.append('unexpected_state_key')
        else:
            for key in (
                    'state_key', 'state_sha256', 'instance_id', 'instance_sha256',
                    'instance_relative_path', 'formal_raw_path', 'formal_raw_sha256',
                    'dataset_origin', 'dataset_role', 'audit_only',
                    'eligible_for_future_training', 'scale', 'CF_level', 'seed'):
                if state.get(key) != frozen.get(key):
                    state_errors.append(f'{key}_mismatch')
        actions = state.get('actions', [])
        action_total += len(actions)
        action_ids = [row.get('action_id') for row in actions]
        if (state.get('schema') != 'ngas-a17ar-full-bank-state-v1'
                or state.get('status') != 'COMPLETE'
                or state.get('protocol_sha256') != protocol_sha):
            state_errors.append('top_level_contract_mismatch')
        if (len(actions) != state.get('evaluated_actions')
                or len(actions) != state.get('full_unique_bank_actions')
                or not actions or len(action_ids) != len(set(action_ids))):
            state_errors.append('full_unique_action_bank_mismatch')
        if state.get('matched_trials_per_action') != 8:
            state_errors.append('matched_trials_scalar_mismatch')
        for action in actions:
            trials = action.get('direct_trials', [])
            continuation = action.get('continuation', {})
            if (len(trials) != 8
                    or [row.get('trial') for row in trials] != list(range(1, 9))):
                state_errors.append('direct_trials_not_exactly_eight')
                break
            if (continuation.get('additional_decoder_evaluations') != 16
                    or len(continuation.get('trace', [])) != 2):
                state_errors.append('continuation_not_exactly_two_eight_trial_steps')
                break
            if set(action.get('utility', {})) != {
                    'U0_immediate_best_gain', 'U1_short_horizon_best_gain',
                    'U2_horizon_gain_per_decoder', 'U2_horizon_gain_per_second',
                    'U2_immediate_gain_per_decoder', 'U3_downside_probability',
                    'U3_improvement_probability', 'U3_mean_signed_improvement',
                    'U3_signed_improvement_variance'}:
                state_errors.append('utility_contract_mismatch')
                break
        if set(state.get('critic_rankings', {})) != set(UTILITY_COLUMNS):
            state_errors.append('critic_ranking_contract_mismatch')
        if state.get('boundaries') != protocol['boundaries']:
            state_errors.append('boundary_mismatch')
        if (state.get('formal_solver_budget_inclusion') is not False
                or not finite_tree(state)):
            state_errors.append('budget_or_numeric_contract_mismatch')
        if state_errors:
            errors.append({'state_key': state.get('state_key'), 'errors': sorted(set(state_errors))})
    seen_keys = [row.get('state_key') for row in raw_states]
    checks = {
        'raw_scope_exact': (
            set(seen_keys) == set(expected) and len(seen_keys) == len(set(seen_keys))
            and len(raw_states) == protocol['state_count'] == 36),
        'raw_manifest_scope_and_hashes_exact': (
            manifest.get('schema') == 'ngas-a17ar-full-bank-raw-manifest-v1'
            and manifest.get('protocol_sha256') == protocol_sha
            and manifest.get('completed_states') == len(raw_states)
            and manifest.get('completed_actions') == action_total
            and manifest.get('files') == actual_hashes),
        'all_state_action_utility_contracts_pass': not errors,
        'origin_split_exact': Counter(row['dataset_origin'] for row in raw_states)
            == Counter({'CLEAN_NON_R12_DEVELOPMENT': 18,
                        'R12_DEVELOPMENT_EXPOSED': 18}),
        'R12_rows_are_audit_only_and_ineligible': all(
            row['audit_only'] is True and row['eligible_for_future_training'] is False
            and row['dataset_role'] == 'DEVELOPMENT_EXPOSED'
            for row in raw_states if row['dataset_origin'] == 'R12_DEVELOPMENT_EXPOSED'),
        'clean_rows_preserve_frozen_roles': all(
            row['audit_only'] is False
            and row['eligible_for_future_training'] == (row['dataset_role'] == 'TRAIN')
            for row in raw_states if row['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT'),
    }
    return raw_states, {'checks': checks, 'errors': errors, 'action_total': action_total,
                        'actual_hashes': actual_hashes}


def state_metric_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        base = {
            'state_key': state['state_key'], 'instance_id': state['instance_id'],
            'dataset_origin': state['dataset_origin'], 'dataset_role': state['dataset_role'],
            'scale': state['scale'], 'CF_level': state['CF_level'],
            'search_stage': state['search_stage'],
            'search_conditions': '|'.join(state['search_condition_tags']),
            'actions': len(state['actions']),
        }
        for utility, metrics in state['critic_rankings'].items():
            rows.append({
                **base, 'utility': utility,
                'informative': metrics['spearman']['valid'],
                'positive_best_available': metrics['best_utility'] > 0.,
                'spearman': metrics['spearman']['rho'],
                'top1_realized_rank': metrics['top1_realized_rank'],
                'top1_normalized_rank': metrics['top1_normalized_rank'],
                'top1_regret': metrics['top1_regret'],
                'best_utility': metrics['best_utility'],
                'top1_utility': metrics['top1_utility'],
                'hit_at_1': metrics['best_action_hit_at_1'],
                'hit_at_5': metrics['best_action_hit_at_5'],
                'hit_at_10': metrics['best_action_hit_at_10'],
            })
    return rows


def category_metric_rows(states: list[dict]) -> list[dict]:
    rows = []
    category_extractors = {
        'neighborhood_size': lambda action: [action['size']],
        'candidate_rule_family': lambda action: action['origin_families'],
        'repair_strategy': lambda action: [action['repair']],
    }
    for state in states:
        for dimension, extractor in category_extractors.items():
            categories = defaultdict(list)
            for action in state['actions']:
                for value in extractor(action):
                    categories[value].append(action)
            for value, actions in categories.items():
                ids = [row['action_id'] for row in actions]
                scores = [row['neural_prior'] for row in actions]
                for utility, column in UTILITY_COLUMNS.items():
                    metrics = ranking_metrics(
                        scores, [row['utility'][column] for row in actions], ids)
                    rows.append({
                        'state_key': state['state_key'], 'instance_id': state['instance_id'],
                        'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
                        'search_stage': state['search_stage'], 'dimension': dimension,
                        'value': value, 'utility': utility, 'actions': len(actions),
                        'informative': metrics['spearman']['valid'],
                        'positive_best_available': metrics['best_utility'] > 0.,
                        'spearman': metrics['spearman']['rho'],
                        'top1_normalized_rank': metrics['top1_normalized_rank'],
                        'top1_regret': metrics['top1_regret'],
                        'hit_at_1': metrics['best_action_hit_at_1'],
                        'hit_at_5': metrics['best_action_hit_at_5'],
                    })
    return rows


def summarize_rank_rows(rows: list[dict]) -> list[dict]:
    output = []
    group_specs = [('dataset_origin',), ('dataset_origin', 'scale'),
                   ('dataset_origin', 'search_stage')]
    for spec in group_specs:
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[key] for key in spec) + (row['utility'],)].append(row)
        for key, values in sorted(grouped.items()):
            valid = [row for row in values if row['informative']]
            positive = [row for row in values if row['positive_best_available']]
            output.append({
                'dimension': '+'.join(spec), 'group': '|'.join(map(str, key[:-1])),
                'utility': key[-1], 'states': len(values),
                'instances': len({row['instance_id'] for row in values}),
                'informative_states': len(valid),
                'positive_best_states': len(positive),
                'mean_spearman_informative': mean_or_none(row['spearman'] for row in valid),
                'median_spearman_informative': median_or_none(row['spearman'] for row in valid),
                'mean_top1_normalized_rank_informative': mean_or_none(
                    row['top1_normalized_rank'] for row in valid),
                'mean_top1_regret_positive_best': mean_or_none(
                    row['top1_regret'] for row in positive),
                'hit_at_1_positive_best': mean_or_none(row['hit_at_1'] for row in positive),
                'hit_at_5_positive_best': mean_or_none(row['hit_at_5'] for row in positive),
            })
    return output


def summarize_category_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset_origin'], row['dimension'], row['value'],
                 row['utility'])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        valid = [row for row in values if row['informative']]
        positive = [row for row in values if row['positive_best_available']]
        output.append({
            'dataset_origin': key[0], 'dimension': key[1], 'value': key[2],
            'utility': key[3], 'states': len(values),
            'informative_states': len(valid), 'positive_best_states': len(positive),
            'mean_actions': mean_or_none(row['actions'] for row in values),
            'mean_spearman_informative': mean_or_none(row['spearman'] for row in valid),
            'mean_top1_normalized_rank_informative': mean_or_none(
                row['top1_normalized_rank'] for row in valid),
            'mean_top1_regret_positive_best': mean_or_none(
                row['top1_regret'] for row in positive),
            'hit_at_1_positive_best': mean_or_none(row['hit_at_1'] for row in positive),
            'hit_at_5_positive_best': mean_or_none(row['hit_at_5'] for row in positive),
        })
    return output


def alignment_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        actions = state['actions']
        for agreement in state['utility_agreement']:
            rows.append({
                'state_key': state['state_key'], 'instance_id': state['instance_id'],
                'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
                'search_stage': state['search_stage'], 'left': agreement['left'],
                'right': agreement['right'], 'informative': agreement['spearman']['valid'],
                'spearman': agreement['spearman']['rho'],
                'same_top1': agreement['same_top1'],
                'top5_overlap': agreement['top5_overlap'],
                'top10_overlap': agreement['top10_overlap'],
            })
        u1 = max(row['utility']['U1_short_horizon_best_gain'] for row in actions)
        u2_top = max(actions, key=lambda row: row['utility']['U2_horizon_gain_per_second'])
        u1_top = max(actions, key=lambda row: row['utility']['U1_short_horizon_best_gain'])
        rows.append({
            'state_key': state['state_key'], 'instance_id': state['instance_id'],
            'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
            'search_stage': state['search_stage'], 'left': 'U1_SHORT_HORIZON',
            'right': 'U2_COST_NORMALIZED_TOP1', 'informative': u1 > 0.,
            'spearman': None, 'same_top1': u1_top['action_id'] == u2_top['action_id'],
            'top5_overlap': None, 'top10_overlap': None,
        })
    return rows


def portfolio_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        actions = state['actions']
        neural = sorted(actions, key=lambda row: (-row['neural_prior'], row['action_id']))
        adjusted = sorted(
            actions, key=lambda row: (-row['portfolio_adjusted_probability'], row['action_id']))
        neural_rank = {row['action_id']: index + 1 for index, row in enumerate(neural)}
        adjusted_rank = {row['action_id']: index + 1 for index, row in enumerate(adjusted)}
        interaction = state['portfolio_interaction']
        rows.append({
            'state_key': state['state_key'], 'instance_id': state['instance_id'],
            'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
            'search_stage': state['search_stage'],
            'top1_changed': interaction['top1_changed'],
            'top5_overlap': interaction['top5_overlap'],
            'neural_top1_adjusted_rank': adjusted_rank[neural[0]['action_id']],
            'portfolio_top1_neural_rank': neural_rank[adjusted[0]['action_id']],
            'mean_neural_top5_rank_shift': statistics.fmean(
                adjusted_rank[row['action_id']] - neural_rank[row['action_id']]
                for row in neural[:5]),
            'production_vs_neural_disagreement': (
                interaction['production_selected_action_id'] != neural[0]['action_id']),
            'production_vs_portfolio_disagreement': (
                interaction['production_selected_action_id'] != adjusted[0]['action_id']),
            'neural_U0': interaction['neural_top1_U0'],
            'portfolio_U0': interaction['portfolio_top1_U0'],
            'selected_U0': interaction['production_selected_U0'],
            'portfolio_minus_neural_U0': (
                interaction['portfolio_top1_U0'] - interaction['neural_top1_U0']),
            'selected_minus_neural_U0': (
                interaction['production_selected_U0'] - interaction['neural_top1_U0']),
            'neural_U1': interaction['neural_top1_U1'],
            'portfolio_U1': interaction['portfolio_top1_U1'],
            'selected_U1': interaction['production_selected_U1'],
            'portfolio_minus_neural_U1': (
                interaction['portfolio_top1_U1'] - interaction['neural_top1_U1']),
            'selected_minus_neural_U1': (
                interaction['production_selected_U1'] - interaction['neural_top1_U1']),
        })
    return rows


def trial_rows(states: list[dict]) -> tuple[list[dict], list[dict]]:
    action_rows = []
    for state in states:
        for action in state['actions']:
            makespans = [float(row['candidate_makespan']) for row in action['direct_trials']]
            signed = [float(row['signed_improvement']) for row in action['direct_trials']]
            final_best = min(makespans)
            previous_best = math.inf
            for trial, makespan in enumerate(makespans, 1):
                prefix_best = min(makespans[:trial])
                raw_marginal = 0. if trial == 1 else max(previous_best - prefix_best, 0.)
                positive_prefix = max(float(state['captured_current_makespan']) - prefix_best, 0.)
                positive_final = max(float(state['captured_current_makespan']) - final_best, 0.)
                action_rows.append({
                    'state_key': state['state_key'], 'instance_id': state['instance_id'],
                    'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
                    'search_stage': state['search_stage'], 'repair': action['repair'],
                    'size': action['size'], 'action_id': action['action_id'], 'trial': trial,
                    'prefix_best_makespan': prefix_best,
                    'raw_marginal_best_reduction': raw_marginal,
                    'final_best_changes_after_k': prefix_best > final_best,
                    'raw_quality_loss_vs_eight': prefix_best - final_best,
                    'positive_gain_at_k': positive_prefix,
                    'positive_gain_loss_vs_eight': positive_final - positive_prefix,
                    'new_prefix_best': trial == 1 or makespan < previous_best,
                    'signed_trial_variance': statistics.pvariance(signed),
                })
                previous_best = prefix_best
    grouped = defaultdict(list)
    dimensions = ('overall', 'dataset_origin', 'repair', 'scale', 'search_stage')
    for row in action_rows:
        for dimension in dimensions:
            value = 'ALL' if dimension == 'overall' else row[dimension]
            grouped[(dimension, value, row['trial'])].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        summary.append({
            'dimension': key[0], 'group': key[1], 'trial': key[2],
            'actions': len(values), 'instances': len({row['instance_id'] for row in values}),
            'mean_raw_marginal_best_reduction': mean_or_none(
                row['raw_marginal_best_reduction'] for row in values),
            'probability_best_changes_after_k': mean_or_none(
                row['final_best_changes_after_k'] for row in values),
            'mean_raw_quality_loss_vs_eight': mean_or_none(
                row['raw_quality_loss_vs_eight'] for row in values),
            'mean_positive_gain_at_k': mean_or_none(row['positive_gain_at_k'] for row in values),
            'mean_positive_gain_loss_vs_eight': mean_or_none(
                row['positive_gain_loss_vs_eight'] for row in values),
            'no_new_best_probability': mean_or_none(not row['new_prefix_best'] for row in values),
            'mean_signed_trial_variance': mean_or_none(
                row['signed_trial_variance'] for row in values),
        })
    return action_rows, summary


def format_number(value, digits: int = 4) -> str:
    if value is None:
        return 'NA'
    return f'{float(value):.{digits}f}'


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)
    return '\n'.join(lines)


def write_reports(rank_summary: list[dict], category_summary: list[dict],
                  state_rows: list[dict], align: list[dict],
                  portfolio: list[dict], trials: list[dict]) -> None:
    origin_rank = [row for row in rank_summary if row['dimension'] == 'dataset_origin']
    ranking_table = markdown_table(
        ['origin', 'utility', 'states', 'informative', 'positive-best',
         'mean rho', 'mean regret*', 'hit@1*', 'hit@5*'], [[
            row['group'], row['utility'], row['states'], row['informative_states'],
            row['positive_best_states'], format_number(row['mean_spearman_informative']),
            format_number(row['mean_top1_regret_positive_best']),
            format_number(row['hit_at_1_positive_best']),
            format_number(row['hit_at_5_positive_best']),
        ] for row in origin_rank])
    scale_stage = [row for row in rank_summary
                   if row['utility'] == 'U0_IMMEDIATE'
                   and row['dimension'] in {'dataset_origin+scale',
                                            'dataset_origin+search_stage'}]
    scale_table = markdown_table(
        ['breakdown', 'group', 'states', 'informative', 'mean rho', 'mean regret*'], [[
            row['dimension'], row['group'], row['states'], row['informative_states'],
            format_number(row['mean_spearman_informative']),
            format_number(row['mean_top1_regret_positive_best']),
        ] for row in scale_stage])
    interval_rows = []
    for origin in ORIGINS:
        values = [row for row in state_rows if row['dataset_origin'] == origin
                  and row['utility'] == 'U0_IMMEDIATE']
        informative = instance_interval(
            [{**row, 'value': float(row['informative'])} for row in values],
            'value', seed_key=f'{origin}:informative')
        regret = instance_interval(
            [row for row in values if row['positive_best_available']],
            'top1_regret', seed_key=f'{origin}:regret')
        interval_rows.append([
            origin,
            f"{format_number(informative['mean'])} "
            f"[{format_number(informative['lower_95'])}, "
            f"{format_number(informative['upper_95'])}]",
            f"{format_number(regret['mean'])} "
            f"[{format_number(regret['lower_95'])}, "
            f"{format_number(regret['upper_95'])}]",
            informative['instances'], regret['instances'],
        ])
    REPORTS['ranking'].write_text(f'''# NGAS A1.7A-R critic ranking audit

The audit enumerated all 12,775 unique joint actions from 36 frozen states. The
clean non-R12 and R12 development-exposed origins are reported separately.
Only states with nonconstant realized utility enter Spearman summaries. Metrics
marked `*` use only states where a positive best action exists; all-zero utility
states are retained in the state count and are not credited as substantive hits.

{ranking_table}

## Scale and stage

{scale_table}

## Instance-cluster uncertainty

{markdown_table(['origin', 'informative rate (95% CI)',
                 'U0 regret given positive best (95% CI)',
                 'rate instances', 'regret instances'], interval_rows)}

Intervals use 10,000 deterministic percentile resamples of unweighted instance
means. They are exploratory uncertainty summaries; no formal hypothesis test or
multiplicity family is declared.

The immediate-utility signal is sparse: the critic's predicted top action did not
hit the best positive U0 action in any informative origin-specific state. Positive
rank correlations are weak for U0/U1 and materially stronger for U3 signed
robustness. These are state-level development diagnostics, not solver-level
generalization evidence. R12 remains development-exposed.

Machine-readable sources: `{relative(DERIVED / 'full_bank_state_metrics.csv')}`,
`{relative(DERIVED / 'critic_ranking_summary.csv')}`, and
`{relative(DERIVED / 'critic_category_breakdowns.csv')}`. The latter supplies the
required neighborhood-size, candidate-family, and repair breakdowns.
''')

    pair_rows = []
    for origin in ORIGINS:
        values = [row for row in align if row['dataset_origin'] == origin
                  and row['left'] == 'U0_IMMEDIATE'
                  and row['right'] == 'U1_SHORT_HORIZON']
        valid = [row for row in values if row['informative']]
        cost = [row for row in align if row['dataset_origin'] == origin
                and row['right'] == 'U2_COST_NORMALIZED_TOP1']
        pair_rows.append([
            origin, len(values), len(valid),
            format_number(mean_or_none(row['spearman'] for row in valid)),
            format_number(mean_or_none(row['same_top1'] for row in values)),
            format_number(mean_or_none(row['top5_overlap'] for row in values)),
            format_number(mean_or_none(row['same_top1'] for row in cost if row['informative'])),
        ])
    REPORTS['alignment'].write_text(f'''# NGAS A1.7A-R utility alignment

{markdown_table(['origin', 'states', 'informative U0/U1', 'mean rho',
                 'same top1', 'mean top5 overlap', 'U1/U2 same top1*'], pair_rows)}

`*` conditions on a state having positive U1 utility. U0 and U1 often share the
same top action because most state banks contain no positive action; the
conditional rank correlation is therefore the more useful diagnostic. U2 uses
U1 improvement per measured repair-and-decode second and can reorder actions with
the same raw horizon gain. No C1-v2 fitting or solver behavior change occurred.

Machine-readable source: `{relative(DERIVED / 'utility_alignment.csv')}`.
''')

    port_rows = []
    for origin in ORIGINS:
        values = [row for row in portfolio if row['dataset_origin'] == origin]
        changed = [row for row in values if row['top1_changed']]
        port_rows.append([
            origin, len(values), format_number(mean_or_none(row['top1_changed'] for row in values)),
            format_number(mean_or_none(row['top5_overlap'] for row in values)),
            format_number(mean_or_none(row['production_vs_neural_disagreement'] for row in values)),
            format_number(mean_or_none(row['portfolio_minus_neural_U0'] for row in values)),
            format_number(mean_or_none(row['portfolio_minus_neural_U0'] for row in changed)),
            format_number(mean_or_none(row['selected_minus_neural_U0'] for row in values)),
        ])
    port_breakdowns = []
    for dimension in ('scale', 'search_stage'):
        grouped = defaultdict(list)
        for row in portfolio:
            grouped[(row['dataset_origin'], row[dimension])].append(row)
        for (origin, value), values in sorted(grouped.items()):
            port_breakdowns.append([
                dimension, origin, value, len(values),
                format_number(mean_or_none(row['top1_changed'] for row in values)),
                format_number(mean_or_none(
                    row['portfolio_minus_neural_U0'] for row in values)),
            ])
    REPORTS['portfolio'].write_text(f'''# NGAS A1.7A-R critic/portfolio interaction

{markdown_table(['origin', 'states', 'top1 change rate', 'top5 overlap',
                 'final/neural disagreement', 'portfolio-neural U0',
                 'conditional U0 change', 'selected-neural U0'], port_rows)}

## Scale and stage dependence

{markdown_table(['dimension', 'origin', 'group', 'states', 'change rate',
                 'portfolio-neural U0'], port_breakdowns)}

The portfolio changes the deterministic neural top-1 rarely. The archived final
selection differs from neural top-1 in every audited state because production
selection samples from the adjusted distribution; its utility comparison therefore
includes both portfolio weighting and frozen exploration. Conditional results are
reported separately so a rare intervention is not averaged into an apparent broad
benefit. Scale/stage rows and rank promotions/demotions are retained in the source.

Machine-readable source: `{relative(DERIVED / 'portfolio_interaction.csv')}`.
''')

    overall = [row for row in trials if row['dimension'] == 'overall']
    trial_table = markdown_table(
        ['trial cap', 'marginal best reduction', 'P(best changes later)',
         'quality loss vs 8', 'positive-gain loss', 'no-new-best rate'], [[
            row['trial'], format_number(row['mean_raw_marginal_best_reduction']),
            format_number(row['probability_best_changes_after_k']),
            format_number(row['mean_raw_quality_loss_vs_eight']),
            format_number(row['mean_positive_gain_loss_vs_eight']),
            format_number(row['no_new_best_probability']),
        ] for row in overall])
    trial_breakdowns = [row for row in trials
                        if row['dimension'] in {'repair', 'scale', 'search_stage'}
                        and row['trial'] == 4]
    breakdown_table = markdown_table(
        ['dimension', 'group', 'actions', 'cap-4 loss', 'trial variance'], [[
            row['dimension'], row['group'], row['actions'],
            format_number(row['mean_raw_quality_loss_vs_eight']),
            format_number(row['mean_signed_trial_variance']),
        ] for row in trial_breakdowns])
    REPORTS['trials'].write_text(f'''# NGAS A1.7A-R candidate-trials audit

Every full-bank action retained exactly eight ordered repair/decode outcomes.
Quality loss is the best candidate makespan under a prefix cap minus the best of
all eight; positive-gain loss additionally clips at the captured incumbent.

{trial_table}

## Frozen subgroup diagnostics at cap 4

{breakdown_table}

Caps 1, 2, and 4 can be read directly from the corresponding rows. The
`no-new-best` rate is the wasted-trial proxy: after trial 1, it records the fraction
of trials that fail to improve the current prefix-best candidate. Repair, scale,
search-stage, and origin breakdowns are stored in the source CSV. This audit can
inform a separately frozen A1.7B racing protocol; it does not implement racing.

Machine-readable source: `{relative(DERIVED / 'candidate_trial_curve.csv')}`.
''')


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(RAW_MANIFEST.read_text())
    progress = json.loads(PROGRESS.read_text())
    states, raw = validate_raw(protocol, manifest)
    expected_keys = {row['state_key'] for row in protocol['states']}
    session, session_paths = session_audit(manifest, expected_keys)
    ledger = verify_exposure_ledger(LEDGER)
    formal_ledger = [row for row in ledger if row.get('phase') == FORMAL_PHASE
                     and row.get('command') == FORMAL_COMMAND]
    expected_instances = Counter(row['instance_id'] for row in protocol['states'])
    execution_commit = session['execution_commit']
    ledger_checks = {
        'ledger_hash_chain_valid': True,
        'exact_formal_exposure_multiset': (
            len(formal_ledger) == len(protocol['states'])
            and Counter(row.get('instance_id') for row in formal_ledger) == expected_instances),
        'formal_exposures_match_frozen_identity_and_execution': all(
            row.get('permitted') is True
            and row.get('requested_purpose') == 'diagnostic'
            and row.get('checkpoint_identifier') == protocol['checkpoint_sha256']
            and row.get('git_sha') == execution_commit
            and row.get('instance_content_sha256')
                == next(item['instance_sha256'] for item in protocol['states']
                        if item['instance_id'] == row.get('instance_id'))
            for row in formal_ledger),
    }
    source_checks = {
        'protocol_frozen_before_results': (
            protocol.get('status') == 'FROZEN_BEFORE_FULL_BANK_RESULTS'),
        'frozen_sources_match': all(
            sha256_file(ROOT / path) == expected
            for path, expected in protocol['source_hashes'].items()),
        'checkpoint_and_upstream_artifacts_match': (
            sha256_file(ROOT / protocol['checkpoint_path']) == protocol['checkpoint_sha256']
            and sha256_file(ROOT / protocol['collection_audit_path'])
                == protocol['collection_audit_sha256']
            and sha256_file(ROOT / protocol['R12_state_protocol_path'])
                == protocol['R12_state_protocol_sha256']),
        'progress_exact_complete': (
            progress.get('status') == 'COMPLETE'
            and progress.get('completed_states') == progress.get('expected_states') == 36
            and progress.get('completed_actions') == manifest['completed_actions'] == 12775
            and progress.get('current_state') is None
            and progress.get('protocol_sha256') == sha256_file(PROTOCOL)),
        'locked_boundaries_preserved': (
            progress.get('R13') == 'LOCKED_NO_ACCESS'
            and progress.get('R14') == 'LOCKED_NO_ACCESS'
            and progress.get('RCIAS_CB1_CORE45') == 'EXCLUDED'
            and progress.get('gurobi_run') is False
            and all(row.get('boundaries') == protocol['boundaries'] for row in states)),
    }
    checks = {**source_checks, **raw['checks'], **session['checks'], **ledger_checks}

    state_rows = state_metric_rows(states)
    category_rows = category_metric_rows(states)
    rank_summary = summarize_rank_rows(state_rows)
    category_summary = summarize_category_rows(category_rows)
    align = alignment_rows(states)
    portfolio = portfolio_rows(states)
    action_trials, trial_summary = trial_rows(states)

    atomic_csv(DERIVED / 'full_bank_state_metrics.csv', state_rows)
    atomic_csv(DERIVED / 'critic_ranking_summary.csv', rank_summary)
    atomic_csv(DERIVED / 'critic_category_breakdowns.csv', category_summary)
    atomic_csv(DERIVED / 'utility_alignment.csv', align)
    atomic_csv(DERIVED / 'portfolio_interaction.csv', portfolio)
    atomic_csv(DERIVED / 'candidate_trial_curve.csv', trial_summary)
    write_reports(rank_summary, category_summary, state_rows, align, portfolio,
                  trial_summary)

    inference = {}
    for origin in ORIGINS:
        origin_rows = [row for row in state_rows
                       if row['dataset_origin'] == origin
                       and row['utility'] == 'U0_IMMEDIATE']
        inference[origin] = {
            'informative_state_rate': instance_interval(
                [{**row, 'value': float(row['informative'])} for row in origin_rows],
                'value', seed_key=f'{origin}:informative'),
            'top1_regret_given_positive_best': instance_interval(
                [row for row in origin_rows if row['positive_best_available']],
                'top1_regret', seed_key=f'{origin}:regret'),
            'portfolio_minus_neural_U0': instance_interval(
                [row for row in portfolio if row['dataset_origin'] == origin],
                'portfolio_minus_neural_U0', seed_key=f'{origin}:portfolio'),
        }
    summary = {
        'schema': 'ngas-a17ar-full-bank-analysis-v1',
        'statistical_scope': {
            'state_level': 'exploratory diagnostic',
            'instance_level': 'unweighted instance-cluster bootstrap interval',
            'solver_level': 'no generalization inference; R12 is development-exposed',
            'formal_hypothesis_tests': 0,
            'multiplicity_correction': 'not applicable',
        },
        'instance_cluster_intervals': inference,
        'origin_action_counts': {
            origin: sum(len(row['actions']) for row in states
                        if row['dataset_origin'] == origin) for origin in ORIGINS},
        'origin_positive_U0_actions': {
            origin: sum(action['utility']['U0_immediate_best_gain'] > 0.
                        for row in states if row['dataset_origin'] == origin
                        for action in row['actions']) for origin in ORIGINS},
        'rank_summary': rank_summary,
        'candidate_trial_caps': [row for row in trial_summary
                                 if row['dimension'] == 'overall'
                                 and row['trial'] in {1, 2, 4, 8}],
        'boundaries': protocol['boundaries'],
    }
    atomic_json(DERIVED / 'full_bank_analysis.json', summary)

    status = 'PASS' if all(checks.values()) else 'FAIL'
    audit = {
        'schema': 'ngas-a17ar-full-bank-completion-audit-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'checks': checks,
        'protocol_sha256': sha256_file(PROTOCOL),
        'execution_commit': execution_commit,
        'checkpoint_sha256': protocol['checkpoint_sha256'],
        'candidate_bank_hash': protocol['candidate_bank_hash'],
        'completed_states': len(states), 'completed_actions': raw['action_total'],
        'raw_bytes': sum(path.stat().st_size for path in RAW.glob('*.json')),
        'raw_manifest_sha256': sha256_file(RAW_MANIFEST),
        'formal_log_sha256': sha256_file(OUT / 'logs/full_bank_formal.log'),
        'session_integrity': session,
        'session_log_sha256': {
            relative(path): sha256_file(path) for path in session_paths},
        'exposure_ledger_sha256': sha256_file(LEDGER),
        'raw_validation_errors': raw['errors'],
        'dataset_origin_counts': dict(Counter(row['dataset_origin'] for row in states)),
        'boundaries': protocol['boundaries'],
        'next_stage': ('PRIOR_STALENESS_PROTOCOL'
                       if status == 'PASS' else 'STOP_FULL_BANK_INTEGRITY_FAILURE'),
    }
    atomic_json(AUDIT, audit)
    print(json.dumps({
        'status': status, 'states': len(states), 'actions': raw['action_total'],
        'checks_passed': sum(checks.values()), 'checks_total': len(checks),
        'audit': relative(AUDIT),
        'reports': {name: relative(path) for name, path in REPORTS.items()},
    }, indent=2))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
