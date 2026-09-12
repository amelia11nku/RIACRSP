#!/usr/bin/env python3
"""Validate and summarize the frozen A1.7A-R prior-staleness audit."""
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
PROTOCOL = ROOT / 'artifacts/ngas_a17ar/prior_staleness_protocol_manifest.json'
PROGRESS = OUT / 'diagnostics/prior_staleness_progress.json'
RAW = OUT / 'diagnostics/prior_staleness_raw'
RAW_MANIFEST = OUT / 'diagnostics/prior_staleness_raw_manifest.json'
SESSION_DIR = OUT / 'integrity/prior_staleness_sessions'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
AUDIT = OUT / 'audit/prior_staleness_completion_audit.json'
DERIVED = OUT / 'derived'
REPORT = ROOT / 'reports/ngas_a17ar_prior_staleness.md'
FORMAL_OWNER = 'NGAS_A1_7AR_STALENESS_OWNER_V1'
FORMAL_PHASE = 'NGAS_A1_7A_R_STAGE7_STALENESS'
FORMAL_COMMAND = 'scripts/run_ngas_a17ar_prior_staleness.py'
ORIGINS = ('CLEAN_NON_R12_DEVELOPMENT', 'R12_DEVELOPMENT_EXPOSED')


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


def close(left, right, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def mean_or_none(values) -> float | None:
    clean = [float(value) for value in values
             if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def format_number(value, digits: int = 4) -> str:
    return 'NA' if value is None else f'{float(value):.{digits}f}'


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |'
                 for row in rows)
    return '\n'.join(lines)


def instance_interval(rows: list[dict], column: str, *, seed_key: str,
                      resamples: int = 10_000) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(column)
        if value is not None and math.isfinite(float(value)):
            grouped[row['instance_id']].append(float(value))
    cluster_values = [statistics.fmean(values) for _, values in sorted(grouped.items())]
    if not cluster_values:
        return {'mean': None, 'lower_95': None, 'upper_95': None,
                'instances': 0, 'resamples': resamples}
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choices(cluster_values, k=len(cluster_values)))
                  for _ in range(resamples))
    return {
        'mean': statistics.fmean(cluster_values),
        'lower_95': boot[int(.025 * resamples)],
        'upper_95': boot[int(.975 * resamples) - 1],
        'instances': len(cluster_values), 'resamples': resamples,
    }


def search_stage(full_bank_path: Path) -> str:
    return json.loads(full_bank_path.read_text())['search_stage']


def session_audit(manifest: dict, expected_keys: set[str]) -> tuple[dict, list[Path]]:
    paths = sorted(SESSION_DIR.glob('*.jsonl'))
    sessions = []
    events = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        sessions.append((path, rows))
        events.extend(rows)
    acquired = [row for row in events if row.get('event') == 'lock_acquired']
    released = [row for row in events if row.get('event') == 'lock_released']
    starts = [row for row in events if row.get('event') == 'staleness_state_started']
    completions = [row for row in events if row.get('event') == 'staleness_state_completed']
    terminal = [row for row in events if row.get('event') == 'staleness_audit_complete']
    execution_commit = acquired[0].get('implementation_commit') if len(acquired) == 1 else None
    commit_exists = bool(execution_commit) and subprocess.run(
        ['git', 'cat-file', '-e', f'{execution_commit}^{{commit}}'], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    completion_hashes = {row.get('raw_path'): row.get('raw_sha256')
                         for row in completions}
    checks = {
        'prior_staleness_lock_released': not (OUT / 'integrity/prior_staleness.lock').exists(),
        'exactly_one_process_session': len(sessions) == 1,
        'event_sequences_contiguous': all(
            [row.get('sequence') for row in rows] == list(range(1, len(rows) + 1))
            for _, rows in sessions),
        'one_logical_formal_owner': (
            {row.get('formal_owner_id') for row in events} == {FORMAL_OWNER}),
        'one_successful_acquire_release': (
            len(acquired) == len(released) == 1
            and released[0].get('status') == 'SUCCESS'),
        'formal_starts_exact_once': (
            Counter(row.get('state_key') for row in starts)
            == Counter({key: 1 for key in expected_keys})),
        'formal_completions_exact_once': (
            Counter(row.get('state_key') for row in completions)
            == Counter({key: 1 for key in expected_keys})),
        'completion_raw_hashes_match_manifest': completion_hashes == manifest['files'],
        'one_terminal_completion_event': (
            len(terminal) == 1
            and terminal[0].get('states') == 36
            and terminal[0].get('offsets') == 180
            and terminal[0].get('action_offset_evaluations')
                == manifest['stale_action_offset_evaluations']),
        'execution_commit_exists': commit_exists,
    }
    return {
        'schema': 'ngas-a17ar-prior-staleness-session-integrity-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'execution_commit': execution_commit,
        'sessions': [{
            'path': relative(path),
            'session_id': rows[0].get('session_id') if rows else None,
            'event_count': len(rows),
            'heartbeat_count': sum(row.get('event') == 'heartbeat' for row in rows),
        } for path, rows in sessions],
    }, paths


def validate_offset(state: dict, offset: dict, full_bank: dict,
                    protocol: dict) -> list[str]:
    errors = []
    actions = offset.get('stale_action_evaluations', [])
    action_ids = [row.get('action_id') for row in actions]
    full_actions = full_bank['actions']
    full_by_id = {row['action_id']: row for row in full_actions}
    if (offset.get('offset') not in protocol['offsets']
            or offset.get('iteration') != state['base_iteration'] + offset.get('offset', -1)):
        errors.append('offset_iteration_mismatch')
    if (len(actions) != offset.get('persistent_stale_actions_evaluated')
            or len(actions) != offset.get('full_unique_stale_bank_actions')
            or len(actions) != len(full_actions)
            or set(action_ids) != set(full_by_id)
            or len(action_ids) != len(set(action_ids))):
        errors.append('persistent_full_bank_action_scope_mismatch')
    trial_seed_rows = []
    for action in actions:
        trials = action.get('trials', [])
        if (len(trials) != 8
                or [row.get('trial') for row in trials] != list(range(1, 9))):
            errors.append('action_trials_not_exactly_ordered_eight')
            continue
        trial_seed_rows.append(tuple(row.get('repair_rng_seed') for row in trials))
        best = min(float(row['candidate_makespan']) for row in trials)
        expected_u0 = max(0., float(offset['current_makespan']) - best)
        if (not close(action.get('best_candidate_makespan'), best)
                or not close(action.get('U0_immediate_best_gain'), expected_u0)):
            errors.append('action_U0_reduction_mismatch')
        frozen = full_by_id.get(action.get('action_id'))
        if frozen and (action.get('size') != frozen.get('size')
                       or action.get('repair') != frozen.get('repair')
                       or action.get('target_operations') != frozen.get('target_operations')):
            errors.append('action_semantics_mismatch')
    if trial_seed_rows and len(set(trial_seed_rows)) != 1:
        errors.append('matched_random_numbers_mismatch')
    if actions:
        scores = [float(full_by_id[action_id]['neural_prior']) for action_id in action_ids]
        utility = [float(row['U0_immediate_best_gain']) for row in actions]
        recomputed = ranking_metrics(scores, utility, action_ids)
        observed = offset.get('stale_ranking_quality_U0', {})
        for key in ('predicted_top1_action_id', 'top1_realized_rank',
                    'best_action_hit_at_1', 'best_action_hit_at_5',
                    'best_action_hit_at_10'):
            if observed.get(key) != recomputed.get(key):
                errors.append('stale_ranking_metric_mismatch')
                break
        for key in ('top1_normalized_rank', 'top1_regret', 'best_utility', 'top1_utility'):
            if not close(observed.get(key), recomputed.get(key)):
                errors.append('stale_ranking_metric_mismatch')
                break
        if not close(observed.get('spearman', {}).get('rho'),
                     recomputed.get('spearman', {}).get('rho')):
            errors.append('stale_ranking_spearman_mismatch')
    drift = offset.get('representation_drift', {})
    overlap = offset.get('stale_vs_fresh', {})
    if not (0. <= float(drift.get('graph_edge_jaccard', -1.)) <= 1.
            and 0. <= float(overlap.get('semantic_bank_jaccard', -1.)) <= 1.
            and 0 <= int(overlap.get('top5_overlap', -1)) <= 5
            and 0 <= int(overlap.get('top10_overlap', -1)) <= 10):
        errors.append('drift_range_mismatch')
    if offset.get('offset') == 0:
        zero_checks = (
            close(drift.get('operation_feature_relative_l2'), 0.)
            and close(drift.get('operation_feature_mean_absolute'), 0.)
            and close(drift.get('graph_edge_jaccard'), 1.)
            and overlap.get('top1_same_semantics') is True
            and close(overlap.get('semantic_bank_jaccard'), 1.)
            and close(overlap.get('mean_absolute_prior_drift_common'), 0.)
            and close(overlap.get('mean_absolute_advantage_drift_common'), 0.)
            and close(overlap.get('mean_absolute_percentile_rank_drift_common'), 0.))
        if not zero_checks:
            errors.append('offset_zero_identity_mismatch')
    quality = offset.get('selected_action_quality', {})
    if actions:
        best_stale = max(float(row['U0_immediate_best_gain']) for row in actions)
        best_union = max(best_stale, float(quality.get('fresh_neural_top1_U0', -math.inf)))
        if (not close(quality.get('best_observed_union_U0'), best_union)
                or not close(quality.get('stale_top1_regret_union'),
                             best_union - float(quality.get('stale_neural_top1_U0', 0.)))
                or not close(quality.get('fresh_top1_regret_union'),
                             best_union - float(quality.get('fresh_neural_top1_U0', 0.)))
                or not close(quality.get('archived_selected_regret_union'),
                             best_union - float(quality.get('archived_selected_matched_U0', 0.)))):
            errors.append('selected_action_quality_mismatch')
    if not finite_tree(offset):
        errors.append('nonfinite_offset_payload')
    return sorted(set(errors))


def validate_raw(protocol: dict, manifest: dict) -> tuple[list[dict], dict]:
    expected = {row['state_key']: row for row in protocol['states']}
    protocol_sha = sha256_file(PROTOCOL)
    actual_hashes = {}
    states = []
    errors = []
    total_evaluations = 0
    stages = {}
    for path in sorted(RAW.glob('*.json')):
        actual_hashes[relative(path)] = sha256_file(path)
        state = json.loads(path.read_text())
        states.append(state)
        frozen = expected.get(state.get('state_key'))
        state_errors = []
        if frozen is None:
            state_errors.append('unexpected_state_key')
        else:
            for key in ('state_key', 'state_sha256', 'instance_id', 'instance_sha256',
                        'instance_relative_path', 'formal_raw_path', 'formal_raw_sha256',
                        'full_bank_raw_path', 'full_bank_raw_sha256', 'dataset_origin',
                        'dataset_role', 'audit_only', 'eligible_for_future_training',
                        'scale', 'CF_level', 'seed'):
                if state.get(key) != frozen.get(key):
                    state_errors.append(f'{key}_mismatch')
        if (state.get('schema') != 'ngas-a17ar-prior-staleness-state-v1'
                or state.get('status') != 'COMPLETE'
                or state.get('protocol_sha256') != protocol_sha
                or state.get('offsets') != protocol['offsets']
                or state.get('refresh_interval') != protocol['refresh_interval']
                or state.get('boundaries') != protocol['boundaries']
                or state.get('formal_solver_budget_inclusion') is not False):
            state_errors.append('top_level_contract_mismatch')
        if frozen is not None:
            full_path = ROOT / frozen['full_bank_raw_path']
            if sha256_file(full_path) != frozen['full_bank_raw_sha256']:
                state_errors.append('full_bank_dependency_hash_mismatch')
                full_bank = {'actions': [], 'search_stage': 'UNKNOWN'}
            else:
                full_bank = json.loads(full_path.read_text())
                stages[state['state_key']] = full_bank['search_stage']
            offset_rows = state.get('offset_results', [])
            if ([row.get('offset') for row in offset_rows] != protocol['offsets']
                    or len(offset_rows) != len(protocol['offsets'])):
                state_errors.append('offset_scope_mismatch')
            else:
                for row in offset_rows:
                    total_evaluations += int(row.get('persistent_stale_actions_evaluated', 0))
                    state_errors.extend(validate_offset(state, row, full_bank, protocol))
        if not finite_tree(state):
            state_errors.append('nonfinite_state_payload')
        if state_errors:
            errors.append({'state_key': state.get('state_key'),
                           'errors': sorted(set(state_errors))})
    keys = [row.get('state_key') for row in states]
    checks = {
        'raw_scope_exact': (set(keys) == set(expected) and len(keys) == len(set(keys))
                            and len(states) == protocol['state_count'] == 36),
        'raw_manifest_scope_and_hashes_exact': (
            manifest.get('schema') == 'ngas-a17ar-prior-staleness-raw-manifest-v1'
            and manifest.get('protocol_sha256') == protocol_sha
            and manifest.get('completed_states') == len(states) == 36
            and manifest.get('completed_offsets') == 180
            and manifest.get('stale_action_offset_evaluations') == total_evaluations
            and manifest.get('files') == actual_hashes),
        'all_state_offset_action_contracts_pass': not errors,
        'origin_split_exact': Counter(row['dataset_origin'] for row in states)
            == Counter({'CLEAN_NON_R12_DEVELOPMENT': 18,
                        'R12_DEVELOPMENT_EXPOSED': 18}),
        'R12_rows_are_audit_only_and_ineligible': all(
            row['audit_only'] is True and row['eligible_for_future_training'] is False
            and row['dataset_role'] == 'DEVELOPMENT_EXPOSED'
            for row in states if row['dataset_origin'] == 'R12_DEVELOPMENT_EXPOSED'),
        'clean_rows_preserve_frozen_roles': all(
            row['audit_only'] is False
            and row['eligible_for_future_training'] == (row['dataset_role'] == 'TRAIN')
            for row in states if row['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT'),
    }
    return states, {'checks': checks, 'errors': errors,
                    'total_evaluations': total_evaluations,
                    'actual_hashes': actual_hashes, 'stages': stages}


def metric_rows(states: list[dict], stages: dict[str, str]) -> list[dict]:
    rows = []
    for state in states:
        for offset in state['offset_results']:
            rank = offset['stale_ranking_quality_U0']
            overlap = offset['stale_vs_fresh']
            representation = offset['representation_drift']
            quality = offset['selected_action_quality']
            rows.append({
                'state_key': state['state_key'], 'instance_id': state['instance_id'],
                'dataset_origin': state['dataset_origin'],
                'dataset_role': state['dataset_role'], 'scale': state['scale'],
                'CF_level': state['CF_level'], 'search_stage': stages[state['state_key']],
                'offset': offset['offset'], 'iteration': offset['iteration'],
                'actions': offset['persistent_stale_actions_evaluated'],
                'fresh_bank_actions': offset['fresh_bank_actions'],
                'informative_U0': rank['spearman']['valid'],
                'positive_best_U0': rank['best_utility'] > 0.,
                'stale_U0_spearman': rank['spearman']['rho'],
                'stale_top1_normalized_rank': rank['top1_normalized_rank'],
                'stale_top1_regret': rank['top1_regret'],
                'semantic_bank_jaccard': overlap['semantic_bank_jaccard'],
                'top1_same_semantics': overlap['top1_same_semantics'],
                'top5_overlap_fraction': overlap['top5_overlap'] / 5.,
                'top10_overlap_fraction': overlap['top10_overlap'] / 10.,
                'common_rank_spearman': overlap['percentile_rank_spearman_common']['rho'],
                'common_prior_mean_absolute_drift': overlap['mean_absolute_prior_drift_common'],
                'common_advantage_mean_absolute_drift': overlap['mean_absolute_advantage_drift_common'],
                'common_percentile_rank_mean_absolute_drift': overlap['mean_absolute_percentile_rank_drift_common'],
                'operation_feature_relative_l2': representation['operation_feature_relative_l2'],
                'operation_feature_mean_absolute': representation['operation_feature_mean_absolute'],
                'graph_edge_jaccard': representation['graph_edge_jaccard'],
                'critical_signature_changed': representation['critical_signature_changed'],
                'dominant_bottleneck_changed': representation['dominant_bottleneck_changed'],
                'stale_top1_U0': quality['stale_neural_top1_U0'],
                'fresh_top1_U0': quality['fresh_neural_top1_U0'],
                'archived_selected_U0': quality['archived_selected_matched_U0'],
                'stale_top1_regret_union': quality['stale_top1_regret_union'],
                'fresh_top1_regret_union': quality['fresh_top1_regret_union'],
                'archived_selected_regret_union': quality['archived_selected_regret_union'],
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        keys = {
            'dataset_origin': row['dataset_origin'],
            'dataset_origin+scale': f"{row['dataset_origin']}|{row['scale']}",
            'dataset_origin+search_stage': (
                f"{row['dataset_origin']}|{row['search_stage']}"),
        }
        for dimension, group in keys.items():
            groups[(dimension, group, row['offset'])].append(row)
    result = []
    for (dimension, group, offset), values in sorted(groups.items()):
        informative = [row for row in values if row['informative_U0']]
        positive = [row for row in values if row['positive_best_U0']]
        result.append({
            'dimension': dimension, 'group': group, 'offset': offset,
            'states': len(values), 'instances': len({row['instance_id'] for row in values}),
            'informative_states': len(informative),
            'positive_best_states': len(positive),
            'informative_state_rate': len(informative) / len(values),
            'mean_stale_U0_spearman_informative': mean_or_none(
                row['stale_U0_spearman'] for row in informative),
            'mean_stale_top1_regret_positive_best': mean_or_none(
                row['stale_top1_regret'] for row in positive),
            'mean_semantic_bank_jaccard': mean_or_none(
                row['semantic_bank_jaccard'] for row in values),
            'top1_same_semantics_rate': mean_or_none(
                row['top1_same_semantics'] for row in values),
            'mean_top5_overlap_fraction': mean_or_none(
                row['top5_overlap_fraction'] for row in values),
            'mean_common_rank_spearman': mean_or_none(
                row['common_rank_spearman'] for row in values),
            'mean_common_percentile_rank_absolute_drift': mean_or_none(
                row['common_percentile_rank_mean_absolute_drift'] for row in values),
            'mean_operation_feature_relative_l2': mean_or_none(
                row['operation_feature_relative_l2'] for row in values),
            'mean_graph_edge_jaccard': mean_or_none(
                row['graph_edge_jaccard'] for row in values),
            'critical_signature_change_rate': mean_or_none(
                row['critical_signature_changed'] for row in values),
            'dominant_bottleneck_change_rate': mean_or_none(
                row['dominant_bottleneck_changed'] for row in values),
            'mean_stale_top1_U0': mean_or_none(row['stale_top1_U0'] for row in values),
            'mean_fresh_top1_U0': mean_or_none(row['fresh_top1_U0'] for row in values),
            'mean_archived_selected_U0': mean_or_none(
                row['archived_selected_U0'] for row in values),
            'mean_stale_top1_regret_union': mean_or_none(
                row['stale_top1_regret_union'] for row in values),
            'mean_fresh_top1_regret_union': mean_or_none(
                row['fresh_top1_regret_union'] for row in values),
        })
    return result


def write_report(summary: list[dict], intervals: dict) -> None:
    origin = [row for row in summary if row['dimension'] == 'dataset_origin']
    table = markdown_table(
        ['origin', 'offset', 'info', 'rho*', 'regret+', 'bank J', 'same top1',
         'top5', 'rank drift', 'feature L2', 'edge J', 'stale/fresh U0'], [[
            row['group'], row['offset'],
            f"{row['informative_states']}/{row['states']}",
            format_number(row['mean_stale_U0_spearman_informative']),
            format_number(row['mean_stale_top1_regret_positive_best']),
            format_number(row['mean_semantic_bank_jaccard']),
            format_number(row['top1_same_semantics_rate']),
            format_number(row['mean_top5_overlap_fraction']),
            format_number(row['mean_common_percentile_rank_absolute_drift']),
            format_number(row['mean_operation_feature_relative_l2']),
            format_number(row['mean_graph_edge_jaccard']),
            (f"{format_number(row['mean_stale_top1_U0'])}/"
             f"{format_number(row['mean_fresh_top1_U0'])}"),
        ] for row in origin])
    interval_table = []
    for key, metrics in intervals.items():
        origin_name, offset = key.split('|')
        interval_table.append([
            origin_name, offset,
            f"{format_number(metrics['informative_state_rate']['mean'])} "
            f"[{format_number(metrics['informative_state_rate']['lower_95'])}, "
            f"{format_number(metrics['informative_state_rate']['upper_95'])}]",
            f"{format_number(metrics['semantic_bank_jaccard']['mean'])} "
            f"[{format_number(metrics['semantic_bank_jaccard']['lower_95'])}, "
            f"{format_number(metrics['semantic_bank_jaccard']['upper_95'])}]",
            f"{format_number(metrics['common_rank_drift']['mean'])} "
            f"[{format_number(metrics['common_rank_drift']['lower_95'])}, "
            f"{format_number(metrics['common_rank_drift']['upper_95'])}]",
        ])
    REPORT.write_text(f'''# NGAS A1.7A-R prior-staleness audit

The frozen audit completed 36 states at offsets 0, 5, 10, 15, and 19 before the
production refresh at iteration 20. It evaluated all persistent base-bank actions
with eight matched repair/decode trials: 180 state-offsets and 63,875 action-offset
evaluations. Clean non-R12 and R12 development-exposed rows remain separate.

{table}

`rho*` includes only nonconstant-U0 states. `regret+` includes only states with a
positive best U0 action. All-zero states remain in denominators and are not credited
as ranking success. Top-5 overlap is normalized by five.

## Instance-cluster uncertainty

{markdown_table(['origin', 'offset', 'informative rate (95% CI)',
                 'bank Jaccard (95% CI)', 'rank drift (95% CI)'], interval_table)}

Intervals use 10,000 deterministic percentile resamples of unweighted instance
means. These are exploratory development diagnostics; no formal hypothesis test is
declared and no solver-level generalization inference is made.

## Interpretation boundary

The fresh candidate bank's semantic overlap falls immediately after offset zero,
including states whose compact operation features and typed graph are unchanged.
Candidate construction incorporates the refreshed state identifier, so this bank
churn is not evidence that the compact relational representation or RT-HGT encoder
itself became stale. Among semantic actions common to both banks, percentile-rank
drift stays small. U0 is sparse and neither origin shows monotonic degradation of
the stale critic's informative-state rank correlation through offset 19.

Fresh neural Top-1 does not provide a consistent U0 advantage over stale Top-1.
The result does not support changing the frozen refresh interval inside A1.7A-R;
any adaptive-refresh design requires a separately frozen development protocol.
Scale- and search-stage rows are retained in
`{relative(DERIVED / 'prior_staleness_summary.csv')}`.

Machine-readable sources: `{relative(DERIVED / 'prior_staleness_state_metrics.csv')}`,
`{relative(DERIVED / 'prior_staleness_summary.csv')}`, and
`{relative(DERIVED / 'prior_staleness_analysis.json')}`.
''')


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    progress = json.loads(PROGRESS.read_text())
    manifest = json.loads(RAW_MANIFEST.read_text())
    states, raw = validate_raw(protocol, manifest)
    expected_keys = {row['state_key'] for row in protocol['states']}
    session, session_paths = session_audit(manifest, expected_keys)
    execution_commit = session['execution_commit']

    ledger = verify_exposure_ledger(LEDGER)
    formal_ledger = [row for row in ledger if row.get('phase') == FORMAL_PHASE
                     and row.get('command') == FORMAL_COMMAND]
    expected_instances = Counter(row['instance_id'] for row in protocol['states'])
    expected_hashes = {row['instance_id']: row['instance_sha256']
                       for row in protocol['states']}
    ledger_checks = {
        'ledger_hash_chain_valid': True,
        'exact_formal_exposure_multiset': (
            len(formal_ledger) == 36
            and Counter(row.get('instance_id') for row in formal_ledger)
                == expected_instances),
        'formal_exposures_match_frozen_identity_and_execution': all(
            row.get('permitted') is True
            and row.get('requested_purpose') == 'diagnostic'
            and row.get('checkpoint_identifier') == protocol['checkpoint_sha256']
            and row.get('git_sha') == execution_commit
            and row.get('instance_content_sha256')
                == expected_hashes.get(row.get('instance_id'))
            for row in formal_ledger),
    }
    source_checks = {
        'protocol_frozen_before_results': (
            protocol.get('status') == 'FROZEN_BEFORE_STALENESS_RESULTS'
            and protocol.get('revision') == 2),
        'frozen_sources_match': all(
            sha256_file(ROOT / path) == expected
            for path, expected in protocol['source_hashes'].items()),
        'checkpoint_and_upstream_artifacts_match': (
            sha256_file(ROOT / protocol['checkpoint_path'])
                == protocol['checkpoint_sha256']
            and sha256_file(ROOT / protocol['full_bank_protocol_path'])
                == protocol['full_bank_protocol_sha256']
            and sha256_file(ROOT / protocol['full_bank_completion_audit_path'])
                == protocol['full_bank_completion_audit_sha256']
            and sha256_file(ROOT / protocol['full_bank_raw_manifest_path'])
                == protocol['full_bank_raw_manifest_sha256']),
        'progress_exact_complete': (
            progress.get('status') == 'COMPLETE'
            and progress.get('completed_states') == progress.get('expected_states') == 36
            and progress.get('completed_offsets') == progress.get('expected_offsets') == 180
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

    rows = metric_rows(states, raw['stages'])
    summary = summarize(rows)
    intervals = {}
    for origin in ORIGINS:
        for offset in protocol['offsets']:
            subset = [row for row in rows if row['dataset_origin'] == origin
                      and row['offset'] == offset]
            intervals[f'{origin}|{offset}'] = {
                'informative_state_rate': instance_interval(
                    [{**row, 'value': float(row['informative_U0'])} for row in subset],
                    'value', seed_key=f'{origin}:{offset}:informative'),
                'semantic_bank_jaccard': instance_interval(
                    subset, 'semantic_bank_jaccard',
                    seed_key=f'{origin}:{offset}:bank'),
                'common_rank_drift': instance_interval(
                    subset, 'common_percentile_rank_mean_absolute_drift',
                    seed_key=f'{origin}:{offset}:rank'),
            }
    atomic_csv(DERIVED / 'prior_staleness_state_metrics.csv', rows)
    atomic_csv(DERIVED / 'prior_staleness_summary.csv', summary)
    analysis = {
        'schema': 'ngas-a17ar-prior-staleness-analysis-v1',
        'statistical_scope': {
            'state_level': 'exploratory diagnostic',
            'instance_level': 'unweighted instance-cluster bootstrap interval',
            'solver_level': 'no generalization inference; R12 is development-exposed',
            'formal_hypothesis_tests': 0,
        },
        'instance_cluster_intervals': intervals,
        'origin_summary': [row for row in summary
                           if row['dimension'] == 'dataset_origin'],
        'interpretation': {
            'candidate_bank_churn': (
                'state-identifier-conditioned bank construction; distinct from '
                'compact-representation or RT-HGT staleness'),
            'adaptive_refresh_supported': False,
            'production_refresh_changed': False,
        },
        'boundaries': protocol['boundaries'],
    }
    atomic_json(DERIVED / 'prior_staleness_analysis.json', analysis)
    write_report(summary, intervals)

    status = 'PASS' if all(checks.values()) else 'FAIL'
    audit = {
        'schema': 'ngas-a17ar-prior-staleness-completion-audit-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'checks': checks,
        'protocol_sha256': sha256_file(PROTOCOL),
        'execution_commit': execution_commit,
        'checkpoint_sha256': protocol['checkpoint_sha256'],
        'candidate_bank_hash': protocol['candidate_bank_hash'],
        'completed_states': len(states), 'completed_offsets': len(rows),
        'stale_action_offset_evaluations': raw['total_evaluations'],
        'raw_bytes': sum(path.stat().st_size for path in RAW.glob('*.json')),
        'raw_manifest_sha256': sha256_file(RAW_MANIFEST),
        'formal_log_sha256': sha256_file(OUT / 'logs/prior_staleness_formal.log'),
        'session_integrity': session,
        'session_log_sha256': {relative(path): sha256_file(path)
                               for path in session_paths},
        'exposure_ledger_sha256': sha256_file(LEDGER),
        'raw_validation_errors': raw['errors'],
        'dataset_origin_counts': dict(Counter(row['dataset_origin'] for row in states)),
        'boundaries': protocol['boundaries'],
        'next_stage': ('A1_7AR_FINALIZATION' if status == 'PASS'
                       else 'STOP_STALENESS_INTEGRITY_FAILURE'),
    }
    atomic_json(AUDIT, audit)
    print(json.dumps({
        'status': status, 'states': len(states), 'offsets': len(rows),
        'action_offset_evaluations': raw['total_evaluations'],
        'checks_passed': sum(checks.values()), 'checks_total': len(checks),
        'audit': relative(AUDIT), 'report': relative(REPORT),
    }, indent=2))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
