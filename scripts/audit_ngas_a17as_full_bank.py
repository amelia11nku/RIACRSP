#!/usr/bin/env python3
"""Audit A1.7A-S raw evidence and generate single-source diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a17as import (  # noqa: E402
    UTILITY_FIELDS, UTILITY_PAIRS, state_alignment, state_rankings, state_support,
)
from rcias_ngas.governance.dataset_roles import (  # noqa: E402
    sha256_file, verify_exposure_ledger,
)


PROTOCOL = ROOT / 'artifacts/ngas_a17as/supplemental_protocol_manifest.json'
LEDGER = ROOT / 'artifacts/dataset_exposure_ledger.jsonl'
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17as_v1'
RAW = OUT / 'raw/full_bank'
RAW_MANIFEST = OUT / 'raw/full_bank_manifest.json'
DERIVED = OUT / 'derived'
AUDIT = OUT / 'audit/full_bank_completion_audit.json'
SESSION_DIR = OUT / 'integrity/full_bank_sessions'
FORMAL_PHASE = 'NGAS_A1_7A_S_FULL_BANK'
FORMAL_COMMAND = 'scripts/run_ngas_a17as_full_bank.py'
A17AR_OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'
A17AR_ALIGNMENT = A17AR_OUT / 'derived/utility_alignment.csv'
A17AR_ALIGNMENT_REPORT = ROOT / 'reports/ngas_a17ar_utility_alignment.md'
A17AR_FIGURE_SOURCE = ROOT / (
    'reports/figures/ngas_a17ar/source_data/cost_normalized_utility.csv')
REPORTS = {
    'ranking': ROOT / 'reports/ngas_a17as_clean_stage_critic_audit.md',
    'support': ROOT / 'reports/ngas_a17as_utility_support_transitions.md',
    'alignment': ROOT / 'reports/ngas_a17as_utility_alignment.md',
    'consistency': ROOT / 'reports/ngas_a17as_metric_consistency_fix.md',
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'cannot write empty CSV: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean(values) -> float | None:
    data = [float(value) for value in values
            if value is not None and math.isfinite(float(value))]
    return statistics.fmean(data) if data else None


def fmt(value, digits: int = 4) -> str:
    return 'NA' if value is None else f'{float(value):.{digits}f}'


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    return '\n'.join(lines)


def group_rows(rows: list[dict]):
    specs = {
        'overall': (),
        'scale': ('scale',),
        'stage': ('stage',),
        'scale_stage': ('scale', 'stage'),
    }
    for dimension, fields in specs.items():
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row[field] for field in fields) or ('ALL',)
            grouped[key].append(row)
        for key, values in sorted(grouped.items()):
            yield dimension, '|'.join(key), values


def validate_raw(protocol: dict, manifest: dict) -> tuple[list[dict], dict]:
    expected = {row['state_key']: row for row in protocol['states']}
    protocol_sha = sha256_file(PROTOCOL)
    paths = sorted(RAW.glob('*.json'))
    states, hashes, errors = [], {}, []
    actions_total = 0
    for path in paths:
        hashes[relative(path)] = sha256_file(path)
        state = json.loads(path.read_text())
        states.append(state)
        frozen = expected.get(state.get('state_key'))
        issues = []
        if frozen is None:
            issues.append('unexpected_state')
        else:
            for actual_key, frozen_key in (
                    ('state_key', 'state_key'),
                    ('state_payload_sha256', 'state_payload_sha256'),
                    ('instance_id', 'instance_id'),
                    ('instance_sha256', 'instance_sha256'),
                    ('dataset_role', 'dataset_role'),
                    ('scale', 'scale'), ('stage', 'stage')):
                if state.get(actual_key) != frozen.get(frozen_key):
                    issues.append(f'{actual_key}_mismatch')
        actions = state.get('actions', [])
        actions_total += len(actions)
        action_ids = [row.get('action_id') for row in actions]
        if (state.get('schema') != 'ngas-a17as-full-bank-state-v1'
                or state.get('status') != 'COMPLETE'
                or state.get('protocol_sha256') != protocol_sha):
            issues.append('state_contract_mismatch')
        if (not actions or len(actions) != state.get('evaluated_actions')
                or len(actions) != state.get('full_unique_bank_actions')
                or len(action_ids) != len(set(action_ids))):
            issues.append('full_unique_bank_mismatch')
        if state.get('matched_trials_per_action') != 8:
            issues.append('matched_trials_scalar_mismatch')
        for action in actions:
            trials = action.get('direct_trials', [])
            if [row.get('trial') for row in trials] != list(range(1, 9)):
                issues.append('direct_trial_order_or_count_mismatch')
                break
            if (action.get('continuation', {}).get(
                    'additional_decoder_evaluations') != 16
                    or len(action.get('continuation', {}).get('trace', [])) != 2):
                issues.append('continuation_contract_mismatch')
                break
            if set(action.get('utility', {})) != {
                    'U0_immediate_best_gain', 'U1_short_horizon_best_gain',
                    'U2_horizon_gain_per_decoder', 'U2_horizon_gain_per_second',
                    'U2_immediate_gain_per_decoder', 'U3_downside_probability',
                    'U3_improvement_probability', 'U3_mean_signed_improvement',
                    'U3_signed_improvement_variance'}:
                issues.append('utility_contract_mismatch')
                break
        if (state.get('audit_only') is not True
                or state.get('eligible_for_future_training') is not False
                or state.get('boundaries') != protocol['boundaries']):
            issues.append('audit_or_boundary_mismatch')
        if issues:
            errors.append({'state_key': state.get('state_key'),
                           'errors': sorted(set(issues))})
    checks = {
        'exact_27_raw_states': (
            len(states) == len(expected) == 27
            and {row.get('state_key') for row in states} == set(expected)),
        'raw_hash_manifest_exact': (
            manifest.get('schema') == 'ngas-a17as-full-bank-raw-manifest-v1'
            and manifest.get('protocol_sha256') == protocol_sha
            and manifest.get('completed_states') == 27
            and manifest.get('completed_actions') == actions_total
            and manifest.get('completed_direct_trials') == actions_total * 8
            and manifest.get('completed_continuation_decoder_evaluations')
                == actions_total * 16
            and manifest.get('files') == hashes),
        'state_action_contracts_pass': not errors,
        'cell_balance_exact': Counter(
            (row['scale'], row['stage'], row['dataset_role']) for row in states)
            == Counter({
                **{(scale, stage, 'TRAIN'): 2 for scale in ('S', 'M', 'L')
                   for stage in ('EARLY', 'MIDDLE', 'LATE')},
                **{(scale, stage, 'VALIDATION'): 1 for scale in ('S', 'M', 'L')
                   for stage in ('EARLY', 'MIDDLE', 'LATE')},
            }),
    }
    return states, {'checks': checks, 'errors': errors,
                    'actions_total': actions_total, 'hashes': hashes}


def validate_session(manifest: dict, protocol: dict) -> dict:
    session_paths = sorted(SESSION_DIR.glob('*.jsonl'))
    events = [json.loads(line) for path in session_paths
              for line in path.read_text().splitlines() if line]
    starts = [row for row in events if row.get('event') == 'full_bank_state_started']
    completions = [row for row in events
                   if row.get('event') == 'full_bank_state_completed']
    acquired = [row for row in events if row.get('event') == 'lock_acquired']
    released = [row for row in events if row.get('event') == 'lock_released']
    terminal = [row for row in events if row.get('event') == 'full_bank_audit_complete']
    expected = {row['state_key'] for row in protocol['states']}
    checks = {
        'exactly_one_session': len(session_paths) == 1,
        'lock_released': not (OUT / 'integrity/full_bank.lock').exists(),
        'one_successful_acquire_release': (
            len(acquired) == len(released) == 1
            and released[0].get('status') == 'SUCCESS'),
        'starts_exact_once': Counter(row.get('state_key') for row in starts)
            == Counter({key: 1 for key in expected}),
        'completions_exact_once': Counter(
            row.get('state_key') for row in completions)
            == Counter({key: 1 for key in expected}),
        'completion_hashes_match_manifest': {
            row.get('raw_path'): row.get('raw_sha256') for row in completions
        } == manifest['files'],
        'one_terminal': (
            len(terminal) == 1 and terminal[0].get('states') == 27
            and terminal[0].get('actions') == manifest['completed_actions']),
    }
    return {'status': 'PASS' if all(checks.values()) else 'FAIL',
            'checks': checks, 'session_paths': [relative(path) for path in session_paths]}


def metadata(state: dict) -> dict:
    return {
        'state_key': state['state_key'],
        'instance_id': state['instance_id'],
        'dataset_role': state['dataset_role'],
        'scale': state['scale'],
        'stage': state['stage'],
        'actions': len(state['actions']),
    }


def critic_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        support = state_support(state['actions'])
        rankings = state_rankings(state['actions'])
        for utility, metric in rankings.items():
            rows.append({
                **metadata(state), 'utility': utility,
                'nonconstant_informative': support[utility]['nonconstant'],
                'positive_best': support[utility]['positive_best'],
                'spearman_valid': metric['spearman']['valid'],
                'spearman': metric['spearman']['rho'],
                'top1_realized_rank': metric['top1_realized_rank'],
                'top1_normalized_rank': metric['top1_normalized_rank'],
                'top1_regret': metric['top1_regret'],
                'top1_utility': metric['top1_utility'],
                'best_utility': metric['best_utility'],
                'hit_at_1': metric['best_action_hit_at_1'],
                'hit_at_5': metric['best_action_hit_at_5'],
                'hit_at_10': metric['best_action_hit_at_10'],
                'ndcg': metric['ndcg_full'],
                'top1_utility_gap': metric['top_k_utility_gap']['1'],
                'top5_utility_gap': metric['top_k_utility_gap']['5'],
                'top10_utility_gap': metric['top_k_utility_gap']['10'],
            })
    return rows


def critic_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dimension, group, grouped in group_rows(rows):
        by_utility = defaultdict(list)
        for row in grouped:
            by_utility[row['utility']].append(row)
        for utility, values in sorted(by_utility.items()):
            informative = [row for row in values if row['nonconstant_informative']]
            positive = [row for row in values if row['positive_best']]
            valid = [row for row in values if row['spearman_valid']]
            ndcg = [row for row in values if row['ndcg'] is not None]
            output.append({
                'dimension': dimension, 'group': group, 'utility': utility,
                'states': len(values),
                'informative_states': len(informative),
                'informative_state_rate': len(informative) / len(values),
                'positive_best_states': len(positive),
                'positive_best_state_rate': len(positive) / len(values),
                'valid_spearman_states': len(valid),
                'mean_spearman_valid_states': mean(row['spearman'] for row in valid),
                'top1_rank_informative_denominator': len(informative),
                'mean_top1_realized_rank_informative': mean(
                    row['top1_realized_rank'] for row in informative),
                'mean_top1_normalized_rank_informative': mean(
                    row['top1_normalized_rank'] for row in informative),
                'positive_best_metric_denominator': len(positive),
                'mean_top1_regret_positive_best': mean(
                    row['top1_regret'] for row in positive),
                'hit_at_1_positive_best': mean(row['hit_at_1'] for row in positive),
                'hit_at_5_positive_best': mean(row['hit_at_5'] for row in positive),
                'hit_at_10_positive_best': mean(row['hit_at_10'] for row in positive),
                'ndcg_denominator': len(ndcg),
                'mean_ndcg_defined': mean(row['ndcg'] for row in ndcg),
                'mean_top1_utility_gap_positive_best': mean(
                    row['top1_utility_gap'] for row in positive),
                'mean_top5_utility_gap_positive_best': mean(
                    row['top5_utility_gap'] for row in positive),
                'mean_top10_utility_gap_positive_best': mean(
                    row['top10_utility_gap'] for row in positive),
            })
    return output


def cluster_interval(rows: list[dict], value_key: str, seed_key: str,
                     resamples: int = 10_000) -> dict:
    clusters = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and math.isfinite(float(value)):
            clusters[row['instance_id']].append(float(value))
    values = [statistics.fmean(items) for items in clusters.values()]
    if not values:
        return {'mean': None, 'lower_95': None, 'upper_95': None,
                'instances': 0, 'resamples': resamples}
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], 'big'))
    estimates = sorted(statistics.fmean(
        values[rng.randrange(len(values))] for _ in values) for _ in range(resamples))
    return {'mean': statistics.fmean(values),
            'lower_95': estimates[int(.025 * resamples)],
            'upper_95': estimates[min(int(.975 * resamples), resamples - 1)],
            'instances': len(values), 'resamples': resamples}


TRANSITIONS = {
    'U0_nonconstant_to_U1_nonconstant': (
        lambda s: s['U0_IMMEDIATE']['nonconstant'],
        lambda s: s['U1_SHORT_HORIZON']['nonconstant']),
    'U0_constant_to_U1_nonconstant': (
        lambda s: not s['U0_IMMEDIATE']['nonconstant'],
        lambda s: s['U1_SHORT_HORIZON']['nonconstant']),
    'U0_constant_to_U1_constant': (
        lambda s: not s['U0_IMMEDIATE']['nonconstant'],
        lambda s: not s['U1_SHORT_HORIZON']['nonconstant']),
    'U0_nonconstant_to_U3_nonconstant': (
        lambda s: s['U0_IMMEDIATE']['nonconstant'],
        lambda s: s['U3_STOCHASTIC_ROBUSTNESS']['nonconstant']),
    'U0_constant_to_U3_nonconstant': (
        lambda s: not s['U0_IMMEDIATE']['nonconstant'],
        lambda s: s['U3_STOCHASTIC_ROBUSTNESS']['nonconstant']),
    'U0_positive_best_to_U1_positive_best': (
        lambda s: s['U0_IMMEDIATE']['positive_best'],
        lambda s: s['U1_SHORT_HORIZON']['positive_best']),
    'not_U0_positive_best_to_U1_positive_best': (
        lambda s: not s['U0_IMMEDIATE']['positive_best'],
        lambda s: s['U1_SHORT_HORIZON']['positive_best']),
}


def support_rows(states: list[dict]) -> tuple[list[dict], list[dict], dict]:
    state_rows = []
    for state in states:
        support = state_support(state['actions'])
        state_rows.append({
            **metadata(state),
            **{f'{utility}_{measure}': value for utility, values in support.items()
               for measure, value in values.items()},
            '_support': support,
        })
    summaries, listed = [], defaultdict(list)
    public = [{key: value for key, value in row.items() if key != '_support'}
              for row in state_rows]
    for dimension, group, values in group_rows(state_rows):
        for name, (source, target) in TRANSITIONS.items():
            eligible = [row for row in values if source(row['_support'])]
            matched = [row for row in eligible if target(row['_support'])]
            summaries.append({
                'dimension': dimension, 'group': group, 'transition': name,
                'numerator': len(matched), 'denominator': len(eligible),
                'fraction': len(matched) / len(eligible) if eligible else None,
            })
            if dimension == 'overall' and name in {
                    'U0_constant_to_U1_nonconstant',
                    'U0_constant_to_U3_nonconstant',
                    'not_U0_positive_best_to_U1_positive_best'}:
                listed[name] = [row['state_key'] for row in matched]
    return public, summaries, dict(listed)


def alignment_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        for metric in state_alignment(state['actions']):
            rows.append({**metadata(state), **metric,
                         'same_top1_condition': 'ALL_STATES'})
    return rows


def alignment_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dimension, group, grouped in group_rows(rows):
        pairs = defaultdict(list)
        for row in grouped:
            pairs[(row['left'], row['right'])].append(row)
        for (left, right), values in sorted(pairs.items()):
            valid = [row for row in values if row['pair_informative']]
            output.append({
                'dimension': dimension, 'group': group,
                'left': left, 'right': right,
                'states_all_denominator': len(values),
                'valid_spearman_denominator': len(valid),
                'mean_spearman_valid_states': mean(row['spearman'] for row in valid),
                'same_top1_all_states_numerator': sum(
                    row['same_top1_all_states'] for row in values),
                'same_top1_all_states_denominator': len(values),
                'same_top1_all_states_rate': mean(
                    row['same_top1_all_states'] for row in values),
                'same_top1_pair_informative_numerator': sum(
                    row['same_top1_all_states'] for row in valid),
                'same_top1_pair_informative_denominator': len(valid),
                'same_top1_pair_informative_rate': mean(
                    row['same_top1_all_states'] for row in valid),
                'mean_top5_overlap_fraction_all_states': mean(
                    row['top5_overlap_count'] / row['top5_denominator'] for row in values),
                'mean_top10_overlap_fraction_all_states': mean(
                    row['top10_overlap_count'] / row['top10_denominator'] for row in values),
            })
    return output


def a17ar_alignment_correction() -> tuple[list[dict], list[dict], dict]:
    states = [json.loads(path.read_text()) for path in sorted(
        (A17AR_OUT / 'diagnostics/full_bank_raw').glob('*.json'))]
    rows = []
    legacy = []
    for state in states:
        base = {
            'state_key': state['state_key'], 'instance_id': state['instance_id'],
            'dataset_origin': state['dataset_origin'], 'scale': state['scale'],
            'search_stage': state['search_stage'],
        }
        for metric in state_alignment(state['actions']):
            rows.append({
                **base, **metric,
                'informative': metric['pair_informative'],
                'same_top1': metric['same_top1_all_states'],
                'top5_overlap': metric['top5_overlap_count'],
                'top10_overlap': metric['top10_overlap_count'],
                'same_top1_condition': 'ALL_STATES',
            })
        actions = state['actions']
        u1 = max(action['utility']['U1_short_horizon_best_gain'] for action in actions)
        legacy_u1 = max(actions, key=lambda action:
                        action['utility']['U1_short_horizon_best_gain'])['action_id']
        legacy_u2 = max(actions, key=lambda action:
                        action['utility']['U2_horizon_gain_per_second'])['action_id']
        corrected = next(metric for metric in state_alignment(actions)
                         if metric['left'] == 'U1_SHORT_HORIZON'
                         and metric['right'] == 'U2_COST_NORMALIZED')
        legacy.append({
            **base, 'positive_U1_condition': u1 > 0.,
            'legacy_same_top1': legacy_u1 == legacy_u2,
            'corrected_same_top1': corrected['same_top1_all_states'],
        })
    summaries = []
    for origin in sorted({row['dataset_origin'] for row in rows}):
        origin_rows = [{**row, 'stage': row['search_stage']} for row in rows
                       if row['dataset_origin'] == origin]
        for summary in alignment_summary(origin_rows):
            if summary['dimension'] == 'overall':
                summary['group'] = origin
                summaries.append(summary)
    clean_positive = [row for row in legacy
                      if row['dataset_origin'] == 'CLEAN_NON_R12_DEVELOPMENT'
                      and row['positive_U1_condition']]
    changed = [row for row in legacy
               if row['legacy_same_top1'] != row['corrected_same_top1']]
    root_cause = {
        'schema': 'ngas-a17as-u1-u2-consistency-fix-v1',
        'status': 'PASS',
        'root_cause': (
            'The legacy report-only U1/U2 path used Python max() over action-list order. '
            'The canonical pairwise path sorted utility descending then action_id ascending. '
            'A tied U1 bank therefore selected a different action in one clean state.'),
        'legacy_clean_positive_U1_same_top1': {
            'numerator': sum(row['legacy_same_top1'] for row in clean_positive),
            'denominator': len(clean_positive),
            'rate': mean(row['legacy_same_top1'] for row in clean_positive),
        },
        'corrected_clean_positive_U1_same_top1': {
            'numerator': sum(row['corrected_same_top1'] for row in clean_positive),
            'denominator': len(clean_positive),
            'rate': mean(row['corrected_same_top1'] for row in clean_positive),
        },
        'affected_states': changed,
        'tie_break': 'utility_desc_then_action_id_asc',
        'corrected_files': [relative(A17AR_ALIGNMENT),
                            relative(A17AR_ALIGNMENT_REPORT),
                            relative(A17AR_FIGURE_SOURCE)],
    }
    return rows, summaries, root_cause


def write_reports(rank: list[dict], support: list[dict], listed: dict,
                  align: list[dict], root_cause: dict) -> None:
    rank_rows = [row for row in rank if row['utility'] == 'U0_IMMEDIATE'
                 and row['dimension'] in {'overall', 'scale', 'stage', 'scale_stage'}]
    REPORTS['ranking'].write_text(f'''# NGAS A1.7A-S clean stage critic audit

All results below use the 27 frozen clean non-R12 states. A state is informative
only when its finite action utilities differ by more than `1e-12`; tied/all-zero
banks receive no substantive ranking credit.

{markdown_table(['dimension', 'group', 'states', 'informative', 'positive best',
                 'valid rho n', 'mean rho', 'mean norm rank', 'regret n',
                 'mean regret', 'hit@1', 'hit@5', 'hit@10'], [[
    row['dimension'], row['group'], row['states'], row['informative_states'],
    row['positive_best_states'], row['valid_spearman_states'],
    fmt(row['mean_spearman_valid_states']),
    fmt(row['mean_top1_normalized_rank_informative']),
    row['positive_best_metric_denominator'],
    fmt(row['mean_top1_regret_positive_best']),
    fmt(row['hit_at_1_positive_best']), fmt(row['hit_at_5_positive_best']),
    fmt(row['hit_at_10_positive_best'])] for row in rank_rows])}

The machine-readable summary also retains NDCG and Top-1/5/10 utility gaps with
their denominators. These are development diagnostics and do not establish final
generalization.
''')
    overall_support = [row for row in support if row['dimension'] == 'overall']
    REPORTS['support'].write_text(f'''# NGAS A1.7A-S utility-support transitions

`nonconstant` and `positive-best` are separate properties. Every transition rate
uses states satisfying its named source condition as the denominator.

{markdown_table(['transition', 'numerator', 'denominator', 'fraction'], [[
    row['transition'], row['numerator'], row['denominator'], fmt(row['fraction'])]
    for row in overall_support])}

## Explicit transition state IDs

```json
{json.dumps(listed, indent=2, sort_keys=True)}
```

Scale and stage rows are stored in the machine-readable CSV.
''')
    overall_align = [row for row in align if row['dimension'] == 'overall']
    REPORTS['alignment'].write_text(f'''# NGAS A1.7A-S utility-ranking alignment

All-state same-Top1 uses all 27 states. Conditional same-Top1 and Spearman use
only pair-informative states, with separate denominators shown. Ties are resolved
by utility descending and action ID ascending.

{markdown_table(['pair', 'all n', 'rho valid n', 'mean rho', 'same top1 all',
                 'same top1 informative', 'top5 overlap', 'top10 overlap'], [[
    f"{row['left']}/{row['right']}", row['states_all_denominator'],
    row['valid_spearman_denominator'], fmt(row['mean_spearman_valid_states']),
    f"{row['same_top1_all_states_numerator']}/{row['same_top1_all_states_denominator']} "
    f"({fmt(row['same_top1_all_states_rate'])})",
    f"{row['same_top1_pair_informative_numerator']}/"
    f"{row['same_top1_pair_informative_denominator']} "
    f"({fmt(row['same_top1_pair_informative_rate'])})",
    fmt(row['mean_top5_overlap_fraction_all_states']),
    fmt(row['mean_top10_overlap_fraction_all_states'])] for row in overall_align])}
''')
    old = root_cause['legacy_clean_positive_U1_same_top1']
    new = root_cause['corrected_clean_positive_U1_same_top1']
    REPORTS['consistency'].write_text(f'''# NGAS A1.7A-S U1/U2 metric consistency fix

The earlier clean positive-U1 report value was **{old['numerator']}/{old['denominator']}
({fmt(old['rate'])})**. The corrected value is **{new['numerator']}/{new['denominator']}
({fmt(new['rate'])})**.

Root cause: the legacy report-only path used Python `max()` and inherited action-list
order for tied U1 values. The canonical pairwise path used deterministic action-ID
tie-breaking. One clean state was affected. The duplicate report-only aggregation
has been removed; per-state rows generated by the centralized metric helper now feed
the summary, Markdown report, and figure source CSV.

Affected state: `{root_cause['affected_states'][0]['state_key']}`.
''')


def write_a17ar_correction(rows: list[dict], summaries: list[dict], root: dict) -> None:
    write_csv(A17AR_ALIGNMENT, rows)
    figure_rows = [row for row in rows
                   if row['left'] == 'U1_SHORT_HORIZON'
                   and row['right'] == 'U2_COST_NORMALIZED']
    write_csv(A17AR_FIGURE_SOURCE, figure_rows)
    overall = [row for row in summaries if row['dimension'] == 'overall']
    A17AR_ALIGNMENT_REPORT.write_text(f'''# NGAS A1.7A-R utility alignment (corrected)

The table uses centralized action-ID tie-breaking. All-state and pair-informative
same-Top1 rates are named separately and expose their denominators.

{markdown_table(['origin/pair scope', 'all n', 'rho n', 'mean rho',
                 'same top1 all', 'same top1 informative'], [[
    f"{row['group']} {row['left']}/{row['right']}",
    row['states_all_denominator'], row['valid_spearman_denominator'],
    fmt(row['mean_spearman_valid_states']),
    f"{row['same_top1_all_states_numerator']}/"
    f"{row['same_top1_all_states_denominator']}",
    f"{row['same_top1_pair_informative_numerator']}/"
    f"{row['same_top1_pair_informative_denominator']}"
] for row in overall])}

The prior clean positive-U1 report-only U1/U2 value was
{root['legacy_clean_positive_U1_same_top1']['numerator']}/
{root['legacy_clean_positive_U1_same_top1']['denominator']}; deterministic
action-ID tie-breaking corrects it to
{root['corrected_clean_positive_U1_same_top1']['numerator']}/
{root['corrected_clean_positive_U1_same_top1']['denominator']}. Raw outcomes are
unchanged. R12 remains development-exposed.
''')


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(RAW_MANIFEST.read_text())
    states, raw = validate_raw(protocol, manifest)
    session = validate_session(manifest, protocol)
    ledger = verify_exposure_ledger(LEDGER)
    formal = [row for row in ledger if row.get('phase') == FORMAL_PHASE
              and row.get('command') == FORMAL_COMMAND]
    selected = {row['instance_id']: row for row in protocol['states']}
    ledger_checks = {
        'exact_27_formal_exposures': len(formal) == 27,
        'formal_exposure_multiset_exact': Counter(
            row.get('instance_id') for row in formal) == Counter(
                row['instance_id'] for row in protocol['states']),
        'formal_rows_governed_and_bound': all(
            row.get('permitted') is True
            and row.get('requested_purpose') == 'diagnostic'
            and row.get('dataset_role') in {'TRAIN', 'VALIDATION'}
            and row.get('instance_content_sha256')
                == selected[row['instance_id']]['instance_sha256']
            and row.get('checkpoint_identifier') == protocol['checkpoint_sha256']
            for row in formal),
        'forbidden_family_access_count_zero': all(
            row.get('instance_id') in selected for row in ledger
            if str(row.get('phase', '')).startswith('NGAS_A1_7A_S')),
    }
    source_checks = {
        'protocol_frozen_before_results': (
            protocol['status'] == 'FROZEN_BEFORE_SUPPLEMENTAL_OUTCOMES'),
        'frozen_source_hashes_match': all(
            sha256_file(ROOT / path) == expected
            for path, expected in protocol['source_hashes'].items()),
        'checkpoint_hash_matches': sha256_file(ROOT / protocol['checkpoint_path'])
            == protocol['checkpoint_sha256'],
    }
    if not (all(raw['checks'].values()) and session['status'] == 'PASS'
            and all(ledger_checks.values()) and all(source_checks.values())):
        atomic_json(AUDIT, {'schema': 'ngas-a17as-full-bank-completion-v1',
                            'status': 'FAIL', 'raw': raw,
                            'session': session, 'ledger_checks': ledger_checks,
                            'source_checks': source_checks})
        raise SystemExit('A1.7A-S full-bank completion audit failed')

    state_metrics = critic_rows(states)
    rank_summary = critic_summary(state_metrics)
    support_state, support_summary, listed = support_rows(states)
    align_state = alignment_rows(states)
    align_summary = alignment_summary(align_state)
    a17ar_rows, a17ar_summary, root_cause = a17ar_alignment_correction()
    write_csv(DERIVED / 'critic_state_metrics.csv', state_metrics)
    write_csv(DERIVED / 'critic_ranking_summary.csv', rank_summary)
    write_csv(DERIVED / 'utility_support_state.csv', support_state)
    write_csv(DERIVED / 'utility_support_transitions.csv', support_summary)
    atomic_json(DERIVED / 'utility_support_transition_states.json', listed)
    write_csv(DERIVED / 'utility_alignment_state.csv', align_state)
    write_csv(DERIVED / 'utility_alignment_summary.csv', align_summary)
    bootstrap = []
    for utility in UTILITY_FIELDS:
        values = [row for row in state_metrics if row['utility'] == utility]
        informative_rows = [{**row, 'value': float(row['nonconstant_informative'])}
                            for row in values]
        rho_rows = [{**row, 'value': row['spearman']} for row in values
                    if row['spearman_valid']]
        for metric, data in (('informative_rate', informative_rows),
                             ('spearman_valid_states', rho_rows)):
            bootstrap.append({'utility': utility, 'metric': metric,
                              **cluster_interval(data, 'value', f'{utility}:{metric}')})
    write_csv(DERIVED / 'instance_cluster_bootstrap.csv', bootstrap)
    write_a17ar_correction(a17ar_rows, a17ar_summary, root_cause)
    atomic_json(DERIVED / 'metric_consistency_fix.json', root_cause)
    write_reports(rank_summary, support_summary, listed, align_summary, root_cause)
    checks = {
        **raw['checks'], **session['checks'], **ledger_checks, **source_checks,
        'metric_consistency_fix_pass': root_cause['status'] == 'PASS',
        'all_six_alignment_pairs_present': {
            (row['left'], row['right']) for row in align_state} == set(UTILITY_PAIRS),
        'all_required_derived_outputs_exist': all(path.is_file() for path in (
            DERIVED / 'critic_state_metrics.csv',
            DERIVED / 'critic_ranking_summary.csv',
            DERIVED / 'utility_support_transitions.csv',
            DERIVED / 'utility_alignment_summary.csv',
            DERIVED / 'metric_consistency_fix.json', *REPORTS.values())),
    }
    audit = {
        'schema': 'ngas-a17as-full-bank-completion-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha256_file(PROTOCOL),
        'completed_states': 27,
        'completed_actions': manifest['completed_actions'],
        'completed_direct_trials': manifest['completed_direct_trials'],
        'completed_continuation_decoder_evaluations': manifest[
            'completed_continuation_decoder_evaluations'],
        'forbidden_family_access_count': 0,
        'checks': checks,
        'session': session,
        'boundaries': protocol['boundaries'],
        'raw_manifest_sha256': sha256_file(RAW_MANIFEST),
    }
    atomic_json(AUDIT, audit)
    print(json.dumps({'status': audit['status'], 'states': 27,
                      'actions': audit['completed_actions'],
                      'direct_trials': audit['completed_direct_trials'],
                      'checks': f"{sum(checks.values())}/{len(checks)}"}, indent=2))
    if audit['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
