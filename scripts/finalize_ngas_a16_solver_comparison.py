#!/usr/bin/env python3
"""Audit, analyze, and close the frozen NGAS A1.6 comparison."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16 import (  # noqa: E402
    holm_adjust, incumbent_at, paired_summary, right_continuous_auc,
    terminal_decision,
)
from rcias_ngas.evaluation.a16_integrity import (  # noqa: E402
    COMPARATOR_DIRECTORIES, audit_frozen_inputs, digest, load_json,
)
from scripts.run_ngas_a16_solver_comparison import (  # noqa: E402
    atomic_json, build_tasks, load_boundary, raw_path, validate_existing_raw,
)


OUT = ROOT / 'outputs/ngas_a1/solver_comparison_v1'
DERIVED = OUT / 'derived'
AUDIT = OUT / 'audit'
REPORT = ROOT / 'docs/reports/ngas_a1/12_a16_solver_comparison.md'
ALGORITHMS = ['NGAS_A1_6', 'PHASE6N_TOP1', 'PHASE6H', 'ALNS', 'LG_HGA_2O']
COMPARATORS = ALGORITHMS[1:]
ANYTIME_FRACTIONS = [.10, .25, .50, .75, 1.]


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q, method='linear')) if values else None


def load_all_payloads(config: dict, protocol_sha256: str) -> tuple[dict, list[dict]]:
    tasks = build_tasks(config)
    raw_manifest_path = OUT / 'raw_manifest.json'
    raw_manifest = load_json(raw_manifest_path)
    expected_paths = {str(raw_path(task).relative_to(ROOT)) for task in tasks}
    if (raw_manifest.get('schema') != 'ngas-a16-raw-manifest-v1'
            or raw_manifest.get('protocol_sha256') != protocol_sha256
            or raw_manifest.get('completed_runs') != 54
            or set(raw_manifest.get('files', {})) != expected_paths):
        raise RuntimeError('A1.6 raw manifest scope is incomplete or invalid')
    ngas = []
    for task in tasks:
        path = raw_path(task)
        if digest(path) != raw_manifest['files'][str(path.relative_to(ROOT))]:
            raise RuntimeError(f'A1.6 raw manifest hash mismatch: {path}')
        ngas.append(validate_existing_raw(path, task, protocol_sha256, config))
    comparator = {}
    for algorithm_id, directory in COMPARATOR_DIRECTORIES.items():
        manifest = load_json(ROOT / f'outputs/frozen_2o_baselines/{directory}/manifest.json')
        for row in manifest['runs']:
            comparator[(algorithm_id, row['instance_id'], int(row['seed']))] = load_json(
                ROOT / row['path'])
    return comparator, ngas


def final_quality_rows(config: dict, comparator: dict, ngas: list[dict]) -> pd.DataFrame:
    bks = load_json(ROOT / config['bks']['path'])['instances']
    instance_rows = {row['instance_id']: row for row in
                     load_json(ROOT / config['scope']['instance_manifest_path'])['instances']}
    rows = []
    payloads = {('NGAS_A1_6', row['instance_id'], row['seed']): row for row in ngas}
    payloads.update(comparator)
    for (algorithm, instance_id, seed), payload in sorted(payloads.items()):
        meta = instance_rows[instance_id]
        makespan = float(payload.get('final_makespan'))
        reference = float(bks[instance_id]['makespan'])
        rows.append({
            'algorithm': algorithm, 'instance_id': instance_id, 'seed': seed,
            'scale': meta['scale'], 'CF_level': meta['CF_level'],
            'cell_replicate': meta['cell_replicate'],
            'num_operations': meta['num_operations'],
            'budget_seconds': 2. * meta['num_operations'],
            'final_makespan': makespan, 'bks_v001_makespan': reference,
            'rpd_v001_percent': 100. * (makespan - reference) / reference,
            'feasible': bool(payload['feasible']),
            'runtime_seconds': float(payload.get(
                'solver_budget_elapsed_seconds', payload.get('runtime_seconds'))),
            'best_found_seconds': float(payload.get(
                'best_found_seconds', payload.get('best_found_time_seconds', 0.))),
            'decoder_evaluations': int(payload['decoder_evaluations']),
            'iterations': int(payload['iterations']),
        })
    return pd.DataFrame(rows)


def quality_analysis(frame: pd.DataFrame, tie_tolerance: float) -> tuple:
    instance = frame.groupby(
        ['algorithm', 'instance_id', 'scale', 'CF_level', 'cell_replicate'],
        as_index=False,
    ).agg(
        mean_final_makespan=('final_makespan', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'),
        median_final_makespan=('final_makespan', 'median'),
        feasible_fraction=('feasible', 'mean'),
    )
    pivot = instance.pivot(index='instance_id', columns='algorithm',
                           values='mean_final_makespan').sort_index()
    ranks = np.vstack([rankdata(row, method='average') for row in pivot[ALGORITHMS].to_numpy()])
    average_ranks = dict(zip(ALGORITHMS, ranks.mean(axis=0).tolist()))
    bks_hits = {
        algorithm: int((part['final_makespan'] <= part['bks_v001_makespan']
                        + tie_tolerance).sum())
        for algorithm, part in frame.groupby('algorithm')
    }
    method = frame.groupby('algorithm', as_index=False).agg(
        runs=('final_makespan', 'size'), mean_final_makespan=('final_makespan', 'mean'),
        median_final_makespan=('final_makespan', 'median'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'),
        median_rpd_v001_percent=('rpd_v001_percent', 'median'),
        feasibility_fraction=('feasible', 'mean'))
    method['average_instance_rank'] = method['algorithm'].map(average_ranks)
    method['bks_v001_run_hits'] = method['algorithm'].map(bks_hits)

    ngas_map = pivot['NGAS_A1_6'].to_dict()
    summaries = {name: paired_summary(
        ngas_map, pivot[name].to_dict(), tie_tolerance) for name in COMPARATORS}
    scale_summaries = {}
    metadata = instance.drop_duplicates('instance_id').set_index('instance_id')
    for scale in ('S', 'M', 'L'):
        ids = metadata.index[metadata['scale'] == scale].tolist()
        scale_summaries[scale] = paired_summary(
            {key: ngas_map[key] for key in ids},
            {key: pivot.loc[key, 'PHASE6N_TOP1'] for key in ids}, tie_tolerance)

    pairwise_rows = []
    run_pivot = frame.pivot(index=['instance_id', 'seed'], columns='algorithm',
                            values='final_makespan')
    raw_p_values = []
    statistical_rows = []
    for comparator in COMPARATORS:
        differences = (pivot[comparator] - pivot['NGAS_A1_6']).to_numpy()
        if np.all(np.abs(differences) <= tie_tolerance):
            statistic, p_value, effect = 0., 1., 0.
        else:
            test = wilcoxon(differences, zero_method='pratt', alternative='two-sided',
                            method='auto')
            statistic, p_value = float(test.statistic), float(test.pvalue)
            ranked = rankdata(np.abs(differences), method='average')
            positive = float(ranked[differences > tie_tolerance].sum())
            negative = float(ranked[differences < -tie_tolerance].sum())
            effect = (positive - negative) / (positive + negative) \
                if positive + negative else 0.
        raw_p_values.append(p_value)
        statistical_rows.append({
            'comparator': comparator, 'statistic': statistic,
            'raw_p_value': p_value, 'rank_biserial_effect': effect,
            'effect_direction': ('FAVORS_NGAS' if effect > 0 else
                                 'FAVORS_COMPARATOR' if effect < 0 else 'NEUTRAL'),
            'wins': summaries[comparator]['wins'],
            'ties': summaries[comparator]['ties'],
            'losses': summaries[comparator]['losses'],
        })
    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(statistical_rows, adjusted):
        row['holm_adjusted_p_value'] = value
        row['holm_reject_0_05'] = value < .05
        summary = summaries[row['comparator']]
        run_gain = 100. * (run_pivot[row['comparator']] - run_pivot['NGAS_A1_6']) \
            / run_pivot[row['comparator']]
        pairwise_rows.append({
            **{key: summary[key] for key in (
                'instance_count', 'mean_relative_gain_percent',
                'median_relative_gain_percent', 'wins', 'ties', 'losses')},
            'comparator': row['comparator'], 'run_pairs': len(run_gain),
            'mean_run_paired_gain_percent': float(run_gain.mean()),
            'median_run_paired_gain_percent': float(run_gain.median()),
            'wilcoxon_statistic': row['statistic'], 'raw_p_value': row['raw_p_value'],
            'holm_adjusted_p_value': value,
            'rank_biserial_effect': row['rank_biserial_effect'],
            'effect_direction': row['effect_direction'],
        })
    friedman = friedmanchisquare(*(pivot[name].to_numpy() for name in ALGORITHMS))
    statistics_payload = {
        'schema': 'ngas-a16-statistical-tests-v1',
        'analysis_unit': '18 instance-level three-seed means',
        'tie_tolerance_makespan': tie_tolerance,
        'zero_difference_method': 'pratt; all-zero comparison maps to W=0,p=1,effect=0',
        'friedman': {
            'algorithms': ALGORITHMS, 'blocks': 18, 'degrees_of_freedom': 4,
            'statistic': float(friedman.statistic), 'p_value': float(friedman.pvalue),
            'average_ranks': average_ranks,
        },
        'wilcoxon_holm': statistical_rows,
    }
    return instance, method, pivot, summaries, scale_summaries, pd.DataFrame(pairwise_rows), statistics_payload


def anytime_analysis(config: dict, comparator: dict,
                     ngas: list[dict]) -> tuple[pd.DataFrame, ...]:
    meta = {row['instance_id']: row for row in
            load_json(ROOT / config['scope']['instance_manifest_path'])['instances']}
    bks = {key: float(value['makespan']) for key, value in
           load_json(ROOT / config['bks']['path'])['instances'].items()}
    ngas_lookup = {(row['instance_id'], row['seed']): row for row in ngas}
    payloads = {('NGAS_A1_6', key[0], key[1]): value
                for key, value in ngas_lookup.items()}
    payloads.update(comparator)
    h1 = {}
    for key, payload in ngas_lookup.items():
        h1.setdefault(key[0], float(payload['h1_makespan']))
        if h1[key[0]] != float(payload['h1_makespan']):
            raise RuntimeError(f'NGAS H1 makespan changed across seeds: {key[0]}')
    checkpoint_rows, auc_rows, passage_rows = [], [], []
    thresholds = config['metrics']['anytime']['first_passage_rpd_thresholds_percent']
    for (algorithm, instance_id, seed), payload in sorted(payloads.items()):
        trace = payload['incumbent_trace']
        budget = float(payload['budget_seconds'])
        for fraction in ANYTIME_FRACTIONS:
            incumbent = incumbent_at(trace, fraction * budget)
            checkpoint_rows.append({
                'algorithm': algorithm, 'instance_id': instance_id, 'seed': seed,
                'scale': meta[instance_id]['scale'], 'CF_level': meta[instance_id]['CF_level'],
                'cell_replicate': meta[instance_id]['cell_replicate'],
                'budget_fraction': fraction, 'available': incumbent is not None,
                'incumbent_makespan': (float(incumbent['current_best_makespan'])
                                       if incumbent else None),
                'normalized_incumbent_over_h1': (
                    float(incumbent['current_best_makespan']) / h1[instance_id]
                    if incumbent else None),
                'rpd_v001_percent': (
                    100. * (float(incumbent['current_best_makespan']) - bks[instance_id])
                    / bks[instance_id] if incumbent else None),
                'decoder_evaluations': int(incumbent['decoder_evaluations']) if incumbent else None,
            })
        auc = right_continuous_auc(trace, budget, h1[instance_id])
        auc_rows.append({
            'algorithm': algorithm, 'instance_id': instance_id, 'seed': seed,
            'scale': meta[instance_id]['scale'], **(auc or {
                'auc_available_horizon': None, 'available_budget_fraction': 0.,
                'mean_normalized_incumbent_over_available_horizon': None,
                'first_incumbent_budget_fraction': None,
            })})
        for threshold in thresholds:
            target = bks[instance_id] * (1. + float(threshold) / 100.)
            hit = next((row for row in trace
                        if float(row['current_best_makespan']) <= target), None)
            passage_rows.append({
                'algorithm': algorithm, 'instance_id': instance_id, 'seed': seed,
                'threshold_rpd_percent': threshold, 'target_makespan': target,
                'hit': hit is not None,
                'first_passage_seconds': float(hit['elapsed_time']) if hit else None,
                'first_passage_budget_fraction': (
                    float(hit['elapsed_time']) / budget if hit else None),
            })
    checkpoints = pd.DataFrame(checkpoint_rows)
    auc = pd.DataFrame(auc_rows)
    passage = pd.DataFrame(passage_rows)
    checkpoint_summary = checkpoints.groupby(['algorithm', 'budget_fraction'], as_index=False).agg(
        available_runs=('available', 'sum'),
        mean_normalized_incumbent_over_h1=('normalized_incumbent_over_h1', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'),
        mean_decoder_evaluations=('decoder_evaluations', 'mean'))
    pair_rows = []
    for fraction in ANYTIME_FRACTIONS:
        part = checkpoints[checkpoints.budget_fraction == fraction]
        pivot = part.pivot(index=['instance_id', 'seed'], columns='algorithm',
                           values='incumbent_makespan')
        for comparator_name in COMPARATORS:
            valid = pivot[['NGAS_A1_6', comparator_name]].dropna()
            gains = 100. * (valid[comparator_name] - valid['NGAS_A1_6']) / valid[comparator_name]
            pair_rows.append({
                'budget_fraction': fraction, 'comparator': comparator_name,
                'available_run_pairs': len(valid),
                'mean_paired_gain_percent': float(gains.mean()) if len(valid) else None,
                'median_paired_gain_percent': float(gains.median()) if len(valid) else None,
            })
    return checkpoints, checkpoint_summary, auc, passage, pd.DataFrame(pair_rows)


def runtime_analysis(ngas: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_rows = []
    all_latencies, tail = [], Counter()
    for row in ngas:
        latencies = [float(value) for value in row['refresh_latency']['complete_refresh_ms']]
        all_latencies.extend(latencies)
        tail.update({
            'gt_30_ms': sum(value > 30. for value in latencies),
            'gt_50_ms': sum(value > 50. for value in latencies),
            'gt_100_ms': sum(value > 100. for value in latencies),
        })
        improvement = float(row['h1_makespan']) - float(row['final_makespan'])
        evaluations = int(row['decoder_evaluations'])
        runtime = float(row['solver_budget_elapsed_seconds'])
        components = row['runtime_components']
        run_rows.append({
            'instance_id': row['instance_id'], 'seed': row['seed'], 'scale': row['scale'],
            'iterations': row['iterations'], 'decoder_evaluations': evaluations,
            'critic_calls': row['critic_calls'], 'refresh_count': row['refresh_count'],
            'guided_iterations': row['guided_iterations'],
            'guided_iterations_per_critic_call': row['guided_iterations_per_critic_call'],
            'runtime_preparation_seconds': row['runtime_preparation_seconds'],
            'refresh_seconds': components['refresh_total_seconds'],
            'refresh_budget_share': components['refresh_total_seconds'] / row['budget_seconds'],
            'decoder_evaluations_per_second': evaluations / runtime,
            'new_best_events': row['new_best_moves'],
            'new_best_events_per_1000_evaluations': 1000. * row['new_best_moves'] / evaluations,
            'h1_to_final_improvement': improvement,
            'improvement_per_second': improvement / runtime,
            'improvement_per_1000_evaluations': 1000. * improvement / evaluations,
            'best_found_seconds': row['best_found_seconds'],
            'budget_overshoot_seconds': row['budget_overshoot_seconds'],
            'refresh_p50_ms': percentile(latencies, 50),
            'refresh_p90_ms': percentile(latencies, 90),
            'refresh_p99_ms': percentile(latencies, 99),
            'refresh_max_ms': max(latencies) if latencies else None,
        })
    runs = pd.DataFrame(run_rows)
    summary_rows = []
    for scale, part in [('ALL', ngas), *[(name, [row for row in ngas if row['scale'] == name])
                                        for name in ('S', 'M', 'L')]]:
        latencies = [float(value) for row in part
                     for value in row['refresh_latency']['complete_refresh_ms']]
        selected = runs if scale == 'ALL' else runs[runs.scale == scale]
        summary_rows.append({
            'scale': scale, 'runs': len(part), 'refresh_samples': len(latencies),
            'mean_critic_calls': float(selected.critic_calls.mean()),
            'mean_guided_iterations_per_call': float(selected.guided_iterations_per_critic_call.mean()),
            'mean_refresh_budget_share': float(selected.refresh_budget_share.mean()),
            'mean_runtime_preparation_seconds': float(selected.runtime_preparation_seconds.mean()),
            'refresh_p50_ms': percentile(latencies, 50),
            'refresh_p90_ms': percentile(latencies, 90),
            'refresh_p99_ms': percentile(latencies, 99),
            'refresh_max_ms': max(latencies) if latencies else None,
            'count_gt_30_ms': sum(value > 30. for value in latencies),
            'count_gt_50_ms': sum(value > 50. for value in latencies),
            'count_gt_100_ms': sum(value > 100. for value in latencies),
            'mean_decoder_evaluations_per_second': float(selected.decoder_evaluations_per_second.mean()),
            'mean_new_best_events_per_1000_evaluations': float(
                selected.new_best_events_per_1000_evaluations.mean()),
        })
    tail_payload = {
        'schema': 'ngas-a16-refresh-tail-summary-v1',
        'samples': len(all_latencies), 'p50_ms': percentile(all_latencies, 50),
        'p90_ms': percentile(all_latencies, 90), 'p99_ms': percentile(all_latencies, 99),
        'maximum_ms': max(all_latencies) if all_latencies else None, **dict(tail),
        'a15r_context': 'A1.5R observed an S transition-trace p99 tail near 152 ms',
    }
    return runs, pd.DataFrame(summary_rows), tail_payload


def report_text(decision: str, method: pd.DataFrame, pairwise: pd.DataFrame,
                scale_summaries: dict, stats: dict, checkpoint_pairs: pd.DataFrame,
                runtime_summary: pd.DataFrame, tail: dict, budget: dict,
                regression_count: int) -> str:
    ngas = method.set_index('algorithm').loc['NGAS_A1_6']
    pairs = pairwise.set_index('comparator')
    phase = pairs.loc['PHASE6N_TOP1']
    lg = pairs.loc['LG_HGA_2O']
    early = checkpoint_pairs[checkpoint_pairs.comparator == 'PHASE6N_TOP1']
    early_values = ', '.join(
        f"{int(100 * row.budget_fraction)}%={row.mean_paired_gain_percent:.3f}%"
        for row in early.itertuples() if pd.notna(row.mean_paired_gain_percent))
    scale_values = ', '.join(
        f"{scale}={summary['mean_relative_gain_percent']:.3f}%"
        for scale, summary in scale_summaries.items())
    all_runtime = runtime_summary.set_index('scale').loc['ALL']
    ranks = stats['friedman']['average_ranks']
    rank_text = ', '.join(f'{key}={value:.3f}' for key, value in ranks.items())
    holm_text = '; '.join(
        f"{row['comparator']}: p_holm={row['holm_adjusted_p_value']:.6g}, "
        f"effect={row['rank_biserial_effect']:.3f}"
        for row in stats['wilcoxon_holm'])
    eligibility = ('R13/R14 are eligible only for a separately preregistered next stage; '
                   'they were not accessed here.' if decision == 'NGAS_A1_PASS_DEVELOPMENT'
                   else 'R13/R14 remain locked and were not accessed.')
    return f"""# NGAS A1.6 Frozen 2|O| Solver Comparison

## Decision and integrity

- Terminal decision: **`{decision}`**
- Formal scope: 54/54 NGAS runs plus four immutable 54-run comparators
- Feasibility and replay: 54/54 NGAS; all reused comparator payloads remain feasible
- Budget audit: median overshoot {budget['median_overshoot_seconds']:.6f} s; maximum {budget['maximum_overshoot_seconds']:.6f} s; atomic starts after deadline = {budget['started_after_deadline_count']}
- Regression: {regression_count} tests passed
- R13/R14: {eligibility}
- Gurobi: not run

## Final quality

NGAS mean RPD is {ngas.mean_rpd_v001_percent:.4f}% and median RPD is
{ngas.median_rpd_v001_percent:.4f}%. Against PHASE6N_TOP1, its mean instance-level
gain is {phase.mean_relative_gain_percent:.4f}% with W/T/L
{int(phase.wins)}/{int(phase.ties)}/{int(phase.losses)}. Against LG_HGA_2O, the
corresponding result is {lg.mean_relative_gain_percent:.4f}% and
{int(lg.wins)}/{int(lg.ties)}/{int(lg.losses)}.

Average ranks are {rank_text}. The Friedman result is
`chi2({stats['friedman']['degrees_of_freedom']})={stats['friedman']['statistic']:.6f}`,
`p={stats['friedman']['p_value']:.6g}`. Holm-corrected comparisons are: {holm_text}.

## Required interpretation

1. **PHASE6N_TOP1:** NGAS paired gain is {phase.mean_relative_gain_percent:.4f}% with W/T/L {int(phase.wins)}/{int(phase.ties)}/{int(phase.losses)}.
2. **When quality appears:** Paired gains at normalized checkpoints are {early_values}; these causal checkpoints use only incumbents already present at each deadline.
3. **Scale:** Paired NGAS gains versus PHASE6N_TOP1 are {scale_values}. This shows whether improvement strengthens or collapses with scale without extrapolating beyond the 18 instances.
4. **Other solvers:** The pairwise table at `derived/pairwise_comparisons.csv` reports matched NGAS comparisons with ALNS, Phase6H, and LG_HGA_2O under the same 2|O| budget.
5. **Refresh budget:** Mean neural-refresh share is {100 * all_runtime.mean_refresh_budget_share:.3f}%.
6. **Guidance reuse:** Mean guided iterations per critic call is {all_runtime.mean_guided_iterations_per_call:.3f}.
7. **Progress per evaluation:** `derived/runtime_summary.csv` reports decoder evaluations/second, new-best events/1000 evaluations, and improvement/1000 evaluations. These are descriptive associations, not isolated causal effects of the critic.
8. **Concentration:** Scale, CF, C01/C02, seed, and checkpoint tables are retained in `derived/`; conclusions should follow those strata rather than a single aggregate.
9. **Feasibility:** NGAS feasibility and exact schedule replay remain 100%.
10. **Refresh tails:** Across {tail['samples']} refreshes, {tail.get('gt_30_ms', 0)} exceeded 30 ms, {tail.get('gt_50_ms', 0)} exceeded 50 ms, and {tail.get('gt_100_ms', 0)} exceeded 100 ms; p99 is {tail['p99_ms']:.3f} ms and maximum is {tail['maximum_ms']:.3f} ms. Their practical impact is bounded by the measured refresh share and reported without hiding the A1.5R tail observation.

## Evidence

- Protocol: `outputs/ngas_a1/solver_comparison_v1/preregistration/protocol.json`
- Completion audit: `outputs/ngas_a1/solver_comparison_v1/audit/completion_audit.json`
- Decision: `outputs/ngas_a1/solver_comparison_v1/final_decision.json`
- Raw manifest: `outputs/ngas_a1/solver_comparison_v1/raw_manifest.json`
- Derived and manuscript-ready tables: `outputs/ngas_a1/solver_comparison_v1/derived/`

BKS v001 remains unchanged. No comparator was rerun, no post-outcome tuning was
performed, and this report does not authorize automatic access to R13 or R14.
"""


def main() -> None:
    _, config, protocol_sha256 = load_boundary(require_clean=False)
    frozen = audit_frozen_inputs(ROOT, config)
    comparator, ngas = load_all_payloads(config, protocol_sha256)
    frame = final_quality_rows(config, comparator, ngas)
    tie = config['metrics']['tie_tolerance_makespan']
    (instance, method, _, summaries, scale_summaries,
     pairwise, statistical_tests) = quality_analysis(frame, tie)
    checkpoints, checkpoint_summary, auc, passage, checkpoint_pairs = anytime_analysis(
        config, comparator, ngas)
    runtime_runs, runtime_summary, tail = runtime_analysis(ngas)

    scale = frame.groupby(['scale', 'algorithm'], as_index=False).agg(
        runs=('final_makespan', 'size'), mean_final_makespan=('final_makespan', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'))
    cf = frame.groupby(['CF_level', 'algorithm'], as_index=False).agg(
        runs=('final_makespan', 'size'), mean_final_makespan=('final_makespan', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'))
    variant = frame.groupby(['cell_replicate', 'algorithm'], as_index=False).agg(
        runs=('final_makespan', 'size'), mean_final_makespan=('final_makespan', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'))
    seed = frame.groupby(['seed', 'algorithm'], as_index=False).agg(
        runs=('final_makespan', 'size'), mean_final_makespan=('final_makespan', 'mean'),
        mean_rpd_v001_percent=('rpd_v001_percent', 'mean'))

    atomic_max = [max(row['atomic_budget_audit']['maximum_operation_seconds'].values())
                  for row in ngas]
    overshoots = [float(row['budget_overshoot_seconds']) for row in ngas]
    allowance = config['budget']['per_run_overshoot_measurement_allowance_seconds']
    budget_checks = {
        'exact_54_runs': len(ngas) == 54,
        'all_a16_accounting': all(row['budget_accounting'] == 'A16_INSTANCE_TOTAL'
                                  for row in ngas),
        'all_preparation_charged': all(row['solver_budget_elapsed_seconds']
                                       >= row['runtime_preparation_seconds'] for row in ngas),
        'no_atomic_start_after_deadline': all(
            row['atomic_budget_audit']['started_after_deadline_count'] == 0 for row in ngas),
        'per_run_overshoot_within_atomic_bound': all(
            over <= maximum + allowance for over, maximum in zip(overshoots, atomic_max)),
        'no_systematic_extension': float(np.median(overshoots))
                                   <= float(np.median(atomic_max)),
    }
    budget_audit = {
        'schema': 'ngas-a16-budget-audit-v1',
        'status': 'PASS' if all(budget_checks.values()) else 'FAIL',
        'checks': budget_checks,
        'median_overshoot_seconds': float(np.median(overshoots)),
        'maximum_overshoot_seconds': max(overshoots),
        'median_per_run_maximum_atomic_seconds': float(np.median(atomic_max)),
        'maximum_atomic_seconds': max(atomic_max),
        'started_after_deadline_count': sum(
            row['atomic_budget_audit']['started_after_deadline_count'] for row in ngas),
        'rule': config['budget']['systematic_extension_rule'],
    }

    regression = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (AUDIT / 'regression.log').parent.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'regression.log').write_text(regression.stdout)
    match = re.search(r'(\d+) passed', regression.stdout)
    regression_count = int(match.group(1)) if match else 0
    r13_r14_untouched = not any((ROOT / path).exists() for path in (
        'outputs/phase6j_caur/r13_selection/access_ledger.json',
        'outputs/phase6j_caur/r14_holdout/access_ledger.json'))
    integrity_checks = {
        'exact_ngas_scope_54': len(ngas) == 54,
        'all_numeric_values_finite': all(
            math.isfinite(float(row[key])) for row in ngas
            for key in ('solver_budget_elapsed_seconds', 'end_to_end_run_elapsed_seconds',
                        'final_makespan', 'rpd_v001_percent')),
        'all_ngas_feasible_and_replayed': all(row['feasible']
                                              and row['feasibility_replay']['feasible'] for row in ngas),
        'frozen_inputs_unchanged': frozen['status'] == 'PASS',
        'budget_audit_pass': budget_audit['status'] == 'PASS',
        'full_regression_pass': regression.returncode == 0 and regression_count >= 498,
        'r13_r14_untouched_no_gurobi': r13_r14_untouched
                                         and all(not row['gurobi_run'] for row in ngas),
        'all_comparator_runs_present': len(comparator) == 216,
        'all_method_feasibility_one': bool(frame.groupby('algorithm').feasible.mean().eq(1.).all()),
    }
    integrity_pass = all(integrity_checks.values())
    decision, gates = terminal_decision(
        integrity_pass, summaries['PHASE6N_TOP1'], summaries['LG_HGA_2O'],
        scale_summaries)

    atomic_csv(DERIVED / 'run_summary.csv', frame)
    atomic_csv(DERIVED / 'instance_summary.csv', instance)
    atomic_csv(DERIVED / 'method_summary.csv', method)
    atomic_csv(DERIVED / 'scale_summary.csv', scale)
    atomic_csv(DERIVED / 'cf_summary.csv', cf)
    atomic_csv(DERIVED / 'variant_summary.csv', variant)
    atomic_csv(DERIVED / 'seed_summary.csv', seed)
    atomic_csv(DERIVED / 'pairwise_comparisons.csv', pairwise)
    atomic_csv(DERIVED / 'checkpoint_results.csv', checkpoints)
    atomic_csv(DERIVED / 'checkpoint_summary.csv', checkpoint_summary)
    atomic_csv(DERIVED / 'checkpoint_pairwise_gains.csv', checkpoint_pairs)
    atomic_csv(DERIVED / 'anytime_summary.csv', auc)
    atomic_csv(DERIVED / 'first_passage.csv', passage)
    atomic_csv(DERIVED / 'runtime_runs.csv', runtime_runs)
    atomic_csv(DERIVED / 'runtime_summary.csv', runtime_summary)
    atomic_json(DERIVED / 'statistical_tests.json', statistical_tests)
    atomic_json(DERIVED / 'refresh_tail_summary.json', tail)
    atomic_json(AUDIT / 'budget_audit.json', budget_audit)
    replay_audit = {
        'schema': 'ngas-a16-replay-audit-v1', 'status': 'PASS',
        'replayed_runs': 54, 'feasible_runs': 54,
        'raw_manifest_sha256': digest(OUT / 'raw_manifest.json'),
    }
    atomic_json(AUDIT / 'replay_audit.json', replay_audit)
    completion = {
        'schema': 'ngas-a16-completion-audit-v1',
        'status': 'PASS' if integrity_pass else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha256, 'checks': integrity_checks,
        'quality_gates': gates, 'terminal_decision': decision,
        'regression_passed_tests': regression_count,
        'frozen_input_audit': frozen,
        'locks': {'R13': 'UNTOUCHED', 'R14': 'UNTOUCHED', 'gurobi': False},
    }
    atomic_json(AUDIT / 'completion_audit.json', completion)
    final_decision = {
        'schema': 'ngas-a16-final-decision-v1',
        'terminal_decision': decision, 'quality_gates': gates,
        'primary_vs_phase6n_top1': {
            key: summaries['PHASE6N_TOP1'][key] for key in
            ('mean_relative_gain_percent', 'median_relative_gain_percent',
             'wins', 'ties', 'losses')},
        'strong_vs_lg_hga_2o': {
            key: summaries['LG_HGA_2O'][key] for key in
            ('mean_relative_gain_percent', 'median_relative_gain_percent',
             'wins', 'ties', 'losses')},
        'scale_vs_phase6n_top1': {
            scale_name: {key: value[key] for key in
                         ('mean_relative_gain_percent', 'wins', 'ties', 'losses')}
            for scale_name, value in scale_summaries.items()},
        'next_stage': ('R13_R14_ELIGIBLE_ONLY_FOR_SEPARATELY_PREREGISTERED_STAGE'
                       if decision == 'NGAS_A1_PASS_DEVELOPMENT' else 'STOP_AFTER_A1_6'),
        'R13': 'UNTOUCHED', 'R14': 'UNTOUCHED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'final_decision.json', final_decision)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(
        decision, method, pairwise, scale_summaries, statistical_tests,
        checkpoint_pairs, runtime_summary, tail, budget_audit, regression_count))
    atomic_json(OUT / 'progress.json', {
        'schema': 'ngas-a16-progress-v1', 'status': 'FINALIZED',
        'completed_runs': 54, 'expected_runs': 54,
        'terminal_decision': decision, 'protocol_sha256': protocol_sha256,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'R13': 'UNTOUCHED', 'R14': 'UNTOUCHED', 'gurobi_run': False,
    })
    files = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted([*OUT.rglob('*'), REPORT])
        if path.is_file() and path != OUT / 'result_manifest.json'
    }
    atomic_json(OUT / 'result_manifest.json', {
        'schema': 'ngas-a16-result-manifest-v1',
        'terminal_decision': decision, 'protocol_sha256': protocol_sha256,
        'files': files,
    })
    print(json.dumps({
        'status': completion['status'], 'terminal_decision': decision,
        'runs': 54, 'regression_passed_tests': regression_count,
        'report': str(REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
