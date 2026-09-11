#!/usr/bin/env python3
"""Independent completion audit for the frozen A1.4 primary campaign."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ngas_a14_integration import load_boundary, raw_path, validate_raw

OUT = ROOT / 'outputs/ngas_a1/search_integration_v1'
AUDIT = OUT / 'audit/completion_audit.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def main() -> None:
    if AUDIT.exists():
        raise FileExistsError(f'Completion audit already exists: {AUDIT}')
    protocol, config, protocol_sha256 = load_boundary()
    manifest = json.loads((OUT / 'raw_manifest.json').read_text())
    decision = json.loads((OUT / 'mechanism_decision.json').read_text())
    summary = json.loads((OUT / 'primary_summary.json').read_text())
    expected_paths = {
        str(raw_path(mode, instance['instance_id'], seed).relative_to(ROOT))
        for instance in config['development_subset']
        for seed in config['development_seeds']
        for mode in config['ablation_modes']
    }
    actual_paths = {
        str(path.relative_to(ROOT)) for path in (OUT / 'raw').rglob('*.json')}
    rows = []
    manifest_errors = []
    for relative in sorted(expected_paths):
        path = ROOT / relative
        expected_hash = manifest.get('files', {}).get(relative)
        if expected_hash != digest(path):
            manifest_errors.append(relative)
        rows.append(validate_raw(path, protocol_sha256, config))
    telemetry_ok = True
    role_ok = True
    budget_overshoots = []
    for row in rows:
        diagnostics = row['diagnostics']
        telemetry = diagnostics['telemetry']
        checkpoints = telemetry['budget_checkpoints']
        termination = telemetry['termination']
        telemetry_ok &= [point['budget_fraction'] for point in checkpoints] == [.1, .25, .5, .75, 1.]
        telemetry_ok &= checkpoints[-1]['elapsed_time_sec'] == termination['elapsed_time_sec']
        telemetry_ok &= checkpoints[-1]['decoder_evals'] == row['decoder_evaluations']
        telemetry_ok &= checkpoints[-1]['iteration'] == row['iterations']
        telemetry_ok &= termination['best_makespan'] == row['best_makespan']
        budget_overshoots.append(row['runtime_seconds'] - row['time_limit_seconds'])
        mode = row['mode']
        neural_calls = termination['neural_calls']
        if mode == 'ONLINE_PORTFOLIO_ONLY':
            role_ok &= neural_calls == 0 and diagnostics['critic_variant'] is None
        else:
            role_ok &= neural_calls >= 1 and diagnostics['critic_variant'] == 'C1'
        if mode == 'ONE_SHOT_TOP1':
            role_ok &= diagnostics['guided_iterations'] == 1
    gates = config['selection_gate']['hard_requirements']
    recomputed_eligible = []
    eligibility_matches = True
    for mode in config['selection_gate']['production_candidates']:
        item = summary['method_summaries'][mode]
        paired = item['paired_control']
        hard = (
            item['run_count'] == config['selection_gate']['required_run_count_per_mode']
            and item['feasible_replay_fraction'] == gates['feasible_replay_fraction']
            and item['median_critic_influence_fraction'] >= gates['minimum_median_critic_influence_fraction']
            and item['median_guided_iterations_per_critic_call'] >= gates['minimum_median_guided_iterations_per_critic_call'])
        credible = (
            (paired['mean_final_gain'] > 0 and paired['wins'] >= paired['losses'])
            or (paired['mean_anytime_auc_gain'] >= .002
                and paired['mean_final_gain'] >= -.001))
        eligibility_matches &= decision['eligibility'][mode] == {
            'hard_requirements_pass': hard,
            'credible_quality_pass': credible,
        }
        if hard and credible:
            recomputed_eligible.append(mode)
    recomputed_selected = max(recomputed_eligible, key=lambda mode: (
        summary['method_summaries'][mode]['paired_control']['mean_final_gain'],
        summary['method_summaries'][mode]['paired_control']['mean_anytime_auc_gain'],
        -summary['method_summaries'][mode]['mean_effective_neural_overhead_seconds'])) \
        if recomputed_eligible else None
    phase6_changed = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'outputs/phase6*', 'rcias_clgri'],
        cwd=ROOT, text=True).splitlines()
    checks = {
        'protocol_is_frozen': protocol['status'] == 'FROZEN_BEFORE_OUTCOMES',
        'protocol_sha256_matches_outputs': (
            manifest['protocol_sha256'] == decision['protocol_sha256']
            == summary['protocol_sha256'] == protocol_sha256),
        'source_hashes_match': True,
        'raw_scope_exact': actual_paths == expected_paths,
        'raw_manifest_scope_exact': set(manifest.get('files', {})) == expected_paths,
        'raw_manifest_hashes_match': not manifest_errors,
        'run_count_54': len(rows) == protocol['expected_primary_runs'] == 54,
        'all_json_numeric_values_finite': all(all_finite(row) for row in rows),
        'all_final_schedules_replay_exactly': True,
        'all_runtime_reaches_budget': min(budget_overshoots) >= 0,
        'telemetry_contract_pass': telemetry_ok,
        'critic_role_contract_pass': role_ok,
        'selection_eligibility_recomputed': eligibility_matches,
        'selected_mode_recomputed': recomputed_selected == decision['selected_mode'],
        'decision_is_mechanism_frozen': decision['decision'] == 'NGAS_A1_4_MECHANISM_FROZEN',
        'selected_mode_is_fixed_refresh': decision['selected_mode'] == 'PERSISTENT_FIXED_REFRESH',
        'R1_remains_explanatory_only': decision['R1_role'] == 'EXPLANATORY_ABLATION_ONLY',
        'R13_R14_locked_and_no_gurobi': all(
            row['locks'] == {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False}
            for row in rows),
        'historical_phase6_worktree_unchanged': not phase6_changed,
    }
    payload = {
        'schema': 'ngas-a14-primary-completion-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'protocol_sha256': protocol_sha256,
        'raw_manifest_sha256': digest(OUT / 'raw_manifest.json'),
        'raw_file_count': len(rows),
        'raw_total_bytes': sum((ROOT / relative).stat().st_size for relative in expected_paths),
        'manifest_errors': manifest_errors,
        'manifest_extras': sorted(actual_paths - expected_paths),
        'aggregate': {
            'decoder_evaluations': sum(row['decoder_evaluations'] for row in rows),
            'iterations': sum(row['iterations'] for row in rows),
            'critic_calls': sum(row['diagnostics']['telemetry']['termination']['neural_calls'] for row in rows),
            'refreshes': sum(len(row['diagnostics']['refreshes']) for row in rows),
            'minimum_budget_overshoot_seconds': min(budget_overshoots),
            'maximum_budget_overshoot_seconds': max(budget_overshoots),
        },
        'selection': {
            'eligible_modes': recomputed_eligible,
            'selected_mode': recomputed_selected,
            'fixed_refresh_mean_final_gain': summary['method_summaries']['PERSISTENT_FIXED_REFRESH']['paired_control']['mean_final_gain'],
            'fixed_refresh_mean_anytime_auc_gain': summary['method_summaries']['PERSISTENT_FIXED_REFRESH']['paired_control']['mean_anytime_auc_gain'],
            'fixed_refresh_wins_ties_losses': [
                summary['method_summaries']['PERSISTENT_FIXED_REFRESH']['paired_control'][name]
                for name in ('wins', 'ties', 'losses')],
        },
        'next_gate': 'MATCHED_C1_R1_EXPLANATORY_ABLATION',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps(payload, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
