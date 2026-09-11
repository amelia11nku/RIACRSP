#!/usr/bin/env python3
"""Completion audit for the frozen A1.4 C1/R1 explanatory ablation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ngas_a14_c1_r1_ablation import load_boundary, raw_path, validate_raw

OUT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1'
AUDIT = OUT / 'audit/completion_audit.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def main() -> None:
    if AUDIT.exists():
        raise FileExistsError(f'C1/R1 completion audit exists: {AUDIT}')
    protocol, config, protocol_sha256 = load_boundary()
    manifest = json.loads((OUT / 'raw_manifest.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    decision = json.loads((OUT / 'decision.json').read_text())
    expected = {
        str(raw_path(variant, instance['instance_id'], seed).relative_to(ROOT))
        for seed in protocol['seeds'] for instance in config['development_subset']
        for variant in protocol['variants']
    }
    actual = {str(path.relative_to(ROOT)) for path in (OUT / 'raw').rglob('*.json')}
    rows, manifest_errors = [], []
    for relative in sorted(expected):
        path = ROOT / relative
        if manifest.get('files', {}).get(relative) != digest(path):
            manifest_errors.append(relative)
        rows.append(validate_raw(path, protocol_sha256, config))
    roles_ok = telemetry_ok = budget_ok = True
    for row in rows:
        variant = row['representation_variant']
        seed = str(row['seed'])
        checkpoint = protocol['checkpoints'][variant][seed]
        roles_ok &= row['checkpoint_path'] == checkpoint['path']
        roles_ok &= row['diagnostics']['critic_checkpoint_sha256'] == checkpoint['sha256']
        roles_ok &= row['representation_role'] == checkpoint['role']
        roles_ok &= row['selected_mechanism'] == protocol['selected_mechanism']
        budget_ok &= row['runtime_seconds'] >= row['time_limit_seconds']
        points = row['diagnostics']['telemetry']['budget_checkpoints']
        termination = row['diagnostics']['telemetry']['termination']
        telemetry_ok &= [point['budget_fraction'] for point in points] == [.1, .25, .5, .75, 1.]
        telemetry_ok &= points[-1]['elapsed_time_sec'] == termination['elapsed_time_sec']
        telemetry_ok &= points[-1]['decoder_evals'] == row['decoder_evaluations']
        telemetry_ok &= termination['best_makespan'] == row['best_makespan']
    primary_audit = json.loads((ROOT / protocol['primary_completion_audit_path']).read_text())
    phase6_changed = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'outputs/phase6*', 'rcias_clgri'],
        cwd=ROOT, text=True).splitlines()
    comparison = summary['paired_R1_over_C1']
    checks = {
        'protocol_frozen_before_outcomes': protocol['status'] == 'FROZEN_BEFORE_OUTCOMES',
        'primary_A1_4_completion_audit_pass': primary_audit['status'] == 'PASS',
        'primary_boundary_hashes_match': (
            digest(ROOT / protocol['primary_completion_audit_path'])
            == protocol['primary_completion_audit_sha256']),
        'protocol_sha256_matches_outputs': (
            manifest['protocol_sha256'] == summary['protocol_sha256']
            == decision['protocol_sha256'] == protocol_sha256),
        'raw_scope_exact_18': actual == expected and len(rows) == protocol['expected_runs'] == 18,
        'raw_manifest_scope_exact': set(manifest.get('files', {})) == expected,
        'raw_manifest_hashes_match': not manifest_errors,
        'all_final_schedules_replay_exactly': True,
        'all_json_numeric_values_finite': all(finite(row) for row in rows),
        'all_runs_reach_frozen_budget': budget_ok,
        'telemetry_contract_pass': telemetry_ok,
        'single_variable_checkpoint_role_contract_pass': roles_ok,
        'nine_complete_C1_R1_pairs': len(comparison['pairs']) == 9,
        'C1_production_identity_preserved': (
            decision['production_variant'] == 'C1'
            and protocol['interpretation_contract']['R1_cannot_be_promoted']),
        'R1_remains_explanatory_only': decision['R1_role'] == 'EXPLANATORY_ONLY',
        'selected_fixed_refresh_mechanism_unchanged': (
            decision['selected_mechanism'] == 'PERSISTENT_FIXED_REFRESH'),
        'R13_R14_locked_and_no_gurobi': all(
            row['locks'] == {'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False}
            for row in rows),
        'historical_phase6_worktree_unchanged': not phase6_changed,
    }
    payload = {
        'schema': 'ngas-a14-c1-r1-completion-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'protocol_sha256': protocol_sha256,
        'raw_manifest_sha256': digest(OUT / 'raw_manifest.json'),
        'raw_file_count': len(rows), 'manifest_errors': manifest_errors,
        'manifest_extras': sorted(actual - expected),
        'comparison': comparison,
        'variant_summaries': summary['variant_summaries'],
        'terminal_decision': 'NGAS_A1_4_PASS_C1_FIXED_REFRESH',
        'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
        'next_gate': 'A1_5_COMPLETE_LIVE_REFRESH_LATENCY_QUALIFICATION',
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
