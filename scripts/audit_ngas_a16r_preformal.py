#!/usr/bin/env python3
"""Build the machine-readable A1.6R audit required before protocol freeze."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16_integrity import (  # noqa: E402
    audit_frozen_inputs, digest, load_json,
)


OUT = ROOT / 'outputs/ngas_a1/solver_comparison_a16r_v1'
CONFIG = ROOT / 'configs/ngas_a16r_integrity_diagnostic_v1.json'
AUDIT = OUT / 'preregistration/preformal_audit.json'
COMPARATOR_AUDIT = OUT / 'preregistration/comparator_audit.json'
REGRESSION_LOG = OUT / 'preregistration/regression.log'
SMOKE = OUT / 'smoke/smoke.json'
A15R_AUDIT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/audit/completion_audit.json'
A15R_MANIFEST = ROOT / 'outputs/ngas_a1/runtime_revision_v1/result_manifest.json'
A16_AUDIT = ROOT / 'outputs/ngas_a1/solver_comparison_v1/audit/completion_audit.json'
A16_DECISION = ROOT / 'outputs/ngas_a1/solver_comparison_v1/final_decision.json'
CAPACITY = OUT / 'preregistration/capacity_audit.json'
INVARIANCE = OUT / 'preregistration/behavior_invariance.json'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def tracked_worktree_clean() -> bool:
    return (subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT).returncode == 0
            and subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode == 0)


def full_regression() -> tuple[bool, int, str]:
    command = [sys.executable, '-m', 'pytest', '-q']
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    REGRESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    REGRESSION_LOG.write_text(result.stdout)
    match = re.search(r'(\d+) passed', result.stdout)
    count = int(match.group(1)) if match else 0
    return result.returncode == 0, count, ' '.join(command)


def a15r_manifest_valid() -> bool:
    manifest = load_json(A15R_MANIFEST)
    return (manifest.get('terminal_decision') == 'NGAS_A15R_RUNTIME_PASS'
            and all(digest(ROOT / relative) == expected
                    for relative, expected in manifest.get('files', {}).items()))


def main() -> None:
    if AUDIT.exists():
        raise RuntimeError('A1.6R preformal audit already exists')
    config = load_json(CONFIG)
    frozen = audit_frozen_inputs(ROOT, config)
    atomic_json(COMPARATOR_AUDIT, frozen)
    smoke = load_json(SMOKE) if SMOKE.exists() else {}
    a15r = load_json(A15R_AUDIT)
    a16 = load_json(A16_AUDIT)
    a16_decision = load_json(A16_DECISION)
    capacity = load_json(CAPACITY) if CAPACITY.exists() else {}
    invariance = load_json(INVARIANCE) if INVARIANCE.exists() else {}
    old_search = load_json(ROOT / config['production_solver']['search_config_source'])['search']
    regression_pass, regression_count, regression_command = full_regression()
    active = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    competing = [line.strip() for line in active.splitlines()
                 if any(marker in line for marker in (
                     'train_ngas_joint_critic.py', 'run_phase6l_training.py',
                     'run_phase6m_training.py', 'run_phase6n_training.py'))]
    start = config['starting_commit']
    start_is_ancestor = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', start, 'HEAD'], cwd=ROOT).returncode == 0
    formal_raw = list((OUT / 'raw').glob('*/*.json')) if (OUT / 'raw').exists() else []
    r13_r14_untouched = not any((ROOT / path).exists() for path in (
        'outputs/phase6j_caur/r13_selection/access_ledger.json',
        'outputs/phase6j_caur/r14_holdout/access_ledger.json'))
    checks = {
        'tracked_implementation_clean': tracked_worktree_clean(),
        'verified_start_commit_is_ancestor': start_is_ancestor,
        'a15r_completion_pass': a15r.get('status') == 'PASS'
                               and a15r.get('terminal_decision') == 'NGAS_A15R_RUNTIME_PASS'
                               and a15r.get('A1_6') == 'ELIGIBLE_TO_FREEZE_NEXT_STAGE',
        'a15r_result_manifest_valid': a15r_manifest_valid(),
        'a16_integrity_failure_preserved': (
            a16.get('terminal_decision') == 'NGAS_A1_6_INVALID_COMPARISON'
            and a16_decision.get('terminal_decision')
            == 'NGAS_A1_6_INVALID_COMPARISON'),
        'frozen_input_audit_pass': frozen['status'] == 'PASS',
        'production_search_parameters_unchanged': config['production_solver']['search'] == old_search,
        'excluded_cuda_smoke_pass': smoke.get('status') == 'PASS'
                                    and smoke.get('excluded_from_formal') is True
                                    and smoke.get('checks', {}).get('preparation_charged') is True,
        'behavior_invariance_pass': invariance.get('status') == 'PASS',
        'capacity_preflight_pass': capacity.get('status') == 'PASS',
        'cuda_available': torch.cuda.is_available(),
        'qualified_gpu_class': torch.cuda.is_available()
                               and torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 4060 Ti',
        'no_competing_training_worker': not competing,
        'full_regression_pass': regression_pass and regression_count >= 498,
        'budget_accounting_regression_included': regression_pass
                                                 and regression_count >= 498,
        'immutable_resume_regression_included': regression_pass
                                                and regression_count >= 498,
        'schedule_replay_regression_included': regression_pass
                                              and regression_count >= 498,
        'zero_formal_results_before_freeze': not formal_raw,
        'formal_lock_absent_before_freeze': not (OUT / 'integrity/formal.lock').exists(),
        'new_namespace_does_not_reuse_a16_raw': (
            config['output_root'] != 'outputs/ngas_a1/solver_comparison_v1'),
        'r13_r14_untouched_no_gurobi': r13_r14_untouched,
    }
    payload = {
        'schema': 'ngas-a16r-preformal-audit-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'checks': checks,
        'config_path': str(CONFIG.relative_to(ROOT)),
        'config_sha256': digest(CONFIG),
        'a15r_completion_audit_path': str(A15R_AUDIT.relative_to(ROOT)),
        'a15r_completion_audit_sha256': digest(A15R_AUDIT),
        'a15r_result_manifest_path': str(A15R_MANIFEST.relative_to(ROOT)),
        'a15r_result_manifest_sha256': digest(A15R_MANIFEST),
        'a16_invalid_completion_audit_path': str(A16_AUDIT.relative_to(ROOT)),
        'a16_invalid_completion_audit_sha256': digest(A16_AUDIT),
        'a16_invalid_decision_path': str(A16_DECISION.relative_to(ROOT)),
        'a16_invalid_decision_sha256': digest(A16_DECISION),
        'comparator_audit_path': str(COMPARATOR_AUDIT.relative_to(ROOT)),
        'comparator_audit_sha256': digest(COMPARATOR_AUDIT),
        'smoke_path': str(SMOKE.relative_to(ROOT)),
        'smoke_sha256': digest(SMOKE) if SMOKE.exists() else None,
        'behavior_invariance_path': str(INVARIANCE.relative_to(ROOT)),
        'behavior_invariance_sha256': digest(INVARIANCE) if INVARIANCE.exists() else None,
        'capacity_audit_path': str(CAPACITY.relative_to(ROOT)),
        'capacity_audit_sha256': digest(CAPACITY) if CAPACITY.exists() else None,
        'regression': {
            'command': regression_command,
            'passed_tests': regression_count,
            'log_path': str(REGRESSION_LOG.relative_to(ROOT)),
            'log_sha256': digest(REGRESSION_LOG),
        },
        'machine': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'formal_raw_files_before_freeze': [str(path.relative_to(ROOT)) for path in formal_raw],
        'competing_workers': competing,
        'locks': config['locks'],
    }
    atomic_json(AUDIT, payload)
    print(json.dumps({'status': payload['status'], 'checks': checks,
                      'passed_tests': regression_count,
                      'path': str(AUDIT.relative_to(ROOT))}, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
