#!/usr/bin/env python3
"""Validate A1.5R development evidence and run the preformal regression."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/audit/preformal_audit.json'
CONFIG = ROOT / 'configs/ngas_a15r_runtime_revision_v1.json'
DEVELOPMENT = ROOT / 'outputs/ngas_a1/runtime_revision_v1/development/development_latency.json'
EQUIVALENCE = ROOT / 'outputs/ngas_a1/runtime_revision_v1/equivalence/equivalence_matrix.json'
SEARCH = ROOT / 'outputs/ngas_a1/runtime_revision_v1/audit/search_integration.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    config = json.loads(CONFIG.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    equivalence = json.loads(EQUIVALENCE.read_text())
    search = json.loads(SEARCH.read_text())
    completed = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    match = re.search(r'(\d+) passed', completed.stdout)
    test_count = int(match.group(1)) if match else None
    historical = json.loads((
        ROOT / config['historical_A1_5']['completion_audit']).read_text())
    checks = {
        'historical_A1_5_terminal_decision_preserved':
            historical['terminal_decision'] == 'NGAS_A1_REVISE_RUNTIME',
        'historical_A1_5_gate_failed': historical['latency_gate_pass'] is False,
        'development_headroom_pass': development['ready_to_freeze_formal'],
        'all_development_p90_le_27_ms': all(
            row['complete_refresh_ms']['p90'] <= 27.
            for row in development['results'].values()),
        'L_L_MAX_medians_below_25_ms': all(
            development['results'][label]['complete_refresh_ms']['p50'] < 25.
            for label in ('L', 'L_MAX')),
        'equivalence_matrix_pass': equivalence['pass'],
        'search_integration_pass': search['pass'],
        'regression_pass': completed.returncode == 0 and test_count == 492,
        'A1_6_locked': config['locks']['A1_6'] == 'LOCKED_UNTIL_A15R_PASS',
        'R13_R14_locked_no_gurobi': config['locks']['R13'] == 'LOCKED'
            and config['locks']['R14'] == 'LOCKED'
            and config['locks']['gurobi'] is False,
    }
    payload = {
        'schema': 'ngas-a15r-preformal-audit-v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': digest(CONFIG),
        'development_latency_sha256': digest(DEVELOPMENT),
        'equivalence_matrix_sha256': digest(EQUIVALENCE),
        'search_integration_sha256': digest(SEARCH),
        'regression': {'command': f'{sys.executable} -m pytest -q',
                       'returncode': completed.returncode,
                       'passed_tests': test_count,
                       'last_output_line': completed.stdout.strip().splitlines()[-1]},
        'checks': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL',
        'historical_A1_5_immutable': True, 'locks': config['locks'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUT)
    print(json.dumps({
        'output': str(OUT.relative_to(ROOT)), 'checks': checks,
        'status': payload['status'],
    }, indent=2))
    if payload['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
