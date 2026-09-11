#!/usr/bin/env python3
"""Launch one persistent A1.6R systemd worker and verify lock ownership."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.a16_io import write_new_json  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import load_json  # noqa: E402
from scripts.run_ngas_a16r_solver_comparison import (  # noqa: E402
    OUT, PROTOCOL, build_tasks, load_boundary,
)


PYTHON = '/home/liulei/miniconda3/envs/gnn311/bin/python'


def main() -> None:
    _, config, protocol_sha256 = load_boundary(require_clean=True)
    tasks = build_tasks(config)
    if (OUT / 'integrity/formal.lock').exists():
        owner = load_json(OUT / 'integrity/formal.lock')
        raise RuntimeError(
            f"A1.6R already owned by session {owner.get('session_id')} "
            f"PID {owner.get('pid')}")
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    unit = f'ngas-a16r-v1-{stamp.lower()}'
    log_path = OUT / 'logs' / f'formal_systemd_{stamp}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        'systemd-run', '--user', f'--unit={unit}', '--collect',
        f'--property=WorkingDirectory={ROOT}',
        f'--property=StandardOutput=append:{log_path}',
        f'--property=StandardError=append:{log_path}',
        '--setenv=PYTHONHASHSEED=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
        PYTHON, '-u', 'scripts/run_ngas_a16r_solver_comparison.py',
        '--device', 'cuda:0',
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    lock_path = OUT / 'integrity/formal.lock'
    progress_path = OUT / 'progress.json'
    deadline = time.monotonic() + 60.
    while time.monotonic() < deadline:
        if lock_path.exists() and progress_path.exists():
            break
        state = subprocess.check_output(
            ['systemctl', '--user', 'show', unit, '--property=ActiveState',
             '--property=SubState', '--property=ExecMainStatus'], text=True)
        if 'ActiveState=failed' in state or 'SubState=failed' in state:
            raise RuntimeError(f'A1.6R service failed before acquiring ownership:\n{state}')
        time.sleep(1.)
    if not lock_path.exists() or not progress_path.exists():
        raise RuntimeError('A1.6R service did not acquire the lock and validate its queue')
    owner = load_json(lock_path)
    progress = load_json(progress_path)
    main_pid = int(subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '--property=MainPID', '--value'],
        text=True).strip())
    if main_pid != owner['pid'] or progress['session_id'] != owner['session_id']:
        raise RuntimeError('systemd PID, lock owner, and progress session disagree')
    nominal = sum(task['budget_seconds'] for task in tasks)
    projected = nominal * 1.10 + 600.
    record = {
        'schema': 'ngas-a16r-launch-record-v1',
        'status': 'RUNNING_VERIFIED',
        'unit': unit,
        'main_pid': main_pid,
        'formal_owner_id': owner['formal_owner_id'],
        'session_id': owner['session_id'],
        'process_start_ticks': owner['process_start_ticks'],
        'hostname': owner['hostname'],
        'command': command,
        'working_directory': str(ROOT),
        'implementation_commit': owner['implementation_commit'],
        'protocol_path': str(PROTOCOL.relative_to(ROOT)),
        'protocol_sha256': protocol_sha256,
        'started_at_utc': started.isoformat(),
        'nominal_budget_seconds': nominal,
        'projected_seconds': projected,
        'expected_completion_utc': (started + timedelta(seconds=projected)).isoformat(),
        'eta_basis': '12,960 seconds formal budget plus 10 percent diagnostics overhead and 600 seconds setup reserve',
        'log_path': str(log_path.relative_to(ROOT)),
        'raw_output_path': str((OUT / 'raw').relative_to(ROOT)),
        'progress_path': str(progress_path.relative_to(ROOT)),
        'session_log_path': owner['event_log_path'],
        'inspection_command': f'cat {progress_path.relative_to(ROOT)}',
        'service_command': f'systemctl --user show {unit}',
        'resume_command': f'{PYTHON} scripts/launch_ngas_a16r_solver_comparison.py',
        'resume_semantics': (
            'after an unclean exit, first prove and quarantine the stale lock; '
            'valid complete raw remains immutable and is revalidated before skipping'),
        'single_launch_policy': 'no detached fallback and no duplicate service',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    write_new_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
