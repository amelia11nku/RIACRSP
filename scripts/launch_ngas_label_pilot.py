#!/usr/bin/env python3
"""Detach the frozen pilot and verify real result production before returning."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ngas_label_pilot import OUT, atomic_json, verify_protocol


def main():
    verify_protocol()
    progress_path = OUT / 'progress.json'
    if progress_path.exists() and json.loads(progress_path.read_text())['status'] == 'COMPLETE':
        raise RuntimeError('Pilot already complete; do not rerun')
    command = [sys.executable, '-u', 'scripts/run_ngas_label_pilot.py']
    started = datetime.now(timezone.utc)
    log = OUT / ('pilot_' + started.strftime('%Y%m%dT%H%M%SZ') + '.log')
    previous = progress_path.read_text() if progress_path.exists() else None
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    deadline = time.monotonic() + 50
    observed = None
    while time.monotonic() < deadline:
        if progress_path.exists() and progress_path.read_text() != previous:
            observed = json.loads(progress_path.read_text())
            if observed['completed_actions'] >= 1 and log.stat().st_size:
                break
        if process.poll() is not None:
            break
        time.sleep(.5)
    if observed is None or (process.poll() is not None and observed['status'] != 'COMPLETE'):
        raise RuntimeError('Pilot launch failed: ' + log.read_text()[-3000:])
    remaining = max(0, 450 - observed['completed_actions'])
    # First state is L: extrapolation is deliberately conservative for M/S.
    eta = max(60, observed['mean_action_seconds'] * remaining * 1.25)
    record = {
        'status': 'RUNNING_VERIFIED' if process.poll() is None else 'COMPLETE',
        'pid': process.pid, 'command': command, 'log_path': str(log.relative_to(ROOT)),
        'output_path': str(OUT.relative_to(ROOT)), 'started_at_utc': started.isoformat(),
        'expected_completion_utc': (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat(),
        'expected_remaining_seconds': eta,
        'eta_basis': 'first L-state completed joint-action runtime x remaining action upper bound x 1.25',
        'resume_command': ' '.join(command),
        'check_command': 'cat outputs/ngas_a1/label_pilot/progress.json',
        'observed_progress': observed,
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    atomic_json(OUT / 'launch_record.json', record)
    if record['status'] == 'RUNNING_VERIFIED':
        atomic_json(ROOT / 'outputs/ngas_a1/final_decision.json', {
            'stage': 'A1.2', 'status': 'LABEL_PILOT_RUNNING', 'decision': 'PENDING_LABEL_PILOT',
            'A1.0': 'PASS', 'A1.1': 'PASS', 'pid': process.pid,
            'training_started': False, 'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
        })
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
