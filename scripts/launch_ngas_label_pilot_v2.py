#!/usr/bin/env python3
"""Launch one persistent V2 worker and verify a fresh durable label."""
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ngas_label_pilot_v2 import OUT, atomic_json, verify_protocol


def launch():
    verify_protocol()
    progress_path = OUT / 'progress.json'
    if progress_path.exists() and json.loads(progress_path.read_text())['status'] == 'COMPLETE':
        raise RuntimeError('V2 already complete; no rerun')
    # Check the worker lock as well as serializing concurrent launcher invocations.
    with (OUT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, '-u', 'scripts/run_ngas_label_pilot_v2.py']
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    log = OUT / f'pilot_v2_{stamp}.log'
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'PYTHONHASHSEED': '0',
                                                               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    deadline = time.monotonic() + 50
    observed = None
    while time.monotonic() < deadline:
        if progress_path.exists():
            value = json.loads(progress_path.read_text())
            if value.get('pid') == process.pid and value.get('new_replicates_this_process', 0) >= 1:
                observed = value
                break
        if process.poll() is not None:
            break
        time.sleep(.5)
    if observed is None or (process.poll() is not None and observed['status'] != 'COMPLETE'):
        raise RuntimeError('V2 worker did not produce a new durable label: ' + log.read_text()[-3000:])
    eta = max(60, observed['remaining_decoder_upper_bound'] * observed['observed_seconds_per_decoder'] * 1.3)
    record = {
        'status': 'RUNNING_VERIFIED' if process.poll() is None else 'COMPLETE', 'pid': process.pid,
        'command': command, 'log_path': str(log.relative_to(ROOT)), 'output_path': str(OUT.relative_to(ROOT)),
        'started_at_utc': started.isoformat(), 'expected_remaining_seconds': eta,
        'expected_completion_utc': (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat(),
        'eta_basis': 'first L-state seconds per decoder x exact remaining decoder upper bound x 1.3',
        'observed_progress': observed,
        'resume_command': f'{sys.executable} scripts/launch_ngas_label_pilot_v2.py',
        'check_command': 'cat outputs/ngas_a1/label_pilot_v2/progress.json',
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch()


if __name__ == '__main__':
    main()
