#!/usr/bin/env python3
"""Detach expanded development labels; require fresh output and a live worker."""
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
from scripts.collect_ngas_development import OUT, atomic_json, verify_protocol


def launch():
    verify_protocol()
    progress_path = OUT / 'progress.json'
    if progress_path.exists() and json.loads(progress_path.read_text())['status'] == 'COMPLETE':
        raise RuntimeError('Collection already complete')
    with (OUT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = datetime.now(timezone.utc)
    log = OUT / ('collection_' + started.strftime('%Y%m%dT%H%M%SZ') + '.log')
    command = [sys.executable, '-u', 'scripts/collect_ngas_development.py']
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    observed = None
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        if progress_path.exists():
            record = json.loads(progress_path.read_text())
            if record.get('pid') == process.pid and record.get('new_actions_this_process', 0) > 0:
                observed = record
                break
        if process.poll() is not None:
            break
        time.sleep(.5)
    if observed is None or (process.poll() is not None and observed['status'] != 'COMPLETE'):
        raise RuntimeError('Development launch failed: ' + log.read_text()[-3000:])
    eta = observed['seconds_per_decoder'] * observed['remaining_label_decoder_upper_bound'] * 1.3 + 600
    record = {'status': 'RUNNING_VERIFIED', 'pid': process.pid, 'command': command,
              'started_at_utc': started.isoformat(), 'expected_remaining_seconds': eta,
              'expected_completion_utc': (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat(),
              'eta_basis': 'first L-state measured seconds/decoder x remaining upper bound x 1.3 + 10 min source/I/O reserve; conservative',
              'log_path': str(log.relative_to(ROOT)), 'output_path': str(OUT.relative_to(ROOT)),
              'observed_progress': observed,
              'resume_command': f'{sys.executable} scripts/launch_ngas_development.py',
              'check_command': 'cat outputs/ngas_a1/development_v1/progress.json',
              'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch()


if __name__ == '__main__':
    main()
