#!/usr/bin/env python3
"""Launch and verify the detached frozen A1.4 development worker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/liulei/miniconda3/envs/gnn311/bin/python'
OUT = ROOT / 'outputs/ngas_a1/search_integration_v1'
PROTOCOL = OUT / 'preregistration/protocol.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def active_workers() -> list[int]:
    listing = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    return sorted({int(line.strip().split(maxsplit=1)[0]) for line in listing.splitlines()
                   if 'scripts/run_ngas_a14_integration.py' in line
                   and 'launch_ngas_a14_integration.py' not in line})


def main() -> None:
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Formal A1.4 launch requires a clean committed worktree')
    if active_workers():
        raise RuntimeError(f'A1.4 worker already active: {active_workers()}')
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha256 = digest(PROTOCOL)
    config = json.loads((ROOT / protocol['config_path']).read_text())
    subprocess.run([
        PYTHON, '-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
        'print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())'],
        cwd=ROOT, check=True)
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    log_path = OUT / f'formal_{stamp}.log'
    command = [PYTHON, '-u', 'scripts/run_ngas_a14_integration.py', '--device', 'cuda:0']
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = '0'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    log = log_path.open('w')
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 55
    observed_start = False
    while time.monotonic() < deadline:
        log.flush()
        for line in log_path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') in ('run_started', 'a14_primary_complete'):
                observed_start = True
                break
        if process.poll() is not None or observed_start:
            break
        time.sleep(1)
    log.close()
    if process.poll() is not None and process.returncode != 0:
        raise RuntimeError('A1.4 worker failed launch verification:\n'
                           + log_path.read_text(errors='replace')[-5000:])
    if not observed_start:
        raise RuntimeError('A1.4 worker produced no verified start event:\n'
                           + log_path.read_text(errors='replace')[-5000:])
    nominal = sum(row['num_operations'] for row in config['development_subset']) \
        * config['budget']['multiplier'] * len(config['development_seeds']) \
        * len(config['ablation_modes'])
    projected = nominal * 1.20 + 180.
    record = {
        'schema': 'ngas-a14-primary-launch-v1',
        'status': 'COMPLETE_DURING_LAUNCH' if process.poll() == 0 else 'RUNNING_VERIFIED',
        'pid': process.pid, 'command': command, 'working_directory': str(ROOT),
        'device': 'cuda:0', 'implementation_commit': protocol['implementation_commit'],
        'launch_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_sha256': protocol_sha256,
        'started_at_utc': started.isoformat(),
        'nominal_budget_seconds': nominal,
        'projected_seconds': projected,
        'expected_completion_utc': (started + timedelta(seconds=projected)).isoformat(),
        'eta_basis': 'sum of frozen per-run wall-clock caps plus 20 percent and 180 seconds overhead',
        'log_path': str(log_path.relative_to(ROOT)),
        'output_path': str(OUT.relative_to(ROOT)),
        'progress_path': str((OUT / 'progress.json').relative_to(ROOT)),
        'check_command': 'cat outputs/ngas_a1/search_integration_v1/progress.json',
        'resume_command': f'{PYTHON} scripts/launch_ngas_a14_integration.py',
        'resume_semantics': 'validated immutable completed raw runs are reused; only missing runs execute',
        'liveness_evidence': ('worker completed during verification' if process.poll() == 0
                              else 'worker alive and emitted a frozen run_started event'),
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
