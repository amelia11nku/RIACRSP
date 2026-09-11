#!/usr/bin/env python3
"""Launch and verify the detached frozen A1.4 C1/R1 ablation."""
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
OUT = ROOT / 'outputs/ngas_a1/search_integration_c1_r1_v1'
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
                   if 'scripts/run_ngas_a14_c1_r1_ablation.py' in line
                   and 'launch_ngas_a14_c1_r1.py' not in line})


def main() -> None:
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Formal C1/R1 launch requires a clean committed worktree')
    if active_workers():
        raise RuntimeError(f'C1/R1 worker already active: {active_workers()}')
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha256 = digest(PROTOCOL)
    subprocess.run([
        PYTHON, '-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
        'print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())'],
        cwd=ROOT, check=True)
    started = datetime.now(timezone.utc)
    log_path = OUT / f"formal_{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    command = [PYTHON, '-u', 'scripts/run_ngas_a14_c1_r1_ablation.py', '--device', 'cuda:0']
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = '0'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    log = log_path.open('w')
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 55
    observed = False
    while time.monotonic() < deadline:
        log.flush()
        for line in log_path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') in ('ablation_run_started', 'c1_r1_ablation_complete'):
                observed = True
                break
        if process.poll() is not None or observed:
            break
        time.sleep(1)
    log.close()
    if process.poll() is not None and process.returncode != 0:
        raise RuntimeError('C1/R1 worker failed launch verification:\n'
                           + log_path.read_text(errors='replace')[-5000:])
    if not observed:
        raise RuntimeError('C1/R1 worker emitted no verified start event')
    nominal = 1113.
    projected = nominal * 1.20 + 120.
    record = {
        'schema': 'ngas-a14-c1-r1-launch-v1',
        'status': 'COMPLETE_DURING_LAUNCH' if process.poll() == 0 else 'RUNNING_VERIFIED',
        'pid': process.pid, 'command': command, 'device': 'cuda:0',
        'working_directory': str(ROOT),
        'implementation_commit': protocol['implementation_commit'],
        'launch_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_sha256': protocol_sha256,
        'started_at_utc': started.isoformat(),
        'nominal_budget_seconds': nominal, 'projected_seconds': projected,
        'expected_completion_utc': (started + timedelta(seconds=projected)).isoformat(),
        'eta_basis': 'sum of 18 frozen wall-clock caps plus 20 percent and 120 seconds overhead',
        'log_path': str(log_path.relative_to(ROOT)),
        'output_path': str(OUT.relative_to(ROOT)),
        'progress_path': str((OUT / 'progress.json').relative_to(ROOT)),
        'check_command': 'cat outputs/ngas_a1/search_integration_c1_r1_v1/progress.json',
        'resume_command': f'{PYTHON} scripts/launch_ngas_a14_c1_r1.py',
        'resume_semantics': 'validated immutable completed raw runs are reused; only missing runs execute',
        'production_variant': 'C1', 'R1_role': 'EXPLANATORY_ONLY',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
