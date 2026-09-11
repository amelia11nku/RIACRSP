#!/usr/bin/env python3
"""Launch and verify the detached frozen A1.6 formal worker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/solver_comparison_v1'
PYTHON = '/home/liulei/miniconda3/envs/gnn311/bin/python'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def active_workers() -> list[int]:
    listing = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    return sorted({int(line.strip().split(maxsplit=1)[0]) for line in listing.splitlines()
                   if ('scripts/run_ngas_a16_worker.py' in line
                       or 'scripts/run_ngas_a16_solver_comparison.py --device' in line)
                   and 'launch_ngas_a16_solver_comparison.py' not in line})


def main() -> None:
    from scripts.run_ngas_a16_solver_comparison import (
        PROTOCOL, build_tasks, load_boundary,
    )
    if active_workers():
        raise RuntimeError(f'A1.6 formal worker already active: {active_workers()}')
    _, config, protocol_sha256 = load_boundary(require_clean=True)
    subprocess.run([
        PYTHON, '-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
        'print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())'],
        cwd=ROOT, check=True)
    tasks = build_tasks(config)
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    log_path = OUT / 'logs' / f'formal_{stamp}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', 'scripts/run_ngas_a16_worker.py']
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = '0'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    log = log_path.open('x')
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 55.
    observed_start = False
    while time.monotonic() < deadline:
        log.flush()
        text = log_path.read_text(errors='replace')
        observed_start = '"event": "a16_run_start"' in text \
            or '"event": "a16_formal_raw_complete"' in text
        if process.poll() is not None or observed_start:
            break
        time.sleep(1.)
    log.close()
    if process.poll() is not None and process.returncode != 0:
        raise RuntimeError(f'A1.6 worker failed launch verification:\n'
                           f'{log_path.read_text(errors="replace")[-6000:]}')
    if not observed_start:
        raise RuntimeError(f'A1.6 worker did not reach the first formal task:\n'
                           f'{log_path.read_text(errors="replace")[-6000:]}')
    nominal = sum(task['budget_seconds'] for task in tasks)
    projected = nominal * 1.08 + 300.
    record = {
        'schema': 'ngas-a16-launch-record-v1',
        'status': 'COMPLETE_DURING_LAUNCH' if process.poll() == 0 else 'RUNNING_VERIFIED',
        'pid': process.pid, 'session': f'detached_process_group_{process.pid}',
        'command': command, 'working_directory': str(ROOT),
        'implementation_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_path': str(PROTOCOL.relative_to(ROOT)),
        'protocol_sha256': protocol_sha256,
        'started_at_utc': started.isoformat(),
        'nominal_budget_seconds': nominal,
        'projected_seconds': projected,
        'expected_completion_utc': (started + timedelta(seconds=projected)).isoformat(),
        'eta_basis': '12,960 frozen solver-budget seconds plus 8 percent run overhead and 300 seconds setup/final replay reserve',
        'log_path': str(log_path.relative_to(ROOT)),
        'raw_output_path': str((OUT / 'raw').relative_to(ROOT)),
        'progress_path': str((OUT / 'progress.json').relative_to(ROOT)),
        'exit_status_path': str((OUT / 'formal_exit_status.json').relative_to(ROOT)),
        'inspection_command': 'cat outputs/ngas_a1/solver_comparison_v1/progress.json',
        'resume_command': f'{PYTHON} scripts/launch_ngas_a16_solver_comparison.py',
        'resume_semantics': 'validate and reuse every complete immutable formal raw; never rerun a valid result',
        'liveness_evidence': 'worker alive and first formal run start event observed',
        'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
