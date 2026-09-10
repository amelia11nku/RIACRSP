#!/usr/bin/env python3
"""Launch and verify the detached A1.3 grouped OOF training worker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/liulei/miniconda3/envs/gnn311/bin/python'
OUT = ROOT / 'outputs/ngas_a1/critic_training_gpu_v1'


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def active_workers() -> list[int]:
    listing = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    return sorted({int(line.strip().split(maxsplit=1)[0]) for line in listing.splitlines()
                   if 'scripts/train_ngas_joint_critic.py --device' in line
                   and 'launch_ngas_joint_critic.py' not in line})


def main() -> None:
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Formal critic training requires a clean committed worktree')
    if active_workers():
        raise RuntimeError(f'NGAS critic training already active: {active_workers()}')
    from scripts.train_ngas_joint_critic import PROTOCOL, validate_protocol
    from rcias_ngas.critic.train import digest
    protocol = validate_protocol()
    if not protocol['environment']['cuda_available']:
        raise RuntimeError('Frozen GPU environment is unavailable')
    subprocess.run([
        PYTHON, '-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
        'print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda").item())'],
        cwd=ROOT, check=True)
    device = 'cuda'
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    log_path = OUT / f'training_{stamp}.log'
    command = [PYTHON, '-u', 'scripts/train_ngas_joint_critic.py', '--device', device]
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = '0'
    if device == 'cuda':
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    log = log_path.open('w')
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 55
    observed_start = False
    first_epoch_seconds = None
    while time.monotonic() < deadline:
        log.flush()
        for line in log_path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'ngas_training_started':
                observed_start = True
            if event.get('event') == 'ngas_oof_epoch' and event.get('epoch') == 1:
                first_epoch_seconds = float(event['elapsed_seconds'])
                break
        if process.poll() is not None or first_epoch_seconds is not None:
            break
        time.sleep(1)
    log.close()
    if process.poll() is not None:
        if process.returncode == 0 and (OUT / 'final_decision.json').exists():
            observed_start = True
        else:
            tail = log_path.read_text(errors='replace')[-5000:]
            raise RuntimeError(f'Training worker failed launch verification:\n{tail}')
    if not observed_start:
        tail = log_path.read_text(errors='replace')[-5000:]
        raise RuntimeError(f'Training worker failed launch verification:\n{tail}')
    projected_seconds = max(900., (first_epoch_seconds or 36.) * 60 * 10 * 1.35)
    record = {
        'schema': 'ngas-a13-training-launch-v1',
        'status': 'COMPLETE_DURING_LAUNCH' if process.poll() == 0 else 'RUNNING_VERIFIED',
        'pid': process.pid, 'command': command, 'device': device,
        'working_directory': str(ROOT),
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'training_protocol_sha256': digest(PROTOCOL),
        'started_at_utc': started.isoformat(), 'projected_seconds': projected_seconds,
        'expected_completion_utc': (started + timedelta(seconds=projected_seconds)).isoformat(),
        'eta_basis': 'first OOF epoch wall time times 60 epochs and at most ten fits, with 35 percent reserve',
        'first_epoch_seconds': first_epoch_seconds,
        'log_path': str(log_path.relative_to(ROOT)),
        'output_path': str(OUT.relative_to(ROOT)),
        'progress_path': str((OUT / 'progress.json').relative_to(ROOT)),
        'check_command': 'cat outputs/ngas_a1/critic_training_gpu_v1/progress.json',
        'resume_command': f'{PYTHON} scripts/launch_ngas_joint_critic.py',
        'resume_semantics': 'validated complete seed/fold runs are reused; only an incomplete current run is recomputed',
        'liveness_evidence': ('worker completed successfully during launch verification'
                              if process.poll() == 0 else
                              'worker alive and cache/protocol validation start event observed'),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    atomic_json(OUT / 'launch_record.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
