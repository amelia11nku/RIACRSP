#!/usr/bin/env python3
"""Persistent A1.6 worker wrapper that preserves the formal runner exit code."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/solver_comparison_v1'
PYTHON = '/home/liulei/miniconda3/envs/gnn311/bin/python'


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main() -> None:
    command = [PYTHON, '-u', 'scripts/run_ngas_a16_solver_comparison.py',
               '--device', 'cuda:0']
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT)
    atomic_json(OUT / 'formal_exit_status.json', {
        'schema': 'ngas-a16-worker-exit-v1',
        'command': command, 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'exit_code': result.returncode,
        'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
    })
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
