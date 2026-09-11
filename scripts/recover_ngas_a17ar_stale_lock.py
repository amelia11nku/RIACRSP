#!/usr/bin/env python3
"""Quarantine a proven-dead A1.7A-R formal collection lock."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.execution import recover_stale_lock  # noqa: E402


OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1'


def main() -> None:
    recovered = recover_stale_lock(
        OUT / 'integrity/formal.lock', OUT / 'integrity/recovered_stale_locks')
    print(json.dumps({
        'status': 'PROVEN_DEAD_LOCK_QUARANTINED',
        'path': str(recovered.relative_to(ROOT)),
        'next_command': (
            '/home/liulei/miniconda3/envs/gnn311/bin/python -u '
            'scripts/run_ngas_a17ar_trajectory_collection.py --device cuda:0'),
    }, indent=2))


if __name__ == '__main__':
    main()
