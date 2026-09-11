#!/usr/bin/env python3
"""Freeze the pre-outcome A1.5 measurement-runner device-index amendment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/latency_qualification_v1'
V1 = OUT / 'preregistration/protocol.json'
V2 = OUT / 'preregistration/protocol_v2.json'
FAILURE = OUT / 'preregistration/rejected_v1/preflight_failure.json'
PROGRESS = OUT / 'progress.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main() -> None:
    if V2.exists():
        raise RuntimeError('A1.5 protocol v2 already exists')
    raw_files = list((OUT / 'raw').glob('*/*.json'))
    if raw_files:
        raise RuntimeError('Cannot amend protocol after formal timing outputs exist')
    protocol = json.loads(V1.read_text())
    failure = {
        'schema': 'ngas-a15-protocol-v1-preflight-failure-v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_path': str(V1.relative_to(ROOT)),
        'protocol_sha256': digest(V1),
        'failure_phase': 'before critic loading, warmup, or timing sample creation',
        'error': "RuntimeError: Invalid device argument from torch.cuda.reset_peak_memory_stats(torch.device('cuda:0'))",
        'raw_timing_files_present': False,
        'outcomes_observed': False,
        'allowed_amendment': 'replace the PyTorch memory-stat device argument with integer CUDA device index only',
    }
    atomic_json(FAILURE, failure)
    protocol.update({
        'schema': 'ngas-a15-latency-protocol-v2',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'supersedes_protocol_path': str(V1.relative_to(ROOT)),
        'supersedes_protocol_sha256': digest(V1),
        'pre_outcome_amendment_path': str(FAILURE.relative_to(ROOT)),
        'pre_outcome_amendment_sha256': digest(FAILURE),
        'measurement_runner_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'amendment_scope': 'PyTorch CUDA memory-stat API receives device index 0; scientific scope unchanged',
    })
    protocol['source_hashes'] = {
        path: digest(ROOT / path) for path in protocol['source_hashes']
    }
    atomic_json(V2, protocol)
    protocol_sha256 = digest(V2)
    atomic_json(PROGRESS, {
        'schema': 'ngas-a15-latency-progress-v1',
        'status': 'PROTOCOL_V2_FROZEN',
        'completed_units': 0, 'expected_units': 8,
        'protocol_sha256': protocol_sha256,
        'supersedes_protocol_sha256': digest(V1),
        'decision': 'PENDING_FORMAL_MEASUREMENT',
        'next_gate': 'A1_5_FORMAL_LATENCY_MEASUREMENT',
        'A1_6': 'LOCKED', 'R13': 'LOCKED', 'R14': 'LOCKED', 'gurobi_run': False,
    })
    print(json.dumps({'protocol': str(V2.relative_to(ROOT)),
                      'protocol_sha256': protocol_sha256,
                      'pre_outcome_amendment_sha256': digest(FAILURE)}, indent=2))


if __name__ == '__main__':
    main()
