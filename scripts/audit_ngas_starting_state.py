#!/usr/bin/env python3
"""Capture the pre-edit NGAS boundary without modifying historical evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with (ROOT / path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch
    destination = ROOT / 'outputs/ngas_a1/audit/starting_state.json'
    if destination.exists():
        raise FileExistsError('Starting boundary is immutable')
    registry_path = 'outputs/frozen_2o_baselines/registry.json'
    manifest_path = 'outputs/frozen_2o_baselines/instance_manifest.json'
    registry = json.loads((ROOT / registry_path).read_text())
    manifest = json.loads((ROOT / manifest_path).read_text())
    protected = {}
    mismatches = []

    def verify(path, expected=None):
        actual = digest(path)
        protected[path] = actual
        if expected is not None and actual != expected:
            mismatches.append(path)

    verify(registry_path)
    verify(manifest_path)
    for entry in registry['entries']:
        for field in ('source_files', 'shared_source_files', 'algorithm_config_hashes'):
            for path, expected in entry[field].items():
                verify(path, expected)
        for prefix in ('run_manifest', 'aggregate_summary'):
            verify(entry[prefix + '_path'], entry[prefix + '_sha256'])
        for row in json.loads((ROOT / entry['run_manifest_path']).read_text())['runs']:
            verify(row['path'], row['sha256'])
    for row in manifest['instances']:
        verify('instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/' + row['relative_path'], row['sha256'])
    checkpoint_manifest = 'outputs/phase6p_adaptive_portfolio_v1/preregistration/phase6n_checkpoint_manifest.json'
    verify(checkpoint_manifest)
    for row in json.loads((ROOT / checkpoint_manifest).read_text())['checkpoints']:
        verify(row['path'], row['sha256'])
    verify('outputs/phase6h_calibration/frozen/phase6h_policy.json')
    for path in sorted((ROOT / 'outputs/phase6p_adaptive_portfolio_v1').rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.csv'}:
            verify(str(path.relative_to(ROOT)))
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = {
        'schema': 'ngas-starting-state-v1', 'starting_commit': sha,
        'python_executable': sys.executable, 'python_version': sys.version,
        'torch_version': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'registry_status': registry['status'], 'registry': registry,
        'canonical_manifest_path': manifest_path, 'canonical_manifest_sha256': digest(manifest_path),
        'development_seeds': manifest['development_seeds'],
        'phase6p_terminal': json.loads((ROOT / 'outputs/phase6p_adaptive_portfolio_v1/final/final_decision.json').read_text()),
        'protected_hashes': protected, 'hash_mismatches': mismatches,
        'historical_bks': 'Phase6P finalizer min(final_makespan) over its 270 raw runs; no standalone immutable BKS version',
        'historical_anytime': 'incumbent_trace last best event <= checkpoint; decoder count is last-best count, not cumulative work',
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    report = ROOT / 'docs/reports/ngas_a1/00_starting_state_audit.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f'''# NGAS-A1 starting-state audit

Starting commit: `{sha}`. Origin: `https://github.com/amelia11nku/RIACRSP.git`.
The working tree was clean before this audit script was added.

Environment: `{sys.executable}`, Python {sys.version.split()[0]}, PyTorch {torch.__version__},
CUDA build {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}.
CPU is sufficient for infrastructure and the bounded label pilot; GPU training is not qualified.

Canonical manifest SHA256: `{digest(manifest_path)}`. 18 R12 DEVELOPMENT instances;
seeds {manifest['development_seeds']}. All four comparators have 54 frozen raw runs.
Verified {len(protected)} files; mismatches: {mismatches}.
Exact implementation commits, source/config/checkpoint identifiers, raw run manifests,
and retained hashes are in `outputs/ngas_a1/audit/starting_state.json`.

Phase 6P is terminal `ADAPTIVE_PORTFOLIO_PILOT_NO_GO`. No historical implementation
or raw result is to be changed. R13/R14 remain locked; no Gurobi work is authorized.

The historical BKS was computed from the 270 Phase 6P development/comparator raw runs.
NGAS will materialize that exact scope as immutable BKS v001 and separate raw provenance
from derived RPD. Historical anytime decoder counts come from incumbent events;
the makespan curve remains useful but the work-count column is not a budget counter.

Execution order: A1.0 infrastructure tests; A1.1 critical-sync/bank validation;
A1.2 preregistered bounded continuation pilot; training only after the pilot gate passes.
No solver-quality comparison before the RNG gate, and no large labeling before the pilot.
''')
    print(json.dumps({'starting_commit': sha, 'verified_files': len(protected), 'mismatches': mismatches}))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
