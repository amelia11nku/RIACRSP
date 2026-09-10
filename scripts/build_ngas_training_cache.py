#!/usr/bin/env python3
"""Build the outcome-preserving, compact A1.3 critic training cache."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.bks import content_hash, write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.collect_ngas_development import OUT as DEVELOPMENT, verify_protocol

OUT = ROOT / 'outputs/ngas_a1/critic_training_v1'
CACHE = OUT / 'training_cache.json.gz'
MANIFEST = OUT / 'training_cache_manifest.json'
AUDIT = ROOT / 'outputs/ngas_a1/audit/development_completion.json'


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop('payload_sha256')
    if content_hash(value) != expected:
        raise ValueError(f'Payload hash mismatch: {path}')
    return value


def publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Immutable cache differs: {path}')
        return
    handle, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    protocol = verify_protocol()
    audit = json.loads(AUDIT.read_text())
    gate = json.loads((DEVELOPMENT / 'data_gate.json').read_text())
    if audit['status'] != 'PASS' or audit['decision'] != 'READY_FOR_JOINT_CRITIC_TRAINING':
        raise RuntimeError('Independent development completion audit has not passed')
    if gate['decision'] != 'READY_FOR_JOINT_CRITIC_TRAINING':
        raise RuntimeError('Development data gate has not passed')
    result_manifest = json.loads((DEVELOPMENT / 'result_hash_manifest.json').read_text())
    actual_files = {str(path.relative_to(ROOT)) for path in (DEVELOPMENT / 'states').rglob('*.json')}
    if actual_files != set(result_manifest):
        raise ValueError('Development raw file set differs from its frozen manifest')
    for relative, expected in result_manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Development raw file drift: {relative}')

    records = []
    actions_total = replicates_total = 0
    instances_by_fold = {str(fold): set() for fold in range(3)}
    for spec in protocol['states']:
        directory = DEVELOPMENT / 'states' / spec['state_id']
        state = unseal(directory / 'state.json')
        action_record = unseal(directory / 'actions.json')
        label_by_id = {}
        for path in (directory / 'labels').glob('*.json'):
            label = unseal(path)
            label_by_id[label['action']['action_id']] = label
        action_ids = [row['action_id'] for row in action_record['actions']]
        if len(label_by_id) != len(action_ids) or set(label_by_id) != set(action_ids):
            raise ValueError(f'Action/label identity mismatch: {spec["state_id"]}')
        advantages = []
        beat_frequencies = []
        for action_id in action_ids:
            label = label_by_id[action_id]
            values = [float(row['advantage']) for row in label['replicates']]
            if len(values) != protocol['config']['replicates']:
                raise ValueError('Replicate count drift')
            if statistics.fmean(values) != label['advantage_mean']:
                raise ValueError('Cached advantage mean differs from raw replicates')
            frequency = statistics.fmean(float(value > 0) for value in values)
            if frequency != label['beats_fallback_frequency']:
                raise ValueError('Cached beats-fallback frequency differs')
            advantages.append(values)
            beat_frequencies.append(frequency)
        instance = spec['instance']
        instances_by_fold[str(spec['fold'])].add(instance['instance_id'])
        records.append({
            'state_id': spec['state_id'],
            'instance_id': instance['instance_id'],
            'fold': int(spec['fold']),
            'scale': instance['scale'],
            'CF_level': instance['CF_level'],
            'source': spec['source'],
            'state_features': state['csg_features'],
            'action_features': action_record['action_features'],
            'actions': action_record['actions'],
            'replicate_advantages': advantages,
            'beats_fallback_frequency': beat_frequencies,
        })
        actions_total += len(action_ids)
        replicates_total += sum(map(len, advantages))

    payload = {
        'schema': 'ngas-a13-training-cache-v1',
        'development_protocol_sha256': digest(DEVELOPMENT / 'protocol.json'),
        'development_result_manifest_sha256': digest(DEVELOPMENT / 'result_hash_manifest.json'),
        'development_data_gate_sha256': digest(DEVELOPMENT / 'data_gate.json'),
        'development_completion_audit_sha256': digest(AUDIT),
        'preprocessing': 'fixed outcome-blind feature construction; no fitted normalization',
        'records': records,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    publish(CACHE, compressed)
    manifest = {
        'schema': 'ngas-a13-training-cache-manifest-v1',
        'cache_path': str(CACHE.relative_to(ROOT)),
        'cache_sha256': hashlib.sha256(compressed).hexdigest(),
        'uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
        'uncompressed_bytes': len(raw),
        'compressed_bytes': len(compressed),
        'states': len(records),
        'instances': len({row['instance_id'] for row in records}),
        'actions': actions_total,
        'paired_replicates': replicates_total,
        'replicates_per_action': protocol['config']['replicates'],
        'instances_per_fold': {fold: len(values) for fold, values in instances_by_fold.items()},
        'all_state_ids_unique': len({row['state_id'] for row in records}) == len(records),
        'all_action_ids_unique_within_state': all(
            len({row['action_id'] for row in record['actions']}) == len(record['actions'])
            for record in records),
        'source_hashes': {
            str((DEVELOPMENT / 'protocol.json').relative_to(ROOT)): digest(DEVELOPMENT / 'protocol.json'),
            str((DEVELOPMENT / 'result_hash_manifest.json').relative_to(ROOT)): digest(DEVELOPMENT / 'result_hash_manifest.json'),
            str((DEVELOPMENT / 'data_gate.json').relative_to(ROOT)): digest(DEVELOPMENT / 'data_gate.json'),
            str(AUDIT.relative_to(ROOT)): digest(AUDIT),
        },
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != manifest:
            raise FileExistsError('Immutable training cache manifest differs')
    else:
        write_immutable(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
