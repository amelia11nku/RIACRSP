#!/usr/bin/env python3
"""Independently verify completed V2 raw records, cached CRN and saved gate."""
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_ngas.critic.label_revision import aggregate, evaluate_replicate
from rcias_ngas.evaluation.bks import content_hash, write_immutable
from scripts.audit_ngas_starting_state import digest
from scripts.run_ngas_label_pilot_v2 import OUT, V1, action_from_metadata, final_summary, verify_protocol


def unseal(path):
    row = json.loads(path.read_text())
    expected = row.pop('payload_sha256')
    assert content_hash(row) == expected, path
    return row


def main():
    protocol = verify_protocol()
    manifest = json.loads((OUT / 'result_hash_manifest.json').read_text())
    assert set(manifest) == {str(p.relative_to(ROOT)) for p in (OUT / 'states').rglob('*.json')}
    for path, expected in manifest.items():
        assert digest(path) == expected, path
    v1 = json.loads((V1 / 'protocol.json').read_text())
    rows = defaultdict(list)
    calls = reps = replayed = 0
    for directory in sorted((OUT / 'states').iterdir()):
        state = json.loads((V1 / 'states' / directory.name / 'state.json').read_text())
        spec = next(r for r in v1['instances'] if r['instance_id'] == directory.name.split('__')[0])
        instance = load_instance(ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14' / spec['relative_path'])
        current = decode_candidate(instance, Candidate(**{k: tuple(v) for k, v in state['candidate'].items()}))
        assert current.feasible and current.makespan == state['makespan']
        for variant in protocol['config']['variants']:
            base = directory / variant['id']
            cache = [json.loads(p.read_text()) for p in sorted((base / 'fallback').glob('*.json'))]
            assert len(cache) == 9
            for p in sorted((base / 'fallback').glob('*.json')):
                f = unseal(p)
                calls += sum(s['decoder_evals'] for s in f['trajectory']['steps'])
            labels = sorted((base / 'labels').glob('*.json'))
            assert len(labels) == 75
            for index, path in enumerate(labels):
                label = unseal(path)['label']
                action = action_from_metadata(label['action'])
                raw = sorted((base / 'replicates' / action.action_id).glob('*.json'))
                assert len(raw) == 9
                for k, rep_path in enumerate(raw):
                    record = unseal(rep_path)
                    rep = record['replicate']
                    assert rep == label['replicates'][k]
                    assert record['context']['fallback_sha256'] == cache[k]['payload_sha256']
                    assert rep['fallback'] == cache[k]['trajectory']
                    assert rep['candidate']['steps'][0]['action_id'] == action.action_id
                    assert rep['candidate']['steps'][0]['repair'] == action.repair
                    assert rep['candidate']['steps'][0]['destroyed_operations'] == list(action.target.operations)
                    assert rep['advantage'] == (rep['fallback']['best_makespan'] - rep['candidate']['best_makespan']) / current.makespan
                    for c, f in zip(rep['candidate']['steps'], rep['fallback']['steps']):
                        assert c['feasible'] and f['feasible']
                        assert c['decoder_evals'] == f['decoder_evals'] == variant['repair_trials']
                        assert c['neighbor_seed'] == f['neighbor_seed'] and c['acceptance_seed'] == f['acceptance_seed']
                    calls += sum(s['decoder_evals'] for s in rep['candidate']['steps'])
                    reps += 1
                derived = aggregate(label['action'], current.makespan, label['replicates'], variant['repair_trials'], 2)
                assert all(value == label[key] for key, value in derived.items())
                if index == 0:
                    rep = label['replicates'][0]
                    replay = evaluate_replicate(instance, current, action, directory.name, rep['crn_seed'], 2, variant['repair_trials'], rep['fallback'])
                    assert json.loads(json.dumps(replay)) == rep
                    replayed += 1
                rows[variant['id']].append(label)
    saved = json.loads((OUT / 'gate.json').read_text())
    recomputed = final_summary(rows, protocol['config'])
    assert all(saved[key] == value for key, value in recomputed.items())
    assert calls == 123120 and reps == 8100 and saved['decision'] == 'NGAS_A1_LABEL_PILOT_PASS'
    audit = {'schema': 'ngas-v2-completion-audit-v1', 'status': 'PASS',
             'starting_commit': '098a58e9d42b1dcee84ca04eb288258c70bc0ccb',
             'verified_files': len(manifest), 'verified_action_rows': 900,
             'paired_replicates': reps, 'actual_label_decoder_evals': calls,
             'source_replays': 6, 'candidate_trajectory_replays': replayed,
             'gate_reproduced_exactly': True, 'gate_sha256': digest(OUT / 'gate.json'),
             'primary': 'T8_R9', 'decision': saved['decision'], 'historical_evidence_unchanged': True}
    verify_protocol()
    write_immutable(ROOT / 'outputs/ngas_a1/audit/v2_completion.json', audit)
    for name in ('protocol.json', 'gate.json', 'progress.json', 'final_decision.json', 'result_hash_manifest.json'):
        path = OUT / name
        manifest[str(path.relative_to(ROOT))] = digest(path)
    archive_path = ROOT / 'outputs/ngas_a1/archive/label_pilot_v2.tar.gz'
    with archive_path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path in sorted(manifest):
                    data = (ROOT / path).read_bytes()
                    info = tarfile.TarInfo(path)
                    info.size, info.mode = len(data), 0o644
                    archive.addfile(info, io.BytesIO(data))
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest[member.name]
    write_immutable(archive_path.with_suffix('.manifest.json'), {
        'archive_sha256': digest(archive_path), 'files': manifest, 'original_files_removed': False})
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
