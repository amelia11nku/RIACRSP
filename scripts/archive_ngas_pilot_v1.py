#!/usr/bin/env python3
"""Create a compact, byte-preserving archive of V1 evidence; delete nothing."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ngas_starting_state import digest
from rcias_ngas.evaluation.bks import write_immutable


def main():
    base = ROOT / 'outputs/ngas_a1/label_pilot'
    manifest = json.loads((base / 'result_hash_manifest.json').read_text())
    for name in ('protocol.json', 'gate.json', 'progress.json', 'result_hash_manifest.json'):
        path = base / name
        manifest[str(path.relative_to(ROOT))] = digest(path)
    destination = ROOT / 'outputs/ngas_a1/archive/label_pilot_v1.tar.gz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path, expected in sorted(manifest.items()):
                    assert digest(path) == expected
                    data = (ROOT / path).read_bytes()
                    info = tarfile.TarInfo(path)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, 'r:gz') as archive:
        import hashlib
        assert {m.name for m in archive.getmembers()} == set(manifest)
        for member in archive.getmembers():
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest[member.name]
    write_immutable(destination.with_suffix('.manifest.json'), {
        'archive_path': str(destination.relative_to(ROOT)), 'archive_sha256': digest(destination),
        'files': manifest, 'raw_files_removed': False, 'decision': 'NGAS_A1_REVISE_LABELS',
    })
    print(json.dumps({'archived_files': len(manifest), 'archive_bytes': destination.stat().st_size}))


if __name__ == '__main__':
    main()
