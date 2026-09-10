#!/usr/bin/env python3
"""Create a deterministic byte-preserving archive of expanded development evidence."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_ngas.evaluation.bks import write_immutable
from scripts.audit_ngas_starting_state import digest


def main() -> None:
    base = ROOT / 'outputs/ngas_a1/development_v1'
    manifest = json.loads((base / 'result_hash_manifest.json').read_text())
    extra = [base / name for name in ('protocol.json', 'data_gate.json', 'progress.json',
                                      'launch_record.json', 'result_hash_manifest.json')]
    extra.append(ROOT / 'outputs/ngas_a1/audit/development_completion.json')
    for path in extra:
        manifest[str(path.relative_to(ROOT))] = digest(path)
    destination = ROOT / 'outputs/ngas_a1/archive/development_v1.tar.gz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for relative, expected in sorted(manifest.items()):
                    path = ROOT / relative
                    if digest(path) != expected:
                        raise ValueError(f'Development evidence drift: {relative}')
                    data = path.read_bytes()
                    info = tarfile.TarInfo(relative)
                    info.size, info.mode = len(data), 0o644
                    archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, 'r:gz') as archive:
        members = archive.getmembers()
        if {member.name for member in members} != set(manifest):
            raise ValueError('Development archive file set mismatch')
        for member in members:
            value = archive.extractfile(member)
            if value is None or hashlib.sha256(value.read()).hexdigest() != manifest[member.name]:
                raise ValueError(f'Development archive member mismatch: {member.name}')
    output = destination.with_suffix('.manifest.json')
    write_immutable(output, {
        'schema': 'ngas-a13-development-archive-v1',
        'archive_path': str(destination.relative_to(ROOT)),
        'archive_sha256': digest(destination), 'files': manifest,
        'raw_files_removed': False, 'decision': 'READY_FOR_JOINT_CRITIC_TRAINING',
    })
    print(json.dumps({'archived_files': len(manifest),
                      'archive_bytes': destination.stat().st_size,
                      'archive_sha256': digest(destination)}, indent=2))


if __name__ == '__main__':
    main()
