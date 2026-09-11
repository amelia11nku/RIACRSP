"""Fail-closed dataset roles, content identity, and exposure logging."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


ROLES = frozenset({
    'TRAIN',
    'VALIDATION',
    'DEVELOPMENT_EXPOSED',
    'AUDIT_ONLY',
    'LOCKED_FINAL_EVAL',
    'LOCKED_GENERALIZATION_EVAL',
    'EXTERNAL_BASELINE_ONLY',
    'UNASSIGNED',
})
LOCKED_ROLES = frozenset({'LOCKED_FINAL_EVAL', 'LOCKED_GENERALIZATION_EVAL'})
HISTORICALLY_EXPOSED_ROLES = frozenset({
    'TRAIN', 'VALIDATION', 'DEVELOPMENT_EXPOSED', 'AUDIT_ONLY',
})
PURPOSE_ROLES = {
    'training': frozenset({'TRAIN'}),
    'model_selection': frozenset({'VALIDATION'}),
    'validation': frozenset({'VALIDATION'}),
    'diagnostic': frozenset({'TRAIN', 'VALIDATION', 'DEVELOPMENT_EXPOSED', 'AUDIT_ONLY'}),
    'regression': frozenset({'TRAIN', 'VALIDATION', 'DEVELOPMENT_EXPOSED', 'AUDIT_ONLY'}),
    'independent_test': LOCKED_ROLES,
    'final_evaluation': frozenset({'LOCKED_FINAL_EVAL'}),
    'generalization_evaluation': frozenset({'LOCKED_GENERALIZATION_EVAL'}),
    'metadata_integrity': ROLES,
}


class AccessDenied(RuntimeError):
    """Raised before a prohibited dataset access can begin."""


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    ).encode('utf-8')


class DatasetRegistry:
    """Validated, immutable-in-memory view of the authoritative role registry."""

    def __init__(self, path: Path, root: Path | None = None):
        self.path = Path(path)
        self.root = Path(root) if root is not None else self.path.resolve().parents[1]
        self.payload = json.loads(self.path.read_text())
        if self.payload.get('schema') != 'ngas-dataset-role-registry-v1':
            raise ValueError('Unsupported dataset-role registry schema')
        if set(self.payload.get('roles', ())) != ROLES:
            raise ValueError('Dataset-role registry has an incomplete role vocabulary')
        if not isinstance(self.payload.get('revision'), int) or self.payload['revision'] < 1:
            raise ValueError('Dataset-role registry revision must be a positive integer')
        instances = self.payload.get('instances')
        if not isinstance(instances, list) or not instances:
            raise ValueError('Dataset-role registry contains no instances')
        self.by_id: dict[str, dict] = {}
        self.by_hash: dict[str, dict] = {}
        for row in instances:
            instance_id = row.get('instance_id')
            content_hash = row.get('content_sha256')
            role = row.get('role')
            if not instance_id or not content_hash or role not in ROLES:
                raise ValueError('Invalid registry instance row')
            if instance_id in self.by_id:
                raise ValueError(f'Duplicate registry instance ID: {instance_id}')
            if content_hash in self.by_hash:
                other = self.by_hash[content_hash]['instance_id']
                raise ValueError(
                    f'Duplicate instance content hashes in registry: {other}, {instance_id}')
            history = row.get('historical_roles', [role])
            if not history or any(item not in ROLES for item in history):
                raise ValueError(f'Invalid role history for {instance_id}')
            self.by_id[instance_id] = row
            self.by_hash[content_hash] = row

    def resolve(self, *, instance_id: str | None = None,
                content_sha256: str | None = None) -> dict:
        by_id = self.by_id.get(instance_id) if instance_id is not None else None
        by_hash = self.by_hash.get(content_sha256) if content_sha256 is not None else None
        if by_id is None and by_hash is None:
            raise AccessDenied('Instance/content hash is absent from the dataset-role registry')
        if by_id is not None and by_hash is not None and by_id is not by_hash:
            raise AccessDenied('Instance ID and content hash resolve to different registry rows')
        row = by_id or by_hash
        assert row is not None
        if content_sha256 is not None and row['content_sha256'] != content_sha256:
            raise AccessDenied('Manifest content hash does not match the registered instance')
        return row

    def authorize(self, row: Mapping[str, object], purpose: str, *,
                  allow_locked_evaluation: bool = False) -> dict:
        if purpose not in PURPOSE_ROLES:
            raise ValueError(f'Unknown dataset access purpose: {purpose}')
        instance_id = str(row.get('instance_id', '')) or None
        content_hash = str(row.get('content_sha256') or row.get('sha256') or '') or None
        registered = self.resolve(instance_id=instance_id, content_sha256=content_hash)
        declared_role = row.get('dataset_role') or row.get('role')
        if declared_role is None:
            raise AccessDenied(f'Manifest omits dataset role for {registered["instance_id"]}')
        if declared_role != registered['role']:
            raise AccessDenied(
                f'Manifest role {declared_role} disagrees with registered role '
                f'{registered["role"]} for {registered["instance_id"]}')
        role = registered['role']
        if purpose == 'independent_test':
            history = set(registered.get('historical_roles', (role,)))
            if history & HISTORICALLY_EXPOSED_ROLES:
                raise AccessDenied(
                    f'{registered["instance_id"]} has development exposure and is not independent')
        if role not in PURPOSE_ROLES[purpose]:
            raise AccessDenied(f'{role} is forbidden for purpose {purpose}')
        if role in LOCKED_ROLES and purpose != 'metadata_integrity' and not allow_locked_evaluation:
            raise AccessDenied(f'{registered["instance_id"]} remains locked for {purpose}')
        return registered

    def verify_file(self, row: Mapping[str, object]) -> dict:
        registered = self.resolve(
            instance_id=str(row.get('instance_id', '')) or None,
            content_sha256=str(row.get('content_sha256') or row.get('sha256') or '') or None,
        )
        path = self.root / registered['relative_path']
        if not path.is_file() or sha256_file(path) != registered['content_sha256']:
            raise AccessDenied(f'Instance bytes differ from registry: {registered["instance_id"]}')
        return registered


def validate_manifest(rows: Iterable[Mapping[str, object]], registry: DatasetRegistry,
                      purpose: str, *, verify_files: bool = True,
                      allow_locked_evaluation: bool = False) -> list[dict]:
    """Validate a complete manifest before any formal work is started."""
    validated = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for row in rows:
        registered = registry.authorize(
            row, purpose, allow_locked_evaluation=allow_locked_evaluation)
        if verify_files:
            registry.verify_file(row)
        instance_id = registered['instance_id']
        content_hash = registered['content_sha256']
        if instance_id in seen_ids or content_hash in seen_hashes:
            raise AccessDenied(f'Duplicate instance identity in manifest: {instance_id}')
        seen_ids.add(instance_id)
        seen_hashes.add(content_hash)
        validated.append(registered)
    if not validated:
        raise AccessDenied('Formal manifest is empty')
    return validated


def validate_training_records(records: Iterable[Mapping[str, object]],
                              registry: DatasetRegistry) -> list[dict]:
    """Reject audit-only and non-TRAIN records for the future C1-v2 loader."""
    validated = []
    for record in records:
        if record.get('audit_only') is not False:
            raise AccessDenied('C1-v2 training records must set audit_only=false')
        if record.get('dataset_role') != 'TRAIN':
            raise AccessDenied('C1-v2 training records must have dataset_role=TRAIN')
        registered = registry.authorize(record, 'training')
        if registered['role'] != 'TRAIN':
            raise AccessDenied('Only governed TRAIN instances may enter C1-v2 training')
        validated.append(dict(record))
    if not validated:
        raise AccessDenied('C1-v2 training records are empty')
    return validated


def append_exposure(path: Path, event: Mapping[str, object]) -> dict:
    """Append one hash-chained exposure event and fsync it to durable storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists() and path.stat().st_size:
        verify_exposure_ledger(path)
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    previous = json.loads(line)['entry_sha256']
    body = {
        'schema': 'ngas-dataset-exposure-ledger-entry-v1',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'previous_entry_sha256': previous,
        **dict(event),
    }
    body['entry_sha256'] = hashlib.sha256(_canonical(body)).hexdigest()
    with path.open('a') as stream:
        stream.write(json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return body


def verify_exposure_ledger(path: Path) -> list[dict]:
    previous = None
    entries = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Blank exposure ledger line {number}')
            entry = json.loads(line)
            claimed = entry.pop('entry_sha256', None)
            actual = hashlib.sha256(_canonical(entry)).hexdigest()
            if claimed != actual:
                raise ValueError(f'Exposure ledger hash mismatch at line {number}')
            if entry.get('previous_entry_sha256') != previous:
                raise ValueError(f'Exposure ledger chain mismatch at line {number}')
            entry['entry_sha256'] = claimed
            entries.append(entry)
            previous = claimed
    return entries
