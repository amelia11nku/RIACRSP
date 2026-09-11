"""Atomic, fail-fast ownership for a formal experiment namespace."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


class LockHeldError(RuntimeError):
    """Raised when another process already owns the formal namespace."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_start_ticks(pid: int) -> int:
    fields_after_command = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return int(fields_after_command[19])


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('x') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ExperimentSessionLock:
    """Own one experiment namespace until explicit release.

    Lock creation uses ``O_EXCL`` and therefore has no check-then-create race.
    Heartbeats use a separate file and thread; they never touch solver/runtime
    objects. Existing locks always fail fast and are never removed by age.
    """

    def __init__(self, lock_path: Path, session_dir: Path, *, experiment: str,
                 stage: str, implementation_commit: str, command: list[str],
                 formal_owner_id: str, heartbeat_interval_seconds: float = 30.) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError('heartbeat interval must be positive')
        self.lock_path = Path(lock_path)
        self.session_dir = Path(session_dir)
        self.experiment = experiment
        self.stage = stage
        self.implementation_commit = implementation_commit
        self.command = list(command)
        self.formal_owner_id = formal_owner_id
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.session_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.process_start_ticks = _process_start_ticks(self.pid)
        self.event_log_path = self.session_dir / f'{self.session_id}.jsonl'
        self.heartbeat_path = self.session_dir / f'{self.session_id}.heartbeat.json'
        self._stop = threading.Event()
        self._event_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._acquired = False
        self._event_sequence = 0

    def _metadata(self) -> dict:
        return {
            'schema': 'ngas-formal-experiment-lock-v1',
            'experiment': self.experiment,
            'stage': self.stage,
            'formal_owner_id': self.formal_owner_id,
            'session_id': self.session_id,
            'pid': self.pid,
            'hostname': self.hostname,
            'process_start_ticks': self.process_start_ticks,
            'implementation_commit': self.implementation_commit,
            'command': self.command,
            'acquired_at_utc': _utc_now(),
            'event_log_path': str(self.event_log_path),
            'heartbeat_path': str(self.heartbeat_path),
        }

    def _append_event(self, event: str, **fields) -> None:
        with self._event_lock:
            self._event_sequence += 1
            payload = {
                'schema': 'ngas-formal-session-event-v1',
                'sequence': self._event_sequence,
                'event': event,
                'at_utc': _utc_now(),
                'monotonic_seconds': time.monotonic(),
                'formal_owner_id': self.formal_owner_id,
                'session_id': self.session_id,
                'pid': self.pid,
                'hostname': self.hostname,
                **fields,
            }
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(payload, sort_keys=True) + '\n'
            descriptor = os.open(
                self.event_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(descriptor, encoded.encode())
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def heartbeat(self, *, source: str = 'periodic') -> None:
        if not self._acquired:
            raise RuntimeError('cannot heartbeat an unowned experiment lock')
        payload = {
            'schema': 'ngas-formal-session-heartbeat-v1',
            'at_utc': _utc_now(),
            'monotonic_seconds': time.monotonic(),
            'source': source,
            'formal_owner_id': self.formal_owner_id,
            'session_id': self.session_id,
            'pid': self.pid,
            'hostname': self.hostname,
            'process_start_ticks': self.process_start_ticks,
        }
        _atomic_json(self.heartbeat_path, payload)
        self._append_event('heartbeat', source=source)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            try:
                self.heartbeat()
            except Exception as error:  # preserved in provenance; solver keeps ownership
                self._append_event('heartbeat_error', error=repr(error))

    def acquire(self) -> 'ExperimentSessionLock':
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        encoded = (json.dumps(metadata, indent=2, sort_keys=True) + '\n').encode()
        try:
            descriptor = os.open(
                self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise LockHeldError(
                f'formal experiment lock already exists: {self.lock_path}') from error
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._acquired = True
        self._append_event(
            'lock_acquired', lock_path=str(self.lock_path),
            process_start_ticks=self.process_start_ticks,
            implementation_commit=self.implementation_commit,
            command=self.command, experiment=self.experiment, stage=self.stage)
        self.heartbeat(source='acquisition')
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f'{self.stage}-heartbeat-{self.session_id[:8]}', daemon=True)
        self._thread.start()
        return self

    def record(self, event: str, **fields) -> None:
        if not self._acquired:
            raise RuntimeError('cannot record against an unowned experiment lock')
        self._append_event(event, **fields)

    def release(self, *, status: str = 'SUCCESS') -> None:
        if not self._acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1., self.heartbeat_interval_seconds + 1.))
        current = json.loads(self.lock_path.read_text())
        if (current.get('session_id') != self.session_id
                or current.get('process_start_ticks') != self.process_start_ticks):
            raise RuntimeError('formal lock ownership changed before release')
        self._append_event('lock_released', status=status, lock_path=str(self.lock_path))
        self.lock_path.unlink()
        self._acquired = False

    def __enter__(self) -> 'ExperimentSessionLock':
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release(status='FAILED' if exc_type else 'SUCCESS')


def recover_stale_lock(lock_path: Path, recovery_dir: Path) -> Path:
    """Quarantine a proven-dead same-host owner; never infer staleness by age."""
    lock_path = Path(lock_path)
    original = lock_path.read_bytes()
    try:
        metadata = json.loads(original)
    except (TypeError, ValueError) as error:
        raise RuntimeError('lock metadata is ambiguous; manual audit required') from error
    if metadata.get('hostname') != socket.gethostname():
        raise RuntimeError('cannot prove a lock owner on another host is dead')
    pid = int(metadata['pid'])
    expected_ticks = int(metadata['process_start_ticks'])
    try:
        actual_ticks = _process_start_ticks(pid)
    except (FileNotFoundError, ProcessLookupError):
        actual_ticks = None
    if actual_ticks == expected_ticks:
        raise LockHeldError(f'lock owner PID {pid} is still alive')
    if lock_path.read_bytes() != original:
        raise RuntimeError('lock metadata changed during stale-owner verification')
    recovery_dir = Path(recovery_dir)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(original).hexdigest()[:16]
    destination = recovery_dir / (
        f"{metadata.get('session_id', 'unknown')}.{digest}.stale-lock.json")
    lock_path.replace(destination)
    return destination
