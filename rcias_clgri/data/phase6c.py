"""Semantic state serialization and reconstruction contract for Phase 6C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, DecodedCandidate, decode_candidate


@dataclass(frozen=True)
class ReconstructedState:
    instance: Instance
    candidate: Candidate
    decoded: DecodedCandidate
    current_makespan: float
    search_progress: float
    search_stage: str


def candidate_to_json(candidate: Candidate) -> str:
    return json.dumps({
        "operation_order": candidate.operation_order,
        "island_assignment": candidate.island_assignment,
        "w_assignment": candidate.w_assignment,
        "f_assignment": candidate.f_assignment,
    }, separators=(",", ":"))


def candidate_from_json(value: str) -> Candidate:
    raw = json.loads(value)
    return Candidate(
        tuple(raw["operation_order"]), tuple(raw["island_assignment"]),
        tuple(raw["w_assignment"]), tuple(raw["f_assignment"]),
    )


def candidate_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def reconstruct_state_from_instance(instance: Instance, record: Mapping[str, object]) -> ReconstructedState:
    candidate_json = str(record["current_candidate"])
    expected_checksum = record.get("candidate_sha256")
    if expected_checksum is not None and str(expected_checksum) != candidate_sha256(candidate_json):
        raise ValueError(f"candidate checksum mismatch for {record['state_id']}")
    candidate = candidate_from_json(candidate_json)
    decoded = decode_candidate(instance, candidate)
    current_makespan = float(record["current_makespan"])
    if abs(decoded.makespan - current_makespan) > 1e-9:
        raise ValueError(f"state reconstruction mismatch for {record['state_id']}")
    return ReconstructedState(
        instance, candidate, decoded, current_makespan,
        float(record["search_progress"]), str(record["search_stage"]),
    )


def reconstruct_state(record: Mapping[str, object], train_root: Path) -> ReconstructedState:
    """Reconstruct one state without consulting a later trajectory row."""
    instance = load_instance(train_root / str(record["instance_relative_path"]))
    if instance.instance_id != str(record["instance_id"]):
        raise ValueError("state record points to the wrong frozen instance")
    return reconstruct_state_from_instance(instance, record)
