"""Interfaces and immutable decisions for live neural intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from rcias_clgri.data.instance import Instance
from rcias_clgri.search.common import DecodedCandidate


@dataclass(frozen=True)
class InterventionDecision:
    intervene: bool
    state_id: str
    selected_target_set_id: str | None
    destroyed_operations: tuple[str, ...]
    calibrated_probability: float | None
    calibrated_utility: float | None
    decision_margin: float | None
    fallback_reason: str | None
    proposal_count: int
    requested_proposal_count: int
    duplicate_proposal_count: int
    selected_origin_family: str | None = None
    selected_origin_operator: str | None = None
    selected_origin_rules: tuple[str, ...] = ()
    graph_hash: str | None = None
    timings_ms: Mapping[str, float] | None = None
    state_feature_summary: Mapping[str, float] | None = None


class LiveInterventionPolicy(Protocol):
    def decide(
        self,
        instance: Instance,
        current: DecodedCandidate,
        *,
        state_id: str,
        destroy_count: int,
        search_progress: float,
        search_stage: str,
    ) -> InterventionDecision: ...


class AlwaysFallbackPolicy:
    """Test/safety policy that performs no neural work."""

    def decide(
        self,
        instance: Instance,
        current: DecodedCandidate,
        *,
        state_id: str,
        destroy_count: int,
        search_progress: float,
        search_stage: str,
    ) -> InterventionDecision:
        return InterventionDecision(
            False, state_id, None, (), None, None, None,
            "POLICY_ABSTAIN", 0, 0, 0,
        )
