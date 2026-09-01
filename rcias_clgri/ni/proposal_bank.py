"""Outcome-blind live proposal banks using the frozen Phase 6C semantics."""

from __future__ import annotations

import json
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.search.common import DecodedCandidate
from rcias_clgri.search.phase6c import ArmGenerationResult, generate_revised_target_arms


def build_live_proposal_bank(
    instance: Instance,
    current: DecodedCandidate,
    *,
    state_id: str,
    destroy_count: int,
    seed_namespace: int,
) -> tuple[ArmGenerationResult, list[Mapping[str, object]]]:
    """Return Phase 6C proposals in the tensorizer's action-record schema.

    Label fields are neutral placeholders required by the offline tensor container;
    they are never used by the model forward pass.
    """
    generated = generate_revised_target_arms(
        instance,
        current,
        state_id,
        destroy_count,
        seed_namespace,
    )
    records: list[Mapping[str, object]] = []
    for arm in generated.arms:
        records.append({
            "state_id": state_id,
            "target_set_id": arm.target_set_id,
            "destroyed_operation_ids": json.dumps(arm.destroyed_operations),
            "arm_family": arm.arm_family,
            "origin_destroy_operator": arm.origin_destroy_operator,
            "origin_rules": json.dumps(arm.origin_rules),
            "origin_families": json.dumps(arm.origin_families),
            "mean_relative_improvement": 0.0,
            "rank_within_state": 0.0,
            "rank_percentile": 0.0,
            "regret_to_best": 0.0,
            "top1": False,
            "top3": False,
        })
    return generated, records
