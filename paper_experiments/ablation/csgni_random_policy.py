"""P1 policy ablation: frozen gate with uniform full-bank target selection."""

from __future__ import annotations

from dataclasses import replace
import random
import time

from rcias_clgri.data.instance import Instance
from rcias_clgri.ni.live_inference import FrozenLiveInference
from rcias_clgri.ni.live_policy import InterventionDecision
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.search.common import DecodedCandidate
from rcias_clgri.search.counterfactual import stable_seed


class UniformFullBankAtFrozenGate:
    """Keep the frozen gate but replace learned target ranking at interventions.

    The reference inference call is deliberately retained without modification:
    it constructs the graph, the 24-rule/deduplicated proposal bank, scores all
    unique targets, and applies the frozen Phase6H calibration gate.  Only when
    that gate intervenes do we regenerate the same deterministic proposal bank
    and select one unique target uniformly using an isolated RNG namespace.

    The second bank construction is included in run wall time.  It is preferable
    to duplicating or editing frozen inference internals for this small ablation,
    and its overhead is reported explicitly in the live log.
    """

    policy_name = "P1_UNIFORM_FULL_BANK_AT_FROZEN_GATE"

    def __init__(
        self,
        reference: FrozenLiveInference,
        *,
        selection_seed_namespace: int = 671201,
    ) -> None:
        self.reference = reference
        self.selection_seed_namespace = int(selection_seed_namespace)
        self.device = reference.device
        self.checkpoint_sha256 = reference.checkpoint_sha256
        self.deployment_artifact_sha256 = reference.deployment_artifact_sha256

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
        gated = self.reference.decide(
            instance,
            current,
            state_id=state_id,
            destroy_count=destroy_count,
            search_progress=search_progress,
            search_stage=search_stage,
        )
        if not gated.intervene:
            return replace(gated, policy_name=self.policy_name)

        selection_started = time.perf_counter()
        generated, _ = build_live_proposal_bank(
            instance,
            current,
            state_id=state_id,
            destroy_count=destroy_count,
            seed_namespace=self.reference.proposal_seed_namespace,
        )
        if generated.requested_arm_count != 24:
            raise RuntimeError(
                f"full production bank changed: requested={generated.requested_arm_count}"
            )
        if generated.unique_arm_count != gated.proposal_count:
            raise RuntimeError("regenerated unique bank differs from frozen inference bank")
        if not generated.arms:
            raise RuntimeError("full production bank unexpectedly contains no unique targets")
        rng = random.Random(
            stable_seed(
                state_id,
                "uniform_full_bank",
                namespace=self.selection_seed_namespace,
            )
        )
        arm = generated.arms[rng.randrange(generated.unique_arm_count)]
        selection_ms = (time.perf_counter() - selection_started) * 1000.0
        timings = dict(gated.timings_ms or {})
        timings["uniform_selection_bank_rebuild"] = selection_ms
        timings["total"] = float(timings.get("total", 0.0)) + selection_ms
        return replace(
            gated,
            selected_target_set_id=arm.target_set_id,
            destroyed_operations=arm.destroyed_operations,
            selected_origin_family=arm.arm_family,
            selected_origin_operator=arm.origin_destroy_operator,
            selected_origin_rules=arm.origin_rules,
            timings_ms=timings,
            policy_name=self.policy_name,
        )
