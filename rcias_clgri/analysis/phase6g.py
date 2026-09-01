"""Efficient per-iteration records for Phase 6G live search."""

from __future__ import annotations

import json
from typing import Mapping


class Phase6GLiveObserver:
    def __init__(self, metadata: Mapping[str, object]) -> None:
        self.metadata = dict(metadata)
        self.rows: list[dict[str, object]] = []
        self.ni_calls = 0
        self.fallbacks = 0
        self.cumulative_overhead_ms = 0.0

    def __call__(self, event: Mapping[str, object]) -> None:
        eligible = bool(event["ni_eligible"])
        intervened = bool(event["ni_intervened"])
        fallback = eligible and not intervened
        self.ni_calls += int(intervened)
        self.fallbacks += int(fallback)
        timing = dict(event.get("ni_timing_ms") or {})
        state_features = dict(event.get("ni_state_feature_summary") or {})
        self.cumulative_overhead_ms += float(timing.get("total", 0.0))
        current = event["current_before"]
        candidate = event["candidate"]
        best = event["best_after"]
        self.rows.append({
            **self.metadata,
            "iteration": int(event["iteration"]),
            "elapsed_time": float(event["elapsed_time"]),
            "current_makespan": float(current.makespan),
            "best_makespan": float(best.makespan),
            "ni_eligible": eligible,
            "ni_intervention": intervened,
            "fallback": fallback,
            "fallback_reason": event.get("ni_fallback_reason"),
            "candidate_bank_size": int(event.get("ni_proposal_count", 0)),
            "requested_bank_size": int(event.get("ni_requested_proposal_count", 0)),
            "duplicate_bank_size": int(event.get("ni_duplicate_proposal_count", 0)),
            "predicted_probability": event.get("ni_calibrated_probability"),
            "predicted_utility": event.get("ni_calibrated_utility"),
            "calibrated_confidence": event.get("ni_calibrated_probability"),
            "score_margin": event.get("ni_decision_margin"),
            "selected_target_set_id": event.get("ni_target_set_id"),
            "selected_origin_family": event.get("ni_selected_origin_family"),
            "selected_origin_operator": event.get("ni_selected_origin_operator"),
            "selected_origin_rules": json.dumps(event.get("ni_selected_origin_rules", ())),
            "selected_operation_ids": json.dumps(event["destroyed_operation_ids"]),
            "csg_graph_hash": event.get("ni_graph_hash"),
            "mean_slack_ratio": state_features.get("mean_slack_ratio"),
            "mean_w_delay_ratio": state_features.get("mean_w_delay_ratio"),
            "mean_f_delay_ratio": state_features.get("mean_f_delay_ratio"),
            "mean_island_relative_load": state_features.get("mean_island_relative_load"),
            "mean_local_reconfiguration_ratio": state_features.get(
                "mean_local_reconfiguration_ratio"
            ),
            "search_progress": state_features.get("search_progress"),
            "csg_build_ms": float(timing.get("csg_build", 0.0)),
            "target_bank_ms": float(timing.get("proposal_bank", 0.0)),
            "tensorization_ms": float(timing.get("tensorization_and_transfer", 0.0)),
            "model_inference_ms": float(timing.get("model_inference", 0.0)),
            "action_scoring_ms": float(timing.get("action_scoring", 0.0)),
            "calibration_gate_ms": float(timing.get("calibration_gate", 0.0)),
            "ni_overhead_ms": float(timing.get("total", 0.0)),
            "repair_time_ms": float(event["repair_runtime"]) * 1000.0,
            "decoder_evaluations": int(event["decoder_evaluations"]),
            "repair_decoder_evaluations": int(event["repair_decoder_evaluations"]),
            "candidate_makespan": float(candidate.makespan),
            "immediate_relative_utility": (
                float(current.makespan - candidate.makespan) / float(current.makespan)
            ),
            "accepted": bool(event["accepted"]),
            "new_global_best": bool(event["new_global_best"]),
            "temperature": float(event["temperature_before"]),
            "cumulative_ni_calls": self.ni_calls,
            "cumulative_fallback_count": self.fallbacks,
            "cumulative_ni_overhead_ms": self.cumulative_overhead_ms,
            "alns_weight_credit": bool(event["alns_weight_credit"]),
            "rng_baseline_namespace": int(event["rng_baseline_namespace"]),
            "rng_proposal_namespace": int(event["rng_proposal_namespace"]),
            "rng_ni_repair_namespace": int(event["rng_ni_repair_namespace"]),
            "rng_acceptance_namespace": int(event["rng_acceptance_namespace"]),
            "rng_diagnostics_namespace": int(event["rng_diagnostics_namespace"]),
        })
