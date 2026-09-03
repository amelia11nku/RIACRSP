"""Phase 6H live-calibration logging and anytime-search primitives."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

import numpy as np


class Phase6HLiveObserver:
    """Retain outcome-blind inputs and post-decoder labels for each live state."""

    def __init__(self, metadata: Mapping[str, object]) -> None:
        self.metadata = dict(metadata)
        self.rows: list[dict[str, object]] = []
        self.interventions = 0
        self.fallbacks = 0
        self.cumulative_overhead_ms = 0.0

    def __call__(self, event: Mapping[str, object]) -> None:
        eligible = bool(event["ni_eligible"])
        intervened = bool(event["ni_intervened"])
        fallback = eligible and not intervened
        self.interventions += int(intervened)
        self.fallbacks += int(fallback)
        timing = dict(event.get("ni_timing_ms") or {})
        features = dict(event.get("ni_state_feature_summary") or {})
        self.cumulative_overhead_ms += float(timing.get("total", 0.0))
        current = event["current_before"]
        candidate = event["candidate"]
        best = event["best_after"]
        self.rows.append({
            **self.metadata,
            "iteration": int(event["iteration"]),
            "elapsed_wall_time": float(event["elapsed_time"]),
            "iteration_runtime": float(event["iteration_runtime"]),
            "decoder_evaluations": int(event["decoder_evaluations"]),
            "repair_decoder_evaluations": int(event["repair_decoder_evaluations"]),
            "current_makespan": float(current.makespan),
            "candidate_makespan": float(candidate.makespan),
            "best_makespan": float(best.makespan),
            "realized_immediate_delta": float(current.makespan - candidate.makespan),
            "realized_immediate_utility": (
                float(current.makespan - candidate.makespan) / float(current.makespan)
            ),
            "realized_positive": bool(candidate.makespan < current.makespan),
            "candidate_feasible": bool(candidate.feasible),
            "accepted": bool(event["accepted"]),
            "new_global_best": bool(event["new_global_best"]),
            "ni_eligible": eligible,
            "ni_intervention": intervened,
            "fallback": fallback,
            "fallback_reason": event.get("ni_fallback_reason"),
            "policy_name": event.get("ni_policy_name"),
            "selected_target_set_id": event.get("ni_target_set_id"),
            "selected_origin_family": event.get("ni_selected_origin_family"),
            "selected_origin_operator": event.get("ni_selected_origin_operator"),
            "selected_origin_rules": json.dumps(event.get("ni_selected_origin_rules", ())),
            "selected_operation_ids": json.dumps(event.get("ni_selected_operation_ids", ())),
            "executed_operation_ids": json.dumps(event["destroyed_operation_ids"]),
            "destroy_size": len(event.get("ni_selected_operation_ids", ())),
            "candidate_bank_size": int(event.get("ni_proposal_count", 0)),
            "requested_bank_size": int(event.get("ni_requested_proposal_count", 0)),
            "duplicate_bank_size": int(event.get("ni_duplicate_proposal_count", 0)),
            "raw_score": event.get("ni_raw_score"),
            "raw_probability": event.get("ni_raw_probability"),
            "raw_utility": event.get("ni_raw_utility"),
            "calibrated_probability": event.get("ni_calibrated_probability"),
            "calibrated_utility": event.get("ni_calibrated_utility"),
            "decision_margin": event.get("ni_decision_margin"),
            "support_in_range": event.get("ni_support_in_range"),
            "support_out_of_range_count": int(
                event.get("ni_support_out_of_range_count", 0)
            ),
            "csg_graph_hash": event.get("ni_graph_hash"),
            "mean_slack_ratio": features.get("mean_slack_ratio"),
            "mean_w_delay_ratio": features.get("mean_w_delay_ratio"),
            "mean_f_delay_ratio": features.get("mean_f_delay_ratio"),
            "mean_island_relative_load": features.get("mean_island_relative_load"),
            "mean_local_reconfiguration_ratio": features.get(
                "mean_local_reconfiguration_ratio"
            ),
            "search_progress": features.get("search_progress"),
            "csg_build_ms": float(timing.get("csg_build", 0.0)),
            "target_bank_ms": float(timing.get("proposal_bank", 0.0)),
            "tensorization_ms": float(timing.get("tensorization_and_transfer", 0.0)),
            "model_inference_ms": float(timing.get("model_inference", 0.0)),
            "action_scoring_ms": float(timing.get("action_scoring", 0.0)),
            "calibration_gate_ms": float(timing.get("calibration_gate", 0.0)),
            "ni_overhead_ms": float(timing.get("total", 0.0)),
            "repair_time_ms": float(event["repair_runtime"]) * 1000.0,
            "repair_excluding_decoder_ms": float(
                event.get("repair_excluding_decoder_runtime", event["repair_runtime"])
            ) * 1000.0,
            "decoder_time_ms": float(event.get("decoder_runtime", 0.0)) * 1000.0,
            "temperature": float(event["temperature_before"]),
            "cumulative_ni_calls": self.interventions,
            "cumulative_fallback_count": self.fallbacks,
            "cumulative_ni_overhead_ms": self.cumulative_overhead_ms,
            "alns_weight_credit": bool(event["alns_weight_credit"]),
        })


def _trace_rows(trace: Iterable[object]) -> list[dict[str, float | int]]:
    rows = []
    for point in trace:
        if isinstance(point, Mapping):
            elapsed = point["elapsed_time"]
            evaluations = point["decoder_evaluations"]
            makespan = point["current_best_makespan"]
        else:
            elapsed = getattr(point, "elapsed_time")
            evaluations = getattr(point, "decoder_evaluations")
            makespan = getattr(point, "current_best_makespan")
        rows.append({
            "elapsed_time": float(elapsed),
            "decoder_evaluations": int(evaluations),
            "current_best_makespan": float(makespan),
        })
    return rows


def validate_incumbent_trace(
    trace: Iterable[object], *, final_best: float | None = None
) -> list[dict[str, float | int]]:
    rows = _trace_rows(trace)
    if not rows:
        raise ValueError("incumbent trace must not be empty")
    elapsed = np.asarray([row["elapsed_time"] for row in rows], dtype=float)
    evaluations = np.asarray([row["decoder_evaluations"] for row in rows], dtype=int)
    makespan = np.asarray([row["current_best_makespan"] for row in rows], dtype=float)
    if np.any(np.diff(elapsed) < 0) or np.any(np.diff(evaluations) < 0):
        raise ValueError("incumbent trace time/evaluations must be monotone")
    if np.any(np.diff(makespan) >= 0):
        raise ValueError("incumbent objective must strictly improve at every event")
    if final_best is not None and not np.isclose(makespan[-1], float(final_best)):
        raise ValueError("last incumbent does not match final best")
    return rows


def sample_incumbent_trace(
    trace: Iterable[object],
    *,
    budget: float,
    fractions: Sequence[float],
) -> list[dict[str, object]]:
    if budget <= 0 or any(not 0 < float(value) <= 1 for value in fractions):
        raise ValueError("budget and normalized fractions must be positive")
    rows = validate_incumbent_trace(trace)
    result = []
    for fraction in fractions:
        checkpoint = float(fraction) * budget
        available = [row for row in rows if float(row["elapsed_time"]) <= checkpoint]
        selected = available[-1] if available else None
        result.append({
            "budget_fraction": float(fraction),
            "elapsed_wall_time": checkpoint,
            "best_makespan": (
                None if selected is None else float(selected["current_best_makespan"])
            ),
            "decoder_evaluations": (
                None if selected is None else int(selected["decoder_evaluations"])
            ),
            "incumbent_available": selected is not None,
        })
    return result


def first_common_target_hit(
    trace: Iterable[object], *, target_makespan: float
) -> dict[str, object]:
    rows = validate_incumbent_trace(trace)
    for row in rows:
        if float(row["current_best_makespan"]) <= target_makespan:
            return {
                "reached": True,
                "right_censored": False,
                "elapsed_wall_time": float(row["elapsed_time"]),
                "decoder_evaluations": int(row["decoder_evaluations"]),
            }
    return {
        "reached": False,
        "right_censored": True,
        "elapsed_wall_time": None,
        "decoder_evaluations": None,
    }


def normalized_gap_auc(
    trace: Iterable[object], *, budget: float, reference_makespan: float
) -> float:
    """Stepwise relative-gap AUC; first incumbent is extended back to time zero."""
    if budget <= 0 or reference_makespan <= 0:
        raise ValueError("budget and reference makespan must be positive")
    rows = validate_incumbent_trace(trace)
    times = [0.0]
    gaps = [
        max(0.0, float(rows[0]["current_best_makespan"]) / reference_makespan - 1.0)
    ]
    for row in rows:
        elapsed = min(float(row["elapsed_time"]), budget)
        gap = max(0.0, float(row["current_best_makespan"]) / reference_makespan - 1.0)
        if elapsed < times[-1]:
            continue
        times.extend((elapsed, elapsed))
        gaps.extend((gaps[-1], gap))
        if elapsed >= budget:
            break
    if times[-1] < budget:
        times.append(budget)
        gaps.append(gaps[-1])
    return float(np.trapz(gaps, times) / budget)
