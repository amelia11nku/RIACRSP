"""Temporal feature semantics for realized CSG relations."""

from __future__ import annotations


DEFAULT_EPSILON = 1e-9
TEMPORAL_FEATURE_NAMES = (
    "source_end_time",
    "target_start_time",
    "temporal_gap",
    "normalized_temporal_gap",
    "binding_indicator",
)


def temporal_features(
    source_end: float,
    target_start: float,
    makespan: float,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """Return an unclipped, leakage-safe temporal gap record."""
    gap = float(target_start) - float(source_end)
    return {
        "source_end_time": float(source_end),
        "target_start_time": float(target_start),
        "temporal_gap": gap,
        "normalized_temporal_gap": gap / max(float(makespan), 1.0),
        "binding_indicator": float(abs(gap) <= epsilon),
    }
