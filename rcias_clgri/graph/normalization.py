"""Central, ID-independent scaling for graph and candidate features."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from rcias_clgri.data.instance import Instance


@dataclass(frozen=True)
class FeatureNormalizer:
    """Deterministic feature scales derived only from numeric instance data."""

    time_scale: float
    distance_scale: float
    cost_scale: float
    load_scale: float
    count_scale: float

    @classmethod
    def from_instance(cls, instance: Instance) -> "FeatureNormalizer":
        distances = [abs(float(value)) for value in instance.distance.values()]
        transition_costs = [abs(float(value)) for value in instance.reconfiguration_cost.values()]
        transport_rates = [
            *map(abs, instance.w_loaded_cost_per_distance.values()),
            *map(abs, instance.w_empty_cost_per_distance.values()),
            *map(abs, instance.f_outbound_cost_per_distance.values()),
            *map(abs, instance.f_return_cost_per_distance.values()),
        ]
        max_distance = max(distances, default=1.0)
        max_cost = max(
            transition_costs + [rate * max_distance for rate in transport_rates],
            default=1.0,
        )
        max_count = max(
            len(instance.operations), len(instance.products), len(instance.islands),
            len(instance.configurations), len(instance.agvs_w), len(instance.agvs_f), 1,
        )
        return cls(
            time_scale=max(1.0, float(instance.horizon)),
            distance_scale=max(1.0, max_distance),
            cost_scale=max(1.0, max_cost),
            load_scale=max(1.0, float(instance.horizon)),
            count_scale=float(max_count),
        )

    def time(self, value: float) -> float:
        return float(value) / self.time_scale

    def distance(self, value: float) -> float:
        return float(value) / self.distance_scale

    def cost(self, value: float) -> float:
        return float(value) / self.cost_scale

    def load(self, value: float) -> float:
        return float(value) / self.load_scale

    def count(self, value: float) -> float:
        return float(value) / self.count_scale

    def to_dict(self) -> dict[str, float]:
        return {
            "time_scale": self.time_scale,
            "distance_scale": self.distance_scale,
            "cost_scale": self.cost_scale,
            "load_scale": self.load_scale,
            "count_scale": self.count_scale,
        }

    @staticmethod
    def assert_finite(features: Mapping[str, float]) -> None:
        invalid = {name: value for name, value in features.items() if not math.isfinite(value)}
        if invalid:
            raise ValueError(f"non-finite graph features: {invalid}")
