"""Typed representation of an RCIAS-2.0 instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OperationData:
    """Static data for one assembly operation."""

    op_id: str
    product_id: str
    required_config: str
    eligible_islands: tuple[str, ...]
    processing_time: Mapping[str, int]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductData:
    """Operation membership and technological DAG for one product."""

    product_id: str
    operations: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IslandData:
    """Supported discrete capability states of one assembly island."""

    island_id: str
    supported_configs: tuple[str, ...]
    initial_config: str
    coordinate: tuple[int, int] | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """Validated and preprocessed RCIAS-2.0 instance.

    Tuple-keyed mappings make hot decoder lookups constant-time. DAG predecessor,
    successor, and transitive sets are computed exactly once by the loader.
    """

    instance_id: str
    schema: str
    seed: int | None
    products: tuple[str, ...]
    operations: tuple[str, ...]
    islands: tuple[str, ...]
    configurations: tuple[str, ...]
    agvs_w: tuple[str, ...]
    agvs_f: tuple[str, ...]
    nodes: tuple[str, ...]
    product_data: Mapping[str, ProductData]
    operation_data: Mapping[str, OperationData]
    island_data: Mapping[str, IslandData]
    predecessors: Mapping[str, frozenset[str]]
    successors: Mapping[str, frozenset[str]]
    transitive_predecessors: Mapping[str, frozenset[str]]
    transitive_successors: Mapping[str, frozenset[str]]
    product_of: Mapping[str, str]
    processing_time: Mapping[tuple[str, str], int]
    reconfiguration_time: Mapping[tuple[str, str, str], int]
    reconfiguration_cost: Mapping[tuple[str, str, str], float]
    distance: Mapping[tuple[str, str], float]
    w_loaded_time: Mapping[tuple[str, str, str], int]
    w_empty_time: Mapping[tuple[str, str, str], int]
    w_loaded_cost_per_distance: Mapping[str, float]
    w_empty_cost_per_distance: Mapping[str, float]
    f_outbound_time: Mapping[tuple[str, str], int]
    f_return_time: Mapping[tuple[str, str], int]
    f_outbound_cost_per_distance: Mapping[str, float]
    f_return_cost_per_distance: Mapping[str, float]
    objective_parameters: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    @property
    def horizon(self) -> float:
        """A deterministic normalization horizon, not a scheduling constraint."""

        processing = sum(
            max(self.processing_time[(op, island)] for island in self.operation_data[op].eligible_islands)
            for op in self.operations
        )
        reconfiguration = max(self.reconfiguration_time.values(), default=0) * max(1, self.num_operations)
        w_travel = max(self.w_loaded_time.values(), default=0) * max(1, self.num_operations)
        f_travel = max(
            (self.f_outbound_time[key] + self.f_return_time[key] for key in self.f_outbound_time),
            default=0,
        ) * max(1, self.num_operations)
        return float(max(1, processing + reconfiguration + w_travel + f_travel))
