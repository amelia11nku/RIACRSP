"""Shared deterministic helpers for the two RCIAS-2.0 generators."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .loader import load_instance_dict

WAREHOUSE = "WH"
SCHEMA_VERSION = "RCIAS-2.0"

DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "generator_version": "2.1.0",
    "configuration_generation": {"minimum": 3, "maximum": 6, "island_offset": 1},
    "dag_generation": {"optional_edge_probability": 0.18, "two_operation_precedence": "none"},
    "reconfiguration": {
        "time_base": 1,
        "time_per_index_separation": 1,
        "time_jitter_min": 0,
        "time_jitter_max": 2,
        "cost_base": 1.5,
        "cost_per_index_separation": 0.75,
        "cost_jitter_min": 0.0,
        "cost_jitter_max": 1.0,
        "cost_decimals": 2,
    },
    "layout": {"coordinate_min": 3, "coordinate_max": 24, "metric": "manhattan"},
    "fleet": {"num_w_agvs": 2, "num_f_agvs": 2},
    "travel": {
        "w_loaded_speed_min": 1.0,
        "w_loaded_speed_max": 1.6,
        "w_empty_speed_min": 1.3,
        "w_empty_speed_max": 2.0,
        "f_outbound_speed_min": 1.1,
        "f_outbound_speed_max": 1.8,
        "f_return_speed_min": 1.4,
        "f_return_speed_max": 2.1,
        "rounding": "ceil_positive",
    },
    "cost": {
        "w_loaded_min": 0.28,
        "w_loaded_max": 0.48,
        "w_empty_min": 0.16,
        "w_empty_max": 0.32,
        "f_outbound_min": 0.18,
        "f_outbound_max": 0.36,
        "f_return_min": 0.14,
        "f_return_max": 0.30,
        "decimals": 3,
    },
    "seed_policy": "sha256(RCIAS-2.0::family::instance_name), first 32 bits big-endian",
    "json_serialization": {"sort_keys": True, "indent": 2, "ensure_ascii": False, "newline": True},
}


def operation_id(product_index: int, operation_index: int, operations_in_product: int) -> str:
    """Return compact IDs when unambiguous and separated IDs otherwise."""

    if product_index < 10 and operations_in_product < 10:
        return f"o{product_index}{operation_index}"
    return f"o{product_index}_{operation_index}"


def manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def ceil_time(distance: float, speed: float) -> int:
    return 0 if distance == 0 else max(1, int(math.ceil(distance / speed)))


def unique_coordinates(
    islands: Sequence[str],
    rng: random.Random,
    lower: int = 3,
    upper: int = 24,
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, tuple[int, int]]:
    if generation_config is not None:
        layout = generation_config["layout"]
        lower = int(layout["coordinate_min"])
        upper = int(layout["coordinate_max"])
    coordinates: dict[str, tuple[int, int]] = {WAREHOUSE: (0, 0)}
    occupied = {(0, 0)}
    for island_id in islands:
        while True:
            coordinate = (rng.randint(lower, upper), rng.randint(lower, upper))
            if coordinate not in occupied:
                occupied.add(coordinate)
                coordinates[island_id] = coordinate
                break
    return coordinates


def build_reconfiguration(
    islands: Mapping[str, Mapping[str, Any]],
    rng: random.Random,
    generation_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete island-specific transition matrices with zero diagonals."""

    times: dict[str, Any] = {}
    costs: dict[str, Any] = {}
    rule = generation_config["reconfiguration"]
    for island_id, island in islands.items():
        supported = island["supported_configurations"]
        times[island_id] = {}
        costs[island_id] = {}
        for source in supported:
            times[island_id][source] = {}
            costs[island_id][source] = {}
            source_index = int(source[1:])
            for target in supported:
                if source == target:
                    duration, cost = 0, 0.0
                else:
                    target_index = int(target[1:])
                    separation = abs(source_index - target_index)
                    duration = (
                        int(rule["time_base"])
                        + int(rule["time_per_index_separation"]) * separation
                        + rng.randint(int(rule["time_jitter_min"]), int(rule["time_jitter_max"]))
                    )
                    cost = round(
                        float(rule["cost_base"])
                        + float(rule["cost_per_index_separation"]) * separation
                        + rng.uniform(float(rule["cost_jitter_min"]), float(rule["cost_jitter_max"])),
                        int(rule["cost_decimals"]),
                    )
                times[island_id][source][target] = duration
                costs[island_id][source][target] = cost
    return times, costs


def build_logistics(
    islands: Sequence[str],
    agvs_w: Sequence[str],
    agvs_f: Sequence[str],
    coordinates: Mapping[str, tuple[int, int]],
    rng: random.Random,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete deterministic travel and cost parameters."""

    nodes = [WAREHOUSE, *islands]
    distance = {
        source: {target: manhattan(coordinates[source], coordinates[target]) for target in nodes}
        for source in nodes
    }
    loaded_time: dict[str, Any] = {}
    empty_time: dict[str, Any] = {}
    loaded_cost: dict[str, float] = {}
    empty_cost: dict[str, float] = {}
    travel_rule = generation_config["travel"]
    cost_rule = generation_config["cost"]
    for agv_id in agvs_w:
        loaded_speed = rng.uniform(float(travel_rule["w_loaded_speed_min"]), float(travel_rule["w_loaded_speed_max"]))
        empty_speed = rng.uniform(float(travel_rule["w_empty_speed_min"]), float(travel_rule["w_empty_speed_max"]))
        loaded_time[agv_id] = {
            source: {target: ceil_time(distance[source][target], loaded_speed) for target in nodes}
            for source in nodes
        }
        empty_time[agv_id] = {
            source: {target: ceil_time(distance[source][target], empty_speed) for target in nodes}
            for source in nodes
        }
        loaded_cost[agv_id] = round(rng.uniform(float(cost_rule["w_loaded_min"]), float(cost_rule["w_loaded_max"])), int(cost_rule["decimals"]))
        empty_cost[agv_id] = round(rng.uniform(float(cost_rule["w_empty_min"]), float(cost_rule["w_empty_max"])), int(cost_rule["decimals"]))

    outbound_time: dict[str, Any] = {}
    return_time: dict[str, Any] = {}
    outbound_cost: dict[str, float] = {}
    return_cost: dict[str, float] = {}
    for agv_id in agvs_f:
        outbound_speed = rng.uniform(float(travel_rule["f_outbound_speed_min"]), float(travel_rule["f_outbound_speed_max"]))
        return_speed = rng.uniform(float(travel_rule["f_return_speed_min"]), float(travel_rule["f_return_speed_max"]))
        outbound_time[agv_id] = {
            island_id: ceil_time(distance[WAREHOUSE][island_id], outbound_speed)
            for island_id in islands
        }
        return_time[agv_id] = {
            island_id: ceil_time(distance[island_id][WAREHOUSE], return_speed)
            for island_id in islands
        }
        outbound_cost[agv_id] = round(rng.uniform(float(cost_rule["f_outbound_min"]), float(cost_rule["f_outbound_max"])), int(cost_rule["decimals"]))
        return_cost[agv_id] = round(rng.uniform(float(cost_rule["f_return_min"]), float(cost_rule["f_return_max"])), int(cost_rule["decimals"]))

    return {
        "warehouse": WAREHOUSE,
        "distance": distance,
        "W": {
            "loaded_time": loaded_time,
            "empty_time": empty_time,
            "loaded_cost_per_distance": loaded_cost,
            "empty_cost_per_distance": empty_cost,
        },
        "F": {
            "outbound_time": outbound_time,
            "return_time": return_time,
            "outbound_cost_per_distance": outbound_cost,
            "return_cost_per_distance": return_cost,
        },
    }


def finalize_instance(
    *,
    instance_id: str,
    generator: str,
    seed: int,
    products: Mapping[str, Mapping[str, Any]],
    operations: Mapping[str, Mapping[str, Any]],
    islands: Mapping[str, Mapping[str, Any]],
    configurations: Sequence[str],
    agvs_w: Sequence[str],
    agvs_f: Sequence[str],
    coordinates: Mapping[str, tuple[int, int]],
    rng: random.Random,
    generation_config: Mapping[str, Any] | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble, strictly validate, and return one RCIAS-2.0 JSON object."""

    island_records = {
        island_id: {
            **record,
            "coordinate": list(coordinates[island_id]),
        }
        for island_id, record in islands.items()
    }
    config = DEFAULT_GENERATION_CONFIG if generation_config is None else generation_config
    reconfiguration_time, reconfiguration_cost = build_reconfiguration(island_records, rng, config)
    island_ids = list(island_records)
    metadata = {
        "schema": SCHEMA_VERSION,
        "instance_id": instance_id,
        "generator": generator,
        "seed": seed,
        "static_problem": True,
    }
    if extra_meta:
        metadata.update(extra_meta)
    raw = {
        "meta": metadata,
        "sets": {
            "products": list(products),
            "operations": list(operations),
            "islands": island_ids,
            "configurations": list(configurations),
            "agvs_w": list(agvs_w),
            "agvs_f": list(agvs_f),
            "nodes": [WAREHOUSE, *island_ids],
        },
        "products": dict(products),
        "operations": dict(operations),
        "islands": island_records,
        "reconfiguration": {
            "time": reconfiguration_time,
            "cost": reconfiguration_cost,
        },
        "logistics": build_logistics(island_ids, agvs_w, agvs_f, coordinates, rng, config),
        "objective_parameters": {
            "makespan_weight": 1.0,
            "cost_weight": 0.05,
        },
    }
    load_instance_dict(raw)
    return raw


def write_json(instance: Mapping[str, Any], output: str | Path) -> None:
    """Write a reproducible UTF-8 JSON instance."""

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(deterministic_json_text(instance).encode("utf-8"))


def deterministic_json_text(instance: Mapping[str, Any]) -> str:
    """Serialize JSON with the frozen byte-level canonical settings."""

    return json.dumps(instance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
