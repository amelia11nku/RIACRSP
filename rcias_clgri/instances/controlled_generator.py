"""Native deterministic generator for the controlled RCIAS-CB1 benchmark."""
from __future__ import annotations

import copy
from collections import Counter
import math
import random
from statistics import mean
from typing import Any, Mapping

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics
from rcias_clgri.data.generation import DEFAULT_GENERATION_CONFIG, finalize_instance, operation_id
from rcias_clgri.data.loader import load_instance_dict


def _operation_counts(products: int, target: int, rng: random.Random) -> list[int]:
    counts = [5] * products
    while sum(counts) < target:
        choices = [index for index, value in enumerate(counts) if value < 10]
        counts[rng.choice(choices)] += 1
    rng.shuffle(counts)
    return counts


def _dag(op_ids: list[str], rng: random.Random) -> list[list[str]]:
    """Connected branching DAG with guaranteed incomparable operations."""
    n = len(op_ids)
    edges = {(op_ids[0], op_ids[1]), (op_ids[0], op_ids[2])}
    for index in range(3, n - 1):
        parent = rng.randrange(0, index)
        edges.add((op_ids[parent], op_ids[index]))
        if rng.random() < 0.30:
            edges.add((op_ids[rng.randrange(0, index)], op_ids[index]))
    for index in range(1, n - 1):
        if not any(left == op_ids[index] for left, _ in edges):
            edges.add((op_ids[index], op_ids[-1]))
    edges.add((op_ids[1], op_ids[-1]))
    edges.add((op_ids[2], op_ids[-1]))
    return [list(edge) for edge in sorted(edges)]


def _capabilities(islands, configurations, center, rng):
    cells = {(island, config) for config in configurations for island in rng.sample(islands, 2)}
    for island in islands:
        if not any(left == island for left, _ in cells):
            cells.add((island, rng.choice(configurations)))
    target = round(center * len(islands) * len(configurations))
    candidates = [(island, config) for island in islands for config in configurations]
    rng.shuffle(candidates)
    for cell in candidates:
        if len(cells) >= target:
            break
        island, _ = cell
        if cell not in cells and sum(left == island for left, _ in cells) < len(configurations) - 1:
            cells.add(cell)
    return {island: sorted(config for left, config in cells if left == island) for island in islands}


def _configuration_demand(configurations, count, rng):
    dominant = rng.randrange(len(configurations))
    weights = [2.5 if index == dominant else 1.0 for index in range(len(configurations))]
    result = list(configurations)
    result.extend(rng.choices(configurations, weights=weights, k=count - len(configurations)))
    rng.shuffle(result)
    return result


def _scale_nonzero(values, factor):
    for outer in values.values():
        for row in outer.values():
            for key, value in row.items():
                row[key] = 0 if value == 0 else max(1, round(value * factor))


def _scale_logistics(raw, w_factor, f_factor):
    for section in ("loaded_time", "empty_time"):
        for matrix in raw["logistics"]["W"][section].values():
            for row in matrix.values():
                for key, value in row.items():
                    row[key] = 0 if value == 0 else max(1, round(value * w_factor))
    for section in ("outbound_time", "return_time"):
        for row in raw["logistics"]["F"][section].values():
            for key, value in row.items():
                row[key] = max(1, round(value * f_factor))


def _normalize_intensities(raw, target_ri, target_w, target_f):
    metrics = benchmark_metrics(load_instance_dict(raw))
    _scale_nonzero(raw["reconfiguration"]["time"], target_ri / max(metrics["RI"], 1e-12))
    metrics = benchmark_metrics(load_instance_dict(raw))
    _scale_logistics(raw, target_w / max(metrics["W_transport_intensity"], 1e-12),
                     target_f / max(metrics["F_transport_intensity"], 1e-12))


def generate_candidate(instance_id: str, suite: str, scale: str, cf_level: str,
                       seed: int, specification: Mapping[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed)
    scale_rule = specification["scales"][scale]
    cf = specification["cf_levels"][cf_level]
    product_count = int(scale_rule["products"])
    total_operations = rng.randint(int(scale_rule["operations_min"]), int(scale_rule["operations_max"]))
    counts = _operation_counts(product_count, total_operations, rng)
    products_ids = [f"J{index}" for index in range(1, product_count + 1)]
    island_ids = [f"M{index}" for index in range(1, int(scale_rule["islands"]) + 1)]
    configurations = [f"C{index}" for index in range(1, int(scale_rule["configurations"]) + 1)]
    capabilities = _capabilities(island_ids, configurations, float(cf["cap_center"]), rng)
    products, all_operations = {}, []
    for product_index, (product, count) in enumerate(zip(products_ids, counts), 1):
        op_ids = [operation_id(product_index, index, count) for index in range(1, count + 1)]
        products[product] = {"operations": op_ids, "precedence": _dag(op_ids, rng)}
        all_operations.extend((op_id, product) for op_id in op_ids)
    demands = _configuration_demand(configurations, len(all_operations), rng)
    operations = {}
    desired = float(cf["route_center"]) * len(island_ids)
    for (op_id, product), config in zip(all_operations, demands):
        supporting = [island for island in island_ids if config in capabilities[island]]
        lower = max(2, math.floor(desired))
        upper = max(2, math.ceil(desired))
        eligible_count = min(len(supporting), rng.choice((lower, upper)))
        eligible = sorted(rng.sample(supporting, eligible_count))
        base = rng.randint(35, 90)
        target_cv = rng.uniform(0.27, 0.38)
        centered = [index - (len(eligible) - 1) / 2 for index in range(len(eligible))]
        deviation = math.sqrt(sum(value * value for value in centered) / len(centered))
        factors = [1.0 + target_cv * value / deviation for value in centered]
        rng.shuffle(factors)
        times = {island: max(1, round(base * factor)) for island, factor in zip(eligible, factors)}
        operations[op_id] = {"product": product, "required_configuration": config,
                             "eligible_islands": eligible, "processing_time": times}
    islands = {island: {"supported_configurations": capabilities[island],
                        "initial_configuration": rng.choice(capabilities[island])} for island in island_ids}
    coordinates = {"WH": (0, 0)}
    occupied = {(0, 0)}
    for island in island_ids:
        coordinate = (rng.randint(4, 50), rng.randint(4, 50))
        while coordinate in occupied:
            coordinate = (rng.randint(4, 50), rng.randint(4, 50))
        coordinates[island] = coordinate
        occupied.add(coordinate)
    generation_config = copy.deepcopy(DEFAULT_GENERATION_CONFIG)
    generation_config["layout"] = {"coordinate_min": 4, "coordinate_max": 50, "metric": "manhattan"}
    raw = finalize_instance(
        instance_id=instance_id, generator="RCIAS-CB1-native-v1", seed=seed,
        products=products, operations=operations, islands=islands, configurations=configurations,
        agvs_w=["W1", "W2"], agvs_f=["F1", "F2"], coordinates=coordinates,
        rng=rng, generation_config=generation_config,
        extra_meta={"suite": suite, "scale": scale, "CF_level": cf_level},
    )
    targets = specification["core_targets"]
    _normalize_intensities(raw, float(targets["ri_center"]), float(targets["w_ti_center"]), float(targets["f_ti_center"]))
    load_instance_dict(raw)
    return raw


def scale_sensitivity_variant(base: Mapping[str, Any], instance_id: str, ri_level: str,
                              ti_level: str, specification: Mapping[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    sensitivity = specification["sensitivity"]
    _scale_nonzero(raw["reconfiguration"]["time"], float(sensitivity["ri_factors"][ri_level]))
    factor = float(sensitivity["ti_factors"][ti_level])
    _scale_logistics(raw, factor, factor)
    raw["meta"].update({"instance_id": instance_id, "RI_level": ri_level, "TI_level": ti_level})
    load_instance_dict(raw)
    return raw


def configuration_entropy(raw: Mapping[str, Any]) -> float:
    configurations = raw["sets"]["configurations"]
    counts = Counter(record["required_configuration"] for record in raw["operations"].values())
    total = sum(counts.values())
    probabilities = [counts[config] / total for config in configurations if counts[config]]
    return -sum(value * math.log(value) for value in probabilities) / math.log(len(configurations))


def acceptance_failures(raw: Mapping[str, Any], scale: str, cf_level: str,
                        specification: Mapping[str, Any]) -> list[str]:
    instance = load_instance_dict(raw)
    metrics = benchmark_metrics(instance)
    scale_rule, cf, targets = (specification["scales"][scale], specification["cf_levels"][cf_level],
                               specification["core_targets"])
    failures = []
    checks = {
        "operation_count": int(scale_rule["operations_min"]) <= instance.num_operations <= int(scale_rule["operations_max"]),
        "minimum_two_eligible": all(len(instance.operation_data[op].eligible_islands) >= 2 for op in instance.operations),
        "capability_consistency": all(instance.operation_data[op].required_config in instance.island_data[island].supported_configs for op in instance.operations for island in instance.operation_data[op].eligible_islands),
        "configuration_two_islands": all(sum(config in instance.island_data[island].supported_configs for island in instance.islands) >= 2 for config in instance.configurations),
        "island_nonempty": all(instance.island_data[island].supported_configs for island in instance.islands),
        "capability_target": float(cf["cap_min"]) <= metrics["F_cap_mean"] <= float(cf["cap_max"]),
        "routing_target": float(cf["route_min"]) <= metrics["F_route_mean"] <= float(cf["route_max"]),
        "no_full_operation": metrics["R_full_op"] == 0,
        "few_full_islands": metrics["R_full_island"] <= 0.10,
        "processing_cv": float(targets["processing_cv_min"]) <= metrics["processing_CV_mean"] <= float(targets["processing_cv_max"]),
        "ri": float(targets["ri_min"]) <= metrics["RI"] <= float(targets["ri_max"]),
        "w_ti": float(targets["w_ti_min"]) <= metrics["W_transport_intensity"] <= float(targets["w_ti_max"]),
        "f_ti": float(targets["f_ti_min"]) <= metrics["F_transport_intensity"] <= float(targets["f_ti_max"]),
        "entropy": float(targets["entropy_min"]) <= configuration_entropy(raw) <= float(targets["entropy_max"]),
        "positive_processing": all(value > 0 for value in instance.processing_time.values()),
        "reconfiguration_diagonal": all(value == 0 for (island, source, target), value in instance.reconfiguration_time.items() if source == target),
        "branching_dags": sum(any(len(instance.successors[op]) > 1 for op in instance.product_data[product].operations) for product in instance.products) > len(instance.products) / 2,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return failures
