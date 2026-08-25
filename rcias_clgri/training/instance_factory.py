"""Independent synthetic instance distribution for Phase 3 learning."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from rcias_clgri.data.generation import (
    DEFAULT_GENERATION_CONFIG,
    finalize_instance,
    operation_id,
    unique_coordinates,
)
from rcias_clgri.data.instance import Instance
from rcias_clgri.data.loader import load_instance_dict


@dataclass(frozen=True)
class LevelSpecification:
    """Integer/continuous ranges defining one genuinely synthetic level."""

    products: tuple[int, int]
    operations_per_product: tuple[int, int]
    total_operations: tuple[int, int] | None
    islands: tuple[int, int]
    configurations: tuple[int, int]
    w_agvs: tuple[int, int]
    f_agvs: tuple[int, int]
    eligible_islands: tuple[int, int]
    processing_time: tuple[int, int]
    dag_probability: tuple[float, float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LevelSpecification":
        def pair(name: str, cast):
            values = data[name]
            return cast(values[0]), cast(values[1])

        total = data.get("total_operations")
        return cls(
            products=pair("products", int),
            operations_per_product=pair("operations_per_product", int),
            total_operations=None if total is None else (int(total[0]), int(total[1])),
            islands=pair("islands", int),
            configurations=pair("configurations", int),
            w_agvs=pair("w_agvs", int),
            f_agvs=pair("f_agvs", int),
            eligible_islands=pair("eligible_islands", int),
            processing_time=pair("processing_time", int),
            dag_probability=pair("dag_probability", float),
        )


def _integer_quantile(rows: list[dict[str, str]], field: str, q: float) -> int:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return int(round(float(np.quantile(values, q))))


class TrainingInstanceFactory:
    """Map ``(seed, level)`` to a new validated :class:`Instance`.

    Canonical operation-level records are never loaded.  Only manifest
    quantiles are used to calibrate the Level-L size envelope.
    """

    LEVELS = ("S", "M", "L")

    def __init__(
        self,
        levels: Mapping[str, Mapping[str, Any]],
        manifest_path: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self._manifest_rows = list(csv.DictReader(
            self.manifest_path.open(encoding="utf-8", newline="")
        ))
        if len(self._manifest_rows) != 130:
            raise ValueError("training calibration manifest must contain 130 rows")
        self.levels = {
            name: self._resolve_level(name, levels[name]) for name in self.LEVELS
        }

    def _resolve_level(
        self, name: str, data: Mapping[str, Any],
    ) -> LevelSpecification:
        resolved = dict(data)
        if name == "L" and data.get("manifest_driven", False):
            low, _middle, high = (float(value) for value in data["manifest_quantiles"])
            resolved["products"] = [
                _integer_quantile(self._manifest_rows, "num_products", low),
                _integer_quantile(self._manifest_rows, "num_products", high),
            ]
            resolved["total_operations"] = [
                _integer_quantile(self._manifest_rows, "num_operations", low),
                _integer_quantile(self._manifest_rows, "num_operations", high),
            ]
            resolved["islands"] = [
                _integer_quantile(self._manifest_rows, "num_islands", low),
                _integer_quantile(self._manifest_rows, "num_islands", high),
            ]
            resolved["configurations"] = [
                max(2, _integer_quantile(self._manifest_rows, "num_configurations", low)),
                _integer_quantile(self._manifest_rows, "num_configurations", high),
            ]
            resolved["eligible_islands"] = [1, max(
                2, _integer_quantile(self._manifest_rows, "mean_eligible_islands", high)
            )]
            resolved["processing_time"] = [
                max(1, _integer_quantile(self._manifest_rows, "mean_processing_time", low) // 3),
                max(2, _integer_quantile(self._manifest_rows, "mean_processing_time", high) * 2),
            ]
        return LevelSpecification.from_mapping(resolved)

    @staticmethod
    def _operation_counts(
        rng: random.Random, spec: LevelSpecification, products: int,
    ) -> list[int]:
        low, high = spec.operations_per_product
        counts = [rng.randint(low, high) for _ in range(products)]
        if spec.total_operations is None:
            return counts
        target = rng.randint(*spec.total_operations)
        target = max(products * low, min(products * high, target))
        while sum(counts) < target:
            choices = [index for index, count in enumerate(counts) if count < high]
            counts[rng.choice(choices)] += 1
        while sum(counts) > target:
            choices = [index for index, count in enumerate(counts) if count > low]
            counts[rng.choice(choices)] -= 1
        return counts

    @staticmethod
    def _supported_configurations(
        rng: random.Random, island_ids: list[str], config_ids: list[str],
    ) -> dict[str, list[str]]:
        support: dict[str, list[str]] = {}
        for island_id in island_ids:
            count = rng.randint(max(1, len(config_ids) // 2), len(config_ids))
            support[island_id] = sorted(rng.sample(config_ids, count))
        for config_id in config_ids:
            if not any(config_id in values for values in support.values()):
                island_id = rng.choice(island_ids)
                support[island_id] = sorted({*support[island_id], config_id})
        return support

    @staticmethod
    def _dag(
        rng: random.Random, operation_ids: list[str], probability: float,
    ) -> list[list[str]]:
        edges: set[tuple[str, str]] = set()
        for right in range(1, len(operation_ids)):
            for left in range(right):
                if rng.random() < probability:
                    edges.add((operation_ids[left], operation_ids[right]))
            # Preserve a meaningful partial order without forcing a full chain.
            if right >= 2 and not any(target == operation_ids[right] for _, target in edges):
                if rng.random() < 0.65:
                    left = rng.randrange(0, right)
                    edges.add((operation_ids[left], operation_ids[right]))
        return [list(edge) for edge in sorted(edges)]

    def sample_raw(
        self, seed: int, level: str, *, scenario: str = "standard",
    ) -> dict[str, Any]:
        normalized = level.upper()
        if normalized not in self.levels:
            raise ValueError(f"unknown curriculum level: {level}")
        if scenario not in {
            "standard", "high_reconfiguration", "fleet_scarcity", "high_travel"
        }:
            raise ValueError(f"unknown synthetic generalization scenario: {scenario}")
        spec = self.levels[normalized]
        rng = random.Random(int(seed))
        num_products = rng.randint(*spec.products)
        operation_counts = self._operation_counts(rng, spec, num_products)
        num_islands = rng.randint(*spec.islands)
        num_configs = rng.randint(*spec.configurations)
        island_ids = [f"M{index}" for index in range(1, num_islands + 1)]
        config_ids = [f"C{index}" for index in range(1, num_configs + 1)]
        num_w = 1 if scenario == "fleet_scarcity" else rng.randint(*spec.w_agvs)
        num_f = 1 if scenario == "fleet_scarcity" else rng.randint(*spec.f_agvs)
        w_ids = [f"W{index}" for index in range(1, num_w + 1)]
        f_ids = [f"F{index}" for index in range(1, num_f + 1)]
        supported = self._supported_configurations(rng, island_ids, config_ids)
        islands = {
            island_id: {
                "supported_configurations": supported[island_id],
                "initial_configuration": rng.choice(supported[island_id]),
            }
            for island_id in island_ids
        }
        products: dict[str, dict[str, Any]] = {}
        operations: dict[str, dict[str, Any]] = {}
        dag_probability = rng.uniform(*spec.dag_probability)
        for product_index, count in enumerate(operation_counts, start=1):
            product_id = f"J{product_index}"
            op_ids = [operation_id(product_index, index, count) for index in range(1, count + 1)]
            products[product_id] = {
                "operations": op_ids,
                "precedence": self._dag(rng, op_ids, dag_probability),
            }
            for op_id in op_ids:
                required = rng.choice(config_ids)
                compatible = [
                    island_id for island_id in island_ids
                    if required in supported[island_id]
                ]
                requested = rng.randint(*spec.eligible_islands)
                eligible = sorted(rng.sample(compatible, min(requested, len(compatible))))
                operations[op_id] = {
                    "product": product_id,
                    "required_configuration": required,
                    "eligible_islands": eligible,
                    "processing_time": {
                        island_id: rng.randint(*spec.processing_time)
                        for island_id in eligible
                    },
                }
        generation_config = copy.deepcopy(DEFAULT_GENERATION_CONFIG)
        generation_config["layout"]["coordinate_max"] = max(24, 3 * num_islands)
        if scenario == "high_reconfiguration":
            rule = generation_config["reconfiguration"]
            rule["time_base"] *= 2
            rule["time_per_index_separation"] *= 2
        elif scenario == "high_travel":
            travel = generation_config["travel"]
            for key in tuple(travel):
                if "speed" in key:
                    travel[key] = float(travel[key]) * 0.55
        raw = finalize_instance(
            instance_id=f"synthetic_{normalized}_{int(seed)}",
            generator="phase3-independent-synthetic-distribution",
            seed=int(seed),
            products=products,
            operations=operations,
            islands=islands,
            configurations=config_ids,
            agvs_w=w_ids,
            agvs_f=f_ids,
            coordinates=unique_coordinates(island_ids, rng, generation_config=generation_config),
            rng=rng,
            generation_config=generation_config,
            extra_meta={
                "data_split": "synthetic_training_or_validation",
                "curriculum_level": normalized,
                "generalization_scenario": scenario,
                "canonical_operation_records_used": False,
                "dag_probability": round(dag_probability, 6),
            },
        )
        return raw

    def sample(
        self, seed: int, level: str, *, scenario: str = "standard",
    ) -> Instance:
        return load_instance_dict(self.sample_raw(seed, level, scenario=scenario))
