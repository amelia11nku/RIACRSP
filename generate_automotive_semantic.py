#!/usr/bin/env python3
"""Generate automotive-semantic RCIAS-2.0 instances."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Sequence

from rcias_clgri.data.generation import finalize_instance, operation_id, unique_coordinates, write_json


# code -> (human label, fixed capability configuration, base processing time)
TASKS: dict[str, tuple[str, str, int]] = {
    "BODY_POS": ("Body positioning", "C1", 8),
    "WIRING": ("Interior wiring harness", "C2", 12),
    "COCKPIT": ("Cockpit installation", "C3", 14),
    "SEAT_F": ("Front-seat installation", "C4", 10),
    "SEAT_R": ("Rear-seat installation", "C4", 9),
    "DOOR": ("Door-module installation", "C3", 13),
    "WHEEL": ("Wheel installation", "C5", 11),
    "FLUID_FILL": ("Fluid filling", "C6", 8),
    "ADAS": ("ADAS calibration", "C7", 10),
    "EOL": ("End-of-line test", "C2", 9),
    "HV_BAT": ("High-voltage battery installation", "C8", 16),
    "FUEL_CHECK": ("Fuel-system check", "C6", 9),
    "ROOF": ("Panoramic-roof bonding", "C9", 12),
    "ROW3": ("Third-row seat installation", "C4", 9),
}


def variant_task_codes(variant: str, panoramic: bool) -> list[str]:
    codes = ["BODY_POS", "WIRING", "COCKPIT", "SEAT_F", "SEAT_R", "DOOR", "WHEEL", "FLUID_FILL", "ADAS", "EOL"]
    if variant in {"EV", "SUV_EV"}:
        codes.insert(1, "HV_BAT")
    if variant == "ICE":
        codes.insert(-2, "FUEL_CHECK")
    if variant == "SUV_EV":
        codes.insert(-3, "ROW3")
    if panoramic:
        codes.insert(3, "ROOF")
    return codes


def build_precedence(op_by_code: dict[str, str]) -> list[list[str]]:
    """Build a multi-branch automotive assembly DAG."""

    edges: set[tuple[str, str]] = set()

    def add(source: str, target: str) -> None:
        if source in op_by_code and target in op_by_code:
            edges.add((op_by_code[source], op_by_code[target]))

    for target in ["WIRING", "HV_BAT", "DOOR", "WHEEL", "ROOF"]:
        add("BODY_POS", target)
    add("WIRING", "COCKPIT")
    for target in ["SEAT_F", "SEAT_R", "FLUID_FILL"]:
        add("COCKPIT", target)
    for source in ["ROOF", "DOOR", "WHEEL", "COCKPIT", "HV_BAT"]:
        add(source, "ADAS")
    for source in ["FLUID_FILL", "ADAS", "SEAT_F", "SEAT_R", "ROW3", "FUEL_CHECK"]:
        add(source, "EOL")
    return [list(edge) for edge in sorted(edges)]


def _build_from_product_codes(
    product_codes: Sequence[Sequence[str]],
    *,
    seed: int,
    num_islands: int,
    num_agv_w: int,
    num_agv_f: int,
    min_eligible_islands: int,
    instance_id: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    if num_islands < 2 or num_agv_w < 1 or num_agv_f < 1:
        raise ValueError("RCIAS instances need >=2 islands and non-empty W/F fleets")
    min_eligible_islands = min(max(1, min_eligible_islands), num_islands)
    island_ids = [f"M{index}" for index in range(1, num_islands + 1)]
    agvs_w = [f"W{index}" for index in range(1, num_agv_w + 1)]
    agvs_f = [f"F{index}" for index in range(1, num_agv_f + 1)]
    used_configs = sorted({TASKS[code][1] for codes in product_codes for code in codes}, key=lambda value: int(value[1:]))

    supported: dict[str, set[str]] = {island_id: set() for island_id in island_ids}
    for config_id in used_configs:
        for island_id in rng.sample(island_ids, min_eligible_islands):
            supported[island_id].add(config_id)
    for island_id in island_ids:
        for config_id in used_configs:
            if rng.random() < 0.45:
                supported[island_id].add(config_id)
        if not supported[island_id]:
            supported[island_id].add(rng.choice(used_configs))

    products: dict[str, Any] = {}
    operations: dict[str, Any] = {}
    efficiency = {island_id: rng.uniform(0.84, 1.18) for island_id in island_ids}
    variants = ["EV", "ICE", "SUV_EV"]
    for product_index, codes in enumerate(product_codes, start=1):
        product_id = f"J{product_index}"
        op_ids = [operation_id(product_index, index, len(codes)) for index in range(1, len(codes) + 1)]
        op_by_code = dict(zip(codes, op_ids))
        products[product_id] = {
            "operations": op_ids,
            "precedence": build_precedence(op_by_code),
            "family": "Automotive-Final-Assembly",
            "variant": variants[(product_index - 1) % len(variants)],
        }
        product_factor = rng.uniform(0.93, 1.10)
        for op_id, code in zip(op_ids, codes):
            label, config_id, base_time = TASKS[code]
            eligible = [island_id for island_id in island_ids if config_id in supported[island_id]]
            processing = {
                island_id: max(1, int(round(base_time * product_factor * efficiency[island_id] * rng.uniform(0.96, 1.05))))
                for island_id in eligible
            }
            operations[op_id] = {
                "product": product_id,
                "required_configuration": config_id,
                "eligible_islands": eligible,
                "processing_time": processing,
                "semantic_code": code,
                "semantic_name": label,
            }

    islands = {
        island_id: {
            "supported_configurations": [config_id for config_id in used_configs if config_id in supported[island_id]],
            "initial_configuration": rng.choice(sorted(supported[island_id], key=lambda value: int(value[1:]))),
        }
        for island_id in island_ids
    }
    coordinates = unique_coordinates(island_ids, rng, lower=4, upper=30)
    return finalize_instance(
        instance_id=instance_id,
        generator="Automotive semantic RCIAS-2.0",
        seed=seed,
        products=products,
        operations=operations,
        islands=islands,
        configurations=used_configs,
        agvs_w=agvs_w,
        agvs_f=agvs_f,
        coordinates=coordinates,
        rng=rng,
        extra_meta={"source": "synthetic automotive final assembly"},
    )


def build_instance(
    seed: int = 42,
    num_products: int = 4,
    num_islands: int = 6,
    num_agv_w: int = 2,
    num_agv_f: int = 2,
    min_eligible_islands: int = 2,
) -> dict[str, Any]:
    """Generate a normal automotive-semantic instance."""

    if num_products < 1:
        raise ValueError("num_products must be positive")
    variants = ["EV", "ICE", "SUV_EV"]
    preview_rng = random.Random(seed)
    product_codes = [
        variant_task_codes(variants[index % len(variants)], preview_rng.random() < 0.45)
        for index in range(num_products)
    ]
    return _build_from_product_codes(
        product_codes,
        seed=seed,
        num_islands=num_islands,
        num_agv_w=num_agv_w,
        num_agv_f=num_agv_f,
        min_eligible_islands=min_eligible_islands,
        instance_id=f"automotive-rcias-{num_products}x{num_islands}-s{seed}",
    )


def build_tiny_instance(seed: int = 11) -> dict[str, Any]:
    """Generate a nonlinear-DAG automotive instance for exact validation."""

    product_codes = [
        ["BODY_POS", "WIRING", "HV_BAT"],
        ["SEAT_F", "SEAT_R", "EOL"],
    ]
    return _build_from_product_codes(
        product_codes,
        seed=seed,
        num_islands=3,
        num_agv_w=1,
        num_agv_f=1,
        min_eligible_islands=2,
        instance_id="automotive-tiny",
    )


def build_small_instance(seed: int = 19) -> dict[str, Any]:
    """Generate three nonlinear four-operation product DAGs."""

    product_codes = [
        ["BODY_POS", "WIRING", "HV_BAT", "COCKPIT"],
        ["BODY_POS", "DOOR", "WHEEL", "ADAS"],
        ["COCKPIT", "SEAT_F", "SEAT_R", "EOL"],
    ]
    return _build_from_product_codes(
        product_codes,
        seed=seed,
        num_islands=3,
        num_agv_w=2,
        num_agv_f=2,
        min_eligible_islands=2,
        instance_id="automotive-small",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an automotive RCIAS-2.0 instance")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/generated/automotive_semantic.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--products", type=int, default=4)
    parser.add_argument("--islands", type=int, default=6)
    parser.add_argument("--agv-w", type=int, default=2)
    parser.add_argument("--agv-f", type=int, default=2)
    parser.add_argument("--min-eligible-islands", type=int, default=2)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    instance = build_tiny_instance(args.seed) if args.tiny else build_instance(
        seed=args.seed,
        num_products=args.products,
        num_islands=args.islands,
        num_agv_w=args.agv_w,
        num_agv_f=args.agv_f,
        min_eligible_islands=args.min_eligible_islands,
    )
    write_json(instance, args.output)
    print(
        f"OK: {args.output} | products={len(instance['sets']['products'])} "
        f"operations={len(instance['sets']['operations'])} islands={len(instance['sets']['islands'])}"
    )


if __name__ == "__main__":
    main()
