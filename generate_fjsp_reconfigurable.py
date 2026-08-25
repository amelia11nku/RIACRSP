#!/usr/bin/env python3
"""Extend an FJSP processing domain into a strict RCIAS-2.0 instance.

The source machine eligibility and processing times are preserved. Original
job-shop chains are intentionally replaced by sparse assembly DAGs, as required
by the RCIAS mathematical model.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

from rcias_clgri.data.generation import (
    DEFAULT_GENERATION_CONFIG,
    finalize_instance,
    operation_id,
    unique_coordinates,
    write_json,
)


def parse_fjsp(path: str | Path) -> dict[str, Any]:
    """Parse the common Brandimarte-style FJSP text format."""

    source = Path(path)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty FJSP file: {source}")
    header_tokens = lines[0].split()
    if len(header_tokens) < 2:
        raise ValueError("FJSP header must contain job and machine counts")
    num_jobs, num_machines = int(header_tokens[0]), int(header_tokens[1])
    if len(lines) - 1 < num_jobs:
        raise ValueError(f"expected {num_jobs} job lines, found {len(lines) - 1}")
    jobs: list[list[list[tuple[int, int]]]] = []
    for job_index in range(num_jobs):
        tokens = [int(token) for token in lines[job_index + 1].split()]
        if not tokens:
            raise ValueError(f"empty job line {job_index + 1}")
        position = 0
        num_operations = tokens[position]
        position += 1
        job: list[list[tuple[int, int]]] = []
        for operation_index in range(num_operations):
            if position >= len(tokens):
                raise ValueError(f"missing alternatives in job {job_index + 1}, operation {operation_index + 1}")
            num_alternatives = tokens[position]
            position += 1
            alternatives: list[tuple[int, int]] = []
            for _ in range(num_alternatives):
                if position + 1 >= len(tokens):
                    raise ValueError(f"malformed alternatives in job {job_index + 1}")
                machine, duration = tokens[position], tokens[position + 1]
                position += 2
                if not 1 <= machine <= num_machines or duration <= 0:
                    raise ValueError(f"invalid FJSP option machine={machine}, duration={duration}")
                alternatives.append((machine, duration))
            if not alternatives:
                raise ValueError("every FJSP operation needs at least one alternative")
            job.append(alternatives)
        if position != len(tokens):
            raise ValueError(f"unexpected trailing values on job line {job_index + 1}")
        jobs.append(job)
    return {"n_jobs": num_jobs, "n_machines": num_machines, "jobs": jobs}


def demo_fjsp() -> dict[str, Any]:
    """Return the reproducible medium demo processing domain."""

    return {
        "n_jobs": 3,
        "n_machines": 4,
        "jobs": [
            [[(1, 8), (2, 7)], [(2, 6), (3, 8)], [(1, 5), (4, 7)]],
            [[(2, 7), (3, 6)], [(1, 9), (4, 6)], [(3, 5), (4, 4)]],
            [[(1, 6), (3, 7)], [(2, 5), (4, 6)], [(1, 8), (2, 7), (3, 6)]],
        ],
    }


def tiny_fjsp() -> dict[str, Any]:
    """Return a two-product domain used by exact validation."""

    return {
        "n_jobs": 2,
        "n_machines": 2,
        "jobs": [
            [[(1, 4), (2, 5)], [(1, 3), (2, 4)], [(1, 5), (2, 3)]],
            [[(1, 3), (2, 4)], [(1, 4), (2, 2)]],
        ],
    }


def small_fjsp() -> dict[str, Any]:
    """Return a three-product, three-island validation domain."""

    return {
        "n_jobs": 3,
        "n_machines": 3,
        "jobs": [
            [[(1, 5), (2, 4)], [(2, 6), (3, 5)], [(1, 4), (3, 3)]],
            [[(1, 4), (3, 6)], [(1, 5), (2, 4)], [(2, 3), (3, 5)]],
            [[(1, 6), (2, 5)], [(2, 4), (3, 4)], [(1, 3), (3, 4)]],
        ],
    }


def stable_seed(family: str, instance_name: str) -> int:
    """Derive a cross-process stable 32-bit seed from public source identity."""

    key = f"RCIAS-2.0::{family}::{instance_name}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _sparse_dag(
    op_ids: list[str],
    rng: random.Random,
    edge_probability: float,
) -> list[list[str]]:
    """Create an acyclic partial order without freezing a job-shop chain."""

    if len(op_ids) < 2:
        return []
    if len(op_ids) == 2:
        return []
    edges: set[tuple[str, str]] = {(op_ids[0], op_ids[-1])}
    for left in range(len(op_ids) - 1):
        for right in range(left + 1, len(op_ids)):
            if (left, right) in {(0, len(op_ids) - 1), (0, 1), (1, 2)}:
                continue
            if rng.random() < edge_probability:
                edges.add((op_ids[left], op_ids[right]))
    return [list(edge) for edge in sorted(edges)]


def build_instance(
    base: Mapping[str, Any],
    seed: int = 42,
    num_configurations: int | None = None,
    num_agv_w: int = 2,
    num_agv_f: int = 2,
    source_name: str = "built-in-demo",
    instance_id: str | None = None,
    family: str = "fjsp_demo",
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate an RCIAS-2.0 instance from an FJSP domain."""

    rng = random.Random(seed)
    num_jobs = int(base["n_jobs"])
    num_machines = int(base["n_machines"])
    jobs = base["jobs"]
    if num_jobs < 1 or num_machines < 2 or len(jobs) != num_jobs:
        raise ValueError("FJSP base must contain jobs and at least two machines")
    if num_agv_w < 1 or num_agv_f < 1:
        raise ValueError("both logistics fleets must be non-empty")
    config = dict(DEFAULT_GENERATION_CONFIG if generation_config is None else generation_config)
    config_rule = config["configuration_generation"]
    config_count = num_configurations or max(
        int(config_rule["minimum"]),
        min(int(config_rule["maximum"]), num_machines + int(config_rule["island_offset"])),
    )
    if config_count < 2:
        raise ValueError("at least two configurations are required")

    island_ids = [f"M{index}" for index in range(1, num_machines + 1)]
    configurations = [f"C{index}" for index in range(1, config_count + 1)]
    agvs_w = [f"W{index}" for index in range(1, num_agv_w + 1)]
    agvs_f = [f"F{index}" for index in range(1, num_agv_f + 1)]
    products: dict[str, Any] = {}
    operations: dict[str, Any] = {}
    required_by_island: dict[str, set[str]] = {island_id: set() for island_id in island_ids}

    config_cursor = 0
    for product_index, job in enumerate(jobs, start=1):
        product_id = f"J{product_index}"
        op_ids = [
            operation_id(product_index, operation_index, len(job))
            for operation_index in range(1, len(job) + 1)
        ]
        products[product_id] = {
            "operations": op_ids,
            "precedence": _sparse_dag(op_ids, rng, float(config["dag_generation"]["optional_edge_probability"])),
            "family": "FJSP-expanded",
        }
        for operation_index, alternatives in enumerate(job, start=1):
            op_id = op_ids[operation_index - 1]
            required_config = configurations[config_cursor % config_count]
            config_cursor += 1
            eligible = [f"M{int(machine)}" for machine, _ in alternatives]
            processing = {f"M{int(machine)}": int(duration) for machine, duration in alternatives}
            if len(eligible) != len(set(eligible)):
                raise ValueError(f"duplicate machine alternative on {op_id}")
            for island_id in eligible:
                required_by_island[island_id].add(required_config)
            operations[op_id] = {
                "product": product_id,
                "required_configuration": required_config,
                "eligible_islands": eligible,
                "processing_time": processing,
                "benchmark_operation_index": operation_index,
            }

    islands: dict[str, Any] = {}
    for island_id in island_ids:
        supported = set(required_by_island[island_id])
        while len(supported) < min(2, config_count):
            supported.add(rng.choice(configurations))
        supported_ordered = [config for config in configurations if config in supported]
        islands[island_id] = {
            "supported_configurations": supported_ordered,
            "initial_configuration": rng.choice(supported_ordered),
        }
    coordinates = unique_coordinates(island_ids, rng, generation_config=config)
    return finalize_instance(
        instance_id=instance_id or f"fjsp-rcias-{num_jobs}x{num_machines}-s{seed}",
        generator="FJSP RCIAS-2.0 extension",
        seed=seed,
        products=products,
        operations=operations,
        islands=islands,
        configurations=configurations,
        agvs_w=agvs_w,
        agvs_f=agvs_f,
        coordinates=coordinates,
        rng=rng,
        generation_config=config,
        extra_meta={
            "source": source_name,
            "family": family,
            "generator_version": str(config["generator_version"]),
            "mapping": {"job": "product", "operation": "operation", "machine": "assembly_island"},
        },
    )


def build_tiny_instance(seed: int = 7) -> dict[str, Any]:
    """Generate the FJSP-family exact-validation instance."""

    return build_instance(
        tiny_fjsp(), seed=seed, num_configurations=3, num_agv_w=1, num_agv_f=1,
        source_name="built-in-tiny", instance_id="fjsp-tiny",
    )


def build_small_instance(seed: int = 17) -> dict[str, Any]:
    """Generate the slightly larger FJSP-family validation instance."""

    return build_instance(
        small_fjsp(), seed=seed, num_configurations=3, num_agv_w=2, num_agv_f=2,
        source_name="built-in-small", instance_id="fjsp-small",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an RCIAS-2.0 FJSP extension")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/generated/fjsp_reconfigurable.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configurations", type=int, default=None)
    parser.add_argument("--agv-w", type=int, default=2)
    parser.add_argument("--agv-f", type=int, default=2)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if args.tiny:
        instance = build_tiny_instance(args.seed)
    else:
        base = parse_fjsp(args.input) if args.input else demo_fjsp()
        source = str(args.input) if args.input else "built-in-demo"
        instance = build_instance(
            base, seed=args.seed, num_configurations=args.configurations,
            num_agv_w=args.agv_w, num_agv_f=args.agv_f, source_name=source,
        )
    write_json(instance, args.output)
    print(
        f"OK: {args.output} | products={len(instance['sets']['products'])} "
        f"operations={len(instance['sets']['operations'])} islands={len(instance['sets']['islands'])}"
    )


if __name__ == "__main__":
    main()
