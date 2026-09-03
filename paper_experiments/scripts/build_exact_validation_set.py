#!/usr/bin/env python3
"""Build the deterministic pre-heuristic exact-validation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.generation import (  # noqa: E402
    DEFAULT_GENERATION_CONFIG,
    deterministic_json_text,
    finalize_instance,
    operation_id,
)
from rcias_clgri.data.loader import load_instance_dict  # noqa: E402


DEFAULT_DESIGN_PATH = ROOT / "paper_experiments/configs/exact_validation/design_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "paper_experiments/benchmarks/exact_validation_10_v2"
GENERATOR_VERSION = "initial-manuscript-exact-v1"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def balanced_counts(total: int, products: int) -> list[int]:
    quotient, remainder = divmod(total, products)
    return [quotient + (index < remainder) for index in range(products)]


def precedence(operations: list[str], pattern: str) -> list[list[str]]:
    if len(operations) < 2:
        return []
    if pattern == "linear" or len(operations) < 3:
        return [[left, right] for left, right in zip(operations, operations[1:])]
    if pattern == "mixed" and len(operations) % 2 == 0:
        return [[left, right] for left, right in zip(operations, operations[1:])]
    if len(operations) == 3:
        return [[operations[0], operations[1]], [operations[0], operations[2]]]
    edges = [
        [operations[0], operations[1]],
        [operations[0], operations[2]],
        [operations[1], operations[-1]],
        [operations[2], operations[-1]],
    ]
    for index in range(3, len(operations) - 1):
        edges.append([operations[index - 2], operations[index]])
        edges.append([operations[index], operations[-1]])
    return sorted({tuple(edge) for edge in edges})


def capabilities(island_ids: list[str], configurations: list[str]) -> dict[str, list[str]]:
    if len(island_ids) == 2:
        return {island: list(configurations) for island in island_ids}
    result = {
        island_ids[0]: ["C1", "C2"],
        island_ids[1]: ["C2", "C3"],
        island_ids[2]: ["C1", "C3"],
    }
    for island in island_ids[3:]:
        result[island] = list(configurations)
    return result


def build_case(case: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(int(case["seed"]))
    product_count = int(case["products"])
    operation_count = int(case["operations"])
    island_ids = [f"M{index}" for index in range(1, int(case["islands"]) + 1)]
    configurations = ["C1", "C2", "C3"]
    island_capabilities = capabilities(island_ids, configurations)
    products: dict[str, Any] = {}
    operations: dict[str, Any] = {}
    global_index = 0
    for product_index, count in enumerate(
        balanced_counts(operation_count, product_count), start=1
    ):
        product_id = f"J{product_index}"
        operation_ids = [
            operation_id(product_index, index, count)
            for index in range(1, count + 1)
        ]
        pattern = str(case["dag_pattern"])
        product_pattern = "linear" if pattern == "mixed" and product_index % 2 == 0 else pattern
        products[product_id] = {
            "operations": operation_ids,
            "precedence": precedence(operation_ids, product_pattern),
        }
        for operation in operation_ids:
            required = configurations[(global_index + product_index) % len(configurations)]
            supporting = [
                island for island in island_ids
                if required in island_capabilities[island]
            ]
            alternative_operations = int(case.get("alternative_operations", operation_count))
            eligible_count = 1 if global_index >= alternative_operations else 2
            if (
                eligible_count > 1
                and case["eligibility_pattern"] == "wide"
                and len(supporting) > 2
                and global_index % 2 == 0
            ):
                eligible_count = len(supporting)
            eligible = sorted(rng.sample(supporting, eligible_count))
            base = rng.randint(7, 28)
            shuffled = list(eligible)
            rng.shuffle(shuffled)
            processing = {
                island: base + 3 * index + rng.randint(0, 2)
                for index, island in enumerate(shuffled)
            }
            operations[operation] = {
                "product": product_id,
                "required_configuration": required,
                "eligible_islands": eligible,
                "processing_time": {island: processing[island] for island in eligible},
            }
            global_index += 1
    islands = {
        island: {
            "supported_configurations": island_capabilities[island],
            "initial_configuration": island_capabilities[island][
                (index + int(case["seed"])) % len(island_capabilities[island])
            ],
        }
        for index, island in enumerate(island_ids)
    }
    occupied = {(0, 0)}
    coordinates: dict[str, tuple[int, int]] = {"WH": (0, 0)}
    for island in island_ids:
        coordinate = (rng.randint(3, 18), rng.randint(3, 18))
        while coordinate in occupied:
            coordinate = (rng.randint(3, 18), rng.randint(3, 18))
        occupied.add(coordinate)
        coordinates[island] = coordinate
    generation_config = json.loads(json.dumps(DEFAULT_GENERATION_CONFIG))
    generation_config["layout"] = {
        "coordinate_min": 3,
        "coordinate_max": 18,
        "metric": "manhattan",
    }
    return finalize_instance(
        instance_id=str(case["instance_id"]),
        generator=GENERATOR_VERSION,
        seed=int(case["seed"]),
        products=products,
        operations=operations,
        islands=islands,
        configurations=configurations,
        agvs_w=[f"W{index}" for index in range(1, int(case["w_agvs"]) + 1)],
        agvs_f=[f"F{index}" for index in range(1, int(case["f_agvs"]) + 1)],
        coordinates=coordinates,
        rng=rng,
        generation_config=generation_config,
        extra_meta={
            "suite": "INITIAL_MANUSCRIPT_EXACT_VALIDATION_CANDIDATE",
            "design_status": "PRE_HEURISTIC_NOT_FROZEN",
            "dag_pattern": case["dag_pattern"],
            "eligibility_pattern": case["eligibility_pattern"],
            "generation_rationale": case["rationale"],
        },
    )


def write_if_identical_or_absent(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to overwrite non-identical candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def describe(case: dict[str, Any], raw: dict[str, Any], path: Path) -> dict[str, Any]:
    instance = load_instance_dict(raw)
    processing = [float(value) for value in instance.processing_time.values()]
    travel = [
        *map(float, instance.w_loaded_time.values()),
        *map(float, instance.w_empty_time.values()),
        *map(float, instance.f_outbound_time.values()),
        *map(float, instance.f_return_time.values()),
    ]
    precedence_arcs = [
        [source, target]
        for product in instance.products
        for source, target in instance.product_data[product].precedence
    ]
    return {
        "instance_id": instance.instance_id,
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "seed": int(case["seed"]),
        "operation_count": instance.num_operations,
        "product_count": len(instance.products),
        "island_count": len(instance.islands),
        "w_agv_count": len(instance.agvs_w),
        "f_agv_count": len(instance.agvs_f),
        "precedence_arcs": precedence_arcs,
        "precedence_arc_count": len(precedence_arcs),
        "eligibility_density": sum(
            len(instance.operation_data[operation].eligible_islands)
            for operation in instance.operations
        ) / (instance.num_operations * len(instance.islands)),
        "processing_time": {
            "minimum": min(processing),
            "maximum": max(processing),
            "mean": statistics.fmean(processing),
            "population_stdev": statistics.pstdev(processing),
        },
        "travel_time": {
            "minimum": min(travel),
            "maximum": max(travel),
            "mean": statistics.fmean(travel),
            "population_stdev": statistics.pstdev(travel),
        },
        "dag_pattern": case["dag_pattern"],
        "eligibility_pattern": case["eligibility_pattern"],
        "rationale": case["rationale"],
    }


def audit(raw: dict[str, Any], case: dict[str, Any]) -> None:
    instance = load_instance_dict(raw)
    checks = {
        "operation_count": instance.num_operations == int(case["operations"]) <= 12,
        "multiple_products": len(instance.products) >= 2,
        "alternative_islands": sum(
            len(instance.operation_data[operation].eligible_islands) >= 2
            for operation in instance.operations
        ) == int(case.get("alternative_operations", instance.num_operations)),
        "heterogeneous_processing": len(set(instance.processing_time.values())) > 1,
        "w_f_fleets_present": bool(instance.agvs_w) and bool(instance.agvs_f),
        "resource_conflicts_possible": instance.num_operations > len(instance.islands),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{instance.instance_id} failed structural checks: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    design_path = args.design if args.design.is_absolute() else ROOT / args.design
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    design = json.loads(design_path.read_text(encoding="utf-8"))
    cases = design["cases"]
    if len(cases) != 10 or len({case["instance_id"] for case in cases}) != 10:
        raise RuntimeError("exact-validation design must contain exactly 10 unique cases")
    records = []
    for case in cases:
        raw = build_case(case)
        audit(raw, case)
        path = output_root / "instances" / f"{case['instance_id']}.json"
        write_if_identical_or_absent(path, deterministic_json_text(raw))
        records.append(describe(case, raw, path))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema": "initial-manuscript-exact-candidate-manifest-v1",
        "status": "CANDIDATE_PRE_HEURISTIC_NOT_FROZEN",
        "heuristic_results_observed_during_design": False,
        "instance_count": len(records),
        "maximum_operations": max(record["operation_count"] for record in records),
        "generator": GENERATOR_VERSION,
        "generator_source": str(Path(__file__).relative_to(ROOT)),
        "generator_source_sha256": digest(Path(__file__)),
        "design_path": str(design_path.relative_to(ROOT)),
        "design_sha256": digest(design_path),
        "git_commit": commit,
        "instances": records,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    write_if_identical_or_absent(output_root / "candidate_manifest.json", manifest_text)
    checksums = "".join(
        f"{record['sha256']}  instances/{record['instance_id']}.json\n"
        for record in records
    )
    write_if_identical_or_absent(output_root / "checksums.sha256", checksums)
    print(f"EXACT_CANDIDATES_READY count={len(records)} max_operations={manifest['maximum_operations']}")


if __name__ == "__main__":
    main()
