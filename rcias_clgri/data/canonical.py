"""Canonical public benchmark discovery, preservation checks, and metrics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from generate_fjsp_reconfigurable import build_instance, parse_fjsp, stable_seed
from rcias_clgri.data.generation import deterministic_json_text
from rcias_clgri.data.loader import load_instance_dict


@dataclass(frozen=True)
class PublicSource:
    instance_id: str
    family: str
    source_path: Path
    relative_source: str
    output_relative: Path


def discover_public_sources(source_root: str | Path) -> tuple[PublicSource, ...]:
    """Return the exact frozen Mk and Hurink la e/r/v source catalogue."""

    root = Path(source_root)
    sources: list[PublicSource] = []
    for index in range(1, 11):
        name = f"Mk{index:02d}"
        path = root / "Mk" / f"{name}.fjs"
        sources.append(PublicSource(
            f"BR_{name}", "brandimarte", path, path.relative_to(root).as_posix(),
            Path("brandimarte") / f"BR_{name}.json",
        ))
    hurink_codes = {"edata": "E", "rdata": "R", "vdata": "V"}
    for subfamily, code in hurink_codes.items():
        for index in range(1, 41):
            name = f"la{index:02d}"
            path = root / "Hurink_Data" / subfamily / f"{name}.fjs"
            sources.append(PublicSource(
                f"HU_{code}_{name}", f"hurink_{subfamily}", path,
                path.relative_to(root).as_posix(),
                Path("hurink") / subfamily / f"HU_{code}_{name}.json",
            ))
    missing = [str(source.source_path) for source in sources if not source.source_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing public benchmark sources: {missing}")
    if len(sources) != 130:
        raise RuntimeError(f"public source catalogue must contain 130 instances, got {len(sources)}")
    return tuple(sources)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def assert_has_incomparable_pair(operations: tuple[str, ...] | list[str], edges: tuple[tuple[str, str], ...] | list[list[str]]) -> None:
    """Raise if a product with >=2 operations is a total order."""

    nodes = tuple(operations)
    if len(nodes) < 2:
        return
    successors = {node: set() for node in nodes}
    for source, target in edges:
        successors[source].add(target)
    closure: dict[str, set[str]] = {}
    for node in nodes:
        reached: set[str] = set()
        stack = list(successors[node])
        while stack:
            current = stack.pop()
            if current not in reached:
                reached.add(current)
                stack.extend(successors[current] - reached)
        closure[node] = reached
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1:]:
            if right not in closure[left] and left not in closure[right]:
                return
    raise ValueError("generated product DAG has no incomparable operation pair")


def verify_public_processing_preservation(base: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    """Compare every source FJSP option against its generated RCIAS operation."""

    for product_index, job in enumerate(base["jobs"], start=1):
        product_id = f"J{product_index}"
        generated_ops = raw["products"][product_id]["operations"]
        if len(generated_ops) != len(job):
            raise ValueError(f"operation count changed for {product_id}")
        for operation_index, alternatives in enumerate(job):
            op_id = generated_ops[operation_index]
            expected_times = {f"M{machine}": int(duration) for machine, duration in alternatives}
            operation = raw["operations"][op_id]
            if operation["processing_time"] != expected_times:
                raise ValueError(f"processing times changed for {op_id}")
            if operation["eligible_islands"] != list(expected_times):
                raise ValueError(f"machine eligibility changed for {op_id}")


def generate_public_instance(source: PublicSource, config: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Generate and fully validate one deterministic public extension."""

    base = parse_fjsp(source.source_path)
    seed = stable_seed(source.family, source.source_path.stem)
    fleet = config["fleet"]
    raw = build_instance(
        base,
        seed=seed,
        num_agv_w=int(fleet["num_w_agvs"]),
        num_agv_f=int(fleet["num_f_agvs"]),
        source_name=source.relative_source,
        instance_id=source.instance_id,
        family=source.family,
        generation_config=config,
    )
    verify_public_processing_preservation(base, raw)
    for product in raw["products"].values():
        assert_has_incomparable_pair(product["operations"], product["precedence"])
    load_instance_dict(raw)
    return raw, base


def manifest_record(
    source: PublicSource,
    raw: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the required frozen manifest row."""

    operation_records = list(raw["operations"].values())
    eligible_counts = [len(operation["eligible_islands"]) for operation in operation_records]
    processing = [
        duration for operation in operation_records for duration in operation["processing_time"].values()
    ]
    product_records = list(raw["products"].values())
    edge_count = sum(len(product["precedence"]) for product in product_records)
    possible_edges = sum(len(product["operations"]) * (len(product["operations"]) - 1) / 2 for product in product_records)
    reconfiguration_values = [
        duration
        for island_id, matrix in raw["reconfiguration"]["time"].items()
        for source_config, row in matrix.items()
        for target_config, duration in row.items()
        if source_config != target_config
    ]
    generated_bytes = deterministic_json_text(raw).encode("utf-8")
    return {
        "instance_id": source.instance_id,
        "family": source.family,
        "source_file": source.relative_source,
        "seed": raw["meta"]["seed"],
        "num_products": len(raw["sets"]["products"]),
        "num_operations": len(raw["sets"]["operations"]),
        "num_islands": len(raw["sets"]["islands"]),
        "num_configurations": len(raw["sets"]["configurations"]),
        "num_w_agvs": len(raw["sets"]["agvs_w"]),
        "num_f_agvs": len(raw["sets"]["agvs_f"]),
        "num_precedence_edges": edge_count,
        "dag_density": round(edge_count / possible_edges if possible_edges else 0.0, 8),
        "mean_eligible_islands": round(mean(eligible_counts), 8),
        "min_eligible_islands": min(eligible_counts),
        "max_eligible_islands": max(eligible_counts),
        "mean_processing_time": round(mean(processing), 8),
        "reconfiguration_time_mean": round(mean(reconfiguration_values), 8),
        "reconfiguration_time_max": max(reconfiguration_values),
        "source_sha256": sha256_file(source.source_path),
        "generated_sha256": sha256_bytes(generated_bytes),
        "schema_version": raw["meta"]["schema"],
        "generator_version": raw["meta"]["generator_version"],
        "valid": True,
    }
