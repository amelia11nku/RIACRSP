"""Strict validation for the RCIAS-2.0 schema and preprocessed instances."""

from __future__ import annotations

from collections import deque

from .instance import Instance


class InstanceValidationError(ValueError):
    """Raised when an instance violates a named RCIAS model requirement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InstanceValidationError(message)


def _check_id(prefix: str, values: tuple[str, ...], category: str) -> None:
    _require(bool(values), f"{category} set must be non-empty")
    _require(len(values) == len(set(values)), f"duplicate IDs in {category}")
    for value in values:
        _require(isinstance(value, str) and value.startswith(prefix), f"invalid {category} ID: {value!r}")


def _is_acyclic(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> bool:
    indegree = {node: 0 for node in nodes}
    successors = {node: [] for node in nodes}
    for source, target in edges:
        if source not in indegree or target not in indegree or source == target:
            return False
        successors[source].append(target)
        indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(nodes)


def validate_instance(instance: Instance) -> None:
    """Validate every static constraint required by the mathematical model.

    The function deliberately has no fallback or repair behavior. A malformed
    reference is reported at the first exact field where it is encountered.
    """

    _require(instance.schema == "RCIAS-2.0", f"unsupported schema: {instance.schema!r}")
    _check_id("J", instance.products, "product")
    _check_id("o", instance.operations, "operation")
    _check_id("M", instance.islands, "island")
    _check_id("C", instance.configurations, "configuration")
    _check_id("W", instance.agvs_w, "W-AGV")
    _check_id("F", instance.agvs_f, "F-AGV")
    _require(instance.nodes == ("WH",) + instance.islands, "nodes must be WH followed by all islands")

    covered: list[str] = []
    for product_id in instance.products:
        _require(product_id in instance.product_data, f"missing product record: {product_id}")
        product = instance.product_data[product_id]
        _require(product.product_id == product_id, f"product key/ID mismatch: {product_id}")
        _require(bool(product.operations), f"product {product_id} has no operations")
        _require(len(product.operations) == len(set(product.operations)), f"duplicate operation in {product_id}")
        _require(
            _is_acyclic(product.operations, product.precedence),
            f"precedence graph for {product_id} is not a valid DAG",
        )
        for source, target in product.precedence:
            _require(source in product.operations and target in product.operations,
                     f"precedence edge {source}->{target} leaves product {product_id}")
        covered.extend(product.operations)
    _require(set(covered) == set(instance.operations) and len(covered) == len(instance.operations),
             "every operation must belong to exactly one product")

    for op_id in instance.operations:
        _require(op_id in instance.operation_data, f"missing operation record: {op_id}")
        operation = instance.operation_data[op_id]
        _require(operation.product_id in instance.product_data, f"unknown product on {op_id}")
        _require(instance.product_of.get(op_id) == operation.product_id,
                 f"product mapping mismatch for {op_id}")
        _require(operation.required_config in instance.configurations,
                 f"unknown required_configuration on {op_id}: {operation.required_config}")
        _require(bool(operation.eligible_islands), f"operation {op_id} has no eligible island")
        _require(len(operation.eligible_islands) == len(set(operation.eligible_islands)),
                 f"duplicate eligible island for {op_id}")
        _require(set(operation.processing_time) == set(operation.eligible_islands),
                 f"processing_time keys must exactly equal eligible_islands for {op_id}")
        for island_id in operation.eligible_islands:
            _require(island_id in instance.island_data, f"unknown eligible island {island_id} for {op_id}")
            _require(operation.required_config in instance.island_data[island_id].supported_configs,
                     f"{op_id} requires {operation.required_config}, unsupported by {island_id}")
            duration = instance.processing_time.get((op_id, island_id))
            _require(duration is not None and duration > 0,
                     f"processing time must be positive for {op_id}@{island_id}")

    for island_id in instance.islands:
        _require(island_id in instance.island_data, f"missing island record: {island_id}")
        island = instance.island_data[island_id]
        _require(bool(island.supported_configs), f"island {island_id} supports no configurations")
        _require(set(island.supported_configs) <= set(instance.configurations),
                 f"island {island_id} references an unknown configuration")
        _require(island.initial_config in island.supported_configs,
                 f"initial configuration of {island_id} is unsupported")
        for source in island.supported_configs:
            for target in island.supported_configs:
                time_key = (island_id, source, target)
                _require(time_key in instance.reconfiguration_time,
                         f"missing reconfiguration time {island_id}:{source}->{target}")
                _require(time_key in instance.reconfiguration_cost,
                         f"missing reconfiguration cost {island_id}:{source}->{target}")
                _require(instance.reconfiguration_time[time_key] >= 0,
                         f"negative reconfiguration time {island_id}:{source}->{target}")
                _require(instance.reconfiguration_cost[time_key] >= 0,
                         f"negative reconfiguration cost {island_id}:{source}->{target}")
                if source == target:
                    _require(instance.reconfiguration_time[time_key] == 0,
                             f"diagonal reconfiguration time must be zero at {island_id}/{source}")
                    _require(instance.reconfiguration_cost[time_key] == 0,
                             f"diagonal reconfiguration cost must be zero at {island_id}/{source}")

    for source in instance.nodes:
        for target in instance.nodes:
            _require((source, target) in instance.distance,
                     f"missing distance {source}->{target}")
            _require(instance.distance[(source, target)] >= 0,
                     f"negative distance {source}->{target}")
            for agv_id in instance.agvs_w:
                _require((agv_id, source, target) in instance.w_loaded_time,
                         f"missing W loaded time {agv_id}:{source}->{target}")
                _require((agv_id, source, target) in instance.w_empty_time,
                         f"missing W empty time {agv_id}:{source}->{target}")
                _require(instance.w_loaded_time[(agv_id, source, target)] >= 0,
                         f"negative W loaded time {agv_id}:{source}->{target}")
                _require(instance.w_empty_time[(agv_id, source, target)] >= 0,
                         f"negative W empty time {agv_id}:{source}->{target}")

    for agv_id in instance.agvs_w:
        _require(instance.w_loaded_cost_per_distance.get(agv_id, -1) >= 0,
                 f"missing/negative loaded cost for {agv_id}")
        _require(instance.w_empty_cost_per_distance.get(agv_id, -1) >= 0,
                 f"missing/negative empty cost for {agv_id}")
    for agv_id in instance.agvs_f:
        _require(instance.f_outbound_cost_per_distance.get(agv_id, -1) >= 0,
                 f"missing/negative outbound cost for {agv_id}")
        _require(instance.f_return_cost_per_distance.get(agv_id, -1) >= 0,
                 f"missing/negative return cost for {agv_id}")
        for island_id in instance.islands:
            _require((agv_id, island_id) in instance.f_outbound_time,
                     f"missing F outbound time {agv_id}:{island_id}")
            _require((agv_id, island_id) in instance.f_return_time,
                     f"missing F return time {agv_id}:{island_id}")
            _require(instance.f_outbound_time[(agv_id, island_id)] >= 0,
                     f"negative F outbound time {agv_id}:{island_id}")
            _require(instance.f_return_time[(agv_id, island_id)] >= 0,
                     f"negative F return time {agv_id}:{island_id}")
    _require(instance.objective_parameters.get("makespan_weight", -1) >= 0,
             "objective makespan_weight must be present and nonnegative")
    _require(instance.objective_parameters.get("cost_weight", -1) >= 0,
             "objective cost_weight must be present and nonnegative")
