"""JSON loader for the strict RCIAS-2.0 data contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .instance import Instance, IslandData, OperationData, ProductData
from .validator import InstanceValidationError, validate_instance


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise InstanceValidationError(f"field {key!r} must be an object")
    return value


def _tuple_ids(sets: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = sets.get(key)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise InstanceValidationError(f"sets.{key} must be a list of strings")
    return tuple(values)


def _flatten_2d(raw: Mapping[str, Any], first: tuple[str, ...], second: tuple[str, ...], name: str) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for left in first:
        row = raw.get(left)
        if not isinstance(row, Mapping):
            raise InstanceValidationError(f"missing row {name}.{left}")
        for right in second:
            value = row.get(right)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InstanceValidationError(f"{name}.{left}.{right} must be numeric")
            result[(left, right)] = float(value)
    return result


def _flatten_3d(raw: Mapping[str, Any], first: tuple[str, ...], second_by_first: Mapping[str, tuple[str, ...]], name: str) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for outer in first:
        matrix = raw.get(outer)
        if not isinstance(matrix, Mapping):
            raise InstanceValidationError(f"missing matrix {name}.{outer}")
        members = second_by_first[outer]
        for source in members:
            row = matrix.get(source)
            if not isinstance(row, Mapping):
                raise InstanceValidationError(f"missing row {name}.{outer}.{source}")
            for target in members:
                value = row.get(target)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise InstanceValidationError(f"{name}.{outer}.{source}.{target} must be numeric")
                result[(outer, source, target)] = float(value)
    return result


def _transitive_sets(
    operations: tuple[str, ...],
    adjacency: Mapping[str, set[str]],
) -> dict[str, frozenset[str]]:
    closure: dict[str, frozenset[str]] = {}
    for operation in operations:
        reached: set[str] = set()
        stack = list(adjacency[operation])
        while stack:
            current = stack.pop()
            if current in reached:
                continue
            reached.add(current)
            stack.extend(adjacency[current] - reached)
        closure[operation] = frozenset(reached)
    return closure


def load_instance_dict(raw: Mapping[str, Any]) -> Instance:
    """Build and validate an :class:`Instance` from a decoded JSON object."""

    meta = _mapping(raw, "meta")
    schema = meta.get("schema")
    if schema != "RCIAS-2.0":
        raise InstanceValidationError(
            f"expected RCIAS-2.0, received {schema!r}; legacy module-based schemas are not accepted"
        )
    sets = _mapping(raw, "sets")
    products = _tuple_ids(sets, "products")
    operations = _tuple_ids(sets, "operations")
    islands = _tuple_ids(sets, "islands")
    configurations = _tuple_ids(sets, "configurations")
    agvs_w = _tuple_ids(sets, "agvs_w")
    agvs_f = _tuple_ids(sets, "agvs_f")
    nodes = _tuple_ids(sets, "nodes")

    products_raw = _mapping(raw, "products")
    product_data: dict[str, ProductData] = {}
    for product_id in products:
        record = products_raw.get(product_id)
        if not isinstance(record, Mapping):
            raise InstanceValidationError(f"missing products.{product_id}")
        op_ids = record.get("operations")
        precedence = record.get("precedence")
        if not isinstance(op_ids, list) or not isinstance(precedence, list):
            raise InstanceValidationError(f"invalid operations/precedence for {product_id}")
        try:
            edges = tuple((str(edge[0]), str(edge[1])) for edge in precedence if len(edge) == 2)
        except (TypeError, IndexError) as error:
            raise InstanceValidationError(f"invalid precedence edge in {product_id}") from error
        if len(edges) != len(precedence):
            raise InstanceValidationError(f"every precedence edge in {product_id} must have two endpoints")
        attributes = {key: value for key, value in record.items() if key not in {"operations", "precedence"}}
        product_data[product_id] = ProductData(product_id, tuple(op_ids), edges, attributes)

    operations_raw = _mapping(raw, "operations")
    operation_data: dict[str, OperationData] = {}
    product_of: dict[str, str] = {}
    processing_time: dict[tuple[str, str], int] = {}
    for op_id in operations:
        record = operations_raw.get(op_id)
        if not isinstance(record, Mapping):
            raise InstanceValidationError(f"missing operations.{op_id}")
        product_id = record.get("product")
        required_config = record.get("required_configuration")
        eligible = record.get("eligible_islands")
        times = record.get("processing_time")
        if not isinstance(product_id, str) or not isinstance(required_config, str):
            raise InstanceValidationError(f"invalid product/configuration on {op_id}")
        if not isinstance(eligible, list) or not isinstance(times, Mapping):
            raise InstanceValidationError(f"invalid eligibility/processing time on {op_id}")
        parsed_times: dict[str, int] = {}
        for island_id, value in times.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise InstanceValidationError(f"processing time {op_id}@{island_id} must be an integer")
            parsed_times[str(island_id)] = value
            processing_time[(op_id, str(island_id))] = value
        attributes = {
            key: value for key, value in record.items()
            if key not in {"product", "required_configuration", "eligible_islands", "processing_time"}
        }
        operation_data[op_id] = OperationData(
            op_id, product_id, required_config, tuple(eligible), parsed_times, attributes
        )
        product_of[op_id] = product_id

    islands_raw = _mapping(raw, "islands")
    island_data: dict[str, IslandData] = {}
    for island_id in islands:
        record = islands_raw.get(island_id)
        if not isinstance(record, Mapping):
            raise InstanceValidationError(f"missing islands.{island_id}")
        supported = record.get("supported_configurations")
        initial = record.get("initial_configuration")
        coordinate_raw = record.get("coordinate")
        if not isinstance(supported, list) or not isinstance(initial, str):
            raise InstanceValidationError(f"invalid configurations on {island_id}")
        coordinate = None
        if coordinate_raw is not None:
            if not isinstance(coordinate_raw, list) or len(coordinate_raw) != 2:
                raise InstanceValidationError(f"coordinate for {island_id} must have two values")
            coordinate = (int(coordinate_raw[0]), int(coordinate_raw[1]))
        attributes = {
            key: value for key, value in record.items()
            if key not in {"supported_configurations", "initial_configuration", "coordinate"}
        }
        island_data[island_id] = IslandData(island_id, tuple(supported), initial, coordinate, attributes)

    predecessors_mutable = {op_id: set() for op_id in operations}
    successors_mutable = {op_id: set() for op_id in operations}
    for product in product_data.values():
        for source, target in product.precedence:
            if source in successors_mutable and target in predecessors_mutable:
                successors_mutable[source].add(target)
                predecessors_mutable[target].add(source)
    predecessors = {op_id: frozenset(values) for op_id, values in predecessors_mutable.items()}
    successors = {op_id: frozenset(values) for op_id, values in successors_mutable.items()}
    transitive_successors = _transitive_sets(operations, successors_mutable)
    transitive_predecessors = _transitive_sets(operations, predecessors_mutable)

    reconfiguration = _mapping(raw, "reconfiguration")
    configs_by_island = {island_id: island_data[island_id].supported_configs for island_id in islands}
    reconfiguration_time_float = _flatten_3d(
        _mapping(reconfiguration, "time"), islands, configs_by_island, "reconfiguration.time"
    )
    reconfiguration_cost = _flatten_3d(
        _mapping(reconfiguration, "cost"), islands, configs_by_island, "reconfiguration.cost"
    )
    reconfiguration_time = {key: int(value) for key, value in reconfiguration_time_float.items()}

    logistics = _mapping(raw, "logistics")
    distance = _flatten_2d(_mapping(logistics, "distance"), nodes, nodes, "logistics.distance")
    w_raw = _mapping(logistics, "W")
    loaded_raw = _mapping(w_raw, "loaded_time")
    empty_raw = _mapping(w_raw, "empty_time")
    w_loaded_time: dict[tuple[str, str, str], int] = {}
    w_empty_time: dict[tuple[str, str, str], int] = {}
    for agv_id in agvs_w:
        loaded = _flatten_2d(_mapping(loaded_raw, agv_id), nodes, nodes, f"logistics.W.loaded_time.{agv_id}")
        empty = _flatten_2d(_mapping(empty_raw, agv_id), nodes, nodes, f"logistics.W.empty_time.{agv_id}")
        w_loaded_time.update({(agv_id, source, target): int(value) for (source, target), value in loaded.items()})
        w_empty_time.update({(agv_id, source, target): int(value) for (source, target), value in empty.items()})
    w_loaded_cost = {str(key): float(value) for key, value in _mapping(w_raw, "loaded_cost_per_distance").items()}
    w_empty_cost = {str(key): float(value) for key, value in _mapping(w_raw, "empty_cost_per_distance").items()}

    f_raw = _mapping(logistics, "F")
    f_outbound_time_float = _flatten_2d(
        _mapping(f_raw, "outbound_time"), agvs_f, islands, "logistics.F.outbound_time"
    )
    f_return_time_float = _flatten_2d(
        _mapping(f_raw, "return_time"), agvs_f, islands, "logistics.F.return_time"
    )
    f_outbound_time = {key: int(value) for key, value in f_outbound_time_float.items()}
    f_return_time = {key: int(value) for key, value in f_return_time_float.items()}
    f_outbound_cost = {str(key): float(value) for key, value in _mapping(f_raw, "outbound_cost_per_distance").items()}
    f_return_cost = {str(key): float(value) for key, value in _mapping(f_raw, "return_cost_per_distance").items()}
    objective_parameters_raw = raw.get("objective_parameters", {"makespan_weight": 1.0, "cost_weight": 0.0})
    if not isinstance(objective_parameters_raw, Mapping):
        raise InstanceValidationError("objective_parameters must be an object")
    objective_parameters = {str(key): float(value) for key, value in objective_parameters_raw.items()}

    instance = Instance(
        instance_id=str(meta.get("instance_id", "unnamed")),
        schema=str(schema),
        seed=int(meta["seed"]) if meta.get("seed") is not None else None,
        products=products,
        operations=operations,
        islands=islands,
        configurations=configurations,
        agvs_w=agvs_w,
        agvs_f=agvs_f,
        nodes=nodes,
        product_data=product_data,
        operation_data=operation_data,
        island_data=island_data,
        predecessors=predecessors,
        successors=successors,
        transitive_predecessors=transitive_predecessors,
        transitive_successors=transitive_successors,
        product_of=product_of,
        processing_time=processing_time,
        reconfiguration_time=reconfiguration_time,
        reconfiguration_cost=reconfiguration_cost,
        distance=distance,
        w_loaded_time=w_loaded_time,
        w_empty_time=w_empty_time,
        w_loaded_cost_per_distance=w_loaded_cost,
        w_empty_cost_per_distance=w_empty_cost,
        f_outbound_time=f_outbound_time,
        f_return_time=f_return_time,
        f_outbound_cost_per_distance=f_outbound_cost,
        f_return_cost_per_distance=f_return_cost,
        objective_parameters=objective_parameters,
        metadata=dict(meta),
    )
    validate_instance(instance)
    return instance


def load_instance(path: str | Path) -> Instance:
    """Load a UTF-8 JSON file, rejecting malformed JSON and legacy schemas."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InstanceValidationError(f"invalid JSON in {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise InstanceValidationError(f"top-level JSON in {source} must be an object")
    return load_instance_dict(raw)
