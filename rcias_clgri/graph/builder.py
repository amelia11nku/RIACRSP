"""ID-invariant dynamic Capability-Logistics Coupled Graph builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.schedule import Schedule

from .candidates import (
    CandidateFeatureExtractor,
    CandidateProbeStats,
    NumericFeatures,
    get_f_candidate_features,
    get_island_candidate_features,
    get_operation_candidate_features,
    get_w_candidate_features,
)
from .normalization import FeatureNormalizer


@dataclass(frozen=True)
class EdgeRecord:
    source_type: str
    relation: str
    target_type: str
    source_id: str
    target_id: str
    features: NumericFeatures


@dataclass(frozen=True)
class GraphState:
    node_features: Mapping[str, Mapping[str, NumericFeatures]]
    edges: tuple[EdgeRecord, ...]
    ready_operations: tuple[str, ...]
    operation_mask: Mapping[str, bool]
    island_masks: Mapping[str, Mapping[str, bool]]
    w_masks: Mapping[tuple[str, str], tuple[str | None, ...]]
    f_masks: Mapping[tuple[str, str], tuple[str, ...]]
    operation_candidates: Mapping[str, NumericFeatures]
    island_candidates: Mapping[tuple[str, str], NumericFeatures]
    w_candidates: Mapping[tuple[str, str, str | None], NumericFeatures]
    f_candidates: Mapping[tuple[str, str, str], NumericFeatures]
    normalization: Mapping[str, float]
    probe_stats: CandidateProbeStats

    @property
    def candidate_choice_count(self) -> int:
        return (
            len(self.ready_operations)
            + sum(sum(mask.values()) for mask in self.island_masks.values())
            + len(self.w_candidates)
            + len(self.f_candidates)
        )


def _largest_gap(intervals: list[tuple[float, float]], horizon: float) -> float:
    if not intervals:
        return horizon
    ordered = sorted(intervals)
    largest = ordered[0][0]
    for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]):
        largest = max(largest, right_start - left_end)
    return max(largest, horizon - ordered[-1][1])


def _assert_numeric_graph(
    node_features: Mapping[str, Mapping[str, NumericFeatures]],
    edges: list[EdgeRecord],
) -> None:
    for nodes in node_features.values():
        for features in nodes.values():
            FeatureNormalizer.assert_finite(features)
            if not all(isinstance(value, (int, float)) for value in features.values()):
                raise TypeError("node features must be numeric and ID-independent")
    for edge in edges:
        FeatureNormalizer.assert_finite(edge.features)
        if not all(isinstance(value, (int, float)) for value in edge.features.values()):
            raise TypeError("edge features must be numeric and ID-independent")


def build_graph_state(instance: Instance, schedule: Schedule) -> GraphState:
    """Build typed nodes/relations and hierarchical hard-masked candidates."""

    norm = FeatureNormalizer.from_instance(instance)
    extractor = CandidateFeatureExtractor(instance, schedule, norm)
    ready = extractor.ready_operations
    ready_set = set(ready)
    operation_candidates = get_operation_candidate_features(extractor)
    node_features: dict[str, dict[str, dict[str, float]]] = {
        "O": {}, "J": {}, "M": {}, "W": {}, "F": {},
    }

    for op_id in instance.operations:
        operation = instance.operation_data[op_id]
        product_size = len(instance.product_data[operation.product_id].operations)
        record = schedule.operation_schedules.get(op_id)
        features = dict(operation_candidates[op_id])
        features.update({
            "dag_in_degree": norm.count(len(instance.predecessors[op_id])),
            "dag_out_degree": norm.count(len(instance.successors[op_id])),
            "transitive_predecessor_ratio": (
                len(instance.transitive_predecessors[op_id]) / max(1, product_size - 1)
            ),
            "transitive_successor_ratio": (
                len(instance.transitive_successors[op_id]) / max(1, product_size - 1)
            ),
            "scheduled_start": 0.0 if record is None else norm.time(record.start_time),
            "scheduled_end": 0.0 if record is None else norm.time(record.completion_time),
        })
        node_features["O"][op_id] = features

    for product_id in instance.products:
        sequence = schedule.product_sequences[product_id]
        operations = instance.product_data[product_id].operations
        remaining = [op_id for op_id in operations if op_id not in schedule.operation_schedules]
        remaining_work = sum(
            min(instance.operation_data[op_id].processing_time.values()) for op_id in remaining
        )
        last_record = None if not sequence else schedule.operation_schedules[sequence[-1]]
        location = "WH" if last_record is None else last_record.island_id
        node_features["J"][product_id] = {
            "operation_count": norm.count(len(operations)),
            "scheduled_ratio": len(sequence) / max(1, len(operations)),
            "remaining_workload": norm.load(remaining_work),
            "last_actual_completion": norm.time(0.0 if last_record is None else last_record.completion_time),
            "at_warehouse": float(location == "WH"),
            "distance_to_warehouse": norm.distance(instance.distance[(location, "WH")]),
        }

    for island_id in instance.islands:
        sequence = schedule.island_timelines[island_id]
        records = [schedule.operation_schedules[op_id] for op_id in sequence]
        intervals = [(record.start_time, record.completion_time) for record in records]
        initial = instance.island_data[island_id].initial_config
        tail = initial if not records else records[-1].config_id
        node_features["M"][island_id] = {
            "processing_load": norm.load(sum(record.processing_time for record in records)),
            "reconfiguration_load": norm.load(sum(
                record.reconfiguration_end - record.reconfiguration_start for record in records
            )),
            "tail_completion": norm.time(0.0 if not records else records[-1].completion_time),
            "scheduled_operation_count": norm.count(len(records)),
            "supported_configuration_count": norm.count(
                len(instance.island_data[island_id].supported_configs)
            ),
            "free_gap_count": norm.count(len(records) + 1),
            "largest_free_gap": norm.time(_largest_gap(intervals, instance.horizon)),
            "tail_equals_initial_configuration": float(tail == initial),
        }

    for vehicle_id in instance.agvs_w:
        tasks = schedule.w_timelines[vehicle_id]
        intervals = [(task.empty_start, task.arrival_time) for task in tasks]
        location = "WH" if not tasks else tasks[-1].destination
        node_features["W"][vehicle_id] = {
            "task_count": norm.count(len(tasks)),
            "loaded_load": norm.load(sum(task.loaded_travel_time for task in tasks)),
            "empty_load": norm.load(sum(task.empty_travel_time for task in tasks)),
            "tail_completion": norm.time(0.0 if not tasks else tasks[-1].arrival_time),
            "largest_free_gap": norm.time(_largest_gap(intervals, instance.horizon)),
            "at_warehouse": float(location == "WH"),
            "distance_to_warehouse": norm.distance(instance.distance[(location, "WH")]),
        }

    for vehicle_id in instance.agvs_f:
        tasks = schedule.f_timelines[vehicle_id]
        intervals = [(task.departure_wh, task.return_wh) for task in tasks]
        node_features["F"][vehicle_id] = {
            "task_count": norm.count(len(tasks)),
            "busy_load": norm.load(sum(task.return_wh - task.departure_wh for task in tasks)),
            "tail_return": norm.time(0.0 if not tasks else tasks[-1].return_wh),
            "largest_free_gap": norm.time(_largest_gap(intervals, instance.horizon)),
            "at_warehouse": 1.0,
        }

    edges: list[EdgeRecord] = []
    for product_id in instance.products:
        for op_id in instance.product_data[product_id].operations:
            edges.append(EdgeRecord("J", "contains", "O", product_id, op_id, {}))
            edges.append(EdgeRecord("O", "belongs_to", "J", op_id, product_id, {}))
        for source, target in instance.product_data[product_id].precedence:
            edges.append(EdgeRecord("O", "precedence", "O", source, target, {}))
            edges.append(EdgeRecord("O", "precedence_rev", "O", target, source, {}))

    island_candidates: dict[tuple[str, str], NumericFeatures] = {}
    for op_id in instance.operations:
        candidates = get_island_candidate_features(extractor, op_id)
        for island_id, features in candidates.items():
            island_candidates[(op_id, island_id)] = features
            edges.append(EdgeRecord("O", "eligible_on", "M", op_id, island_id, features))
            edges.append(EdgeRecord("M", "can_process", "O", island_id, op_id, features))

    max_w_loaded = max(instance.w_loaded_time.values(), default=1)
    for source in instance.islands:
        for target in instance.islands:
            if source != target:
                edges.append(EdgeRecord("M", "spatial", "M", source, target, {
                    "distance": norm.distance(instance.distance[(source, target)]),
                    "minimum_loaded_time": norm.time(min(
                        instance.w_loaded_time[(vehicle, source, target)]
                        for vehicle in instance.agvs_w
                    )),
                }))
    for vehicle_id in instance.agvs_w:
        for island_id in instance.islands:
            features = {
                "warehouse_distance": norm.distance(instance.distance[("WH", island_id)]),
                "loaded_time_from_warehouse": norm.time(
                    instance.w_loaded_time[(vehicle_id, "WH", island_id)]
                ),
                "relative_loaded_time": (
                    instance.w_loaded_time[(vehicle_id, "WH", island_id)]
                    / max(1.0, float(max_w_loaded))
                ),
            }
            edges.append(EdgeRecord("W", "reachable_to", "M", vehicle_id, island_id, features))
            edges.append(EdgeRecord("M", "reachable_by", "W", island_id, vehicle_id, features))
    for vehicle_id in instance.agvs_f:
        for island_id in instance.islands:
            features = {
                "outbound_time": norm.time(instance.f_outbound_time[(vehicle_id, island_id)]),
                "return_time": norm.time(instance.f_return_time[(vehicle_id, island_id)]),
                "round_trip_time": norm.time(
                    instance.f_outbound_time[(vehicle_id, island_id)]
                    + instance.f_return_time[(vehicle_id, island_id)]
                ),
                "warehouse_distance": norm.distance(instance.distance[("WH", island_id)]),
            }
            edges.append(EdgeRecord("F", "deliver_to", "M", vehicle_id, island_id, features))
            edges.append(EdgeRecord("M", "served_by", "F", island_id, vehicle_id, features))
    for sequence in schedule.product_sequences.values():
        for source, target in zip(sequence, sequence[1:]):
            edges.append(EdgeRecord("O", "actual_product_prev", "O", source, target, {}))
    for sequence in schedule.island_timelines.values():
        for source, target in zip(sequence, sequence[1:]):
            edges.append(EdgeRecord("O", "machine_prev", "O", source, target, {}))

    operation_mask = {op_id: op_id in ready_set for op_id in instance.operations}
    island_masks = {
        op_id: {
            island_id: op_id in ready_set and island_id in instance.operation_data[op_id].eligible_islands
            for island_id in instance.islands
        }
        for op_id in instance.operations
    }
    w_masks: dict[tuple[str, str], tuple[str | None, ...]] = {}
    f_masks: dict[tuple[str, str], tuple[str, ...]] = {}
    w_candidates: dict[tuple[str, str, str | None], NumericFeatures] = {}
    f_candidates: dict[tuple[str, str, str], NumericFeatures] = {}
    for op_id in ready:
        for island_id in instance.operation_data[op_id].eligible_islands:
            w_items = get_w_candidate_features(extractor, op_id, island_id)
            f_items = get_f_candidate_features(extractor, op_id, island_id)
            w_masks[(op_id, island_id)] = tuple(w_items)
            f_masks[(op_id, island_id)] = tuple(f_items)
            for vehicle_id, features in w_items.items():
                w_candidates[(op_id, island_id, vehicle_id)] = features
            for vehicle_id, features in f_items.items():
                f_candidates[(op_id, island_id, vehicle_id)] = features

    _assert_numeric_graph(node_features, edges)
    return GraphState(
        node_features=node_features,
        edges=tuple(edges),
        ready_operations=ready,
        operation_mask=operation_mask,
        island_masks=island_masks,
        w_masks=w_masks,
        f_masks=f_masks,
        operation_candidates=operation_candidates,
        island_candidates=island_candidates,
        w_candidates=w_candidates,
        f_candidates=f_candidates,
        normalization=norm.to_dict(),
        probe_stats=extractor.stats,
    )
