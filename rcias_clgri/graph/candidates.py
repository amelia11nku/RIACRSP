"""Hierarchical candidate features without W-by-F Cartesian probing."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping

from rcias_clgri.data.instance import Instance
from rcias_clgri.env.schedule import Schedule
from rcias_clgri.env.timelines import (
    probe_f_insertion,
    probe_machine_insertion,
    probe_w_insertion,
)

from .normalization import FeatureNormalizer

NumericFeatures = Mapping[str, float]


@dataclass
class CandidateProbeStats:
    """Instrumentation used to enforce and profile linear candidate probing."""

    w_probes: int = 0
    f_probes: int = 0
    machine_probes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "w_probes": self.w_probes,
            "f_probes": self.f_probes,
            "machine_probes": self.machine_probes,
            "total_probes": self.w_probes + self.f_probes + self.machine_probes,
        }


class CandidateFeatureExtractor:
    """State-bound, cached feature extractor for autoregressive decisions.

    Only topologically ready operations invoke dynamic timeline probes. For an
    operation-island pair, W and F candidates are evaluated independently and
    the machine is probed once using their best arrival lower bounds.
    """

    def __init__(
        self,
        instance: Instance,
        schedule: Schedule,
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        self.instance = instance
        self.schedule = schedule
        self.normalizer = normalizer or FeatureNormalizer.from_instance(instance)
        selected = schedule.scheduled_operations
        self.ready_operations = tuple(
            op_id for op_id in instance.operations
            if op_id not in selected and instance.predecessors[op_id] <= selected
        )
        self._ready = set(self.ready_operations)
        self.stats = CandidateProbeStats()
        self._operation_cache: dict[str, dict[str, float]] | None = None
        self._island_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._w_cache: dict[tuple[str, str], dict[str | None, dict[str, float]]] = {}
        self._f_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        self._w_arrival: dict[tuple[str, str, str | None], float] = {}
        self._f_arrival: dict[tuple[str, str, str], float] = {}

    def _product_context(self, op_id: str) -> tuple[str | None, str, float]:
        product_id = self.instance.product_of[op_id]
        sequence = self.schedule.product_sequences[product_id]
        if not sequence:
            return None, "WH", 0.0
        predecessor = sequence[-1]
        record = self.schedule.operation_schedules[predecessor]
        return predecessor, record.island_id, record.completion_time

    def operation_features(self) -> Mapping[str, NumericFeatures]:
        if self._operation_cache is not None:
            return self._operation_cache
        result: dict[str, dict[str, float]] = {}
        selected = self.schedule.scheduled_operations
        norm = self.normalizer
        for op_id in self.instance.operations:
            operation = self.instance.operation_data[op_id]
            product = self.instance.product_data[operation.product_id]
            times = list(operation.processing_time.values())
            remaining = [item for item in product.operations if item not in selected]
            product_ready = 0.0
            if op_id in self._ready:
                _, _, product_ready = self._product_context(op_id)
            features = {
                "readiness": float(op_id in self._ready),
                "is_actionable": float(op_id in self._ready),
                "dynamic_feature_valid": float(op_id in self._ready),
                "is_scheduled": float(op_id in selected),
                "remaining_workload": norm.load(sum(
                    min(self.instance.operation_data[item].processing_time.values())
                    for item in remaining
                )),
                "criticality": norm.count(len(self.instance.transitive_successors[op_id])),
                "eligible_island_count": norm.count(len(operation.eligible_islands)),
                "min_processing_time": norm.time(min(times)),
                "mean_processing_time": norm.time(mean(times)),
                "product_progress": (
                    len(self.schedule.product_sequences[operation.product_id])
                    / max(1, len(product.operations))
                ),
                # Non-ready operations deliberately receive no future actual predecessor time.
                "product_ready_time": norm.time(product_ready),
            }
            norm.assert_finite(features)
            result[op_id] = features
        self._operation_cache = result
        return result

    def w_features(self, op_id: str, island_id: str) -> Mapping[str | None, NumericFeatures]:
        key = (op_id, island_id)
        if key in self._w_cache:
            return self._w_cache[key]
        if op_id not in self._ready:
            return {}
        instance, schedule, norm = self.instance, self.schedule, self.normalizer
        predecessor, pickup, product_ready = self._product_context(op_id)
        product_id = instance.product_of[op_id]
        result: dict[str | None, dict[str, float]] = {}
        if pickup == island_id:
            features = {
                "vehicle_workload": 0.0,
                "is_at_pickup": 1.0,
                "current_to_pickup_distance": 0.0,
                "empty_reposition_time": 0.0,
                "empty_reposition_distance": 0.0,
                "loaded_travel_time": 0.0,
                "loaded_travel_distance": 0.0,
                "earliest_arrival": norm.time(product_ready),
                "incremental_occupancy": 0.0,
                "dynamic_feature_valid": 1.0,
            }
            result[None] = features
            self._w_arrival[(op_id, island_id, None)] = product_ready
        else:
            for vehicle_id in instance.agvs_w:
                probe = probe_w_insertion(
                    instance, schedule, op_id, product_id, predecessor, pickup,
                    island_id, product_ready, vehicle_id,
                )
                self.stats.w_probes += 1
                task = probe.task
                workload = sum(
                    item.empty_travel_time + item.loaded_travel_time
                    for item in schedule.w_timelines[vehicle_id]
                )
                features = {
                    "vehicle_workload": norm.load(workload),
                    "is_at_pickup": float(task.empty_distance == 0.0),
                    "current_to_pickup_distance": norm.distance(task.empty_distance),
                    "empty_reposition_time": norm.time(task.empty_travel_time),
                    "empty_reposition_distance": norm.distance(task.empty_distance),
                    "loaded_travel_time": norm.time(task.loaded_travel_time),
                    "loaded_travel_distance": norm.distance(task.loaded_distance),
                    "earliest_arrival": norm.time(task.arrival_time),
                    "incremental_occupancy": norm.load(
                        task.empty_travel_time + task.loaded_travel_time
                    ),
                    "dynamic_feature_valid": 1.0,
                }
                norm.assert_finite(features)
                result[vehicle_id] = features
                self._w_arrival[(op_id, island_id, vehicle_id)] = task.arrival_time
        self._w_cache[key] = result
        return result

    def f_features(self, op_id: str, island_id: str) -> Mapping[str, NumericFeatures]:
        key = (op_id, island_id)
        if key in self._f_cache:
            return self._f_cache[key]
        if op_id not in self._ready:
            return {}
        instance, schedule, norm = self.instance, self.schedule, self.normalizer
        result: dict[str, dict[str, float]] = {}
        for vehicle_id in instance.agvs_f:
            probe = probe_f_insertion(instance, schedule, op_id, island_id, vehicle_id)
            self.stats.f_probes += 1
            task = probe.task
            workload = sum(
                item.return_wh - item.departure_wh for item in schedule.f_timelines[vehicle_id]
            )
            features = {
                "vehicle_workload": norm.load(workload),
                "earliest_departure": norm.time(task.departure_wh),
                "arrival_island": norm.time(task.arrival_island),
                "return_wh": norm.time(task.return_wh),
                "outbound_travel_time": norm.time(task.outbound_time),
                "return_travel_time": norm.time(task.return_time),
                "round_trip_occupancy": norm.load(task.return_wh - task.departure_wh),
                "outbound_distance": norm.distance(task.outbound_distance),
                "return_distance": norm.distance(task.return_distance),
                "dynamic_feature_valid": 1.0,
            }
            norm.assert_finite(features)
            result[vehicle_id] = features
            self._f_arrival[(op_id, island_id, vehicle_id)] = task.arrival_island
        self._f_cache[key] = result
        return result

    def _static_island_features(self, op_id: str, island_id: str) -> dict[str, float]:
        instance, norm = self.instance, self.normalizer
        operation = instance.operation_data[op_id]
        processing = float(instance.processing_time[(op_id, island_id)])
        configs = instance.island_data[island_id].supported_configs
        reconfig_time = min(
            instance.reconfiguration_time[(island_id, source, operation.required_config)]
            for source in configs
        )
        reconfig_cost = min(
            instance.reconfiguration_cost[(island_id, source, operation.required_config)]
            for source in configs
        )
        f_lower = min(instance.f_outbound_time[(vehicle, island_id)] for vehicle in instance.agvs_f)
        # The future product location is unknown, so only the global static lower bound is valid.
        w_lower = min(
            instance.w_loaded_time[(vehicle, origin, island_id)]
            for vehicle in instance.agvs_w for origin in instance.nodes
        )
        sync = max(float(reconfig_time), float(f_lower), float(w_lower))
        return {
            "processing_time": norm.time(processing),
            "same_configuration": 0.0,
            "reconfiguration_time": norm.time(reconfig_time),
            "reconfiguration_cost": norm.cost(reconfig_cost),
            "earliest_machine_insertion": 0.0,
            "w_lower_bound": norm.time(w_lower),
            "f_lower_bound": norm.time(f_lower),
            "synchronization_lower_bound": norm.time(sync),
            "estimated_completion": norm.time(sync + processing),
            "dynamic_feature_valid": 0.0,
            "is_actionable": 0.0,
        }

    def island_features(self, op_id: str) -> Mapping[str, NumericFeatures]:
        operation = self.instance.operation_data[op_id]
        for island_id in operation.eligible_islands:
            key = (op_id, island_id)
            if key in self._island_cache:
                continue
            if op_id not in self._ready:
                features = self._static_island_features(op_id, island_id)
                self.normalizer.assert_finite(features)
                self._island_cache[key] = features
                continue
            w_features = self.w_features(op_id, island_id)
            f_features = self.f_features(op_id, island_id)
            w_arrival = min(
                self._w_arrival[(op_id, island_id, candidate)] for candidate in w_features
            )
            f_arrival = min(
                self._f_arrival[(op_id, island_id, candidate)] for candidate in f_features
            )
            _, _, product_ready = self._product_context(op_id)
            base_ready = max(product_ready, w_arrival, f_arrival)
            machine = probe_machine_insertion(
                self.instance, self.schedule, op_id, island_id, base_ready
            )
            self.stats.machine_probes += 1
            if machine.previous_operation is None:
                previous_config = self.instance.island_data[island_id].initial_config
            else:
                previous_config = self.schedule.operation_schedules[
                    machine.previous_operation
                ].config_id
            required = operation.required_config
            sync = max(base_ready, machine.reconfiguration_end)
            norm = self.normalizer
            features = {
                "processing_time": norm.time(
                    self.instance.processing_time[(op_id, island_id)]
                ),
                "same_configuration": float(previous_config == required),
                "reconfiguration_time": norm.time(machine.setup_before),
                "reconfiguration_cost": norm.cost(machine.incremental_reconfiguration_cost),
                "earliest_machine_insertion": norm.time(machine.processing_start),
                "w_lower_bound": norm.time(w_arrival),
                "f_lower_bound": norm.time(f_arrival),
                "synchronization_lower_bound": norm.time(sync),
                "estimated_completion": norm.time(machine.processing_end),
                "dynamic_feature_valid": 1.0,
                "is_actionable": 1.0,
            }
            norm.assert_finite(features)
            self._island_cache[key] = features
        return {
            island_id: self._island_cache[(op_id, island_id)]
            for island_id in operation.eligible_islands
        }


def get_operation_candidate_features(
    state: CandidateFeatureExtractor,
) -> Mapping[str, NumericFeatures]:
    return state.operation_features()


def get_island_candidate_features(
    state: CandidateFeatureExtractor, op_id: str,
) -> Mapping[str, NumericFeatures]:
    return state.island_features(op_id)


def get_w_candidate_features(
    state: CandidateFeatureExtractor, op_id: str, island_id: str,
) -> Mapping[str | None, NumericFeatures]:
    return state.w_features(op_id, island_id)


def get_f_candidate_features(
    state: CandidateFeatureExtractor, op_id: str, island_id: str,
) -> Mapping[str, NumericFeatures]:
    return state.f_features(op_id, island_id)
