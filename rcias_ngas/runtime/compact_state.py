"""Compact, allocation-bounded construction of frozen NGAS C1 inputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
import statistics
import time

import numpy as np
import torch

from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS
from rcias_ngas.bank.ngas_bank_v1 import build_all_banks
from rcias_ngas.csg.critical_mapping import CRITICAL_FEATURE_NAMES
from rcias_ngas.csg.critical_sync import CRITICAL_CATEGORIES, relation_category
from rcias_ngas.csg.revised_features import (
    BASE_NODE_DIM, BASE_NODE_NAMES, BOUNDARY_DIM, EDGE_FEATURE_DIM, NODE_DIM,
)
from rcias_ngas.csg.revised_schema import FAMILIES, OPERATORS, PROVENANCE_DIM, RULES
from rcias_ngas.evaluation.bks import content_hash


NODE_TYPES = ('OP', 'ISLAND', 'CONFIG', 'W_AGV', 'F_AGV',
              'W_EVENT', 'F_EVENT', 'RECONF_EVENT')
EDGE_TYPES = (
    'OP__PRECEDES__OP', 'OP__ELIGIBLE_ON__ISLAND', 'OP__ASSIGNED_TO__ISLAND',
    'OP__REQUIRES__CONFIG', 'ISLAND__SUPPORTS__CONFIG',
    'ISLAND__CURRENT_CONFIG__CONFIG', 'OP__PRODUCT_NEXT__OP',
    'OP__ISLAND_NEXT__OP', 'W_EVENT__W_NEXT__W_EVENT',
    'F_EVENT__F_NEXT__F_EVENT', 'W_EVENT__ENABLES__OP',
    'F_EVENT__ENABLES__OP', 'RECONF_EVENT__ENABLES__OP',
    'W_EVENT__EXECUTED_BY__W_AGV', 'F_EVENT__EXECUTED_BY__F_AGV',
    'RECONF_EVENT__OCCURS_ON__ISLAND', 'OP__TRIGGERS_RECONF__RECONF_EVENT',
    'OP__RELEASES_WORKPIECE_TO__W_EVENT',
    'RECONF_EVENT__FROM_CONFIG__CONFIG', 'RECONF_EVENT__TO_CONFIG__CONFIG',
)
CRITICAL_OFFSET = BASE_NODE_DIM
ZERO_SLACK_COLUMN = CRITICAL_OFFSET + CRITICAL_FEATURE_NAMES.index('zero_slack')
EPSILON = 1e-9

# Compact event kinds.
SOURCE, SINK, RECONFIG, OPERATION, W_EMPTY, W_LOADED, F_OUTBOUND, F_RETURN, IDLE = range(9)
EVENT_KIND_NAME = (
    'SOURCE', 'SINK', 'RECONFIGURATION', 'OPERATION', 'W_EMPTY', 'W_LOADED',
    'F_OUTBOUND', 'F_RETURN', 'UNEXPLAINED_IDLE',
)
AUTOMATIC_CATEGORY = {
    RECONFIG: CRITICAL_CATEGORIES.index('RECONFIGURATION'),
    W_EMPTY: CRITICAL_CATEGORIES.index('W_LOGISTICS'),
    W_LOADED: CRITICAL_CATEGORIES.index('W_LOGISTICS'),
    F_OUTBOUND: CRITICAL_CATEGORIES.index('F_LOGISTICS'),
    F_RETURN: CRITICAL_CATEGORIES.index('F_LOGISTICS'),
    IDLE: CRITICAL_CATEGORIES.index('REALIZED_IDLE'),
}
AUTOMATIC_CATEGORY_BY_KIND = np.asarray(
    [AUTOMATIC_CATEGORY.get(kind, -1) for kind in range(len(EVENT_KIND_NAME))],
    dtype=np.int8)


def _positive_mean(values) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    return statistics.mean(positive) if positive else 1.0


@dataclass(frozen=True)
class StaticInstanceContext:
    instance: object
    operations: tuple[str, ...]
    operation_index: dict[str, int]
    islands: tuple[str, ...]
    island_index: dict[str, int]
    configurations: tuple[str, ...]
    config_index: dict[str, int]
    w_resources: tuple[str, ...]
    w_index: dict[str, int]
    f_resources: tuple[str, ...]
    f_index: dict[str, int]
    resource_index: dict[str, int]
    mean_processing: float
    mean_travel: float
    mean_reconfiguration: float
    static_node_count: int
    max_neural_nodes: int
    max_forward_edges: int
    max_event_nodes: int
    max_event_arcs: int
    static_precedence: tuple[tuple[int, int], ...]
    static_eligibility: tuple[tuple[int, int], ...]
    static_requires: tuple[tuple[int, int], ...]
    static_supports: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, instance):
        operations = tuple(sorted(instance.operations))
        islands = tuple(sorted(instance.islands))
        configurations = tuple(sorted(instance.configurations))
        w_resources = tuple(sorted(instance.agvs_w))
        f_resources = tuple(sorted(instance.agvs_f))
        op_index = {value: index for index, value in enumerate(operations)}
        island_index = {value: index for index, value in enumerate(islands)}
        config_index = {value: index for index, value in enumerate(configurations)}
        w_index = {value: index for index, value in enumerate(w_resources)}
        f_index = {value: index for index, value in enumerate(f_resources)}
        resources = tuple(sorted({*islands, *w_resources, *f_resources}))
        resource_index = {value: index for index, value in enumerate(resources)}
        processing = [
            instance.processing_time[operation, island]
            for operation in instance.operations
            for island in instance.operation_data[operation].eligible_islands
        ]
        travel = [*instance.w_loaded_time.values(), *instance.w_empty_time.values(),
                  *instance.f_outbound_time.values(), *instance.f_return_time.values()]
        precedence = tuple(sorted(
            (op_index[source], op_index[target])
            for product in instance.products
            for source, target in instance.product_data[product].precedence))
        eligibility = tuple(sorted(
            (op_index[operation], island_index[island])
            for operation in instance.operations
            for island in instance.operation_data[operation].eligible_islands))
        requires = tuple(sorted(
            (op_index[operation], config_index[instance.operation_data[operation].required_config])
            for operation in instance.operations))
        supports = tuple(sorted(
            (island_index[island], config_index[config])
            for island in instance.islands
            for config in instance.island_data[island].supported_configs))
        static_count = (len(operations) + len(islands) + len(configurations)
                        + len(w_resources) + len(f_resources))
        n = len(operations)
        # Forward construction has the four static relation families plus at
        # most 16 operation-indexed dynamic families and one edge per island.
        max_forward_edges = (
            len(precedence) + len(eligibility) + len(supports)
            + len(islands) + 16 * n)
        # The event DAG has at most 2 + 6n base events and one idle event for
        # every non-terminal base event. Base arcs are bounded by 11n + |P|;
        # idle anchors add at most one copy of those arcs plus 6n idle exits.
        max_event_nodes = 12 * n + 2
        max_event_arcs = 28 * n + 2 * len(precedence)
        return cls(
            instance, operations, op_index, islands, island_index,
            configurations, config_index, w_resources, w_index, f_resources, f_index,
            resource_index, _positive_mean(processing), _positive_mean(travel),
            _positive_mean(instance.reconfiguration_time.values()), static_count,
            static_count + 3 * n, max_forward_edges,
            max_event_nodes, max_event_arcs,
            precedence, eligibility, requires, supports,
        )


class ReusableWorkspace:
    def __init__(self, context: StaticInstanceContext):
        n = len(context.operations)
        nodes = context.max_neural_nodes
        edges = context.max_forward_edges
        events = context.max_event_nodes
        arcs = context.max_event_arcs
        targets = len(SIZE_FRACTIONS) * len(RULES)
        actions = targets * len(NGAS_REPAIR_IDS)
        self.resize_events = 0
        self.capacities = {
            'neural_nodes': nodes, 'forward_edges': edges, 'event_nodes': events,
            'event_arcs': arcs, 'targets': targets, 'actions': actions,
        }
        # Keep feature arithmetic in float64, matching the reference Python
        # feature path, and cast once into the reusable model input buffer.
        self.x = np.zeros((nodes, NODE_DIM), np.float64)
        self.tensor_x = np.zeros((nodes, NODE_DIM), np.float32)
        self.types = np.zeros(nodes, np.int64)
        self.critical_mask = np.zeros(nodes, np.bool_)
        self.forward_source = np.zeros(edges, np.int64)
        self.forward_target = np.zeros(edges, np.int64)
        self.forward_relation = np.zeros(edges, np.int64)
        self.forward_features = np.zeros((edges, EDGE_FEATURE_DIM), np.float32)
        self.edge_index = np.zeros((2, 2 * edges), np.int64)
        self.relations = np.zeros(2 * edges, np.int64)
        self.edge_features = np.zeros((2 * edges, EDGE_FEATURE_DIM), np.float32)
        self.operation_nodes = np.arange(n, dtype=np.int64)
        self.target_membership = np.zeros((targets, n), np.float32)
        self.target_node_membership = np.zeros((targets, nodes), np.bool_)
        self.target_provenance = np.zeros((targets, PROVENANCE_DIM), np.float32)
        self.target_boundary = np.zeros((targets, nodes), np.float32)
        self.target_boundary_stats = np.zeros((targets, BOUNDARY_DIM), np.float32)
        self.target_overlap = np.zeros((targets, 1), np.float32)
        self.membership = np.zeros((actions, n), np.float32)
        self.provenance = np.zeros((actions, PROVENANCE_DIM), np.float32)
        self.sizes = np.zeros(actions, np.int64)
        self.repairs = np.zeros(actions, np.int64)
        self.boundary = np.zeros((actions, nodes), np.float32)
        self.boundary_stats = np.zeros((actions, BOUNDARY_DIM), np.float32)
        self.overlap = np.zeros((actions, 1), np.float32)
        self.edge_selected_source = np.zeros(edges, np.bool_)
        self.edge_selected_target = np.zeros(edges, np.bool_)
        self.edge_crosses = np.zeros(edges, np.bool_)
        self.w_node_by_op = np.full(n, -1, np.int64)
        self.f_node_by_op = np.full(n, -1, np.int64)
        self.reconf_node_by_op = np.full(n, -1, np.int64)
        self.previous_island_op = np.full(n, -1, np.int64)
        self.reconf_source_config = np.full(n, -1, np.int64)
        self.event_keys = [''] * events
        self.event_kind = np.zeros(events, np.int8)
        self.event_start = np.zeros(events, np.float64)
        self.event_end = np.zeros(events, np.float64)
        self.event_operation = np.full(events, -1, np.int64)
        self.event_resource = np.full(events, -1, np.int64)
        self.event_neural = np.full(events, -1, np.int64)
        self.event_idle_successor = np.full(events, -1, np.int64)
        self.operation_event = np.full(n, -1, np.int64)
        self.arc_source = np.zeros(arcs, np.int64)
        self.arc_target = np.zeros(arcs, np.int64)
        self.arc_category = np.zeros(arcs, np.int8)
        self.arc_margin = np.zeros(arcs, np.float64)
        self.arc_active_margin = np.zeros(arcs, np.float64)
        self.arc_critical = np.zeros(arcs, np.bool_)
        self.in_offsets = np.zeros(events + 1, np.int64)
        self.in_sources = np.zeros(arcs, np.int64)
        self.out_offsets = np.zeros(events + 1, np.int64)
        self.out_targets = np.zeros(arcs, np.int64)
        self.topological = np.zeros(events, np.int64)
        self.earliest_start = np.zeros(events, np.float64)
        self.earliest_finish = np.zeros(events, np.float64)
        self.distance_sink = np.zeros(events, np.float64)
        self.latest_start = np.zeros(events, np.float64)
        self.event_slack = np.zeros(events, np.float64)
        self.zero_slack = np.zeros(events, np.bool_)
        self.critical_degree = np.zeros(events, np.int64)
        self.category_mask = np.zeros(events, np.uint16)
        self.min_margin = np.zeros(events, np.float64)
        self.operation_node_count = np.zeros(n, np.int64)
        self.operation_edge_count = np.zeros(n, np.int64)
        self.operation_resource = np.zeros((n, len(context.resource_index)), np.bool_)
        self.operation_slack = np.zeros(n, np.float64)
        self.agg_count = np.zeros(nodes, np.int64)
        self.agg_finite = np.zeros(nodes, np.int64)
        self.agg_zero = np.zeros(nodes, np.int64)
        self.agg_degree = np.zeros(nodes, np.int64)
        self.agg_categories = np.zeros(nodes, np.uint16)
        self.agg_min_slack = np.zeros(nodes, np.float64)
        self.agg_min_margin = np.zeros(nodes, np.float64)
        self.agg_margin_defined = np.zeros(nodes, np.bool_)
        self.agg_unreachable = np.zeros(nodes, np.int64)


@dataclass(frozen=True)
class CompactBuildResult:
    actions: tuple[JointAction, ...]
    cpu_batch: dict[str, torch.Tensor]
    critical_signature: str
    dominant_bottleneck: str | None
    bank_summary: dict
    node_count: int
    forward_edge_count: int
    target_count: int
    component_seconds: dict[str, float]


class _BankAnalysis:
    def __init__(self, context, workspace, ranked_operations):
        self.context = context
        self.workspace = workspace
        self.ranked_operations = ranked_operations

    def operation_slack(self, operation):
        value = self.workspace.operation_slack[self.context.operation_index[operation]]
        return None if not math.isfinite(value) else float(value)


class CompactStateBuilder:
    def __init__(self, instance):
        self.context = StaticInstanceContext.build(instance)
        self.workspace = ReusableWorkspace(self.context)

    def capacity_audit(self) -> dict:
        """Return derived bounds and allocated mutable-buffer shapes."""
        c, w = self.context, self.workspace
        n = len(c.operations)
        resource_nodes = (
            len(c.islands) + len(c.configurations)
            + len(c.w_resources) + len(c.f_resources))
        return {
            'schema': 'ngas-compact-capacity-audit-v1',
            'operation_nodes': n,
            'resource_and_configuration_nodes': resource_nodes,
            'required_upper_bounds': dict(w.capacities),
            'allocated_capacities': dict(w.capacities),
            'membership_matrix_shapes': {
                'target_by_operation': list(w.target_membership.shape),
                'target_by_node': list(w.target_node_membership.shape),
                'action_by_operation': list(w.membership.shape),
                'action_by_node_boundary': list(w.boundary.shape),
            },
            'derivations': {
                'targets': 'len(SIZE_FRACTIONS) * len(RULES)',
                'actions': 'targets * len(NGAS_REPAIR_IDS)',
                'neural_nodes': 'static_nodes + 3 * num_operations',
                'forward_edges': (
                    'precedence + eligibility + supports + islands '
                    '+ 16 * num_operations'),
                'event_nodes': '12 * num_operations + 2',
                'event_arcs': '28 * num_operations + 2 * precedence_edges',
            },
            'resize_events': w.resize_events,
            'silent_truncation_allowed': False,
        }

    @staticmethod
    def _position(index, length):
        return float(index) / max(length - 1, 1)

    def _neural_nodes(self, schedule):
        c, w = self.context, self.workspace
        n = len(c.operations)
        w.w_node_by_op.fill(-1)
        w.f_node_by_op.fill(-1)
        w.reconf_node_by_op.fill(-1)
        w.previous_island_op.fill(-1)
        w.reconf_source_config.fill(-1)
        w_tasks = sorted(
            (task for tasks in schedule.w_timelines.values() for task in tasks),
            key=lambda task: task.task_id)
        f_tasks = sorted(
            (task for tasks in schedule.f_timelines.values() for task in tasks),
            key=lambda task: task.task_id)
        positive_reconfig = sorted(
            operation for operation in c.operations
            if (schedule.operation_schedules[operation].reconfiguration_end
                - schedule.operation_schedules[operation].reconfiguration_start) > EPSILON)
        op_offset = 0
        island_offset = n
        config_offset = island_offset + len(c.islands)
        w_resource_offset = config_offset + len(c.configurations)
        f_resource_offset = w_resource_offset + len(c.w_resources)
        w_event_offset = f_resource_offset + len(c.f_resources)
        f_event_offset = w_event_offset + len(w_tasks)
        reconf_offset = f_event_offset + len(f_tasks)
        node_count = reconf_offset + len(positive_reconfig)
        if node_count > c.max_neural_nodes:
            raise RuntimeError('Compact neural-node capacity exceeded')
        w.x[:node_count].fill(0)
        w.types[:node_count].fill(0)
        w.critical_mask[:node_count].fill(False)
        w.types[island_offset:config_offset] = 1
        w.types[config_offset:w_resource_offset] = 2
        w.types[w_resource_offset:f_resource_offset] = 3
        w.types[f_resource_offset:w_event_offset] = 4
        w.types[w_event_offset:f_event_offset] = 5
        w.types[f_event_offset:reconf_offset] = 6
        w.types[reconf_offset:node_count] = 7

        for index, task in enumerate(w_tasks):
            w.w_node_by_op[c.operation_index[task.operation_id]] = w_event_offset + index
        for index, task in enumerate(f_tasks):
            w.f_node_by_op[c.operation_index[task.operation_id]] = f_event_offset + index
        for index, operation in enumerate(positive_reconfig):
            w.reconf_node_by_op[c.operation_index[operation]] = reconf_offset + index

        product_position = np.zeros(n, np.int64)
        product_length = np.ones(n, np.int64)
        island_position = np.zeros(n, np.int64)
        island_length = np.ones(n, np.int64)
        w_position = np.zeros(n, np.int64)
        w_length = np.ones(n, np.int64)
        f_position = np.zeros(n, np.int64)
        f_length = np.ones(n, np.int64)
        for sequence in schedule.product_sequences.values():
            for position, operation in enumerate(sequence):
                index = c.operation_index[operation]
                product_position[index], product_length[index] = position, len(sequence)
        for island in c.islands:
            sequence = schedule.island_timelines[island]
            previous_config = c.instance.island_data[island].initial_config
            previous_index = -1
            for position, operation in enumerate(sequence):
                index = c.operation_index[operation]
                island_position[index], island_length[index] = position, len(sequence)
                w.previous_island_op[index] = previous_index
                w.reconf_source_config[index] = c.config_index[previous_config]
                previous_index = index
                previous_config = schedule.operation_schedules[operation].config_id
        for tasks in schedule.w_timelines.values():
            for position, task in enumerate(tasks):
                index = c.operation_index[task.operation_id]
                w_position[index], w_length[index] = position, len(tasks)
        for tasks in schedule.f_timelines.values():
            for position, task in enumerate(tasks):
                index = c.operation_index[task.operation_id]
                f_position[index], f_length[index] = position, len(tasks)

        makespan = max(row.completion_time for row in schedule.operation_schedules.values())
        time_scale = max(float(makespan), 1.)
        island_counts = np.asarray(
            [len(schedule.island_timelines[value]) for value in c.islands], dtype=np.float64)
        max_island_count = max(float(island_counts.max(initial=0)), 1.)
        island_processing = np.asarray([
            sum(schedule.operation_schedules[operation].processing_time
                for operation in schedule.island_timelines[island])
            for island in c.islands
        ], dtype=np.float64)
        max_island_processing = max(float(island_processing.max(initial=0)), 1.)

        op_columns = {name: index for index, name in enumerate(BASE_NODE_NAMES['OP'])}
        bank_features = {}
        for row, operation in enumerate(c.operations):
            record = schedule.operation_schedules[operation]
            data = c.instance.operation_data[operation]
            index = c.operation_index[operation]
            sequence = schedule.island_timelines[record.island_id]
            next_reconfiguration = 0.
            pos = int(island_position[index])
            if pos + 1 < len(sequence):
                following = schedule.operation_schedules[sequence[pos + 1]]
                next_reconfiguration = following.reconfiguration_end - following.reconfiguration_start
            reconfiguration = record.reconfiguration_end - record.reconfiguration_start
            values = {
                'processing_time_normalized': record.processing_time / c.mean_processing,
                'mean_eligible_processing_time_normalized':
                    statistics.mean(data.processing_time.values()) / c.mean_processing,
                'eligible_island_fraction': len(data.eligible_islands) / len(c.islands),
                'start_time_normalized': record.start_time / time_scale,
                'completion_time_normalized': record.completion_time / time_scale,
                'w_delay_normalized': max(0., record.w_ready_time - record.product_ready_time) / time_scale,
                'f_delay_normalized': max(0., record.f_ready_time - record.product_ready_time) / time_scale,
                'synchronization_wait_normalized': max(0., record.start_time - max(
                    record.product_ready_time, record.config_ready_time,
                    record.w_ready_time, record.f_ready_time)) / time_scale,
                'local_reconfiguration_normalized':
                    (reconfiguration + next_reconfiguration) / c.mean_reconfiguration,
                'island_relative_load': island_counts[c.island_index[record.island_id]] / max_island_count,
                'product_position_normalized': self._position(
                    int(product_position[index]), int(product_length[index])),
                'island_position_normalized': self._position(pos, int(island_length[index])),
                'w_chain_position_normalized': self._position(
                    int(w_position[index]), int(w_length[index])) if w.w_node_by_op[index] >= 0 else 0.,
                'f_chain_position_normalized': self._position(
                    int(f_position[index]), int(f_length[index])) if w.f_node_by_op[index] >= 0 else 0.,
                'has_w_event': float(w.w_node_by_op[index] >= 0),
                'has_f_event': float(w.f_node_by_op[index] >= 0),
                'predecessor_count_normalized': len(c.instance.predecessors[operation]) / n,
                'successor_count_normalized': len(c.instance.successors[operation]) / n,
            }
            for name, value in values.items():
                w.x[row, op_columns[name]] = value
            bank_features[operation] = {
                'assigned_island': record.island_id,
                'W_waiting_or_delay_contribution': max(
                    0., record.w_ready_time - record.product_ready_time),
                'F_waiting_or_delay_contribution': max(
                    0., record.f_ready_time - record.product_ready_time),
                'operation_slack': max(0., makespan - record.completion_time),
            }

        island_columns = {name: index for index, name in enumerate(BASE_NODE_NAMES['ISLAND'])}
        for local, island in enumerate(c.islands):
            sequence = schedule.island_timelines[island]
            reconfiguration_time = sum(
                schedule.operation_schedules[operation].reconfiguration_end
                - schedule.operation_schedules[operation].reconfiguration_start
                for operation in sequence)
            reconfiguration_count = sum(
                schedule.operation_schedules[operation].reconfiguration_end
                - schedule.operation_schedules[operation].reconfiguration_start > EPSILON
                for operation in sequence)
            processing = island_processing[local]
            busy = processing + reconfiguration_time
            last = 0. if not sequence else schedule.operation_schedules[sequence[-1]].completion_time
            capability = len(c.instance.island_data[island].supported_configs) / len(c.configurations)
            values = (
                processing / time_scale, processing / max_island_processing, len(sequence) / n,
                capability, capability, reconfiguration_count / n,
                reconfiguration_time / time_scale, busy / time_scale,
                max(0., time_scale - busy) / time_scale, last / time_scale,
            )
            w.x[island_offset + local, :len(values)] = values

        supporting = Counter(config for island in c.instance.islands
                             for config in c.instance.island_data[island].supported_configs)
        required = Counter(c.instance.operation_data[operation].required_config
                           for operation in c.instance.operations)
        for local, config in enumerate(c.configurations):
            w.x[config_offset + local, :2] = (
                supporting[config] / len(c.islands), required[config] / n)

        w_busy = np.asarray([
            sum(task.arrival_time - task.empty_start for task in schedule.w_timelines[value])
            for value in c.w_resources], dtype=np.float64)
        max_w_busy = max(float(w_busy.max(initial=0)), 1.)
        for local, resource_id in enumerate(c.w_resources):
            tasks = schedule.w_timelines[resource_id]
            travel = sum(task.empty_travel_time + task.loaded_travel_time for task in tasks)
            waiting = sum(max(0., task.loaded_start - task.empty_arrival) for task in tasks)
            last = 0. if not tasks else tasks[-1].arrival_time
            w.x[w_resource_offset + local, :6] = (
                len(tasks) / n, w_busy[local] / time_scale, travel / c.mean_travel,
                waiting / time_scale, w_busy[local] / max_w_busy, last / time_scale)
        f_busy = np.asarray([
            sum(task.return_wh - task.departure_wh for task in schedule.f_timelines[value])
            for value in c.f_resources], dtype=np.float64)
        max_f_busy = max(float(f_busy.max(initial=0)), 1.)
        for local, resource_id in enumerate(c.f_resources):
            tasks = schedule.f_timelines[resource_id]
            travel = sum(task.outbound_time + task.return_time for task in tasks)
            last = 0. if not tasks else tasks[-1].return_wh
            w.x[f_resource_offset + local, :5] = (
                len(tasks) / n, f_busy[local] / time_scale, travel / c.mean_travel,
                f_busy[local] / max_f_busy, last / time_scale)
        for local, task in enumerate(w_tasks):
            position = int(w_position[c.operation_index[task.operation_id]])
            length = int(w_length[c.operation_index[task.operation_id]])
            w.x[w_event_offset + local, :7] = (
                (task.arrival_time - task.empty_start) / time_scale,
                task.empty_travel_time / c.mean_travel,
                max(0., task.loaded_start - task.empty_arrival) / time_scale,
                task.loaded_travel_time / c.mean_travel,
                self._position(position, length), float(task.pickup == 'WH'),
                float(task.predecessor_op is None))
        for local, task in enumerate(f_tasks):
            position = int(f_position[c.operation_index[task.operation_id]])
            length = int(f_length[c.operation_index[task.operation_id]])
            w.x[f_event_offset + local, :4] = (
                (task.return_wh - task.departure_wh) / time_scale,
                task.outbound_time / c.mean_travel, task.return_time / c.mean_travel,
                self._position(position, length))
        for local, operation in enumerate(positive_reconfig):
            index = c.operation_index[operation]
            record = schedule.operation_schedules[operation]
            duration = record.reconfiguration_end - record.reconfiguration_start
            w.x[reconf_offset + local, :3] = (
                duration / c.mean_reconfiguration,
                self._position(int(island_position[index]), int(island_length[index])),
                float(w.previous_island_op[index] < 0))
        np.log1p(w.x[:node_count, :BASE_NODE_DIM],
                 out=w.x[:node_count, :BASE_NODE_DIM])
        return {
            'node_count': node_count, 'makespan': float(makespan),
            'offsets': (op_offset, island_offset, config_offset, w_resource_offset,
                        f_resource_offset, w_event_offset, f_event_offset, reconf_offset),
            'w_tasks': w_tasks, 'f_tasks': f_tasks,
            'positive_reconfig': positive_reconfig,
            'bank_features': bank_features,
            'time_scale': time_scale,
        }

    def _add_event(self, count, key, kind, start, end, operation=-1, resource=-1,
                   neural=-1, idle_successor=-1):
        w = self.workspace
        if count >= len(w.event_kind):
            raise RuntimeError('Compact event-node capacity exceeded')
        w.event_keys[count] = key
        w.event_kind[count] = kind
        w.event_start[count] = start
        w.event_end[count] = end
        w.event_operation[count] = operation
        w.event_resource[count] = resource
        w.event_neural[count] = neural
        w.event_idle_successor[count] = idle_successor
        return count + 1

    def _add_arc(self, count, source, target, relation):
        w = self.workspace
        if count >= len(w.arc_source):
            raise RuntimeError('Compact event-arc capacity exceeded')
        w.arc_source[count], w.arc_target[count] = source, target
        w.arc_category[count] = CRITICAL_CATEGORIES.index(relation_category(relation))
        return count + 1

    def _csr(self, node_count, arc_count):
        w = self.workspace
        w.in_offsets[:node_count + 1].fill(0)
        w.out_offsets[:node_count + 1].fill(0)
        np.add.at(w.in_offsets, w.arc_target[:arc_count] + 1, 1)
        np.add.at(w.out_offsets, w.arc_source[:arc_count] + 1, 1)
        np.cumsum(w.in_offsets[:node_count + 1], out=w.in_offsets[:node_count + 1])
        np.cumsum(w.out_offsets[:node_count + 1], out=w.out_offsets[:node_count + 1])
        in_cursor = w.in_offsets[:node_count].copy()
        out_cursor = w.out_offsets[:node_count].copy()
        for index in range(arc_count):
            source, target = int(w.arc_source[index]), int(w.arc_target[index])
            w.in_sources[in_cursor[target]] = source
            in_cursor[target] += 1
            w.out_targets[out_cursor[source]] = target
            out_cursor[source] += 1

    def _compact_critical(self, schedule, neural):
        c, w = self.context, self.workspace
        n = len(c.operations)
        _, _, _, _, _, _, _, _ = neural['offsets']
        event_by_key = {}
        w.operation_event.fill(-1)
        count = 0
        count = self._add_event(count, 'S', SOURCE, 0., 0.)
        count = self._add_event(count, 'E', SINK, neural['makespan'], neural['makespan'])
        for operation, record in schedule.operation_schedules.items():
            op = c.operation_index[operation]
            count = self._add_event(
                count, 'RECONFIG:' + operation, RECONFIG,
                record.reconfiguration_start, record.reconfiguration_end, op,
                c.resource_index[record.island_id],
                int(w.reconf_node_by_op[op] if w.reconf_node_by_op[op] >= 0 else op))
            count = self._add_event(
                count, 'OP:' + operation, OPERATION, record.start_time,
                record.completion_time, op, c.resource_index[record.island_id], op)
            w.operation_event[op] = count - 1
        for resource_id, tasks in schedule.w_timelines.items():
            for task in tasks:
                op = c.operation_index[task.operation_id]
                count = self._add_event(
                    count, 'W_EMPTY:' + task.task_id, W_EMPTY,
                    task.empty_start, task.empty_arrival, op, c.resource_index[resource_id],
                    int(w.w_node_by_op[op]))
                count = self._add_event(
                    count, 'W_LOADED:' + task.task_id, W_LOADED,
                    task.loaded_start, task.arrival_time, op, c.resource_index[resource_id],
                    int(w.w_node_by_op[op]))
        for resource_id, tasks in schedule.f_timelines.items():
            for task in tasks:
                op = c.operation_index[task.operation_id]
                count = self._add_event(
                    count, 'F_OUTBOUND:' + task.task_id, F_OUTBOUND,
                    task.departure_wh, task.arrival_island, op, c.resource_index[resource_id],
                    int(w.f_node_by_op[op]))
                count = self._add_event(
                    count, 'F_RETURN:' + task.task_id, F_RETURN,
                    task.arrival_island, task.return_wh, op, c.resource_index[resource_id],
                    int(w.f_node_by_op[op]))
        for index in range(count):
            event_by_key[w.event_keys[index]] = index
        arcs = 0
        for island_id, sequence in schedule.island_timelines.items():
            previous = None
            for operation in sequence:
                arcs = self._add_arc(
                    arcs, event_by_key['OP:' + previous] if previous else 0,
                    event_by_key['RECONFIG:' + operation], 'ISLAND_ORDER')
                arcs = self._add_arc(
                    arcs, event_by_key['RECONFIG:' + operation],
                    event_by_key['OP:' + operation], 'CONFIGURATION_READY')
                previous = operation
        for operation, record in schedule.operation_schedules.items():
            operation_event = event_by_key['OP:' + operation]
            arcs = self._add_arc(arcs, operation_event, 1, 'MAKESPAN_COMPLETION')
            if record.product_predecessor is not None:
                arcs = self._add_arc(
                    arcs, event_by_key['OP:' + record.product_predecessor], operation_event,
                    'REALIZED_PRODUCT_CHAIN')
            for predecessor in c.instance.predecessors[operation]:
                arcs = self._add_arc(
                    arcs, event_by_key['OP:' + predecessor], operation_event,
                    'TECHNOLOGICAL_PRECEDENCE')
        for resource_id, tasks in schedule.w_timelines.items():
            previous_loaded = None
            for task in tasks:
                empty = event_by_key['W_EMPTY:' + task.task_id]
                loaded = event_by_key['W_LOADED:' + task.task_id]
                arcs = self._add_arc(arcs, previous_loaded if previous_loaded is not None else 0,
                                     empty, 'W_RESOURCE_ORDER')
                arcs = self._add_arc(arcs, empty, loaded, 'W_EMPTY_BEFORE_LOADED')
                if task.predecessor_op is not None:
                    arcs = self._add_arc(
                        arcs, event_by_key['OP:' + task.predecessor_op], loaded,
                        'WORKPIECE_RELEASE')
                else:
                    arcs = self._add_arc(arcs, 0, loaded, 'WAREHOUSE_RELEASE')
                arcs = self._add_arc(
                    arcs, loaded, event_by_key['OP:' + task.operation_id], 'W_ARRIVAL_READY')
                previous_loaded = loaded
        for resource_id, tasks in schedule.f_timelines.items():
            previous_return = None
            for task in tasks:
                outbound = event_by_key['F_OUTBOUND:' + task.task_id]
                returned = event_by_key['F_RETURN:' + task.task_id]
                arcs = self._add_arc(arcs, previous_return if previous_return is not None else 0,
                                     outbound, 'F_RESOURCE_ORDER')
                arcs = self._add_arc(arcs, outbound, returned, 'F_OUTBOUND_BEFORE_RETURN')
                arcs = self._add_arc(
                    arcs, outbound, event_by_key['OP:' + task.operation_id], 'F_ARRIVAL_READY')
                previous_return = returned

        base_count, base_arcs = count, arcs
        self._csr(base_count, base_arcs)
        for node in range(base_count):
            if node in (0, 1):
                continue
            begin, end = w.in_offsets[node], w.in_offsets[node + 1]
            sources = w.in_sources[begin:end]
            latest = max((w.event_end[int(source)] for source in sources), default=0.)
            if w.event_start[node] - latest <= 1e-8:
                continue
            idle = count
            count = self._add_event(
                count, 'IDLE:' + w.event_keys[node], IDLE, latest,
                w.event_start[node], neural=int(w.event_neural[node]), idle_successor=node)
            anchors = sorted(
                (int(source) for source in sources
                 if math.isclose(w.event_end[int(source)], latest, abs_tol=1e-8)),
                key=lambda value: w.event_keys[value]) or [0]
            for source in anchors:
                arcs = self._add_arc(arcs, source, idle, 'REALIZED_IDLE_AFTER')
            arcs = self._add_arc(arcs, idle, node, 'REALIZED_IDLE_BEFORE')
        self._csr(count, arcs)

        indegree = np.diff(w.in_offsets[:count + 1]).copy()
        ready = [(w.event_keys[index], index) for index in range(count) if indegree[index] == 0]
        heapq.heapify(ready)
        order_count = 0
        while ready:
            _, node = heapq.heappop(ready)
            w.topological[order_count] = node
            order_count += 1
            for position in range(w.out_offsets[node], w.out_offsets[node + 1]):
                target = int(w.out_targets[position])
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, (w.event_keys[target], target))
        if order_count != count:
            raise ValueError('Compact generalized event graph is cyclic')

        tolerance = max(1e-8, 1e-10 * max(1., neural['makespan']))
        for position in range(count):
            node = int(w.topological[position])
            begin, end = w.in_offsets[node], w.in_offsets[node + 1]
            earliest = max((w.earliest_finish[int(source)]
                            for source in w.in_sources[begin:end]), default=0.)
            finish = earliest + w.event_end[node] - w.event_start[node]
            if abs(finish - w.event_end[node]) > tolerance:
                raise ValueError(f'Compact event DAG does not reproduce {w.event_keys[node]}')
            w.earliest_start[node], w.earliest_finish[node] = earliest, finish
        w.distance_sink[:count].fill(math.inf)
        for position in range(count - 1, -1, -1):
            node = int(w.topological[position])
            if node == 1:
                w.distance_sink[node] = 0.
                continue
            begin, end = w.out_offsets[node], w.out_offsets[node + 1]
            candidates = [
                w.event_end[int(target)] - w.event_start[int(target)]
                + w.distance_sink[int(target)]
                for target in w.out_targets[begin:end]
                if math.isfinite(w.distance_sink[int(target)])
            ]
            if candidates:
                w.distance_sink[node] = max(candidates)
        duration = w.event_end[:count] - w.event_start[:count]
        finite = np.isfinite(w.distance_sink[:count])
        w.latest_start[:count] = math.inf
        w.latest_start[:count][finite] = (
            neural['makespan'] - duration[finite] - w.distance_sink[:count][finite])
        w.zero_slack[:count] = finite & (
            np.abs(w.latest_start[:count] - w.earliest_start[:count]) <= tolerance)
        w.critical_degree[:count].fill(0)
        w.category_mask[:count].fill(0)
        w.min_margin[:count].fill(math.inf)
        w.operation_node_count.fill(0)
        w.operation_edge_count.fill(0)
        w.operation_resource.fill(False)

        zero_nodes = np.flatnonzero(w.zero_slack[:count])
        automatic = AUTOMATIC_CATEGORY_BY_KIND[w.event_kind[zero_nodes]]
        automatic_nodes = zero_nodes[automatic >= 0]
        automatic_categories = automatic[automatic >= 0]
        np.bitwise_or.at(
            w.category_mask, automatic_nodes,
            np.left_shift(np.uint16(1), automatic_categories.astype(np.uint16)))
        zero_operations = w.event_operation[zero_nodes]
        valid = zero_operations >= 0
        zero_operations = zero_operations[valid]
        np.add.at(w.operation_node_count, zero_operations, 1)
        zero_resources = w.event_resource[zero_nodes][valid]
        with_resource = zero_resources >= 0
        w.operation_resource[
            zero_operations[with_resource], zero_resources[with_resource]] = True

        sources = w.arc_source[:arcs]
        targets = w.arc_target[:arcs]
        np.subtract(w.event_start[targets], w.event_end[sources],
                    out=w.arc_margin[:arcs])
        np.maximum(w.arc_margin[:arcs], 0., out=w.arc_active_margin[:arcs])
        np.minimum.at(w.min_margin, sources, w.arc_active_margin[:arcs])
        np.minimum.at(w.min_margin, targets, w.arc_active_margin[:arcs])
        np.logical_and(w.zero_slack[sources], w.zero_slack[targets],
                       out=w.arc_critical[:arcs])
        w.arc_critical[:arcs] &= np.abs(w.arc_margin[:arcs]) <= tolerance
        critical_indices = np.flatnonzero(w.arc_critical[:arcs])
        critical_sources = sources[critical_indices]
        critical_targets = targets[critical_indices]
        critical_categories = w.arc_category[critical_indices]
        category_counts = np.bincount(
            critical_categories, minlength=len(CRITICAL_CATEGORIES))
        category_masks = np.left_shift(
            np.uint16(1), critical_categories.astype(np.uint16))
        np.bitwise_or.at(w.category_mask, critical_sources, category_masks)
        np.bitwise_or.at(w.category_mask, critical_targets, category_masks)
        np.add.at(w.critical_degree, critical_sources, 1)
        np.add.at(w.critical_degree, critical_targets, 1)
        source_operations = w.event_operation[critical_sources]
        target_operations = w.event_operation[critical_targets]
        valid_source = source_operations >= 0
        np.add.at(w.operation_edge_count, source_operations[valid_source], 1)
        valid_target = (target_operations >= 0) & (target_operations != source_operations)
        np.add.at(w.operation_edge_count, target_operations[valid_target], 1)
        resource_counts = w.operation_resource.sum(axis=1)
        scores = 4 * w.operation_node_count + w.operation_edge_count + resource_counts
        has_reason = (w.operation_node_count > 0) | (w.operation_edge_count > 0)
        ranked = tuple(sorted(
            (operation for operation in c.operations
             if has_reason[c.operation_index[operation]]),
            key=lambda operation: (-scores[c.operation_index[operation]], operation)))
        operation_events = w.operation_event
        w.operation_slack[:] = (
            w.latest_start[operation_events] - w.earliest_start[operation_events])

        node_count = neural['node_count']
        w.agg_count[:node_count].fill(0)
        w.agg_finite[:node_count].fill(0)
        w.agg_zero[:node_count].fill(0)
        w.agg_degree[:node_count].fill(0)
        w.agg_categories[:node_count].fill(0)
        w.agg_min_slack[:node_count].fill(math.inf)
        w.agg_min_margin[:node_count].fill(math.inf)
        w.agg_margin_defined[:node_count].fill(False)
        w.agg_unreachable[:node_count].fill(0)
        maximum_degree = max(int(w.critical_degree[:count].max(initial=0)), 1)
        np.subtract(w.latest_start[:count], w.earliest_start[:count],
                    out=w.event_slack[:count])
        mapped_events = np.flatnonzero(w.event_neural[:count] >= 0)
        mapped_nodes = w.event_neural[mapped_events]
        np.add.at(w.agg_count, mapped_nodes, 1)
        mapped_finite = finite[mapped_events]
        finite_events = mapped_events[mapped_finite]
        finite_nodes = mapped_nodes[mapped_finite]
        np.add.at(w.agg_finite, finite_nodes, 1)
        np.minimum.at(w.agg_min_slack, finite_nodes, w.event_slack[finite_events])
        np.add.at(w.agg_unreachable, mapped_nodes[~mapped_finite], 1)
        np.add.at(w.agg_zero, mapped_nodes, w.zero_slack[mapped_events])
        np.add.at(w.agg_degree, mapped_nodes, w.critical_degree[mapped_events])
        np.bitwise_or.at(w.agg_categories, mapped_nodes, w.category_mask[mapped_events])
        margin_finite = np.isfinite(w.min_margin[mapped_events])
        margin_nodes = mapped_nodes[margin_finite]
        w.agg_margin_defined[margin_nodes] = True
        np.minimum.at(
            w.agg_min_margin, margin_nodes,
            w.min_margin[mapped_events[margin_finite]])
        critical = w.x[:node_count, CRITICAL_OFFSET:]
        critical.fill(0)
        eligible = (w.agg_count[:node_count] > 0) & np.isin(
            w.types[:node_count], (0, 5, 6, 7))
        nodes = np.flatnonzero(eligible)
        mapped_count = w.agg_count[nodes]
        finite_count = w.agg_finite[nodes]
        degree = w.agg_degree[nodes]
        zero_count = w.agg_zero[nodes]
        finite_nodes = finite_count > 0
        critical[nodes[finite_nodes], 0] = (
            w.agg_min_slack[nodes[finite_nodes]] / neural['time_scale'])
        critical[nodes, 1] = finite_nodes
        critical[nodes, 2] = zero_count > 0
        critical[nodes, 3] = degree > 0
        normalized_degree = degree / (maximum_degree * mapped_count)
        critical[nodes, 4] = normalized_degree
        for category in range(len(CRITICAL_CATEGORIES)):
            critical[nodes, 5 + category] = (
                w.agg_categories[nodes] & (1 << category)) != 0
        margin_nodes = nodes[w.agg_margin_defined[nodes]]
        critical[margin_nodes, 12] = (
            w.agg_min_margin[margin_nodes] / neural['time_scale'])
        critical[nodes, 13] = w.agg_margin_defined[nodes]
        critical[nodes, 14] = .5 * (
            zero_count / mapped_count + np.minimum(1., normalized_degree))
        critical[nodes, 15] = w.agg_unreachable[nodes] / mapped_count
        np.not_equal(w.x[:node_count, ZERO_SLACK_COLUMN], 0.,
                     out=w.critical_mask[:node_count])
        categories = {
            CRITICAL_CATEGORIES[index]: int(value)
            for index, value in enumerate(category_counts) if value
        }
        dominant = min(categories, key=lambda name: (-categories[name], name)) \
            if categories else None
        signature = content_hash({
            'critical_operations': list(ranked),
            'critical_edge_categories': dict(sorted(categories.items())),
        })
        return _BankAnalysis(c, w, ranked), signature, dominant

    def _forward_edges(self, schedule, neural):
        c, w = self.context, self.workspace
        n = len(c.operations)
        (_, island_offset, config_offset, w_resource_offset, f_resource_offset,
         _, _, _) = neural['offsets']
        makespan = neural['time_scale']
        count = 0

        def add(relation, source, target, source_end=None, target_start=None):
            nonlocal count
            if count >= len(w.forward_source):
                raise RuntimeError('Compact CSG edge capacity exceeded')
            w.forward_source[count], w.forward_target[count] = source, target
            w.forward_relation[count] = relation
            if source_end is None:
                w.forward_features[count] = 0.
            else:
                gap = float(target_start) - float(source_end)
                w.forward_features[count] = gap / makespan, float(abs(gap) <= EPSILON)
            count += 1

        records = schedule.operation_schedules
        for source, target in c.static_precedence:
            add(0, source, target, records[c.operations[source]].completion_time,
                records[c.operations[target]].start_time)
        for operation, island in c.static_eligibility:
            add(1, operation, island_offset + island)
        for operation, operation_id in enumerate(c.operations):
            add(2, operation, island_offset + c.island_index[records[operation_id].island_id])
        for operation, config in c.static_requires:
            add(3, operation, config_offset + config)
        for island, config in c.static_supports:
            add(4, island_offset + island, config_offset + config)
        for island_local, island in enumerate(c.islands):
            sequence = schedule.island_timelines[island]
            config = (c.instance.island_data[island].initial_config if not sequence
                      else records[sequence[-1]].config_id)
            add(5, island_offset + island_local, config_offset + c.config_index[config])
        pairs = sorted((source, target) for sequence in schedule.product_sequences.values()
                       for source, target in zip(sequence, sequence[1:]))
        for source, target in pairs:
            add(6, c.operation_index[source], c.operation_index[target],
                records[source].completion_time, records[target].start_time)
        pairs = sorted((source, target) for sequence in schedule.island_timelines.values()
                       for source, target in zip(sequence, sequence[1:]))
        for source, target in pairs:
            add(7, c.operation_index[source], c.operation_index[target],
                records[source].completion_time, records[target].reconfiguration_start)
        w_tasks = neural['w_tasks']
        f_tasks = neural['f_tasks']
        w_task_global = {task.task_id: int(w.w_node_by_op[c.operation_index[task.operation_id]])
                         for task in w_tasks}
        f_task_global = {task.task_id: int(w.f_node_by_op[c.operation_index[task.operation_id]])
                         for task in f_tasks}
        pairs = sorted(
            ((source, target) for tasks in schedule.w_timelines.values()
             for source, target in zip(tasks, tasks[1:])),
            key=lambda pair: (pair[0].task_id, pair[1].task_id))
        for source, target in pairs:
            add(8, w_task_global[source.task_id], w_task_global[target.task_id],
                source.arrival_time, target.empty_start)
        pairs = sorted(
            ((source, target) for tasks in schedule.f_timelines.values()
             for source, target in zip(tasks, tasks[1:])),
            key=lambda pair: (pair[0].task_id, pair[1].task_id))
        for source, target in pairs:
            add(9, f_task_global[source.task_id], f_task_global[target.task_id],
                source.return_wh, target.departure_wh)
        for task in w_tasks:
            add(10, w_task_global[task.task_id], c.operation_index[task.operation_id],
                task.arrival_time, records[task.operation_id].start_time)
        for task in f_tasks:
            add(11, f_task_global[task.task_id], c.operation_index[task.operation_id],
                task.arrival_island, records[task.operation_id].start_time)
        for operation in neural['positive_reconfig']:
            op = c.operation_index[operation]
            add(12, int(w.reconf_node_by_op[op]), op,
                records[operation].reconfiguration_end, records[operation].start_time)
        for task in w_tasks:
            add(13, w_task_global[task.task_id],
                w_resource_offset + c.w_index[task.vehicle_id])
        for task in f_tasks:
            add(14, f_task_global[task.task_id],
                f_resource_offset + c.f_index[task.vehicle_id])
        for operation in neural['positive_reconfig']:
            op = c.operation_index[operation]
            add(15, int(w.reconf_node_by_op[op]),
                island_offset + c.island_index[records[operation].island_id])
        trigger = sorted(
            (c.operations[int(w.previous_island_op[c.operation_index[operation]])], operation)
            for operation in neural['positive_reconfig']
            if w.previous_island_op[c.operation_index[operation]] >= 0)
        for previous, operation in trigger:
            add(16, c.operation_index[previous], int(w.reconf_node_by_op[c.operation_index[operation]]),
                records[previous].completion_time, records[operation].reconfiguration_start)
        for task in sorted((task for task in w_tasks if task.predecessor_op is not None),
                           key=lambda task: (task.predecessor_op, task.task_id)):
            add(17, c.operation_index[task.predecessor_op], w_task_global[task.task_id],
                records[task.predecessor_op].completion_time, task.loaded_start)
        for operation in neural['positive_reconfig']:
            op = c.operation_index[operation]
            add(18, int(w.reconf_node_by_op[op]),
                config_offset + int(w.reconf_source_config[op]))
        for operation in neural['positive_reconfig']:
            op = c.operation_index[operation]
            add(19, int(w.reconf_node_by_op[op]),
                config_offset + c.config_index[records[operation].config_id])
        directed = 2 * count
        w.edge_index[0, :directed:2] = w.forward_source[:count]
        w.edge_index[1, :directed:2] = w.forward_target[:count]
        w.edge_index[0, 1:directed:2] = w.forward_target[:count]
        w.edge_index[1, 1:directed:2] = w.forward_source[:count]
        w.relations[:directed:2] = w.forward_relation[:count]
        w.relations[1:directed:2] = w.forward_relation[:count] + len(EDGE_TYPES)
        w.edge_features[:directed:2] = w.forward_features[:count]
        w.edge_features[1:directed:2] = w.forward_features[:count]
        return count

    def _actions(self, actions, node_count, forward_count):
        c, w = self.context, self.workspace
        repairs = len(NGAS_REPAIR_IDS)
        targets = actions[::repairs]
        target_count, action_count = len(targets), len(actions)
        if target_count > w.capacities['targets'] or action_count > w.capacities['actions']:
            raise RuntimeError('Compact action capacity exceeded')
        w.target_membership[:target_count].fill(0)
        w.target_node_membership[:target_count, :node_count].fill(False)
        w.target_provenance[:target_count].fill(0)
        w.target_boundary[:target_count, :node_count].fill(0)
        w.target_boundary_stats[:target_count].fill(0)
        w.target_overlap[:target_count].fill(0)
        for target_index, action in enumerate(targets):
            operation_indices = [c.operation_index[value] for value in action.target.operations]
            w.target_membership[target_index, operation_indices] = 1.
            w.target_node_membership[target_index, operation_indices] = True
            values = list(action.target.provenance_features(RULES, FAMILIES, OPERATORS))
            values[-1] /= len(RULES)
            w.target_provenance[target_index] = values
            np.take(w.target_node_membership[target_index], w.forward_source[:forward_count],
                    out=w.edge_selected_source[:forward_count])
            np.take(w.target_node_membership[target_index], w.forward_target[:forward_count],
                    out=w.edge_selected_target[:forward_count])
            np.logical_xor(w.edge_selected_source[:forward_count],
                           w.edge_selected_target[:forward_count],
                           out=w.edge_crosses[:forward_count])
            crossing = np.flatnonzero(w.edge_crosses[:forward_count])
            source_selected = w.edge_selected_source[crossing]
            boundary_nodes = np.where(
                source_selected, w.forward_target[crossing], w.forward_source[crossing])
            w.target_boundary[target_index, boundary_nodes] = 1.
            groups = w.forward_relation[crossing] + np.where(
                source_selected, len(EDGE_TYPES), 0)
            scale = max(1., float(len(operation_indices)))
            w.target_boundary_stats[target_index] = np.bincount(
                groups, minlength=BOUNDARY_DIM) / scale
            w.target_overlap[target_index, 0] = (
                w.critical_mask[operation_indices].sum() / scale)
        size_index = {size: index for index, size in enumerate(SIZE_FRACTIONS)}
        for target_index, action in enumerate(targets):
            begin, end = repairs * target_index, repairs * (target_index + 1)
            w.membership[begin:end] = w.target_membership[target_index]
            w.provenance[begin:end] = w.target_provenance[target_index]
            w.sizes[begin:end] = size_index[action.size]
            w.repairs[begin:end] = np.arange(repairs)
            w.boundary[begin:end, :node_count] = w.target_boundary[target_index, :node_count]
            w.boundary_stats[begin:end] = w.target_boundary_stats[target_index]
            w.overlap[begin:end] = w.target_overlap[target_index]
        return target_count

    def build(self, current, state_id, streams):
        schedule = current.schedule
        timings = {}
        started = time.perf_counter()
        neural = self._neural_nodes(schedule)
        timings['compact_node_features'] = time.perf_counter() - started
        started = time.perf_counter()
        analysis, signature, bottleneck = self._compact_critical(schedule, neural)
        timings['compact_event_critical_mapping'] = time.perf_counter() - started
        started = time.perf_counter()
        forward_count = self._forward_edges(schedule, neural)
        timings['compact_edge_update'] = time.perf_counter() - started
        started = time.perf_counter()
        banks = build_all_banks(
            self.context.instance, current, state_id, streams, analysis,
            schedule_feature_cache=neural['bank_features'])
        actions = tuple(
            JointAction(bank.size, target, repair)
            for bank in banks for target in bank.targets for repair in NGAS_REPAIR_IDS)
        timings['shared_candidate_bank'] = time.perf_counter() - started
        started = time.perf_counter()
        target_count = self._actions(actions, neural['node_count'], forward_count)
        action_count = len(actions)
        node_count = neural['node_count']
        directed = 2 * forward_count
        np.copyto(
            self.workspace.tensor_x[:node_count], self.workspace.x[:node_count],
            casting='unsafe')
        batch = {
            'x': torch.from_numpy(self.workspace.tensor_x[:node_count]),
            'types': torch.from_numpy(self.workspace.types[:node_count]),
            'edge_index': torch.from_numpy(self.workspace.edge_index[:, :directed]),
            'relations': torch.from_numpy(self.workspace.relations[:directed]),
            'edge_features': torch.from_numpy(self.workspace.edge_features[:directed]),
            'operation_nodes': torch.from_numpy(self.workspace.operation_nodes),
            'critical_mask': torch.from_numpy(self.workspace.critical_mask[:node_count]),
            'membership': torch.from_numpy(self.workspace.membership[:action_count]),
            'provenance': torch.from_numpy(self.workspace.provenance[:action_count]),
            'sizes': torch.from_numpy(self.workspace.sizes[:action_count]),
            'repairs': torch.from_numpy(self.workspace.repairs[:action_count]),
            'boundary_membership': torch.from_numpy(
                self.workspace.boundary[:action_count, :node_count]),
            'boundary_stats': torch.from_numpy(self.workspace.boundary_stats[:action_count]),
            'target_critical_overlap': torch.from_numpy(self.workspace.overlap[:action_count]),
        }
        timings['vectorized_action_features_and_tensor_views'] = (
            time.perf_counter() - started)
        summary = {
            'requested_targets': sum(bank.requested_count for bank in banks),
            'unique_targets': sum(len(bank.targets) for bank in banks),
            'duplicate_targets': sum(bank.duplicate_count for bank in banks),
            'joint_actions': len(actions),
            'targets_by_size': {bank.size: len(bank.targets) for bank in banks},
        }
        return CompactBuildResult(
            actions, batch, signature, bottleneck, summary,
            node_count, forward_count, target_count, timings)
