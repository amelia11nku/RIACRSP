"""Deterministic generalized-event to neural-CSG critical feature mapping."""
from __future__ import annotations

from dataclasses import dataclass
import math

from rcias_ngas.csg.critical_sync import CRITICAL_CATEGORIES, CriticalSync


CRITICAL_FEATURE_NAMES = (
    'normalized_slack',
    'slack_defined',
    'zero_slack',
    'incident_critical_edge',
    'normalized_critical_degree',
    *(f'critical_category_{name.lower()}' for name in CRITICAL_CATEGORIES),
    'normalized_min_active_margin',
    'active_margin_defined',
    'critical_participation_score',
    'mapped_unreachable_fraction',
)

CRITICAL_NODE_TYPES = ('OP', 'W_EVENT', 'F_EVENT', 'RECONF_EVENT')


@dataclass(frozen=True)
class CriticalCSGMapping:
    event_to_neural: dict[str, tuple[str, str]]
    projection_reason: dict[str, str]
    node_features: dict[tuple[str, str], tuple[float, ...]]
    mapped_events: int
    aggregated_events: int
    omitted_boundaries: tuple[str, ...]


def _direct_mapping(event_id: str, kind: str, operation_id: str | None,
                    neural_keys: set[tuple[str, str]]) -> tuple[tuple[str, str] | None, str]:
    suffix = event_id.split(':', 1)[1] if ':' in event_id else event_id
    if kind == 'OPERATION':
        return ('OP', operation_id or suffix), 'DIRECT_OPERATION'
    if kind == 'RECONFIGURATION':
        direct = ('RECONF_EVENT', f'R:{operation_id or suffix}')
        if direct in neural_keys:
            return direct, 'DIRECT_RECONFIGURATION'
        return ('OP', operation_id or suffix), 'ZERO_DURATION_RECONFIGURATION_PROJECTED_TO_OP'
    if kind in ('W_EMPTY', 'W_LOADED'):
        return ('W_EVENT', suffix), 'W_EMPTY_LOADED_AGGREGATED'
    if kind in ('F_OUTBOUND', 'F_RETURN'):
        return ('F_EVENT', suffix), 'F_OUTBOUND_RETURN_AGGREGATED'
    return None, 'BOUNDARY_OR_IDLE'


def map_critical_events(event_graph, csg_graph, analysis: CriticalSync) -> CriticalCSGMapping:
    """Map every representable event once and aggregate CPM features by CSG node.

    W empty/loaded and F outbound/return activities intentionally share their
    corresponding transport-event node. Zero-duration reconfigurations absent
    from CSG-1.0 are projected to their OP node. Realized-idle activities are
    projected to the neural node of the activity that follows the idle gap.
    Source and sink are explicitly omitted graph boundaries.
    """
    neural_keys = {(kind, node.key) for kind in csg_graph.nodes for node in csg_graph.nodes[kind]}
    event_to_neural: dict[str, tuple[str, str]] = {}
    reasons: dict[str, str] = {}
    omitted = []
    for event_id in event_graph.topological_order:
        event = event_graph.nodes[event_id]
        if event.kind in ('SOURCE', 'SINK'):
            reasons[event_id] = 'INTENTIONALLY_OMITTED_GRAPH_BOUNDARY'
            omitted.append(event_id)
            continue
        mapped, reason = _direct_mapping(event_id, event.kind, event.operation_id, neural_keys)
        if event.kind == 'UNEXPLAINED_IDLE':
            successor_event = event_id.removeprefix('IDLE:')
            successor = event_graph.nodes[successor_event]
            mapped, _ = _direct_mapping(
                successor_event, successor.kind, successor.operation_id, neural_keys)
            reason = 'REALIZED_IDLE_PROJECTED_TO_SUCCESSOR'
        if mapped not in neural_keys:
            raise ValueError(f'Generalized event has no neural CSG mapping: {event_id} -> {mapped}')
        event_to_neural[event_id] = mapped
        reasons[event_id] = reason

    expected = {event_id for event_id, node in analysis.nodes.items()
                if node['zero_slack'] and node['kind'] not in ('SOURCE', 'SINK')}
    missing = expected - set(event_to_neural)
    if missing:
        raise ValueError(f'Critical generalized events are unmapped: {sorted(missing)}')

    grouped: dict[tuple[str, str], list[str]] = {key: [] for key in neural_keys}
    for event_id, key in event_to_neural.items():
        grouped[key].append(event_id)
    maximum_degree = max((node['critical_degree'] for node in analysis.nodes.values()), default=1) or 1
    incident_margins: dict[str, list[float]] = {key: [] for key in analysis.nodes}
    for edge in analysis.edges:
        incident_margins[edge['source']].append(float(edge['active_margin']))
        incident_margins[edge['target']].append(float(edge['active_margin']))

    feature_by_node = {}
    makespan = max(float(event_graph.makespan), 1.)
    for key in neural_keys:
        event_ids = grouped[key]
        if key[0] not in CRITICAL_NODE_TYPES or not event_ids:
            feature_by_node[key] = (0.,) * len(CRITICAL_FEATURE_NAMES)
            continue
        records = [analysis.nodes[event_id] for event_id in event_ids]
        finite_slacks = [float(row['slack']) for row in records if row['slack'] is not None]
        degree = sum(int(row['critical_degree']) for row in records)
        categories = {category for row in records for category in row['critical_categories']}
        margins = [max(0., margin) for event_id in event_ids for margin in incident_margins[event_id]]
        zero_count = sum(bool(row['zero_slack']) for row in records)
        values = [
            min(finite_slacks) / makespan if finite_slacks else 0.,
            float(bool(finite_slacks)),
            float(bool(zero_count)),
            float(degree > 0),
            degree / (maximum_degree * len(event_ids)),
            *(float(category in categories) for category in CRITICAL_CATEGORIES),
            min(margins) / makespan if margins else 0.,
            float(bool(margins)),
            .5 * (zero_count / len(event_ids) + min(1., degree / (maximum_degree * len(event_ids)))),
            sum(not row['reaches_makespan'] for row in records) / len(event_ids),
        ]
        if len(values) != len(CRITICAL_FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid critical feature aggregation for {key}')
        feature_by_node[key] = tuple(values)
    return CriticalCSGMapping(
        event_to_neural=event_to_neural,
        projection_reason=reasons,
        node_features=feature_by_node,
        mapped_events=len(event_to_neural),
        aggregated_events=sum(max(0, len(items) - 1) for items in grouped.values()),
        omitted_boundaries=tuple(sorted(omitted)),
    )
