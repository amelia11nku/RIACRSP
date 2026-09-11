"""CPM analysis of every makespan-critical chain in the realized event DAG.

``operation_scores`` are retained only as a destroy-target prioritization
heuristic.  Criticality itself is defined solely by finite CPM slack within the
frozen tolerance.
"""
from dataclasses import dataclass
import math

from rcias_clgri.search.dabc_chdg import build_generalized_chdg


@dataclass(frozen=True)
class CriticalSync:
    tolerance: float
    nodes: dict
    edges: tuple[dict, ...]
    operation_reasons: dict[str, tuple[str, ...]]
    operation_scores: dict[str, int]
    ranked_operations: tuple[str, ...]


CRITICAL_CATEGORIES = (
    'PRECEDENCE', 'ISLAND', 'RECONFIGURATION', 'W_LOGISTICS',
    'F_LOGISTICS', 'SYNCHRONIZATION', 'REALIZED_IDLE',
)

RELATION_CATEGORY = {
    # Generic relation used only by hand-built CPM fixtures.
    'ready': 'SYNCHRONIZATION',
    'TECHNOLOGICAL_PRECEDENCE': 'PRECEDENCE',
    'REALIZED_PRODUCT_CHAIN': 'PRECEDENCE',
    'ISLAND_ORDER': 'ISLAND',
    'CONFIGURATION_READY': 'RECONFIGURATION',
    'W_RESOURCE_ORDER': 'W_LOGISTICS',
    'W_EMPTY_BEFORE_LOADED': 'W_LOGISTICS',
    'WORKPIECE_RELEASE': 'W_LOGISTICS',
    'WAREHOUSE_RELEASE': 'W_LOGISTICS',
    'W_ARRIVAL_READY': 'W_LOGISTICS',
    'F_RESOURCE_ORDER': 'F_LOGISTICS',
    'F_OUTBOUND_BEFORE_RETURN': 'F_LOGISTICS',
    'F_ARRIVAL_READY': 'F_LOGISTICS',
    'MAKESPAN_COMPLETION': 'SYNCHRONIZATION',
    'REALIZED_IDLE_AFTER': 'REALIZED_IDLE',
    'REALIZED_IDLE_BEFORE': 'REALIZED_IDLE',
}


def relation_category(relation: str) -> str:
    """Return the frozen physical/realized cause category for an event arc."""
    try:
        return RELATION_CATEGORY[relation]
    except KeyError as error:
        raise ValueError(f'Unclassified generalized-CHDG relation: {relation}') from error


def analyze_graph(graph, absolute_tolerance=1e-8, relative_tolerance=1e-10):
    if absolute_tolerance <= 0 or relative_tolerance < 0:
        raise ValueError("Invalid time tolerance")
    tolerance = max(absolute_tolerance, relative_tolerance * max(1., graph.makespan))
    earliest_start, earliest_finish = {}, {}
    for key in graph.topological_order:
        node = graph.nodes[key]
        earliest_start[key] = max((earliest_finish[p] for p in graph.predecessors[key]), default=0.)
        earliest_finish[key] = earliest_start[key] + node.duration
        if abs(earliest_finish[key] - node.end_time) > tolerance:
            raise ValueError(f"DAG does not reproduce schedule: {key}")
    if abs(earliest_finish['E'] - graph.makespan) > tolerance:
        raise ValueError("DAG sink differs from makespan")
    # Longest remaining duration after each node. Nodes with no path to E (e.g.
    # a final F return after makespan) remain undefined and are never made
    # critical by imposing an artificial sink.
    distance_to_sink = {}
    for key in reversed(graph.topological_order):
        if key == 'E':
            distance_to_sink[key] = 0.
            continue
        candidates = [graph.nodes[s].duration + distance_to_sink[s]
                      for s in graph.successors[key] if math.isfinite(distance_to_sink[s])]
        distance_to_sink[key] = max(candidates) if candidates else math.inf
    latest_start = {
        key: graph.makespan - graph.nodes[key].duration - distance_to_sink[key]
        if math.isfinite(distance_to_sink[key]) else math.inf
        for key in graph.nodes
    }
    latest_finish = {
        key: latest_start[key] + graph.nodes[key].duration
        if math.isfinite(latest_start[key]) else math.inf
        for key in graph.nodes
    }
    zero = {key for key in graph.nodes
            if math.isfinite(latest_start[key])
            and abs(latest_start[key] - earliest_start[key]) <= tolerance}
    reasons = {n.operation_id: set() for n in graph.nodes.values() if n.operation_id}
    resources = {op: set() for op in reasons}
    node_count = {op: 0 for op in reasons}
    edge_count = {op: 0 for op in reasons}
    for key in sorted(zero):
        node = graph.nodes[key]
        if node.operation_id:
            op = node.operation_id
            reasons[op].add(f"zero_slack:{node.kind}:{key}")
            node_count[op] += 1
            if node.resource_id:
                resources[op].add(node.resource_id)
    edges = []
    critical_incident = {key: [] for key in graph.nodes}
    for arc in graph.arcs:
        margin = graph.nodes[arc.target].start_time - graph.nodes[arc.source].end_time
        critical = arc.source in zero and arc.target in zero and abs(margin) <= tolerance
        category = relation_category(arc.relation)
        edges.append({'source': arc.source, 'target': arc.target, 'relation': arc.relation,
                      'category': category, 'active_margin': margin, 'critical': critical})
        if critical:
            critical_incident[arc.source].append(edges[-1])
            critical_incident[arc.target].append(edges[-1])
            for op in {graph.nodes[k].operation_id for k in (arc.source, arc.target)} - {None}:
                edge_count[op] += 1
                reasons[op].add(f"critical_edge:{category}:{arc.source}->{arc.target}:{arc.relation}")
    scores = {op: 4 * node_count[op] + edge_count[op] + len(resources[op]) for op in reasons}
    nodes = {}
    for key, node in sorted(graph.nodes.items()):
        finite = math.isfinite(latest_start[key])
        incident = critical_incident[key]
        categories = {edge['category'] for edge in incident}
        if key in zero:
            categories.update({
                'RECONFIGURATION' if node.kind == 'RECONFIGURATION' else
                'W_LOGISTICS' if node.kind in ('W_EMPTY', 'W_LOADED') else
                'F_LOGISTICS' if node.kind in ('F_OUTBOUND', 'F_RETURN') else
                'REALIZED_IDLE' if node.kind == 'UNEXPLAINED_IDLE' else None
            } - {None})
        nodes[key] = {
            'kind': node.kind, 'operation_id': node.operation_id, 'resource_id': node.resource_id,
            'earliest_start': earliest_start[key], 'earliest_finish': earliest_finish[key],
            'latest_start': latest_start[key] if finite else None,
            'latest_finish': latest_finish[key] if finite else None,
            'slack': latest_start[key] - earliest_start[key] if finite else None,
            'distance_from_source': earliest_start[key],
            'distance_to_sink': distance_to_sink[key] if finite else None,
            'zero_slack': key in zero, 'reaches_makespan': finite,
            'critical_degree': len(incident),
            'critical_categories': tuple(sorted(categories)),
        }
    return CriticalSync(tolerance, nodes, tuple(sorted(edges, key=lambda e: (e['source'], e['target'], e['relation']))),
                        {op: tuple(sorted(value)) for op, value in reasons.items()}, scores,
                        tuple(sorted((op for op in reasons if reasons[op]), key=lambda op: (-scores[op], op))))


def critical_sync(instance, decoded):
    if not decoded.feasible:
        raise ValueError("Critical analysis requires a feasible schedule")
    return analyze_graph(build_generalized_chdg(instance, decoded))
