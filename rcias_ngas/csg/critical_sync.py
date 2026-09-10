"""All makespan-critical chains, including resource and readiness activities."""
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
    # Nodes with no path to E (e.g. a final F return after makespan) have infinite
    # latest time and are NEVER declared critical by imposing an artificial sink.
    latest_start, latest_finish = {}, {}
    for key in reversed(graph.topological_order):
        latest_finish[key] = graph.makespan if key == 'E' else min(
            (latest_start[s] for s in graph.successors[key]), default=math.inf)
        latest_start[key] = latest_finish[key] - graph.nodes[key].duration
    zero = {key for key in graph.nodes if abs(latest_start[key] - earliest_start[key]) <= tolerance}
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
    for arc in graph.arcs:
        margin = graph.nodes[arc.target].start_time - graph.nodes[arc.source].end_time
        critical = arc.source in zero and arc.target in zero and abs(margin) <= tolerance
        edges.append({'source': arc.source, 'target': arc.target, 'relation': arc.relation,
                      'active_margin': margin, 'critical': critical})
        if critical:
            for op in {graph.nodes[k].operation_id for k in (arc.source, arc.target)} - {None}:
                edge_count[op] += 1
                reasons[op].add(f"critical_edge:{arc.source}->{arc.target}:{arc.relation}")
    scores = {op: 4 * node_count[op] + edge_count[op] + len(resources[op]) for op in reasons}
    nodes = {}
    for key, node in sorted(graph.nodes.items()):
        finite = math.isfinite(latest_start[key])
        nodes[key] = {
            'kind': node.kind, 'operation_id': node.operation_id, 'resource_id': node.resource_id,
            'earliest_start': earliest_start[key], 'earliest_finish': earliest_finish[key],
            'latest_start': latest_start[key] if finite else None,
            'latest_finish': latest_finish[key] if finite else None,
            'slack': latest_start[key] - earliest_start[key] if finite else None,
            'zero_slack': key in zero, 'reaches_makespan': finite,
        }
    return CriticalSync(tolerance, nodes, tuple(sorted(edges, key=lambda e: (e['source'], e['target'], e['relation']))),
                        {op: tuple(sorted(value)) for op, value in reasons.items()}, scores,
                        tuple(sorted((op for op in reasons if reasons[op]), key=lambda op: (-scores[op], op))))


def critical_sync(instance, decoded):
    if not decoded.feasible:
        raise ValueError("Critical analysis requires a feasible schedule")
    return analyze_graph(build_generalized_chdg(instance, decoded))
