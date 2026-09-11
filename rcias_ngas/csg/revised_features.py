"""A1.3R outcome-blind CSG tensors with typed CPM critical features."""
from __future__ import annotations

import math
import torch

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.csg.schema import SCHEMA, NODE_TYPE_ORDER, EDGE_TYPE_ORDER
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.repair import repair_index
from rcias_ngas.csg.critical_mapping import CRITICAL_FEATURE_NAMES, map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph
from rcias_ngas.csg.revised_schema import FAMILIES, OPERATORS, PROVENANCE_DIM, RULES
from rcias_ngas.evaluation.bks import content_hash


BASE_NODE_NAMES = {
    kind: tuple(name for name, spec in SCHEMA['node_types'][kind]['features'].items()
                if spec['semantic_type'] not in ('CONTINUOUS_TIME', 'COUNT', 'CURRENT_STATE_DESCRIPTOR')
                and name not in ('operation_slack_normalized', 'is_processing_critical_proxy',
                                 'is_resource_terminal_proxy'))
    for kind in NODE_TYPE_ORDER
}
BASE_NODE_DIM = max(map(len, BASE_NODE_NAMES.values()))
NODE_DIM = BASE_NODE_DIM + len(CRITICAL_FEATURE_NAMES)
EDGE_FEATURE_DIM = 2
BOUNDARY_DIM = 2 * len(EDGE_TYPE_ORDER)
ZERO_SLACK_INDEX = BASE_NODE_DIM + CRITICAL_FEATURE_NAMES.index('zero_slack')
MAJOR_NODE_TYPES = ('OP', 'W_EVENT', 'F_EVENT', 'RECONF_EVENT')


def state_features_from_components(instance, graph, mapping, *, include_hash=True,
                                   include_diagnostics=True):
    """Assemble the frozen neural inputs from already computed live graphs."""
    node_index, x, kinds, critical_masks = {}, [], [], []
    for type_index, kind in enumerate(NODE_TYPE_ORDER):
        for node in graph.nodes[kind]:
            node_index[kind, node.key] = len(x)
            values = [float(node.features[name]) for name in BASE_NODE_NAMES[kind]]
            values = [math.copysign(math.log1p(abs(value)), value) for value in values]
            values += [0.] * (BASE_NODE_DIM - len(values))
            critical_values = mapping.node_features[kind, node.key]
            values += critical_values
            x.append(values)
            kinds.append(type_index)
            critical_masks.append(critical_values[CRITICAL_FEATURE_NAMES.index('zero_slack')])

    edges, relations, edge_features, edge_directions = [], [], [], []
    for relation, name in enumerate(EDGE_TYPE_ORDER):
        for edge in graph.edges[name]:
            source = node_index[edge.source_type, edge.source_key]
            target = node_index[edge.target_type, edge.target_key]
            gap = float(edge.features.get('normalized_temporal_gap', 0.))
            binding = float(edge.features.get('binding_indicator', 0.))
            for s, t, r, direction in (
                    (source, target, relation, 1),
                    (target, source, relation + len(EDGE_TYPE_ORDER), -1)):
                edges.append([s, t])
                relations.append(r)
                edge_features.append([gap, binding])
                edge_directions.append(direction)
    operations = tuple(sorted(instance.operations))
    payload = {
        'schema': 'ngas-csg-features-v2',
        'node_features': x,
        'node_types': kinds,
        'edge_index': edges,
        'edge_types': relations,
        'edge_features': edge_features,
        'edge_directions': edge_directions,
        'operation_nodes': [node_index['OP', operation] for operation in operations],
        'operation_ids': operations,
        'critical_mask': critical_masks,
        'graph_hash': graph.graph_hash if include_diagnostics else '',
        'critical_mapping': {
            'mapped_events': mapping.mapped_events,
            'aggregated_events': mapping.aggregated_events,
            'omitted_boundaries': mapping.omitted_boundaries,
            'event_to_neural': {event: list(mapped) for event, mapped in sorted(mapping.event_to_neural.items())},
            'projection_reason': dict(sorted(mapping.projection_reason.items())),
        } if include_diagnostics else {},
        'feature_schema': {
            'base_node_dim': BASE_NODE_DIM,
            'node_dim': NODE_DIM,
            'critical_feature_names': CRITICAL_FEATURE_NAMES,
            'edge_feature_names': ('normalized_temporal_gap', 'binding_indicator'),
        },
    }
    payload['feature_hash'] = content_hash(payload) if include_hash else ''
    return payload


def state_features(instance, current, state_id):
    graph = build_csg_from_schedule(
        instance, current.schedule, state_id=state_id,
        search_progress=0., search_stage='0-20%')
    event_graph = build_generalized_chdg(instance, current)
    critical = analyze_graph(event_graph)
    mapping = map_critical_events(event_graph, graph, critical)
    return state_features_from_components(instance, graph, mapping)


def action_features(state, actions):
    operations = state['operation_ids']
    operation_set = set(operations)
    operation_position = {operation: index for index, operation in enumerate(operations)}
    operation_nodes = state['operation_nodes']
    memberships, provenance, boundary_memberships, boundary_stats = [], [], [], []
    critical_overlap = []
    target_cache = {}
    original_edges = [
        (edge, relation) for edge, relation, direction in zip(
            state['edge_index'], state['edge_types'], state['edge_directions']) if direction == 1
    ]
    incident_edges = [[] for _ in state['node_features']]
    for edge_index, ((source, target), relation) in enumerate(original_edges):
        record = edge_index, source, target, relation
        incident_edges[source].append(record)
        incident_edges[target].append(record)
    for action in actions:
        cache_key = (action.size, action.target)
        cached = target_cache.get(cache_key)
        if cached is not None:
            membership, values, boundary, relation_counts, overlap = cached
            memberships.append(membership)
            provenance.append(values)
            boundary_memberships.append(boundary)
            boundary_stats.append(relation_counts)
            critical_overlap.append(overlap)
            continue
        targets = set(action.target.operations)
        if not targets <= operation_set:
            raise ValueError('Target operation not present in CSG')
        membership = [float(operation in targets) for operation in operations]
        selected_nodes = {operation_nodes[operation_position[operation]] for operation in targets}
        memberships.append(membership)
        values = list(action.target.provenance_features(RULES, FAMILIES, OPERATORS))
        values[-1] /= len(RULES)
        provenance.append(values)
        boundary = [0.] * len(state['node_features'])
        relation_counts = [0.] * BOUNDARY_DIM
        candidate_edges = {
            record for node in selected_nodes for record in incident_edges[node]
        }
        for _, source, target, relation in candidate_edges:
            source_selected = source in selected_nodes
            if source_selected == (target in selected_nodes):
                continue
            boundary[target if source_selected else source] = 1.
            offset = len(EDGE_TYPE_ORDER) if source_selected else 0
            relation_counts[offset + relation] += 1.
        scale = max(1., float(len(targets)))
        boundary_memberships.append(boundary)
        relation_counts = [value / scale for value in relation_counts]
        boundary_stats.append(relation_counts)
        overlap = sum(
            state['critical_mask'][operation_nodes[operation_position[operation]]]
            for operation in targets) / scale
        critical_overlap.append(overlap)
        target_cache[cache_key] = membership, values, boundary, relation_counts, overlap
    return {
        'membership': memberships,
        'provenance': provenance,
        'sizes': [tuple(SIZE_FRACTIONS).index(action.size) for action in actions],
        'repairs': [repair_index(action.repair) for action in actions],
        'boundary_membership': boundary_memberships,
        'boundary_stats': boundary_stats,
        'target_critical_overlap': critical_overlap,
        'repair_ids': [action.repair for action in actions],
    }


def tensorize(state, actions, device='cpu'):
    """Tensorize one state and every action without a candidate graph loop."""
    return {
        'x': torch.tensor(state['node_features'], dtype=torch.float32, device=device),
        'types': torch.tensor(state['node_types'], dtype=torch.long, device=device),
        'edge_index': torch.tensor(state['edge_index'], dtype=torch.long, device=device).reshape(-1, 2).T,
        'relations': torch.tensor(state['edge_types'], dtype=torch.long, device=device),
        'edge_features': torch.tensor(state['edge_features'], dtype=torch.float32, device=device).reshape(-1, EDGE_FEATURE_DIM),
        'operation_nodes': torch.tensor(state['operation_nodes'], dtype=torch.long, device=device),
        'critical_mask': torch.tensor(state['critical_mask'], dtype=torch.bool, device=device),
        'membership': torch.tensor(actions['membership'], dtype=torch.float32, device=device),
        'provenance': torch.tensor(actions['provenance'], dtype=torch.float32, device=device),
        'sizes': torch.tensor(actions['sizes'], dtype=torch.long, device=device),
        'repairs': torch.tensor(actions['repairs'], dtype=torch.long, device=device),
        'boundary_membership': torch.tensor(actions['boundary_membership'], dtype=torch.float32, device=device),
        'boundary_stats': torch.tensor(actions['boundary_stats'], dtype=torch.float32, device=device),
        'target_critical_overlap': torch.tensor(actions['target_critical_overlap'], dtype=torch.float32, device=device).unsqueeze(1),
    }
