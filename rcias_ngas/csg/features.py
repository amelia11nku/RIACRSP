"""Outcome-blind compact CSG tensors with true critical-sync OP features."""
import math
import torch

from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.csg.schema import SCHEMA, NODE_TYPE_ORDER, EDGE_TYPE_ORDER
from rcias_clgri.search.alns import REPAIR
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.csg.critical_sync import critical_sync

RULE_GROUPS = (
    ('operator_random', 'operator_overloaded_island', 'operator_high_reconfiguration',
     'operator_w_bottleneck', 'operator_f_bottleneck', 'operator_related'),
    tuple(f'related_variant_{i}' for i in range(1, 5)),
    tuple(f'matched_random_{i}' for i in range(1, 4)),
    ('one_operation_swap', 'two_operation_swap', 'related_replace_25', 'related_replace_50'),
    ('near_same_product', 'near_precedence_neighbor', 'near_same_island_chain',
     'near_high_W_delay', 'near_high_F_delay', 'near_legacy_completion_tail'),
)
RULES = tuple(sorted(('csg_critical_sync', *(r for group in RULE_GROUPS for r in group))))
FAMILIES = ('ORIGINAL_OPERATOR', 'RELATED_VARIANT', 'MATCHED_RANDOM', 'LOCAL_PERTURBATION', 'STRUCTURED_NEAR_NEIGHBOR')
OPERATORS = ('random', 'csg_critical_sync', 'overloaded_island', 'high_reconfiguration', 'w_bottleneck', 'f_bottleneck', 'related')
NODE_NAMES = {kind: tuple(name for name, spec in SCHEMA['node_types'][kind]['features'].items()
                         if spec['semantic_type'] not in ('CONTINUOUS_TIME', 'COUNT', 'CURRENT_STATE_DESCRIPTOR')
                         and name not in ('operation_slack_normalized', 'is_processing_critical_proxy', 'is_resource_terminal_proxy'))
              for kind in NODE_TYPE_ORDER}
NODE_DIM = max(map(len, NODE_NAMES.values())) + 4
PROVENANCE_DIM = len(RULES) + len(FAMILIES) + len(OPERATORS) + 1


def state_features(instance, current, state_id):
    graph = build_csg_from_schedule(instance, current.schedule, state_id=state_id,
                                    search_progress=0., search_stage='0-20%')
    critical = critical_sync(instance, current)
    node_index, x, kinds = {}, [], []
    max_score = max(1, max(critical.operation_scores.values()))
    for type_index, kind in enumerate(NODE_TYPE_ORDER):
        for node in graph.nodes[kind]:
            node_index[kind, node.key] = len(x)
            values = [float(node.features[name]) for name in NODE_NAMES[kind]]
            values = [math.copysign(math.log1p(abs(v)), v) for v in values]
            values += [0.] * (NODE_DIM - 4 - len(values))
            if kind == 'OP':
                temporal = critical.nodes['OP:' + node.key]
                values += [temporal['slack'] / max(current.makespan, 1.),
                           float(temporal['zero_slack']), float(bool(critical.operation_reasons[node.key])),
                           critical.operation_scores[node.key] / max_score]
            else:
                values += [0.] * 4
            x.append(values)
            kinds.append(type_index)
    edges, relations, edge_features = [], [], []
    for relation, name in enumerate(EDGE_TYPE_ORDER):
        for edge in graph.edges[name]:
            source, target = node_index[edge.source_type, edge.source_key], node_index[edge.target_type, edge.target_key]
            gap = float(edge.features.get('normalized_temporal_gap', 0.))
            binding = float(edge.features.get('binding_indicator', 0.))
            for s, t, r in ((source, target, relation), (target, source, relation + len(EDGE_TYPE_ORDER))):
                edges.append([s, t])
                relations.append(r)
                edge_features.append([gap, binding])
    operations = tuple(sorted(instance.operations))
    return {'schema': 'ngas-csg-features-v1', 'node_features': x, 'node_types': kinds,
            'edge_index': edges, 'edge_types': relations, 'edge_features': edge_features,
            'operation_nodes': [node_index['OP', op] for op in operations],
            'operation_ids': operations, 'graph_hash': graph.graph_hash}


def action_features(state, actions):
    operations = state['operation_ids']
    memberships, provenance = [], []
    for action in actions:
        if not set(action.target.operations) <= set(operations):
            raise ValueError('Target operation not present in CSG')
        memberships.append([float(op in action.target.operations) for op in operations])
        values = list(action.target.provenance_features(RULES, FAMILIES, OPERATORS))
        values[-1] /= len(RULES)
        provenance.append(values)
    return {'membership': memberships, 'provenance': provenance,
            'sizes': [tuple(SIZE_FRACTIONS).index(a.size) for a in actions],
            'repairs': [REPAIR.index(a.repair) for a in actions]}


def tensorize(state, actions, device='cpu'):
    """Identifiers remain lookup-only. All action scores share one state forward."""
    return {
        'x': torch.tensor(state['node_features'], dtype=torch.float32, device=device),
        'types': torch.tensor(state['node_types'], dtype=torch.long, device=device),
        'edge_index': torch.tensor(state['edge_index'], dtype=torch.long, device=device).reshape(-1, 2).T,
        'relations': torch.tensor(state['edge_types'], dtype=torch.long, device=device),
        'edge_features': torch.tensor(state['edge_features'], dtype=torch.float32, device=device).reshape(-1, 2),
        'operation_nodes': torch.tensor(state['operation_nodes'], dtype=torch.long, device=device),
        'membership': torch.tensor(actions['membership'], dtype=torch.float32, device=device),
        'provenance': torch.tensor(actions['provenance'], dtype=torch.float32, device=device),
        'sizes': torch.tensor(actions['sizes'], dtype=torch.long, device=device),
        'repairs': torch.tensor(actions['repairs'], dtype=torch.long, device=device),
    }
