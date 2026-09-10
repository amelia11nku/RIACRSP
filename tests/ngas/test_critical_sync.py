from types import SimpleNamespace

import pytest

from rcias_clgri.search.dabc_chdg import (
    EventArc, EventNode, GeneralizedCHDG, _topological_order,
)
from rcias_ngas.csg.critical_mapping import CRITICAL_FEATURE_NAMES, map_critical_events
from rcias_ngas.csg.critical_sync import analyze_graph


def graph_fixture(branch_end=2.):
    """Three tied length-8 chains plus one noncritical and one orphan branch."""
    nodes = {node.node_id: node for node in (
        EventNode('S', 'SOURCE', 0, 0),
        EventNode('W_EMPTY:w1', 'W_EMPTY', 0, 1, 'a', 'W1'),
        EventNode('W_LOADED:w1', 'W_LOADED', 1, 2, 'a', 'W1'),
        EventNode('RECONFIG:a', 'RECONFIGURATION', 2, 3, 'a', 'M1'),
        EventNode('F_OUTBOUND:f1', 'F_OUTBOUND', 3, 5, 'a', 'F1'),
        EventNode('OP:a', 'OPERATION', 5, 8, 'a', 'M1'),
        EventNode('F_RETURN:f1', 'F_RETURN', 5, 15, 'a', 'F1'),
        EventNode('OP:c', 'OPERATION', 0, 4, 'c', 'M2'),
        EventNode('RECONFIG:d', 'RECONFIGURATION', 4, 4, 'd', 'M2'),
        EventNode('OP:d', 'OPERATION', 4, 8, 'd', 'M2'),
        EventNode('OP:p', 'OPERATION', 0, 2, 'p', 'M3'),
        EventNode('IDLE:OP:q', 'UNEXPLAINED_IDLE', 2, 5),
        EventNode('OP:q', 'OPERATION', 5, 8, 'q', 'M3'),
        EventNode('OP:b', 'OPERATION', 0, branch_end, 'b', 'M4'),
        EventNode('E', 'SINK', 8, 8),
    )}
    arcs = tuple(EventArc(*values) for values in (
        ('S', 'W_EMPTY:w1', 'W_RESOURCE_ORDER'),
        ('W_EMPTY:w1', 'W_LOADED:w1', 'W_EMPTY_BEFORE_LOADED'),
        ('W_LOADED:w1', 'RECONFIG:a', 'W_ARRIVAL_READY'),
        ('RECONFIG:a', 'F_OUTBOUND:f1', 'CONFIGURATION_READY'),
        ('F_OUTBOUND:f1', 'OP:a', 'F_ARRIVAL_READY'),
        ('OP:a', 'E', 'MAKESPAN_COMPLETION'),
        ('F_OUTBOUND:f1', 'F_RETURN:f1', 'F_OUTBOUND_BEFORE_RETURN'),
        ('S', 'OP:c', 'TECHNOLOGICAL_PRECEDENCE'),
        ('OP:c', 'RECONFIG:d', 'ISLAND_ORDER'),
        ('RECONFIG:d', 'OP:d', 'CONFIGURATION_READY'),
        ('OP:d', 'E', 'MAKESPAN_COMPLETION'),
        ('S', 'OP:p', 'TECHNOLOGICAL_PRECEDENCE'),
        ('OP:p', 'IDLE:OP:q', 'REALIZED_IDLE_AFTER'),
        ('IDLE:OP:q', 'OP:q', 'REALIZED_IDLE_BEFORE'),
        ('OP:q', 'E', 'MAKESPAN_COMPLETION'),
        ('S', 'OP:b', 'TECHNOLOGICAL_PRECEDENCE'),
        ('OP:b', 'OP:a', 'REALIZED_PRODUCT_CHAIN'),
        ('OP:b', 'E', 'MAKESPAN_COMPLETION'),
        ('W_EMPTY:w1', 'OP:a', 'W_ARRIVAL_READY'),
    ))
    order, predecessors, successors = _topological_order(nodes, arcs)
    return GeneralizedCHDG(nodes, arcs, predecessors, successors, order, 8)


def assert_longest_path_identity(graph, result):
    for key, row in result.nodes.items():
        if row['zero_slack']:
            total = row['distance_from_source'] + graph.nodes[key].duration + row['distance_to_sink']
            assert total == pytest.approx(graph.makespan, abs=result.tolerance)


def test_union_of_tied_critical_chains_and_noncritical_cases():
    graph = graph_fixture()
    result = analyze_graph(graph)
    assert_longest_path_identity(graph, result)
    expected = {
        'S', 'W_EMPTY:w1', 'W_LOADED:w1', 'RECONFIG:a', 'F_OUTBOUND:f1', 'OP:a',
        'OP:c', 'RECONFIG:d', 'OP:d', 'OP:p', 'IDLE:OP:q', 'OP:q', 'E',
    }
    assert {key for key, row in result.nodes.items() if row['zero_slack']} == expected
    assert result.nodes['OP:b']['slack'] == 3
    assert result.nodes['F_RETURN:f1']['slack'] is None
    assert not result.nodes['F_RETURN:f1']['reaches_makespan']
    redundant = next(edge for edge in result.edges
                     if edge['source'] == 'W_EMPTY:w1' and edge['target'] == 'OP:a')
    assert redundant['active_margin'] == 4
    assert not redundant['critical']
    assert 'REALIZED_IDLE' in result.nodes['IDLE:OP:q']['critical_categories']
    assert result.nodes['RECONFIG:d']['zero_slack']


def test_tolerance_boundary_is_frozen():
    close_graph, far_graph = graph_fixture(5 - 5e-9), graph_fixture(5 - 5e-7)
    close, far = analyze_graph(close_graph), analyze_graph(far_graph)
    assert_longest_path_identity(close_graph, close)
    assert_longest_path_identity(far_graph, far)
    assert close.nodes['OP:b']['zero_slack']
    assert not far.nodes['OP:b']['zero_slack']


def test_event_to_csg_mapping_and_typed_critical_aggregation():
    graph = graph_fixture()
    result = analyze_graph(graph)
    csg = SimpleNamespace(nodes={
        'OP': tuple(SimpleNamespace(key=key) for key in ('a', 'b', 'c', 'd', 'p', 'q')),
        'W_EVENT': (SimpleNamespace(key='w1'),),
        'F_EVENT': (SimpleNamespace(key='f1'),),
        'RECONF_EVENT': (SimpleNamespace(key='R:a'),),
    })
    mapping = map_critical_events(graph, csg, result)
    assert mapping.event_to_neural['W_EMPTY:w1'] == ('W_EVENT', 'w1')
    assert mapping.event_to_neural['W_LOADED:w1'] == ('W_EVENT', 'w1')
    assert mapping.event_to_neural['F_RETURN:f1'] == ('F_EVENT', 'f1')
    assert mapping.event_to_neural['RECONFIG:a'] == ('RECONF_EVENT', 'R:a')
    assert mapping.event_to_neural['RECONFIG:d'] == ('OP', 'd')
    assert mapping.event_to_neural['IDLE:OP:q'] == ('OP', 'q')
    zero = CRITICAL_FEATURE_NAMES.index('zero_slack')
    unreachable = CRITICAL_FEATURE_NAMES.index('mapped_unreachable_fraction')
    for key in (('W_EVENT', 'w1'), ('F_EVENT', 'f1'), ('RECONF_EVENT', 'R:a')):
        assert mapping.node_features[key][zero] == 1
    assert mapping.node_features['F_EVENT', 'f1'][unreachable] == .5
    assert set(mapping.omitted_boundaries) == {'S', 'E'}
