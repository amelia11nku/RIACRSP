from rcias_clgri.search.dabc_chdg import EventNode, EventArc, GeneralizedCHDG, _topological_order
from rcias_ngas.csg.critical_sync import analyze_graph


def graph_fixture(branch_delta=0.):
    # Hand solution: W(2)->R(1)->F(2)->A(3) ends at 8.
    # B(2) joins A at t=5: B slack=3. Final F return ends at 15 and
    # has no route to makespan E, so it must NOT get a negative deadline.
    nodes = {n.node_id: n for n in (
        EventNode('S', 'SOURCE', 0, 0),
        EventNode('W', 'W_LOADED', 0, 2, 'a', 'W1'),
        EventNode('R', 'RECONFIGURATION', 2, 3, 'a', 'M1'),
        EventNode('F', 'F_OUTBOUND', 3, 5, 'a', 'F1'),
        EventNode('B', 'OPERATION', 0, 2 + branch_delta, 'b', 'M2'),
        EventNode('A', 'OPERATION', 5, 8, 'a', 'M1'),
        EventNode('FR', 'F_RETURN', 5, 15, 'a', 'F1'),
        EventNode('E', 'SINK', 8, 8),
    )}
    arcs = tuple(EventArc(s, t, 'ready') for s, t in (
        ('S', 'W'), ('W', 'R'), ('R', 'F'), ('F', 'A'), ('S', 'B'),
        ('B', 'A'), ('A', 'E'), ('B', 'E'), ('F', 'FR')))
    order, pred, succ = _topological_order(nodes, arcs)
    return GeneralizedCHDG(nodes, arcs, pred, succ, order, 8)


def test_hand_computed_slack_and_projection():
    result = analyze_graph(graph_fixture())
    assert result.nodes['B']['slack'] == 3
    assert result.nodes['B']['latest_finish'] == 5
    assert result.nodes['FR']['slack'] is None
    assert result.nodes['A']['earliest_start'] == result.nodes['A']['latest_start'] == 5
    for key in ('W', 'R', 'F', 'A'):
        assert result.nodes[key]['zero_slack']
        assert any(':' + key in r for r in result.operation_reasons['a'])
    assert result.ranked_operations == ('a',)
    assert not next(e for e in result.edges if e['source'] == 'B' and e['target'] == 'A')['critical']


def test_zero_slack_uses_tolerance():
    close = analyze_graph(graph_fixture(3 - 5e-9))
    far = analyze_graph(graph_fixture(3 - 5e-7))
    assert close.nodes['B']['zero_slack']
    assert not far.nodes['B']['zero_slack']
