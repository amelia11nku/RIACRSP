from __future__ import annotations

from rcias_clgri.env.insertion_decoder import InsertionDecoder
from rcias_clgri.graph.builder import build_graph_state


def test_graph_has_five_node_types_and_relations(automotive_instance):
    graph = build_graph_state(automotive_instance, InsertionDecoder(automotive_instance).empty_schedule())
    assert set(graph.node_features) == {"O", "J", "M", "W", "F"}
    relations = {edge.relation for edge in graph.edges}
    assert {
        "precedence", "contains", "belongs_to", "eligible_on", "spatial",
        "reachable_to", "deliver_to",
    } <= relations
    assert graph.ready_operations == ("o11", "o21", "o22")
    assert graph.w_masks[("o11", "M1")] == ("W1",)


def test_graph_adds_realized_dynamic_relations(controlled_env):
    graph = build_graph_state(controlled_env.instance, controlled_env.schedule)
    relations = {edge.relation for edge in graph.edges}
    assert "actual_product_prev" in relations
    assert "machine_prev" in relations
