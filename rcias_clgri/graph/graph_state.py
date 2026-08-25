"""Compatibility module exposing the dynamic graph-state public API."""

from .builder import EdgeRecord, GraphState, build_graph_state

__all__ = ["EdgeRecord", "GraphState", "build_graph_state"]
