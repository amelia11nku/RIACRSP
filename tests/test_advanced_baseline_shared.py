from __future__ import annotations

import math
import random

from rcias_clgri.search.common import decode_candidate, random_candidate
from rcias_clgri.search.dabc_chdg import (
    build_generalized_chdg,
    critical_island_blocks,
    critical_path,
)
from rcias_clgri.search.operators import pox_pair, uniform_pair


def test_generalized_chdg_is_acyclic_and_matches_decoder_makespan(automotive_instance):
    for seed in range(20):
        decoded = decode_candidate(
            automotive_instance,
            random_candidate(automotive_instance, random.Random(seed)),
        )
        graph = build_generalized_chdg(automotive_instance, decoded)
        path = critical_path(graph, random.Random(seed))
        assert len(graph.topological_order) == len(graph.nodes)
        assert math.isclose(path.duration, decoded.makespan)
        assert path.node_ids[0] == "S"
        assert path.node_ids[-1] == "E"
        for node_id in path.node_ids:
            node = graph.nodes[node_id]
            assert node.start_time <= node.end_time


def test_critical_island_blocks_are_maximal_realized_runs(automotive_instance):
    decoded = decode_candidate(
        automotive_instance,
        random_candidate(automotive_instance, random.Random(17)),
    )
    graph = build_generalized_chdg(automotive_instance, decoded)
    path = critical_path(graph)
    blocks = critical_island_blocks(decoded, path)
    critical = {
        node.removeprefix("OP:") for node in path.node_ids if node.startswith("OP:")
    }
    for block in blocks:
        timeline = decoded.schedule.island_timelines[block.island_id]
        positions = [timeline.index(operation) for operation in block.operation_ids]
        assert positions == list(range(positions[0], positions[-1] + 1))
        assert all(operation in critical for operation in block.operation_ids)
        if positions[0] > 0:
            assert timeline[positions[0] - 1] not in critical
        if positions[-1] + 1 < len(timeline):
            assert timeline[positions[-1] + 1] not in critical


def test_shared_pox_and_uniform_crossover_preserve_domains(automotive_instance):
    rng = random.Random(23)
    left = random_candidate(automotive_instance, rng)
    right = random_candidate(automotive_instance, rng)
    orders = pox_pair(
        automotive_instance, left.operation_order, right.operation_order, rng
    )
    assert all(len(order) == len(set(order)) == automotive_instance.num_operations for order in orders)
    assert all(set(order) == set(automotive_instance.operations) for order in orders)

    for left_layer, right_layer in (
        (left.island_assignment, right.island_assignment),
        (left.w_assignment, right.w_assignment),
        (left.f_assignment, right.f_assignment),
    ):
        first, second = uniform_pair(left_layer, right_layer, rng)
        assert all(value in {a, b} for value, a, b in zip(first, left_layer, right_layer))
        assert all(value in {a, b} for value, a, b in zip(second, left_layer, right_layer))
