from dataclasses import replace
import os
import subprocess
import sys

import pytest

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.actions.destroy_size import destroy_count, SIZE_FRACTIONS
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import execute_action
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.bank.provenance import deduplicate
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.rng import RNGStreams


@pytest.fixture
def small():
    instance = load_instance('instances/tiny/tiny_01.json')
    h1 = solve_dispatching(instance, 'H1')
    return instance, decode_candidate(instance, candidate_from_actions(instance, h1.actions))


def test_full_bank_dedup_provenance(small):
    instance, current = small
    rngs = RNGStreams(instance.instance_id, 1)
    bank = build_bank(instance, current, 's0', 'small', rngs)
    assert bank == build_bank(instance, current, 's0', 'small', rngs)
    assert bank.requested_count == 24
    assert len(bank.targets) + bank.duplicate_count == 24
    assert bank.duplicate_count > 0
    assert all(len(t.operations) == bank.destroy_count for t in bank.targets)
    assert deduplicate(tuple(reversed(bank.proposals)), 's0', 'small') == bank.targets
    vocab = tuple(sorted(p.origin_rule for p in bank.proposals))
    for target in bank.targets:
        assert target.provenance_features(vocab, (), ()) == replace(target, origin_rules=tuple(reversed(target.origin_rules))).provenance_features(vocab, (), ())
        assert not any('makespan' in field or 'reward' in field for field in target.__dataclass_fields__)
    assert any('csg_critical_sync' in t.origin_rules for t in bank.targets)
    assert not any('operator_critical' in t.origin_rules for t in bank.targets)


@pytest.mark.parametrize('filename', ['tiny_01', 'tiny_02', 'tiny_03'])
def test_critical_projection(filename):
    instance = load_instance(f'instances/tiny/{filename}.json')
    h1 = solve_dispatching(instance, 'H1')
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    analysis = critical_sync(instance, current)
    assert analysis.ranked_operations
    assert all(analysis.operation_reasons[op] for op in analysis.ranked_operations)
    for key, node in analysis.nodes.items():
        if node['zero_slack'] and node['operation_id']:
            assert any(key in reason for reason in analysis.operation_reasons[node['operation_id']])
    for edge in analysis.edges:
        if edge['critical']:
            assert abs(edge['active_margin']) <= analysis.tolerance
            assert analysis.nodes[edge['source']]['zero_slack']
            assert analysis.nodes[edge['target']]['zero_slack']


def test_joint_execution_reproduces_trajectory(small):
    instance, initial = small
    def trajectory(extra_target_draw):
        current = initial
        result = []
        rngs = RNGStreams(instance.instance_id, 7)
        for iteration in range(4):
            state_id = str(iteration)
            bank = build_bank(instance, current, state_id, 'medium', rngs)
            action = JointAction('medium', bank.targets[0], 'regret2')
            if extra_target_draw:
                rngs.stream('target', state_id).random()
            current, _ = execute_action(instance, current, action, rngs, state_id, 2)
            result.append((action.action_id, current.candidate, current.makespan))
        return result
    assert trajectory(False) == trajectory(False) == trajectory(True)


def test_hash_seed_does_not_change_repair(small):
    code = '''from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import execute_action
from rcias_ngas.bank.provenance import Target
from rcias_ngas.rng import RNGStreams
i = load_instance('instances/tiny/tiny_01.json')
d = decode_candidate(i, candidate_from_actions(i, solve_dispatching(i, 'H1').actions))
a = JointAction('small', Target('t', tuple(i.operations[:3]), (), (), ()), 'regret2')
r, _ = execute_action(i, d, a, RNGStreams(i.instance_id, 42), 's', 2)
print(r.candidate, r.makespan)
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed}) for seed in ('1', '987')]
    assert outputs[0] == outputs[1]


def test_size_bounds():
    for n in (1, 2, 3, 6, 61, 194):
        for size in SIZE_FRACTIONS:
            assert 1 <= destroy_count(n, size) <= n
    assert len({destroy_count(61, s) for s in SIZE_FRACTIONS}) == 3
