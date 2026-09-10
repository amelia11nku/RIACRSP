from dataclasses import replace

import torch

from rcias_clgri.csg.schema import NODE_TYPE_ORDER
from rcias_ngas.actions.repair import NGAS_REPAIR_IDS, repair_index
from rcias_ngas.csg.critical_mapping import CRITICAL_FEATURE_NAMES
from rcias_ngas.csg.revised_features import (
    BASE_NODE_DIM, NODE_DIM, action_features, state_features, tensorize,
)
from rcias_ngas.critic.dataset import balanced_actions
from rcias_ngas.critic.pooling import TypeCriticalPooling
from rcias_ngas.critic.revised_critic import RevisedJointCritic
from rcias_ngas.rng import RNGStreams
from tests.ngas.test_bank import small


def sample(small):
    instance, current = small
    state = state_features(instance, current, 'revised_test')
    actions, _ = balanced_actions(
        instance, current, 'revised_test', RNGStreams(instance.instance_id, 7))
    return state, actions


def test_typed_features_are_versioned_outcome_blind_and_reachable(small):
    state, _ = sample(small)
    assert state['schema'] == 'ngas-csg-features-v2'
    assert all(len(row) == NODE_DIM for row in state['node_features'])
    assert not any('advantage' in key or 'label' in key for key in state)
    defined = CRITICAL_FEATURE_NAMES.index('slack_defined') + BASE_NODE_DIM
    zero = CRITICAL_FEATURE_NAMES.index('zero_slack') + BASE_NODE_DIM
    unreachable = CRITICAL_FEATURE_NAMES.index('mapped_unreachable_fraction') + BASE_NODE_DIM
    f_type = NODE_TYPE_ORDER.index('F_EVENT')
    f_rows = [row for row, kind in zip(state['node_features'], state['node_types'])
              if kind == f_type]
    assert f_rows and all(row[defined] == 1 for row in f_rows)
    assert any(row[zero] == 1 for row in f_rows)
    assert any(row[unreachable] > 0 for row in f_rows)


def test_pooling_handles_empty_and_nonempty_resource_groups():
    hidden = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    batch = {
        'types': torch.tensor([
            NODE_TYPE_ORDER.index('OP'), NODE_TYPE_ORDER.index('OP'),
            NODE_TYPE_ORDER.index('W_EVENT'), NODE_TYPE_ORDER.index('F_EVENT')]),
        'critical_mask': torch.tensor([True, False, True, False]),
    }
    pool = TypeCriticalPooling(8)
    result = pool(hidden, batch)
    assert result.shape == (1, 8)
    assert torch.isfinite(result).all()


def test_rt_hgt_and_compact_batch_all_actions_once_and_deterministically(small):
    torch.manual_seed(11)
    state, actions = sample(small)
    chosen = actions[:4]
    batch = tensorize(state, action_features(state, chosen))
    for encoder in ('compact_relational', 'rt_hgt'):
        model = RevisedJointCritic(encoder, hidden=16, layers=1, heads=4).eval()
        calls = []
        hook = model.encoder.register_forward_hook(lambda *_: calls.append(1))
        with torch.inference_mode():
            together = model(batch)
            repeated = model(batch)
        hook.remove()
        assert len(calls) == 2
        assert together['advantage'].shape == (4,)
        assert torch.equal(together['advantage'], repeated['advantage'])
        assert torch.isfinite(together['advantage']).all()
        separate = []
        with torch.inference_mode():
            for action in chosen:
                one = tensorize(state, action_features(state, [action]))
                separate.append(model(one)['advantage'])
        assert torch.allclose(together['advantage'], torch.cat(separate), atol=1e-6)


def test_ngas_repair_ids_and_provenance_order_are_stable(small):
    assert NGAS_REPAIR_IDS == (
        'greedy', 'regret2', 'regret3', 'reconfiguration_aware', 'transport_aware')
    assert [repair_index(value) for value in NGAS_REPAIR_IDS] == list(range(5))
    state, actions = sample(small)
    action = actions[0]
    permuted = replace(action, target=replace(
        action.target,
        origin_rules=tuple(reversed(action.target.origin_rules)),
        origin_families=tuple(reversed(action.target.origin_families)),
        origin_operators=tuple(reversed(action.target.origin_operators)),
    ))
    assert action_features(state, [action]) == action_features(state, [permuted])
