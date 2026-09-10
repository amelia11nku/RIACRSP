from dataclasses import replace
import torch

from rcias_ngas.csg.features import action_features, state_features, tensorize, RULES
from rcias_ngas.critic.dataset import balanced_actions
from rcias_ngas.critic.development import instance_fold, state_plan, selected_rules
from rcias_ngas.critic.joint_critic import JointCritic
from rcias_ngas.critic.losses import joint_loss
from rcias_ngas.rng import RNGStreams
from tests.ngas.test_bank import small


def sample(small):
    instance, current = small
    actions, _ = balanced_actions(instance, current, 'model_test', RNGStreams(instance.instance_id, 1))
    state = state_features(instance, current, 'model_test')
    return state, actions


def test_batched_save_load_and_repair_identity(small, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    state, actions = sample(small)
    actions = [actions[0], replace(actions[0], repair='greedy'), replace(actions[0], repair='regret3')]
    batch = tensorize(state, action_features(state, actions))
    model = JointCritic().eval()
    with torch.inference_mode():
        together = model(batch)
        separate = [model(tensorize(state, action_features(state, [a]))) for a in actions]
    assert torch.isfinite(together['advantage']).all()
    assert torch.allclose(together['advantage'], torch.cat([r['advantage'] for r in separate]), atol=1e-6)
    assert not torch.equal(together['advantage'][1:2], together['advantage'][2:3])
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    loaded = JointCritic().eval()
    loaded.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
    with torch.inference_mode():
        assert torch.equal(together['advantage'], loaded(batch)['advantage'])


def test_origin_permutation_and_no_label_features(small):
    state, actions = sample(small)
    permuted = [replace(a, target=replace(a.target, origin_rules=tuple(reversed(a.target.origin_rules)),
                                          origin_families=tuple(reversed(a.target.origin_families)))) for a in actions]
    assert action_features(state, actions) == action_features(state, permuted)
    assert not any('advantage' in name or 'label' in name for name in state)
    assert len(RULES) == 24


def test_noise_loss_finite_gradients_and_ties(small):
    state, actions = sample(small)
    model = JointCritic()
    batch = tensorize(state, action_features(state, actions[:4]))
    y = torch.tensor([[0.] * 9, [.00001] * 9, [.01] * 9, [-.01] * 9])
    losses = joint_loss(model(batch), y)
    assert losses['material_pairs'].item() == 5
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    tied = joint_loss(model(batch), torch.zeros(4, 9))
    assert tied['material_pairs'] == 0
    assert tied['pairwise'].item() == tied['listwise'].item() == 0


def test_structural_fold_isolation_and_rule_coverage():
    specs = [{'instance_id': f'{s}_{c}_{rep}', 'scale': s, 'CF_level': c, 'replicate': rep}
             for s in ('S', 'M', 'L') for c in ('CF1', 'CF2', 'CF3') for rep in (1, 2)]
    plan = state_plan(specs, (746101, 746102, 746103))
    assert len(plan) == 72
    for spec in specs:
        assert len({r['fold'] for r in plan if r['instance']['scale'] == spec['scale'] and r['instance']['CF_level'] == spec['CF_level']}) == 1
    for fold in range(3):
        subset = [s for s in specs if instance_fold(s) == fold]
        assert len(subset) == 6
        assert {s['scale'] for s in subset} == {'S', 'M', 'L'}
        assert {s['CF_level'] for s in subset} == {'CF1', 'CF2', 'CF3'}
    assert {rule for r in plan for i in range(3) for rule in selected_rules(r['ordinal'], i)} == set(RULES)
