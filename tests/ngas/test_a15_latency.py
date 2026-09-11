from rcias_clgri.csg.builder import build_csg_from_schedule
from rcias_clgri.csg.serialize import canonical_payload
from rcias_ngas.csg.revised_features import action_features, state_features, tensorize
from rcias_ngas.critic.dataset import balanced_actions
from rcias_ngas.latency.live_refresh import _tensorize_cpu
from rcias_ngas.rng import RNGStreams
from tests.ngas.test_bank import small


def test_hash_free_live_csg_preserves_graph_semantics(small):
    instance, current = small
    kwargs = dict(state_id='a15-test', search_progress=0., search_stage='0-20%')
    reference = build_csg_from_schedule(instance, current.schedule, **kwargs)
    live = build_csg_from_schedule(instance, current.schedule, **kwargs, attach_hash=False)
    assert reference.graph_hash
    assert live.graph_hash == ''
    assert canonical_payload(reference) == canonical_payload(live)


def test_cached_action_features_equal_independent_action_features(small):
    instance, current = small
    state = state_features(instance, current, 'a15-action-test')
    actions, _ = balanced_actions(
        instance, current, 'a15-action-test', RNGStreams(instance.instance_id, 7))
    chosen = actions[:15]
    together = action_features(state, chosen)
    row_fields = (
        'membership', 'provenance', 'sizes', 'repairs', 'boundary_membership',
        'boundary_stats', 'target_critical_overlap', 'repair_ids',
    )
    for index, action in enumerate(chosen):
        separate = action_features(state, [action])
        assert all(together[name][index] == separate[name][0] for name in row_fields)


def test_numpy_tensorization_is_bitwise_equal(small):
    instance, current = small
    state = state_features(instance, current, 'a15-tensor-test')
    actions, _ = balanced_actions(
        instance, current, 'a15-tensor-test', RNGStreams(instance.instance_id, 11))
    features = action_features(state, actions[:15])
    reference = tensorize(state, features)
    live = _tensorize_cpu(state, features)
    assert reference.keys() == live.keys()
    assert all(reference[name].dtype == live[name].dtype for name in reference)
    assert all(reference[name].shape == live[name].shape for name in reference)
    assert all(reference[name].equal(live[name]) for name in reference)
