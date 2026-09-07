from dataclasses import asdict
from copy import deepcopy

import numpy as np
import pytest
import torch

from rcias_clgri.ni.phase6j_deployment import SharedFrozenCAUREnsemble
from rcias_clgri.ni.phase6j_caur_model import CAURModel
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.encoder import NIModelConfig
from rcias_clgri.ni.phase6k_runtime import (eager_reconstruction, prepare_topology,
    DeviceDecision, extract_decision, segment_softmax)
from scripts.audit_phase6k_preprocessing_decisions import inference_inputs
from scripts import prepare_phase6j_caur_deployment as deploy
from test_phase6k_preprocessing_diagnosis import fixture


def model_fixture():
    sample, frame, transform, tensorizer = fixture()
    base = CSGTargetSetScorer(tensorizer, NIModelConfig(hidden_dim=16, heads=4, layers=2, dropout=0))
    models = [CAURModel(deepcopy(base), (2, 2, 2), family='J1_CONT_FROZEN').eval() for _ in range(3)]
    packed = inference_inputs(sample, frame, transform, 'cpu')
    prepare_topology(packed)
    return SharedFrozenCAUREnsemble(models), packed


def test_e1_full_graph_and_three_independent_heads_bit_exact():
    e0, packed = model_fixture()
    before = deepcopy(e0.state_dict())
    e1 = eager_reconstruction(e0)
    with torch.inference_mode():
        ref = e0(packed['batch'], **deploy.model_inputs(packed))
        actual = e1(packed['batch'], **deploy.model_inputs(packed))
    assert all(torch.equal(a, b) for a, b in zip(ref, actual))
    assert all(torch.equal(e0.state_dict()[k], v) and torch.equal(e1.state_dict()[k], v) for k, v in before.items())
    assert e0.base.state_encoder.__class__ != e1.base.state_encoder.__class__


def test_explicit_softmax_retains_sparse_empty_segments():
    scores = torch.tensor([[1., 2.], [3., 4.], [2., 5.]])
    segments = torch.tensor([0, 0, 3])
    result = segment_softmax(scores, segments, 7)
    torch.testing.assert_close(result[:2], scores[:2].softmax(0))
    assert torch.equal(result[2], torch.ones(2))
    assert segment_softmax(scores[:0], segments[:0], 7).shape == (0, 2)


@pytest.mark.parametrize('mode', ['tie', 'probability', 'lcb', 'support', 'harm', 'fallback', 'intervene'])
def test_gpu_gate_semantics_against_original_with_lexical_ties(mode):
    _, packed = model_fixture()
    packed['frame']['target_set_id'] = ['z', 'a']
    ids = ('z', 'a')
    packed['lexical_rank'] = torch.tensor([1, 0])
    a = torch.tensor([[1., 2.]]).repeat(3, 1)
    logit = torch.full_like(a, 10.)
    immediate = torch.ones_like(a)
    support = np.array([True, True])
    if mode == 'tie': a.fill_(1.)
    if mode == 'probability': logit.fill_(-10.)
    if mode == 'lcb': a.fill_(-1.)
    if mode == 'support': support[1] = False
    if mode == 'harm': immediate.fill_(-1.)
    if mode == 'fallback': a[:, 0] = 3.
    protocol = {'calibrator': {'method': 'PLATT', 'parameters': {'coefficient': 1., 'intercept': 0.}},
                'gate': {'p_min': .5, 'lcb_lambda': 1., 'delta_min': 0.}, 'immediate_harm_floor': 0.}
    packed['supported'] = support
    outputs = (a, logit, immediate)
    expected = asdict(deploy.deployment_decision(outputs, packed, protocol))
    expected.pop('lcb')
    result = DeviceDecision(protocol)(outputs, torch.as_tensor(support), packed['lexical_rank'], packed['fallback_indices'])
    assert extract_decision(result, ids) == expected


def test_e2_keeps_each_seed_state_and_matches_e0_outputs():
    from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble
    e0, packed = model_fixture()
    e2 = VectorizedEnsemble(eager_reconstruction(e0))
    for name, stacked in zip(e2.heads.parameter_names, e2.heads.stacked_parameters):
        for seed, head in enumerate(e0.heads):
            assert torch.equal(stacked[seed], head.state_dict()[name])
    with torch.inference_mode():
        ref = e0(packed['batch'], **deploy.model_inputs(packed))
        actual = e2(packed['batch'], **deploy.model_inputs(packed))
    for a, b in zip(ref, actual):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert all(p.dtype == torch.float32 for p in e2.parameters())


def test_e4s_hoisted_qkv_preserves_weights_and_outputs_within_contract():
    from rcias_clgri.ni.phase6k_runtime import hoisted_qkv_reconstruction
    from rcias_clgri.ni.phase6k_vectorized import VectorizedEnsemble
    e0, packed = model_fixture()
    e2 = VectorizedEnsemble(eager_reconstruction(e0))
    before = {name: value.clone() for name, value in e2.state_dict().items()}
    e4s = hoisted_qkv_reconstruction(e2)
    with torch.inference_mode():
        reference = e0(packed['batch'], **deploy.model_inputs(packed))
        actual = e4s(packed['batch'], **deploy.model_inputs(packed))
    assert all(torch.equal(e4s.state_dict()[name], value) for name, value in before.items())
    for ref, value in zip(reference, actual):
        torch.testing.assert_close(ref, value, atol=1e-5, rtol=1e-5)


def test_runtime_selection_respects_fixed_order_margin_and_both_caps():
    from scripts.measure_phase6k_formal_latency import select_variant
    def result(neural, live, status='PASS'):
        return {'status':status, 'diagnostic_latency':{'neural_ms':{'p90':neural}, 'full_live_ms':{'p90':live}}}
    results = {'E1':result(29, 90), 'E2':result(26, 95), 'E3':result(12, 80)}
    assert select_variant(results) == 'E2'
    results['E2'] = result(26, 101)
    results['E3'] = result(12, 80, 'INELIGIBLE_PARITY')
    assert select_variant(results) == 'E1'
    results['E1'] = result(31, 90)
    assert select_variant(results) is None


def test_nonfinite_device_outputs_cannot_silently_select_an_action():
    _, packed = model_fixture()
    protocol = {'calibrator':{'method':'PLATT','parameters':{'coefficient':1.,'intercept':0.}},
                'gate':{'p_min':.5,'lcb_lambda':1.,'delta_min':0.},'immediate_harm_floor':0.}
    output = (torch.full((3, 2), float('nan')), torch.ones(3,2), torch.ones(3,2))
    value = DeviceDecision(protocol)(output, packed['support_tensor'], packed['lexical_rank'], packed['fallback_indices'])
    with pytest.raises(RuntimeError, match='non-finite'):
        extract_decision(value, packed['batch'].target_set_ids)


def test_historical_bank_capture_calls_original_builder_once(monkeypatch):
    from types import SimpleNamespace
    from scripts import phase6k_runtime_common as common

    generated, records = object(), object()
    calls = []

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return generated, records

    def scorer(*args, **kwargs):
        observed = common.phase6i_mr.build_live_proposal_bank('instance', 'current', state_id='state',
            destroy_count=2, seed_namespace=3)
        assert observed == (generated, records)
        return SimpleNamespace(arms=(), graph=object(), timings_ms={})

    monkeypatch.setattr(common.phase6i_mr, 'build_live_proposal_bank', builder)
    monkeypatch.setattr(common, 'score_frozen_candidate_bank', scorer)
    monkeypatch.setattr(common, 'select_forced_candidate_roles', lambda arms: ())
    ctx = {'instance': object(), 'current': object(), 'state_id': 'state', 'count': 2,
           'snapshot': {'search_progress': 0.5}}
    with pytest.raises(StopIteration):
        common.realize({'policy': SimpleNamespace(proposal_seed_namespace=3)}, ctx)
    assert len(calls) == 1


def test_historical_membership_reordering_reuses_projected_indices():
    from types import SimpleNamespace
    from scripts.phase6k_runtime_common import reorder_historical_membership

    actions = SimpleNamespace(
        target_set_ids=('z', 'a', 'm'),
        action_ptr=torch.tensor([0, 2, 3, 6]),
        target_operation_indices=torch.tensor([9, 8, 1, 4, 5, 6]),
    )
    reordered = reorder_historical_membership(actions)
    assert reordered.target_set_ids == ('a', 'm', 'z')
    assert torch.equal(reordered.target_operation_indices, torch.tensor([1, 4, 5, 6, 9, 8]))
    assert torch.equal(reordered.target_action_index, torch.tensor([0, 1, 1, 1, 2, 2]))


@pytest.mark.parametrize('fault', [None, 'missing_state', 'missing_candidate', 'duplicate_repetition', 'failed_pair'])
def test_equivalence_coverage_cannot_be_replaced_by_a_pass_label(fault):
    from scripts.measure_phase6k_formal_latency import verify_equivalence
    ids = [f'state-{i:03}' for i in range(288)]
    rows = [{'state_id':sid,'candidate_rows':23 if i<103 else 24,'pass':True,'checks':{'decision':True}}
            for i,sid in enumerate(ids)]
    repeats = [{'state_id':sid,'repetition':rep,'pass':True,'checks':{'decision':True}}
               for sid in ids[:5] for rep in range(30)]
    record = {'status':'PASS','rows':rows,'robustness':repeats}
    if fault == 'missing_state': rows.pop()
    if fault == 'missing_candidate': rows[0]['candidate_rows']-=1
    if fault == 'duplicate_repetition': repeats[-1] = repeats[0]
    if fault == 'failed_pair': rows[0]['checks']['decision']=False
    if fault:
        with pytest.raises(ValueError):
            verify_equivalence(record,ids,ids[:5])
    else:
        verify_equivalence(record,ids,ids[:5])
