from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.encoder import NIModelConfig
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.tensorize import CSGTensorizer
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from scripts import train_phase6j_caur as r
from scripts.audit_phase6k_preprocessing_decisions import inference_inputs
from scripts.diagnose_phase6k_reductions import CaptureIndexAdds, replay_operation


def fixture():
    import json
    instance = load_instance("instances/tiny/tiny_01.json")
    current = decode_candidate(instance, candidate_from_actions(instance, solve_dispatching(instance, "H1").actions))
    graph = build_csg_from_schedule(instance, current.schedule, state_id="diagnostic", search_progress=0.4, search_stage="40-60%")
    ops = tuple(graph.operation_to_node)
    records = [{"state_id": "diagnostic", "target_set_id": f"arm-{i}", "destroyed_operation_ids": json.dumps(target),
                "mean_relative_improvement": i, "rank_within_state": i + 1, "rank_percentile": i,
                "regret_to_best": 1 - i, "top1": i == 1, "top3": True}
               for i, target in enumerate((ops[:2], ops[-2:]))]
    tensorizer = CSGTensorizer()
    sample = NIStateSample(tensorizer.tensorize(graph), tensorize_action_records(graph, records), {})
    frame = pd.DataFrame({"target_set_id": ["arm-0", "arm-1"], **{k: ["x", "x"] for k in r.CATEGORICAL_COLUMNS},
                          **{k: [0., 0.] for k in r.NUMERIC_COLUMNS}})
    frame["is_fallback"] = [True, False]
    transform = r.FeatureTransform({k: ("x",) for k in r.CATEGORICAL_COLUMNS},
                                   {k: 0. for k in r.NUMERIC_COLUMNS}, {k: 1. for k in r.NUMERIC_COLUMNS})
    return sample, frame, transform, tensorizer


def test_outcome_free_batch_matches_legacy_model_inputs_and_outputs():
    sample, frame, transform, tensorizer = fixture()
    legacy = batch_state_samples([sample])
    allowed = ("target_set_ids", "target_operation_indices", "target_action_index")
    membership_only = SimpleNamespace(graph=sample.graph,
                                      actions=SimpleNamespace(**{k: getattr(sample.actions, k) for k in allowed}))
    packed = inference_inputs(membership_only, frame, transform, "cpu")
    assert not hasattr(packed["batch"], "utility")
    for name in ("node_features", "node_ptr", "node_batch_index"):
        for key, value in getattr(legacy, name).items():
            assert torch.equal(getattr(packed["batch"], name)[key], value)
    model = CSGTargetSetScorer(tensorizer, NIModelConfig(hidden_dim=16, heads=4, layers=1, dropout=0)).eval()
    with torch.inference_mode():
        actual, expected = model(packed["batch"]), model(legacy)
    assert torch.equal(actual.scores, expected.scores)
    assert torch.equal(actual.action_embeddings, expected.action_embeddings)
    assert np.all(packed["supported"])


@pytest.mark.parametrize("fault", ["order", "fallback"])
def test_inference_identity_errors_are_rejected(fault):
    sample, frame, transform, _ = fixture()
    if fault == "order":
        frame = frame.iloc[::-1]
    else:
        frame["is_fallback"] = False
    with pytest.raises(ValueError):
        inference_inputs(sample, frame, transform, "cpu")


def test_operator_capture_keeps_original_destination_and_source_unchanged():
    original = torch.tensor([[10.], [20.]])
    target, source = original.clone(), torch.tensor([[1.], [2.], [3.]])
    capture = CaptureIndexAdds()
    with capture:
        target.index_add_(0, torch.tensor([0, 0, 1]), source)
    assert torch.equal(target, torch.tensor([[13.], [23.]]))
    record = capture.records[0]
    assert torch.equal(record["target"], original)
    assert torch.equal(record["source"], source)
    assert replay_operation(record, 3)["variants"] == 1
    assert torch.equal(record["target"], original)
