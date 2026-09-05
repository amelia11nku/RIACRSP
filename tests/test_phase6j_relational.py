from __future__ import annotations

import fcntl
import json

import pytest
import torch

from rcias_clgri.ni.encoder import NIModelConfig
from rcias_clgri.csg import build_csg_from_schedule
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.ni.batching import batch_state_samples
from rcias_clgri.ni.dataset import NIStateSample, tensorize_action_records
from rcias_clgri.ni.phase6j_relational import FallbackRelativeInteraction, RelationalCAURModel
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.tensorize import CSGTensorizer
from scripts import run_phase6j_caur_j3 as j3
from scripts.launch_phase6j_caur_j3 import is_training_command
from rcias_clgri.search.common import candidate_from_actions, decode_candidate


def make_model():
    return RelationalCAURModel(
        CSGTargetSetScorer(CSGTensorizer(), NIModelConfig(layers=2, utility_head=True)),
        (25, 8, 6),
    )


def test_interaction_preserves_fallback_and_depends_on_all_four_inputs():
    torch.manual_seed(9)
    layer = FallbackRelativeInteraction()
    inputs = [torch.randn(4, width, requires_grad=True) for width in (128, 128, 128, 12)]
    action, fallback, state, origin = inputs
    output = layer(*inputs)
    assert output.shape == action.shape
    assert torch.equal(layer(action, action, state, origin), torch.zeros_like(action))
    output.square().mean().backward()
    assert all(tensor.grad is not None and tensor.grad.abs().sum() > 0 for tensor in inputs)
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in layer.parameters())
    permutation = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(layer(*(tensor[permutation] for tensor in inputs)), output[permutation])


def test_j3_caps_trainability_and_checkpoint_round_trip():
    model = make_model()
    total, trainable = model.parameter_counts()
    assert total == 5_336_159
    assert trainable == 2_597_855
    assert total <= 5_350_000 and trainable <= 2_600_000
    assert sum(p.numel() for p in model.interaction.parameters()) == 3168
    model.train()
    assert not model.base.state_encoder.layers[0].training
    assert model.base.state_encoder.layers[-1].training
    assert model.base.action_encoder.projection.training
    assert model.interaction.training
    assert not any(p.requires_grad for p in model.base.state_encoder.layers[0].parameters())
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    allowed = ("heads.", "interaction.", "base.state_encoder.layers.1.", "base.action_encoder.projection.")
    assert all(name.startswith(allowed) for name in trainable_names)
    state = j3.checkpoint_state(model)
    assert set(state) == trainable_names
    clone = make_model()
    j3.restore_trainable(clone, state)
    for name, value in j3.checkpoint_state(clone).items():
        torch.testing.assert_close(value, state[name])
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="missing or unexpected"):
        j3.restore_trainable(clone, missing)
    model.eval()
    assert not any(module.training for module in model.modules())


def test_worker_lock_rejects_second_worker_before_loading_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(j3, "OUT", tmp_path)
    with (tmp_path / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns the lock"):
            j3.run_training()


def test_j3_forward_backward_updates_only_authorized_parameters():
    instance = load_instance(j3.ROOT / "instances/tiny/tiny_01.json")
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    graph = build_csg_from_schedule(instance, decoded.schedule, state_id="j3-test",
                                    search_progress=0.4, search_stage="40-60%")
    ops = tuple(graph.operation_to_node)
    rows = [{
        "state_id": graph.state_id, "target_set_id": f"arm-{i}",
        "destroyed_operation_ids": json.dumps(selected),
        "mean_relative_improvement": utility, "rank_within_state": i + 1,
        "rank_percentile": 1.0 - i, "regret_to_best": 0.2 - utility,
        "top1": i == 0, "top3": True, "arm_family": "ORIGINAL_OPERATOR",
        "origin_destroy_operator": "related",
    } for i, (selected, utility) in enumerate(((ops[:2], 0.2), (ops[-2:], 0.0)))]
    sample = NIStateSample(CSGTensorizer().tensorize(graph), tensorize_action_records(graph, rows), {"scale": "S"})
    batch = batch_state_samples([sample])
    model = make_model().train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    output = model(batch, fallback_action_indices=torch.tensor([1]),
                   categorical=torch.tensor([[1, 1, 1], [2, 1, 1]]), numeric=torch.zeros(2, 12))
    assert output.advantage.shape == output.beats_fallback_logit.shape == (2,)
    loss = output.advantage.square().sum() + output.beats_fallback_logit.square().sum() + output.immediate_utility.square().sum()
    loss.backward()
    assert model.interaction.output.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    assert not torch.equal(model.interaction.output.weight, before["interaction.output.weight"])


def test_process_detection_ignores_shell_wrappers_and_freeze_commands():
    assert is_training_command(["/env/bin/python", "scripts/train_phase6j_caur.py", "--device", "cuda"])
    assert is_training_command(["python", "scripts/run_phase6j_caur_j3.py", "--mode", "train"])
    assert not is_training_command(["python", "scripts/run_phase6j_caur_j3.py", "--mode", "freeze"])
    assert not is_training_command(["bash", "-c", "python scripts/run_phase6j_caur_j3.py --mode train"])
    assert not is_training_command(["rg", "train_phase6j_caur.py"])


def test_j3_sources_do_not_open_old_holdout_payloads():
    for relative in (*j3.CODE_FILES, "scripts/launch_phase6j_caur_j3.py"):
        source = (j3.ROOT / relative).read_text().lower()
        assert "outputs/phase6i_mr/r11_validation" not in source
        assert "r11_live_rev_holdout" not in source
