from __future__ import annotations

import copy
from dataclasses import replace
import fcntl
import json

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
from rcias_clgri.ni.phase6j_caur_model import CAURModel
from rcias_clgri.ni.phase6j_deployment import SharedFrozenCAUREnsemble
from rcias_clgri.ni.scorer import CSGTargetSetScorer
from rcias_clgri.ni.tensorize import CSGTensorizer
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from scripts import prepare_phase6j_caur_deployment as deploy
from scripts.audit_phase6j_caur_readiness import data_gate_checks


def models():
    base = CSGTargetSetScorer(CSGTensorizer(), NIModelConfig(layers=2, utility_head=True))
    return [CAURModel(copy.deepcopy(base), (25, 8, 6), family=deploy.FAMILY).eval() for _ in range(3)]


def test_shared_encoder_is_exact_and_does_not_consume_outcome_tensors():
    instance = load_instance(deploy.ROOT / "instances/tiny/tiny_01.json")
    candidate = candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    decoded = decode_candidate(instance, candidate)
    graph = build_csg_from_schedule(instance, decoded.schedule, state_id="deployment-test",
                                    search_progress=0.4, search_stage="40-60%")
    ops = tuple(graph.operation_to_node)
    rows = [{"state_id": graph.state_id, "target_set_id": f"arm-{i}",
             "destroyed_operation_ids": json.dumps(selected), "mean_relative_improvement": 0.1 * i,
             "rank_within_state": i + 1, "rank_percentile": 1.0 - i, "regret_to_best": 0.1 * (1 - i),
             "top1": i == 0, "top3": True, "arm_family": "ORIGINAL_OPERATOR", "origin_destroy_operator": "related"}
            for i, selected in enumerate((ops[:2], ops[-2:]))]
    sample = NIStateSample(CSGTensorizer().tensorize(graph), tensorize_action_records(graph, rows), {"scale": "S"})
    batch = batch_state_samples([sample])
    seeds = models()
    ensemble = SharedFrozenCAUREnsemble(seeds)
    inputs = {"fallback_action_indices": torch.tensor([1]), "categorical": torch.tensor([[1, 1, 1], [2, 1, 1]]),
              "numeric": torch.zeros(2, 12)}
    with torch.inference_mode():
        individual = [model(batch, **inputs) for model in seeds]
        shared = ensemble(batch, **inputs)
        poisoned = replace(batch, **{key: torch.full_like(getattr(batch, key), float("nan"))
                                     for key in ("utility", "positive", "rank_percentile", "regret_to_best")})
        ignored_labels = ensemble(poisoned, **inputs)
    for i, key in enumerate(("advantage", "beats_fallback_logit", "immediate_utility")):
        assert torch.equal(shared[i], torch.stack([getattr(o, key) for o in individual]))
        assert torch.equal(shared[i], ignored_labels[i])
    assert sum(p.numel() for p in ensemble.parameters()) == 5_406_717
    assert not any(module.training for module in ensemble.modules())


def test_shared_encoder_rejects_nonidentical_or_unfrozen_bases_and_wrong_family():
    seeds = models()
    with pytest.raises(ValueError, match="exactly three J1"):
        SharedFrozenCAUREnsemble(seeds[:2])
    seeds[1].family = "J2_CONT_LASTBLOCK"
    with pytest.raises(ValueError, match="exactly three J1"):
        SharedFrozenCAUREnsemble(seeds)
    seeds[1].family = deploy.FAMILY
    parameter = next(seeds[1].base.parameters())
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        SharedFrozenCAUREnsemble(seeds)
    parameter.requires_grad_(False)
    with torch.no_grad():
        parameter.add_(1)
    with pytest.raises(ValueError, match="encoders differ"):
        SharedFrozenCAUREnsemble(seeds)


def test_full_fit_epochs_are_per_seed_medians_without_metric_selection(monkeypatch):
    values = {1: [6, 5, 3], 2: [25, 16, 3], 3: [3, 15, 4]}
    monkeypatch.setattr(deploy.r, "run_paths", lambda family, seed, fold: (None, None, (seed, fold)))
    monkeypatch.setattr(deploy.r, "load_json", lambda key: {"best_epoch": values[key[0]][key[1]]})
    plan = deploy.epoch_plan({"training": {"seeds": [1, 2, 3]}})
    assert [v["full_fit_epochs"] for v in plan.values()] == [5, 16, 4]


def test_inference_gate_uses_three_seed_uncertainty_and_ignores_labels():
    protocol = {"calibrator": {"method": "SIGMOID_IDENTITY", "parameters": {}},
                "gate": {"p_min": 0.55, "lcb_lambda": 1.0, "delta_min": 0}, "immediate_harm_floor": -0.005}
    packed = {"frame": pd.DataFrame({"target_set_id": ["fallback", "neural"], "is_fallback": [True, False]}),
              "supported": np.array([True, True])}
    output = (torch.tensor([[0., 0.2], [0., 0.3], [0., 0.4]]), torch.ones(3, 2), torch.zeros(3, 2))
    decision = deploy.deployment_decision(output, packed, protocol)
    assert decision.intervened and decision.selected_target_set_id == "neural"
    packed["frame"]["continuation_advantage_mean"] = [999.0, -999.0]
    assert deploy.deployment_decision(output, packed, protocol) == decision
    output[0][2, 1] = -0.6
    assert not deploy.deployment_decision(output, packed, protocol).intervened


@pytest.mark.parametrize("failure", ["ece", "gate", "origin", "scale"])
def test_essential_data_gates_cannot_be_bypassed(failure):
    config = {"r12_acceptance": {"essential": {"overall_spearman_gt": 0, "minimum_scale_mean_spearman": 0,
              "selected_lift_gt": 0, "selected_bootstrap_lcb_gt": 0, "selected_winner_ece_max": 0.1}}}
    summary = {"metrics": {"overall_spearman": 0.18, "mean_spearman_by_scale": {"S": 0.1, "M": 0.1, "L": 0.1},
                          "selected_lift": 0.01, "selected_lift_lcb": 0.001},
               "calibration": {"metrics": [{"expected_calibration_error": 0.04}]},
               "selected_gate": {"retained": True, "coverage_pass": True, "selected_lift": 0.005,
                                 "selected_lift_lcb": 0.001, "scale_S_lift": 0.01, "scale_M_lift": 0.01, "scale_L_lift": 0.01}}
    assert all(data_gate_checks(summary, config, origin_diversity=True).values())
    if failure == "ece":
        summary["calibration"]["metrics"][0]["expected_calibration_error"] = 0.1148
    elif failure == "gate":
        summary["selected_gate"] = None
    elif failure == "scale":
        summary["metrics"]["mean_spearman_by_scale"]["L"] = -0.01
    assert not all(data_gate_checks(summary, config, origin_diversity=failure != "origin").values())


def test_deployment_worker_rejects_duplicate_before_loading_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "OUT", tmp_path)
    with (tmp_path / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns the lock"):
            deploy.run()


def test_completed_seed_hash_failure_never_triggers_a_refit(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "OUT", tmp_path)
    checkpoint, record = deploy.seed_paths(1)
    checkpoint.write_bytes(b"corrupted")
    record.write_text(json.dumps({"status": "COMPLETE", "seed": 1, "r13_accessed": False, "r14_accessed": False,
                                  "protocol_sha256": "fake", "checkpoint_sha256": "wrong"}))
    monkeypatch.setattr(deploy.r, "digest", lambda path: "fake")
    with pytest.raises(RuntimeError, match="completed seed hash changed"):
        deploy.load_seed(1, None, {}, {}, torch.device("cpu"))


def test_new_deployment_sources_do_not_open_old_holdout_payloads():
    for relative in deploy.CODE_FILES:
        source = (deploy.ROOT / relative).read_text().lower()
        assert "outputs/phase6i_mr/r11_validation" not in source
        assert "r11_live_rev_holdout" not in source
