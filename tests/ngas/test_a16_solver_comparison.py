import time
import json
from pathlib import Path

import pytest
import torch

from rcias_clgri.data.loader import load_instance
from rcias_ngas.critic.revised_critic import RevisedJointCritic
from rcias_ngas.evaluation.a16 import (
    holm_adjust, incumbent_at, paired_summary, right_continuous_auc,
    terminal_decision,
)
from rcias_ngas.evaluation.a16_io import validate_raw_contract, write_new_json
from rcias_ngas.evaluation.a16_integrity import audit_frozen_inputs, digest
from rcias_ngas.runtime import ProductionRefreshRuntime
from rcias_ngas.search.ngas_solver import NGASSearchConfig, solve_ngas


class TinyCritic:
    variant = 'C1'
    sha256 = 'tiny-c1'
    device = torch.device('cpu')

    def __init__(self):
        torch.manual_seed(17)
        self.model = RevisedJointCritic('rt_hgt', hidden=16, layers=1, heads=4).eval()


class DelayedRuntime:
    def __init__(self, critic, delay=.05):
        self.runtime = ProductionRefreshRuntime(critic)
        self.delay = delay

    def prepare_instance(self, instance):
        time.sleep(self.delay)
        return self.runtime.prepare_instance(instance)

    def refresh(self, *args, **kwargs):
        return self.runtime.refresh(*args, **kwargs)


def test_a16_charges_runtime_preparation_but_legacy_default_does_not():
    instance = load_instance('instances/tiny/tiny_01.json')
    critic = TinyCritic()
    config = NGASSearchConfig(candidate_trials=1, iteration_limit=0)
    legacy = solve_ngas(
        instance, 10., 746101, 'PERSISTENT_FIXED_REFRESH', critic, config,
        refresh_runtime=DelayedRuntime(critic))
    a16 = solve_ngas(
        instance, 10., 746101, 'PERSISTENT_FIXED_REFRESH', critic, config,
        refresh_runtime=DelayedRuntime(critic),
        budget_accounting='A16_INSTANCE_TOTAL')

    legacy_preparation = legacy.diagnostics['runtime_components']['runtime_preparation_seconds']
    a16_preparation = a16.diagnostics['runtime_components']['runtime_preparation_seconds']
    assert legacy_preparation >= .05
    assert a16_preparation >= .05
    assert legacy.runtime < legacy_preparation
    assert a16.runtime >= a16_preparation
    assert a16.diagnostics['solver_budget_elapsed_seconds'] == a16.runtime
    assert a16.diagnostics['budget_accounting'] == 'A16_INSTANCE_TOTAL'


def test_paired_metrics_and_terminal_decisions_use_instance_blocks():
    ngas = {'i1': 90., 'i2': 105., 'i3': 100.}
    baseline = {'i1': 100., 'i2': 100., 'i3': 100.}
    summary = paired_summary(ngas, baseline, 1e-9)
    assert (summary['wins'], summary['ties'], summary['losses']) == (1, 1, 1)
    assert summary['mean_relative_gain_percent'] == pytest.approx(5. / 3.)

    by_scale = {scale: summary for scale in ('S', 'M', 'L')}
    decision, gates = terminal_decision(True, summary, summary, by_scale)
    assert decision == 'NGAS_A1_PASS_DEVELOPMENT'
    assert all(gates['no_material_scale_collapse'].values())
    failed, _ = terminal_decision(True, summary, {'mean_relative_gain_percent': -1.01}, by_scale)
    assert failed == 'NGAS_A1_REVISE_QUALITY'
    invalid, _ = terminal_decision(False, summary, summary, by_scale)
    assert invalid == 'NGAS_A1_6_INVALID_COMPARISON'


def test_anytime_metrics_are_causal_steps_and_do_not_invent_early_values():
    trace = [
        {'elapsed_time': 2., 'decoder_evaluations': 1, 'current_best_makespan': 100.},
        {'elapsed_time': 5., 'decoder_evaluations': 3, 'current_best_makespan': 80.},
    ]
    assert incumbent_at(trace, 1.) is None
    assert incumbent_at(trace, 4.)['current_best_makespan'] == 100.
    auc = right_continuous_auc(trace, 10., 100.)
    assert auc['available_budget_fraction'] == pytest.approx(.8)
    assert auc['auc_available_horizon'] == pytest.approx(.7)
    assert auc['mean_normalized_incumbent_over_available_horizon'] == pytest.approx(.875)


def test_holm_adjustment_preserves_input_order_and_monotonicity():
    assert holm_adjust([.04, .01, .03, .20]) == pytest.approx([.09, .04, .09, .20])


def test_formal_raw_is_created_once_and_only_valid_payloads_are_resumable(tmp_path):
    task = {
        'instance_id': 'i1', 'instance_sha256': 'abc',
        'seed': 746101, 'budget_seconds': 10.,
    }
    payload = {
        'schema': 'ngas-a16-formal-run-v1', 'status': 'COMPLETE',
        'protocol_sha256': 'protocol', 'algorithm_id': 'NGAS_A1_6',
        'instance_id': 'i1', 'instance_sha256': 'abc', 'seed': 746101,
        'budget_seconds': 10., 'budget_accounting': 'A16_INSTANCE_TOTAL',
        'feasible': True, 'r13_accessed': False, 'r14_accessed': False,
        'gurobi_run': False,
    }
    path = tmp_path / 'raw.json'
    write_new_json(path, payload)
    validate_raw_contract(payload, task, 'protocol')
    with pytest.raises(FileExistsError):
        write_new_json(path, payload)
    with pytest.raises(RuntimeError):
        validate_raw_contract({**payload, 'seed': 746102}, task, 'protocol')


def test_frozen_comparator_and_bks_hashes_are_reusable_and_unchanged():
    root = Path(__file__).resolve().parents[2]
    config_path = root / 'configs/ngas_a16_solver_comparison_v1.json'
    config = json.loads(config_path.read_text())
    audit = audit_frozen_inputs(root, config)
    assert audit['status'] == 'PASS'
    assert audit['checks']['all_comparators_pass']
    assert audit['bks_sha256'] == digest(root / config['bks']['path'])
