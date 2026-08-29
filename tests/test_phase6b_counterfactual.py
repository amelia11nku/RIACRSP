from __future__ import annotations
import json
from pathlib import Path
import random

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.instances.controlled_generator import generate_candidate, scale_sensitivity_variant
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_clgri.search.counterfactual import (
    evaluate_counterfactual, generate_target_arms, stable_seed, swap_target,
)

ROOT = Path(__file__).resolve().parents[1]


def state():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    h1 = solve_dispatching(instance, "H1")
    candidate = candidate_from_actions(instance, h1.actions)
    return instance, candidate, decode_candidate(instance, candidate)


def test_counterfactual_is_immutable_deterministic_and_rng_isolated():
    instance, candidate, decoded = state()
    before_candidate = candidate
    before_schedule = json.dumps(decoded.schedule.to_dict(), sort_keys=True)
    global_state = random.getstate()
    arguments = (instance, candidate, decoded, tuple(instance.operations[:2]), "transport_aware", 665001, 8)
    first = evaluate_counterfactual(*arguments); second = evaluate_counterfactual(*arguments)
    assert first.counterfactual.candidate == second.counterfactual.candidate
    assert first.counterfactual.makespan == second.counterfactual.makespan
    assert first.decoder_evaluations == 8
    assert candidate == before_candidate
    assert json.dumps(decoded.schedule.to_dict(), sort_keys=True) == before_schedule
    assert random.getstate() == global_state


def test_counterfactual_sign_fixed_repair_and_count():
    instance, candidate, decoded = state()
    result = evaluate_counterfactual(instance, candidate, decoded, tuple(instance.operations[:2]), "transport_aware", 1, 2)
    assert result.absolute_improvement == decoded.makespan - result.counterfactual.makespan
    assert result.improved == (result.absolute_improvement > 0)
    with pytest.raises(ValueError, match="transport_aware"):
        evaluate_counterfactual(instance, candidate, decoded, tuple(instance.operations[:2]), "greedy", 1, 2)


def test_target_arm_generation_deduplication_order_invariance_and_swaps():
    instance, candidate, decoded = state()
    count = max(2, round(instance.num_operations * .15))
    arms = generate_target_arms(instance, decoded, "state", count, 664000000)
    assert 1 < len(arms) <= 14
    assert len({arm.destroyed_operations for arm in arms}) == len(arms)
    assert sum(len(arm.duplicate_origin_labels) for arm in arms) == 14 - len(arms)
    assert all(len(arm.destroyed_operations) == count for arm in arms)
    forward = {arm.arm_id: evaluate_counterfactual(instance, candidate, decoded, arm.destroyed_operations, "transport_aware", stable_seed("state", arm.arm_id, 0), 2).counterfactual.makespan for arm in arms}
    reverse = {arm.arm_id: evaluate_counterfactual(instance, candidate, decoded, arm.destroyed_operations, "transport_aware", stable_seed("state", arm.arm_id, 0), 2).counterfactual.makespan for arm in reversed(arms)}
    assert forward == reverse
    reference = arms[0].destroyed_operations
    outside = next(operation for operation in instance.operations if operation not in reference)
    swapped = swap_target(reference, reference[0], outside)
    assert len(swapped) == len(reference) and outside in swapped and reference[0] not in swapped


def test_training_distribution_factorial_seed_isolation_and_deterministic_generation():
    path = ROOT / "instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv"
    if not path.exists(): pytest.skip("training distribution not generated")
    manifest = pd.read_csv(path)
    assert len(manifest) == 405
    assert manifest.training_split.value_counts().to_dict() == {"TRAIN": 243, "TRAIN_VALIDATION": 81, "TRAIN_INTERNAL_HOLDOUT": 81}
    assert len(manifest.groupby(["scale", "CF_level", "RI_level", "TI_level"])) == 81
    assert manifest.trajectory_seed.is_unique
    assert manifest.state_sampling_seed.is_unique
    assert not set(manifest.trajectory_seed) & set(range(610001, 610011))
    sample = manifest.sort_values("instance_id").iloc[0]
    spec = json.loads((ROOT / "configs/rcias_cb1_generation.json").read_text())
    base = generate_candidate(sample.base_structure, "TRAIN_BASE", sample.scale, sample.CF_level, int(sample.final_generation_seed), spec)
    regenerated = scale_sensitivity_variant(base, sample.instance_id, sample.RI_level, sample.TI_level, spec)
    regenerated["meta"].update({"suite": "TRAIN_ONLY", "training_split": sample.training_split, "base_structure": sample.base_structure})
    stored = json.loads((ROOT / "instances/controlled/RCIAS-CB1-TRAIN" / sample.relative_path).read_text())
    assert regenerated == stored


def test_observer_side_counterfactual_does_not_change_alns_trajectory():
    instance, _, _ = state(); config = ALNSConfig(candidate_trials=2, iteration_limit=12)
    baseline = []
    solve_alns(instance, 60, 700001, config, lambda event: baseline.append((event["destroy_operator"], event["repair_operator"], event["candidate"].makespan, event["accepted"])))
    observed = []
    def observer(event):
        current = event["current_before"]
        evaluate_counterfactual(instance, current.candidate, current, tuple(instance.operations[:2]), "transport_aware", 665999, 2)
        observed.append((event["destroy_operator"], event["repair_operator"], event["candidate"].makespan, event["accepted"]))
    solve_alns(instance, 60, 700001, config, observer)
    assert observed == baseline


def test_repair_seed_repetition_is_replayable():
    instance, candidate, decoded = state(); removed = tuple(instance.operations[:2])
    repeated = [evaluate_counterfactual(instance, candidate, decoded, removed, "transport_aware", seed, 2) for seed in (665001, 665002, 665003)]
    replayed = [evaluate_counterfactual(instance, candidate, decoded, removed, "transport_aware", seed, 2) for seed in (665001, 665002, 665003)]
    assert [result.counterfactual.makespan for result in repeated] == [result.counterfactual.makespan for result in replayed]


def test_instance_grouped_splitting_has_no_instance_overlap():
    manifest = pd.read_csv(ROOT / "instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv")
    pilot = manifest[manifest.replicate == "R01"].reset_index(drop=True)
    for train, test in GroupKFold(5).split(pilot, groups=pilot.instance_id):
        assert not set(pilot.iloc[train].instance_id) & set(pilot.iloc[test].instance_id)


def test_information_leakage_outcomes_are_not_model_inputs():
    path = ROOT / "outputs/phase6b/audit/information_leakage_audit.csv"
    if not path.exists(): pytest.skip("Phase 6B analysis not generated")
    audit = pd.read_csv(path).set_index(["table", "field"])
    for field in ("counterfactual_makespan", "absolute_improvement", "relative_improvement", "rank_within_state", "best_arm"):
        assert audit.loc[("counterfactual_arm_results", field), "classification"] == "LABEL_ONLY"
