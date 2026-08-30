from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c import (
    candidate_sha256, candidate_to_json, reconstruct_state_from_instance,
)
from rcias_clgri.data.phase6c_io import (
    atomic_write_json, atomic_write_parquet, remove_partial_files, sha256_file,
)
from rcias_clgri.data.phase6c_contract import (
    ANALYSIS_ONLY, FORBIDDEN_FUTURE_INFORMATION, IDENTIFIER_ONLY, LABEL_ONLY,
    MODEL_INPUT_ALLOWED, TABLE_FIELDS,
)
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_clgri.search.counterfactual import evaluate_counterfactual, stable_seed
from rcias_clgri.search.phase6c import (
    aggregate_repair_outcomes, generate_revised_target_arms, pairwise_preference, target_set_id,
)
from scripts.audit_phase6c_integrity import integrity_passed, normalize_boolean_record

ROOT = Path(__file__).resolve().parents[1]


def decoded_state():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    actions = solve_dispatching(instance, "H1").actions
    candidate = candidate_from_actions(instance, actions)
    return instance, candidate, decode_candidate(instance, candidate)


def test_revised_arm_design_is_outcome_blind_deduplicated_and_fixed_count():
    instance, _, decoded = decoded_state()
    count = max(2, round(instance.num_operations * .15))
    result = generate_revised_target_arms(instance, decoded, "state", count, 667000000)
    replay = generate_revised_target_arms(instance, decoded, "state", count, 667000000)
    assert result.requested_arm_count == 24
    assert result.arms == replay.arms
    assert result.unique_arm_count + result.duplicate_arm_count == 24
    assert len({arm.destroyed_operations for arm in result.arms}) == len(result.arms)
    assert all(len(arm.destroyed_operations) == count for arm in result.arms)
    assert {proposal.arm_family for proposal in result.proposals} >= {
        "ORIGINAL_OPERATOR", "RELATED_VARIANT", "MATCHED_RANDOM",
        "LOCAL_PERTURBATION", "STRUCTURED_NEAR_NEIGHBOR",
    }


def test_three_seed_aggregation_and_conditional_pair_label():
    rows = [
        {"repair_seed": 11, "counterfactual_makespan": 9, "absolute_improvement": 1, "relative_improvement": .1},
        {"repair_seed": 12, "counterfactual_makespan": 10, "absolute_improvement": 0, "relative_improvement": 0},
        {"repair_seed": 13, "counterfactual_makespan": 11, "absolute_improvement": -1, "relative_improvement": -.1},
    ]
    aggregate = aggregate_repair_outcomes(rows)
    assert aggregate["mean_relative_improvement"] == pytest.approx(0)
    assert aggregate["improvement_probability"] == pytest.approx(1 / 3)
    assert aggregate["positive_under_1_of_3"] and not aggregate["positive_under_2_of_3"]
    assert pairwise_preference(.01) == 1
    assert pairwise_preference(-.01) == -1
    assert pairwise_preference(0) == 0
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_repair_outcomes(rows[:2])


def test_semantic_state_reconstruction_uses_only_current_record():
    instance, candidate, decoded = decoded_state()
    serialized = candidate_to_json(candidate)
    record = {
        "state_id": "semantic_state", "current_candidate": serialized,
        "candidate_sha256": candidate_sha256(serialized), "current_makespan": decoded.makespan,
        "search_progress": .25, "search_stage": "20-40%",
    }
    reconstructed = reconstruct_state_from_instance(instance, record)
    assert reconstructed.candidate == candidate
    assert reconstructed.decoded.schedule.to_dict() == decoded.schedule.to_dict()
    broken = dict(record, current_makespan=decoded.makespan + 1)
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        reconstruct_state_from_instance(instance, broken)


def test_phase6c_repair_seed_determinism_order_invariance_and_target_ids():
    instance, candidate, decoded = decoded_state()
    result = generate_revised_target_arms(instance, decoded, "state", 2, 667000000)
    forward = {}
    for arm in result.arms:
        seed = stable_seed("state", arm.target_set_id, "repair_group", 0, namespace=668000000)
        forward[arm.target_set_id] = evaluate_counterfactual(
            instance, candidate, decoded, arm.destroyed_operations, "transport_aware", seed, 2,
        ).counterfactual.makespan
        assert arm.target_set_id == target_set_id("state", arm.destroyed_operations)
    reverse = {}
    for arm in reversed(result.arms):
        seed = stable_seed("state", arm.target_set_id, "repair_group", 0, namespace=668000000)
        reverse[arm.target_set_id] = evaluate_counterfactual(
            instance, candidate, decoded, arm.destroyed_operations, "transport_aware", seed, 2,
        ).counterfactual.makespan
    assert forward == reverse


def test_atomic_parquet_checksum_and_partial_recovery(tmp_path):
    frame = pd.DataFrame({"state_id": ["a", "b"], "value": [1, 2]})
    path = tmp_path / "states.parquet"
    atomic_write_parquet(frame, path)
    first = sha256_file(path)
    atomic_write_parquet(frame, path)
    assert sha256_file(path) == first
    partial = tmp_path / ".states.parquet.partial.999"
    partial.write_text("incomplete")
    assert remove_partial_files(tmp_path) == 1
    assert not partial.exists()
    status = tmp_path / "status.json"
    atomic_write_json({"status": "COMPLETE", "sha256": first}, status)
    assert json.loads(status.read_text())["sha256"] == first


def test_phase6c_seed_and_split_contracts_exclude_frozen_suites():
    manifest = pd.read_csv(ROOT / "instances/controlled/RCIAS-CB1-TRAIN/manifests/train_instance_manifest.csv")
    trajectory_seeds = {
        stable_seed(instance_id, run, namespace=666000000)
        for instance_id in manifest.instance_id for run in (1, 2)
    }
    assert len(trajectory_seeds) == 810
    assert not trajectory_seeds & set(manifest.trajectory_seed)
    assert manifest.groupby("instance_id").training_split.nunique().max() == 1
    assert not manifest.instance_id.str.contains("DEV|CORE|SENSITIVITY|LEGACY", case=False).any()


def test_every_contract_field_has_one_legal_leakage_classification():
    legal = {MODEL_INPUT_ALLOWED, LABEL_ONLY, IDENTIFIER_ONLY, ANALYSIS_ONLY, FORBIDDEN_FUTURE_INFORMATION}
    assert TABLE_FIELDS
    for fields in TABLE_FIELDS.values():
        assert fields
        assert set(fields.values()) <= legal
    assert TABLE_FIELDS["target_set_aggregates"]["mean_relative_improvement"] == LABEL_ONLY
    assert TABLE_FIELDS["target_membership"]["operation_slack"] == MODEL_INPUT_ALLOWED


def test_integrity_flags_are_normalized_to_json_safe_booleans():
    flags = normalize_boolean_record({
        "schema": "test-schema-v1",
        "pandas_comparison": pd.Series([0]).sum() == 0,
        "future_information_stored": False,
    })
    assert flags["schema"] == "test-schema-v1"
    assert type(flags["pandas_comparison"]) is bool
    assert integrity_passed(flags)
    assert json.loads(json.dumps(flags)) == {
        "schema": "test-schema-v1",
        "pandas_comparison": True,
        "future_information_stored": False,
    }


def test_gate_a_shard_resume_and_three_seed_integrity_when_available():
    root = ROOT / "outputs/phase6c/gates/gate_a/dataset"
    status_paths = sorted(root.glob("*/*/status.json"))
    if not status_paths:
        pytest.skip("Phase 6C Gate A has not been generated")
    statuses = [json.loads(path.read_text()) for path in status_paths]
    assert len(statuses) == 81
    assert sum(status["state_count"] for status in statuses) == 1000
    assert all(status["repair_seed_row_count"] == 3 * status["arm_count"] for status in statuses)
    assert all(status["status"] == "COMPLETE" for status in statuses)


def test_final_state_manifest_integrity_when_available():
    path = ROOT / "outputs/phase6c/manifests/state_manifest.csv"
    if not path.exists():
        pytest.skip("full Phase 6C state selection has not completed")
    states = pd.read_csv(path)
    assert len(states) == states.state_id.nunique() == 100000
    assert states.training_split.value_counts().to_dict() == {
        "TRAIN": 60000, "TRAIN_VALIDATION": 20000, "TRAIN_INTERNAL_HOLDOUT": 20000,
    }
    assert states.groupby("instance_id").training_split.nunique().max() == 1
    assert states.groupby("training_split").apply(
        lambda part: len(part[["scale", "CF_level", "RI_level", "TI_level"]].drop_duplicates())
    ).eq(81).all()
