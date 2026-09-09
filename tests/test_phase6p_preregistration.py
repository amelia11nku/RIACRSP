import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6p_adaptive_portfolio_v1.json"
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/preregistration"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase6p_policy_is_frozen_before_solver_outcomes() -> None:
    config = load_json(CONFIG)
    protocol = load_json(OUT / "preregistration.json")
    assert config["status"] == "PREREGISTERED_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME"
    assert protocol["status"] == "FROZEN_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME"
    assert protocol["solver_quality_outcomes_observed"] is False
    assert protocol["config_sha256"] == digest(CONFIG)
    assert config["primary"]["candidate_bank"]["shortlist_k"] == 6
    assert config["primary"]["portfolio"]["neural_rank_prior"] == (
        "1 / rank for ranks 1 through 6"
    )
    assert config["primary"]["portfolio"]["mapped_destroy_weight"].startswith("mean ")
    assert config["primary"]["repair"]["candidate_trials"] == 8
    assert config["primary"]["critic"]["precision"] == "FP32_ONLY"
    assert config["primary"]["critic"]["historical_score_calls"] == 0
    assert config["primary"]["critic"]["phase6o_critic_calls"] == 0
    assert config["optional_preregistered_rescue"] is None


def test_phase6p_frozen_manifests_are_complete() -> None:
    protocol = load_json(OUT / "preregistration.json")
    names = {
        "checkpoint_manifest_sha256": "phase6n_checkpoint_manifest.json",
        "instance_manifest_sha256": "development_instance_manifest.json",
        "command_manifest_sha256": "command_manifest.json",
        "source_hashes_sha256": "source_hashes.json",
    }
    for field, name in names.items():
        assert protocol[field] == digest(OUT / name)

    checkpoints = load_json(OUT / "phase6n_checkpoint_manifest.json")
    assert checkpoints["status"] == "FROZEN_BEFORE_PHASE6P_SOLVER_OUTCOMES"
    assert len(checkpoints["checkpoints"]) == 9
    assert {
        (row["training_seed"], row["held_fold"])
        for row in checkpoints["checkpoints"]
    } == {
        (seed, fold) for seed in (726101, 726102, 726103) for fold in range(3)
    }
    assert checkpoints["qualification_boundary"].endswith(
        "no Phase 6N deployable bundle was created"
    )

    instances = load_json(OUT / "development_instance_manifest.json")
    assert instances["instance_count"] == 18
    assert len(instances["instances"]) == 18
    assert instances["development_seeds"] == [746101, 746102, 746103]
    assert instances["formal_seeds"] == [746101, 746102, 746103, 746104, 746105]
    assert {row["scale"] for row in instances["instances"]} == {"S", "M", "L"}
    assert {row["CF_level"] for row in instances["instances"]} == {"CF1", "CF2", "CF3"}


def test_phase6p_holdout_and_comparator_guards_are_frozen() -> None:
    config = load_json(CONFIG)
    commands = load_json(OUT / "command_manifest.json")
    assert config["development"]["budget_seconds"] == "2 * instance.num_operations"
    assert config["formal_r12"]["development_seed_subset"] == config["development"]["seeds"]
    assert config["baseline_registry"]["audit_before_any_comparator_execution"] is True
    assert commands["commands"][1]["stage"] == "P3_comparator_reuse_audit"
    assert commands["comparator_guard"] == (
        "do not execute any comparator before comparator_reuse_audit"
    )
    assert config["boundaries"]["gurobi"] is False
    assert config["boundaries"]["r13"] == "LOCKED_UNTIL_FORMAL_R12_PASSES"
    assert config["boundaries"]["r14"] == (
        "LOCKED_UNTIL_PREREGISTERED_R13_CONDITION_PASSES"
    )
