#!/usr/bin/env python3
"""Freeze the selected Phase 6I policy and code before R10 translation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
COMMAND_MANIFEST = ROOT / "configs/phase6i_mr_command_manifest.json"
SELECTION_PATH = ROOT / "outputs/phase6i_mr/r10_selection/selection_decision.json"
REGISTRY_PATH = ROOT / "outputs/phase6i_mr/pre_r10/candidate_registry.json"
PRE_R10_FREEZE = ROOT / "outputs/phase6i_mr/pre_r10/pre_r10_freeze.json"
CACHE_INTEGRITY = ROOT / "outputs/phase6i_mr/r10_selection/cache/cache_integrity.json"
TRAINING_DATA_FREEZE = ROOT / "outputs/phase6i_mr/training_data/training_data_freeze.json"
TRAINING_CONSTANTS = ROOT / "outputs/phase6i_mr/training_data/training_constants.json"
POLICY_PATH = ROOT / "outputs/phase6i_mr/frozen/r10_translation_policy.json"
AUTHORIZATION_PATH = ROOT / "outputs/phase6i_mr/frozen/r10_translation_authorization.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    if POLICY_PATH.exists() or AUTHORIZATION_PATH.exists():
        if not (POLICY_PATH.exists() and AUTHORIZATION_PATH.exists()):
            raise RuntimeError("partial R10 translation freeze exists")
        authorization = load_json(AUTHORIZATION_PATH)
        if authorization.get("translation_policy_sha256") != digest(POLICY_PATH):
            raise RuntimeError("existing R10 translation policy hash mismatch")
        print(json.dumps({
            "event": "r10_translation_freeze_already_exists",
            "translation_policy_sha256": digest(POLICY_PATH),
        }), flush=True)
        return

    config = load_json(CONFIG_PATH)
    command_manifest = load_json(COMMAND_MANIFEST)
    selection = load_json(SELECTION_PATH)
    registry = load_json(REGISTRY_PATH)
    pre_r10 = load_json(PRE_R10_FREEZE)
    cache = load_json(CACHE_INTEGRITY)
    training_freeze = load_json(TRAINING_DATA_FREEZE)
    constants = load_json(TRAINING_CONSTANTS)
    selected = selection.get("selected")
    if not all([
        selection.get("status") == "SELECTED_IMMUTABLY_FOR_R10_TRANSLATION",
        selection.get("r10_refit_performed") is False,
        selection.get("r11_accessed") is False,
        selected is not None,
        pre_r10.get("status") == "PASS",
        pre_r10.get("r11_accessed") is False,
        cache.get("status") == "PASS",
        cache.get("r11_accessed") is False,
        training_freeze.get("status") == "FROZEN_BEFORE_MODEL_FIT_AND_R10_ACCESS",
        training_freeze.get("r11_accessed") is False,
        command_manifest.get("status")
        == "FROZEN_BEFORE_FORMAL_R09_COLLECTION_OR_R10_R11_CONTENT_ACCESS",
    ]):
        raise RuntimeError("selected policy inputs are not eligible for translation freeze")
    for path, expected in selection["output_hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"R10 selection output hash mismatch: {path}")
    if selection["input_hashes"]["candidate_registry"] != digest(REGISTRY_PATH):
        raise RuntimeError("candidate registry hash mismatch")
    candidate = next(
        item for item in registry["immediate_candidates"]
        if item["candidate_id"] == selected["candidate_id"]
    )
    if candidate["model_family"] not in {"U1", "U2"}:
        raise RuntimeError("selected live deployment family is unsupported")
    if candidate["ensemble_rule"] != (
        "arithmetic mean of the three seed-specific normalized utility outputs"
    ):
        raise RuntimeError("selected model is not the preregistered three-seed ensemble")
    for model in candidate["artifacts"]:
        if digest(ROOT / model["checkpoint_path"]) != model["checkpoint_sha256"]:
            raise RuntimeError("selected model checkpoint hash mismatch")
        if digest(ROOT / model["record_path"]) != model["record_sha256"]:
            raise RuntimeError("selected model record hash mismatch")
    calibration_path = ROOT / candidate["calibration_path"]
    support_path = ROOT / registry["support_bounds_path"]
    if digest(calibration_path) != candidate["calibration_sha256"]:
        raise RuntimeError("selected calibration hash mismatch")
    if digest(support_path) != registry["support_bounds_sha256"]:
        raise RuntimeError("selected support-bounds hash mismatch")
    expected_constants = training_freeze["frozen_outputs"]["training_constants"]["sha256"]
    if digest(TRAINING_CONSTANTS) != expected_constants:
        raise RuntimeError("training constants hash mismatch")
    calibration = load_json(calibration_path)
    support = load_json(support_path)
    experiment_freeze = ROOT / config["locked_inputs"]["phase6f_experiment_freeze"]
    code_paths = [
        "rcias_clgri/ni/phase6i_live_inference.py",
        "scripts/run_phase6i_mr_r10_translation.py",
        "scripts/freeze_phase6i_mr_r10_translation.py",
    ]
    code_hashes = {path: digest(ROOT / path) for path in code_paths}
    policy = {
        "schema": "phase6i-mr-r10-translation-policy-v1",
        "status": "FROZEN_FOR_R10_TRANSLATION_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_name": "PHASE6I_MR_U2_MIXED_OLD_NEW_3SEED",
        "candidate_id": candidate["candidate_id"],
        "model_family": candidate["model_family"],
        "data_variant": candidate["data_variant"],
        "ensemble_rule": "ARITHMETIC_MEAN_THREE_TRAINING_SEEDS",
        "model_artifacts": candidate["artifacts"],
        "base_experiment_freeze_path": str(experiment_freeze.relative_to(ROOT)),
        "base_experiment_freeze_sha256": digest(experiment_freeze),
        "base_checkpoint_sha256": config["locked_inputs"]["phase6f_checkpoint_sha256"],
        "tensor_schema_hash": cache["tensor_schema_hash"],
        "probability_calibrator": calibration["probability_calibrator"],
        "utility_calibrator": calibration["utility_calibrator"],
        "threshold_set_id": selected["threshold_set_id"],
        "thresholds": {
            "probability": selected["probability_threshold"],
            "utility": selected["utility_threshold"],
        },
        "support_rule": "ALL_19_R09_LITERAL_RAW_CONTEXT_FEATURES_IN_RANGE",
        "support_bounds": support["bounds"],
        "context_feature_order": constants["context_feature_order"],
        "context_normalization": constants["context_normalization"],
        "proposal_seed_namespace": config["rng_namespaces"]["frozen_live_proposal"],
        "candidate_bank_contract": config["search"]["candidate_bank_contract"],
        "source_hashes": {
            "config": digest(CONFIG_PATH),
            "command_manifest": digest(COMMAND_MANIFEST),
            "selection_decision": digest(SELECTION_PATH),
            "candidate_registry": digest(REGISTRY_PATH),
            "pre_r10_freeze": digest(PRE_R10_FREEZE),
            "r10_cache_integrity": digest(CACHE_INTEGRITY),
            "training_data_freeze": digest(TRAINING_DATA_FREEZE),
            "training_constants": digest(TRAINING_CONSTANTS),
            "calibration": digest(calibration_path),
            "support_bounds": digest(support_path),
        },
        "code_hashes": code_hashes,
        "r09_accessed": True,
        "r10_accessed": True,
        "r10_refit_performed": False,
        "r11_accessed": False,
        "r11_unlock": False,
    }
    atomic_json(policy, POLICY_PATH)
    authorization = {
        "schema": "phase6i-mr-r10-translation-authorization-v1",
        "status": "FROZEN_BEFORE_R10_TRANSLATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "access_limit": "ONE_COMPLETE_RESUMABLE_R10_TRANSLATION_PASS",
        "translation_started": False,
        "r11_accessed": False,
        "config_sha256": digest(CONFIG_PATH),
        "selection_decision_sha256": digest(SELECTION_PATH),
        "translation_policy_sha256": digest(POLICY_PATH),
        "code_hashes": code_hashes,
    }
    atomic_json(authorization, AUTHORIZATION_PATH)
    print(json.dumps({
        "status": authorization["status"],
        "translation_policy_sha256": digest(POLICY_PATH),
        "authorization_sha256": digest(AUTHORIZATION_PATH),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
