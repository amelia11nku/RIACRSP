import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rcias_clgri.data.phase6i_access import (
    Phase6IHoldoutAccessError,
    load_phase6i_instance,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase6i_preregistration_locks_splits_and_seed_namespaces():
    config = json.loads(
        (ROOT / "configs/phase6i_mr_live_utility_revision.json").read_text()
    )
    assert config["status"] == (
        "REVISION_1_2_PREREGISTERED_BEFORE_FORMAL_R09_COLLECTION_OR_R10_R11_CONTENT_ACCESS"
    )
    assert config["instance_suite"]["replicate_split"] == {
        "R09": "LIVE_REV_FIT",
        "R10": "LIVE_REV_SELECT",
        "R11": "LIVE_REV_HOLDOUT",
    }
    assert config["instance_suite"]["r11_access"]["before_freeze"] == (
        "MANIFEST_ID_AND_CONTENT_HASH_ONLY"
    )
    assert config["instance_suite"]["instances_per_split"] == 18
    assert config["instance_suite"]["instances_per_cell"] == 2
    seed_groups = [set(values) for values in config["seeds"].values()]
    for index, group in enumerate(seed_groups):
        assert all(group.isdisjoint(other) for other in seed_groups[index + 1 :])
    assert "GUROBI_OR_TINY_RERUN" in config["forbidden"]


def test_phase6i_instance_manifest_and_integrity_audit_are_disjoint():
    manifest = pd.read_csv(SUITE / "manifests/phase6i_instance_manifest.csv")
    assert len(manifest) == manifest.instance_id.nunique() == 54
    assert manifest.groupby(
        ["live_revision_split", "scale", "CF_level"]
    ).size().eq(2).all()
    assert set(manifest.cell_replicate) == {"C01", "C02"}
    hash_sets = {
        split: set(frame.sha256)
        for split, frame in manifest.groupby("live_revision_split")
    }
    assert hash_sets["LIVE_REV_FIT"].isdisjoint(hash_sets["LIVE_REV_SELECT"])
    assert hash_sets["LIVE_REV_FIT"].isdisjoint(hash_sets["LIVE_REV_HOLDOUT"])
    assert hash_sets["LIVE_REV_SELECT"].isdisjoint(hash_sets["LIVE_REV_HOLDOUT"])
    audit = json.loads(
        (SUITE / "manifests/phase6i_integrity_audit.json").read_text()
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["r11_access_scope"] == "MANIFEST_ID_HASH_SIZE_ONLY"


def test_phase6i_command_manifest_freezes_seed_robust_and_translation_stages():
    commands = json.loads(
        (ROOT / "configs/phase6i_mr_command_manifest.json").read_text()
    )
    stage_ids = {stage["id"] for stage in commands["stages"]}
    assert "08_train_seed_robust_candidates" in stage_ids
    assert "11_r10_solver_translation_once" in stage_ids
    assert "13_r11_once" in stage_ids
    assert commands["protected_access"]["r11"].startswith("reject until")


def test_r11_content_access_is_rejected_without_valid_freeze(tmp_path):
    locked_path = tmp_path / "example_R11.json"
    with pytest.raises(Phase6IHoldoutAccessError, match="locked"):
        load_phase6i_instance(locked_path)

    artifact = tmp_path / "selected_artifact.json"
    artifact.write_text('{"model":"U2"}\n')
    bad_record = tmp_path / "freeze_record.json"
    bad_record.write_text(json.dumps({
        "schema": "phase6i-mr-artifact-freeze-v1",
        "status": "FROZEN_BEFORE_R11",
        "r11_content_accessed": False,
        "artifact_sha256": "not-the-artifact-hash",
    }))
    with pytest.raises(Phase6IHoldoutAccessError, match="invalid"):
        load_phase6i_instance(
            locked_path,
            freeze_record_path=bad_record,
            artifact_path=artifact,
        )


def test_valid_frozen_artifact_unlocks_r11_loader(tmp_path):
    locked_path = tmp_path / "tiny_copy_R11.json"
    locked_path.write_bytes((ROOT / "instances/tiny/tiny_03.json").read_bytes())
    artifact = tmp_path / "selected_artifact.json"
    artifact.write_text('{"model":"U2"}\n')
    record = tmp_path / "freeze_record.json"
    record.write_text(json.dumps({
        "schema": "phase6i-mr-artifact-freeze-v1",
        "status": "FROZEN_BEFORE_R11",
        "r11_content_accessed": False,
        "artifact_sha256": digest(artifact),
    }))
    instance = load_phase6i_instance(
        locked_path,
        freeze_record_path=record,
        artifact_path=artifact,
    )
    assert instance.instance_id == "tiny_03"
