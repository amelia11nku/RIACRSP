import json
from pathlib import Path

from rcias_clgri.ni.phase6m_selective_risk import (
    CATEGORICAL_COLUMNS,
    STRUCTURAL_NUMERIC_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bottleneck_proxy_schema_correction_is_narrow():
    assert "bottleneck_proxy" in CATEGORICAL_COLUMNS
    assert "bottleneck_proxy" not in STRUCTURAL_NUMERIC_COLUMNS
    base = json.loads((ROOT / "configs/phase6m_selective_confidence_v1.json").read_text())
    assert "bottleneck_proxy" in base["selector_features"]["structural_numeric"]


def test_schema_amendment_preserves_failed_preflight_and_precedes_training():
    amendment = json.loads((
        ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/amendment_m2_schema.json"
    ).read_text())
    failure = json.loads((
        ROOT / amendment["preflight_failure"]["path"]
    ).read_text())
    assert amendment["status"] == "FROZEN_SCHEMA_CORRECTION_BEFORE_FIRST_OPTIMIZER_STEP"
    assert amendment["active_feature_patch"] == {
        "remove_from_structural_numeric": ["bottleneck_proxy"],
        "add_to_categorical_one_hot": ["bottleneck_proxy"],
    }
    assert failure["status"] == "INVALIDATED_BEFORE_FIRST_OPTIMIZER_STEP"
    assert failure["optimizer_steps_started"] is False
    assert failure["outer_oof_generated"] is False


def test_corrected_implementation_protocol_is_frozen_and_locked():
    protocol = json.loads((
        ROOT / "outputs/phase6m_selective_confidence_v1/implementation/implementation_protocol_v2.json"
    ).read_text())
    schema = json.loads((ROOT / protocol["feature_schema"]["path"]).read_text())
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_PHASE6M_OPTIMIZER_STEP"
    assert protocol["inner_ranker_runs"] == 18
    assert protocol["selector_runs"] == 9
    assert protocol["optimizer_steps_started"] is False
    assert protocol["r13_accessed"] is False
    assert protocol["r14_accessed"] is False
    assert "bottleneck_proxy" in schema["categorical_columns"]
    assert "bottleneck_proxy" not in schema["structural_numeric_columns"]
