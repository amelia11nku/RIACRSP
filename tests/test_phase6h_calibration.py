import json
import inspect
from pathlib import Path

import numpy as np

from rcias_clgri.analysis.phase6h import (
    first_common_target_hit,
    normalized_gap_auc,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.ni.calibration import FrozenCalibrator, fit_probability_calibrator
from rcias_clgri.ni.live_inference import FrozenLiveInference
from rcias_clgri.ni.live_policy import AlwaysFallbackPolicy
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni


ROOT = Path(__file__).resolve().parents[1]


def test_calibrator_serialization_and_beta_determinism():
    scores = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1])
    calibrator = fit_probability_calibrator(scores, labels, "BETA")
    replay = FrozenCalibrator(**json.loads(json.dumps(calibrator.to_dict())))
    assert np.array_equal(calibrator.predict(scores), replay.predict(scores))
    identity = fit_probability_calibrator(scores, labels, "SIGMOID_IDENTITY")
    assert np.all((identity.predict(scores) > 0) & (identity.predict(scores) < 1))


def test_phase6h_split_is_disjoint_and_holdout_is_unopened_by_config():
    config = json.loads((ROOT / "configs/phase6h_live_calibration.json").read_text())
    manifest = (ROOT / config["calibration_instances"]["manifest"]).read_text()
    assert "CB1_DEV" not in manifest
    assert "CB1_CORE" not in manifest
    assert "CB1_SENS" not in manifest
    assert config["calibration_collection"]["split"] == "CAL_FIT"
    assert config["calibration_collection"]["holdout_access"] == "FORBIDDEN"
    assert set(config["seeds"]["CAL_FIT_COLLECTION"]).isdisjoint(
        config["seeds"]["CAL_HOLDOUT"]
    )
    audit = json.loads((
        ROOT / "instances/controlled/RCIAS-CB1-CAL/manifests/calibration_instance_audit.json"
    ).read_text())
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())


def test_phase6h_exact_protocol_uses_only_tiny_and_cal_fit():
    protocol = json.loads((
        ROOT / "configs/phase6h_exact_validation.json"
    ).read_text())
    assert protocol["status"] == "FROZEN_BEFORE_PHASE6H_EXACT_VALIDATION"
    assert set(protocol["tiny_instances"]) == {"tiny_01", "tiny_03"}
    assert all(
        case["relative_path"].startswith("cal_fit/")
        for case in protocol["additional_cal_fit_small_cases"]
    )
    selected_cases = json.dumps(protocol["additional_cal_fit_small_cases"])
    assert not any(
        forbidden in selected_cases
        for forbidden in ("CB1_CORE", "CB1_SENSITIVITY", "LEGACY_130")
    )


def test_live_inference_api_is_outcome_blind():
    parameters = set(inspect.signature(FrozenLiveInference.decide).parameters)
    assert parameters == {
        "self", "instance", "current", "state_id", "destroy_count",
        "search_progress", "search_stage",
    }


def test_phase6h_fallback_executes_frozen_alns_path():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    events = []
    result = solve_csgni(
        instance,
        60.0,
        671198,
        AlwaysFallbackPolicy(),
        alns_config=ALNSConfig(candidate_trials=2, iteration_limit=3),
        csgni_config=CSGNIConfig(intervention_rate=100),
        observer=events.append,
    )
    assert result.diagnostics["ni_interventions"] == 0
    assert result.diagnostics["ni_fallbacks"] == 3
    assert all(event["alns_weight_credit"] for event in events)
    assert all(event["ni_fallback_reason"] == "POLICY_ABSTAIN" for event in events)


def test_incumbent_trace_reconstruction_and_first_hits():
    trace = [
        {"elapsed_time": 1.0, "decoder_evaluations": 1, "current_best_makespan": 120},
        {"elapsed_time": 4.0, "decoder_evaluations": 10, "current_best_makespan": 110},
        {"elapsed_time": 8.0, "decoder_evaluations": 20, "current_best_makespan": 100},
    ]
    assert validate_incumbent_trace(trace, final_best=100)[-1]["decoder_evaluations"] == 20
    sampled = sample_incumbent_trace(trace, budget=10.0, fractions=(0.05, 0.5, 1.0))
    assert sampled[0]["incumbent_available"] is False
    assert [row["best_makespan"] for row in sampled[1:]] == [110.0, 100.0]
    hit = first_common_target_hit(trace, target_makespan=110)
    assert hit == {
        "reached": True,
        "right_censored": False,
        "elapsed_wall_time": 4.0,
        "decoder_evaluations": 10,
    }
    assert first_common_target_hit(trace, target_makespan=99)["right_censored"] is True
    assert normalized_gap_auc(trace, budget=10.0, reference_makespan=100) > 0
