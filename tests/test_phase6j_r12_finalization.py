from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import finalize_phase6j_caur_r12 as final


def latency_fixture():
    states = pd.DataFrame({"state_id": ["s", "l"], "scale": ["S", "L"]})
    settings = {"states": 2, "measured_repetitions": 3,
                "caps": {"p90_neural_decision_ms_max": 30.0, "p90_total_decision_ms_max": 100.0}}
    table = pd.DataFrame([{"state_id": state, "scale": scale, "repetition": rep,
                           "neural_ms": 32.0 + rep, "cached_total_ms": 38.0 + rep}
                          for state, scale in zip(states.state_id, states.scale) for rep in range(3)])
    summary = {name: {column: {f"p{q}": float(np.percentile(group[column], q)) for q in (50, 90, 99)}
                     for column in ("neural_ms", "cached_total_ms")}
               for name, group in [("overall", table), *list(table.groupby("scale"))]}
    report = {"states": 2, "measured_decisions": 6, "scope": settings, "latency_ms": summary,
              "neural_cap_pass": False, "cached_total_cap_pass": True, "live_total_cap_pass": None,
              "r13_eligible": False, "status": "CACHED_LATENCY_GATE_FAILED_BEFORE_R13"}
    return table, states, settings, report


def test_all_raw_samples_reproduce_latency_quantiles_and_failed_gate():
    table, states, settings, report = latency_fixture()
    assert final.audit_latency(table, states, settings, report) == report["latency_ms"]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "scale", "nan", "negative", "quantile", "gate", "live"])
def test_latency_audit_rejects_incomplete_or_corrupt_evidence(corruption):
    table, states, settings, report = latency_fixture()
    if corruption == "missing":
        table = table.iloc[:-1]
    elif corruption == "duplicate":
        table.loc[1] = table.loc[0]
    elif corruption == "scale":
        table.loc[0, "scale"] = "L"
    elif corruption == "nan":
        table.loc[0, "neural_ms"] = np.nan
    elif corruption == "negative":
        table.loc[0, "neural_ms"] = -1
    elif corruption == "quantile":
        report["latency_ms"]["overall"]["neural_ms"]["p90"] = 29
    elif corruption == "gate":
        report["neural_cap_pass"] = True
    else:
        report["live_total_cap_pass"] = True
    with pytest.raises((RuntimeError, AssertionError)):
        final.audit_latency(table, states, settings, report)


def test_neural_failure_cannot_be_rescued_by_fast_cached_total():
    readiness = {"data_eligible_families": [final.deploy.FAMILY], "families": {
        final.deploy.FAMILY: {"data_gate_pass": True, "failed_checks": []},
        "J2_CONT_LASTBLOCK": {"data_gate_pass": False, "failed_checks": ["no_gate"]},
        "J3_CONT_RELATIONAL": {"data_gate_pass": False, "failed_checks": ["ece", "no_gate"]}}}
    report = {"neural_cap_pass": False, "cached_total_cap_pass": True}
    reasons = final.rejection_reasons(readiness, report)
    assert reasons[final.deploy.FAMILY] == ["p90_neural_decision_latency_exceeds_frozen_cap"]
    assert readiness["families"][final.deploy.FAMILY]["failed_checks"] == []
    report["neural_cap_pass"] = True
    with pytest.raises(RuntimeError, match="no terminal rejection"):
        final.rejection_reasons(readiness, report)


def test_neural_cap_equality_is_inclusive():
    table, states, settings, report = latency_fixture()
    settings["caps"]["p90_neural_decision_ms_max"] = report["latency_ms"]["overall"]["neural_ms"]["p90"]
    report.update(neural_cap_pass=True, status="CACHED_PROFILE_COMPLETE_LIVE_TOTAL_PENDING")
    final.audit_latency(table, states, settings, report)


def test_terminal_publication_is_idempotent_and_refuses_changed_decisions(tmp_path):
    path = tmp_path / "decision.json"
    payload = {"decision": "MODEL_REVISION"}
    final.publish_once([(path, payload)])
    original = path.read_bytes()
    final.publish_once([(path, copy.deepcopy(payload))])
    assert path.read_bytes() == original
    new_path = tmp_path / "new.json"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        final.publish_once([(new_path, {}), (path, {"decision": "PROCEED_FREEZE_V1"})])
    assert not new_path.exists() and json.loads(path.read_text()) == payload


def test_terminal_audit_never_calls_training_or_timing():
    source = (final.ROOT / "scripts/finalize_phase6j_caur_r12.py").read_text()
    assert "optimize_model(" not in source and "deploy.profile(" not in source
    assert "outputs/phase6i_mr/r11_validation" not in source
