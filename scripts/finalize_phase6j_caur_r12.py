#!/usr/bin/env python3
"""Audit completed R12 evidence and close a failed gate without rerunning it."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_phase6j_caur_deployment as deploy  # noqa: E402
from scripts.audit_phase6j_caur_readiness import audit_j3, audit_regular_training  # noqa: E402

r = deploy.r
FINAL = ROOT / "outputs/phase6j_caur/final"
AUDIT_PATH = deploy.OUT / "completion_integrity_audit.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_latency(table, states, settings, report):
    """Recompute the exact registered population; never filter slow samples."""
    require(len(states) == settings["states"] and not states.state_id.duplicated().any(), "invalid source state population")
    expected = pd.MultiIndex.from_product([states.state_id, range(settings["measured_repetitions"])],
                                          names=["state_id", "repetition"]).sort_values()
    actual = table.set_index(["state_id", "repetition"]).sort_index()
    pd.testing.assert_index_equal(actual.index, expected)
    expected_scale = states.set_index("state_id").scale.loc[table.state_id].to_numpy()
    require(np.array_equal(table.scale.to_numpy(), expected_scale), "latency scale identities changed")
    times = table[["neural_ms", "cached_total_ms"]].to_numpy(float)
    require(np.isfinite(times).all() and (times > 0).all(), "invalid latency measurements")
    summary = {}
    for name, group in [("overall", table), *list(table.groupby("scale", sort=True))]:
        summary[name] = {}
        for column in ("neural_ms", "cached_total_ms"):
            values = {f"p{q}": float(np.percentile(group[column], q)) for q in (50, 90, 99)}
            for key, value in values.items():
                np.testing.assert_allclose(value, report["latency_ms"][name][column][key], rtol=1e-12, atol=1e-12)
            summary[name][column] = values
    caps = settings["caps"]
    neural_pass = summary["overall"]["neural_ms"]["p90"] <= caps["p90_neural_decision_ms_max"]
    cached_pass = summary["overall"]["cached_total_ms"]["p90"] <= caps["p90_total_decision_ms_max"]
    expected_status = ("CACHED_PROFILE_COMPLETE_LIVE_TOTAL_PENDING" if neural_pass and cached_pass
                       else "CACHED_LATENCY_GATE_FAILED_BEFORE_R13")
    require(report["neural_cap_pass"] is neural_pass and report["cached_total_cap_pass"] is cached_pass
            and report["status"] == expected_status, "recorded latency gate disagrees with raw samples")
    require(report["states"] == len(states) and report["measured_decisions"] == len(table)
            and report["scope"] == settings, "recorded latency population or scope changed")
    require(report["live_total_cap_pass"] is None and report["r13_eligible"] is False,
            "cached measurements cannot certify live total latency or R13 eligibility")
    return summary


def rejection_reasons(readiness, report):
    reasons = {family: list(row["failed_checks"]) for family, row in readiness["families"].items()}
    eligible = [family for family, row in readiness["families"].items() if row["data_gate_pass"]]
    require(eligible == readiness["data_eligible_families"] == [deploy.FAMILY],
            "this terminal audit requires J1 as the only data-eligible family")
    if not report["neural_cap_pass"]:
        reasons[deploy.FAMILY].append("p90_neural_decision_latency_exceeds_frozen_cap")
    if not report["cached_total_cap_pass"]:
        reasons[deploy.FAMILY].append("cached_total_latency_already_exceeds_full_total_cap")
    require(all(reasons.values()), "no terminal rejection: complete missing live validation instead")
    return reasons


def audit_deployment():
    protocol, parent = deploy.validate_protocol(), r.validate_protocol()
    sha = r.digest(deploy.PROTOCOL_PATH)
    paths = [deploy.PROTOCOL_PATH, deploy.READINESS]
    require(audit_regular_training() == r.load_json(deploy.j3.REGULAR_AUDIT), "regular OOF audit changed")
    require(audit_j3() == r.load_json(deploy.READINESS.parent / "j3_completion_integrity_audit.json"), "J3 OOF audit changed")
    paths.extend([deploy.j3.REGULAR_AUDIT, deploy.READINESS.parent / "j3_completion_integrity_audit.json"])
    collection_path = r.COLLECTION / "collection_integrity.json"
    collection = r.load_json(collection_path)
    cache = r.load_json(r.CACHE / "tensor_cache_integrity.json")
    require(collection["status"] == cache["status"] == "PASS" and all(collection["checks"].values())
            and all(cache["checks"].values()) and cache["collection_integrity_sha256"] == r.digest(collection_path),
            "collection/feasibility integrity boundary changed")
    paths.extend([collection_path, r.CACHE / "tensor_cache_integrity.json"])
    for filename, key in (("r12_seed_labels.parquet", "r12_seed_labels_sha256"),
                          ("r12_grouped_labels.parquet", "r12_grouped_labels_sha256"),
                          ("collection_shard_manifest.csv", "shard_manifest_sha256")):
        path = r.COLLECTION / filename
        require(r.digest(path) == collection[key], f"collection artifact changed: {filename}")
        paths.append(path)
    progress, worker, launch, report = [r.load_json(deploy.OUT / name) for name in
                                      ("progress.json", "worker_status.json", "launch_record.json", "cached_latency_report.json")]
    require(worker["status"] == "COMPLETE" and worker["exit_code"] == 0
            and progress["completed_seeds"] == progress["expected_seeds"] == 3
            and progress["profiled_states"] == report["states"] == 288
            and progress["worker_pid"] == worker["pid"] == launch["pid"]
            and progress["status"] == report["status"], "worker completion boundary failed")
    require(all(x["protocol_sha256"] == sha for x in (progress, launch, report)), "worker protocol hash mismatch")
    require(all(x["r13_accessed"] is False and x["r14_accessed"] is False for x in (progress, report)),
            "deployment access declaration failed")
    require(report["scope"]["caps"] == r.load_json(r.CONFIG_PATH)["runtime"], "latency caps changed")
    require(protocol["epoch_plan"] == deploy.epoch_plan(parent), "final-fit epoch plan changed")
    source = pd.read_parquet(r.SOURCE_PATH)
    transform = r.fit_feature_transform(source)
    models, epochs = [], {}
    for seed in parent["training"]["seeds"]:
        model = deploy.load_seed(seed, transform, protocol, parent, torch.device("cpu"))
        models.append(model)
        checkpoint, record_path = deploy.seed_paths(seed)
        record = r.load_json(record_path)
        require(r.digest(checkpoint) == report["seed_checkpoint_sha256"][str(seed)], "profiled checkpoint changed")
        require(np.isfinite(pd.DataFrame(record["history"]).to_numpy(float)).all(), "non-finite training history")
        require([row["epoch"] for row in record["history"]] == list(range(1, len(record["history"]) + 1)),
                "training epochs are missing or duplicated")
        epochs[str(seed)] = len(record["history"])
        paths.extend([checkpoint, record_path])
    ensemble = deploy.SharedFrozenCAUREnsemble(models)
    require(report["physical_shared_parameters"] == sum(p.numel() for p in ensemble.parameters())
            and report["per_seed_parameters"] == [list(m.parameter_counts()) for m in models], "parameter counts changed")
    require(report["three_seed_output_parity"] == "EXACT" and report["maximum_absolute_parity_error"] == 0,
            "frozen worker did not establish exact inference parity")
    table_path = deploy.OUT / "cached_latency_samples.csv"
    require(r.digest(table_path) == report["samples_sha256"], "raw latency sample hash changed")
    table = pd.read_csv(table_path, float_precision="round_trip")
    summary = audit_latency(table, source[["state_id", "scale"]].drop_duplicates(), protocol["latency"], report)
    log_path = ROOT / launch["log_path"]
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    logged_epochs = [row for row in events if row.get("event") == "phase6j_caur_epoch"]
    for seed, count in epochs.items():
        require([row["epoch"] for row in logged_epochs if row["seed"] == int(seed)] == list(range(1, count + 1)),
                "logged epochs disagree with checkpoints")
    require(events[-1]["status"] == report["status"] and events[-1]["profiled_states"] == 288,
            "log has no matching terminal completion")
    require(datetime.fromisoformat(protocol["frozen_at_utc"]) < datetime.fromisoformat(events[0]["updated_at_utc"]),
            "deployment fit began before protocol freeze")
    paths.extend([table_path, log_path, *(deploy.OUT / name for name in
                 ("progress.json", "worker_status.json", "launch_record.json", "cached_latency_report.json"))])
    return {"schema": "phase6j-caur-deployment-completion-audit-v1", "status": "PASS",
            "protocol_sha256": sha, "completed_seeds": 3, "epochs_by_seed": epochs,
            "profiled_states": 288, "measured_decisions": 864, "latency_ms_recomputed": summary,
            "checks": {"prior_oof_audits_reproduced": True, "collection_and_feasibility_evidence_unchanged": True,
                       "frozen_inputs_code_and_caps": True, "checkpoint_reload_and_finite_parameters": True,
                       "normalization_and_fixed_epochs": True, "shared_encoder_identity": True,
                       "worker_log_and_exit": True, "all_latency_samples_unique_complete_positive": True,
                       "quantiles_and_gate_recomputed": True, "r13_r14_locked": True},
            "parity_scope": "all-state exact assertions in the hash-frozen worker; no profiling rerun in this audit",
            "artifact_sha256": {str(p.relative_to(ROOT)): r.digest(p) for p in paths},
            "audit_code_sha256": r.digest(Path(__file__)), "r13_accessed": False, "r14_accessed": False}, report


def publish_once(payloads):
    for path, value in payloads:
        require(not path.exists() or r.load_json(path) == value, f"refusing to overwrite terminal evidence: {path}")
    for path, value in payloads:
        if not path.exists():
            r.atomic_json(value, path)


def main():
    audit, report = audit_deployment()
    reasons = rejection_reasons(r.load_json(deploy.READINESS), report)
    p90 = report["latency_ms"]["overall"]["neural_ms"]["p90"]
    cap = report["scope"]["caps"]["p90_neural_decision_ms_max"]
    decision = {"schema": "phase6j-caur-r12-final-decision-v1", "status": "COMPLETE",
                "decision": "MODEL_REVISION", "stop_boundary": "BEFORE_R13", "integrity": audit["status"],
                "family_rejection_reasons": reasons, "r13_eligible_families": [],
                "selected_model": None, "csg_ni_v1_frozen": False, "phase6h_replacement_authorized": False,
                "neural_p90_ms": p90, "neural_p90_cap_ms": cap, "neural_over_cap_percent": (p90 / cap - 1) * 100,
                "cached_total_p90_ms": report["latency_ms"]["overall"]["cached_total_ms"]["p90"],
                "live_total_latency": "NOT_MEASURED_STOPPED_ON_NEURAL_GATE", "solver_translation": "NOT_RUN",
                "r13_accessed": False, "r14_accessed": False, "r13_locked": True, "r14_locked": True,
                "next_step": "propose one separately preregistered runtime-only revision; no automatic restart or cap relaxation",
                "audit_path": str(AUDIT_PATH.relative_to(ROOT)), "deployment_report_sha256": r.digest(deploy.OUT / "cached_latency_report.json")}
    status = {key: decision[key] for key in ("status", "decision", "stop_boundary", "integrity", "selected_model",
              "csg_ni_v1_frozen", "phase6h_replacement_authorized", "r13_accessed", "r14_accessed", "r13_locked", "r14_locked", "next_step")}
    status.update(schema="phase6j-caur-final-status-v1", decision_path=str((FINAL / "final_decision.json").relative_to(ROOT)))
    publish_once([(AUDIT_PATH, audit), (FINAL / "final_decision.json", decision), (FINAL / "final_status.json", status)])
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
