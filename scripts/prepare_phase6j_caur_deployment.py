#!/usr/bin/env python3
"""Freeze J1 full-R12 fitting and measure cached inference without opening R13."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import choose_caur_action  # noqa: E402
from rcias_clgri.ni.calibration import FrozenCalibrator  # noqa: E402
from rcias_clgri.ni.phase6j_deployment import SharedFrozenCAUREnsemble  # noqa: E402
from scripts import run_phase6j_caur_j3 as j3  # noqa: E402
from scripts.launch_phase6j_caur_j3 import active_training_processes  # noqa: E402

r = j3.regular
FAMILY = "J1_CONT_FROZEN"
OUT = ROOT / "outputs/phase6j_caur/deployment/j1_full_r12"
PROTOCOL_PATH = ROOT / "outputs/phase6j_caur/frozen/r12_j1_deployment_protocol.json"
READINESS = ROOT / "outputs/phase6j_caur/r12_acceptance/data_readiness.json"
CODE_FILES = (
    "scripts/prepare_phase6j_caur_deployment.py",
    "scripts/audit_phase6j_caur_readiness.py",
    "scripts/launch_phase6j_caur_j3.py",
    "rcias_clgri/ni/phase6j_deployment.py",
)


def epoch_plan(parent):
    plan = {}
    for seed in parent["training"]["seeds"]:
        epochs = [r.load_json(r.run_paths(FAMILY, seed, fold)[2])["best_epoch"] for fold in range(3)]
        plan[str(seed)] = {"outer_fold_selected_epochs": epochs, "full_fit_epochs": int(np.median(epochs))}
    return plan


def freeze_protocol():
    if PROTOCOL_PATH.exists():
        return validate_protocol()
    j3.validate_protocol()
    parent = r.validate_protocol()
    readiness = r.load_json(READINESS)
    if readiness["data_eligible_families"] != [FAMILY]:
        raise RuntimeError("this bounded deployment step requires J1 as the only data-eligible family")
    paths = [READINESS, j3.PROTOCOL_PATH, j3.REGULAR_AUDIT,
             READINESS.parent / "j3_completion_integrity_audit.json",
             r.OUT / f"{FAMILY}_oof_summary.json"]
    for seed in parent["training"]["seeds"]:
        paths.extend(r.run_paths(FAMILY, seed, fold)[2] for fold in range(3))
    summary = r.load_json(r.OUT / f"{FAMILY}_oof_summary.json")
    protocol = {
        "schema": "phase6j-caur-j1-deployment-protocol-v1",
        "status": "FROZEN_BEFORE_FULL_R12_FIT", "family": FAMILY,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_sha256": r.digest(r.PROTOCOL_PATH),
        "input_hashes": {str(p.relative_to(ROOT)): r.digest(p) for p in paths},
        "code_hashes": {p: r.digest(ROOT / p) for p in CODE_FILES},
        "implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  capture_output=True, text=True, check=True).stdout.strip(),
        "epoch_rule": "per-seed median of three nested-OOF selected epochs; no validation or early stopping",
        "epoch_plan": epoch_plan(parent), "shuffle_salt": 303,
        "normalization": "fit once on all 288 R12 states; no R13 normalization",
        "calibrator": summary["calibration"]["deployment_calibrator"],
        "gate": summary["selected_gate"], "immediate_harm_floor": parent["gate"]["immediate_harm_floor"],
        "calibration_boundary": "unchanged OOF-selected-winner calibrator and gate; never fit to final-fit predictions",
        "shared_encoder": "identical frozen J1 encoder evaluated once, three independently trained heads; exact parity required",
        "latency": {
            "states": 288, "batch_size": 1, "warmup_decisions": 10, "measured_repetitions": 3,
            "device": "cuda", "synchronize_cuda": True,
            "neural_scope": "on-device full bank; shared encoder and all three heads",
            "cached_total_scope": "cached CPU graph/action tensors and source features through batch, transfer, model, calibration and gate",
            "excluded_from_cached_total": ["live CSG construction", "proposal generation", "frozen-score source features"],
            "live_total_required_before_R13": True,
            "caps": r.load_json(r.CONFIG_PATH)["runtime"],
        },
        "r13_accessed": False, "r14_accessed": False,
    }
    r.atomic_json(protocol, PROTOCOL_PATH)
    return protocol


def validate_protocol():
    j3.validate_protocol()
    protocol = r.load_json(PROTOCOL_PATH)
    if not (protocol["schema"] == "phase6j-caur-j1-deployment-protocol-v1"
            and protocol["status"] == "FROZEN_BEFORE_FULL_R12_FIT"
            and protocol["family"] == FAMILY
            and protocol["r13_accessed"] is False and protocol["r14_accessed"] is False
            and protocol["parent_protocol_sha256"] == r.digest(r.PROTOCOL_PATH)):
        raise RuntimeError("deployment authorization or parent training protocol changed")
    for section in ("input_hashes", "code_hashes"):
        for relative, sha in protocol[section].items():
            if r.digest(ROOT / relative) != sha:
                raise RuntimeError(f"deployment freeze changed: {relative}")
    return protocol


def seed_paths(seed):
    return OUT / f"seed_{seed}.pt", OUT / f"seed_{seed}.json"


def load_seed(seed, transform, protocol, parent, device):
    checkpoint_path, record_path = seed_paths(seed)
    record = r.load_json(record_path)
    if not (record["status"] == "COMPLETE" and record["seed"] == seed
            and record["r13_accessed"] is False and record["r14_accessed"] is False
            and record["protocol_sha256"] == r.digest(PROTOCOL_PATH)
            and record["checkpoint_sha256"] == r.digest(checkpoint_path)):
        raise RuntimeError(f"completed seed hash changed: {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {"model_family": FAMILY, "training_seed": seed,
                "feature_transform": transform.to_dict(), "protocol_sha256": r.digest(PROTOCOL_PATH),
                "base_checkpoint_sha256": parent["base_checkpoint"]["sha256"],
                "epochs": protocol["epoch_plan"][str(seed)]["full_fit_epochs"], "training_states": 288}
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"final-fit checkpoint identity changed: {seed}")
    if len(record["history"]) != expected["epochs"] or any("validation" in k for row in record["history"] for k in row):
        raise RuntimeError("final fit must use fixed epochs without validation")
    model = r.initialize_model(FAMILY, seed, transform, parent, device)
    j3.restore_trainable(model, checkpoint["trainable_model_state"])
    return model.eval()


def model_inputs(packed):
    return {"fallback_action_indices": packed["fallback_indices"],
            "categorical": packed["categorical"], "numeric": packed["numeric"]}


def deployment_decision(output, packed, protocol):
    advantage, logits, immediate = [value.detach().float().cpu().numpy() for value in output]
    if not all(np.isfinite(value).all() for value in (advantage, logits, immediate)):
        raise RuntimeError("non-finite deployment prediction")
    mean, std = advantage.mean(axis=0), advantage.std(axis=0, ddof=0)
    probability = FrozenCalibrator(**protocol["calibrator"]).predict(logits.mean(axis=0))
    immediate = immediate.mean(axis=0)
    frame = packed["frame"]
    rows = [{"target_set_id": str(row.target_set_id), "continuation_advantage_mean": float(mean[i]),
             "continuation_advantage_std": float(std[i]), "beats_fallback_probability": float(probability[i]),
             "supported": bool(packed["supported"][i]), "immediate_utility_prediction": float(immediate[i])}
            for i, row in enumerate(frame.itertuples(index=False))]
    gate = protocol["gate"]
    return choose_caur_action(rows, fallback_target_set_id=str(frame.loc[frame.is_fallback, "target_set_id"].item()),
                             **{k: gate[k] for k in ("p_min", "lcb_lambda", "delta_min")},
                             immediate_harm_floor=protocol["immediate_harm_floor"])


def profile(models, ids, samples, frames, transform, protocol, update):
    ensemble = SharedFrozenCAUREnsemble(models)
    rows, parity_max = [], 0.0
    device = torch.device("cuda")
    settings = protocol["latency"]
    with torch.inference_mode():
        for state_id in ids[:settings["warmup_decisions"]]:
            packed = r.build_batch([state_id], samples, frames, transform, device)
            deployment_decision(ensemble(packed["batch"], **model_inputs(packed)), packed, protocol)
        for i, state_id in enumerate(ids):
            packed = r.build_batch([state_id], samples, frames, transform, device)
            outputs = [model(packed["batch"], **model_inputs(packed)) for model in models]
            reference = tuple(torch.stack([getattr(o, key) for o in outputs])
                              for key in ("advantage", "beats_fallback_logit", "immediate_utility"))
            shared = ensemble(packed["batch"], **model_inputs(packed))
            for actual, expected in zip(shared, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                parity_max = max(parity_max, float((actual - expected).abs().max()))
            reference_decision = deployment_decision(reference, packed, protocol)
            if reference_decision != deployment_decision(shared, packed, protocol):
                raise RuntimeError("shared and independent ensemble decisions differ")
            for repetition in range(settings["measured_repetitions"]):
                torch.cuda.synchronize()
                start = time.perf_counter()
                ensemble(packed["batch"], **model_inputs(packed))
                torch.cuda.synchronize()
                neural_ms = (time.perf_counter() - start) * 1000
                torch.cuda.synchronize()
                start = time.perf_counter()
                full = r.build_batch([state_id], samples, frames, transform, device)
                decision = deployment_decision(ensemble(full["batch"], **model_inputs(full)), full, protocol)
                torch.cuda.synchronize()
                cached_ms = (time.perf_counter() - start) * 1000
                if decision != reference_decision:
                    raise RuntimeError("cached end-to-end decision differs from reference")
                rows.append({"state_id": state_id, "scale": str(frames[state_id].scale.iloc[0]),
                             "repetition": repetition, "neural_ms": neural_ms, "cached_total_ms": cached_ms})
            if (i + 1) % 24 == 0:
                update(status="PROFILING", profiled_states=i + 1)
    table = pd.DataFrame(rows)
    r.atomic_csv(table, OUT / "cached_latency_samples.csv")
    summary = {}
    for name, group in [("overall", table), *list(table.groupby("scale", sort=True))]:
        summary[name] = {column: {f"p{q}": float(np.percentile(group[column], q)) for q in (50, 90, 99)}
                         for column in ("neural_ms", "cached_total_ms")}
    caps = settings["caps"]
    report = {"status": "CACHED_PROFILE_COMPLETE_LIVE_TOTAL_PENDING", "protocol_sha256": r.digest(PROTOCOL_PATH),
              "states": len(ids), "measured_decisions": len(rows), "three_seed_output_parity": "EXACT",
              "maximum_absolute_parity_error": parity_max, "latency_ms": summary,
              "neural_cap_pass": summary["overall"]["neural_ms"]["p90"] <= caps["p90_neural_decision_ms_max"],
              "cached_total_cap_pass": summary["overall"]["cached_total_ms"]["p90"] <= caps["p90_total_decision_ms_max"],
              "live_total_cap_pass": None, "r13_eligible": False,
              "physical_shared_parameters": sum(p.numel() for p in ensemble.parameters()),
              "per_seed_parameters": [list(m.parameter_counts()) for m in models],
              "seed_checkpoint_sha256": {str(seed): r.digest(seed_paths(seed)[0]) for seed in protocol["epoch_plan"]},
              "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "samples_sha256": r.digest(OUT / "cached_latency_samples.csv"),
              "scope": settings, "r13_accessed": False, "r14_accessed": False}
    if not report["neural_cap_pass"] or not report["cached_total_cap_pass"]:
        report["status"] = "CACHED_LATENCY_GATE_FAILED_BEFORE_R13"
    r.atomic_json(report, OUT / "cached_latency_report.json")
    return report


def run_locked():
    protocol, parent = validate_protocol(), r.validate_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("deployment preparation requires verified host CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    source = pd.read_parquet(r.SOURCE_PATH)
    source["oof_fold"] = [r.grouped_oof_fold(str(s), str(c)) for s, c in zip(source.scale, source.CF_level)]
    samples, frames = r.load_samples(), r.state_frames(source)
    ids, transform = sorted(frames), r.fit_feature_transform(source)
    started, progress = time.perf_counter(), {"worker_pid": os.getpid(), "completed_seeds": 0, "expected_seeds": 3,
                                             "protocol_sha256": r.digest(PROTOCOL_PATH),
                                             "r13_accessed": False, "r14_accessed": False}

    def update(**values):
        progress.update(values)
        progress.update(elapsed_seconds=time.perf_counter() - started, updated_at_utc=datetime.now(timezone.utc).isoformat())
        r.atomic_json(progress, OUT / "progress.json")
        print(json.dumps(progress), flush=True)

    models = []
    for seed_string, plan in protocol["epoch_plan"].items():
        seed = int(seed_string)
        checkpoint_path, record_path = seed_paths(seed)
        update(status="FITTING", current_seed=seed, fixed_epochs=plan["full_fit_epochs"])
        if not record_path.exists():
            if checkpoint_path.exists():
                raise RuntimeError(f"incomplete seed checkpoint requires inspection before resuming: {checkpoint_path}")
            begin = time.perf_counter()
            model, history, epochs = r.optimize_model(
                FAMILY, seed, ids, None, samples, frames, transform, parent, torch.device("cuda"),
                maximum_epochs=plan["full_fit_epochs"], shuffle_salt=protocol["shuffle_salt"], early_stopping=False)
            r.save_checkpoint({"schema": "phase6j-caur-full-r12-seed-v1", "model_family": FAMILY,
                               "training_seed": seed, "epochs": epochs, "training_states": len(ids),
                               "feature_transform": transform.to_dict(), "protocol_sha256": r.digest(PROTOCOL_PATH),
                               "base_checkpoint_sha256": parent["base_checkpoint"]["sha256"],
                               "trainable_model_state": j3.checkpoint_state(model)}, checkpoint_path)
            r.atomic_json({"status": "COMPLETE", "seed": seed, "history": history,
                           "runtime_seconds": time.perf_counter() - begin,
                           "protocol_sha256": r.digest(PROTOCOL_PATH), "checkpoint_sha256": r.digest(checkpoint_path),
                           "r13_accessed": False, "r14_accessed": False}, record_path)
            del model
        models.append(load_seed(seed, transform, protocol, parent, torch.device("cuda")))
        update(completed_seeds=len(models))
    report_path = OUT / "cached_latency_report.json"
    if report_path.exists():
        report = r.load_json(report_path)
        if report["protocol_sha256"] != r.digest(PROTOCOL_PATH) or report["samples_sha256"] != r.digest(OUT / "cached_latency_samples.csv"):
            raise RuntimeError("completed latency evidence changed")
    else:
        report = profile(models, ids, samples, frames, transform, protocol, update)
    update(status=report["status"], profiled_states=report["states"], neural_cap_pass=report["neural_cap_pass"])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a deployment worker already owns the lock") from exc
        try:
            run_locked()
        except BaseException as exc:
            r.atomic_json({"status": "FAILED", "exit_code": 1, "pid": os.getpid(), "error": repr(exc)}, OUT / "worker_status.json")
            raise
        r.atomic_json({"status": "COMPLETE", "exit_code": 0, "pid": os.getpid()}, OUT / "worker_status.json")


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (OUT / "worker.lock").open("a+") as worker_lock:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_training_processes():
            raise RuntimeError("another Phase 6J training worker is active")
        validate_protocol()
        if (OUT / "cached_latency_report.json").exists():
            raise RuntimeError("deployment preparation is complete; audit instead of relaunching")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = OUT / f"j1_deployment_{stamp}.log"
        command = [sys.executable, "scripts/prepare_phase6j_caur_deployment.py", "--mode", "run"]
        environment = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
        with log.open("ab", buffering=0) as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"status": "STARTING", "pid": process.pid, "command": command,
                  "log_path": str(log.relative_to(ROOT)), "protocol_sha256": r.digest(PROTOCOL_PATH)}
        path = OUT / f"launch_{stamp}.json"
        r.atomic_json(record, path)
        deadline, verified = time.monotonic() + 30, False
        while time.monotonic() < deadline and process.poll() is None:
            if '"event": "phase6j_caur_epoch"' in log.read_text() or '"status": "PROFILING"' in log.read_text():
                verified = r.load_json(OUT / "progress.json")["worker_pid"] == process.pid
                if verified:
                    break
            time.sleep(1)
        if not verified:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            record.update(status="FAILED_STARTUP", exit_code=process.returncode)
        else:
            record["status"] = "RUNNING_VERIFIED"
        r.atomic_json(record, path)
        r.atomic_json(record, OUT / "launch_record.json")
        print(json.dumps(record, indent=2), flush=True)
        if not verified:
            raise RuntimeError(f"deployment startup failed: {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "run", "launch"), required=True)
    mode = parser.parse_args().mode
    if mode == "freeze":
        print(json.dumps(freeze_protocol(), indent=2))
    elif mode == "run":
        run()
    else:
        launch()
