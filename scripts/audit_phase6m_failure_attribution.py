#!/usr/bin/env python3
"""Build the read-only Phase 6M M0 failure-attribution audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from rcias_clgri.ni.calibration import (  # noqa: E402
    calibration_metrics,
    reliability_table,
)
PHASE6L_COMMIT = "8f0d37397af82e6a8bfdb437c88459712cdf5ec7"
NAMESPACE = ROOT / "outputs/phase6m_selective_confidence_v1"
AUDIT = NAMESPACE / "audit"
TRAINING = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training"
DATA = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data"
J1_TRAINING = ROOT / "outputs/phase6j_caur/training"
REPORT = ROOT / "docs/reports/phase6m_failure_attribution_audit.md"
MANUAL = Path("/home/liulei/下载/phase6m_score_free_selective_confidence_codex_instructions.md")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def verify_manifest(path: Path, *, records: bool) -> dict:
    manifest = json.loads(path.read_text())
    total_bytes = 0
    for relative, expected in manifest.items():
        target = ROOT / relative
        require(target.is_file(), f"protected file missing: {relative}")
        expected_hash = expected["sha256"] if records else expected
        require(digest(target) == expected_hash, f"protected file changed: {relative}")
        if records:
            require(target.stat().st_size == expected["bytes"],
                    f"protected size changed: {relative}")
            total_bytes += int(expected["bytes"])
    return {"files": len(manifest), "bytes": total_bytes, "status": "PASS"}


def verify_predecessor_evidence() -> dict:
    phase6l_root = ROOT / "outputs/phase6l_legacy_score_decoupling_v1"
    start = json.loads((phase6l_root / "audit/starting_audit.json").read_text())
    boundary = start["protected_evidence"]
    paths = {
        "phase6i_phase6j": ROOT / boundary["phase6i_phase6j_manifest_path"],
        "phase6k": ROOT / boundary["phase6k_manifest_path"],
        "tracked_predecessor": ROOT / boundary["tracked_predecessor_manifest_path"],
    }
    require(digest(paths["phase6i_phase6j"]) == boundary["phase6i_phase6j_manifest_sha256"],
            "Phase 6I/6J protection manifest changed")
    require(digest(paths["phase6k"]) == boundary["phase6k_manifest_sha256"],
            "Phase 6K protection manifest changed")
    require(digest(paths["tracked_predecessor"]) == boundary["tracked_predecessor_manifest_sha256"],
            "tracked predecessor manifest changed")
    return {
        "phase6i_phase6j": verify_manifest(paths["phase6i_phase6j"], records=True),
        "phase6k": verify_manifest(paths["phase6k"], records=True),
        "tracked_predecessor": verify_manifest(paths["tracked_predecessor"], records=False),
        "manifest_hashes_match_l0": True,
    }


def validate_training_protocol() -> dict:
    protocol_path = TRAINING / "training_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checks = (
        protocol.get("schema") == "phase6l-score-free-training-protocol-v1",
        protocol.get("status") == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        protocol.get("r13_accessed") is False,
        protocol.get("r14_accessed") is False,
        digest(DATA / "r12_score_free_grouped_labels.parquet")
        == protocol["input_hashes"]["score_free_grouped_labels"],
    )
    require(all(checks), "Phase 6L training protocol boundary failed")
    for relative, expected in protocol["code_hashes"].items():
        require(digest(ROOT / relative) == expected,
                f"frozen Phase 6L code changed: {relative}")
    return protocol


def grouped_bootstrap_interval(state_frame: pd.DataFrame, value_column: str,
                               *, seed: int, resamples: int) -> tuple[float, float]:
    values = state_frame.groupby("instance_id")[value_column].mean().to_numpy(dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "invalid grouped bootstrap values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fit_feature_transform(frame: pd.DataFrame) -> dict:
    return {
        "vocabularies": {
            column: set(frame[column].astype(str).unique())
            for column in CATEGORICAL_COLUMNS
        },
        "medians": {
            column: float(np.median(frame[column].to_numpy(dtype=float)))
            for column in NUMERIC_COLUMNS
        },
        "iqrs": {
            column: max(float(
                np.quantile(frame[column].to_numpy(dtype=float), 0.75)
                - np.quantile(frame[column].to_numpy(dtype=float), 0.25)
            ), 1e-6)
            for column in NUMERIC_COLUMNS
        },
    }


def starting_boundary() -> dict:
    require(MANUAL.is_file(), "Phase 6M execution manual is missing")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PHASE6L_COMMIT, "HEAD"],
            cwd=ROOT, check=False,
        ).returncode == 0,
        "the frozen Phase 6L terminal commit is not an ancestor of HEAD",
    )
    for relative in (
        "outputs/phase6j_caur/r13_selection/access_ledger.json",
        "outputs/phase6j_caur/r14_holdout/access_ledger.json",
    ):
        require(not (ROOT / relative).exists(), f"forbidden access ledger exists: {relative}")
    protected = verify_predecessor_evidence()
    phase6l_root = ROOT / "outputs/phase6l_legacy_score_decoupling_v1"
    phase6l_files = sorted(path for path in phase6l_root.rglob("*") if path.is_file())
    manifest = {
        str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in phase6l_files
    }
    protected_path = AUDIT / "protected_phase6l_evidence.json"
    if protected_path.exists():
        require(json.loads(protected_path.read_text()) == manifest,
                "protected Phase 6L evidence changed after M0 started")
    else:
        write_json(protected_path, manifest)

    required_inputs = [
        "docs/reports/phase6l_final_report.md",
        "docs/reports/phase6l_project_handoff.md",
        "docs/reports/phase6l_development_quality_report.md",
        "docs/reports/phase6l_l3_training_report.md",
        "docs/reports/phase6l_score_free_data_report.md",
        "docs/reports/phase6l_legacy_score_dependency_audit.md",
        "docs/reports/phase6l_legacy_score_decoupling_preregistered_protocol.md",
        "outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json",
        "outputs/phase6l_legacy_score_decoupling_v1/quality/development_quality.json",
        "outputs/phase6l_legacy_score_decoupling_v1/training/gate_grid.csv",
        "outputs/phase6j_caur/frozen/r12_training_protocol.json",
        "outputs/phase6j_caur/training/J1_CONT_FROZEN_oof_summary.json",
        "outputs/phase6k_runtime_v1/final/final_decision_v2.json",
        "docs/reports/phase6k_runtime_v1_final_report.md",
    ]
    hashes = {}
    for relative in required_inputs:
        path = ROOT / relative
        require(path.is_file(), f"required frozen input missing: {relative}")
        hashes[relative] = digest(path)
    return {
        "phase6l_terminal_commit": PHASE6L_COMMIT,
        "phase6l_terminal_is_ancestor": True,
        "head_at_audit": git_output("rev-parse", "HEAD"),
        "manual": {"path": str(MANUAL), "sha256": digest(MANUAL)},
        "input_sha256": hashes,
        "protected_phase6l": {
            "manifest": str(protected_path.relative_to(ROOT)),
            "files": len(manifest),
            "bytes": int(sum(item["bytes"] for item in manifest.values())),
        },
        "predecessor_manifest_verification": protected,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def gate_waterfall(selected: pd.DataFrame, gate_grid: pd.DataFrame,
                   protocol: dict) -> tuple[pd.DataFrame, dict]:
    gate = protocol["gate"]
    bootstrap = protocol["bootstrap"]
    rows: list[dict] = []
    intervention_sets: dict[str, dict[str, list[str]]] = {}
    local_flags = (
        "fallback_rejected", "probability_rejected", "lower_bound_rejected",
        "support_rejected", "immediate_harm_rejected",
    )
    for gate_index, saved in gate_grid.reset_index(drop=True).iterrows():
        current = selected.copy()
        current["lcb"] = (
            current.ensemble_advantage_mean
            - float(saved.lcb_lambda) * current.ensemble_advantage_std
        )
        current["fallback_rejected"] = current.is_fallback.astype(bool)
        current["probability_rejected"] = current.calibrated_probability.lt(float(saved.p_min))
        current["lower_bound_rejected"] = current.lcb.le(float(saved.delta_min))
        current["support_rejected"] = ~current.supported.astype(bool)
        current["immediate_harm_rejected"] = current.ensemble_immediate_utility.lt(
            float(gate["immediate_harm_floor"])
        )
        current["intervened"] = ~current[list(local_flags)].any(axis=1)
        current["gate_selected_lift"] = current.continuation_advantage_mean.where(
            current.intervened, 0.0
        )
        lower, upper = grouped_bootstrap_interval(
            current, "gate_selected_lift", seed=int(bootstrap["seed"]),
            resamples=int(bootstrap["resamples"]),
        )
        scale_lift = current.groupby("scale").gate_selected_lift.mean().to_dict()
        scale_interventions = current.groupby("scale").intervened.sum().to_dict()
        forced = json.loads(saved.forced_abstention)
        scale_coverage = {
            scale: (
                int(scale_interventions[scale])
                >= int(gate["minimum_direct_interventions_per_scale"])
                or bool(forced[scale]["exception_pass"])
            ) for scale in ("S", "M", "L")
        }
        require(int(current.intervened.sum()) == int(saved.interventions),
                f"gate {gate_index} intervention count changed")
        require(np.isclose(current.gate_selected_lift.mean(), saved.selected_lift),
                f"gate {gate_index} lift changed")
        require(np.isclose(lower, saved.selected_lift_lcb) and np.isclose(upper, saved.selected_lift_ucb),
                f"gate {gate_index} bootstrap interval changed")
        require(all(int(scale_interventions[s]) == int(saved[f"scale_{s}_interventions"])
                    for s in ("S", "M", "L")), f"gate {gate_index} scale count changed")
        require(all(np.isclose(scale_lift[s], saved[f"scale_{s}_lift"])
                    for s in ("S", "M", "L")), f"gate {gate_index} scale lift changed")
        require(bool(all(scale_coverage.values())) == bool(saved.coverage_pass),
                f"gate {gate_index} coverage changed")

        gate_id = (
            f"p{saved.p_min:.2f}_lambda{saved.lcb_lambda:.1f}_delta{saved.delta_min:.4f}"
        )
        key = f"p_min={saved.p_min:.2f}"
        intervention_sets.setdefault(key, {})[gate_id] = sorted(
            current.loc[current.intervened, "state_id"].astype(str).tolist()
        )
        for item in current.sort_values("state_id", kind="stable").to_dict("records"):
            rejected = [name for name in local_flags if bool(item[name])]
            first = next((name for name in local_flags if bool(item[name])), "accepted")
            scale = str(item["scale"])
            rows.append({
                "gate_id": gate_id,
                "gate_index": gate_index,
                "p_min": float(saved.p_min),
                "lcb_lambda": float(saved.lcb_lambda),
                "delta_min": float(saved.delta_min),
                "state_id": item["state_id"],
                "instance_id": item["instance_id"],
                "scale": scale,
                "CF_level": item["CF_level"],
                "search_stage": item["search_stage"],
                "target_set_id": item["target_set_id"],
                "calibrated_probability": item["calibrated_probability"],
                "ensemble_advantage_mean": item["ensemble_advantage_mean"],
                "ensemble_advantage_std": item["ensemble_advantage_std"],
                "lcb": item["lcb"],
                "ensemble_immediate_utility": item["ensemble_immediate_utility"],
                "realized_continuation_advantage": item["continuation_advantage_mean"],
                **{name: bool(item[name]) for name in local_flags},
                "local_rejection_count": len(rejected),
                "local_rejection_signature": "+".join(rejected) if rejected else "accepted",
                "first_local_rejection": first,
                "intervened": bool(item["intervened"]),
                "overall_mean_lift_rejected": bool(current.gate_selected_lift.mean() <= 0.0),
                "grouped_bootstrap_lift_rejected": bool(lower <= 0.0),
                "scale_lift_rejected": bool(scale_lift[scale] < 0.0),
                "scale_coverage_rejected": not bool(scale_coverage[scale]),
                "coverage_aggregation_rejected": not bool(all(scale_coverage.values())),
                "gate_retained": bool(saved.retained),
            })
    frame = pd.DataFrame(rows)
    require(len(frame) == 288 * 18, "gate waterfall is not 288 x 18")

    signatures = frame.local_rejection_signature.value_counts().sort_index().to_dict()
    exclusive = {
        flag: int(((frame.local_rejection_count == 1) & frame[flag]).sum())
        for flag in local_flags
    }
    fixed_p = {}
    for key, sets in intervention_sets.items():
        hashes = {
            gate_id: hashlib.sha256("\n".join(ids).encode()).hexdigest()
            for gate_id, ids in sets.items()
        }
        fixed_p[key] = {
            "configuration_count": len(sets),
            "unique_intervention_set_count": len(set(hashes.values())),
            "identical_across_lambda_and_delta": len(set(hashes.values())) == 1,
            "intervention_count": len(next(iter(sets.values()))),
            "set_sha256": next(iter(hashes.values())),
        }
    least = frame[(frame.p_min == 0.55) & (frame.lcb_lambda == 0.5) & (frame.delta_min == 0.0)]
    nonfallback_rejected = least[~least.fallback_rejected & ~least.intervened]
    prob_support = nonfallback_rejected.probability_rejected | nonfallback_rejected.support_rejected
    summary = {
        "rows": len(frame),
        "states": int(frame.state_id.nunique()),
        "gates": int(frame.gate_id.nunique()),
        "local_flag_counts_all_state_gate_rows": {
            flag: int(frame[flag].sum()) for flag in local_flags
        },
        "exclusive_local_rejection_counts": exclusive,
        "overlap_signatures": {str(key): int(value) for key, value in signatures.items()},
        "fixed_p_intervention_identity": fixed_p,
        "least_restrictive_gate": {
            "gate_id": str(least.gate_id.iloc[0]),
            "interventions": int(least.intervened.sum()),
            "nonfallback_rejected_states": len(nonfallback_rejected),
            "probability_or_support_rejected_states": int(prob_support.sum()),
            "probability_or_support_share": float(prob_support.mean()),
            "dominated_by_probability_or_support": bool(prob_support.mean() >= 0.9),
            "immediate_harm_rejected_states": int(nonfallback_rejected.immediate_harm_rejected.sum()),
            "immediate_harm_share": float(nonfallback_rejected.immediate_harm_rejected.mean()),
            "single_condition_bypass_additional_interventions": {
                flag: int((~least[[other for other in local_flags if other != flag]].any(axis=1)).sum()
                          - least.intervened.sum())
                for flag in local_flags
            },
        },
    }
    return frame, summary


def gate_component_snapshot(selected: pd.DataFrame) -> dict:
    flags = {
        "nonfallback": ~selected.is_fallback.astype(bool),
        "probability_at_least_0_55": selected.calibrated_probability.ge(0.55),
        "lcb_lambda_1_above_0": (
            selected.ensemble_advantage_mean - selected.ensemble_advantage_std
        ).gt(0.0),
        "supported": selected.supported.astype(bool),
        "immediate_harm_floor_pass": selected.ensemble_immediate_utility.ge(-0.005),
    }
    all_pass = np.logical_and.reduce([value.to_numpy(dtype=bool) for value in flags.values()])
    predicted = selected.ensemble_immediate_utility.to_numpy(dtype=float)
    actual = selected.immediate_utility.to_numpy(dtype=float)
    return {
        "component_pass_counts": {key: int(value.sum()) for key, value in flags.items()},
        "all_components_pass": int(all_pass.sum()),
        "all_components_pass_by_scale": {
            scale: int(all_pass[selected.scale.eq(scale).to_numpy()].sum())
            for scale in ("S", "M", "L")
        },
        "immediate_utility_head": {
            "predicted_mean": float(predicted.mean()),
            "actual_mean": float(actual.mean()),
            "predicted_actual_pearson": float(pearsonr(predicted, actual).statistic),
            "predicted_actual_spearman": float(spearmanr(predicted, actual).statistic),
            "mean_absolute_error": float(np.abs(predicted - actual).mean()),
            "predicted_floor_pass": int((predicted >= -0.005).sum()),
            "actual_floor_pass": int((actual >= -0.005).sum()),
            "floor_classification_agreement": float(np.mean(
                (predicted >= -0.005) == (actual >= -0.005)
            )),
        },
    }


def brier_decomposition(probability: np.ndarray, labels: np.ndarray,
                        bins: int = 10) -> dict[str, float]:
    table = reliability_table(probability, labels, bins=bins)
    nonempty = table[table["count"] > 0]
    weights = nonempty["count"].to_numpy(dtype=float) / len(labels)
    confidence = nonempty.mean_confidence.to_numpy(dtype=float)
    observed = nonempty.positive_fraction.to_numpy(dtype=float)
    base = float(np.mean(labels))
    reliability = float(np.sum(weights * (confidence - observed) ** 2))
    resolution = float(np.sum(weights * (observed - base) ** 2))
    uncertainty = base * (1.0 - base)
    brier = float(np.mean((probability - labels) ** 2))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_identity_residual": brier - (reliability - resolution + uncertainty),
        "method": "10_equal_width_bin_approximation",
    }


def confidence_audit(name: str, selected: pd.DataFrame) -> tuple[dict, list[dict]]:
    ordered = selected.sort_values(["calibrated_probability", "state_id"],
                                   ascending=[False, True], kind="stable").reset_index(drop=True)
    probability = ordered.calibrated_probability.to_numpy(dtype=float)
    labels = ordered.continuation_advantage_mean.gt(0).to_numpy(dtype=int)
    advantage = ordered.continuation_advantage_mean.to_numpy(dtype=float)
    basic = calibration_metrics(probability, labels)
    decomposition = brier_decomposition(probability, labels)
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped))
    calibration = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    calibration.fit(logits.reshape(-1, 1), labels)
    pearson = pearsonr(probability, advantage)
    spearman = spearmanr(probability, advantage)
    quantile_levels = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    summary = {
        **basic,
        **decomposition,
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "calibration_intercept": float(calibration.intercept_[0]),
        "confidence_advantage_pearson": float(pearson.statistic),
        "confidence_advantage_spearman": float(spearman.statistic),
        "probability_std": float(np.std(probability, ddof=0)),
        "probability_iqr": float(np.quantile(probability, 0.75) - np.quantile(probability, 0.25)),
        "probability_quantiles": {
            str(level): float(np.quantile(probability, level)) for level in quantile_levels
        },
        "fraction_above": {
            str(threshold): float(np.mean(probability >= threshold))
            for threshold in (0.55, 0.65, 0.75)
        },
    }
    rows: list[dict] = []
    for key, value in summary.items():
        if isinstance(value, (int, float, np.floating)):
            rows.append({"model": name, "record_type": "summary", "metric": key, "value": value})
    for level, value in summary["probability_quantiles"].items():
        rows.append({"model": name, "record_type": "probability_quantile",
                     "metric": "probability", "x": float(level), "value": value})
    for threshold, value in summary["fraction_above"].items():
        rows.append({"model": name, "record_type": "threshold_fraction",
                     "metric": "fraction_at_or_above", "x": float(threshold), "value": value})
    reliable = reliability_table(probability, labels)
    for item in reliable.to_dict("records"):
        rows.append({"model": name, "record_type": "reliability_bin", "metric": "calibration",
                     "x": item["bin"], "lower": item["lower"], "upper": item["upper"],
                     "count": item["count"], "mean_probability": item["mean_confidence"],
                     "positive_rate": item["positive_fraction"]})
    ordered["confidence_decile"] = np.minimum(
        np.floor(np.arange(len(ordered)) * 10 / len(ordered)).astype(int) + 1, 10
    )
    for decile, group in ordered.groupby("confidence_decile", sort=True):
        rows.append({"model": name, "record_type": "ranked_confidence_decile",
                     "metric": "outcome_by_decile", "x": int(decile), "count": len(group),
                     "mean_probability": float(group.calibrated_probability.mean()),
                     "positive_rate": float(group.continuation_advantage_mean.gt(0).mean()),
                     "mean_advantage": float(group.continuation_advantage_mean.mean())})
    for coverage in np.arange(0.05, 1.001, 0.05):
        count = max(1, int(math.ceil(coverage * len(ordered))))
        group = ordered.iloc[:count]
        rows.append({"model": name, "record_type": "selective_coverage_curve",
                     "metric": "top_confidence_subset", "x": float(coverage), "count": count,
                     "mean_probability": float(group.calibrated_probability.mean()),
                     "positive_rate": float(group.continuation_advantage_mean.gt(0).mean()),
                     "negative_outcome_rate": float(group.continuation_advantage_mean.le(0).mean()),
                     "mean_advantage": float(group.continuation_advantage_mean.mean())})
    return summary, rows


def support_audit(ensemble: pd.DataFrame, selected: pd.DataFrame,
                  source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    transforms = {
        fold: fit_feature_transform(source[source.oof_fold.ne(fold)])
        for fold in sorted(source.oof_fold.unique())
    }
    selected_keys = set(zip(selected.state_id.astype(str), selected.target_set_id.astype(str)))
    detail_rows = []
    all_supported = []
    for item in ensemble.sort_values(["state_id", "target_set_id"], kind="stable").to_dict("records"):
        transform = transforms[int(item["oof_fold"])]
        reasons = []
        categorical_reasons = []
        numeric_reasons = []
        max_abs_z = 0.0
        for column in CATEGORICAL_COLUMNS:
            if str(item[column]) not in transform["vocabularies"][column]:
                reason = f"unseen_category:{column}"
                reasons.append(reason)
                categorical_reasons.append(reason)
        for column in NUMERIC_COLUMNS:
            robust_z = (
                float(item[column]) - transform["medians"][column]
            ) / transform["iqrs"][column]
            max_abs_z = max(max_abs_z, abs(robust_z))
            if not np.isfinite(robust_z) or robust_z < -8.0 or robust_z > 8.0:
                reason = f"numeric_robust_z:{column}"
                reasons.append(reason)
                numeric_reasons.append(reason)
        computed_supported = len(reasons) == 0
        all_supported.append(computed_supported)
        if not computed_supported:
            detail_rows.append({
                "state_id": item["state_id"], "instance_id": item["instance_id"],
                "target_set_id": item["target_set_id"], "oof_fold": int(item["oof_fold"]),
                "scale": item["scale"], "CF_level": item["CF_level"],
                "search_stage": item["search_stage"], "origin_family": item["origin_family"],
                "primary_origin_rule": item["primary_origin_rule"],
                "origin_destroy_operator": item["origin_destroy_operator"],
                "selected_winner": (str(item["state_id"]), str(item["target_set_id"])) in selected_keys,
                "categorical_reason_count": len(categorical_reasons),
                "numeric_reason_count": len(numeric_reasons),
                "reason_count": len(reasons), "multiple_causes": len(reasons) > 1,
                "maximum_absolute_robust_z": max_abs_z,
                "categorical_reasons": "+".join(categorical_reasons),
                "numeric_reasons": "+".join(numeric_reasons),
                "reason_signature": "+".join(reasons),
            })
    computed = np.asarray(all_supported, dtype=bool)
    require(np.array_equal(computed, ensemble.supported.to_numpy(dtype=bool)),
            "support reason reconstruction does not match frozen supported bits")
    detail = pd.DataFrame(detail_rows)
    require(len(detail) == int((~ensemble.supported.astype(bool)).sum()),
            "unsupported detail row count mismatch")

    expanded = ensemble.copy()
    expanded["selected_winner"] = [
        (str(s), str(t)) in selected_keys
        for s, t in zip(expanded.state_id, expanded.target_set_id)
    ]
    summary_rows = []
    for dimension in ("overall", "scale", "CF_level", "search_stage", "origin_family", "selected_winner"):
        groups = [("all", expanded)] if dimension == "overall" else expanded.groupby(dimension, sort=True)
        for value, group in groups:
            unsupported = group[~group.supported.astype(bool)]
            group_keys = set(zip(
                unsupported.state_id.astype(str), unsupported.target_set_id.astype(str)
            ))
            counter = Counter(
                reason
                for item in detail.itertuples(index=False)
                if (str(item.state_id), str(item.target_set_id)) in group_keys
                for reason in str(item.reason_signature).split("+") if reason
            )
            summary_rows.append({
                "dimension": dimension, "group": str(value), "candidates": len(group),
                "unsupported_candidates": len(unsupported),
                "support_rate": float(group.supported.mean()),
                "selected_winners": int(group.selected_winner.sum()),
                "unsupported_selected_winners": int((group.selected_winner & ~group.supported.astype(bool)).sum()),
                "reason_counts_json": json.dumps(counter, sort_keys=True),
            })
    support_summary = pd.DataFrame(summary_rows)
    reason_counts = Counter(
        reason for signature in detail.reason_signature
        for reason in str(signature).split("+") if reason
    )
    result = {
        "candidate_support_rate": float(ensemble.supported.mean()),
        "selected_winner_support_rate": float(selected.supported.mean()),
        "unsupported_candidates": len(detail),
        "unsupported_selected_winners": int((~selected.supported.astype(bool)).sum()),
        "unseen_category_rejections": int(sum(
            count for reason, count in reason_counts.items() if reason.startswith("unseen_category:")
        )),
        "numeric_excursion_rejections": int(sum(
            count for reason, count in reason_counts.items() if reason.startswith("numeric_robust_z:")
        )),
        "multiple_cause_candidates": int(detail.multiple_causes.sum()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "support_by_scale": {
            str(row.group): float(row.support_rate)
            for row in support_summary[support_summary.dimension.eq("scale")].itertuples()
        },
        "winner_support_by_scale": {
            scale: float(group.supported.mean()) for scale, group in selected.groupby("scale", sort=True)
        },
        "support_by_CF_level": {
            str(row.group): float(row.support_rate)
            for row in support_summary[support_summary.dimension.eq("CF_level")].itertuples()
        },
        "support_by_search_stage": {
            str(row.group): float(row.support_rate)
            for row in support_summary[support_summary.dimension.eq("search_stage")].itertuples()
        },
    }
    return detail, support_summary, result


def normalized_entropy(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    scale = float(np.std(values, ddof=0))
    standardized = np.zeros_like(values) if scale == 0.0 else (values - values.mean()) / scale
    shifted = standardized - standardized.max()
    probability = np.exp(shifted) / np.exp(shifted).sum()
    return float(-(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum() / np.log(len(values)))


def uncertainty_audit(predictions: pd.DataFrame, ensemble: pd.DataFrame,
                      selected: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected_by_state = selected.set_index("state_id")
    rows = []
    for state_id, mean_group in ensemble.groupby("state_id", sort=True):
        group = mean_group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
        seed_group = predictions[predictions.state_id.eq(state_id)]
        seeds = sorted(seed_group.training_seed.unique())
        matrices = []
        seed_top = []
        seed_probabilities = []
        seed_logits = []
        for seed in seeds:
            current = seed_group[seed_group.training_seed.eq(seed)].sort_values(
                "target_set_id", kind="stable"
            ).reset_index(drop=True)
            require(current.target_set_id.tolist() == group.target_set_id.tolist(),
                    f"candidate order drift in {state_id}, seed {seed}")
            scores = current.predicted_continuation_advantage.to_numpy(dtype=float)
            matrices.append(scores)
            best = np.lexsort((current.target_set_id.astype(str).to_numpy(), -scores))[0]
            seed_top.append(str(current.iloc[best].target_set_id))
            seed_probabilities.append(current.predicted_beats_fallback_probability_raw.to_numpy(dtype=float))
            seed_logits.append(current.predicted_beats_fallback_logit.to_numpy(dtype=float))
        score_matrix = np.vstack(matrices)
        probability_matrix = np.vstack(seed_probabilities)
        logit_matrix = np.vstack(seed_logits)
        mean_scores = group.ensemble_advantage_mean.to_numpy(dtype=float)
        winner = selected_by_state.loc[state_id]
        winner_index = int(np.flatnonzero(group.target_set_id.eq(winner.target_set_id))[0])
        fallback_index = int(np.flatnonzero(group.is_fallback.astype(bool))[0])
        ordered_indices = np.lexsort((group.target_set_id.astype(str).to_numpy(), -mean_scores))
        top2_margin = float(mean_scores[ordered_indices[0]] - mean_scores[ordered_indices[1]])
        pair_rank = []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                pair_rank.append(float(spearmanr(score_matrix[left], score_matrix[right]).statistic))
        votes = Counter(seed_top)
        row = {
            "state_id": state_id, "instance_id": winner.instance_id,
            "scale": winner.scale, "CF_level": winner.CF_level,
            "search_stage": winner.search_stage, "winner_target_set_id": winner.target_set_id,
            "winner_origin_family": winner.origin_family,
            "candidate_bank_size": len(group),
            "ensemble_winner_advantage_mean": float(mean_scores[winner_index]),
            "ensemble_winner_advantage_std": float(np.std(score_matrix[:, winner_index], ddof=0)),
            "winner_beats_fallback_probability_raw_mean": float(probability_matrix[:, winner_index].mean()),
            "winner_beats_fallback_probability_raw_std": float(probability_matrix[:, winner_index].std(ddof=0)),
            "winner_beats_fallback_logit_mean": float(logit_matrix[:, winner_index].mean()),
            "winner_beats_fallback_logit_std": float(logit_matrix[:, winner_index].std(ddof=0)),
            "calibrated_probability": float(winner.calibrated_probability),
            "top1_top2_advantage_margin": top2_margin,
            "winner_fallback_advantage_margin": float(mean_scores[winner_index] - mean_scores[fallback_index]),
            "seed_top1_vote_for_ensemble_winner_fraction": votes[str(winner.target_set_id)] / len(seeds),
            "seed_top1_max_vote_fraction": max(votes.values()) / len(seeds),
            "seed_top1_unique_candidates": len(votes),
            "mean_pairwise_seed_rank_correlation": float(np.mean(pair_rank)),
            "mean_pairwise_seed_rank_disagreement": float(np.mean([1.0 - value for value in pair_rank])),
            "candidate_score_normalized_entropy": normalized_entropy(mean_scores),
            "candidate_score_std": float(np.std(mean_scores, ddof=0)),
            "candidate_score_spread": float(mean_scores.max() - mean_scores.min()),
            "fraction_candidates_predicted_above_fallback": float(np.mean(mean_scores > mean_scores[fallback_index])),
            "supported_candidate_count": int(group.supported.astype(bool).sum()),
            "supported_candidate_fraction": float(group.supported.mean()),
            "selected_winner_supported": bool(winner.supported),
            "selected_winner_is_fallback": bool(winner.is_fallback),
            "realized_winner_lift": float(winner.continuation_advantage_mean),
            "reference_p055_intervened": bool(
                not bool(winner.is_fallback)
                and float(winner.calibrated_probability) >= 0.55
                and float(winner.ensemble_advantage_mean - 0.5 * winner.ensemble_advantage_std) > 0.0
                and bool(winner.supported)
                and float(winner.ensemble_immediate_utility) >= -0.005
            ),
        }
        row["false_positive_intervention"] = bool(
            row["reference_p055_intervened"] and row["realized_winner_lift"] <= 0.0
        )
        row["false_negative_abstention"] = bool(
            not row["reference_p055_intervened"] and row["realized_winner_lift"] > 0.0
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    require(len(frame) == 288, "uncertainty audit must contain 288 states")
    feature_columns = [
        "ensemble_winner_advantage_mean", "ensemble_winner_advantage_std",
        "top1_top2_advantage_margin", "winner_fallback_advantage_margin",
        "seed_top1_vote_for_ensemble_winner_fraction", "mean_pairwise_seed_rank_disagreement",
        "candidate_score_normalized_entropy", "candidate_score_std", "candidate_score_spread",
        "supported_candidate_fraction", "winner_beats_fallback_probability_raw_std",
    ]
    correlations = {}
    for column in feature_columns:
        value = spearmanr(frame[column], frame.realized_winner_lift).statistic
        correlations[column] = None if not np.isfinite(value) else float(value)
    by_scale = {}
    for scale, group in frame.groupby("scale", sort=True):
        by_scale[str(scale)] = {
            "states": len(group), "mean_realized_lift": float(group.realized_winner_lift.mean()),
            "false_positive_interventions": int(group.false_positive_intervention.sum()),
            "false_negative_abstentions": int(group.false_negative_abstention.sum()),
            "mean_supported_candidate_fraction": float(group.supported_candidate_fraction.mean()),
            "winner_support_rate": float(group.selected_winner_supported.mean()),
        }
    result = {
        "states": len(frame),
        "reference_gate": {"p_min": 0.55, "lambda": 0.5, "delta_min": 0.0},
        "reference_interventions": int(frame.reference_p055_intervened.sum()),
        "false_positive_interventions": int(frame.false_positive_intervention.sum()),
        "false_negative_abstentions": int(frame.false_negative_abstention.sum()),
        "spearman_with_realized_winner_lift": correlations,
        "by_scale": by_scale,
    }
    return frame, result


def render_report(result: dict) -> str:
    gate = result["gate_rejection"]
    support = result["support"]
    l = result["confidence"]["phase6l_score_free"]
    j = result["confidence"]["phase6j_j1"]
    uncertainty = result["candidate_selection_uncertainty"]
    fixed = gate["fixed_p_intervention_identity"]
    reasons = support["reason_counts"]
    main_reasons = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:5]
    reason_text = "、".join(f"`{key}`={value}" for key, value in main_reasons)
    dominance = (
        "因此 probability/support 足以解释绝大多数实际拒绝。"
        if gate["least_restrictive_gate"]["dominated_by_probability_or_support"]
        else (
            "这不支持 probability/support 单独主导的预设；immediate-harm 条件覆盖了 "
            f"{gate['least_restrictive_gate']['immediate_harm_share']:.3%} 的这些拒绝状态，"
            "是当前最强的单项约束。"
        )
    )
    components = result["gate_component_comparison"]
    l_harm = components["phase6l_score_free"]["immediate_utility_head"]
    j_harm = components["phase6j_j1"]["immediate_utility_head"]
    return f"""# Phase 6M M0：冻结失败归因审计

状态：**M0_COMPLETE — READY_TO_PREREGISTER_M1**。本阶段只读取已冻结的 Phase 6L/6J 证据，没有训练新模型、没有运行 Gurobi，也没有访问 R13/R14。

## 结论

Phase 6L 的失败位于部署选择层，而不是已验证的原始排序路径。5,184 条状态×门控复算与冻结 `gate_grid.csv` 完全一致。最宽松门控 `p_min=0.55, lambda=0.5, delta=0` 的 {gate['least_restrictive_gate']['nonfallback_rejected_states']} 个非 fallback 拒绝状态中，{gate['least_restrictive_gate']['probability_or_support_rejected_states']} 个至少触发 probability 或 support，比例为 {gate['least_restrictive_gate']['probability_or_support_share']:.3%}。{dominance}

逐项绕过实验进一步定位了原因：只绕过 probability 可新增 {gate['least_restrictive_gate']['single_condition_bypass_additional_interventions']['probability_rejected']} 次干预，只绕过 support 新增 {gate['least_restrictive_gate']['single_condition_bypass_additional_interventions']['support_rejected']} 次，而只绕过 immediate-harm 可新增 {gate['least_restrictive_gate']['single_condition_bypass_additional_interventions']['immediate_harm_rejected']} 次。Phase 6L immediate-utility head 只有 {l_harm['predicted_floor_pass']}/288 通过 `-0.005`，冻结 J1 为 {j_harm['predicted_floor_pass']}/288；Phase 6L 该预测与真实 immediate utility 的 Spearman 仅 {l_harm['predicted_actual_spearman']:.6f}。M1 的 selective-risk 表征必须纳入并审计这一路信号，同时保持最终 immediate-harm floor 不变。

固定 `p_min` 后，`lambda` 和 `delta_min` 的 6 个组合均只有一个唯一干预集合：`p_min=0.55/0.65/0.75` 分别为 {fixed['p_min=0.55']['intervention_count']}、{fixed['p_min=0.65']['intervention_count']}、{fixed['p_min=0.75']['intervention_count']} 次干预。该结论是逐状态集合哈希的精确相等，不是近似判断。`p_min=0.55` 的 Medium gated lift 为负，而更高阈值又无法达到每个规模至少 20 次直接干预；原 18 个 gate 因而全部拒绝。

Phase 6L winner probability 的 ECE 为 {l['expected_calibration_error']:.6f}，但 AUROC 仅 {l['auroc']:.6f}、AUPRC 为 {l['auprc']:.6f}、Brier resolution 为 {l['resolution']:.6f}、概率标准差为 {l['probability_std']:.6f}。冻结 J1 对应值为 ECE {j['expected_calibration_error']:.6f}、AUROC {j['auroc']:.6f}、AUPRC {j['auprc']:.6f}、resolution {j['resolution']:.6f}、概率标准差 {j['probability_std']:.6f}。这些量把全局校准与选择分辨率分开；M1 不应继续仅用 288 个 winner 的低 ECE 作为置信度充分证据。

Phase 6L 候选 support 为 {support['candidate_support_rate']:.3%}，winner support 为 {support['selected_winner_support_rate']:.3%}。共 {support['unsupported_candidates']} 个候选、{support['unsupported_selected_winners']} 个 winner 不支持；未见类别原因计数为 {support['unseen_category_rejections']}，数值 robust-z 原因计数为 {support['numeric_excursion_rejections']}，多原因候选为 {support['multiple_cause_candidates']}。最高频原因是 {reason_text}。其中二值 `is_fallback` 在训练折 IQR 为零时被当作连续异常值，导致全部 288 个 fallback 候选 support 失败；`critical_overlap_fraction` 同样受零/极小 IQR 影响。这是硬 support 表示的具体缺陷。M1 应保留高层语义类别 fail-closed，将二值特征从连续距离中分离，并把其余数值 support 改为训练折内连续距离及明确硬边界。

三 seed 不确定性表覆盖全部 {uncertainty['states']} 个状态。以原 `p_min=0.55, lambda=0.5, delta=0` 为参照，复现 {uncertainty['reference_interventions']} 次干预、{uncertainty['false_positive_interventions']} 个非正收益干预和 {uncertainty['false_negative_abstentions']} 个正收益弃权。winner 均值/标准差、top1-top2 margin、winner-fallback margin、seed vote/rank disagreement、候选分数熵/离散度和 support 比例现已形成 M1 特征选择的冻结依据。

## 方法边界

- Gate 审计按原 Phase 6L 顺序复算 fallback、probability、LCB、support 和 immediate-harm 五个局部条件；overall lift、grouped-bootstrap LCB、对应 scale lift 与 coverage 作为聚合条件单列。
- Brier 分解采用 10 个等宽概率箱，是离散近似；JSON 同时保存 identity residual，避免把分箱近似写成精确恒等式。
- Support 原因由每个 held fold 的两个训练折重新拟合原 median/IQR 与类别词表后反演，并与冻结 OOF 的每个 `supported` 位逐项一致。
- 不确定性只使用现有三 seed Phase 6L OOF 预测与已存 R12 continuation outcome；没有生成新的 qualification output。

## 证据

- `outputs/phase6m_selective_confidence_v1/audit/failure_attribution.json`
- `outputs/phase6m_selective_confidence_v1/audit/gate_rejection_waterfall.csv`
- `outputs/phase6m_selective_confidence_v1/audit/confidence_diagnostics.csv`
- `outputs/phase6m_selective_confidence_v1/audit/support_rejection_breakdown.csv`
- `outputs/phase6m_selective_confidence_v1/audit/support_rejection_summary.csv`
- `outputs/phase6m_selective_confidence_v1/audit/candidate_selection_uncertainty.csv`
- `outputs/phase6m_selective_confidence_v1/audit/protected_phase6l_evidence.json`

M1 只能在上述归因基础上冻结一个 promotable family：`M1_SCORE_FREE_SELECTIVE_RISK`。R13/R14 继续锁定；最终规模覆盖、正 grouped-bootstrap LCB、各规模非负 lift 及后续延迟门槛保持不变。
"""


def main() -> None:
    boundary = starting_boundary()
    protocol = validate_training_protocol()
    source = pd.read_parquet(DATA / "r12_score_free_grouped_labels.parquet")
    predictions = pd.read_parquet(TRAINING / "oof_predictions.parquet")
    ensemble = pd.read_parquet(TRAINING / "ensemble_oof.parquet")
    selected = pd.read_parquet(TRAINING / "selected_winners.parquet")
    gate_grid = pd.read_csv(TRAINING / "gate_grid.csv")
    require(len(source) == len(ensemble) == 6809 and selected.state_id.nunique() == 288,
            "frozen Phase 6L row counts changed")
    waterfall, gate_result = gate_waterfall(selected, gate_grid, protocol)

    diagnostics_rows = []
    confidence = {}
    for name, path in (
        ("phase6l_score_free", TRAINING / "selected_winners.parquet"),
        ("phase6j_j1", J1_TRAINING / "J1_CONT_FROZEN_selected_winners.parquet"),
    ):
        model_result, rows = confidence_audit(name, pd.read_parquet(path))
        confidence[name] = model_result
        diagnostics_rows.extend(rows)
    diagnostics = pd.DataFrame(diagnostics_rows)
    gate_components = {
        "phase6l_score_free": gate_component_snapshot(selected),
        "phase6j_j1": gate_component_snapshot(pd.read_parquet(
            J1_TRAINING / "J1_CONT_FROZEN_selected_winners.parquet"
        )),
    }
    support_detail, support_summary, support_result = support_audit(ensemble, selected, source)
    uncertainty, uncertainty_result = uncertainty_audit(predictions, ensemble, selected)

    result = {
        "schema": "phase6m-frozen-failure-attribution-v1",
        "status": "M0_COMPLETE",
        "decision": "READY_TO_PREREGISTER_M1",
        "scope": "read-only frozen Phase 6L and predecessor OOF evidence",
        "boundary": boundary,
        "gate_rejection": gate_result,
        "confidence": confidence,
        "gate_component_comparison": gate_components,
        "support": support_result,
        "candidate_selection_uncertainty": uncertainty_result,
        "scientific_attribution": {
            "raw_ranker_invalidated": False,
            "failure_layer": "SELECTIVE_CONFIDENCE_SUPPORT_AND_IMMEDIATE_HARM_AUXILIARY",
            "probability_support_dominated": gate_result["least_restrictive_gate"]["dominated_by_probability_or_support"],
            "m1_ranker_architecture_change_authorized": False,
            "m1_selector_support_revision_justified": True,
        },
        "new_training_performed": False,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    write_csv(AUDIT / "gate_rejection_waterfall.csv", waterfall)
    write_csv(AUDIT / "confidence_diagnostics.csv", diagnostics)
    write_csv(AUDIT / "support_rejection_breakdown.csv", support_detail)
    write_csv(AUDIT / "support_rejection_summary.csv", support_summary)
    write_csv(AUDIT / "candidate_selection_uncertainty.csv", uncertainty)
    write_json(AUDIT / "failure_attribution.json", result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(result))
    print(json.dumps({
        "status": result["status"], "decision": result["decision"],
        "waterfall_rows": len(waterfall), "unsupported_candidates": len(support_detail),
        "uncertainty_states": len(uncertainty),
        "probability_support_share": gate_result["least_restrictive_gate"]["probability_or_support_share"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
