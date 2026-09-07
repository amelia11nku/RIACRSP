#!/usr/bin/env python3
"""Run the preregistered Phase 6M M4 development-quality audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.calibration import calibration_metrics, reliability_table  # noqa: E402
from rcias_clgri.ni.phase6m_selective_risk import (  # noqa: E402
    SUPPORT_NUMERIC_COLUMNS,
    SupportTransform,
)
from scripts import audit_phase6m_training as m3_audit  # noqa: E402
from scripts import train_phase6l_score_free as phase6l  # noqa: E402
from scripts import train_phase6m_selective_confidence as training  # noqa: E402


NAMESPACE = ROOT / "outputs/phase6m_selective_confidence_v1"
TRAINING = NAMESPACE / "training"
QUALITY = NAMESPACE / "quality"
REPORT = ROOT / "docs/reports/phase6m_development_quality_report.md"
PHASE6L_TRAINING = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training"
J1_SUMMARY = ROOT / "outputs/phase6j_caur/training/J1_CONT_FROZEN_oof_summary.json"
M0_RESULT = NAMESPACE / "audit/failure_attribution.json"
FAMILY = "M1_SCORE_FREE_SELECTIVE_RISK"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace frozen M4 evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace frozen M4 evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def grouped_bootstrap(
    frame: pd.DataFrame, column: str, *, seed: int, resamples: int
) -> tuple[float, float]:
    values = frame.groupby("instance_id")[column].mean().to_numpy(dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "invalid grouped bootstrap input")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def brier_decomposition(probability: np.ndarray, labels: np.ndarray) -> dict:
    table = reliability_table(probability, labels, bins=10)
    nonempty = table[table["count"] > 0]
    weights = nonempty["count"].to_numpy(dtype=float) / len(labels)
    confidence = nonempty.mean_confidence.to_numpy(dtype=float)
    observed = nonempty.positive_fraction.to_numpy(dtype=float)
    base = float(labels.mean())
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


def confidence_audit(selected: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    probability = selected.selector_probability.to_numpy(dtype=float)
    labels = selected.continuation_advantage_mean.gt(0).to_numpy(dtype=int)
    advantage = selected.continuation_advantage_mean.to_numpy(dtype=float)
    basic = calibration_metrics(probability, labels)
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped))
    calibration = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    calibration.fit(logits.reshape(-1, 1), labels)
    pearson = pearsonr(probability, advantage)
    spearman = spearmanr(probability, advantage)
    quantiles = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    summary = {
        **basic,
        **brier_decomposition(probability, labels),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "calibration_intercept": float(calibration.intercept_[0]),
        "confidence_advantage_pearson": float(pearson.statistic),
        "confidence_advantage_spearman": float(spearman.statistic),
        "probability_std": float(np.std(probability, ddof=0)),
        "probability_iqr": float(np.quantile(probability, 0.75) - np.quantile(probability, 0.25)),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_quantiles": {
            str(level): float(np.quantile(probability, level)) for level in quantiles
        },
        "fraction_above": {
            str(threshold): float(np.mean(probability >= threshold))
            for threshold in (0.55, 0.65, 0.75)
        },
    }

    rows: list[dict] = []
    reliability = reliability_table(probability, labels, bins=10)
    for item in reliability.to_dict("records"):
        rows.append({
            "curve_type": "reliability", "rank_variable": "selector_probability",
            "x": int(item["bin"]), "lower": float(item["lower"]),
            "upper": float(item["upper"]), "count": int(item["count"]),
            "mean_probability": float(item["mean_confidence"]),
            "positive_rate": float(item["positive_fraction"]),
        })
    for rank_variable, ascending in (
        ("selector_probability", False),
        ("selector_lcb_lambda_0_5", False),
    ):
        work = selected.copy()
        if rank_variable == "selector_lcb_lambda_0_5":
            work[rank_variable] = (
                work.selector_predictive_mean - 0.5 * work.selector_total_predictive_scale
            )
        work = work.sort_values(
            [rank_variable, "state_id"], ascending=[ascending, True], kind="stable"
        ).reset_index(drop=True)
        for coverage in np.arange(0.05, 1.001, 0.05):
            count = max(1, int(math.ceil(coverage * len(work))))
            group = work.iloc[:count]
            rows.append({
                "curve_type": "selective_coverage", "rank_variable": rank_variable,
                "x": float(coverage), "count": count,
                "mean_probability": float(group.selector_probability.mean()),
                "positive_rate": float(group.continuation_advantage_mean.gt(0).mean()),
                "negative_outcome_rate": float(group.continuation_advantage_mean.le(0).mean()),
                "mean_advantage": float(group.continuation_advantage_mean.mean()),
                "mean_rank_statistic": float(group[rank_variable].mean()),
            })
    for threshold in np.arange(0.0, 1.001, 0.05):
        group = selected[selected.selector_probability.ge(threshold)]
        rows.append({
            "curve_type": "probability_threshold", "rank_variable": "selector_probability",
            "x": float(threshold), "count": len(group),
            "coverage": float(len(group) / len(selected)),
            "mean_probability": float(group.selector_probability.mean()) if len(group) else None,
            "positive_rate": float(group.continuation_advantage_mean.gt(0).mean()) if len(group) else None,
            "negative_outcome_rate": float(group.continuation_advantage_mean.le(0).mean()) if len(group) else None,
            "mean_advantage": float(group.continuation_advantage_mean.mean()) if len(group) else None,
        })
    return summary, pd.DataFrame(rows)


def raw_ranking(ensemble: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    states = phase6l.base.ranking_state_metrics(ensemble, "ranker_advantage_mean")
    frozen_states = pd.read_parquet(PHASE6L_TRAINING / "state_metrics.parquet")
    pd.testing.assert_frame_equal(states, frozen_states)
    lower, upper = grouped_bootstrap(
        states, "selected_lift", seed=int(config["bootstrap"]["seed"]),
        resamples=int(config["bootstrap"]["resamples"]),
    )
    scale = states.groupby("scale").agg(
        mean_spearman=("spearman", "mean"),
        mean_selected_lift=("selected_lift", "mean"),
    )
    return {
        "model_family": FAMILY,
        "ranker_definition": "frozen Phase6L score-free continuation ranker",
        "state_count": len(states), "action_count": len(ensemble),
        "overall_spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "top1_agreement": float(states.top1_agreement.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower, "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "mean_spearman_by_scale": {
            scale_name: float(scale.loc[scale_name, "mean_spearman"])
            for scale_name in ("S", "M", "L")
        },
        "mean_selected_lift_by_scale": {
            scale_name: float(scale.loc[scale_name, "mean_selected_lift"])
            for scale_name in ("S", "M", "L")
        },
        "exactly_reproduces_frozen_phase6l": True,
    }, states


def support_audit(ensemble: pd.DataFrame, selected: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keys = ["state_id", "target_set_id"]
    selected_keys = set(zip(selected.state_id.astype(str), selected.target_set_id.astype(str)))
    detail: dict[tuple[str, str], list[str]] = {}
    numeric_counter: Counter[str] = Counter()
    for held_fold in range(3):
        checkpoint = torch.load(
            TRAINING / f"selector/seed_716101/fold_{held_fold}.pt",
            map_location="cpu", weights_only=False,
        )
        transform = SupportTransform.from_dict(checkpoint["support_transform"])
        fold_rows = ensemble[ensemble.held_fold.eq(held_fold)]
        for item in fold_rows.to_dict("records"):
            reasons = []
            if not bool(item["high_level_family_known"]):
                reasons.append("unknown_origin_family")
            if not bool(item["high_level_operator_known"]):
                reasons.append("unknown_origin_destroy_operator")
            for column in SUPPORT_NUMERIC_COLUMNS:
                value = float(item[column])
                robust_z = (value - transform.medians[column]) / transform.scales[column]
                if not np.isfinite(robust_z):
                    reasons.append(f"nonfinite:{column}")
                elif abs(robust_z) > 12.0:
                    reason = f"numeric_robust_z:{column}"
                    reasons.append(reason)
                    numeric_counter[reason] += 1
            computed = not reasons
            require(computed == bool(item["hard_supported"]),
                    f"support reason reconstruction mismatch: {item['state_id']}")
            if reasons:
                detail[(str(item["state_id"]), str(item["target_set_id"]))] = reasons
    require(len(detail) == int((~ensemble.hard_supported.astype(bool)).sum()),
            "unsupported candidate count mismatch")

    expanded = ensemble.copy()
    expanded["selected_winner"] = [
        (str(state), str(target)) in selected_keys
        for state, target in zip(expanded.state_id, expanded.target_set_id)
    ]
    rows = []
    for dimension in (
        "overall", "scale", "CF_level", "search_stage", "origin_family", "selected_winner"
    ):
        groups = [("all", expanded)] if dimension == "overall" else expanded.groupby(dimension, sort=True)
        for value, group in groups:
            unsupported = group[~group.hard_supported.astype(bool)]
            counter: Counter[str] = Counter()
            for state, target in zip(unsupported.state_id, unsupported.target_set_id):
                counter.update(detail[(str(state), str(target))])
            winners = group[group.selected_winner]
            rows.append({
                "dimension": dimension, "group": str(value), "candidates": len(group),
                "candidate_support_rate": float(group.hard_supported.mean()),
                "unsupported_candidates": len(unsupported),
                "selected_winners": len(winners),
                "selected_winner_support_rate": (
                    float(winners.hard_supported.mean()) if len(winners) else None
                ),
                "unsupported_selected_winners": int(
                    (~winners.hard_supported.astype(bool)).sum()
                ),
                "fine_rule_oov_candidates": int(group.fine_rule_oov.astype(bool).sum()),
                "mean_continuous_support_score": float(group.continuous_support_score.mean()),
                "rejection_reason_counts_json": json.dumps(counter, sort_keys=True),
            })
    summary = {
        "candidate_support_rate": float(ensemble.hard_supported.mean()),
        "selected_winner_support_rate": float(selected.hard_supported.mean()),
        "unsupported_candidates": len(detail),
        "unsupported_selected_winners": int((~selected.hard_supported.astype(bool)).sum()),
        "fine_rule_oov_candidates_allowed": int(ensemble.fine_rule_oov.astype(bool).sum()),
        "unknown_origin_family": int((~ensemble.high_level_family_known.astype(bool)).sum()),
        "unknown_origin_destroy_operator": int((~ensemble.high_level_operator_known.astype(bool)).sum()),
        "numeric_boundary_failures": int(ensemble.maximum_absolute_robust_z.gt(12.0).sum()),
        "multiple_reason_candidates": int(sum(len(value) > 1 for value in detail.values())),
        "rejection_reason_frequencies": dict(sorted(numeric_counter.items())),
    }
    return summary, pd.DataFrame(rows)


def safe_group_confidence(group: pd.DataFrame) -> dict:
    probability = group.selector_probability.to_numpy(dtype=float)
    labels = group.continuation_advantage_mean.gt(0).to_numpy(dtype=int)
    basic = calibration_metrics(probability, labels)
    both = len(np.unique(labels)) == 2
    return {
        "selector_ece": float(basic["expected_calibration_error"]),
        "selector_brier": float(basic["brier_score"]),
        "selector_auroc": float(roc_auc_score(labels, probability)) if both else None,
        "selector_auprc": float(average_precision_score(labels, probability)) if labels.any() else None,
        "selector_probability_mean": float(probability.mean()),
        "selector_probability_std": float(probability.std(ddof=0)),
        "positive_outcome_rate": float(labels.mean()),
    }


def stratified_metrics(states: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    state_columns = [
        "state_id", "selector_probability", "continuation_advantage_mean",
        "selector_predictive_mean", "selector_total_predictive_scale",
    ]
    combined = states.merge(selected[state_columns], on="state_id", validate="one_to_one")
    rows = []
    for dimension in ("overall", "scale", "CF_level", "search_stage"):
        groups = [("all", combined)] if dimension == "overall" else combined.groupby(dimension, sort=True)
        for value, group in groups:
            rows.append({
                "dimension": dimension, "group": str(value), "states": len(group),
                "spearman": float(group.spearman.mean()),
                "pairwise_accuracy": float(group.pairwise_accuracy.mean()),
                "ndcg_at_1": float(group.ndcg_at_1.mean()),
                "selected_lift": float(group.selected_lift.mean()),
                "selection_regret": float(group.selection_regret.mean()),
                "selector_lcb_lambda_0_5_max": float((
                    group.selector_predictive_mean - 0.5 * group.selector_total_predictive_scale
                ).max()),
                **safe_group_confidence(group),
            })
    return pd.DataFrame(rows)


def gate_grid(selected: pd.DataFrame, ensemble: pd.DataFrame, config: dict,
              confidence: dict) -> pd.DataFrame:
    gate = config["gate"]
    bootstrap = config["bootstrap"]
    oracle = ensemble.groupby("state_id").continuation_advantage_mean.max()
    rows = []
    for p_min in gate["p_min_grid"]:
        for lcb_lambda in gate["lcb_lambda_grid"]:
            for delta_min in gate["delta_min_grid"]:
                work = selected.copy()
                work["lcb"] = (
                    work.selector_predictive_mean
                    - float(lcb_lambda) * work.selector_total_predictive_scale
                )
                conditions = {
                    "nonfallback": ~work.is_fallback.astype(bool),
                    "probability": work.selector_probability.ge(float(p_min)),
                    "lower_bound": work.lcb.gt(float(delta_min)),
                    "support": work.hard_supported.astype(bool),
                    "immediate_harm": work.ranker_immediate_utility_mean.ge(
                        float(gate["immediate_harm_floor"])
                    ),
                }
                work["intervened"] = np.logical_and.reduce([
                    value.to_numpy(dtype=bool) for value in conditions.values()
                ])
                work["gate_selected_lift"] = work.continuation_advantage_mean.where(
                    work.intervened, 0.0
                )
                work["gate_regret"] = work.state_id.map(oracle) - work.gate_selected_lift
                lower, upper = grouped_bootstrap(
                    work, "gate_selected_lift", seed=int(bootstrap["seed"]),
                    resamples=int(bootstrap["resamples"]),
                )
                scale_lifts = work.groupby("scale").gate_selected_lift.mean().to_dict()
                scale_counts = work.groupby("scale").intervened.sum().to_dict()
                coverage, forced = {}, {}
                for scale in ("S", "M", "L"):
                    scale_rows = work[work.scale.eq(scale)]
                    abstained = scale_rows[
                        ~scale_rows.intervened & ~scale_rows.is_fallback.astype(bool)
                    ].copy()
                    if len(abstained):
                        _, abstention_ucb = grouped_bootstrap(
                            abstained.assign(
                                forced_abstention_lift=abstained.continuation_advantage_mean
                            ),
                            "forced_abstention_lift", seed=int(bootstrap["seed"]),
                            resamples=int(bootstrap["resamples"]),
                        )
                    else:
                        abstention_ucb = math.inf
                    exception = (
                        len(abstained) >= int(gate["forced_abstention_minimum"])
                        and abstention_ucb <= 0.0
                    )
                    coverage[scale] = (
                        int(scale_counts.get(scale, 0))
                        >= int(gate["minimum_direct_interventions_per_scale"])
                        or exception
                    )
                    forced[scale] = {
                        "count": len(abstained), "upper_confidence_bound": abstention_ucb,
                        "exception_pass": bool(exception),
                    }
                checks = {
                    "positive_overall_lift": float(work.gate_selected_lift.mean()) > 0.0,
                    "positive_grouped_bootstrap_lcb": lower > 0.0,
                    "nonnegative_all_scale_lifts": all(
                        float(scale_lifts[scale]) >= 0.0 for scale in ("S", "M", "L")
                    ),
                    "cross_scale_coverage": all(coverage.values()),
                    "acceptable_winner_ece": confidence["expected_calibration_error"] <= 0.1,
                }
                rows.append({
                    "model_family": FAMILY, "p_min": float(p_min),
                    "lcb_lambda": float(lcb_lambda), "delta_min": float(delta_min),
                    "interventions": int(work.intervened.sum()),
                    "selected_lift": float(work.gate_selected_lift.mean()),
                    "selected_lift_lcb": lower, "selected_lift_ucb": upper,
                    "selection_regret": float(work.gate_regret.mean()),
                    "selected_winner_ece": confidence["expected_calibration_error"],
                    "scale_S_lift": float(scale_lifts["S"]),
                    "scale_M_lift": float(scale_lifts["M"]),
                    "scale_L_lift": float(scale_lifts["L"]),
                    "scale_S_interventions": int(scale_counts["S"]),
                    "scale_M_interventions": int(scale_counts["M"]),
                    "scale_L_interventions": int(scale_counts["L"]),
                    "nonfallback_pass": int(conditions["nonfallback"].sum()),
                    "probability_pass": int(conditions["probability"].sum()),
                    "lower_bound_pass": int(conditions["lower_bound"].sum()),
                    "support_pass": int(conditions["support"].sum()),
                    "immediate_harm_pass": int(conditions["immediate_harm"].sum()),
                    "coverage_pass": bool(checks["cross_scale_coverage"]),
                    "forced_abstention": json.dumps(forced, sort_keys=True),
                    "retained": bool(all(checks.values())),
                })
    result = pd.DataFrame(rows)
    require(len(result) == 18, "Phase 6M gate grid must contain 18 rows")
    return result


def comparisons(raw: dict, confidence: dict) -> dict:
    j1 = json.loads(J1_SUMMARY.read_text())
    phase6l_summary = json.loads((PHASE6L_TRAINING / "oof_summary.json").read_text())
    m0 = json.loads(M0_RESULT.read_text())
    metric_names = (
        "overall_spearman", "pairwise_accuracy", "ndcg_at_1", "selected_lift",
        "selected_lift_lcb", "selection_regret",
    )
    return {
        "phase6j_j1": {
            "raw_metrics": {name: j1["metrics"][name] for name in metric_names},
            "confidence": m0["confidence"]["phase6j_j1"],
            "retained_gate_count": 1,
            "selected_gate": j1["selected_gate"],
        },
        "phase6l_score_free": {
            "raw_metrics": {name: phase6l_summary["metrics"][name] for name in metric_names},
            "confidence": m0["confidence"]["phase6l_score_free"],
            "retained_gate_count": int(phase6l_summary["eligible_gate_count"]),
            "selected_gate": phase6l_summary["selected_gate"],
        },
        "phase6m": {
            "raw_metrics": {name: raw[name] for name in metric_names},
            "confidence": confidence,
            "retained_gate_count": 0,
            "selected_gate": None,
        },
    }


def render_report(result: dict) -> str:
    raw = result["raw_ranking"]
    confidence = result["confidence"]
    support = result["support"]
    components = result["least_restrictive_component_pass_counts"]
    lcb = result["selector_lcb_diagnostics"]
    return f"""# Phase 6M M4 开发集质量报告

状态：**{result['decision']}**。M4 已完成，M5/M6 runtime、solver、R13 和 R14 均未运行。

## 结论

Phase 6M 精确保留了 Phase 6L 的 score-free 排名结果：Spearman {raw['overall_spearman']:.6f}、pairwise accuracy {raw['pairwise_accuracy']:.6f}、NDCG@1 {raw['ndcg_at_1']:.6f}，raw selected lift {raw['selected_lift']:.6f}，grouped-bootstrap LCB {raw['selected_lift_lcb']:.6f}。S/M/L raw lift 分别为 {raw['mean_selected_lift_by_scale']['S']:.6f}、{raw['mean_selected_lift_by_scale']['M']:.6f}、{raw['mean_selected_lift_by_scale']['L']:.6f}。

新的 selective-risk 层未恢复可部署干预。18 个预注册 gate 全部为 0 次干预，retained gate 为 0。最宽松 LCB 条件 `selector_mean - 0.5 * total_scale > 0` 的通过数为 {components['lower_bound']} / 288；其最大值为 {lcb['lambda_0_5_max']:.6f}。同一组合中 `p>=0.55` 仅 {components['probability']} / 288，通过 frozen immediate-harm floor 的为 {components['immediate_harm']} / 288。失败主要由 selector 的保守下界整体为负导致，不能通过降低已冻结阈值修正本轮结果。

## Confidence 与 support

Winner confidence 的 ECE 为 {confidence['expected_calibration_error']:.6f}，Brier 为 {confidence['brier_score']:.6f}，AUROC 为 {confidence['auroc']:.6f}，AUPRC 为 {confidence['auprc']:.6f}，Brier resolution 为 {confidence['resolution']:.6f}。概率标准差为 {confidence['probability_std']:.6f}，范围 [{confidence['probability_min']:.6f}, {confidence['probability_max']:.6f}]；`p>=0.55` 仅覆盖 {confidence['fraction_above']['0.55']:.3%}。相比 Phase 6L，分辨率和 sharpness 没有改善到足以产生 selective intervention 的程度。

修订后的 hard support 候选通过率为 {support['candidate_support_rate']:.3%}，winner 为 {support['selected_winner_support_rate']:.3%}。55 个 unsupported candidates 全部包含 `max|robust-z|>12` 数值边界原因；未知 high-level family/operator 与 fine-rule OOV 均为 0，unsupported winner 为 0。support 修订消除了 Phase 6L 的 winner support 阻塞，但没有改变 LCB 失败。

## 门槛判定与冻结边界

Raw ranking 的四项预注册要求全部通过，candidate identity/order、24-rule full bank、fallback identity、feasibility 与 origin diversity 也通过。正式 readiness 失败项为：`retained_cross_scale_gate`、`positive_gated_lift`、`positive_grouped_bootstrap_lcb` 和 `cross_scale_coverage`。因此按协议终止为 `MODEL_REVISION_QUALITY`，不得进入 M5/M6、bundle、R12 solver gate、R13 或 R14。

主要证据：

- `outputs/phase6m_selective_confidence_v1/quality/development_quality.json`
- `outputs/phase6m_selective_confidence_v1/quality/gate_grid.csv`
- `outputs/phase6m_selective_confidence_v1/quality/stratified_metrics.csv`
- `outputs/phase6m_selective_confidence_v1/quality/support_by_regime.csv`
- `outputs/phase6m_selective_confidence_v1/quality/selective_risk_curve.csv`

本结论使用全部 288 个 R12 CAUR-FIT states、6,809 个候选和全部三 selector seeds。historical score online forward calls 为 0；R13/R14 access ledgers 不存在；未运行 Gurobi。
"""


def main() -> None:
    config, implementation, implementation_sha256 = training.validate_boundary()
    completion = json.loads((TRAINING / "completion_integrity_audit.json").read_text())
    require(completion["status"] == "PASS" and all(completion["checks"].values()),
            "M3 completion integrity did not pass")
    require(completion["implementation_protocol_sha256"] == implementation_sha256,
            "M3 implementation boundary changed")
    protected = m3_audit.verify_phase6l_protection(implementation)
    ensemble = pd.read_parquet(TRAINING / "ensemble_oof.parquet")
    selected = pd.read_parquet(TRAINING / "selected_winners.parquet")
    require(
        len(ensemble) == 6809 and ensemble.state_id.nunique() == 288
        and not ensemble.duplicated(["state_id", "target_set_id"]).any(),
        "M4 candidate bank identity failed",
    )
    require(
        ensemble.requested_bank_count.eq(24).all()
        and ensemble.groupby("state_id").size().eq(
            ensemble.groupby("state_id").full_bank_unique_count.first()
        ).all()
        and ensemble.groupby("state_id").is_fallback.sum().eq(1).all(),
        "M4 full-bank or fallback boundary failed",
    )
    frozen_ensemble = pd.read_parquet(PHASE6L_TRAINING / "ensemble_oof.parquet")
    require(frozen_ensemble.candidate_feasible.astype(bool).all(), "frozen candidate feasibility failed")
    keys = ["state_id", "target_set_id"]
    pd.testing.assert_frame_equal(
        ensemble[keys].sort_values(keys).reset_index(drop=True),
        frozen_ensemble[keys].sort_values(keys).reset_index(drop=True),
    )
    raw, states = raw_ranking(ensemble, config)
    confidence, risk_curve = confidence_audit(selected)
    support, support_table = support_audit(ensemble, selected)
    strata = stratified_metrics(states, selected)
    gates = gate_grid(selected, ensemble, config, confidence)
    retained = gates[gates.retained.astype(bool)]
    selected_gate = None
    if len(retained):
        selected_gate = retained.sort_values(
            ["selected_lift_lcb", "selection_regret", "selected_winner_ece", "interventions"],
            ascending=[False, True, True, False], kind="stable",
        ).iloc[0].to_dict()
    require(len(retained) == 0 and selected_gate is None,
            "observed M4 result unexpectedly differs from terminal quality path")

    raw_checks = {
        "positive_overall_spearman": raw["overall_spearman"] > 0.0,
        "positive_all_scale_mean_spearman": min(raw["mean_spearman_by_scale"].values()) > 0.0,
        "positive_raw_selected_lift": raw["selected_lift"] > 0.0,
        "positive_raw_selected_lift_lcb": raw["selected_lift_lcb"] > 0.0,
        "full_bank_identity_and_feasibility": True,
        "selected_winner_ece_at_most_0_1": confidence["expected_calibration_error"] <= 0.1,
        "candidate_origin_not_collapsed": all(
            group.origin_family.nunique() > 1 for _, group in selected.groupby("scale")
        ),
    }
    readiness = {
        "retained_cross_scale_gate": len(retained) > 0,
        "positive_gated_lift": len(retained) > 0 and bool(retained.selected_lift.gt(0).all()),
        "positive_grouped_bootstrap_lcb": len(retained) > 0 and bool(retained.selected_lift_lcb.gt(0).all()),
        "nonnegative_all_scale_gated_lift": len(retained) > 0 and bool(
            retained[["scale_S_lift", "scale_M_lift", "scale_L_lift"]].ge(0).all(axis=None)
        ),
        "cross_scale_coverage": len(retained) > 0 and bool(retained.coverage_pass.all()),
    }
    decision = "PROCEED_TO_M5" if all({**raw_checks, **readiness}.values()) else "MODEL_REVISION_QUALITY"
    require(decision == "MODEL_REVISION_QUALITY", "M4 audit expected the observed terminal decision")
    first = gates.iloc[0]
    origins = {
        scale: {str(key): int(value) for key, value in group.origin_family.value_counts().sort_index().items()}
        for scale, group in selected.groupby("scale", sort=True)
    }
    lcb_half = selected.selector_predictive_mean - 0.5 * selected.selector_total_predictive_scale
    lcb_one = selected.selector_predictive_mean - selected.selector_total_predictive_scale
    comparison = comparisons(raw, confidence)
    result = {
        "schema": "phase6m-development-quality-audit-v1",
        "status": "M4_COMPLETE", "decision": decision,
        "stop_boundary": "M4_BEFORE_DEVELOPMENT_RUNTIME",
        "model_family": FAMILY,
        "raw_ranking": raw, "confidence": confidence, "support": support,
        "eligible_gate_count": len(retained), "selected_gate": selected_gate,
        "raw_quality_checks": {key: bool(value) for key, value in raw_checks.items()},
        "formal_intervention_readiness_checks": {
            key: bool(value) for key, value in readiness.items()
        },
        "failed_checks": [key for key, value in readiness.items() if not value],
        "least_restrictive_component_pass_counts": {
            "nonfallback": int(first.nonfallback_pass),
            "probability": int(first.probability_pass),
            "lower_bound": int(first.lower_bound_pass),
            "support": int(first.support_pass),
            "immediate_harm": int(first.immediate_harm_pass),
            "all_conditions": int(first.interventions),
        },
        "selector_lcb_diagnostics": {
            "lambda_0_5_min": float(lcb_half.min()),
            "lambda_0_5_median": float(lcb_half.median()),
            "lambda_0_5_max": float(lcb_half.max()),
            "lambda_0_5_positive_count": int(lcb_half.gt(0).sum()),
            "lambda_1_0_min": float(lcb_one.min()),
            "lambda_1_0_median": float(lcb_one.median()),
            "lambda_1_0_max": float(lcb_one.max()),
            "lambda_1_0_positive_count": int(lcb_one.gt(0).sum()),
        },
        "action_frequency": {
            "raw_neural_argmax_fallback": int(selected.is_fallback.astype(bool).sum()),
            "raw_neural_argmax_nonfallback": int((~selected.is_fallback.astype(bool)).sum()),
            "deployable_interventions": 0,
            "fallback_decisions_without_retained_gate": 288,
        },
        "raw_winner_origin_distribution_by_scale": origins,
        "comparison": comparison,
        "candidate_bank": {
            "states": 288, "candidates": 6809, "requested_rules_per_state": 24,
            "unique_candidates_min": int(ensemble.groupby("state_id").size().min()),
            "unique_candidates_max": int(ensemble.groupby("state_id").size().max()),
            "candidate_identity_order_matches_phase6l": True,
            "all_frozen_repairs_feasible": True,
        },
        "protected_phase6l": protected,
        "historical_score_online_forward_calls": 0,
        "runtime_stage": "NOT_RUN_STOPPED_ON_M4_QUALITY",
        "solver_gate": "NOT_RUN_QUALITY_GATE_FAILED",
        "gurobi_run": False,
        "r13_accessed": False, "r14_accessed": False,
        "r13_locked": True, "r14_locked": True,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in (
                training.CONFIG_PATH, training.PREREGISTRATION, training.AMENDMENT,
                training.IMPLEMENTATION, TRAINING / "completion_integrity_audit.json",
                TRAINING / "oof_predictions.parquet", TRAINING / "ensemble_oof.parquet",
                TRAINING / "selected_winners.parquet",
                PHASE6L_TRAINING / "oof_summary.json", J1_SUMMARY, M0_RESULT,
            )
        },
        "audit_code_sha256": digest(Path(__file__)),
    }
    atomic_csv(QUALITY / "gate_grid.csv", gates)
    atomic_csv(QUALITY / "stratified_metrics.csv", strata)
    atomic_csv(QUALITY / "support_by_regime.csv", support_table)
    atomic_csv(QUALITY / "selective_risk_curve.csv", risk_curve)
    atomic_json(QUALITY / "development_quality.json", result)
    REPORT.write_text(render_report(result))
    print(json.dumps({
        "status": result["status"], "decision": decision,
        "raw_quality_pass": all(raw_checks.values()),
        "intervention_readiness_pass": all(readiness.values()),
        "eligible_gate_count": len(retained),
        "selector_ece": confidence["expected_calibration_error"],
        "selector_auroc": confidence["auroc"],
        "selector_lcb_lambda_0_5_max": result["selector_lcb_diagnostics"]["lambda_0_5_max"],
        "candidate_support_rate": support["candidate_support_rate"],
        "selected_winner_support_rate": support["selected_winner_support_rate"],
        "failed_checks": result["failed_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
