#!/usr/bin/env python3
"""Build the read-only Phase 6N N0 architecture and data audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.ni.live_policy import InterventionDecision  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from scripts.audit_phase6m_failure_attribution import (  # noqa: E402
    verify_predecessor_evidence,
)


STARTING_COMMIT = "8efe396e6cca44493a18c07720c4effaf2add7c1"
MANUAL = Path("/home/liulei/下载/phase6n_candidate_conditioned_csg_codex_instructions.md")
NAMESPACE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1"
AUDIT = NAMESPACE / "audit"
REPORT = ROOT / "docs/reports/phase6n_architecture_data_audit.md"
GROUPED = (
    ROOT
    / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet"
)
RAW = (
    ROOT
    / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_seed_labels.parquet"
)
ENSEMBLE = (
    ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
)
PHASE6L_PROTOCOL = (
    ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/training_protocol.json"
)
BASE_CHECKPOINT = (
    ROOT / "outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt"
)
PHASE6L_CHECKPOINT = (
    ROOT
    / "outputs/phase6l_legacy_score_decoupling_v1/training/oof"
    / "L1_SCORE_FREE_CONT_FROZEN/seed_706101/fold_0.pt"
)

CHEAP_COLLISION_FEATURES = (
    "destroy_target_fraction",
    "critical_overlap_fraction",
    "bottleneck_overlap_fraction",
    "fallback_overlap_fraction",
    "fallback_jaccard",
    "normalized_diversity_rank",
)
SIMILAR_DISTANCE = 0.10
MATERIAL_ADVANTAGE_GAP = 0.01
STRONG_ADVANTAGE_GAP = 0.02


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = text.rstrip() + "\n"
    if path.exists() and path.read_text() != value:
        raise RuntimeError(f"refusing to replace changed N0 evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame.to_csv(index=False)
    if path.exists() and path.read_text() != value:
        raise RuntimeError(f"refusing to replace changed N0 evidence: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def protected_phase6l_phase6m() -> dict:
    paths: list[Path] = []
    for phase in ("phase6l", "phase6m"):
        paths.extend(sorted((ROOT / "docs/reports").glob(f"{phase}*.md")))
    for namespace in (
        "outputs/phase6l_legacy_score_decoupling_v1",
        "outputs/phase6m_selective_confidence_v1",
    ):
        paths.extend(sorted(path for path in (ROOT / namespace).rglob("*") if path.is_file()))
    unique = sorted(set(paths))
    return {
        str(path.relative_to(ROOT)): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in unique
    }


def starting_boundary() -> tuple[dict, dict]:
    require(MANUAL.is_file(), "Phase 6N execution manual is missing")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "Phase 6M terminal commit is not an ancestor of HEAD",
    )
    phase6m_decision_path = (
        ROOT / "outputs/phase6m_selective_confidence_v1/final/final_decision.json"
    )
    phase6m_decision = json.loads(phase6m_decision_path.read_text())
    require(
        phase6m_decision["decision"] == "MODEL_REVISION_QUALITY",
        "Phase 6M terminal decision changed",
    )
    ledgers = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        NAMESPACE / "r13_selection/access_ledger.json",
        NAMESPACE / "r14_holdout/access_ledger.json",
    ]
    existing = [str(path.relative_to(ROOT)) for path in ledgers if path.exists()]
    require(not existing, f"R13/R14 access detected: {existing}")

    chained = verify_predecessor_evidence()
    manifest = protected_phase6l_phase6m()
    manifest_path = AUDIT / "protected_phase6l_phase6m_evidence.json"
    if manifest_path.exists():
        require(
            json.loads(manifest_path.read_text()) == manifest,
            "protected Phase 6L/6M evidence changed after N0 started",
        )
    else:
        atomic_json(manifest_path, manifest)
    boundary = {
        "starting_commit": STARTING_COMMIT,
        "starting_commit_is_ancestor": True,
        "head_at_n0": git("rev-parse", "HEAD"),
        "manual": {"path": str(MANUAL), "sha256": digest(MANUAL)},
        "phase6m_terminal": {
            "path": str(phase6m_decision_path.relative_to(ROOT)),
            "sha256": digest(phase6m_decision_path),
            "decision": phase6m_decision["decision"],
        },
        "protected_phase6l_phase6m": {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "files": len(manifest),
            "bytes": int(sum(item["bytes"] for item in manifest.values())),
        },
        "pre_phase6l_chained_manifest_verification": chained,
        "r13_r14_checked_paths": [str(path.relative_to(ROOT)) for path in ledgers],
        "r13_accessed": False,
        "r14_accessed": False,
        "gurobi_run": False,
    }
    return boundary, manifest


def checkpoint_audit() -> dict:
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    phase6l = torch.load(PHASE6L_CHECKPOINT, map_location="cpu", weights_only=False)
    base_state = base["model_state"]
    trainable_state = phase6l["trainable_model_state"]
    tensorizer = CSGTensorizer(include_reverse=True, dtype=torch.float32)
    protocol = json.loads(PHASE6L_PROTOCOL.read_text())
    model_config = base["model_config"]
    require(model_config["layers"] == 2, "unexpected frozen graph layer count")
    require(
        phase6l["base_checkpoint_sha256"] == digest(BASE_CHECKPOINT),
        "Phase 6L checkpoint does not reference the actual Phase 6F checkpoint",
    )
    relation_specs = tensorizer.relation_specs
    return {
        "graph_construction": {
            "schema": "CSG-1.0",
            "node_types": 8,
            "canonical_relation_types": 20,
            "derived_reverse_relations": True,
            "actual_tensor_relation_types": len(relation_specs),
            "tensor_dtype": str(tensorizer.dtype),
            "tensor_schema_hash": tensorizer.tensor_schema_hash,
        },
        "initialization_checkpoint": {
            "path": str(BASE_CHECKPOINT.relative_to(ROOT)),
            "sha256": digest(BASE_CHECKPOINT),
            "schema": base["schema"],
            "model_config": model_config,
            "state_tensors": len(base_state),
            "stored_parameters": int(sum(value.numel() for value in base_state.values())),
        },
        "phase6l_checkpoint": {
            "path": str(PHASE6L_CHECKPOINT.relative_to(ROOT)),
            "sha256": digest(PHASE6L_CHECKPOINT),
            "schema": phase6l["schema"],
            "model_family": phase6l["model_family"],
            "base_checkpoint_sha256": phase6l["base_checkpoint_sha256"],
            "trainable_state_tensors": len(trainable_state),
            "trainable_parameters": int(
                sum(value.numel() for value in trainable_state.values())
            ),
            "reported_total_parameters": protocol["families"][
                "L1_SCORE_FREE_CONT_FROZEN"
            ]["total_parameters"],
        },
        "actual_phase6l_path": [
            "CSGState -> CSGTensorizer(float32, canonical plus reverse relations)",
            "two-layer FULL_CSG CSGStateEncoder once per state",
            "final node embeddings remain available for all eight node types",
            "graph embedding pools each node type then adds progress and bottleneck category",
            "TargetSetEncoder indexes target OP embeddings for every candidate",
            "target mean, max, graph-query attention, graph embedding and normalized size",
            "Phase6L fuses candidate, fallback, their difference and score-free cheap features",
            "direct continuation, beats-fallback and auxiliary immediate heads",
        ],
        "candidate_identity_present": True,
        "candidate_identity_entry": (
            "batch.target_operation_indices and batch.target_action_index select the actual "
            "destroyed OP membership before target mean/max/attention pooling"
        ),
        "information_lost_by_current_aggregation": [
            "which typed relations cross from target OP nodes to non-target boundary nodes",
            "direction-separated and relation-family-separated boundary identities",
            "explicit embeddings of candidate intersect critical operations",
            "explicit embeddings of candidate intersect active bottleneck operations",
            "explicit critical synchronization-chain boundary embeddings",
            "individual target node identities after permutation-invariant pooling",
        ],
        "important_interpretation": (
            "Phase6L is already target-membership-conditioned; Phase6N tests whether explicit "
            "candidate boundary and critical/bottleneck structure adds material discrimination"
        ),
        "code_sha256": {
            relative: digest(ROOT / relative)
            for relative in (
                "configs/csg_v1_schema.json",
                "rcias_clgri/ni/tensorize.py",
                "rcias_clgri/ni/batching.py",
                "rcias_clgri/ni/encoder.py",
                "rcias_clgri/ni/action_encoder.py",
                "rcias_clgri/ni/phase6l_score_free_model.py",
            )
        },
    }


def immediate_audit(grouped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ensemble = pd.read_parquet(
        ENSEMBLE,
        columns=["state_id", "target_set_id", "ensemble_immediate_utility"],
    )
    frame = grouped.merge(
        ensemble, on=["state_id", "target_set_id"], validate="one_to_one"
    ).copy()
    frame["continuation_positive"] = frame.continuation_advantage_mean.gt(0)
    frame["realized_immediate_negative"] = frame.immediate_utility.lt(0)
    frame["realized_immediate_below_old_floor"] = frame.immediate_utility.lt(-0.005)
    frame["old_predicted_immediate_gate_blocked"] = (
        frame.ensemble_immediate_utility.lt(-0.005)
    )
    columns = [
        "instance_id",
        "scale",
        "CF_level",
        "state_id",
        "target_set_id",
        "origin_family",
        "is_fallback",
        "immediate_utility",
        "ensemble_immediate_utility",
        "continuation_advantage_mean",
        "continuation_advantage_std",
        "beats_fallback",
        "continuation_positive",
        "realized_immediate_negative",
        "realized_immediate_below_old_floor",
        "old_predicted_immediate_gate_blocked",
    ]
    output = frame[columns].sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)

    def summarize(part: pd.DataFrame) -> dict:
        positive = part.continuation_positive
        blocked = part.old_predicted_immediate_gate_blocked
        return {
            "candidates": len(part),
            "continuation_positive": int(positive.sum()),
            "continuation_positive_with_negative_realized_immediate": int(
                (positive & part.realized_immediate_negative).sum()
            ),
            "continuation_positive_with_realized_immediate_below_minus_0_005": int(
                (positive & part.realized_immediate_below_old_floor).sum()
            ),
            "old_predicted_gate_blocked": int(blocked.sum()),
            "old_predicted_gate_blocked_fraction": float(blocked.mean()),
            "useful_positive_candidates_blocked": int((positive & blocked).sum()),
            "blocked_positive_rate": float(part.loc[blocked, "continuation_positive"].mean()),
            "blocked_mean_continuation_advantage": float(
                part.loc[blocked, "continuation_advantage_mean"].mean()
            ),
        }

    winners = frame.sort_values(
        ["state_id", "ensemble_immediate_utility", "target_set_id"],
        ascending=[True, False, True],
        kind="stable",
    ).groupby("state_id", sort=False).head(1)
    actual_pearson = pearsonr(frame.immediate_utility, frame.continuation_advantage_mean)
    actual_spearman = spearmanr(frame.immediate_utility, frame.continuation_advantage_mean)
    predicted_pearson = pearsonr(
        frame.ensemble_immediate_utility, frame.continuation_advantage_mean
    )
    predicted_spearman = spearmanr(
        frame.ensemble_immediate_utility, frame.continuation_advantage_mean
    )
    summary = {
        "definition": (
            "realized immediate utility diagnoses local deterioration; the historical hard "
            "gate used the Phase6L ensemble immediate-utility prediction at -0.005"
        ),
        "old_gate_floor": -0.005,
        "overall": summarize(frame),
        "by_scale": {
            str(scale): summarize(part) for scale, part in frame.groupby("scale", sort=True)
        },
        "correlation": {
            "realized_immediate_vs_continuation_pearson": float(actual_pearson.statistic),
            "realized_immediate_vs_continuation_spearman": float(actual_spearman.statistic),
            "predicted_immediate_vs_continuation_pearson": float(
                predicted_pearson.statistic
            ),
            "predicted_immediate_vs_continuation_spearman": float(
                predicted_spearman.statistic
            ),
        },
        "diagnostic_immediate_argmax_winners": {
            "states": len(winners),
            "continuation_positive": int(winners.continuation_positive.sum()),
            "mean_continuation_advantage": float(
                winners.continuation_advantage_mean.mean()
            ),
        },
        "conclusion": (
            "The old predicted immediate gate removes many continuation-positive actions and "
            "its prediction is nearly uncorrelated with continuation advantage; it is not a "
            "valid Phase6N promotion dependency. Structural best-so-far safety remains mandatory."
        ),
    }
    return output, summary


def _target_jaccard(left: str, right: str) -> float:
    a, b = set(json.loads(left)), set(json.loads(right))
    union = a | b
    return 0.0 if not union else len(a & b) / len(union)


def collision_audit(grouped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    top_ids = set(
        grouped.sort_values(
            ["state_id", "continuation_advantage_mean", "target_set_id"],
            ascending=[True, False, True],
            kind="stable",
        ).groupby("state_id", sort=False).head(1)[
            ["state_id", "target_set_id"]
        ].itertuples(index=False, name=None)
    )
    # One directional nearest neighbor for every candidate with a same-state,
    # same-origin-family peer. Directional rows keep top-candidate coverage auditable.
    for (state_id, family), part in grouped.groupby(
        ["state_id", "origin_family"], sort=True
    ):
        ordered = part.sort_values("target_set_id", kind="stable").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        values = ordered[list(CHEAP_COLLISION_FEATURES)].to_numpy(dtype=float)
        distance = np.sqrt(np.mean((values[:, None, :] - values[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distance, np.inf)
        for index, anchor in ordered.iterrows():
            minimum = float(distance[index].min())
            choices = np.flatnonzero(np.isclose(distance[index], minimum, atol=1e-15, rtol=0))
            neighbor_index = min(
                choices, key=lambda value: str(ordered.iloc[int(value)].target_set_id)
            )
            neighbor = ordered.iloc[int(neighbor_index)]
            advantage_gap = abs(
                float(anchor.continuation_advantage_mean)
                - float(neighbor.continuation_advantage_mean)
            )
            record = {
                "instance_id": str(anchor.instance_id),
                "scale": str(anchor.scale),
                "CF_level": str(anchor.CF_level),
                "state_id": str(state_id),
                "origin_family": str(family),
                "anchor_target_set_id": str(anchor.target_set_id),
                "neighbor_target_set_id": str(neighbor.target_set_id),
                "cheap_feature_rms_distance": minimum,
                "target_operation_jaccard": _target_jaccard(
                    str(anchor.target_operation_ids), str(neighbor.target_operation_ids)
                ),
                "anchor_continuation_advantage": float(anchor.continuation_advantage_mean),
                "neighbor_continuation_advantage": float(
                    neighbor.continuation_advantage_mean
                ),
                "absolute_continuation_advantage_gap": advantage_gap,
                "cheap_similar_at_0_10": minimum <= SIMILAR_DISTANCE,
                "material_gap_at_0_01": advantage_gap >= MATERIAL_ADVANTAGE_GAP,
                "strong_gap_at_0_02": advantage_gap >= STRONG_ADVANTAGE_GAP,
                "collision_at_0_10_0_01": (
                    minimum <= SIMILAR_DISTANCE
                    and advantage_gap >= MATERIAL_ADVANTAGE_GAP
                ),
                "anchor_is_state_continuation_best": (
                    str(state_id), str(anchor.target_set_id)
                ) in top_ids,
            }
            for feature in CHEAP_COLLISION_FEATURES:
                record[f"anchor_{feature}"] = float(anchor[feature])
                record[f"neighbor_{feature}"] = float(neighbor[feature])
            rows.append(record)
    nearest = pd.DataFrame(rows).sort_values(
        ["state_id", "anchor_target_set_id"], kind="stable"
    ).reset_index(drop=True)
    require(len(nearest) > 0, "collision audit produced no nearest-neighbor rows")

    bucketed = grouped.copy()
    for feature in CHEAP_COLLISION_FEATURES:
        bucketed[f"bucket_{feature}"] = np.floor(
            bucketed[feature].to_numpy(dtype=float) / 0.10 + 0.5
        ).astype(int)
    bucket_keys = [
        "state_id",
        "origin_family",
        *(f"bucket_{feature}" for feature in CHEAP_COLLISION_FEATURES),
    ]
    buckets = bucketed.groupby(bucket_keys, sort=True).continuation_advantage_mean.agg(
        count="count", variance=lambda values: float(np.var(values, ddof=0))
    ).reset_index()
    multi = buckets[buckets["count"].ge(2)].copy()
    candidate_weighted_variance = float(
        np.average(multi["variance"], weights=multi["count"])
    ) if len(multi) else 0.0

    top = nearest[nearest.anchor_is_state_continuation_best]
    poor = top.collision_at_0_10_0_01
    similar = nearest.cheap_similar_at_0_10
    gap = nearest.absolute_continuation_advantage_gap
    corr_pearson = pearsonr(nearest.cheap_feature_rms_distance, gap)
    corr_spearman = spearmanr(nearest.cheap_feature_rms_distance, gap)
    examples = nearest[
        nearest.collision_at_0_10_0_01
    ].sort_values(
        ["absolute_continuation_advantage_gap", "cheap_feature_rms_distance"],
        ascending=[False, True],
        kind="stable",
    ).head(12)
    summary = {
        "features": list(CHEAP_COLLISION_FEATURES),
        "distance": "root-mean-square difference over six naturally [0,1] features",
        "same_origin_family_required": True,
        "directional_nearest_neighbor_rows": len(nearest),
        "candidates_without_same_family_peer": int(len(grouped) - len(nearest)),
        "similar_distance_threshold": SIMILAR_DISTANCE,
        "material_advantage_gap": MATERIAL_ADVANTAGE_GAP,
        "strong_advantage_gap": STRONG_ADVANTAGE_GAP,
        "similar_nearest_neighbors": int(similar.sum()),
        "material_collisions": int(nearest.collision_at_0_10_0_01.sum()),
        "strong_collisions": int((similar & nearest.strong_gap_at_0_02).sum()),
        "nearest_distance_vs_outcome_gap": {
            "pearson": float(corr_pearson.statistic),
            "spearman": float(corr_spearman.statistic),
        },
        "conditional_bucket_variance": {
            "bucket_width": 0.10,
            "multi_candidate_buckets": len(multi),
            "candidates_in_multi_candidate_buckets": int(multi["count"].sum()),
            "candidate_weighted_population_variance_approximation": candidate_weighted_variance,
            "median_bucket_population_variance": float(multi["variance"].median())
            if len(multi)
            else 0.0,
            "p90_bucket_population_variance": float(multi["variance"].quantile(0.9))
            if len(multi)
            else 0.0,
        },
        "continuation_best_candidate_separation": {
            "states": int(grouped.state_id.nunique()),
            "states_with_same_family_neighbor_for_best": len(top),
            "poorly_separated_best_candidates": int(poor.sum()),
            "poorly_separated_fraction_of_all_states": float(
                poor.sum() / grouped.state_id.nunique()
            ),
            "poorly_separated_fraction_when_peer_exists": float(poor.mean()),
        },
        "largest_collision_examples": examples[
            [
                "state_id",
                "anchor_target_set_id",
                "neighbor_target_set_id",
                "origin_family",
                "cheap_feature_rms_distance",
                "target_operation_jaccard",
                "absolute_continuation_advantage_gap",
            ]
        ].to_dict("records"),
        "interpretation": (
            "Cheap-context collisions are measured within state and origin family. Material "
            "outcome gaps among these neighbors support testing explicit target/boundary graph "
            "structure; this diagnostic alone does not prove that the new representation will win."
        ),
    }
    return nearest, summary


def sample_size_and_noise(grouped: pd.DataFrame) -> dict:
    raw = pd.read_parquet(RAW)
    keys = ["instance_id", "scale", "CF_level", "state_id", "target_set_id"]
    pivot = raw.pivot(index=keys, columns="continuation_seed", values="continuation_advantage")
    seeds = sorted(int(value) for value in pivot.columns)
    require(len(seeds) == 2, "Phase 6N N0 expects exactly two archived CRN seeds")
    left = pivot[seeds[0]].to_numpy(dtype=float)
    right = pivot[seeds[1]].to_numpy(dtype=float)
    absolute = np.abs(left - right)
    fold = grouped.groupby("oof_fold", sort=True).agg(
        instances=("instance_id", "nunique"),
        states=("state_id", "nunique"),
        candidates=("target_set_id", "size"),
    )
    per_instance_states = grouped.groupby("instance_id").state_id.nunique()
    per_state_candidates = grouped.groupby("state_id").size()
    return {
        "candidate_rows": len(grouped),
        "raw_crn_rows": len(raw),
        "states": int(grouped.state_id.nunique()),
        "instances": int(grouped.instance_id.nunique()),
        "states_per_instance": {
            "minimum": int(per_instance_states.min()),
            "maximum": int(per_instance_states.max()),
            "mean": float(per_instance_states.mean()),
        },
        "candidates_per_state": {
            "minimum": int(per_state_candidates.min()),
            "maximum": int(per_state_candidates.max()),
            "mean": float(per_state_candidates.mean()),
        },
        "structural_folds": {
            str(int(index)): {
                key: int(value) for key, value in row.items()
            }
            for index, row in fold.to_dict("index").items()
        },
        "independence_interpretation": {
            "candidate_rows_are_independent_examples": False,
            "state_clusters": 288,
            "instance_level_qualification_units": 18,
            "required_split": "whole instance/structural cell",
        },
        "two_crn_label_noise": {
            "continuation_seeds": seeds,
            "seed_outcome_pearson": float(np.corrcoef(left, right)[0, 1]),
            "mean_absolute_seed_difference": float(absolute.mean()),
            "median_absolute_seed_difference": float(np.median(absolute)),
            "p90_absolute_seed_difference": float(np.quantile(absolute, 0.9)),
            "p99_absolute_seed_difference": float(np.quantile(absolute, 0.99)),
            "sign_disagreement_fraction": float(np.mean((left > 0) != (right > 0))),
            "candidate_mean_population_std_mean": float(
                grouped.continuation_advantage_std.mean()
            ),
            "candidate_mean_population_std_p90": float(
                grouped.continuation_advantage_std.quantile(0.9)
            ),
        },
        "clustering_descriptives": {
            "raw_label_population_variance": float(
                raw.continuation_advantage.var(ddof=0)
            ),
            "variance_of_state_means": float(
                raw.groupby("state_id").continuation_advantage.mean().var(ddof=0)
            ),
            "variance_of_instance_means": float(
                raw.groupby("instance_id").continuation_advantage.mean().var(ddof=0)
            ),
            "note": (
                "These are descriptive variance components, not an independence-adjusted "
                "effective sample-size estimator. Inferential resampling remains instance-grouped."
            ),
        },
    }


class _AuditInterventionPolicy:
    def prepare_instance(self, instance, h1_schedule) -> None:
        self.prepared = (instance.instance_id, len(h1_schedule.operation_schedules))

    def decide(
        self,
        instance,
        current,
        *,
        state_id,
        destroy_count,
        search_progress,
        search_stage,
    ) -> InterventionDecision:
        del current, search_progress, search_stage
        return InterventionDecision(
            True,
            state_id,
            "phase6n-n0-safety-target",
            tuple(instance.operations[:destroy_count]),
            0.9,
            0.1,
            0.5,
            None,
            20,
            24,
            4,
        )


def best_so_far_behavior_audit() -> dict:
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    config = ALNSConfig(
        candidate_trials=1,
        iteration_limit=20,
        initial_temperature=10.0,
        cooling_rate=0.999,
    )

    def execute() -> tuple[object, list[dict]]:
        events: list[dict] = []
        result = solve_csgni(
            instance,
            60.0,
            1,
            _AuditInterventionPolicy(),
            alns_config=config,
            csgni_config=CSGNIConfig(intervention_rate=100),
            observer=events.append,
        )
        return result, events

    first, events = execute()
    second, repeated = execute()

    def signature(rows: list[dict]) -> list[tuple]:
        return [
            (
                row["iteration"],
                row["current_before"].makespan,
                row["candidate"].makespan,
                row["current_after"].makespan,
                row["best_before"].makespan,
                row["best_after"].makespan,
                row["accepted"],
                row["destroyed_operation_ids"],
                row["candidate"].candidate,
            )
            for row in rows
        ]

    accepted_worse = [
        row
        for row in events
        if row["accepted"]
        and row["current_after"].makespan > row["best_before"].makespan
    ]
    checks = {
        "all_decoded_candidates_feasible": all(
            row["candidate"].feasible
            and row["current_after"].feasible
            and row["best_after"].feasible
            for row in events
        ),
        "best_so_far_monotone_nonincreasing": all(
            row["best_after"].makespan <= row["best_before"].makespan
            for row in events
        ),
        "accepted_worse_current_observed": len(accepted_worse) > 0,
        "accepted_worse_never_overwrites_best": all(
            row["best_after"].makespan == row["best_before"].makespan
            for row in accepted_worse
        ),
        "returned_best_equals_final_best": (
            first.best.candidate == events[-1]["best_after"].candidate
            and first.best.makespan == events[-1]["best_after"].makespan
        ),
        "deterministic_repeated_execution": (
            signature(events) == signature(repeated)
            and first.best.candidate == second.best.candidate
            and first.best.makespan == second.best.makespan
        ),
    }
    require(all(checks.values()), f"best-so-far safety audit failed: {checks}")
    return {
        "status": "PASS",
        "instance": instance.instance_id,
        "seed": 1,
        "iterations": len(events),
        "accepted_worse_current_events": len(accepted_worse),
        "initial_best_makespan": events[0]["best_before"].makespan,
        "returned_best_makespan": first.best.makespan,
        "maximum_accepted_current_makespan": max(
            row["current_after"].makespan for row in accepted_worse
        ),
        "checks": checks,
        "code_sha256": {
            relative: digest(ROOT / relative)
            for relative in (
                "rcias_clgri/search/common.py",
                "rcias_clgri/search/alns.py",
                "rcias_clgri/search/csgni.py",
            )
        },
        "interpretation": (
            "The current solution may deteriorate under simulated annealing, while the "
            "independently stored best remains feasible and monotone. No production solver "
            "change is needed before Phase6N qualification."
        ),
    }


def render_report(audit: dict) -> str:
    immediate = audit["immediate_vs_continuation"]
    overall = immediate["overall"]
    collision = audit["candidate_representation_collision"]
    noise = audit["effective_sample_size_and_label_noise"]
    safety = audit["search_safety"]
    checkpoint = audit["architecture"]
    return f"""# Phase 6N 架构与数据审计

## N0 结论

**`N0_COMPLETE — PROCEED_TO_PREREGISTRATION`**。

Phase 6M 终态 commit `{STARTING_COMMIT}` 与 `MODEL_REVISION_QUALITY` 已核验。R13/R14 未访问，未运行 Gurobi；Phase 6I-MR/6J/6K 的既有保护链重新通过，Phase 6L/6M 的 {audit['starting_boundary']['protected_phase6l_phase6m']['files']} 个文件（{audit['starting_boundary']['protected_phase6l_phase6m']['bytes']:,} bytes）已在 N0 起点逐文件冻结。

审计支持进入一个新的、唯一可晋级的 candidate-conditioned CSG critic 预注册。这里的科学假设比“Phase 6L 没有候选身份”更精确：Phase 6L 已使用真实 destroy target 的 OP membership 做 mean/max/attention pooling；缺口是 target 边界关系、critical/bottleneck 子集和 critical synchronization boundary 没有被显式保留。

## 实际模型路径

冻结基座是 `{checkpoint['initialization_checkpoint']['path']}`（SHA256 `{checkpoint['initialization_checkpoint']['sha256']}`），配置为 hidden=128、{checkpoint['initialization_checkpoint']['model_config']['layers']} 个 `FULL_CSG` relation blocks、4 heads、edge features、FP32。CSG tensorizer包含 20 个 canonical 与 20 个机械 reverse relations。encoder 在每个 state 上运行一次，并返回全部 8 类 node embeddings 与 graph embedding；node embeddings 在 global pooling 前后仍可供候选索引使用。

Phase 6L checkpoint `{checkpoint['phase6l_checkpoint']['path']}` 明确引用同一基座。其 `batch.target_operation_indices`/`target_action_index` 先选出每个候选的 OP nodes，再做 target mean、max 和 graph-query attention；随后拼接 fallback action、差值及 score-free cheap context。当前聚合不显式表示：哪些 typed relations 穿过 destroy-set boundary、边界方向/关系族、candidate∩critical、candidate∩bottleneck、critical synchronization chain boundary。Phase 6N 应补这些量，并保持一次 state encoder、全 bank 批量 pooling。

## Immediate 与 continuation

冻结 R12 CAUR-FIT 共 {overall['candidates']:,} candidates，其中 {overall['continuation_positive']:,} 个 continuation-positive；{overall['continuation_positive_with_negative_realized_immediate']:,} 个同时具有负 realized immediate utility，{overall['continuation_positive_with_realized_immediate_below_minus_0_005']:,} 个低于 `-0.005`。

旧 Phase 6L predicted-immediate hard gate 会阻断 {overall['old_predicted_gate_blocked']:,} 个 candidates（{overall['old_predicted_gate_blocked_fraction']:.2%}），其中 {overall['useful_positive_candidates_blocked']:,} 个实际 continuation-positive。被阻断集合的 positive rate 为 {overall['blocked_positive_rate']:.2%}，平均 continuation advantage 为 {overall['blocked_mean_continuation_advantage']:.6f}。realized immediate 与 continuation 的 Pearson/Spearman 为 {immediate['correlation']['realized_immediate_vs_continuation_pearson']:.6f}/{immediate['correlation']['realized_immediate_vs_continuation_spearman']:.6f}；旧 immediate head prediction 与 continuation 仅为 {immediate['correlation']['predicted_immediate_vs_continuation_pearson']:.6f}/{immediate['correlation']['predicted_immediate_vs_continuation_spearman']:.6f}。

因此旧 immediate head 只保留为诊断，不作为 6N 晋级硬门槛。负的“blocked mean”说明不能把全部 blocked candidates 描述成有益；有效结论是该门槛同时删除了大量有益 continuation actions，且其预测量不适合作为 continuation 的代理。

## Cheap-context collision

碰撞距离固定为六个自然落在 `[0,1]` 的 cheap features 的 RMS，且只在同 state、同 origin family 内找确定性最近邻。阈值为 distance `<= {collision['similar_distance_threshold']:.2f}`；material/strong continuation gap 分别为 `>= {collision['material_advantage_gap']:.2f}`/`>= {collision['strong_advantage_gap']:.2f}`。

{collision['directional_nearest_neighbor_rows']:,} 个可比较 candidates 中，{collision['similar_nearest_neighbors']:,} 个具有 cheap-similar 最近邻；material collisions 为 {collision['material_collisions']:,}，strong collisions 为 {collision['strong_collisions']:,}。在 {collision['continuation_best_candidate_separation']['states_with_same_family_neighbor_for_best']} 个有同-family peer 的 state-best candidates 中，{collision['continuation_best_candidate_separation']['poorly_separated_best_candidates']} 个按该门槛 poorly separated，占全部 288 states 的 {collision['continuation_best_candidate_separation']['poorly_separated_fraction_of_all_states']:.2%}。0.1-width 条件桶中有 {collision['conditional_bucket_variance']['multi_candidate_buckets']:,} 个 multi-candidate buckets，candidate-weighted within-bucket variance 为 {collision['conditional_bucket_variance']['candidate_weighted_population_variance_approximation']:.8f}。

这些是 representation-necessity 的支持证据，不是未来模型必然提升的证明。CSV 保留每个 anchor 的同-family nearest neighbor、target Jaccard、cheap values 与真实 outcome gap，避免只摘录有利案例。

## 有效样本与标签噪声

数据包含 {noise['candidate_rows']:,} candidate means、{noise['raw_crn_rows']:,} raw CRN rows、{noise['states']} states、{noise['instances']} instances；每实例恰好 {noise['states_per_instance']['mean']:.0f} states。三个 structural folds 各有 6 instances、96 states，candidates 为 2,270/2,269/2,270。可用于 instance-grouped qualification 的独立顶层单位是 18 个 instances，不能把 6,809 行当成 6,809 个独立 state examples。

两个 CRN seeds 的 outcome Pearson 为 {noise['two_crn_label_noise']['seed_outcome_pearson']:.6f}，平均/中位 absolute gap 为 {noise['two_crn_label_noise']['mean_absolute_seed_difference']:.6f}/{noise['two_crn_label_noise']['median_absolute_seed_difference']:.6f}，p90/p99 为 {noise['two_crn_label_noise']['p90_absolute_seed_difference']:.6f}/{noise['two_crn_label_noise']['p99_absolute_seed_difference']:.6f}；符号不一致率为 {noise['two_crn_label_noise']['sign_disagreement_fraction']:.2%}。这支持增加等量、按进度分层的 R12 states，并继续按 whole instance/cell cross-fit。

## 搜索安全

生产路径无需修复 best-so-far。`decode_candidate` 每次调用 `check_schedule` 并对 infeasible 结果失败；`solve_csgni` 独立维护 `current` 与 `best`，只在严格 makespan 改善时更新 `best`。确定性 tiny 行为审计运行 {safety['iterations']} iterations，观察到 {safety['accepted_worse_current_events']} 次“接受更差 current”，最大 accepted current makespan {safety['maximum_accepted_current_makespan']:.1f}，而返回 best 保持 {safety['returned_best_makespan']:.1f}；两次相同 seed 的非计时轨迹完全一致，所有 candidate/current/best 均 feasible。

N1 必须在任何新 rollout 或 optimizer step 前冻结唯一 primary family、candidate boundary 索引、controlled fine-tuning、whole-instance folds、数据生成数量、四项 loss、empirical residual calibration 与 direct decision gates。R13/R14 继续锁定。

## 证据

- `outputs/phase6n_candidate_conditioned_csg_v1/audit/architecture_data_audit.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/immediate_vs_continuation.csv`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/candidate_representation_collision.csv`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/protected_phase6l_phase6m_evidence.json`
"""


def main() -> None:
    boundary, _manifest = starting_boundary()
    grouped = pd.read_parquet(GROUPED)
    require(len(grouped) == 6809, "frozen Phase 6L candidate count changed")
    require(grouped.state_id.nunique() == 288, "frozen Phase 6L state count changed")
    require(grouped.instance_id.nunique() == 18, "frozen Phase 6L instance count changed")
    require(grouped.candidate_feasible.all(), "frozen candidate feasibility changed")
    require((grouped.requested_bank_count == 24).all(), "full-bank request count changed")
    require(
        (
            grouped.full_bank_unique_count + grouped.duplicate_bank_count
            == grouped.requested_bank_count
        ).all(),
        "full-bank dedup accounting changed",
    )

    immediate_rows, immediate = immediate_audit(grouped)
    collision_rows, collision = collision_audit(grouped)
    audit = {
        "schema": "phase6n-architecture-data-audit-v1",
        "status": "N0_COMPLETE",
        "decision": "PROCEED_TO_PREREGISTRATION",
        "starting_boundary": boundary,
        "architecture": checkpoint_audit(),
        "immediate_vs_continuation": immediate,
        "candidate_representation_collision": collision,
        "effective_sample_size_and_label_noise": sample_size_and_noise(grouped),
        "search_safety": best_so_far_behavior_audit(),
        "data_integrity": {
            "grouped_path": str(GROUPED.relative_to(ROOT)),
            "grouped_sha256": digest(GROUPED),
            "raw_path": str(RAW.relative_to(ROOT)),
            "raw_sha256": digest(RAW),
            "phase6l_ensemble_path": str(ENSEMBLE.relative_to(ROOT)),
            "phase6l_ensemble_sha256": digest(ENSEMBLE),
            "candidate_rows": len(grouped),
            "states": int(grouped.state_id.nunique()),
            "instances": int(grouped.instance_id.nunique()),
            "full_bank_scope": str(grouped.label_scope.unique().item()),
            "candidate_identity_order_preserved": True,
            "canonical_fallback_exactly_one_per_state": bool(
                grouped.groupby("state_id").is_fallback.sum().eq(1).all()
            ),
        },
        "historical_score_online_forward_calls": 0,
        "new_continuation_rollouts": False,
        "optimizer_steps_started": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "gurobi_run": False,
    }
    require(
        audit["data_integrity"]["canonical_fallback_exactly_one_per_state"],
        "canonical fallback is not unique",
    )
    atomic_csv(AUDIT / "immediate_vs_continuation.csv", immediate_rows)
    atomic_csv(AUDIT / "candidate_representation_collision.csv", collision_rows)
    atomic_json(AUDIT / "architecture_data_audit.json", audit)
    atomic_text(REPORT, render_report(audit))
    print(
        json.dumps(
            {
                "status": audit["status"],
                "decision": audit["decision"],
                "candidate_rows": len(grouped),
                "material_collisions": collision["material_collisions"],
                "accepted_worse_events": audit["search_safety"][
                    "accepted_worse_current_events"
                ],
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
