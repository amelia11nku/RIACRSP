#!/usr/bin/env python3
"""Freeze Phase 6N N1 before new rollout or optimizer execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
N0_COMMIT = "f38725a"
CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
REPORT = ROOT / "docs/reports/phase6n_candidate_conditioned_csg_preregistered_protocol.md"
AUDIT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/audit"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text() == text, f"refusing to replace frozen file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def verify_phase6l_phase6m_manifest() -> dict:
    path = AUDIT / "protected_phase6l_phase6m_evidence.json"
    manifest = json.loads(path.read_text())
    total = 0
    for relative, expected in manifest.items():
        target = ROOT / relative
        require(target.is_file(), f"protected predecessor missing: {relative}")
        require(target.stat().st_size == expected["bytes"], f"size changed: {relative}")
        require(digest(target) == expected["sha256"], f"hash changed: {relative}")
        total += int(expected["bytes"])
    return {
        "manifest_path": str(path.relative_to(ROOT)),
        "manifest_sha256": digest(path),
        "files": len(manifest),
        "bytes": total,
        "status": "PASS",
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    require(
        config["status"] == "PREREGISTERED_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP",
        "invalid Phase 6N preregistration status",
    )
    require(config["primary_family"] == "N1_CANDIDATE_CONDITIONED_CSG", "wrong primary")
    require(config["promotable_families"] == [config["primary_family"]], "not one primary")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", N0_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "N0 commit is not an ancestor of HEAD",
    )
    n0 = json.loads((AUDIT / "architecture_data_audit.json").read_text())
    require(n0["status"] == "N0_COMPLETE", "N0 audit is incomplete")
    require(n0["decision"] == "PROCEED_TO_PREREGISTRATION", "N0 does not authorize N1")
    require(n0["r13_accessed"] is False and n0["r14_accessed"] is False, "holdout access")
    forbidden = [
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/implementation",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/quality",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/runtime",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/solver",
    ]
    require(not any(path.exists() for path in forbidden), "post-N1 output exists before freeze")
    ledgers = [
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json",
    ]
    require(not any(path.exists() for path in ledgers), "R13/R14 access detected")
    protected = verify_phase6l_phase6m_manifest()

    inputs = [
        CONFIG,
        REPORT,
        Path(__file__),
        AUDIT / "architecture_data_audit.json",
        AUDIT / "immediate_vs_continuation.csv",
        AUDIT / "candidate_representation_collision.csv",
        AUDIT / "protected_phase6l_phase6m_evidence.json",
        ROOT / config["locked_inputs"]["instance_manifest"],
        ROOT / config["locked_inputs"]["phase6j_config"],
        ROOT / config["locked_inputs"]["phase6j_horizon_freeze"],
        ROOT / config["locked_inputs"]["phase6f_base_checkpoint"],
        ROOT / config["locked_inputs"]["phase6l_grouped_labels"],
        ROOT / config["locked_inputs"]["phase6l_raw_labels"],
        ROOT / config["locked_inputs"]["phase6l_ensemble_oof"],
        ROOT / config["locked_inputs"]["phase6l_training_protocol"],
        ROOT / "configs/csg_v1_schema.json",
        ROOT / "rcias_clgri/csg/builder.py",
        ROOT / "rcias_clgri/ni/tensorize.py",
        ROOT / "rcias_clgri/ni/batching.py",
        ROOT / "rcias_clgri/ni/encoder.py",
        ROOT / "rcias_clgri/search/alns.py",
        ROOT / "rcias_clgri/search/common.py",
        ROOT / "rcias_clgri/search/phase6c.py",
    ]
    require(all(path.is_file() for path in inputs), "a locked Phase 6N input is missing")

    data = config["data_generation"]
    require(data["instances"] == 18, "data instance count changed")
    require(data["new_states"] == 576 and data["total_states"] == 864, "state plan changed")
    require(data["total_states_per_instance"] == 48, "unequal data plan")
    require(data["continuation_horizon"] == 4, "continuation horizon changed")
    require(len(data["continuation_crn_seeds"]) == 2, "CRN count changed")
    require(data["estimate"]["projected_new_process_seconds"] > 300, "long-run plan missing")

    data_plan = {
        "schema": "phase6n-data-generation-plan-v1",
        "status": "FROZEN_BEFORE_OUTCOME_GENERATION",
        "split": data["split"],
        "instances": data["instances"],
        "source_sampler": data["new_source_sampler"],
        "source_budget_seconds_per_instance": data["source_budget_seconds_per_instance"],
        "new_trajectory_seeds": data["new_trajectory_seeds"],
        "states_per_new_trajectory": data["states_per_new_trajectory"],
        "progress_anchors": data["progress_anchors"],
        "new_source_trajectories": data["new_source_trajectories"],
        "new_states": data["new_states"],
        "combined_states": data["total_states"],
        "states_per_instance": data["total_states_per_instance"],
        "candidate_scope": config["boundaries"]["candidate_scope"],
        "candidate_rules": config["boundaries"]["candidate_generator_rules"],
        "candidate_trials_per_target": data["candidate_trials_per_target"],
        "fallback": data["canonical_fallback"],
        "continuation_horizon": data["continuation_horizon"],
        "continuation_crn_seeds": data["continuation_crn_seeds"],
        "seed_namespaces": {
            "proposal": data["proposal_seed_namespace"],
            "repair": data["repair_seed_namespace"],
            "continuation": data["continuation_seed_namespace"],
        },
        "outcome_independent_stop_rule": data["outcome_independent_stop_rule"],
        "estimate": data["estimate"],
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    data_plan_path = OUT / "data_generation_plan.json"
    atomic_json(data_plan_path, data_plan)
    source_hashes = {
        str(path.relative_to(ROOT)): digest(path) for path in inputs
    }
    source_path = OUT / "source_hashes.json"
    atomic_json(
        source_path,
        {
            "schema": "phase6n-preregistration-source-hashes-v1",
            "files": source_hashes,
        },
    )
    payload = {
        "schema": "phase6n-candidate-conditioned-csg-preregistration-v1",
        "status": "FROZEN_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP",
        "primary_family": config["primary_family"],
        "promotable_families": config["promotable_families"],
        "starting_commit": config["starting_commit"],
        "n0_commit": N0_COMMIT,
        "head_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes_path": str(source_path.relative_to(ROOT)),
        "source_hashes_sha256": digest(source_path),
        "data_generation_plan_path": str(data_plan_path.relative_to(ROOT)),
        "data_generation_plan_sha256": digest(data_plan_path),
        "protected_predecessor": protected,
        "new_rollouts_started": False,
        "optimizer_steps_started": False,
        "outer_oof_generated": False,
        "historical_score_primary_input_calls": 0,
        "historical_score_deployable_runtime_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    preregistration_path = OUT / "preregistration.json"
    atomic_json(preregistration_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_family": payload["primary_family"],
                "source_files": len(source_hashes),
                "new_states": data_plan["new_states"],
                "combined_states": data_plan["combined_states"],
                "projected_hours": data_plan["estimate"]["projected_new_hours"],
                "preregistration_sha256": digest(preregistration_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
