#!/usr/bin/env python3
"""Freeze P1 manifests and replay-audit the two reused ablation arms."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.env.insertion_decoder import Action  # noqa: E402
from rcias_clgri.env.rcias_env import RCIASConstructionEnv  # noqa: E402


PAPER_ROOT = ROOT / "paper_experiments"
ABLATION_ROOT = PAPER_ROOT / "ablation"
CORE_ROOT = ROOT / "instances/controlled/RCIAS-CB1"
CORE_MANIFEST = CORE_ROOT / "manifests/core_manifest.csv"
CORE_CHECKSUMS = CORE_ROOT / "manifests/checksums.sha256"
REFERENCE_CONFIG = PAPER_ROOT / "configs/main_core/phase6h_provisional.json"
MANIFEST_PATH = ABLATION_ROOT / "ablation_instance_manifest.csv"
CONFIG_PATH = ABLATION_ROOT / "configs/p1_ablation_protocol.json"
IMPLEMENTATION_PATH = ABLATION_ROOT / "configs/p1_implementation_manifest.json"
AUDIT_PATH = ABLATION_ROOT / "audit/reference_replay_audit.csv"
INVENTORY_PATH = ABLATION_ROOT / "audit/reference_source_inventory.json"
PROVENANCE_PATH = ABLATION_ROOT / "audit/environment_provenance.json"
SEEDS = (530101, 530102, 530103, 530104, 530105)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    if digest(source) != digest(destination):
        raise RuntimeError(f"copied raw result hash mismatch: {destination}")


def replay(instance_path: Path, payload: dict) -> dict[str, object]:
    instance = load_instance(instance_path)
    environment = RCIASConstructionEnv(instance)
    for raw in payload["best_actions"]:
        environment.step(Action(**raw))
    audit = check_schedule(instance, environment.schedule)
    replay_makespan = environment.objective().makespan
    stored_makespan = float(payload["best_makespan"])
    if not environment.done or not audit["feasible"] or replay_makespan != stored_makespan:
        raise RuntimeError({
            "instance": instance.instance_id,
            "done": environment.done,
            "audit": audit,
            "replay_makespan": replay_makespan,
            "stored_makespan": stored_makespan,
        })
    return {
        "replay_feasible": True,
        "replay_makespan": replay_makespan,
        "replayed_action_count": len(payload["best_actions"]),
        "violation_count": 0,
    }


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> int:
    reference = read_json(REFERENCE_CONFIG)
    if reference.get("status") != "FROZEN_BEFORE_PROVISIONAL_PHASE6H_CORE":
        raise RuntimeError("Full CSG-NI reference is not frozen")
    if tuple(reference["seeds"]) != SEEDS:
        raise RuntimeError("P1 seeds differ from the frozen manuscript prefix")

    checksum_lookup = {}
    for line in CORE_CHECKSUMS.read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        checksum_lookup[relative_path] = checksum
    with CORE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["replicate"] in {"R01", "R02"}]
    selected.sort(key=lambda row: row["instance_id"])
    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in selected:
        cells.setdefault((row["scale"], row["CF_level"]), []).append(row)
    expected_cells = {(scale, cf) for scale in ("S", "M", "L") for cf in ("CF1", "CF2", "CF3")}
    if set(cells) != expected_cells or any(len(group) != 2 for group in cells.values()):
        raise RuntimeError("deterministic P1 selection did not yield two instances per cell")

    manifest_rows = []
    for row in selected:
        relative_path = row["relative_path"]
        instance_path = CORE_ROOT / relative_path
        expected_hash = checksum_lookup.get(relative_path)
        if expected_hash is None or digest(instance_path) != expected_hash:
            raise RuntimeError(f"Core instance checksum mismatch: {row['instance_id']}")
        manifest_rows.append({
            "instance_id": row["instance_id"],
            "scale": row["scale"],
            "CF_level": row["CF_level"],
            "replicate": row["replicate"],
            "selection_rule": "lexicographically_first_two_canonical_ids_per_scale_cf",
            "number_of_operations": int(row["number_of_operations"]),
            "instance_path": str(instance_path.relative_to(ROOT)),
            "instance_sha256": expected_hash,
        })
    atomic_csv(MANIFEST_PATH, manifest_rows)

    audit_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    arms = (
        (
            "CSG-NI Full",
            PAPER_ROOT / "raw_results/core45/CSG_NI_PROVISIONAL_PHASE6H/runs",
            ABLATION_ROOT / "raw_results/full_reference/runs",
        ),
        (
            "No NI intervention (ALNS-H1 equivalence)",
            ROOT / "outputs/phase5c/search/cb1_core/formal/alns_h1",
            ABLATION_ROOT / "raw_results/no_ni_alns_h1_equivalence/runs",
        ),
    )
    for arm, source_root, destination_root in arms:
        for row in manifest_rows:
            instance_path = ROOT / str(row["instance_path"])
            for seed in SEEDS:
                source = source_root / str(row["instance_id"]) / f"seed_{seed}.json"
                destination = destination_root / str(row["instance_id"]) / f"seed_{seed}.json"
                if not source.is_file():
                    raise RuntimeError(f"missing reused P1 source result: {source}")
                payload = read_json(source)
                expected_limit = 2.0 * int(row["number_of_operations"])
                if (
                    payload.get("instance_id") != row["instance_id"]
                    or int(payload.get("seed")) != seed
                    or float(payload.get("time_limit_seconds")) != expected_limit
                    or payload.get("feasible") is not True
                ):
                    raise RuntimeError(f"reused result metadata mismatch: {source}")
                replay_result = replay(instance_path, payload)
                atomic_copy(source, destination)
                audit_rows.append({
                    "arm": arm,
                    "instance_id": row["instance_id"],
                    "scale": row["scale"],
                    "CF_level": row["CF_level"],
                    "seed": seed,
                    "time_limit_seconds": expected_limit,
                    "stored_makespan": float(payload["best_makespan"]),
                    **replay_result,
                    "source_path": str(source.relative_to(ROOT)),
                    "source_sha256": digest(source),
                    "copied_raw_path": str(destination.relative_to(ROOT)),
                    "copied_raw_sha256": digest(destination),
                })
                inventory_rows.append({
                    "arm": arm,
                    "instance_id": row["instance_id"],
                    "seed": seed,
                    "source_path": str(source.relative_to(ROOT)),
                    "source_sha256": digest(source),
                    "copied_raw_path": str(destination.relative_to(ROOT)),
                    "copied_raw_sha256": digest(destination),
                })
    if len(audit_rows) != 180 or not all(row["replay_feasible"] for row in audit_rows):
        raise RuntimeError("P1 reused-arm replay gate failed")
    atomic_csv(AUDIT_PATH, audit_rows)
    atomic_json(INVENTORY_PATH, {
        "schema": "initial-manuscript-p1-reference-inventory-v1",
        "status": "PASS_REPLAY_VALIDATED",
        "copied_raw_files": len(inventory_rows),
        "records": inventory_rows,
    })

    implementation_files = (
        "paper_experiments/ablation/csgni_random_policy.py",
        "paper_experiments/ablation/run_p1_ablation.py",
        "paper_experiments/ablation/prepare_p1_ablation.py",
        "rcias_clgri/ni/live_inference.py",
        "rcias_clgri/ni/proposal_bank.py",
        "rcias_clgri/search/phase6c.py",
        "rcias_clgri/search/csgni.py",
        "rcias_clgri/search/alns.py",
        "rcias_clgri/env/insertion_decoder.py",
        "rcias_clgri/env/feasibility.py",
    )
    implementation = {
        "schema": "initial-manuscript-p1-implementation-manifest-v1",
        "status": "FROZEN_BEFORE_P1_EXECUTION",
        "files": [
            {"path": path, "sha256": digest(ROOT / path)} for path in implementation_files
        ],
    }
    atomic_json(IMPLEMENTATION_PATH, implementation)

    config = {
        "schema": "initial-manuscript-p1-ablation-protocol-v1",
        "status": "FROZEN_BEFORE_P1_EXECUTION",
        "starting_git_commit": current_commit(),
        "suite": "RCIAS-CB1 Core P1 deterministic subset",
        "arms": [
            "CSG-NI Full",
            "Uniform full-bank selection at frozen gate",
            "No NI intervention (ALNS-H1 equivalence)",
        ],
        "new_experiment_arm": "CSG_NI_UNIFORM_FULL_BANK_FROZEN_GATE",
        "selection_rule": "lexicographically first two canonical IDs per Scale x CF cell",
        "ablation_instance_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "ablation_instance_manifest_sha256": digest(MANIFEST_PATH),
        "instance_count": 18,
        "seeds": list(SEEDS),
        "matched_seeds_per_instance_arm": 5,
        "expected_total_arm_records": 270,
        "new_arm_expected_runs": 90,
        "wall_clock_seconds_per_operation": 2.0,
        "solver_concurrency": 2,
        "task_partition": "global_task_index_modulo_shard_count",
        "intervention_rate": int(reference["intervention_rate"]),
        "uniform_selection_seed_namespace": 671201,
        "production_candidate_rule_count": 24,
        "reference_core_config": str(REFERENCE_CONFIG.relative_to(ROOT)),
        "reference_core_config_sha256": digest(REFERENCE_CONFIG),
        "phase6h_policy": reference["phase6h_policy"],
        "phase6h_policy_sha256": reference["phase6h_policy_sha256"],
        "phase6f_experiment_freeze": reference["phase6f_experiment_freeze"],
        "phase6f_experiment_freeze_sha256": reference["phase6f_experiment_freeze_sha256"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "phase6h_config": reference["phase6h_config"],
        "phase6h_config_sha256": reference["phase6h_config_sha256"],
        "alns_config": reference["alns_config"],
        "alns_config_sha256": reference["alns_config_sha256"],
        "implementation_manifest": str(IMPLEMENTATION_PATH.relative_to(ROOT)),
        "implementation_manifest_sha256": digest(IMPLEMENTATION_PATH),
        "reused_reference_audit": str(AUDIT_PATH.relative_to(ROOT)),
        "reused_reference_audit_sha256": digest(AUDIT_PATH),
        "new_arm_budget_seconds_sum": 2.0 * sum(
            int(row["number_of_operations"]) for row in manifest_rows
        ) * len(SEEDS),
        "fairness_boundary": (
            "Every run uses the same instance, seed, H1 initialization, frozen ALNS parameters, "
            "decoder, feasibility logic, and 2|O| ceiling. Full and ALNS-H1 are reused exact raw "
            "results; only uniform target selection is newly executed."
        ),
    }
    atomic_json(CONFIG_PATH, config)

    packages = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "torch", "pyarrow", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    provenance = {
        "schema": "initial-manuscript-p0-p1-p3-provenance-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "core_artifact_hashes": {
            "main_runs.csv": digest(PAPER_ROOT / "processed_data/main/main_runs.csv"),
            "main_scale_summary.csv": digest(PAPER_ROOT / "processed_data/main/main_scale_summary.csv"),
            "statistical_analysis.json": digest(PAPER_ROOT / "processed_data/main/statistical_analysis.json"),
            "phase6h_policy.json": reference["phase6h_policy_sha256"],
            "checkpoint": reference["checkpoint_sha256"],
        },
        "gpu_note": "GPU availability is verified separately at launch because the workspace sandbox does not expose CUDA devices.",
    }
    atomic_json(PROVENANCE_PATH, provenance)
    print(json.dumps({
        "status": "PASS_PREPARED",
        "instances": len(manifest_rows),
        "reused_runs_replay_validated": len(audit_rows),
        "new_runs": config["new_arm_expected_runs"],
        "new_arm_budget_seconds_sum": config["new_arm_budget_seconds_sum"],
        "config": str(CONFIG_PATH.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
