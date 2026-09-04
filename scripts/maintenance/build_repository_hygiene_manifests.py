#!/usr/bin/env python3
"""Build pre-cleanup classifications, archive records, and local evidence copies."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
STARTING_COMMIT = "fa97fec6cac2a94cd1bdc39cbb8c0001f411fa2c"
PRE_INVENTORY = ROOT / "docs/reports/repository_inventory_pre_phase6j.csv"
CLASSIFICATION = ROOT / "docs/reports/repository_cleanup_classification_pre_phase6j.csv"
ARCHIVE_MANIFEST = ROOT / "docs/history/archive_manifest.csv"
EVIDENCE_MANIFEST = ROOT / "docs/history/phase6i_mr_local_evidence_manifest.csv"
P1_REFERENCE_MANIFEST = ROOT / "paper_experiments/ablation/audit/canonical_reference_manifest.csv"
LOCAL_EVIDENCE_ROOT = ROOT / "artifacts/evidence/phase6i_mr_r11"


EVIDENCE_FILES = {
    "outputs/phase6i_mr/collection/r09/access_ledger.json": "R09 access boundary",
    "outputs/phase6i_mr/collection/r09/collection_integrity.json": "R09 collection integrity",
    "outputs/phase6i_mr/collection/r10/access_ledger.json": "R10 access boundary",
    "outputs/phase6i_mr/collection/r10/collection_integrity.json": "R10 collection integrity",
    "outputs/phase6i_mr/final/final_status.json": "final machine-readable status",
    "outputs/phase6i_mr/frozen/artifact_freeze.json": "selected-artifact freeze",
    "outputs/phase6i_mr/frozen/collection_protocol.json": "collection protocol freeze",
    "outputs/phase6i_mr/frozen/r10_collection_authorization.json": "R10 collection authorization",
    "outputs/phase6i_mr/frozen/r10_translation_authorization.json": "R10 translation authorization",
    "outputs/phase6i_mr/frozen/r10_translation_policy.json": "R10 translation policy",
    "outputs/phase6i_mr/frozen/selected_artifact.json": "selected model artifact",
    "outputs/phase6i_mr/frozen/training_protocol.json": "training protocol freeze",
    "outputs/phase6i_mr/pre_r10/pre_r10_freeze.json": "pre-R10 model-selection freeze",
    "outputs/phase6i_mr/r11_validation/anytime_target_metrics.csv": "R11 anytime target metrics",
    "outputs/phase6i_mr/r11_validation/completion_integrity_audit.json": "R11 completion and replay audit",
    "outputs/phase6i_mr/r11_validation/final_decision.json": "R11 final decision",
    "outputs/phase6i_mr/r11_validation/forced_diagnostics.parquet": "R11 forced-action diagnostics",
    "outputs/phase6i_mr/r11_validation/instance_method_summary.csv": "R11 instance-method summary",
    "outputs/phase6i_mr/r11_validation/launch_record.json": "R11 launch provenance",
    "outputs/phase6i_mr/r11_validation/pause_record_20260903T0930CST.json": "R11 pause provenance",
    "outputs/phase6i_mr/r11_validation/promotion_gate_table.csv": "R11 promotion-gate table",
    "outputs/phase6i_mr/r11_validation/r11_access_ledger.json": "R11 one-access ledger",
    "outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json": "R11 anytime and runtime summary",
    "outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json": "R11 ranking and calibration summary",
    "outputs/phase6i_mr/r11_validation/r11_result_hash_manifest.csv": "R11 raw-result hash manifest",
    "outputs/phase6i_mr/r11_validation/resume_record_20260904T0650CST.json": "R11 resume provenance",
    "outputs/phase6i_mr/r11_validation/runtime_environment.json": "R11 runtime environment",
    "outputs/phase6i_mr/r11_validation/scale_support_gate.csv": "R11 support-aware coverage gate",
    "outputs/phase6i_mr/r11_validation/summary_compatibility_audit.json": "R11 repaired-summary compatibility audit",
    "outputs/phase6i_mr/r11_validation/validation_run_summary.csv": "R11 compact run summary",
    "outputs/phase6i_mr/training_data/training_data_freeze.json": "training-data freeze",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived_sha256(path: str) -> str:
    local_path = ROOT / path
    if local_path.is_file():
        return sha256(local_path)
    payload = subprocess.run(
        ["git", "show", f"{STARTING_COMMIT}:{path}"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(path: str) -> tuple[str, str, str]:
    if path.startswith("paper_experiments/ablation/raw_results/full_reference/runs/"):
        return (
            "ARCHIVE_POINTER_ONLY", "remove_byte_identical_duplicate",
            "The retained Phase 6H Core45 source has the same SHA256.",
        )
    if "/live_logs/" in path:
        return (
            "KEEP_LOCAL_IGNORED", "untrack_preserve_local_path",
            "Per-run Parquet process logs are generated inputs; compact derived summaries remain tracked.",
        )
    if path.endswith("/.progress.lock"):
        return (
            "DELETE_SAFE", "delete_stale_runtime_lock",
            "The associated job is complete or superseded and no active process uses the lock.",
        )
    if "/figures/" in path or path.startswith("paper_experiments/figures/"):
        return (
            "KEEP_TRACKED_ACTIVE", "retain",
            "The current paper export and QA contract explicitly consumes this figure family.",
        )
    if "/raw_results/" in path:
        return (
            "KEEP_TRACKED_PROVENANCE", "retain",
            "No redundant tracked canonical copy was proven; the payload supports current paper provenance.",
        )
    if path.startswith("scripts/") or "/scripts/" in path:
        return (
            "KEEP_TRACKED_PROVENANCE", "retain",
            "The script preserves phase reproduction logic; no safe superseding implementation was proven.",
        )
    if path.startswith("docs/reports/"):
        return (
            "KEEP_TRACKED_PROVENANCE", "retain",
            "The report records a frozen phase decision or scientific evidence boundary.",
        )
    if path in {"notes.txt", "implementation_validation_report.md"}:
        return (
            "ARCHIVE_POINTER_ONLY", "remove_superseded_document",
            "The file identifies docs/reports/phase2_final_validation_report.md as its current replacement.",
        )
    return (
        "KEEP_TRACKED_ACTIVE", "retain",
        "Active project or compact derived artifact; no safe replacement was proven.",
    )


def pre_cleanup_paths() -> list[str]:
    with PRE_INVENTORY.open(encoding="utf-8", newline="") as handle:
        candidates = {
            row["path"] for row in csv.DictReader(handle)
            if row["record_type"] == "tracked_cleanup_candidate"
        }
    tracked = set(
        item for item in subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "-z", STARTING_COMMIT],
            cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.split("\0") if item
    )
    candidates.update(path for path in tracked if path.startswith("scripts/"))
    candidates.update(path for path in tracked if path.startswith("docs/reports/"))
    candidates.update(path for path in tracked if "/" not in path)
    return sorted(candidates)


def build_classification() -> None:
    rows = []
    for path in pre_cleanup_paths():
        classification, action, reason = classify(path)
        rows.append({
            "path": path,
            "classification": classification,
            "planned_action": action,
            "reason": reason,
            "starting_commit": STARTING_COMMIT,
        })
    write_csv(
        CLASSIFICATION, rows,
        ["path", "classification", "planned_action", "reason", "starting_commit"],
    )


def build_archive_manifest() -> None:
    rows: list[dict[str, object]] = []
    audit_path = ROOT / "paper_experiments/ablation/audit/reference_replay_audit.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    for row in audit_rows:
        if row["arm"] != "CSG-NI Full":
            continue
        rows.append({
            "original_path": row["copied_raw_path"],
            "sha256": row["copied_raw_sha256"],
            "classification": "ARCHIVE_POINTER_ONLY",
            "action": "remove_byte_identical_duplicate",
            "canonical_or_retained_path": row["source_path"],
            "last_authoritative_commit": STARTING_COMMIT,
            "reason": "Byte-identical duplicate; source and copy SHA256 match in reference_replay_audit.csv.",
            "recovery": f"git show {STARTING_COMMIT}:{row['copied_raw_path']}",
        })

    for path in pre_cleanup_paths():
        if "/live_logs/" in path:
            rows.append({
                "original_path": path, "sha256": archived_sha256(path),
                "classification": "KEEP_LOCAL_IGNORED",
                "action": "untrack_preserve_local_path",
                "canonical_or_retained_path": path,
                "last_authoritative_commit": STARTING_COMMIT,
                "reason": "Generated Parquet live log retained locally and recoverable from Git history.",
                "recovery": f"git show {STARTING_COMMIT}:{path}",
            })
        elif path.endswith("/.progress.lock"):
            rows.append({
                "original_path": path, "sha256": archived_sha256(path),
                "classification": "DELETE_SAFE", "action": "delete_stale_runtime_lock",
                "canonical_or_retained_path": "",
                "last_authoritative_commit": STARTING_COMMIT,
                "reason": "Zero-byte completed or superseded job lock with no active process.",
                "recovery": f"git show {STARTING_COMMIT}:{path}",
            })
    for path in ("implementation_validation_report.md", "notes.txt"):
        rows.append({
            "original_path": path, "sha256": archived_sha256(path),
            "classification": "ARCHIVE_POINTER_ONLY", "action": "remove_superseded_document",
            "canonical_or_retained_path": "docs/reports/phase2_final_validation_report.md",
            "last_authoritative_commit": STARTING_COMMIT,
            "reason": "Superseded by the retained Phase 2 final validation report.",
            "recovery": f"git show {STARTING_COMMIT}:{path}",
        })
    rows.sort(key=lambda row: str(row["original_path"]))
    write_csv(
        ARCHIVE_MANIFEST, rows,
        ["original_path", "sha256", "classification", "action",
         "canonical_or_retained_path", "last_authoritative_commit", "reason", "recovery"],
    )


def build_p1_reference_manifest() -> None:
    audit_path = ROOT / "paper_experiments/ablation/audit/reference_replay_audit.csv"
    rows: list[dict[str, object]] = []
    with audit_path.open(encoding="utf-8", newline="") as handle:
        for audit in csv.DictReader(handle):
            if audit["arm"] == "CSG-NI Full":
                arm = audit["arm"]
                result_path = audit["source_path"]
                result_hash = audit["source_sha256"]
            else:
                arm = "No NI (ALNS-H1)"
                result_path = audit["copied_raw_path"]
                result_hash = audit["copied_raw_sha256"]
            rows.append({
                "arm": arm, "instance_id": audit["instance_id"],
                "seed": audit["seed"], "canonical_result_path": result_path,
                "canonical_result_sha256": result_hash,
                "storage_policy": "single_tracked_canonical_result",
                "source_audit": "paper_experiments/ablation/audit/reference_replay_audit.csv",
            })
    rows.sort(key=lambda row: (str(row["arm"]), str(row["instance_id"]), int(row["seed"])))
    write_csv(
        P1_REFERENCE_MANIFEST, rows,
        ["arm", "instance_id", "seed", "canonical_result_path",
         "canonical_result_sha256", "storage_policy", "source_audit"],
    )


def archive_phase6i_evidence() -> None:
    rows: list[dict[str, object]] = []
    for source_path, role in sorted(EVIDENCE_FILES.items()):
        source = ROOT / source_path
        if not source.is_file():
            raise FileNotFoundError(f"required Phase 6I-MR evidence is missing: {source_path}")
        destination = LOCAL_EVIDENCE_ROOT / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"local evidence copy hash mismatch: {source_path}")
        rows.append({
            "role": role, "source_path": source_path,
            "local_archive_path": destination.relative_to(ROOT).as_posix(),
            "byte_size": source.stat().st_size, "sha256": source_hash,
            "source_status": "ignored_local_output",
        })
    write_csv(
        EVIDENCE_MANIFEST, rows,
        ["role", "source_path", "local_archive_path", "byte_size", "sha256", "source_status"],
    )


def main() -> int:
    build_classification()
    build_archive_manifest()
    build_p1_reference_manifest()
    archive_phase6i_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
