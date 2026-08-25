#!/usr/bin/env python3
"""Audit repository redundancy and optionally remove only safe cache artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAFE_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache"}
SAFE_FILE_NAMES = {".DS_Store"}
SAFE_SUFFIXES = {".pyc", ".tmp", ".bak", ".log"}
PROTECTED_PREFIXES = (
    ROOT / "FJSP-benchmark-main",
    ROOT / "instances" / "canonical",
    ROOT / "instances" / "tiny",
)


def _inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _safe_artifacts() -> tuple[list[Path], list[Path]]:
    directories: list[Path] = []
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir() and path.name in SAFE_DIRECTORY_NAMES:
            directories.append(path)
        elif path.is_file() and (path.name in SAFE_FILE_NAMES or path.suffix.lower() in SAFE_SUFFIXES):
            files.append(path)
    # Avoid listing descendants of a cache directory twice.
    directories.sort(key=lambda path: len(path.parts))
    minimal = [path for path in directories if not any(parent in path.parents for parent in directories)]
    files = [path for path in files if not any(directory in path.parents for directory in minimal)]
    return minimal, files


def _legacy_schema_files() -> list[str]:
    findings: list[str] = []
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts or "FJSP-benchmark-main" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if '"RAIS-1.0"' in text or '"module_instances"' in text or '"agvs_a"' in text:
            findings.append(path.relative_to(ROOT).as_posix())
    return sorted(findings)


def _duplicate_groups() -> list[list[str]]:
    hashes: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() in {".pyc"}:
            continue
        data = path.read_bytes()
        hashes[(len(data), hashlib.sha256(data).hexdigest())].append(path)
    groups = []
    for paths in hashes.values():
        if len(paths) > 1:
            groups.append(sorted(path.relative_to(ROOT).as_posix() for path in paths))
    return sorted(groups)


def _candidate_redundant_python() -> list[str]:
    candidates: list[str] = []
    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in ROOT.rglob("*.py") if ".git" not in path.parts
    )
    public_cli = {
        "generate_fjsp_reconfigurable.py", "generate_automotive_semantic.py",
        "run_small_validation.py", "stress_random_validation.py", "validate_instances.py",
        "generate_canonical_benchmarks.py", "audit_repo_structure.py",
        "generate_tiny_suite.py", "plot_schedule_gantt.py", "run_tiny_validation.py",
        "profile_graph_builder.py",
        "plot_bc_results.py", "run_bc_validation.py",
        "plot_native_solver_comparison.py", "run_native_tiny_validation.py",
    }
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or "FJSP-benchmark-main" in path.parts or "tests" in path.parts:
            continue
        if path.name == "__init__.py" or path.name in public_cli:
            continue
        module_name = path.stem
        if all_text.count(module_name) <= 1:
            candidates.append(path.relative_to(ROOT).as_posix())
    return sorted(candidates)


def _explicit_redundancy_checks() -> dict[str, list[str]]:
    expected_tiny = {"tiny_01.json", "tiny_02.json", "tiny_03.json"}
    tiny_json = {
        path.name for path in (ROOT / "instances" / "tiny").glob("*.json")
    }
    root_demos = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("*demo*.json")
    )
    obsolete_markers = ("_old.py", "_new.py", "_v2.py", "_final.py", "a_agv", "module_instance")
    obsolete_python = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if ".git" not in path.parts and any(marker in path.name.lower() for marker in obsolete_markers)
    )
    expected_exact_solvers = {"tiny_exact_solver.py", "native_tiny_solvers.py"}
    unexpected_exact_solvers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "rcias_clgri" / "exact").glob("*.py")
        if path.name not in expected_exact_solvers and path.name != "__init__.py"
    )
    graph_builders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "rcias_clgri" / "graph").glob("*builder*.py")
    )
    return {
        "unexpected_tiny_json": sorted(tiny_json - expected_tiny),
        "root_demo_json": root_demos,
        "obsolete_versioned_or_module_python": obsolete_python,
        "duplicate_exact_solver_candidates": unexpected_exact_solvers,
        "duplicate_graph_builder_candidates": graph_builders[1:],
    }


def audit(clean_safe: bool) -> dict[str, object]:
    directories, files = _safe_artifacts()
    removed: list[str] = []
    if clean_safe:
        for path in files:
            if not _inside_root(path):
                raise RuntimeError(f"refusing to clean outside repository: {path}")
            path.unlink(missing_ok=True)
            removed.append(path.relative_to(ROOT).as_posix())
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            if not _inside_root(path) or path.resolve() == ROOT.resolve():
                raise RuntimeError(f"refusing unsafe cache removal: {path}")
            shutil.rmtree(path)
            removed.append(path.relative_to(ROOT).as_posix())
        directories, files = _safe_artifacts()
    legacy = _legacy_schema_files()
    duplicates = _duplicate_groups()
    redundant_python = _candidate_redundant_python()
    explicit = _explicit_redundancy_checks()
    result = {
        "safe_cache_directories": [path.relative_to(ROOT).as_posix() for path in directories],
        "safe_temporary_files": [path.relative_to(ROOT).as_posix() for path in files],
        "removed_safe_artifacts": removed,
        "legacy_schema_files": legacy,
        "duplicate_content_groups": duplicates,
        "candidate_redundant_files": redundant_python,
        **explicit,
        "repository_clean": not any((
            directories, files, legacy, duplicates, redundant_python, *explicit.values()
        )),
        "protected_roots": [path.relative_to(ROOT).as_posix() for path in PROTECTED_PREFIXES],
    }
    output = ROOT / "outputs" / "audit" / "repo_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-safe", action="store_true",
        help="explicit compatibility flag; safe cleanup is the default",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="report cache/temp artifacts without deleting them",
    )
    args = parser.parse_args()
    result = audit(not args.report_only)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
