#!/usr/bin/env python3
"""Write deterministic repository-size inventories for hygiene audits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY_CANDIDATES = {
    "raw_results", "log", "logs", "output", "outputs", "checkpoint",
    "checkpoints", "cache", "tmp", "figure", "figures",
}
SUFFIX_CANDIDATES = {
    ".lock", ".pid", ".log", ".tmp", ".bak", ".parquet", ".pt",
    ".pth", ".tiff", ".png", ".pdf", ".eps", ".svg",
}


def git(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        input=input_text, capture_output=True,
    ).stdout


def nul_paths(*args: str) -> list[str]:
    return [item for item in git(*args, "-z").split("\0") if item]


def size(path: str) -> int:
    try:
        return (ROOT / path).stat().st_size
    except FileNotFoundError:
        return 0


def top_level(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "(root)"


def extension(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return suffix or "(none)"


def working_tree_paths() -> list[str]:
    paths: list[str] = []
    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        directories[:] = sorted(item for item in directories if item != ".git")
        for filename in sorted(files):
            path = current_path / filename
            try:
                if path.is_file():
                    paths.append(path.relative_to(ROOT).as_posix())
            except FileNotFoundError:
                continue
    return paths


def aggregate_rows(scope: str, paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    top_counts: Counter[str] = Counter()
    top_bytes: defaultdict[str, int] = defaultdict(int)
    extension_counts: Counter[str] = Counter()
    extension_bytes: defaultdict[str, int] = defaultdict(int)
    for path in paths:
        byte_size = size(path)
        top = top_level(path)
        ext = extension(path)
        top_counts[top] += 1
        top_bytes[top] += byte_size
        extension_counts[ext] += 1
        extension_bytes[ext] += byte_size
    for key in sorted(top_counts):
        rows.append({
            "record_type": "top_level_summary", "scope": scope, "key": key,
            "path": "", "extension": "", "file_count": top_counts[key],
            "byte_size": top_bytes[key], "detail": "",
        })
    for key in sorted(extension_counts):
        rows.append({
            "record_type": "extension_summary", "scope": scope, "key": key,
            "path": "", "extension": key, "file_count": extension_counts[key],
            "byte_size": extension_bytes[key], "detail": "",
        })
    rows.append({
        "record_type": "scope_total", "scope": scope, "key": "TOTAL",
        "path": "", "extension": "", "file_count": len(paths),
        "byte_size": sum(size(path) for path in paths), "detail": "",
    })
    return rows


def candidate_rows(tracked: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in tracked:
        parts = set(Path(path).parts[:-1])
        suffix = Path(path).suffix.lower()
        reasons: list[str] = []
        if parts & DIRECTORY_CANDIDATES:
            reasons.append("candidate_directory")
        if suffix in SUFFIX_CANDIDATES or path.endswith(".progress.lock"):
            reasons.append("candidate_suffix")
        if reasons:
            rows.append({
                "record_type": "tracked_cleanup_candidate", "scope": "tracked",
                "key": "+".join(reasons), "path": path, "extension": extension(path),
                "file_count": 1, "byte_size": size(path), "detail": "",
            })
    return rows


def object_store_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw = git("count-objects", "-v")
    values = dict(line.split(": ", 1) for line in raw.splitlines() if ": " in line)
    for key in sorted(values):
        value = values[key]
        byte_size = int(value) * 1024 if key.startswith("size") else ""
        rows.append({
            "record_type": "git_count_objects", "scope": "git_object_store",
            "key": key, "path": "", "extension": "", "file_count": "",
            "byte_size": byte_size, "detail": value,
        })
    disk_bytes = 0
    for current, _, files in os.walk(ROOT / ".git" / "objects"):
        for filename in files:
            try:
                disk_bytes += (Path(current) / filename).stat().st_size
            except FileNotFoundError:
                continue
    rows.append({
        "record_type": "git_object_store_disk", "scope": "git_object_store",
        "key": "objects_disk_bytes", "path": ".git/objects", "extension": "",
        "file_count": "", "byte_size": disk_bytes,
        "detail": git("count-objects", "-vH").strip().replace("\n", "; "),
    })
    return rows


def history_blobs() -> list[tuple[int, str, str]]:
    objects = []
    oid_paths: dict[str, str] = {}
    for line in git("rev-list", "--objects", "--all").splitlines():
        oid, _, path = line.partition(" ")
        objects.append(oid)
        if path and oid not in oid_paths:
            oid_paths[oid] = path
    metadata = git(
        "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_text="\n".join(objects) + "\n",
    )
    blobs: list[tuple[int, str, str]] = []
    for line in metadata.splitlines():
        oid, object_type, object_size = line.split()
        if object_type == "blob":
            blobs.append((int(object_size), oid_paths.get(oid, "(path unavailable)"), oid))
    return sorted(blobs, reverse=True)[:50]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--largest", required=True, type=Path)
    args = parser.parse_args()

    tracked = sorted(nul_paths("ls-files"))
    ignored = sorted(nul_paths("ls-files", "--others", "--ignored", "--exclude-standard"))
    untracked = sorted(nul_paths("ls-files", "--others", "--exclude-standard"))
    working = sorted(working_tree_paths())

    rows: list[dict[str, object]] = []
    for scope, paths in (
        ("tracked", tracked), ("working_tree", working),
        ("local_ignored", ignored), ("local_untracked", untracked),
    ):
        rows.extend(aggregate_rows(scope, paths))
    rows.extend(candidate_rows(tracked))
    rows.extend(object_store_rows())
    write_csv(
        ROOT / args.inventory, rows,
        ["record_type", "scope", "key", "path", "extension", "file_count", "byte_size", "detail"],
    )

    largest: list[dict[str, object]] = []
    for scope, paths in (("tracked", tracked), ("local_ignored", ignored), ("local_untracked", untracked)):
        for rank, path in enumerate(sorted(paths, key=lambda item: (size(item), item), reverse=True)[:50], 1):
            largest.append({"scope": scope, "rank": rank, "path": path, "byte_size": size(path), "oid": ""})
    for rank, (byte_size, path, oid) in enumerate(history_blobs(), 1):
        largest.append({"scope": "reachable_git_history", "rank": rank, "path": path, "byte_size": byte_size, "oid": oid})
    write_csv(ROOT / args.largest, largest, ["scope", "rank", "path", "byte_size", "oid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
