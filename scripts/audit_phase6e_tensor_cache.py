#!/usr/bin/env python3
"""Cross-audit the complete Phase 6E cache against frozen Phase 6C manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.cache import file_sha256  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs/phase6c/manifests/shard_manifest.csv",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/tensor_cache_audit.json",
    )
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()

    source = pd.read_csv(args.source_manifest)
    source = source[source["status"] == "COMPLETE"].copy()
    cache = pd.read_csv(args.cache_manifest)
    cache = cache[cache["status"] == "COMPLETE"].copy()
    source_ids = set(source["shard_id"])
    cache_ids = set(cache["instance_id"])
    merged = source.merge(
        cache,
        left_on="shard_id",
        right_on="instance_id",
        how="outer",
        suffixes=("_source", "_cache"),
        indicator=True,
    )
    cache_paths = [Path(value) for value in cache["cache_path"]]
    existence = [path.exists() for path in cache_paths]
    sizes = [path.stat().st_size if path.exists() else -1 for path in cache_paths]

    def digest(path: Path) -> str:
        return file_sha256(path) if path.exists() else "MISSING"

    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        actual_hashes = list(executor.map(digest, cache_paths))
    checks = {
        "exact_shard_coverage": source_ids == cache_ids,
        "unique_cache_shards": not cache["instance_id"].duplicated().any(),
        "all_manifest_rows_matched": merged["_merge"].eq("both").all(),
        "split_exact": merged["split"].eq(merged["training_split"]).all(),
        "state_counts_exact": merged["state_count_source"].eq(
            merged["state_count_cache"]
        ).all(),
        "action_counts_exact": merged["arm_count"].eq(merged["action_count"]).all(),
        "source_hashes_exact": merged["shard_sha256"].eq(
            merged["source_shard_sha256"]
        ).all(),
        "tensor_schema_exact": cache["tensor_schema_hash"].eq(
            CSGTensorizer().tensor_schema_hash
        ).all(),
        "all_cache_files_exist": all(existence),
        "cache_sizes_exact": sizes == cache["cache_bytes"].astype(int).tolist(),
        "cache_hashes_exact": actual_hashes == cache["cache_sha256"].tolist(),
        "no_partial_files": not any(args.cache_root.rglob("*.partial")),
        "total_states_100000": int(cache["state_count"].sum()) == 100_000,
    }
    split_summary = (
        cache.groupby("training_split")
        .agg(
            shard_count=("instance_id", "size"),
            state_count=("state_count", "sum"),
            action_count=("action_count", "sum"),
            positive_count=("positive_count", "sum"),
            cache_bytes=("cache_bytes", "sum"),
        )
        .reset_index()
        .to_dict("records")
    )
    result = {
        "schema": "phase6e-tensor-cache-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "source_shards": len(source),
        "cache_shards": len(cache),
        "total_states": int(cache["state_count"].sum()),
        "total_actions": int(cache["action_count"].sum()),
        "total_positive_actions": int(cache["positive_count"].sum()),
        "total_cache_bytes": int(cache["cache_bytes"].sum()),
        "tensor_schema_hash": CSGTensorizer().tensor_schema_hash,
        "split_summary": split_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
