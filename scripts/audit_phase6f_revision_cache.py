#!/usr/bin/env python3
"""Cross-audit the Phase 6F R06 tensor cache against sealed labels."""

from __future__ import annotations

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


SOURCE_MANIFEST = ROOT / "outputs/phase6f/revision_holdout/sealed_label_shard_manifest.csv"
CACHE_MANIFEST = ROOT / "outputs/phase6f/tensorization/revision_holdout_cache/cache_manifest.csv"
CACHE_ROOT = ROOT / "outputs/phase6f/tensorization/revision_holdout_cache"
OUTPUT = ROOT / "outputs/phase6f/audit/revision_holdout_cache_audit.json"


def main() -> None:
    source = pd.read_csv(SOURCE_MANIFEST)
    source = source[source["status"].eq("COMPLETE")].copy()
    cache = pd.read_csv(CACHE_MANIFEST)
    cache = cache[cache["status"].eq("COMPLETE")].copy()
    merged = source.merge(
        cache,
        left_on="shard_id",
        right_on="instance_id",
        how="outer",
        suffixes=("_source", "_cache"),
        indicator=True,
    )
    cache_paths = [ROOT / Path(value) for value in cache["cache_path"]]

    def digest(path: Path) -> str:
        return file_sha256(path) if path.exists() else "MISSING"

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual_hashes = list(executor.map(digest, cache_paths))
    checks = {
        "exact_81_shard_coverage": (
            len(source) == 81
            and set(source["shard_id"]) == set(cache["instance_id"])
        ),
        "unique_cache_shards": not cache["instance_id"].duplicated().any(),
        "all_manifest_rows_matched": merged["_merge"].eq("both").all(),
        "split_exact": merged["split"].eq(merged["training_split"]).all(),
        "state_counts_exact": merged["state_count_source"].eq(
            merged["state_count_cache"]
        ).all(),
        "action_counts_exact": merged["arm_count"].eq(
            merged["action_count"]
        ).all(),
        "source_hashes_exact": merged["shard_sha256"].eq(
            merged["source_shard_sha256"]
        ).all(),
        "tensor_schema_exact": cache["tensor_schema_hash"].eq(
            CSGTensorizer().tensor_schema_hash
        ).all(),
        "all_cache_files_exist": all(path.exists() for path in cache_paths),
        "cache_sizes_exact": [
            path.stat().st_size if path.exists() else -1 for path in cache_paths
        ] == cache["cache_bytes"].astype(int).tolist(),
        "cache_hashes_exact": actual_hashes == cache["cache_sha256"].tolist(),
        "no_partial_files": not any(CACHE_ROOT.rglob("*.partial")),
        "total_states_8100": int(cache["state_count"].sum()) == 8_100,
        "total_actions_191416": int(cache["action_count"].sum()) == 191_416,
    }
    result = {
        "schema": "phase6f-r06-tensor-cache-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "source_shards": len(source),
        "cache_shards": len(cache),
        "total_states": int(cache["state_count"].sum()),
        "total_actions": int(cache["action_count"].sum()),
        "total_positive_actions": int(cache["positive_count"].sum()),
        "total_cache_bytes": int(cache["cache_bytes"].sum()),
        "tensor_schema_hash": CSGTensorizer().tensor_schema_hash,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
