#!/usr/bin/env python3
"""Read-only predecessor, holdout-byte, and frozen DTR audits for Phase 6K."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.search.lghga_learning import load_dtr_bundle, predict_rates
from scripts.run_advanced_baseline_v2 import _validate_models, _verify_implementation

OUT = ROOT / "outputs/phase6k_runtime_v1/audit"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, record):
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to replace audit: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main():
    config = json.loads((ROOT / "configs/phase6j_caur.json").read_text())
    suite = config["instance_suite"]
    manifest = ROOT / suite["manifest"]
    assert digest(manifest) == suite["manifest_sha256"]
    rows = list(csv.DictReader(manifest.open()))
    checks = {row["relative_path"]: digest(ROOT / suite["root"] / row["relative_path"]) == row["sha256"]
              for row in rows}
    assert len(checks) == 54 and all(checks.values())
    # Byte hashes only: no parsing or loading of the unopened instance payloads.
    paths = list((ROOT / "outputs/phase6j_caur").rglob("*"))
    accessed = [str(p.relative_to(ROOT)) for p in paths if p.is_file()
                and any(token in p.name.lower() for token in ("r13", "r14"))]
    assert not accessed, accessed
    locked = config["locked_inputs"]
    assert digest(ROOT / locked["phase6h_policy"]) == locked["phase6h_policy_sha256"]
    assert digest(ROOT / locked["frozen_alns_config"]) == locked["frozen_alns_config_sha256"]
    protected = json.loads((OUT / "protected_evidence.json").read_text())
    assert all(digest(ROOT / path) == item["sha256"] for path, item in protected.items())
    model_root = ROOT / "outputs/baselines/lghga_kb_v2/models"
    implementation = _verify_implementation()
    knowledge, model_manifest = _validate_models(model_root)
    regimes = {}
    for regime, entry in sorted(model_manifest["regimes"].items()):
        folder = model_root / regime
        assert digest(folder / "model_manifest.json") == entry["model_manifest_sha256"]
        bundle = load_dtr_bundle(folder)
        predictions = {str(g): predict_rates(bundle, g, 100) for g in (1, 50, 99, 100, 101, 200, 1000, 1000000)}
        maxima = {name: float(model.tree_.threshold[model.tree_.feature >= 0].max())
                  for name, model in bundle.models.items()}
        assert all(x <= 1.0 for x in maxima.values())
        assert all(predictions[str(g)] == predictions["100"] for g in (101, 200, 1000, 1000000))
        regimes[regime] = {"predictions": predictions, "maximum_tree_split": maxima,
                           "model_hashes": dict(bundle.model_hashes)}
    result = {"status": "PASS", "protected_files_unchanged": len(protected),
              "instance_hashes": checks, "instance_check_scope": "OPAQUE_BYTES_ONLY_NO_HOLDOUT_PARSE",
              "r13_r14_access_paths": accessed, "r13_accessed": False, "r14_accessed": False,
              "phase6h_policy_sha256": locked["phase6h_policy_sha256"],
              "budget": "2 * instance.num_operations seconds; N equals |O|",
              "canonical_method": "LG_HGA-RIACRSP-v2-N4M",
              "canonical_selection_basis": ["docs/reports/lghga_v2_revision.md", "scripts/run_advanced_baseline_v2.py"],
              "implementation_manifest_sha256": implementation,
              "knowledge_manifest_hash": knowledge["knowledge_manifest_hash"], "dtr_regimes": regimes,
              "post_100_rule": "keep generation / 100; no new cap needed: all frozen tree splits <= 1"}
    write_once(OUT / "boundary_and_baseline_audit.json", result)
    print(json.dumps({"status": result["status"], "protected_files": len(protected),
                      "instance_hashes": len(checks), "dtr_regimes": len(regimes)}), flush=True)


if __name__ == "__main__":
    main()
