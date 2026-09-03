#!/usr/bin/env python3
"""Generate and verify the isolated Phase 6I-MR R09/R10/R11 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.benchmark_structure import benchmark_metrics  # noqa: E402
from rcias_clgri.data.generation import write_json  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json  # noqa: E402
from rcias_clgri.instances.controlled_generator import (  # noqa: E402
    acceptance_failures,
    configuration_entropy,
    generate_candidate,
    scale_sensitivity_variant,
)


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
GENERATION_SPEC_PATH = ROOT / "configs/rcias_cb1_generation.json"
TARGET = ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11"
MANIFESTS = TARGET / "manifests"
SPLITS = {
    9: ("LIVE_REV_FIT", "r09_live_rev_fit"),
    10: ("LIVE_REV_SELECT", "r10_live_rev_select"),
    11: ("LIVE_REV_HOLDOUT", "r11_live_rev_holdout"),
}
SCALES = ("S", "M", "L")
CFS = ("CF1", "CF2", "CF3")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_base(
    instance_id: str,
    scale: str,
    cf: str,
    base_seed: int,
    spec: dict,
) -> tuple[dict, int, list[dict]]:
    history = []
    for attempt in range(1, int(spec["max_attempts"]) + 1):
        final_seed = base_seed * 1000 + attempt
        raw = generate_candidate(
            instance_id, "TRAIN_LIVE_REVISION_BASE", scale, cf, final_seed, spec
        )
        failures = acceptance_failures(raw, scale, cf, spec)
        history.append({
            "attempt": attempt,
            "final_seed": final_seed,
            "failures": failures,
        })
        if not failures:
            return raw, final_seed, history
    raise RuntimeError(f"{instance_id} failed structural acceptance: {history[-1]}")


def historical_inventory() -> tuple[set[str], set[str], set[int]]:
    hashes: set[str] = set()
    ids: set[str] = set()
    seeds: set[int] = set()
    for path in (ROOT / "instances").rglob("*.json"):
        if TARGET in path.parents or "manifests" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
        if meta.get("instance_id") is not None:
            hashes.add(digest(path))
            ids.add(str(meta["instance_id"]))
        if meta.get("seed") is not None:
            seeds.add(int(meta["seed"]))
    for path in (ROOT / "instances").rglob("*.csv"):
        if TARGET in path.parents:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "instance_id" in frame:
            ids.update(frame["instance_id"].dropna().astype(str))
        for column in (
            "base_seed",
            "final_seed",
            "base_generation_seed",
            "final_generation_seed",
            "trajectory_seed",
            "state_sampling_seed",
        ):
            if column in frame:
                seeds.update(frame[column].dropna().astype(int))
    return hashes, ids, seeds


def generate_cell_replicate(
    config: dict,
    spec: dict,
    cell_replicate: str,
) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    if cell_replicate not in {"C01", "C02"}:
        raise ValueError(f"unknown cell replicate: {cell_replicate}")
    base_namespace = int(config["instance_suite"]["base_seed_namespace"])
    offset = (
        0
        if cell_replicate == "C01"
        else int(config["instance_suite"]["second_cell_replicate_seed_offset"])
    )
    suffix = "" if cell_replicate == "C01" else "_C02"
    rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}
    for replicate, (split, folder) in SPLITS.items():
        for scale_index, scale in enumerate(SCALES, 1):
            for cf_index, cf in enumerate(CFS, 1):
                base_id = (
                    f"CB1_LIVE_REV_BASE_{scale}_{cf}_R{replicate:02d}{suffix}"
                )
                base_seed = (
                    base_namespace
                    + offset
                    + scale_index * 100
                    + cf_index * 10
                    + replicate
                )
                base, final_seed, history = accepted_base(
                    base_id, scale, cf, base_seed, spec
                )
                histories[base_id] = history
                instance_id = (
                    f"CB1_LIVE_REV_{scale}_{cf}_RI2_TI2_R{replicate:02d}{suffix}"
                )
                raw = scale_sensitivity_variant(base, instance_id, "RI2", "TI2", spec)
                raw["meta"].update({
                    "suite": "TRAIN_LIVE_REVISION_ONLY",
                    "live_revision_split": split,
                    "base_structure": base_id,
                    "cell_replicate": cell_replicate,
                })
                path = TARGET / folder / f"{instance_id}.json"
                write_json(raw, path)
                row = {
                    "instance_id": instance_id,
                    "live_revision_split": split,
                    "scale": scale,
                    "CF_level": cf,
                    "RI_level": "RI2",
                    "TI_level": "TI2",
                    "replicate": f"R{replicate:02d}",
                    "cell_replicate": cell_replicate,
                    "base_structure": base_id,
                    "base_generation_seed": base_seed,
                    "final_generation_seed": final_seed,
                    "relative_path": str(path.relative_to(TARGET)),
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
                rows.append(row)
                if split != "LIVE_REV_HOLDOUT":
                    instance = load_instance(path)
                    metric_rows.append({
                        **{key: row[key] for key in (
                            "instance_id",
                            "live_revision_split",
                            "scale",
                            "CF_level",
                            "RI_level",
                            "TI_level",
                            "replicate",
                            "cell_replicate",
                            "base_structure",
                        )},
                        **benchmark_metrics(instance),
                        "configuration_entropy": configuration_entropy(raw),
                    })

    return rows, metric_rows, histories


def write_generation_spec(config: dict) -> None:
    atomic_write_json({
        "schema": "rcias-cb1-train-live-r09r11-generation-v1.2",
        "source_spec_sha256": digest(GENERATION_SPEC_PATH),
        "phase6i_config_sha256": digest(CONFIG_PATH),
        "base_seed_namespace": config["instance_suite"]["base_seed_namespace"],
        "second_cell_replicate_seed_offset": config["instance_suite"][
            "second_cell_replicate_seed_offset"
        ],
        "factorial_design": {
            "scale": list(SCALES),
            "CF": list(CFS),
            "RI": ["RI2"],
            "TI": ["TI2"],
            "cell_replicate": ["C01", "C02"],
            "replicate_split": {
                f"R{key:02d}": value[0] for key, value in SPLITS.items()
            },
        },
        "r11_generation_boundary": (
            "generated deterministically and hashed here; downstream content access "
            "is forbidden until the selected Phase 6I-MR artifact is frozen"
        ),
        "r11_pre_freeze_metrics": "WITHHELD",
    }, MANIFESTS / "generation_spec.json")


def write_suite(
    config: dict,
    rows: list[dict],
    metric_rows: list[dict],
    histories: dict[str, list[dict]],
) -> None:
    manifest = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values("instance_id").reset_index(drop=True)
    atomic_write_csv(manifest, MANIFESTS / "phase6i_instance_manifest.csv")
    atomic_write_csv(metrics, MANIFESTS / "r09_r10_structural_metrics.csv")
    atomic_write_json(histories, MANIFESTS / "generation_history.json")
    write_generation_spec(config)


def generate() -> None:
    if TARGET.exists() and any(TARGET.rglob("*.json")):
        raise RuntimeError(
            "Phase 6I-MR suite already exists; use --verify-only instead of overwriting"
        )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(GENERATION_SPEC_PATH.read_text(encoding="utf-8"))
    for _, folder in SPLITS.values():
        (TARGET / folder).mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[dict]] = {}
    for cell_replicate in config["instance_suite"]["cell_replicates"]:
        new_rows, new_metrics, new_histories = generate_cell_replicate(
            config, spec, str(cell_replicate)
        )
        rows.extend(new_rows)
        metric_rows.extend(new_metrics)
        histories.update(new_histories)
    write_suite(config, rows, metric_rows, histories)
    verify()
    print("PHASE6I_MR_INSTANCES_CREATED count=54 r09=18 r10=18 r11=18")


def extend_v12() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(GENERATION_SPEC_PATH.read_text(encoding="utf-8"))
    manifest_path = MANIFESTS / "phase6i_instance_manifest.csv"
    metrics_path = MANIFESTS / "r09_r10_structural_metrics.csv"
    history_path = MANIFESTS / "generation_history.json"
    if not manifest_path.exists() or not metrics_path.exists() or not history_path.exists():
        raise RuntimeError("Revision 1.1 C01 suite is incomplete; refusing extension")
    manifest = pd.read_csv(manifest_path)
    metrics = pd.read_csv(metrics_path)
    if len(manifest) != 27 or any(manifest.instance_id.str.endswith("_C02")):
        raise RuntimeError("--extend-v12 requires exactly the unextended 27-instance suite")
    manifest["cell_replicate"] = "C01"
    metrics["cell_replicate"] = "C01"
    histories = json.loads(history_path.read_text(encoding="utf-8"))
    new_rows, new_metrics, new_histories = generate_cell_replicate(
        config, spec, "C02"
    )
    histories.update(new_histories)
    write_suite(
        config,
        [*manifest.to_dict("records"), *new_rows],
        [*metrics.to_dict("records"), *new_metrics],
        histories,
    )
    verify()
    print("PHASE6I_MR_INSTANCES_EXTENDED_V12 count=54 r09=18 r10=18 r11=18")


def verify() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = MANIFESTS / "phase6i_instance_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("Phase 6I-MR instance manifest is missing")
    manifest = pd.read_csv(manifest_path)
    historical_hashes, historical_ids, historical_seeds = historical_inventory()
    failures: list[str] = []
    parsed_splits: set[str] = set()
    for row in manifest.to_dict("records"):
        path = TARGET / row["relative_path"]
        checksum = digest(path)
        common_failure = (
            checksum != row["sha256"]
            or int(path.stat().st_size) != int(row["size_bytes"])
            or checksum in historical_hashes
            or row["instance_id"] in historical_ids
            or int(row["base_generation_seed"]) in historical_seeds
            or int(row["final_generation_seed"]) in historical_seeds
        )
        if row["live_revision_split"] == "LIVE_REV_HOLDOUT":
            instance_failure = False
        else:
            instance = load_instance(path)
            parsed_splits.add(str(row["live_revision_split"]))
            instance_failure = instance.instance_id != row["instance_id"]
        if common_failure or instance_failure:
            failures.append(row["instance_id"])

    split_counts = manifest.live_revision_split.value_counts().to_dict()
    cell_counts = manifest.groupby(
        ["live_revision_split", "scale", "CF_level"]
    ).size()
    expected_search_seeds = {
        int(seed) for values in config["seeds"].values() for seed in values
    } | {int(value) for value in config["rng_namespaces"].values()}
    generation_seeds = set(manifest.base_generation_seed.astype(int)) | set(
        manifest.final_generation_seed.astype(int)
    )
    split_hashes = {
        split: set(group.sha256)
        for split, group in manifest.groupby("live_revision_split")
    }
    checks = {
        "exactly_54_unique_instances": (
            len(manifest) == 54 and manifest.instance_id.nunique() == 54
        ),
        "exactly_18_per_split": split_counts == {
            "LIVE_REV_FIT": 18,
            "LIVE_REV_SELECT": 18,
            "LIVE_REV_HOLDOUT": 18,
        },
        "two_instances_per_scale_cf_cell": (
            len(cell_counts) == 27 and set(cell_counts) == {2}
        ),
        "cell_replicates_exact": (
            set(manifest.cell_replicate) == {"C01", "C02"}
            and manifest.groupby(
                ["live_revision_split", "scale", "CF_level"]
            ).cell_replicate.apply(set).eq({"C01", "C02"}).all()
        ),
        "fixed_ri2_ti2": (
            set(manifest.RI_level) == {"RI2"} and set(manifest.TI_level) == {"TI2"}
        ),
        "replicate_split_exact": (
            set(manifest.loc[
                manifest.live_revision_split == "LIVE_REV_FIT", "replicate"
            ]) == {"R09"}
            and set(manifest.loc[
                manifest.live_revision_split == "LIVE_REV_SELECT", "replicate"
            ]) == {"R10"}
            and set(manifest.loc[
                manifest.live_revision_split == "LIVE_REV_HOLDOUT", "replicate"
            ]) == {"R11"}
        ),
        "all_hashes_and_sizes_exact": not failures,
        "zero_historical_id_hash_or_generation_seed_overlap": not failures,
        "all_instance_hashes_unique": manifest.sha256.nunique() == 54,
        "split_content_hashes_pairwise_disjoint": (
            not (split_hashes["LIVE_REV_FIT"] & split_hashes["LIVE_REV_SELECT"])
            and not (split_hashes["LIVE_REV_FIT"] & split_hashes["LIVE_REV_HOLDOUT"])
            and not (
                split_hashes["LIVE_REV_SELECT"] & split_hashes["LIVE_REV_HOLDOUT"]
            )
        ),
        "generation_seeds_unique": (
            manifest.base_generation_seed.nunique() == 54
            and manifest.final_generation_seed.nunique() == 54
        ),
        "generation_and_search_seeds_disjoint": not bool(
            generation_seeds & expected_search_seeds
        ),
        "r11_content_not_parsed_during_verify": (
            "LIVE_REV_HOLDOUT" not in parsed_splits
        ),
        "phase6i_config_hash_matches": (
            json.loads((MANIFESTS / "generation_spec.json").read_text(encoding="utf-8"))[
                "phase6i_config_sha256"
            ]
            == digest(CONFIG_PATH)
        ),
        "r09_r10_structural_metrics_complete_and_r11_withheld": (
            len(pd.read_csv(MANIFESTS / "r09_r10_structural_metrics.csv")) == 36
            and set(pd.read_csv(
                MANIFESTS / "r09_r10_structural_metrics.csv"
            ).live_revision_split) == {"LIVE_REV_FIT", "LIVE_REV_SELECT"}
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "failures": failures})
    checksum_lines = "".join(
        f"{row['sha256']}  {row['relative_path']}\n"
        for row in manifest.sort_values("relative_path").to_dict("records")
    )
    (MANIFESTS / "checksums.sha256").write_text(checksum_lines, encoding="utf-8")
    atomic_write_json({
        "schema": "phase6i-mr-instance-integrity-audit-v1.2",
        "checks": checks,
        "split_counts": split_counts,
        "r11_access_scope": "MANIFEST_ID_HASH_SIZE_ONLY",
        "status": "PASS",
    }, MANIFESTS / "phase6i_integrity_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extend-v12", action="store_true")
    args = parser.parse_args()
    if args.verify_only and args.extend_v12:
        parser.error("--verify-only and --extend-v12 are mutually exclusive")
    if args.verify_only:
        verify()
    elif args.extend_v12:
        extend_v12()
    else:
        generate()
    if args.verify_only:
        print("PHASE6I_MR_INSTANCES_VERIFY = TRUE")


if __name__ == "__main__":
    main()
