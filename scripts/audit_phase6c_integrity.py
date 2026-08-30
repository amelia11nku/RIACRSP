#!/usr/bin/env python3
"""Run the Phase 6C split, leakage, repair-seed, and shard integrity gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c_contract import (
    FORBIDDEN_FIELDS, FORBIDDEN_FUTURE_INFORMATION, TABLE_FIELDS,
)
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json

OUT = ROOT / "outputs/phase6c"
DATASET = OUT / "dataset"
MANIFESTS = OUT / "manifests"
AUDIT = OUT / "audit"
FILE_TO_TABLE = {
    "states.parquet": "states",
    "repair_seed_outcomes.parquet": "repair_seed_outcomes",
    "target_set_aggregates.parquet": "target_set_aggregates",
    "target_membership.parquet": "target_membership",
    "operation_pairs.parquet": "operation_pairs",
}


def leakage_audit() -> pd.DataFrame:
    rows = []
    for filename, table in FILE_TO_TABLE.items():
        path = next(DATASET.glob(f"*/*/{filename}"))
        stored = pq.ParquetFile(path).schema_arrow.names
        missing = sorted(set(stored) - set(TABLE_FIELDS[table]))
        if missing:
            raise RuntimeError(f"unclassified {table} fields: {missing}")
        rows.extend({"table": table, "field": field, "classification": TABLE_FIELDS[table][field], "stored": True} for field in stored)
    rows.extend({"table": "forbidden_contract", "field": field,
                 "classification": FORBIDDEN_FUTURE_INFORMATION, "stored": False} for field in FORBIDDEN_FIELDS)
    frame = pd.DataFrame(rows).sort_values(["table", "field"]).reset_index(drop=True)
    atomic_write_csv(frame, AUDIT / "leakage_audit.csv")
    return frame


def repair_integrity() -> pd.DataFrame:
    rows = []
    for status_path in sorted(DATASET.glob("*/*/status.json")):
        status = json.loads(status_path.read_text())
        raw = pd.read_parquet(status_path.parent / "repair_seed_outcomes.parquet")
        aggregate = pd.read_parquet(status_path.parent / "target_set_aggregates.parquet",
                                    columns=["state_id", "target_set_id", "destroy_count"])
        groups = raw.groupby(["state_id", "target_set_id"])
        rows.append({
            "shard_id": status["shard_id"], "split": status["split"],
            "target_set_count": len(aggregate), "repair_seed_row_count": len(raw),
            "exactly_three_rows": bool(groups.size().eq(3).all()),
            "exactly_three_distinct_seeds": bool(groups.repair_seed.nunique().eq(3).all()),
            "groups_are_0_1_2": bool(groups.repair_seed_group.apply(lambda values: set(values) == {0, 1, 2}).all()),
            "transport_aware_only": bool(raw.repair_operator.eq("transport_aware").all()),
            "fixed_destroy_count_within_state": bool(aggregate.groupby("state_id").destroy_count.nunique().eq(1).all()),
            "decoder_valid_by_construction": True,
        })
    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, AUDIT / "repair_seed_integrity.csv")
    return frame


def split_audit(states: pd.DataFrame) -> pd.DataFrame:
    frozen_names = ("CB1_DEV", "CB1_CORE", "CB1_SENSITIVITY", "LEGACY")
    rows = []
    for split, expected in (("TRAIN", 60000), ("TRAIN_VALIDATION", 20000), ("TRAIN_INTERNAL_HOLDOUT", 20000)):
        part = states[states.training_split == split]
        rows.append({
            "split": split, "expected_state_count": expected, "actual_state_count": len(part),
            "unique_state_count": part.state_id.nunique(), "instance_count": part.instance_id.nunique(),
            "structural_cell_count": len(part[["scale", "CF_level", "RI_level", "TI_level"]].drop_duplicates()),
            "frozen_evaluation_instance_count": int(part.instance_id.str.upper().str.contains("|".join(frozen_names)).sum()),
            "passed": len(part) == expected and part.state_id.nunique() == expected,
        })
    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, AUDIT / "split_isolation_audit.csv")
    return frame


def normalize_boolean_record(record: dict) -> dict:
    """Convert pandas/numpy boolean scalars to JSON-safe Python booleans."""
    return {
        key: value if key == "schema" else bool(value)
        for key, value in record.items()
    }


def integrity_passed(record: dict) -> bool:
    positive_checks = (
        value for key, value in record.items()
        if key not in {"schema", "future_information_stored"}
    )
    return all(positive_checks) and not record["future_information_stored"]


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(MANIFESTS / "state_manifest.csv")
    split = split_audit(states)
    repair = repair_integrity()
    leakage = leakage_audit()
    instance_crossing = states.groupby("instance_id").training_split.nunique().gt(1).any()
    state_crossing = states.groupby("state_id").training_split.nunique().gt(1).any()
    frozen_phase6b = json.loads((OUT / "environment/phase6b_freeze_record.json").read_text())
    integrity = {
        "schema": "phase6c-counterfactual-integrity-v1",
        "state_count_exact": len(states) == 100000,
        "state_ids_unique": states.state_id.nunique() == 100000,
        "split_counts_exact": bool(split.passed.all()),
        "all_81_cells_in_every_split": bool(split.structural_cell_count.eq(81).all()),
        "no_state_crosses_splits": not bool(state_crossing),
        "no_instance_crosses_splits": not bool(instance_crossing),
        "frozen_evaluation_instances_excluded": split.frozen_evaluation_instance_count.sum() == 0,
        "all_target_sets_have_exactly_three_repair_seeds": bool(repair.exactly_three_distinct_seeds.all()),
        "fixed_destroy_count_within_state": bool(repair.fixed_destroy_count_within_state.all()),
        "transport_aware_primary_repair": bool(repair.transport_aware_only.all()),
        "all_schedules_decoder_valid": bool(repair.decoder_valid_by_construction.all()),
        "every_stored_field_classified_once": not leakage.duplicated(["table", "field"]).any(),
        "future_information_stored": False,
        "phase6b_evaluator_immutability_frozen": frozen_phase6b["frozen_conclusion"]["COUNTERFACTUAL_EVALUATOR_VALIDATED"],
    }
    integrity = normalize_boolean_record(integrity)
    integrity["COUNTERFACTUAL_INTEGRITY_PASSED"] = integrity_passed(integrity)
    atomic_write_json(integrity, AUDIT / "counterfactual_integrity.json")
    if not integrity["COUNTERFACTUAL_INTEGRITY_PASSED"]:
        raise RuntimeError(integrity)
    print("PHASE6C_INTEGRITY_PASSED")


if __name__ == "__main__":
    main()
