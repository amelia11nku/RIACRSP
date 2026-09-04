import pandas as pd

from scripts.repair_phase6i_mr_r11_summary import add_decoder_seconds


def test_derived_r11_summary_recovers_decoder_seconds_from_frozen_payloads():
    summary = pd.DataFrame([
        {
            "method": "ALNS",
            "instance_id": "R11_A",
            "seed": 681401.0,
            "repair_seconds": 1.5,
        },
        {
            "method": "H1",
            "instance_id": "R11_A",
            "seed": float("nan"),
            "repair_seconds": 0.0,
        },
    ])
    records = [
        {
            "method": "ALNS",
            "instance_id": "R11_A",
            "seed": 681401,
            "runtime_components": {"decoder_seconds": 2.25},
        },
        {
            "method": "H1",
            "instance_id": "R11_A",
            "seed": None,
            "runtime_components": {"decoder_seconds": 0.0},
        },
    ]

    repaired = add_decoder_seconds(summary, records)

    assert repaired.decoder_seconds.tolist() == [2.25, 0.0]
    assert repaired.columns.tolist() == [
        "method",
        "instance_id",
        "seed",
        "repair_seconds",
        "decoder_seconds",
    ]
