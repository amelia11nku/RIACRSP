from __future__ import annotations

import hashlib

from rcias_clgri.learning.experiment import run_metadata
from rcias_clgri.learning.experiment import load_phase3_config


def test_run_metadata_records_reproducibility_fields(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"test": true}\n', encoding="utf-8")
    metadata = run_metadata(config, device="cpu", training_seed=123)
    assert metadata["training_seed"] == 123
    assert metadata["device"] == "cpu"
    assert metadata["gpu"] is None
    assert metadata["vram_bytes"] == 0
    assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    for key in ("hostname", "cpu", "ram_bytes", "python", "torch_version", "git_commit"):
        assert key in metadata


def test_config_inheritance_deep_merges(tmp_path):
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text('{"ppo": {"lr": 1, "clip": 2}, "seed": 3}', encoding="utf-8")
    child.write_text('{"extends": "base.json", "ppo": {"lr": 4}}', encoding="utf-8")
    assert load_phase3_config(child) == {"ppo": {"lr": 4, "clip": 2}, "seed": 3}
