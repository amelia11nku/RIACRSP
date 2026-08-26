from __future__ import annotations

import hashlib

from rcias_clgri.learning.experiment import run_metadata


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
