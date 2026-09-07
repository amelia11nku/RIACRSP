import numpy as np

from scripts.audit_phase6m_failure_attribution import (
    brier_decomposition,
    normalized_entropy,
)


def test_brier_decomposition_tracks_brier_with_exact_bins():
    probability = np.asarray([0.1, 0.1, 0.9, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    result = brier_decomposition(probability, labels)
    reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
    assert np.isclose(reconstructed, np.mean((probability - labels) ** 2))
    assert np.isclose(result["brier_identity_residual"], 0.0)


def test_normalized_entropy_has_expected_ordering():
    tied = normalized_entropy(np.asarray([0.0, 0.0, 0.0]))
    separated = normalized_entropy(np.asarray([4.0, 0.0, -4.0]))
    assert np.isclose(tied, 1.0)
    assert 0.0 < separated < tied
