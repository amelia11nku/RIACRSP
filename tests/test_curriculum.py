from __future__ import annotations

from rcias_clgri.training.curriculum import CurriculumManager


def test_curriculum_requires_feasible_stable_validation_and_replays_old_levels():
    manager = CurriculumManager(
        plateau_window=3,
        plateau_relative_improvement=0.02,
        minimum_updates=3,
        current_level_probability=0.75,
    )
    assert not manager.record_validation(100.0, feasibility_rate=1.0, normalized_entropy=0.5)
    assert not manager.record_validation(99.5, feasibility_rate=0.9, normalized_entropy=0.5)
    assert manager.record_validation(99.4, feasibility_rate=1.0, normalized_entropy=0.5)
    assert manager.current_level == "M"
    assert manager.sampling_probabilities() == {"M": 0.75, "S": 0.25}
