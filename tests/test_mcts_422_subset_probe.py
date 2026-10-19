from __future__ import annotations

import importlib

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
probe_module = importlib.import_module("mcts.422.subset_probe")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_atlas_pattern_filter_preserves_identity_and_hides_other_patterns() -> None:
    atlas = atlas_module.HandoffPatternAtlas(
        HANDOFF,
        allowed_pattern_ids={"BFS422-H-1193", "BFS422-H-1195"},
    )

    assert len(atlas.patterns) == 3
    assert atlas.patterns[0].pattern_id == atlas_module.IDENTITY_PATTERN_ID
    assert set(atlas.by_id) == {
        atlas_module.IDENTITY_PATTERN_ID,
        "BFS422-H-1193",
        "BFS422-H-1195",
    }
    assert atlas.to_dict()["answer_files_read_by_atlas"] == []
    assert atlas.to_dict()["pattern_filter_active"] is True


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_subset_selection_uses_only_structural_metadata() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    selected = probe_module.select_pattern_ids(
        atlas,
        structures=("C6", "S3"),
        limit=8,
        seed=17,
    )

    assert len(selected) == 8
    assert {
        atlas.pattern(pattern_id).structure for pattern_id in selected
    } == {"C6", "S3"}
    assert set(selected) == set(
        probe_module.select_pattern_ids(
            atlas,
            structures=("C6", "S3"),
            limit=8,
            seed=17,
        )
    )
