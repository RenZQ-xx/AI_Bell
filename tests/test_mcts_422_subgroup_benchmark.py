from __future__ import annotations

import importlib

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
benchmark_module = importlib.import_module("mcts.422.subgroup_benchmark")
v25_module = importlib.import_module("mcts.422.v25_search")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT
NODE_ID = "BFS422-H-0000"


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_h0000_evaluation_checklist_has_sixteen_unique_classes() -> None:
    targets = benchmark_module.load_target_class_ids(HANDOFF, NODE_ID)

    assert len(targets) == 16
    assert len(set(targets)) == 16


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_fixed_subgroup_search_exposes_only_selected_partition() -> None:
    engine = benchmark_module.FixedSubgroupSearch(
        v25_module.V25SearchConfig(iterations=1, progress_interval=0),
        fixed_pattern_id=NODE_ID,
        handoff_root=HANDOFF,
    )

    assert engine.root.state.context_pattern_id == NODE_ID
    assert engine.pattern_order == (NODE_ID,)
    assert engine._candidate_pattern_ids(engine.root) == (NODE_ID,)
    assert engine._rollout_pattern_ids(engine.root, "mixed") == [NODE_ID]
    assert engine.atlas.to_dict()["answer_files_read_by_atlas"] == []
    assert set(engine.atlas.by_id) == {atlas_module.IDENTITY_PATTERN_ID, NODE_ID}
