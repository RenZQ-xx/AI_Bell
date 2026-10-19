from __future__ import annotations

import importlib
import time
from dataclasses import asdict, replace

import pytest

atlas = importlib.import_module("mcts.422.atlas")
model = importlib.import_module("mcts.422.model")
v29 = importlib.import_module("mcts.422.continuation_search_v29")
v30 = importlib.import_module("mcts.422.continuation_search_v30")
HANDOFF = atlas.DEFAULT_HANDOFF_ROOT
IDENTITY = atlas.IDENTITY_PATTERN_ID
SOURCE = int("ffffffccffccffccccfffcfccfcfffccccffcfcffcfcffccccffccffccffffff", 16)
NEIGHBOR = int("ffffa3c4a3c4a3c4c4a3fafa8585a3c4c4a38585fafaa3c4c4a3c4a3c4a3ffff", 16)
pytestmark = pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")


@pytest.fixture
def engine():
    engine = v30.Bell422EarlyNoopSearch(v30.EarlyNoopSearchConfig(
        iterations=4, progress_interval=0, use_catalog=False,
        canonicalize_classes=False, stabilizer_probes=4,
    ), handoff_root=HANDOFF)
    engine.started_at = time.perf_counter()
    assert engine.facet_oracle.validate(SOURCE).valid
    return engine


def redundant_points(engine):
    return [1 << index for index in range(256)
            if SOURCE & (1 << index) and engine.rank_oracle.rank(SOURCE ^ (1 << index)) == 79]


def correction(kind, removed, added=0, pattern=IDENTITY):
    return model.MacroAction(kind, pattern, (), removed | added, "test", "test",
                             remove_word=removed, add_word=added)


@pytest.mark.parametrize("kind", ["remove", "rewrite"])
def test_certified_noop_never_reconstructs_facet(engine, monkeypatch, kind):
    removed = redundant_points(engine)[0]
    state = model.SearchState(SOURCE, "corrector", 7, IDENTITY, SOURCE)
    node = engine._node(state, 79)
    action = correction(kind, removed, pattern="BFS422-H-0000")

    def forbidden_validate(word):
        raise AssertionError("certified no-op should not reconstruct a facet")

    monkeypatch.setattr(engine.facet_oracle, "validate", forbidden_validate)
    assert engine._transition(state, action) is None
    assert state.corrections_used == 7
    assert action.pattern_id in node.context_pattern_ids
    assert engine.stats["early_noop_actions_skipped"] == 1
    assert engine.stats["uncached_facet_validations_avoided"] == 1
    assert engine._transition(state, action) is None
    assert engine.stats["geometry_noop_cache_hits"] == 1
    assert not node.edges


def test_effect_certificate_reused_across_charts_without_rank_check(engine, monkeypatch):
    removed = redundant_points(engine)[0]
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    first = correction("remove", removed)
    second = correction("remove", removed, pattern="BFS422-H-0000")
    assert engine._transition(state, first) is None

    def forbidden_rank(word):
        raise AssertionError("identical geometry should reuse its certificate")

    monkeypatch.setattr(engine.rank_oracle, "rank", forbidden_rank)
    assert engine._transition(state, second) is None
    assert engine.stats["noop_effect_cache_hits"] == 1
    assert engine.stats["geometry_noop_corrections_skipped"] == 2
    assert len(engine.noop_actions) == 2


def test_unknown_parent_falls_back_to_validation(engine, monkeypatch):
    removed = redundant_points(engine)[0]
    engine.facet_oracle.cache.pop(SOURCE)
    calls = []
    original = engine.facet_oracle.validate

    def counted(word):
        calls.append(word)
        return original(word)

    monkeypatch.setattr(engine.facet_oracle, "validate", counted)
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    assert engine._transition(state, correction("remove", removed)) is None
    assert calls == [SOURCE & ~removed]
    assert engine.stats["early_noop_uncertified_parents"] == 1
    assert engine.stats["early_noop_actions_skipped"] == 0


def test_real_ridge_keeps_correction_and_source(engine):
    state = model.SearchState(SOURCE, "corrector", 7, IDENTITY, SOURCE)
    moved = engine._transition(state, correction("remove", SOURCE & ~NEIGHBOR))
    assert moved.rank == 78
    assert moved.state.support_word == SOURCE & NEIGHBOR
    assert moved.state.corrections_used == 8
    assert moved.state.source_facet_word == SOURCE
    assert engine.stats["early_noop_actions_skipped"] == 0


def test_rewrite_outside_parent_can_reach_other_facet(engine):
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    moved = engine._transition(state, correction(
        "rewrite", SOURCE & ~NEIGHBOR, NEIGHBOR & ~SOURCE,
    ))
    assert moved.facet.valid
    assert moved.state.support_word == NEIGHBOR
    assert moved.state.corrections_used == 1
    assert engine.stats["early_noop_actions_skipped"] == 0


def test_unchanged_word_does_not_create_new_noop_chart(engine):
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    action = correction("remove", 0, pattern="BFS422-H-0000")
    assert engine._transition(state, action) is None
    assert not engine.noop_actions
    assert not engine.noop_charts


def test_effect_cache_is_bounded_and_eviction_keeps_correctness(engine):
    engine.config = replace(engine.config, noop_effect_cache_size=1)
    points = redundant_points(engine)[:2]
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    for point in points:
        assert engine._transition(state, correction("remove", point)) is None
    assert len(engine.noop_effects) == 1
    assert engine.stats["noop_effect_cache_evictions"] == 1
    assert engine._transition(state, correction("remove", points[0], pattern="BFS422-H-0000")) is None
    assert engine.stats["early_noop_actions_skipped"] == 3


def test_disabling_noop_filter_also_disables_guard(engine):
    engine.config = replace(engine.config, skip_geometry_noop_corrections=False)
    removed = redundant_points(engine)[0]
    state = model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE)
    moved = engine._transition(state, correction("remove", removed))
    assert moved.facet.valid
    assert moved.state.support_word == SOURCE
    assert moved.state.corrections_used == 1
    assert not engine.noop_effects


@pytest.mark.parametrize("enabled", [False, True])
def test_v30_preserves_v29_trajectory(enabled):
    config = v30.EarlyNoopSearchConfig(
        iterations=12, max_depth=6, tail_checks=2, stabilizer_probes=4,
        quotient_candidate_limit=4, quotient_secondary_limit=1,
        partial_rank_checks=2, ridge_rank_checks=4, ridge_basis_rank_checks=8,
        progress_interval=0, use_catalog=False, canonicalize_classes=False,
        enable_early_noop_guard=enabled,
    )
    baseline_config = v29.ContinuationSearchConfig(**{
        key: value for key, value in asdict(config).items()
        if key in v29.ContinuationSearchConfig.__dataclass_fields__
    })
    old = v29.Bell422ContinuationSearch(baseline_config, handoff_root=HANDOFF).run()
    new = v30.Bell422EarlyNoopSearch(config, handoff_root=HANDOFF).run()
    assert new["tree"] == old["tree"]
    assert new["coverage"]["discovered_class_ids"] == old["coverage"]["discovered_class_ids"]
    assert [event["path"] for event in new["coverage"]["discoveries"]] == [
        event["path"] for event in old["coverage"]["discoveries"]
    ]
    assert new["continuation_game"]["anchors"] == old["continuation_game"]["anchors"]
    assert new["continuation_game"]["unique_noop_actions"] == old["continuation_game"]["unique_noop_actions"]
    if enabled:
        assert new["early_noop_guard"]["actions_skipped"] > 0
        assert new["facet_oracle"]["validation_count"] < old["facet_oracle"]["validation_count"]
