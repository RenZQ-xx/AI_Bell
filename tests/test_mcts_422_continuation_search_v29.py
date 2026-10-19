from __future__ import annotations

import importlib
import time
from dataclasses import asdict

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
model = importlib.import_module("mcts.422.model")
base = importlib.import_module("mcts.422.v25_search")
v27 = importlib.import_module("mcts.422.quotient_search_v27")
v29 = importlib.import_module("mcts.422.continuation_search_v29")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT
IDENTITY = atlas_module.IDENTITY_PATTERN_ID
SOURCE = int("ffffffccffccffccccfffcfccfcfffccccffcfcffcfcffccccffccffccffffff", 16)
NEIGHBOR = int("ffffa3c4a3c4a3c4c4a3fafa8585a3c4c4a38585fafaa3c4c4a3c4a3c4a3ffff", 16)
pytestmark = pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")


@pytest.fixture
def engine():
    config = v29.ContinuationSearchConfig(
        iterations=4, progress_interval=0, use_catalog=False,
        canonicalize_classes=False, stabilizer_probes=4,
    )
    search = v29.Bell422ContinuationSearch(config, handoff_root=HANDOFF)
    search.started_at = time.perf_counter()
    return search


def observe_source(engine):
    action = model.MacroAction("add", IDENTITY, (), SOURCE, "test_seed", "test", add_word=SOURCE)
    moved = engine._transition(engine.root.state, action)
    assert moved.facet.valid
    edge, moved = engine._edge_for(engine.root, action, moved)
    step = base.PathStep(engine.root, edge, 0.0, True)
    engine.current_iteration = 1
    outcome = engine._terminal_outcome(moved.facet, path=[step], depth=1, source="test")
    assert outcome.new_class
    return moved.facet, step


def test_continuation_resets_budget_without_replaying_or_rediscovering(engine):
    facet, step = observe_source(engine)
    family = engine._family_key(facet)
    anchor = engine.anchors[family][SOURCE]
    historical_remove = model.MacroAction("remove", IDENTITY, (), 1, "history", "test", remove_word=1)
    anchor.witness = (step.edge.action,) + (historical_remove,) * 8

    node, replay = engine._iteration_start(4)

    assert node.state.support_word == SOURCE
    assert node.state.source_facet_word == SOURCE
    assert node.state.corrections_used == 0
    assert replay == ()
    assert engine.stats["historical_corrections_reset"] == 8
    assert engine.stats["historical_replay_steps_saved"] == 9
    assert not engine._correction_limit_reached(node)
    known = engine._terminal_outcome(facet, path=[], depth=0, source="test")
    assert not known.new_class
    assert len(engine.discovered) == 1


def test_noop_deletion_is_chart_only_and_cached(engine):
    facet = engine.facet_oracle.validate(SOURCE)
    assert facet.valid
    removed = next(1 << index for index in range(256)
                   if SOURCE & (1 << index) and engine.rank_oracle.rank(SOURCE ^ (1 << index)) == 79)
    state = model.SearchState(SOURCE, "corrector", 7, IDENTITY, SOURCE)
    node = engine._node(state, 79)
    pattern = "BFS422-H-0000"
    action = model.MacroAction("remove", pattern, (), removed, "test_noop", "test", remove_word=removed)

    assert engine._transition(state, action) is None
    assert node.state.corrections_used == 7
    assert pattern in node.context_pattern_ids
    assert not node.edges
    assert engine.stats["geometry_noop_corrections_skipped"] == 1
    assert engine._transition(state, action) is None
    assert engine.stats["geometry_noop_cache_hits"] == 1
    assert len(engine.noop_actions) == 1


def test_real_ridge_deletion_still_consumes_one_local_correction(engine):
    state = model.SearchState(SOURCE, "corrector", 7, IDENTITY, SOURCE)
    removed = SOURCE & ~NEIGHBOR
    action = model.MacroAction("remove", IDENTITY, (), removed, "test_ridge", "test", remove_word=removed)

    moved = engine._transition(state, action)

    assert moved.state.role == "filler"
    assert moved.rank == 78
    assert moved.state.corrections_used == 8
    assert moved.state.support_word == SOURCE & NEIGHBOR


def test_known_noops_do_not_occupy_rollout_candidate_slots(engine, monkeypatch):
    node = engine._node(model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE), 79)
    noop = model.MacroAction("remove", IDENTITY, (), 1, "noop", "test", remove_word=1)
    valid = model.MacroAction("remove", IDENTITY, (), SOURCE & ~NEIGHBOR,
                             "ridge", "test", remove_word=SOURCE & ~NEIGHBOR)
    engine.noop_actions.add((SOURCE, noop.action_id))
    monkeypatch.setattr(v27.Bell422AdaptiveQuotientSearch, "_actions_for_pattern",
                        lambda self, node, pattern_id, lane: [noop, valid])
    assert engine._actions_for_pattern(node, IDENTITY, "remove_small") == [valid]
    assert engine.stats["noop_candidates_filtered"] == 1


def test_rollout_refills_after_new_geometry_rejections(engine, monkeypatch):
    node = engine._node(model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE), 79)
    remove = model.MacroAction("remove", IDENTITY, (), SOURCE & ~NEIGHBOR,
                              "ridge", "test", remove_word=SOURCE & ~NEIGHBOR)
    valid = engine._edge_for(node, remove)
    calls = []

    def rollout(self, node, policy):
        calls.append(policy)
        if len(calls) == 1:
            self.noop_actions.add((SOURCE, "newly_rejected_action"))
            return self._edge_for(node, self._stop_action())
        return valid

    monkeypatch.setattr(v27.Bell422AdaptiveQuotientSearch, "_rollout_edge", rollout)
    assert engine._rollout_edge(node, "mixed") == valid
    assert len(calls) == 2
    assert engine.stats["noop_rollout_refills"] == 1


def test_rollout_does_not_refill_an_intentional_stop(engine, monkeypatch):
    node = engine._node(model.SearchState(SOURCE, "corrector", 0, IDENTITY, SOURCE), 79)
    calls = []

    def rollout(self, node, policy):
        calls.append(policy)
        return self._edge_for(node, self._stop_action())

    monkeypatch.setattr(v27.Bell422AdaptiveQuotientSearch, "_rollout_edge", rollout)
    result = engine._rollout_edge(node, "mixed")
    assert result[0].action.kind == "stop"
    assert len(calls) == 1


def test_local_credit_excludes_historical_proof_and_paths_remain_complete(engine):
    source_facet, seed = observe_source(engine)
    node, _ = engine._iteration_start(4)
    engine.current_iteration = 4
    origin = engine._terminal_outcome(source_facet, path=[], depth=0, source="test")
    removed = SOURCE & ~NEIGHBOR
    remove = model.MacroAction("remove", IDENTITY, (), removed, "test_ridge", "test", remove_word=removed)
    edge, moved = engine._edge_for(node, remove)
    first = engine._make_step(node, edge, moved.state, current_terminal=origin, is_tree_edge=True)
    filler = engine._node(moved.state, moved.rank)
    added = NEIGHBOR & ~moved.state.support_word
    add = model.MacroAction("add", IDENTITY, (), added, "fullspace_ridge_frontier", "test", add_word=added)
    edge, moved = engine._edge_for(filler, add)
    second = engine._make_step(filler, edge, moved.state, current_terminal=None, is_tree_edge=True)
    outcome = engine._terminal_outcome(moved.facet, path=[first, second], depth=2, source="test")

    event = engine.discovered[outcome.class_id]
    assert outcome.new_class
    assert len(event["local_path"]) == 2
    assert len(event["path"]) == 3
    assert event["path"][0] == seed.edge.action.to_dict()
    assert event["continuation_origin"]["class_id"] == source_facet.class_id
    assert first.corrector_novelty_credit == pytest.approx(engine.config.new_class_reward)
    assert seed.corrector_novelty_credit == 0.0
    assert event["local_depth"] == 2
    assert event["depth"] == 3


def test_continuation_rotates_concrete_embeddings_inside_one_class(engine):
    family = (123,)
    engine.anchors[family] = {
        SOURCE: v29.FacetAnchor("known", family, SOURCE, (), 1, {IDENTITY}),
        NEIGHBOR: v29.FacetAnchor("known", family, NEIGHBOR, (), 1, {IDENTITY}),
    }
    first, _ = engine._iteration_start(4)
    second, _ = engine._iteration_start(8)
    assert first.state.support_word != second.state.support_word
    assert engine.facet_replay_counts[family] == 2


def test_filler_archives_positive_historically_productive_edge(monkeypatch):
    engine = v29.Bell422ContinuationSearch(v29.ContinuationSearchConfig(
        iterations=2, root_max_children=1, progress_interval=0,
        use_catalog=False, canonicalize_classes=False,
    ), handoff_root=HANDOFF)
    words = iter((SOURCE, NEIGHBOR))
    nonces = []

    def proposals(state, pattern_id, *, nonce, count):
        nonces.append(nonce)
        word = next(words)
        return [model.MacroAction("add", pattern_id, (), word,
                                 "fullspace_supporting_face_lp", "test", add_word=word)]

    monkeypatch.setattr(engine.actions, "supporting_face_actions", proposals)
    engine.current_iteration = 1
    first = engine._widen_root_supporting_face()
    first.update(4.4, 0)
    first.discovery_hits = 10
    old_child = engine.nodes[first.child_key]
    engine.current_iteration = 2
    second = engine._widen_root_supporting_face()

    assert second is not None
    assert len(engine.root.edges) == 1
    assert first.action.action_id in engine.archived_root_edges
    assert engine.nodes[first.child_key] is old_child
    assert first.discovery_hits == 10
    assert nonces == [0, 1]
    assert engine.stats["root_entrances_archived"] == 1


def test_disabled_features_preserve_v27_search_trajectory():
    config = v29.ContinuationSearchConfig(
        iterations=5, max_depth=4, tail_checks=2, stabilizer_probes=4,
        quotient_candidate_limit=4, quotient_secondary_limit=1,
        partial_rank_checks=2, ridge_rank_checks=4, ridge_basis_rank_checks=8,
        progress_interval=0, use_catalog=False, canonicalize_classes=False,
        enable_local_continuation=False, skip_geometry_noop_corrections=False,
        fresh_root_entrances=False,
    )
    baseline_config = v27.AdaptiveQuotientSearchConfig(**{
        key: value for key, value in asdict(config).items()
        if key in v27.AdaptiveQuotientSearchConfig.__dataclass_fields__
    })
    old = v27.Bell422AdaptiveQuotientSearch(baseline_config, handoff_root=HANDOFF).run()
    new = v29.Bell422ContinuationSearch(config, handoff_root=HANDOFF).run()
    assert new["tree"] == old["tree"]
    assert [event["path"] for event in new["coverage"]["discoveries"]] == [
        event["path"] for event in old["coverage"]["discoveries"]
    ]
    assert new["coverage"]["discovered_class_ids"] == old["coverage"]["discovered_class_ids"]
    assert new["quotient_game"]["quotient_stats"] == old["quotient_game"]["quotient_stats"]
