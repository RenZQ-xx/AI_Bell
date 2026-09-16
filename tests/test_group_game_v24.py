import time

import mcts.group_game.embedding_search as embedding_search
import mcts.group_game.embedding_ablation_search as embedding_ablation_search
import mcts.group_game.lean_embedding_search as lean_embedding_search
import mcts.group_game.search as search_module
from mcts.group_game.knowledge import GroupKnowledgeBase, PatternSpec
from mcts.group_game.model import GameNode, MacroAction, SymmetryState
from mcts.group_game.search import GroupGameConfig
from mcts.subgroup_patterns import SupportPatternAnalysis
from tests.test_group_game_search import _engine, _partition


def _crossed_pairs() -> tuple[tuple[int, ...], ...]:
    blocks = []
    for start in range(0, 64, 4):
        blocks.extend(((start, start + 2), (start + 1, start + 3)))
    return tuple(blocks)


def test_pattern_embeddings_share_family_value() -> None:
    family_id = "OBS-family"
    first = PatternSpec(
        "OBS-family-E1",
        3,
        "Stab(2)",
        _partition(2),
        "observed_stabilizer",
        family_id=family_id,
    )
    second = PatternSpec(
        "OBS-family-E2",
        3,
        "Stab(2)",
        _crossed_pairs(),
        "observed_stabilizer",
        family_id=family_id,
    )
    knowledge = GroupKnowledgeBase((first, second), share_family_values=True)
    action = MacroAction(
        kind="add",
        pattern_id=first.pattern_id,
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="observed_stabilizer",
        structure="Stab(2)",
        block_size=2,
        arity=1,
    )

    knowledge.update_action(
        "filler",
        action,
        rank_before=0,
        rank_after=2,
        reward=4.0,
        exact_hit=True,
        new_class=True,
    )

    assert knowledge.family_stats[("filler", family_id)].visits == 1
    assert knowledge.pattern_score("filler", second.pattern_id) > 4.0


def test_pattern_embeddings_can_keep_independent_values() -> None:
    family_id = "OBS-family"
    first = PatternSpec(
        "OBS-family-E1",
        3,
        "Stab(2)",
        _partition(2),
        "observed_stabilizer",
        family_id=family_id,
    )
    second = PatternSpec(
        "OBS-family-E2",
        3,
        "Stab(2)",
        _crossed_pairs(),
        "observed_stabilizer",
        family_id=family_id,
    )
    knowledge = GroupKnowledgeBase((first, second), share_family_values=False)
    action = MacroAction(
        kind="add",
        pattern_id=first.pattern_id,
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="observed_stabilizer",
        structure="Stab(2)",
        block_size=2,
        arity=1,
    )
    knowledge.update_action(
        "filler",
        action,
        rank_before=0,
        rank_after=2,
        reward=4.0,
        exact_hit=True,
        new_class=True,
    )

    assert not knowledge.family_stats
    assert knowledge.pattern_score("filler", second.pattern_id) == 1.25


def test_repeated_canonical_support_can_register_distinct_embeddings(monkeypatch) -> None:
    canonical_word = 0x1234
    monkeypatch.setattr(
        search_module,
        "canonical_support_word",
        lambda _word: canonical_word,
    )
    engine = _engine(
        retain_pattern_embeddings=True,
        embedding_relation_policy=True,
        hierarchical_pattern_policy=True,
        directed_pattern_relations=True,
        max_pattern_embeddings_per_family=4,
        max_pattern_analyses_per_class=4,
    )

    def analyze_support(word, **_kwargs):
        blocks = _partition(2) if int(word) & 1 else _crossed_pairs()
        return SupportPatternAnalysis(
            support_word=int(word),
            support_size=int(word).bit_count(),
            stabilizer_order=2,
            stabilizer_blocks=blocks,
            stabilizer_generators=(),
            root_pattern_ids=("PAIR",),
            root_subgroup_multiplicities=(("PAIR", 1),),
        )

    monkeypatch.setattr(engine.atlas, "analyze_support", analyze_support)
    first_word = (1 << 25) - 1
    second_word = first_word << 1
    engine.started_at = time.perf_counter()
    engine.current_iteration = 1
    engine._observe_terminal(
        GameNode(SymmetryState(first_word, "corrector", 0), rank=25),
        path=[],
    )
    engine.current_iteration = 2
    engine._observe_terminal(
        GameNode(SymmetryState(second_word, "corrector", 0), rank=25),
        path=[],
    )

    family_id = f"OBS-{canonical_word:016x}"
    embeddings = engine.knowledge.family_embeddings(family_id)
    assert len(embeddings) == 2
    assert engine.knowledge.observed_supports[canonical_word].stabilizer_pattern_ids == set(
        embeddings
    )
    context_node = GameNode(
        SymmetryState(first_word, "filler", 0, embeddings[0]),
        rank=24,
    )
    assert engine._node_pattern_relation(context_node, embeddings[1]) == "reembed"


def test_v24_entry_enables_embeddings_without_geometric_merge(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(embedding_search, "run_main", fake_main)
    embedding_search.main(["--iterations", "2", "--output", "test.json"])

    arguments = captured["arguments"]
    assert "--retain-pattern-embeddings" in arguments
    assert "--share-embedding-family-values" in arguments
    assert "--embedding-relation-policy" in arguments
    assert "--hierarchical-pattern-policy" in arguments
    assert "--geometric-transpositions" not in arguments
    assert arguments[arguments.index("--max-pattern-embeddings-per-family") + 1] == "4"


def test_v24_switch_remains_opt_in() -> None:
    assert not GroupGameConfig().retain_pattern_embeddings
    assert not GroupGameConfig().share_embedding_family_values
    assert not GroupGameConfig().embedding_relation_policy


def test_ablation_entry_disables_only_selected_mechanism(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(embedding_ablation_search, "run_main", fake_main)
    embedding_ablation_search.main([
        "--variant",
        "no_family_value",
        "--iterations",
        "2",
        "--output",
        "test.json",
    ])

    arguments = captured["arguments"]
    assert "--retain-pattern-embeddings" in arguments
    assert "--embedding-relation-policy" in arguments
    assert "--learn-relation-values" in arguments
    assert "--adaptive-filler-reheat" in arguments
    assert "--share-embedding-family-values" not in arguments


def test_cap2_ablation_uses_lean_policy_with_smaller_caps(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(embedding_ablation_search, "run_main", fake_main)
    embedding_ablation_search.main([
        "--variant",
        "simplified_cap2",
        "--iterations",
        "2",
        "--output",
        "test.json",
    ])

    arguments = captured["arguments"]
    assert "--share-embedding-family-values" not in arguments
    assert "--adaptive-filler-reheat" not in arguments
    assert "--embedding-relation-policy" in arguments
    assert arguments[
        arguments.index("--max-pattern-embeddings-per-family") + 1
    ] == "2"
    assert arguments[arguments.index("--max-pattern-analyses-per-class") + 1] == "2"


def test_lean_entry_keeps_only_supported_embedding_mechanisms(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(lean_embedding_search, "run_main", fake_main)
    lean_embedding_search.main(["--iterations", "2", "--output", "test.json"])

    arguments = captured["arguments"]
    assert "--retain-pattern-embeddings" in arguments
    assert "--embedding-relation-policy" in arguments
    assert "--learn-relation-values" in arguments
    assert "--share-embedding-family-values" not in arguments
    assert "--adaptive-filler-reheat" not in arguments
    assert "--no-leaf-value-shaping" in arguments
