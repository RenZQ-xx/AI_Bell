from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace

from mcts.filler_corrector_search import (
    BlockCorrector,
    CorrectorConfig,
    remove_blocks,
)
from mcts.search import TerminalHit


class _FakeScorer:
    blocks = [(0,), (1,), (2,), (3,)]
    target_classes = {1, 2, 3}

    @staticmethod
    def affine_rank(key: tuple[int, ...]) -> int:
        return sum(int(value) for value in key)

    @staticmethod
    def rank24_entrance_metrics(key: tuple[int, ...]) -> dict[str, int]:
        return {
            "rare": 1 if key[3] == 0 else 0,
            "class44": 0,
            "other_valid": 1,
            "invalid": 0,
        }


class _FakeCompatibilityBank:
    @staticmethod
    def active_signature(
        _key: tuple[int, ...],
        class_ids: list[int],
    ) -> tuple[int, ...]:
        return tuple(class_ids)


def test_remove_blocks_does_not_mutate_source_key() -> None:
    source = (1, 1, 0, 1)

    corrected = remove_blocks(source, [1, 3])

    assert source == (1, 1, 0, 1)
    assert corrected == (1, 0, 0, 0)


def test_corrector_protects_core_and_returns_nonterminal_high_rank_prefix() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter({"exact:class1": 5}),
        ),
        terminal_bests={source.label: source},
        compatibility_bank=_FakeCompatibilityBank(),
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            max_remove_blocks=1,
            candidate_pool=4,
        )
    )

    proposal = corrector.propose(state, rng=random.Random(7))

    assert proposal is not None
    assert proposal.protected_blocks == (0,)
    assert 0 not in proposal.removed_blocks
    assert proposal.removed_blocks == (3,)
    assert proposal.corrected_rank == 3
    assert proposal.corrected_rank < corrector.config.terminal_rank
    assert proposal.entrance_rare_count == 1


def test_corrector_does_not_repeat_the_same_corrected_prefix() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter(),
        ),
        terminal_bests={source.label: source},
        compatibility_bank=None,
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            candidate_pool=4,
        )
    )

    first = corrector.propose(state, rng=random.Random(3))
    second = corrector.propose(state, rng=random.Random(3))

    assert first is not None
    assert second is not None
    assert first.corrected_key != second.corrected_key
