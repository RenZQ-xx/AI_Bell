from types import SimpleNamespace

import numpy as np

from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.search import GroupGameConfig
from tests.test_group_game_search import _engine


def test_adaptive_replay_preserves_v20_cadence_at_zero_coverage() -> None:
    engine = _engine(
        adaptive_corrector_replay=True,
        target_classes=tuple(range(1, 47)),
    )

    due = [iteration for iteration in range(1, 13) if engine._corrector_replay_due(iteration)[0]]

    assert due == [4, 8, 12]
    assert engine._free_filler_ratio() == 0.75


def test_adaptive_replay_increases_after_coverage() -> None:
    engine = _engine(
        adaptive_corrector_replay=True,
        target_classes=(1, 2, 3, 4),
        adaptive_replay_min_coverage=0.50,
    )
    engine.class_terminal_counts.update({1: 1, 2: 1, 3: 1})

    due = sum(engine._corrector_replay_due(iteration)[0] for iteration in range(1, 21))

    assert engine._free_filler_ratio() < 0.15
    assert due >= 16


def test_newly_productive_ridge_source_preempts_yield_scheduler(monkeypatch) -> None:
    bank = FacetCorrectorBank(
        np.zeros((64, 26)),
        seed=1,
        endpoint_dedup=True,
        yield_frontier=True,
    )
    bank.anchors[100] = SimpleNamespace(word=100)
    bank.anchors[200] = SimpleNamespace(word=200)
    bank.shared.coordinates[100] = (1, (), ())
    bank.shared.coordinates[200] = (2, (), ())
    bank.ridge_enumerators[1] = SimpleNamespace(finished=False)
    bank.ridge_enumerators[2] = SimpleNamespace(finished=False)
    monkeypatch.setattr(bank, "frontier_lanes", lambda _word: ("ridge",))

    assert bank.focus_ridges(200, batches=3)
    assert bank.choose_yield_frontier({1: 100, 2: 200}) == ("ridge", 2)


def test_v20_feature_switches_remain_disabled_by_default() -> None:
    config = GroupGameConfig()

    assert not config.ridge_pivot
    assert not config.adaptive_corrector_replay
