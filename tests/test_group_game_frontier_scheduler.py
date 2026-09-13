from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from mcts.group_game.frontier_scheduler import FrontierScheduler
from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.search import GroupGameConfig, build_parser


@pytest.mark.parametrize("pressure,expected", [
    (False, {"ridge": 4, "completion": 6, "orbit": 4, "delivery": 2}),
    (True, {"ridge": 4, "completion": 8, "orbit": 2, "delivery": 2}),
])
def test_lane_quotas_and_fair_source_selection(pressure, expected):
    scheduler = FrontierScheduler(high_water=24)
    available = {lane: (1, 2) for lane in ("ridge", "completion", "orbit", "delivery")}
    remaining = {1: 24 if pressure else 23, 2: 0}
    choices = [scheduler.choose(available, remaining, scope=("replay", 0)) for _ in range(16)]
    assert Counter(lane for lane, _ in choices) == expected
    for lane, count in expected.items():
        assert sum(choice == (lane, 1) for choice in choices) == count // 2
        assert sum(choice == (lane, 2) for choice in choices) == count // 2
    assert scheduler.pressure_counts["replay"] == (16 if pressure else 0)


def test_empty_lanes_skipped_and_pressure_does_not_disable_orbit():
    scheduler = FrontierScheduler(high_water=24)
    assert scheduler.choose({}, {}, scope=("replay", 0)) is None
    choices = [scheduler.choose({"completion": (1,), "orbit": (1,)}, {1: 1000},
                                scope=("grow", 1))[0] for _ in range(10)]
    assert Counter(choices) == {"completion": 8, "orbit": 2}
    assert scheduler.choose({"delivery": (7,)}, {}, scope=("replay", 0)) == ("delivery", 7)
    with pytest.raises(ValueError, match="unknown"):
        scheduler.choose({"invalid": (1,)}, {}, scope=("replay", 0))


def test_bank_shared_canonical_progress_and_one_dispatch_per_opportunity(monkeypatch):
    bank = FacetCorrectorBank(np.zeros((64, 26)), seed=1, fair_frontier=True,
                             lower_face_max_attempts=6)
    for raw, canonical in ((100, 1), (101, 1), (200, 2)):
        bank.anchors[raw] = SimpleNamespace(word=raw)
        bank.shared.coordinates[raw] = (canonical, (), ())
        bank.shared.untried_exits[canonical] = set()
    bank.ridge_enumerators[1] = SimpleNamespace(finished=False)
    bank.orbit_faces[2] = SimpleNamespace(finished=False)
    for canonical, raw in ((1, 100), (2, 200)):
        for r in range(8):
            job = bank.completions.register((canonical, r, "p"), source=raw, retained=r, pattern_id="p")
            job.attempts = 2
    assert bank.completions.remaining_attempts(1) == 32
    calls = []
    for lane, method in (("ridge", "_grow_systematic"), ("orbit", "_grow_orbit_faces"),
                         ("completion", "_grow_lower_faces")):
        monkeypatch.setattr(bank, method, lambda anchor, lane=lane: calls.append(lane))
    for raw in (100, 101, 100, 101, 100, 101):
        before = len(calls)
        bank._scheduled_local_work(bank.anchors[raw])
        assert len(calls) == before + 1
    assert bank.frontier_scheduler.cursors[("grow", 1)] > 0
    assert ("grow", 100) not in bank.frontier_scheduler.cursors
    assert ("grow", 101) not in bank.frontier_scheduler.cursors
    assert Counter(calls) == {"ridge": 2, "completion": 4}
    replay = [bank.choose_replay_frontier({1: 100, 2: 200}) for _ in range(14)]
    assert Counter(lane for lane, _ in replay) == {"ridge": 4, "completion": 8, "orbit": 2}
    assert all(word == 1 for lane, word in replay if lane == "ridge")
    assert all(word == 2 for lane, word in replay if lane == "orbit")
    assert not bank.shared.exit_uses
    assert not bank.shared.endpoints.targets
    bank.ridge_enumerators[1].finished = True
    for job in bank.completions.jobs.values():
        if job.key[0] == 1:
            job.attempts = 6
    assert bank.completions.remaining_attempts(1) == 0
    assert bank.frontier_lanes(101) == ()


def test_scheduler_config_and_disable_switch():
    assert not build_parser().parse_args([]).no_fair_frontier
    assert GroupGameConfig(fair_frontier=False).fair_frontier is False
    with pytest.raises(ValueError, match="high water"):
        GroupGameConfig(frontier_high_water=0)
    with pytest.raises(ValueError, match="high water"):
        FrontierScheduler(high_water=0)
