from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.search import GroupGameConfig, build_parser
from mcts.group_game.yield_frontier import YieldFrontier


def test_novelty_excludes_self_aliases_and_observed_targets():
    before = {(1, 2), (2, 2)}
    after = before | {(1, 1), (3, 2), (1, 4), (3, 4), (1, 5)}
    assert YieldFrontier.novelty(before, {1, 2, 5}, after) == (1, 4)
    assert YieldFrontier.novelty(before, {1, 2}, before) == (0, 0)


def test_dispatch_novelty_and_recent_yield_decay_ignore_wall_time():
    scheduler = YieldFrontier()
    scheduler.observe(1, "ridge", 0.02, 1, 1)
    scheduler.observe(2, "ridge", 0.01, 1, 1)
    assert scheduler.score(2, "ridge") == pytest.approx(
        scheduler.score(1, "ridge")
    )
    scheduler.observe(3, "ridge", 0.01, 0, 1)
    assert scheduler.score(2, "ridge") > scheduler.score(3, "ridge")
    for _ in range(40):
        scheduler.observe(2, "ridge", 0.01, 0, 0)
    assert scheduler.score(2, "ridge") < scheduler.score(3, "ridge")
    assert scheduler.choose({"ridge": (2, 3)}, iteration=1) == ("ridge", 2)
    assert scheduler.choose({"ridge": (2, 3)}, iteration=2) == ("ridge", 3)


def test_periodic_lane_probes_even_for_low_yield_and_shared_source_service():
    scheduler = YieldFrontier()
    scheduler.observe(1, "ridge", 0.001, 1, 1)
    for lane in ("orbit", "completion"):
        scheduler.observe(1, lane, 10, 0, 0)
    available = {lane: (1, 1, 2) for lane in scheduler.lanes}
    for i in range(32):
        scheduler.choose(available, iteration=i + 1)
    probes = [d for d in scheduler.decisions if d["reason"] == "probe"]
    assert Counter(d["lane"] for d in probes) == dict.fromkeys(scheduler.lanes, 2)
    assert {d["canonical_source_hex"] for d in probes} == {"0x0000000000000001", "0x0000000000000002"}
    assert scheduler.choose({}, iteration=33) is None
    assert scheduler.choices == 32
    assert scheduler.choose({"delivery": (9,)}, iteration=34) == ("delivery", 9)


@pytest.mark.parametrize("seconds,targets,pairs", [(float("nan"), 0, 0), (-1, 0, 0), (0, 2, 1)])
def test_invalid_measurements(seconds, targets, pairs):
    with pytest.raises(ValueError):
        YieldFrontier().observe(1, "ridge", seconds, targets, pairs)


def test_zero_cost_floor():
    scheduler = YieldFrontier()
    scheduler.observe(1, "ridge", 0, 1, 1)
    assert 0 < scheduler.score(1, "ridge") < 3000


def test_measurement_preserves_legacy_work_and_does_not_consume(monkeypatch):
    bank = FacetCorrectorBank(np.zeros((64, 26)), seed=1, endpoint_dedup=True, yield_frontier=True)
    anchor = SimpleNamespace(word=100)
    bank.shared.coordinates[100] = (1, (), ())
    bank.shared.pools[1] = []
    bank.shared.endpoints.pairs.add((1, 2))
    monkeypatch.setattr(bank, "frontier_lanes", lambda word: ("ridge",))
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr("mcts.group_game.facet_corrector.time.perf_counter", lambda: next(ticks))
    plan = object()

    def work(current):
        assert current is anchor
        bank.shared.endpoints.pairs.update(((1, 1), (1, 3)))
        return plan

    monkeypatch.setattr(bank, "_grow_systematic", work)
    assert bank._measured_work(anchor, "ridge") is plan
    measured = bank.yield_scheduler.estimates[(1, "ridge")]
    assert (measured.observations, measured.seconds, measured.new_targets, measured.new_pairs) == (1, 0.25, 1, 1)
    assert not bank.shared.exit_uses
    assert not bank.shared.endpoints.executions
    bank.yield_frontier = False
    assert bank._measured_work(anchor, "ridge") is plan
    assert measured.observations == 1


def test_configuration_modes():
    assert GroupGameConfig().yield_frontier
    assert not GroupGameConfig().fair_frontier
    assert build_parser().parse_args(["--fair-frontier"]).fair_frontier
    assert build_parser().parse_args(["--no-fair-frontier"]).no_fair_frontier
    assert build_parser().parse_args(["--no-yield-frontier"]).no_yield_frontier
    for dedup, fair in ((False, False), (True, True)):
        bank = FacetCorrectorBank(np.zeros((64, 26)), seed=1, endpoint_dedup=dedup,
                                 fair_frontier=fair, yield_frontier=True)
        assert not bank.yield_frontier
