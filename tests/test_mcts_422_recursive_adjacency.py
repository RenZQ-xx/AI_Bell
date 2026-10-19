from __future__ import annotations

import copy
import importlib
import random
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

adj = importlib.import_module("mcts.422.recursive_adjacency")
search = importlib.import_module("mcts.422.reduced_adjacency_search")
multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
bench = importlib.import_module("mcts.422.adjacency_benchmark")


class ReportingOracle:
    def __init__(self):
        self.calls = []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(False, "fixture", word)

    def to_dict(self):
        return {"calls": len(self.calls)}


def problem(points):
    points = np.asarray(points, dtype=float)
    count = len(points)
    return reduced.ReducedSubgroupProblem(
        "fixture", "fixture", points.shape[1], points, points,
        tuple(range(count)), tuple(range(count)), tuple(1 << i for i in range(count)),
        count, Path("fixture"))


def test_leaf_all_square_edges_and_no_joggled_geometry():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=2)))
    facets = geometry.enumerate_leaf(15)
    assert {facet.word for facet in facets} == {3, 5, 10, 12}
    assert geometry.enumerate_leaf(15) is facets
    assert geometry.stats["leaf_cache_hits"] == 1
    assert not geometry.diagnostics()["formal_enumeration_completeness_certificate"]


def test_coplanar_qhull_triangles_close_to_six_cube_faces():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=3)))
    facets = geometry.enumerate_leaf(255)
    assert {facet.word for facet in facets} == {15, 240, 51, 204, 85, 170}
    assert all(facet.word.bit_count() == 4 for facet in facets)


def test_simplex_leaf_without_qhull():
    geometry = adj.RecursiveAdjacency([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert {f.word for f in geometry.enumerate_leaf(15)} == {7, 11, 13, 14}


def test_degenerate_qhull_triangulations_are_not_new_facets():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=4)))
    facets = geometry.enumerate_leaf(65535)
    assert len(facets) == 8
    assert all(facet.word.bit_count() == 8 for facet in facets)


def test_non_supporting_or_nonclosed_face_is_not_accepted():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=2)))
    with pytest.raises(adj.GeometryUncertain):
        geometry.certify(15, 9)
    with pytest.raises(adj.GeometryUncertain):
        geometry.certify(15, 1)
    with pytest.raises(adj.GeometryUncertain):
        geometry.certify(3, 4)


def test_cube_rotation_finds_opposite_incident_facet_and_caches():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=3)))
    word, proof = geometry.rotate(255, 15, 3)
    assert word == 51
    assert proof["ridge_word_hex"] == "0x3"
    assert geometry.rotate(255, 15, 3)[0] == 51
    assert geometry.stats["rotation_cache_hits"] == 1


def test_recursive_descent_and_return_have_real_parent_face_incidence():
    points = list(product((0, 1), repeat=4))
    geometry = adj.RecursiveAdjacency(points, adj.AdjacencyConfig(cutoff_vertices=4, max_macro_seconds=5))
    batch = geometry.sample(255, random.Random(4))
    assert batch.status == "chain_finished" and batch.leaf_enumeration_finished
    assert len(batch.chain) == 3 and batch.chain[-1].bit_count() == 4
    assert len(batch.outputs) == 4
    for proof in batch.outputs:
        assert adj.audit_adjacency_proof(points, proof)
        assert int(proof["destination_word_hex"], 16) != 255
        assert len(proof["rotations"]) == 2
    bad = copy.deepcopy(batch.outputs[0])
    bad["rotations"][-1]["destination_word_hex"] = hex(255)
    with pytest.raises(ValueError):
        adj.audit_adjacency_proof(points, bad)


def test_non_axis_aligned_integer_intrinsic_embedding():
    points = np.asarray(list(product((0, 1), repeat=3)))
    embedded = points @ np.asarray([[2, 1, 0, 3], [0, 3, 1, 2], [1, 0, 2, 4]]) + 7
    geometry = adj.RecursiveAdjacency(embedded)
    assert geometry.frame(255).dimension == 3
    assert geometry.rotate(255, 15, 3)[0] == 51


def test_deadline_and_output_limit_never_claim_adjacency_exhaustion():
    geometry = adj.RecursiveAdjacency(list(product((0, 1), repeat=3)),
                                     adj.AdjacencyConfig(max_outputs=1))
    batch = geometry.sample(15, random.Random(1))
    assert batch.status == "output_limit" and len(batch.outputs) == 1
    expired = geometry.sample(15, random.Random(1), deadline=time.perf_counter()-1)
    assert expired.status == "time_limit" and not expired.outputs
    assert not geometry.diagnostics()["finished_chain_implies_facet_adjacency_exhaustion"]


def test_config_rejects_unbounded_leaves_and_noninteger_points():
    for kwargs in ({"cutoff_vertices": 100}, {"max_macro_seconds": 0}, {"max_outputs": 0}):
        with pytest.raises(ValueError):
            adj.AdjacencyConfig(**kwargs)
    with pytest.raises(ValueError):
        adj.RecursiveAdjacency([[0.5], [1.5]])
    with pytest.raises(ValueError):
        search.ReducedAdjacencyConfig(adjacency_interval=0)


def test_disabled_hybrid_preserves_v6_actions_rewards_and_rng():
    p = problem(list(product((0, 1), repeat=3)))
    common = {"iterations": 8, "seed": 32, "progress_interval": 0}
    old = multi.ReducedMultiExitMCTS(p, multi.ReducedMultiExitConfig(**common),
                                   handoff_root=Path("fixture"), full_oracle=ReportingOracle())
    new = search.ReducedAdjacencyMCTS(p, search.ReducedAdjacencyConfig(
        **common, enable_recursive_adjacency=False), handoff_root=Path("fixture"), full_oracle=ReportingOracle())
    defaults = asdict(search.ReducedAdjacencyConfig())
    assert all(defaults[k] == v for k, v in asdict(multi.ReducedMultiExitConfig()).items())
    for i in range(1, 9):
        old.run_iteration(i)
        new.run_iteration(i)
    assert old.rng.getstate() == new.rng.getstate()
    assert old.discovered_reduced_facets == new.discovered_reduced_facets
    assert old.stats == new.stats
    assert old.reduced_witnesses == new.reduced_witnesses


def test_hybrid_proposals_do_not_discover_or_lift_unexecuted_candidates():
    p = problem(list(product((0, 1), repeat=3)))
    oracle = ReportingOracle()
    engine = search.ReducedAdjacencyMCTS(p, search.ReducedAdjacencyConfig(iterations=3, progress_interval=0),
                                       handoff_root=Path("fixture"), full_oracle=oracle)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    actions = engine._corrector_actions(node, count=3, rng=random.Random(1), persistent=True)
    assert len(actions) <= 3 and any(a.source == "recursive_as_ridge" for a in actions)
    assert not engine.discovered_reduced_facets and not oracle.calls
    for i in range(1, 4):
        engine.run_iteration(i)
    assert engine.discovered_reduced_facets
    assert oracle.calls
    assert bench.audit_reduced_paths(p, engine.result()["reduced_facets"]["witnesses"], Path("fixture"))["failure_count"] == 0
    for proof in engine.as_executed_proofs.values():
        assert adj.audit_adjacency_proof(p.irredundant_points, proof)


def test_as_only_revisits_sources_without_target_witness_or_mcts():
    p = problem(list(product((0, 1), repeat=3)))
    oracle = ReportingOracle()
    engine = search.ReducedAdjacencyExplorer(p, search.ReducedAdjacencyConfig(iterations=20),
                                           handoff_root=Path("fixture"), full_oracle=oracle)
    engine.started_at = time.perf_counter()
    for iteration in range(1, 21):
        engine.run_iteration(iteration)
    result = engine.result()
    assert len(engine.witnesses) == 6
    assert len(oracle.calls) == 6
    assert not result["prior_boundary"]["target_labels_affect_search"]
    assert bench.audit_reduced_paths(p, result["reduced_facets"]["witnesses"], Path("fixture"))["failure_count"] == 0
