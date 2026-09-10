"""Regression checks for the extracted compatibility calculation."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
import unittest

from baseline.orbit_blocks import add_block, build_orbit_patterns_from_support, empty_key
from baseline.reference_classes import parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScorer
from mcts.decision_trace import set_trace_sink
from mcts.search import MCTSConfig, MCTSNode, _prepare_actions, _rollout_top_k_pool, _select_child

from .compatibility import ClassCompatibilityIndex, project_support_to_blocks
from .rollout_scorer import RZQRolloutScorer, RolloutScoreConfig
from .symmetry_quotient import build_partition_block_maps


def _fake_scores(actions, scores):
    return [SimpleNamespace(action=action, score=score) for action, score in zip(actions, scores)]


class RolloutTieBreakingTests(unittest.TestCase):
    def test_all_tied_high_actions_can_enter_and_are_uniform(self) -> None:
        actions = list(range(12))
        counts = {action: 0 for action in actions}
        for seed in range(2400):
            pool, _trace = _rollout_top_k_pool(
                _fake_scores(actions, [1.0] * len(actions)),
                4,
                tie_rng=random.Random(seed),
            )
            for item in pool:
                counts[item.action] += 1
        expected = 2400 * 4 / len(actions)
        self.assertGreater(counts[max(actions)], 0)
        self.assertTrue(all(abs(count - expected) < 0.12 * expected for count in counts.values()))

    def test_symmetric_state_action_frequencies_align_under_group_map(self) -> None:
        support = support_mask_from_row(parse_example_rows()[8][1])
        pattern = build_orbit_patterns_from_support(support, class_id=8, max_patterns=1)[0]
        block_maps = build_partition_block_maps(pattern.orbits)
        symmetry = next(block_map for block_map in block_maps if block_map[0] != 0)
        actions = [action for action in range(len(pattern.orbits)) if action != 0]
        mapped_actions = [action for action in range(len(pattern.orbits)) if action != symmetry[0]]
        left = {action: 0 for action in actions}
        right = {action: 0 for action in mapped_actions}
        for seed in range(2400):
            pool, _ = _rollout_top_k_pool(
                _fake_scores(actions, [0.0] * len(actions)),
                4,
                tie_rng=random.Random(seed),
            )
            mapped_pool, _ = _rollout_top_k_pool(
                _fake_scores(mapped_actions, [0.0] * len(mapped_actions)),
                4,
                tie_rng=random.Random(seed + 10_000),
            )
            for item in pool:
                left[item.action] += 1
            for item in mapped_pool:
                right[item.action] += 1
        expected = 2400 * 4 / len(actions)
        self.assertTrue(
            all(abs(left[action] - right[symmetry[action]]) < 0.3 * expected for action in actions)
        )

    def test_non_tied_cutoff_matches_stable_top_k(self) -> None:
        scored = _fake_scores([8, 2, 11, 4, 9], [2.0, 5.0, 1.0, 4.0, 3.0])
        expected = sorted(scored, key=lambda item: item.score, reverse=True)[:3]
        pool, trace = _rollout_top_k_pool(scored, 3, tie_rng=random.Random(99))
        self.assertEqual([item.action for item in pool], [item.action for item in expected])
        self.assertEqual(trace["boundary_tied_actions"], [9])


class CompatibilityTests(unittest.TestCase):
    def test_v3_minimum_visit_phase_uses_prior_for_least_visited_tie(self) -> None:
        key = (0, 0, 0)
        parent = MCTSNode(key, [], 0, None, None, visits=20)
        almost_ready = MCTSNode((1, 0, 0), [0], 1, parent, 0, visits=4)
        zero_prior = MCTSNode((0, 1, 0), [1], 1, parent, 1, visits=0)
        selected_by_prior = MCTSNode((0, 0, 1), [2], 1, parent, 2, visits=0)
        parent.children = {0: almost_ready, 1: zero_prior, 2: selected_by_prior}
        parent.action_priors = {0: 0.5, 1: 0.0, 2: 0.5}
        parent.edge_value_visits = {0: 4.0, 1: 1.0, 2: 1.0}
        parent.edge_survival_sums = {0: 400.0, 1: 100.0, 2: -100.0}

        chosen = _select_child(
            parent,
            1.4,
            rng=random.Random(7),
            use_real_node_visits=True,
            min_action_visits=5,
            normalize_edge_survival_q=True,
        )
        self.assertIs(chosen, selected_by_prior)
        self.assertEqual(parent.selected_edge_action, 2)

    def test_v3_normalized_ucb_uses_parent_edge_survival(self) -> None:
        key = (0, 0)
        parent = MCTSNode(key, [], 0, None, None, visits=100)
        higher_q = MCTSNode((1, 0), [0], 1, parent, 0, visits=5)
        lower_q = MCTSNode((0, 1), [1], 1, parent, 1, visits=5)
        parent.children = {0: higher_q, 1: lower_q}
        parent.action_priors = {0: 0.5, 1: 0.5}
        parent.edge_value_visits = {0: 5.0, 1: 5.0}
        parent.edge_survival_sums = {0: 500.0, 1: 450.0}

        chosen = _select_child(
            parent,
            1.4,
            rng=random.Random(7),
            use_real_node_visits=True,
            min_action_visits=5,
            normalize_edge_survival_q=True,
        )
        self.assertIs(chosen, higher_q)

    def test_v3_equal_edge_q_is_traced_as_one_half(self) -> None:
        key = (0, 0)
        parent = MCTSNode(key, [], 0, None, None, visits=20)
        left = MCTSNode((1, 0), [0], 1, parent, 0, visits=5)
        right = MCTSNode((0, 1), [1], 1, parent, 1, visits=5)
        parent.children = {0: left, 1: right}
        parent.action_priors = {0: 0.5, 1: 0.5}
        parent.edge_value_visits = {0: 5.0, 1: 5.0}
        parent.edge_survival_sums = {0: 50.0, 1: 50.0}
        trace: list[dict[str, object]] = []
        set_trace_sink(trace.append)
        try:
            _select_child(
                parent,
                1.4,
                rng=random.Random(7),
                use_real_node_visits=True,
                min_action_visits=5,
                normalize_edge_survival_q=True,
            )
        finally:
            set_trace_sink(None)
        event = trace[-1]
        self.assertEqual(event["selection_phase"], "normalized_ucb")
        self.assertTrue(all(item["normalized_q"] == 0.5 for item in event["candidates"]))

    def test_rzq_ucb_uses_real_parent_and_child_visits(self) -> None:
        root_key = (0, 0)
        parent = MCTSNode(root_key, [], 0, None, None, visits=100, value_visits=4.0)
        frequently_visited = MCTSNode((1, 0), [0], 1, parent, 0, visits=99)
        rarely_visited = MCTSNode((0, 1), [1], 1, parent, 1, visits=0)
        parent.children = {0: frequently_visited, 1: rarely_visited}
        parent.action_priors = {0: 0.5, 1: 0.5}
        # Deliberately reverse the edge value-visit ordering.  The two modes
        # must therefore choose different children when exploitation is zero.
        parent.edge_value_visits = {0: 0.0, 1: 99.0}

        self.assertIs(
            _select_child(parent, 1.0, use_real_node_visits=True),
            rarely_visited,
        )
        self.assertIs(
            _select_child(parent, 1.0, use_real_node_visits=False),
            frequently_visited,
        )

    def test_partial_block_support_is_incompatible(self) -> None:
        blocks = ((0, 1), (2, 3))
        self.assertEqual(project_support_to_blocks((1, 1, 0, 0), blocks), frozenset({0}))
        self.assertIsNone(project_support_to_blocks((1, 0, 0, 0), blocks))

    def test_class8_known_state_counts(self) -> None:
        support = support_mask_from_row(parse_example_rows()[8][1])
        pattern = build_orbit_patterns_from_support(support, class_id=8, max_patterns=1)[0]
        index = ClassCompatibilityIndex.build(pattern.orbits)
        key = list(empty_key(len(pattern.orbits)))
        for block in (0, 7, 10):
            key[block] = 1
        counts = index.counts(key)
        self.assertEqual((counts[7], counts[8], counts[9]), (6, 12, 4))
        self.assertEqual(sum(count > 0 for count in counts.values()), 24)
        self.assertEqual(sum(counts.values()), 168)

        base = ExpansionScorer(
            blocks=pattern.orbits,
            rare_target_classes=set(),
            target_classes=set(range(1, 47)),
        )
        rollout = RZQRolloutScorer(
            base,
            index,
            discovered_class_ids={1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46},
            config=RolloutScoreConfig(),
        )
        root = empty_key(len(pattern.orbits))
        root_families = rollout.representative_action_families(root)
        self.assertEqual(
            sorted(action for family in root_families.values() for action in family),
            list(range(len(pattern.orbits))),
        )
        self.assertTrue(all(rep == min(family) for rep, family in root_families.items()))
        self.assertLess(len(root_families), len(pattern.orbits))
        first_rep = min(root_families)
        child_key = tuple(int(index == first_rep) for index in range(len(pattern.orbits)))
        child_families = rollout.representative_action_families(child_key)
        self.assertTrue(all(rep == min(family) for rep, family in child_families.items()))
        self.assertEqual(
            sorted(action for family in child_families.values() for action in family),
            [index for index in range(len(pattern.orbits)) if index != first_rep],
        )

        flat_key = root
        for action in (0, 1, 2):
            flat_key = add_block(flat_key, action)
        self.assertGreater(base.flat_capacity(flat_key), 0)
        closed_key, closure_blocks = rollout.flat_closure(flat_key)
        self.assertEqual(closure_blocks, (24,))
        self.assertEqual(base.flat_capacity(closed_key), 0)

        nodes = {}
        managed_root = rollout.node_manager.get_or_create(
            nodes, root, path=[], parent=None, action=None
        )
        _prepare_actions(managed_root, rollout, MCTSConfig())
        managed_root_families = rollout.representative_action_families(managed_root.key)
        self.assertEqual(set(managed_root.expansion_scores), set(managed_root_families))
        self.assertEqual(managed_root.action_scores, {})
        self.assertAlmostEqual(sum(managed_root.action_priors.values()), 1.0)
        for action, details in managed_root.expansion_score_details.items():
            class_count = int(details["compatible_class_count"])
            mean_log = float(details["mean_log_compatible_masks"])
            self.assertAlmostEqual(
                managed_root.expansion_scores[action],
                2.0 * math.log1p(class_count) + mean_log,
            )
        rollout.expansion_prior_mode = "structural_score"
        structural_root = MCTSNode(root, [], 0, None, None)
        _prepare_actions(structural_root, rollout, MCTSConfig())
        self.assertEqual(structural_root.expansion_prior_source, "structural_score")
        self.assertAlmostEqual(sum(structural_root.action_priors.values()), 1.0)
        for action in structural_root.action_families:
            self.assertAlmostEqual(
                structural_root.expansion_scores[action],
                rollout.score_action(root, action).score,
            )
            self.assertEqual(
                structural_root.expansion_score_details[action]["prior_source"],
                "structural_score",
            )
        rollout.expansion_prior_mode = "compatibility_richness"
        symmetric_a = rollout.node_manager.get_or_create(
            nodes, add_block(root, 0), path=[0], parent=managed_root, action=0
        )
        symmetric_b = rollout.node_manager.get_or_create(
            nodes, add_block(root, 1), path=[1], parent=managed_root, action=1
        )
        self.assertIs(symmetric_a, symmetric_b)
        self.assertEqual(set(symmetric_a.arrival_paths), {(0,), (1,)})

        reordered_key = root
        for action in (2, 20, 12):
            reordered_key = add_block(reordered_key, action)
        reordered_a = rollout.node_manager.get_or_create(
            nodes, reordered_key, path=[2, 20, 12], parent=None, action=None
        )
        reordered_b = rollout.node_manager.get_or_create(
            nodes, reordered_key, path=[2, 12, 20], parent=None, action=None
        )
        self.assertIs(reordered_a, reordered_b)
        self.assertIn((2, 20, 12), reordered_a.arrival_paths)
        self.assertIn((2, 12, 20), reordered_a.arrival_paths)

        first_counts = reordered_a.refresh_compatibility(rollout, [7], 1)
        second_counts = reordered_a.refresh_compatibility(rollout, [7, 8], 2)
        self.assertEqual(first_counts[7], second_counts[7])
        self.assertIn(8, reordered_a.compatibility_counts)
        self.assertEqual(reordered_a.compatibility_epoch, 2)
        scored = rollout.score_action(root, 0)
        breakdown = rollout.breakdown(root, 0)
        self.assertAlmostEqual(
            scored.score,
            breakdown.rank_gain_score
            + breakdown.flat_score
            + breakdown.supportability_score
            + breakdown.decline_bonus,
        )
        self.assertLessEqual(breakdown.child_compat_total, breakdown.parent_compat_total)
        self.assertLessEqual(breakdown.child_compat_classes, breakdown.parent_compat_classes)
        self.assertEqual(breakdown.rank_gain_score, 5.0 if breakdown.rank_gain > 0 else -5.0)
        self.assertGreaterEqual(breakdown.decline_bonus, 0.0)
        self.assertTrue(all(value >= 0 for _class_id, value in breakdown.decline_distribution))

        no_decline = rollout.decline_components({7: 10}, {7: 10})
        moderate = rollout.decline_components({7: 10}, {7: 8})
        excessive = rollout.decline_components({7: 10}, {7: 0})
        bridge = rollout.decline_components({7: 1, 8: 1}, {7: 0, 8: 1})
        self.assertEqual(no_decline, (0.0, 0.0, False, 0.0, 0.0))
        self.assertAlmostEqual(moderate[0], 0.2)
        self.assertAlmostEqual(moderate[1], 1.0)
        self.assertLess(excessive[1], moderate[1])
        self.assertTrue(bridge[2])
        self.assertAlmostEqual(bridge[3], 0.5)
        self.assertAlmostEqual(bridge[4], 2.5)

        class44_mask = index.masks_by_class[44][0]
        class44_key = tuple(int(block in class44_mask) for block in range(index.block_count))
        self.assertEqual(base.terminal_label(class44_key).label, "exact:class44")
        self.assertEqual(rollout.terminal_score(class44_key), 10.0)

        whole_polytope = tuple(1 for _ in range(index.block_count))
        self.assertEqual(rollout.terminal_score(whole_polytope), -10.0)


if __name__ == "__main__":
    unittest.main()
