"""Check that closure is independent of symmetry quotienting in v4.1."""
import unittest
from types import SimpleNamespace

from mcts.rzq_.node_manager import RZQNodeManager
from mcts.search import MCTSConfig, _prepare_actions


class ClosureOnlyTests(unittest.TestCase):
    def test_closure_reuses_exact_state_but_not_symmetric_state(self):
        scorer = SimpleNamespace(
            block_symmetry_maps=((0, 1, 2), (1, 0, 2)),
            flat_closure=lambda key: ((1, 0, 1), (2,)) if key == (1, 0, 0) else (key, ()),
            affine_rank=lambda key: sum(key),
            refresh_node_compatibility=lambda node: None,
        )
        manager = RZQNodeManager(scorer, symmetry_dedup=False)
        nodes = {}
        def get(key):
            return manager.get_or_create(nodes, key, path=[], parent=None, action=None)
        first = get((1, 0, 0))
        self.assertEqual(first.key, (1, 0, 1))
        self.assertIs(first, get((1, 0, 1)))
        other = get((0, 1, 1))
        self.assertIsNot(first, other)
        self.assertEqual(len(nodes), 2)
        with self.assertRaises(ValueError):
            manager.merge_nodes(nodes, first, other)
        symmetric = RZQNodeManager(scorer)
        self.assertEqual(symmetric.canonicalize(first.key)[0], symmetric.canonicalize(other.key)[0])

    def test_all_actions_and_two_buckets_can_coexist(self):
        from mcts.search import MCTSNode
        node = MCTSNode(key=(0, 0, 0), path=[], rank=0, parent=None, action_from_parent=None)
        def prepare(node, temperature):
            node.expansion_scores = {a: 1.0 for a in node.action_families}
        scorer = SimpleNamespace(representative_action_families=None, prepare_expansion_priors=prepare)
        _prepare_actions(node, scorer, MCTSConfig())
        self.assertEqual(node.unexpanded_actions, [0, 1, 2])
        self.assertEqual(node.action_families, {0: (0,), 1: (1,), 2: (2,)})
        self.assertEqual(len(node.bucket_expansion_counts), 2)

    def test_expansion_prior_bucket_count_can_be_six(self):
        from mcts.search import MCTSNode
        node = MCTSNode(key=(0, 0), path=[], rank=0, parent=None, action_from_parent=None)
        def prepare(node, temperature):
            node.expansion_scores = {a: 1.0 for a in node.action_families}
        scorer = SimpleNamespace(
            representative_action_families=None,
            prepare_expansion_priors=prepare,
            expansion_bucket_count=6,
        )
        _prepare_actions(node, scorer, MCTSConfig())
        self.assertEqual(len(node.bucket_expansion_counts), 6)


if __name__ == '__main__':
    unittest.main()
