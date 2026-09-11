"""Exercise the all-actions-expanded branch even when the natural run misses it."""
import unittest
from types import SimpleNamespace
from mcts.search import MCTSConfig, MCTSValueComponents
from mcts.rzq_.test_phase_restart import node
from mcts.rzq_.sealed_restart import reset_sealed_phase, sealed_select_child, sealed_backpropagate, used_slots


def fixture():
    p=node(0);p.visits=p.phase_visits=6
    p.children={i:node(i) for i in range(1,7)}
    p.actions_initialized=True
    for a,c in p.children.items():
        c.visits=c.phase_visits=1
        p.edge_total_visits[a]=p.edge_phase_visits[a]=1
        p.action_priors[a]=(7-a)/21
    state=SimpleNamespace(nodes={0:p,**p.children},root=p,config=MCTSConfig(),
                          global_discovery=SimpleNamespace(discovered_exact_classes=set()))
    reset_sealed_phase(state)
    return state,p

class SealedTests(unittest.TestCase):
    def test_six_distinct_old_edges_despite_large_positive_q(self):
        state,p=fixture();chosen=[]
        for _ in range(6):
            c=sealed_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1)
            chosen.append(p.selected_edge_action)
            sealed_backpropagate([p,c],MCTSValueComponents(survival=100+p.selected_edge_action),survival_weight=1,novelty_weight=0)
        self.assertEqual(chosen,[1,2,3,4,5,6])
        self.assertEqual(used_slots(p),6)
        self.assertFalse(p.sealed_edges)
        self.assertEqual(p.phase_visits,6)
        sealed_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1)
        self.assertEqual(p.selected_edge_action,6) # normal Q competition resumes
        self.assertEqual(p.special_slots_used,6)

    def test_new_edges_already_consumed_budget(self):
        state,p=fixture()
        for i in range(7,12):
            p.children[i]=node(i)
            p.children[i].visits=1
            p.action_priors[i]=.01
            p.edge_phase_visits[i]=1
            p.edge_survival_sums[i]=1000
        c=sealed_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1)
        self.assertEqual(p.selected_edge_action,1)
        self.assertEqual(used_slots(p),6)
        sealed_backpropagate([p,c],MCTSValueComponents(survival=50),survival_weight=1,novelty_weight=0)
        sealed_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1)
        self.assertIn(p.selected_edge_action,range(7,12))
        self.assertEqual(p.special_slots_used,1)

    def test_shared_child_does_not_unseal_other_parent(self):
        state,p=fixture()
        q=node(9);q.children={1:p.children[1]};q.edge_total_visits={1:1};q.action_priors={1:1}
        state.nodes[9]=q
        reset_sealed_phase(state)
        child=sealed_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1)
        sealed_backpropagate([p,child],MCTSValueComponents(survival=100),survival_weight=1,novelty_weight=0)
        self.assertNotIn(1,p.sealed_edges)
        self.assertIn(1,q.sealed_edges)

if __name__=='__main__':unittest.main()
