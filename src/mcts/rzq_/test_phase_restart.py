"""Phase counts, shared children, Q source, and six immediate slots."""
import unittest
from types import SimpleNamespace
from mcts.search import MCTSNode, MCTSConfig, MCTSValueComponents
from mcts.rzq_.phase_restart import init_node, phase_backpropagate, phase_select_child, reset_phase, phase_child_limit


def node(i):
    n=MCTSNode(key=(i,),path=[] if i==0 else [i],rank=i,parent=None,action_from_parent=None,creation_index=i)
    init_node(n)
    return n

class PhaseRestartTests(unittest.TestCase):
    def test_real_edge_counts_with_shared_child(self):
        p,q,c=node(0),node(1),node(2)
        p.children[7]=c;q.children[8]=c
        p.action_priors[7]=1;q.action_priors[8]=1
        for parent,action,reward in ((p,7,10),(q,8,100),(q,8,100)):
            parent.selected_edge_action=action
            phase_backpropagate([parent,c],MCTSValueComponents(survival=reward),survival_weight=1,novelty_weight=0)
        self.assertEqual(c.visits,3)
        self.assertEqual(p.edge_phase_visits[7],1)
        self.assertEqual(q.edge_phase_visits[8],2)
        self.assertEqual(p.edge_survival_sums[7]/p.edge_phase_visits[7],10)
        self.assertEqual(q.edge_survival_sums[8]/q.edge_phase_visits[8],100)
        state=SimpleNamespace(nodes={0:p,1:q,2:c,3:c},root=p,config=MCTSConfig(),global_discovery=SimpleNamespace(discovered_exact_classes=set()))
        event=reset_phase(state,expansion=True)
        self.assertEqual(event['affected_nodes'],3)
        self.assertEqual(c.visits,3);self.assertEqual(c.phase_visits,0)
        p.selected_edge_action=7
        phase_backpropagate([p,c],MCTSValueComponents(survival=6),survival_weight=1,novelty_weight=0)
        self.assertEqual(c.visits,4);self.assertEqual(c.phase_visits,1)
        self.assertEqual(p.edge_total_visits[7],2);self.assertEqual(p.edge_phase_visits[7],1)
        self.assertEqual(p.edge_survival_sums[7],6)

    def test_selection_ignores_legacy_value_denominator(self):
        p,a,b=node(0),node(1),node(2)
        p.children={1:a,2:b};p.action_priors={1:.5,2:.5}
        p.visits=10;a.visits=b.visits=5
        p.edge_phase_visits={1:1,2:10};p.edge_survival_sums={1:10,2:20}
        p.edge_value_visits={1:1000.,2:.01}
        child=phase_select_child(p,1.4,use_real_node_visits=True,min_action_visits=1,novelty_weight=0)
        self.assertIs(child,a)
        self.assertEqual(p.selected_edge_action,1)

    def test_immediate_six_slots_and_phase_growth(self):
        p=node(0)
        p.children={i:node(i) for i in range(1,7)}
        p.unexpanded_actions=list(range(7,40));p.visits=10;p.phase_visits=10
        state=SimpleNamespace(nodes={0:p,**p.children},root=p,config=MCTSConfig(),global_discovery=SimpleNamespace(discovered_exact_classes=set()))
        self.assertEqual(phase_child_limit(p,state.config),6)
        reset_phase(state,expansion=True)
        self.assertEqual(phase_child_limit(p,state.config),12)
        self.assertEqual(p.phase_visits,0);self.assertEqual(p.visits,10)
        p.phase_visits=100
        self.assertEqual(phase_child_limit(p,state.config),17)
        reset_phase(state,expansion=True)
        self.assertEqual(phase_child_limit(p,state.config),12)

if __name__=='__main__':unittest.main()
