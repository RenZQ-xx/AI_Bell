"""Structural restart invariants and verification of real smoke snapshots."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from audit_environment import HERE
from mcts.search import MCTSNode, MCTSConfig
from mcts.rzq_.run_class8_v4_4_restart import restart, restart_limit

class RestartTests(unittest.TestCase):
    def test_shared_node_discount_and_widening(self):
        root=MCTSNode(key=(0,),path=[],rank=0,parent=None,action_from_parent=None,visits=100,creation_index=1)
        a=MCTSNode(key=(1,),path=[0],rank=1,parent=root,action_from_parent=0,visits=80,creation_index=2)
        b=MCTSNode(key=(2,),path=[1],rank=1,parent=root,action_from_parent=1,visits=20,creation_index=3)
        root.children={0:a,1:b}; root.action_priors={0:.5,1:.5}
        root.edge_value_visits={0:80.,1:20.};root.edge_survival_sums={0:8000.,1:400.}
        root.unexpanded_actions=list(range(2,30))
        state=SimpleNamespace(nodes={'r':root,'a':a,'alias':a,'b':b},config=MCTSConfig())
        r=restart(state,100,[8])
        self.assertEqual(r['affected_nodes'],3)
        self.assertEqual(a.visits,20)
        self.assertEqual(root.visits,25)
        self.assertEqual(root.visits+root._restart_visit_offset,100)
        self.assertGreaterEqual(restart_limit(root,state.config),8)
        self.assertEqual(root.edge_value_visits[0],20.)
        self.assertAlmostEqual(root.edge_survival_sums[0]/20,100*r['lambda_Q_ratio'])
        es=r['after']['node_statistics'][0]['edges']
        self.assertLessEqual(abs(es[0]['q']-es[1]['q']),.25*abs(es[0]['exploration']-es[1]['exploration'])+1e-12)
        restart(state,101,[15])
        self.assertEqual(root.visits+root._restart_visit_offset,100)

    def test_real_smoke(self):
        path=HERE/'runs/class8_v4_4_restart_smoke_verified'
        events=[json.loads(x) for x in (path/'decisions.jsonl').read_text().splitlines()]
        restarts=[e for e in events if e['event']=='restart']
        self.assertGreaterEqual(len(restarts),1)
        reopened=0; boosted=0
        for r in restarts:
            for pre,post in zip(r['before']['node_statistics'],r['after']['node_statistics']):
                self.assertEqual(pre['lifetime_visits'],post['lifetime_visits'])
                if pre['unexpanded'] and pre['children']>=pre['k']:
                    self.assertGreater(post['k'],post['children']);reopened+=1
                for old,new in zip(pre['edges'],post['edges']):
                    self.assertAlmostEqual(new['q'],old['q']*r['lambda_Q_ratio'])
                    boosted+=new['exploration']>old['exploration']
                es=post['edges']
                if len(es)>1:
                    q=max(e['q'] for e in es)-min(e['q'] for e in es)
                    x=max(e['exploration'] for e in es)-min(e['exploration'] for e in es)
                    if x>0:self.assertLessEqual(q,.25*x+1e-12)
        self.assertGreater(reopened,0)
        self.assertGreater(boosted,0)
        print({'smoke_restarts':len(restarts),'reopened_saturated_nodes':reopened,'exploration_increased_edges':boosted})

if __name__=='__main__':unittest.main()
