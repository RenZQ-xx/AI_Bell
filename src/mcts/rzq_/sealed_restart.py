"""D: spend restart exploration slots on new actions, then sealed old edges."""
from contextlib import contextmanager
import copy
from mcts import interrupt_search as engine
from mcts import search
from mcts.decision_trace import emit
from mcts.rzq_.phase_restart import (
    phase_policy, phase_select_child, phase_child_limit, phase_backpropagate,
    reset_phase, nodes,
)


def capacity(node):
    proxy=copy.copy(node)
    proxy.visits=node.phase_visits
    return max(6,search._progressive_child_limit(proxy,node.sealed_config))


def used_slots(node):
    return len(node.children)-node.expansion_phase_base+node.special_slots_used


def sealed_child_limit(node,cfg):
    if not hasattr(node,'sealed_edges'):
        return phase_child_limit(node,cfg)
    # Original expansion condition compares total children with this threshold.
    return node.expansion_phase_base+capacity(node)-node.special_slots_used


def sealed_select_child(node,exploration_constant,**kwargs):
    if hasattr(node,'sealed_edges') and not node.unexpanded_actions and used_slots(node)<capacity(node):
        eligible=[a for a in node.children if a in node.sealed_edges]
        if eligible:
            rows=[]
            for a in eligible:
                child=node.children[a]
                assert node.edge_phase_visits.get(a,0)==0
                e=exploration_constant*node.action_priors.get(a,0)*max(1,node.visits)**.5/(1+child.visits)
                rows.append({'action':a,'child_id':child.creation_index,'exploration':e,
                             'child_total_visits':child.visits,'edge_phase_visits':0})
            chosen=max(rows,key=lambda r:r['exploration'])['action']
            before=used_slots(node)
            node.sealed_edges.remove(chosen)
            node.special_slots_used+=1
            node.selected_edge_action=chosen
            emit('sealed_selection',parent_id=node.creation_index,node_path=list(node.path),
                 chosen_action=chosen,candidates=rows,excluded_actions=[a for a in node.children if a not in eligible],
                 capacity=capacity(node),slots_used_before=before,slots_used_after=used_slots(node),
                 remaining_sealed=sorted(node.sealed_edges),parent_total_visits=node.visits,
                 parent_phase_visits=node.phase_visits)
            return node.children[chosen]
        emit('sealed_candidates_exhausted',parent_id=node.creation_index,
             capacity=capacity(node),slots_used=used_slots(node),fallback='normal_selection')
    return phase_select_child(node,exploration_constant,**kwargs)


def sealed_backpropagate(path_nodes,value,**kwargs):
    traversed=[(p,int(p.selected_edge_action)) for p in path_nodes[:-1]]
    phase_backpropagate(path_nodes,value,**kwargs)
    # Normal selection may also revisit an old edge after the budget is spent.
    for parent,action in traversed:
        if hasattr(parent,'sealed_edges'):
            parent.sealed_edges.discard(action)


def reset_sealed_phase(state):
    event=reset_phase(state,expansion=True)
    for n in nodes(state):
        n.sealed_edges={a for a in n.children if n.edge_total_visits.get(a,0)>0}
        n.special_slots_used=0
        n.sealed_config=state.config
    event.update(mode='D',special_selection='exploration_only_among_sealed_parent_edges',
                 budget='new_edges + special_selections < max(6,K(phase_visits))',
                 sealed_edges={n.creation_index:sorted(n.sealed_edges) for n in nodes(state)},
                 no_sealed_candidates='fallback_to_normal_selection')
    return event


@contextmanager
def sealed_policy():
    with phase_policy():
        saved=(engine._select_child,engine._progressive_child_limit,engine._backpropagate_components)
        engine._select_child=sealed_select_child
        engine._progressive_child_limit=sealed_child_limit
        engine._backpropagate_components=sealed_backpropagate
        try:
            yield
        finally:
            engine._select_child,engine._progressive_child_limit,engine._backpropagate_components=saved
