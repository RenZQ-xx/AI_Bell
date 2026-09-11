"""User-defined phase restart, isolated from the historical v4.4 entry point.

Selection keeps the original PUCT implementation and total node visits, but
supplies independently counted integer parent-edge phase traversals for Q.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import replace
from functools import partial
import math

from mcts import interrupt_search as engine
from mcts import search
from mcts.decision_trace import emit
from mcts.rzq_.rollout_scorer import RZQRolloutScorer


def discovered_classes(global_state):
    return global_state.discovered_exact_classes


def discovery_epoch(global_state):
    return global_state.discovery_epoch


class SnapshotScorer(RZQRolloutScorer):
    """Make the inherited delegation safe during pickle reconstruction."""
    def __getattr__(self, name):
        base = self.__dict__.get('base_scorer')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


def prepare_task(task):
    task.state.config = replace(task.state.config, discovery_epoch_value_decay=1.0)
    scorer = task.state.scorer
    scorer.__class__ = SnapshotScorer
    scorer.configure_discovery_cache(
        partial(discovered_classes, task.state.global_discovery),
        partial(discovery_epoch, task.state.global_discovery),
    )
    for node in nodes(task.state):
        init_node(node)
    return task


def nodes(state):
    return list({id(n): n for n in state.nodes.values()}.values())


def init_node(node):
    if not hasattr(node, 'phase_visits'):
        # Only fresh, unvisited nodes may enter this experiment.
        assert node.visits == 0
        node.phase_visits = 0
        node.edge_phase_visits = {}
        node.edge_total_visits = {}
        node.expansion_phase_base = None


def phase_backpropagate(path_nodes, value, *, survival_weight, novelty_weight):
    for node in path_nodes:
        init_node(node)
    traversed = [(p, int(p.selected_edge_action), c) for p, c in zip(path_nodes, path_nodes[1:])]
    search._backpropagate_components(path_nodes, value, survival_weight=survival_weight, novelty_weight=novelty_weight)
    for node in path_nodes:
        node.phase_visits += 1
        assert node.value_visits == node.phase_visits
    updates = []
    for parent, action, child in traversed:
        assert parent.children[action] is child
        parent.edge_phase_visits[action] = parent.edge_phase_visits.get(action, 0) + 1
        parent.edge_total_visits[action] = parent.edge_total_visits.get(action, 0) + 1
        n = parent.edge_phase_visits[action]
        assert parent.edge_value_visits[action] == n
        updates.append({'parent_id': parent.creation_index, 'child_id': child.creation_index,
                        'action': action, 'phase_visits': n,
                        'total_visits': parent.edge_total_visits[action],
                        'q': parent.edge_survival_sums[action] / n})
    emit('real_visit_backpropagation', edges=updates,
         nodes=[{'id':n.creation_index,'total_visits':n.visits,'phase_visits':n.phase_visits} for n in path_nodes])


def phase_select_child(node, exploration_constant, **kwargs):
    init_node(node)
    proxy = copy.copy(node)
    # This adapter supplies actual edge traversal counters, never value weights.
    proxy.edge_value_visits = node.edge_phase_visits
    child = search._select_child(proxy, exploration_constant, **kwargs)
    node.selected_edge_action = proxy.selected_edge_action
    return child


def phase_child_limit(node, cfg):
    init_node(node)
    if node.expansion_phase_base is None:
        return search._progressive_child_limit(node, cfg)
    proxy = copy.copy(node)
    proxy.visits = node.phase_visits
    return node.expansion_phase_base + max(6, search._progressive_child_limit(proxy, cfg))


def candidates(node, cfg):
    init_node(node)
    rows = []
    for action, child in node.children.items():
        init_node(child)
        n = node.edge_phase_visits.get(action, 0)
        q = node.edge_survival_sums.get(action, 0) / n if n else 0.0
        e = cfg.exploration_constant * node.action_priors.get(action, 0) * math.sqrt(max(1, node.visits)) / (1 + child.visits)
        rows.append({'action':action,'child_id':child.creation_index,'q':q,'exploration':e,'ucb':q+e,
                     'edge_phase_visits':n,'edge_total_visits':node.edge_total_visits.get(action,0),
                     'child_total_visits':child.visits,'child_phase_visits':child.phase_visits})
    return rows


def tree_snapshot(state):
    rows = []
    for n in nodes(state):
        init_node(n)
        es = candidates(n, state.config)
        best = max(es, key=lambda e:e['ucb'])['action'] if es else None
        rows.append({'id':n.creation_index,'key':list(n.key),'path':list(n.path), 'rank':n.rank,
                     'total_visits':n.visits,'phase_visits':n.phase_visits,
                     'children':len(n.children),'unexpanded':len(n.unexpanded_actions),
                     'actions_initialized':n.actions_initialized,'terminal':n.is_terminal,
                     'original_k':search._progressive_child_limit(n,state.config),
                     'effective_limit':phase_child_limit(n,state.config),
                     'expansion_phase_base':n.expansion_phase_base,
                     'next_bucket':n.next_bucket,'bucket_expansion_counts':list(n.bucket_expansion_counts),
                     'can_expand':bool(n.unexpanded_actions) and len(n.children)<phase_child_limit(n,state.config),
                     'ucb_best_action':best,'edges':es})
    return {'nodes':len(rows),'edges':sum(r['children'] for r in rows),
            'root_id':state.root.creation_index,'global_classes':sorted(state.global_discovery.discovered_exact_classes),
            'node_statistics':rows}


def reset_phase(state, *, expansion):
    before = tree_snapshot(state)
    for node in nodes(state):
        node.phase_visits = 0
        node.edge_phase_visits = {a:0 for a in node.children}
        node.value_visits = 0.0
        for name in ('value_sum','escape_sum','survival_sum','novelty_sum'):
            setattr(node,name,0.0)
        for name in ('edge_value_visits','edge_value_sums','edge_escape_sums','edge_survival_sums','edge_novelty_sums'):
            setattr(node,name,{a:0.0 for a in node.children})
        if expansion:
            node.expansion_phase_base = len(node.children)
    after = tree_snapshot(state)
    for a,b in zip(before['node_statistics'],after['node_statistics']):
        assert a['total_visits']==b['total_visits']
        assert a['children']==b['children']
        assert a['next_bucket']==b['next_bucket']
        for x,y in zip(a['edges'],b['edges']):
            assert x['exploration']==y['exploration'] and y['q']==0
            assert x['edge_total_visits']==y['edge_total_visits']
    return {'before':before,'after':after,'affected_nodes':len(after['node_statistics']),
            'clear_q':True,'total_visits_preserved':True,'immediate_slots':6 if expansion else 0,
            'q_denominator':'integer phase parent-edge traversals',
            'expansion_rule':'old_child_count + max(6, K(phase_visits))' if expansion else 'original total-visits rule'}


@contextmanager
def phase_policy():
    saved = (engine._select_child, engine._backpropagate_components, engine._progressive_child_limit)
    engine._select_child = phase_select_child
    engine._backpropagate_components = phase_backpropagate
    engine._progressive_child_limit = phase_child_limit
    try:
        yield
    finally:
        engine._select_child, engine._backpropagate_components, engine._progressive_child_limit = saved


def dump_task(task):
    """Serialize weak cache entries by value while preserving graph aliases."""
    import pickle
    cache = task.state.global_discovery.structure_cache
    active = cache._active_partitions
    try:
        cache._active_partitions = dict(active.items())
        return pickle.dumps(task, protocol=5)
    finally:
        cache._active_partitions = active


def load_task(blob):
    import pickle
    import weakref
    task = pickle.loads(blob)
    cache = task.state.global_discovery.structure_cache
    cache._active_partitions = weakref.WeakValueDictionary(cache._active_partitions)
    return task
