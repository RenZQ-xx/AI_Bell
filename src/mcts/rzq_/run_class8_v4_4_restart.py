"""Isolated v4.4 restart experiment; no changes to baseline search modules."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter

from audit_environment import HERE, ROOT, snapshot
import mcts.interrupt_search as engine
from mcts.search import _progressive_child_limit as baseline_limit
from mcts.decision_trace import set_trace_sink
from mcts.rzq_.run_class8_v4_4 import v4_4_task_factory

PRE = [44, 43, 23, 38, 41, 46, 45, 35, 39]

def unique_nodes(state):
    return list({id(n): n for n in state.nodes.values()}.values())

def restart_limit(node, cfg):
    proxy = copy.copy(node)
    proxy.visits = node.visits + getattr(node, '_restart_visit_offset', 0)
    return max(baseline_limit(proxy, cfg), getattr(node, '_restart_child_floor', 0))

def edge_stats(node, cfg):
    result = []
    for a, child in node.children.items():
        n = node.edge_value_visits.get(a, 0)
        q = node.edge_survival_sums.get(a, 0) / n if n else 0
        e = cfg.exploration_constant * node.action_priors.get(a, 0) * math.sqrt(max(1, node.visits)) / (1 + child.visits)
        result.append({'action': a, 'q': q, 'exploration': e, 'child_visits': child.visits, 'edge_value_visits': n})
    return result

def summary(state):
    nodes = unique_nodes(state)
    rows = []
    for n in nodes:
        es = edge_stats(n, state.config)
        rows.append({'id': n.creation_index, 'path': n.path, 'visits': n.visits,
                     'lifetime_visits': n.visits + getattr(n, '_restart_visit_offset', 0),
                     'children': len(n.children), 'unexpanded': len(n.unexpanded_actions),
                     'k': restart_limit(n, state.config), 'edges': es})
    return {'nodes': len(nodes), 'edges': sum(len(n.children) for n in nodes),
            'max_stored_path_depth': max(len(n.path) for n in nodes), 'node_statistics': rows}

def restart(state, iteration, classes):
    nodes = unique_nodes(state)
    before = summary(state)
    for n in nodes:
        old = n.visits
        n.visits = max(1, math.ceil(old * .25)) if old else 0
        n._restart_visit_offset = getattr(n, '_restart_visit_offset', 0) + old - n.visits
        if n.unexpanded_actions:
            n._restart_child_floor = max(getattr(n, '_restart_child_floor', 0),
                                        len(n.children) + min(6, len(n.unexpanded_actions)))
    # Calibrate a global shrinkage against the actual post-discount PUCT terms.
    # Nonconstant sibling Q ranges are <= 1/4 of nonconstant exploration ranges.
    rho = .01
    for n in nodes:
        es = edge_stats(n, state.config)
        if len(es) > 1:
            qs = max(e['q'] for e in es) - min(e['q'] for e in es)
            xs = max(e['exploration'] for e in es) - min(e['exploration'] for e in es)
            if qs > 0 and xs > 0:
                rho = min(rho, .25 * xs / qs)
    for n in nodes:
        n.value_visits *= .25
        for field in ('value_sum', 'escape_sum', 'survival_sum', 'novelty_sum'):
            setattr(n, field, getattr(n, field) * .25 * rho)
        for a in n.edge_value_visits:
            n.edge_value_visits[a] *= .25
        for field in ('edge_value_sums', 'edge_escape_sums', 'edge_survival_sums', 'edge_novelty_sums'):
            mapping = getattr(n, field)
            for a in mapping:
                mapping[a] *= .25 * rho
    after = summary(state)
    return {'event': 'restart', 'iteration': iteration, 'classes': classes,
            'lambda_N': .25, 'lambda_Q_ratio': rho, 'lambda_S': .25 * rho,
            'visit_rounding': 'ceil; visited nodes floor 1', 'expansion_slots': 6,
            'scope': 'all retained nodes of sole class8 tree; unique object once',
            'affected_nodes': len(nodes), 'before': before, 'after': after}

def run(output, *, mode='restart', patience=300, cap=None, seed=20260502):
    output.mkdir(parents=True, exist_ok=False)
    config = engine.InterruptSearchConfig(initial_class_id=1, iterations=300, seed=seed, rare_target_classes=tuple(range(1,47)))
    global_state = engine.InterruptGlobalDiscoveryState.from_config(config)
    for c in PRE:
        global_state.mark_discovered(c)
    task = v4_4_task_factory(config, global_state)(8,29,27)
    env = snapshot()
    env['rzq_source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (output/'environment.json').write_text(json.dumps(env, indent=2))
    meta = {'version': 'v4.4-restart-1', 'mode': mode, 'pre_discovered_class_ids': PRE,
            'base_seed': seed, 'seed': task.state.config.seed, 'patience': patience,
            'smoke_cap': cap, 'mcts_config': asdict(task.state.config),
            'interrupt_config': asdict(config), 'bucket_count': task.state.scorer.expansion_bucket_count}
    (output/'config.json').write_text(json.dumps(meta, indent=2, default=str))
    iteration = last_new = 0
    discoveries, restarts, iteration_rows = [], [], []
    counts = Counter()
    current = []
    old_limit = engine._progressive_child_limit
    start = time.monotonic()
    with (output/'decisions.jsonl').open('w') as stream:
        def sink(event):
            if event['event'] == 'expand_action':
                parent = next(n for n in unique_nodes(task.state) if n.path == event['parent_path'])
                proxy = copy.copy(parent)
                proxy.visits += getattr(parent, '_restart_visit_offset', 0)
                event = dict(event, parent_id=parent.creation_index,
                             children_before=len(parent.children),
                             ordinary_k=baseline_limit(parent, task.state.config),
                             lifetime_k=baseline_limit(proxy, task.state.config),
                             restart_k=restart_limit(parent, task.state.config),
                             restart_granted=(mode == 'restart' and len(parent.children) >= baseline_limit(proxy, task.state.config)))
            stream.write(json.dumps(event, default=str) + '\n')
            counts[event['event']] += 1
            current.append(event)
        set_trace_sink(sink)
        if mode == 'restart':
            engine._progressive_child_limit = restart_limit
        try:
            while iteration-last_new < patience and (cap is None or iteration < cap):
                iteration += 1
                current.clear()
                before_nodes = {id(n) for n in unique_nodes(task.state)}
                # Keep the exact v4.4 iteration kernel; the requested patience rule
                # replaces task.step's unrelated early-stop and budget policies.
                task.state.stop_reason = None
                task.iteration_limit = max(task.iteration_limit, last_new + patience)
                fresh = task.step()
                assert task.state.iterations_completed == iteration
                for d in fresh:
                    assert d.class_id in global_state.discovered_exact_classes
                    record = {'event':'new_class_discovery', 'iteration':iteration, 'class':d.class_id,
                              'discovery':asdict(d), 'global_classes':sorted(global_state.discovered_exact_classes)}
                    discoveries.append(record)
                    sink(record)
                expansions = [e for e in current if e['event']=='expand_action']
                terminals = [e for e in current if e['event']=='terminal_observation']
                visits = [e for e in current if e['event']=='node_visit']
                iteration_rows.append({'iteration':iteration,'phase':len(restarts),
                    'new_classes':[d.class_id for d in fresh], 'expansions':expansions,
                    'terminal':terminals, 'tree_path':visits[-1]['path'] if visits else [],
                    'nodes_added':sum(id(n) not in before_nodes for n in unique_nodes(task.state))})
                if fresh:
                    last_new = iteration
                    if mode == 'restart':
                        record = restart(task.state, iteration, [d.class_id for d in fresh])
                        restarts.append(record)
                        sink(record)
                if iteration % 25 == 0 or fresh:
                    stream.flush()
                    print(json.dumps({'iteration':iteration,'stale':iteration-last_new,'new':[d.class_id for d in fresh], 'nodes':len(task.state.nodes)}), flush=True)
        finally:
            engine._progressive_child_limit = old_limit
            set_trace_sink(None)
    final = summary(task.state)
    payload = {'meta':meta, 'result':task.result().to_dict(), 'iterations':iteration,
               'last_new_iteration':last_new,'stale_iterations':iteration-last_new,
               'stop_reason':'no_new_class_patience' if iteration-last_new == patience else 'smoke_cap',
               'global_classes':sorted(global_state.discovered_exact_classes), 'discoveries':discoveries,
               'restart_count':len(restarts), 'event_counts':dict(counts), 'elapsed_seconds':time.monotonic()-start,
               'tree':final}
    (output/'result.json').write_text(json.dumps(payload, indent=2, default=str))
    (output/'iterations.json').write_text(json.dumps(iteration_rows, indent=2, default=str))
    print(json.dumps({'output':str(output),'iterations':iteration,'restart_count':len(restarts),'classes':payload['global_classes']}), flush=True)
    return payload

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['restart','control'], default='restart')
    p.add_argument('--patience', type=int, default=300)
    p.add_argument('--cap', type=int)
    p.add_argument('--seed', type=int, default=20260502)
    a = p.parse_args()
    run(a.output, mode=a.mode, patience=a.patience, cap=a.cap, seed=a.seed)


