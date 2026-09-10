"""100-iteration common snapshot; A/B/C paired 60-iteration diagnostic."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

from audit_environment import HERE, ROOT, snapshot
from mcts.interrupt_search import InterruptSearchConfig, InterruptGlobalDiscoveryState
from mcts.decision_trace import set_trace_sink, emit, set_trace_iteration
from mcts.rzq_.run_class8_v4_4 import v4_4_task_factory
from mcts.rzq_.phase_restart import prepare_task, phase_policy, tree_snapshot, reset_phase, nodes, phase_child_limit, dump_task, load_task
from mcts.search import _progressive_child_limit

PRE = [44,43,23,38,41,46,45,35,39]

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')

def run_steps(task,count,path,label):
    path.mkdir(parents=True,exist_ok=False)
    records=[]
    initial=task.state.iterations_completed
    by_path={tuple(n.path):n for n in nodes(task.state)}
    with (path/'decisions.jsonl').open('w',encoding='utf-8') as stream:
        def sink(record):
            record=dict(record,branch=label,relative_iteration=int(record.get('iteration',initial))-initial)
            if record['event'] in ('expand_action','ucb_choice'):
                key=tuple(record.get('parent_path',record.get('node_path',[])))
                node=by_path.get(key)
                if node is None:
                    by_path.update({tuple(n.path):n for n in nodes(task.state)})
                    node=by_path[key]
                record['parent_id']=node.creation_index
                record['parent_total_visits']=node.visits
                record['parent_phase_visits']=node.phase_visits
                if record['event']=='expand_action':
                    record.update(children_before=len(node.children),
                                  original_limit=_progressive_child_limit(node,task.state.config),
                                  effective_limit=phase_child_limit(node,task.state.config),
                                  restart_only_slot=len(node.children)>=_progressive_child_limit(node,task.state.config))
                else:
                    record['q_denominator_source']='integer_phase_edge_traversals'
                    for c in record['candidates']:
                        c['edge_phase_visits']=node.edge_phase_visits.get(c['action'],0)
                        c['edge_total_visits']=node.edge_total_visits.get(c['action'],0)
                        assert c['edge_value_visits']==c['edge_phase_visits']
            stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
            records.append(record)
        set_trace_sink(sink)
        try:
            for offset in range(1,count+1):
                task.state.stop_reason=None
                task.iteration_limit=max(task.iteration_limit,initial+count)
                fresh=task.step()
                assert task.state.iterations_completed==initial+offset
                for d in fresh:
                    emit('new_class_discovery',class_id=d.class_id,discovery=asdict(d),
                         global_classes=sorted(task.state.global_discovery.discovered_exact_classes),restart_triggered=False)
                if offset%10==0 or fresh:
                    stream.flush()
                    print(json.dumps({'branch':label,'step':offset,'iteration':task.state.iterations_completed,
                                      'new':[d.class_id for d in fresh],'nodes':len(task.state.nodes)}),flush=True)
        finally:
            set_trace_sink(None)
    task.state.stop_reason='paired_fixed_budget'
    write_json(path/'tree.json',tree_snapshot(task.state))
    write_json(path/'result.json',task.result().to_dict())
    return records


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    config=InterruptSearchConfig(initial_class_id=1,iterations=300,seed=20260502,rare_target_classes=tuple(range(1,47)))
    shared=InterruptGlobalDiscoveryState.from_config(config)
    for c in PRE:shared.mark_discovered(c)
    task=prepare_task(v4_4_task_factory(config,shared)(8,29,27))
    assert tree_snapshot(load_task(dump_task(task)).state) == tree_snapshot(task.state)
    env=snapshot()
    env['rzq_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    write_json(output/'environment.json',env)
    write_json(output/'config.json',{'warmup':100,'branch_iterations':60,'pre_discovered':PRE,
               'base_seed':20260502,'task_seed':task.state.config.seed,
               'selection_rng_seed':task.state.config.seed+1009,
               'rollout_tie_rng_seed':task.state.config.seed+2147483647,
               'config':asdict(task.state.config),'interrupt_config':asdict(config),
               'baseline_correction':'all arms use real phase-edge Q; historical epoch value decay disabled',
               'A':'no restart','B':'clear Q and phase visits, keep original widening',
               'C':'clear Q and phase visits, grant 6 immediate slots with phase widening',
               'automatic_restarts':False,'bucket_pointer_reset':False})
    start=time.monotonic()
    with phase_policy():
        run_steps(task,100,output/'warmup','warmup')
        original=tree_snapshot(task.state)
        blob=dump_task(task)
        (output/'snapshot.pkl').write_bytes(blob)
        digest=hashlib.sha256(blob).hexdigest()
        # Fresh unpickling reconstructs every mutable cache and provider target.
        clone=load_task(blob)
        assert tree_snapshot(clone.state)==original
        assert clone.state._rng.getstate()==task.state._rng.getstate()
        assert clone.state._rollout_tie_rng.getstate()==task.state._rollout_tie_rng.getstate()
        assert clone.state.global_discovery is not task.state.global_discovery
        assert clone.state.scorer._discovered_classes_provider() is clone.state.global_discovery.discovered_exact_classes
        assert clone.state.scorer.node_manager.scorer is clone.state.scorer
        assert not clone.state.scorer.node_manager.aliases
        assert {id(n) for n in nodes(clone.state)}.isdisjoint(id(n) for n in nodes(task.state))
        del clone
        write_json(output/'snapshot_manifest.json',{'sha256':digest,'bytes':len(blob),
                   'iteration':100,'tree_roundtrip_equal':True,'rng_roundtrip_equal':True,
                   'providers_rebound_by_pickle':True,'mutable_nodes_disjoint':True})
        del task
        gc.collect()
        arm_a_prefix=None
        for label in ('A','B','C'):
            task=load_task(blob)
            assert tree_snapshot(task.state)==original
            write_json(output/f'{label}_start.json',tree_snapshot(task.state))
            if label!='A':
                event=reset_phase(task.state,expansion=label=='C')
                write_json(output/f'{label}_restart.json',dict(event,event='restart',iteration=100,branch=label))
            records=run_steps(task,60,output/label,label)
            if label=='A':arm_a_prefix=[{k:v for k,v in e.items() if k!='branch'} for e in records if e['relative_iteration']<=10]
            del task,records
            gc.collect()
        task=load_task(blob)
        replay=run_steps(task,10,output/'A_replay10','A_replay10')
        replay=[{k:v for k,v in e.items() if k!='branch'} for e in replay]
        assert replay==arm_a_prefix,'Identical snapshot replay diverged'
        write_json(output/'verification.json',{'A_first10_trace_exact_replay':True,
                   'all_arms_start_from_same_snapshot':True,'snapshot_sha256':digest,
                   'elapsed_seconds':time.monotonic()-start,'completed':True})
    print(json.dumps({'output':str(output),'completed':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=HERE/'runs/class8_phase_restart_paired_100_60')
    run(p.parse_args().output)

