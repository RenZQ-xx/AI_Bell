"""v11: v4.4 plus real phase-edge Q and discovery-triggered D restarts."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from audit_environment import HERE, ROOT, snapshot
from mcts.interrupt_search import InterruptSearchConfig, InterruptGlobalDiscoveryState
from mcts.decision_trace import set_trace_sink, emit
from mcts.rzq_.run_class8_v4_4 import v4_4_task_factory
from mcts.rzq_.phase_restart import prepare_task, nodes, tree_snapshot, dump_task
from mcts.rzq_.sealed_restart import sealed_policy, reset_sealed_phase, sealed_child_limit, capacity, used_slots
from mcts.rzq_.run_class8_phase_restart_paired import write_json
from mcts.rzq_.analyze_phase_restart_paired import graph_depth, distribution

PRE=[44,43,23,38,41,46,45,35,39]


def snapshot_v11(state):
    result=tree_snapshot(state)
    lookup={n.creation_index:n for n in nodes(state)}
    for row in result['node_statistics']:
        n=lookup[row['id']]
        row['effective_limit']=sealed_child_limit(n,state.config)
        row['can_expand']=bool(n.unexpanded_actions) and len(n.children)<row['effective_limit']
        if hasattr(n,'sealed_edges'):
            row.update(sealed_edges=sorted(n.sealed_edges),special_slots_used=n.special_slots_used,
                       stage_capacity=capacity(n),stage_slots_used=used_slots(n))
    result['graph_depth']=graph_depth(result)
    return result


def run(output, *, cap=None):
    output.mkdir(parents=True,exist_ok=False)
    cfg=InterruptSearchConfig(initial_class_id=1,iterations=300,seed=20260502,rare_target_classes=tuple(range(1,47)))
    shared=InterruptGlobalDiscoveryState.from_config(cfg)
    for c in PRE:shared.mark_discovered(c)
    task=prepare_task(v4_4_task_factory(cfg,shared)(8,29,27))
    env=snapshot()
    env['rzq_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    env['started_utc']=datetime.now(timezone.utc).isoformat()
    write_json(output/'environment.json',env)
    meta={'version':'v11','based_on':'v4.4','restart_policy':'D/sealed_parent_edges',
          'pre_discovered_class_ids':PRE,'seed':task.state.config.seed,'base_seed':20260502,
          'selection_rng_seed':task.state.config.seed+1009,'rollout_tie_rng_seed':task.state.config.seed+2147483647,
          'patience':300,'smoke_cap':cap,'initial_restart':False,'trigger':'first_global_exact_class_discovery',
          'timing':'global set updated in discovery callback; restart after iteration backpropagation',
          'multiple_discoveries':'one restart per iteration, all classes registered',
          'scope':'sole retained class8 graph, unique nodes once',
          'config':asdict(task.state.config),'interrupt_config':asdict(cfg),
          'q':'phase edge survival sum / integer phase edge traversals, zero if unvisited',
          'exploration':'1.4*prior*sqrt(parent.total_visits)/(1+child.total_visits)',
          'restart_slots':'immediately 6; cumulative capacity max(6,K(phase_visits))',
          'slot_usage':'new parent edges + special selections',
          'special_selection':'no unexpanded actions, slots remain: exploration-only over sealed old parent edges',
          'normal_selection':'all children when budget exhausted or no sealed candidates',
          'bucket_rotation_preserved':True,'expansion_bucket_count':6,
          'epoch_value_decay_disabled_for_real_phase_Q':True}
    write_json(output/'config.json',meta)
    counts=Counter();stage_counts=Counter();stage_root=Counter();stage_exact=Counter()
    discoveries=[];stages=[];last_new=0;phase=0;stage_start=0;iteration=0
    stage_node_count=len(task.state.nodes);by_path={tuple(n.path):n for n in nodes(task.state)}
    started=time.monotonic()
    stop_reason='running'
    with (output/'decisions.jsonl').open('w',encoding='utf-8') as stream, (output/'restart_events.jsonl').open('w',encoding='utf-8') as resets:
        def sink(record):
            record=dict(record,phase=phase)
            event=record['event'];counts[event]+=1;stage_counts[event]+=1
            if event in ('expand_action','ucb_choice'):
                path=tuple(record.get('parent_path',record.get('node_path',[])))
                if path not in by_path:by_path.update({tuple(n.path):n for n in nodes(task.state)})
                n=by_path[path]
                record.update(parent_id=n.creation_index,parent_total_visits=n.visits,parent_phase_visits=n.phase_visits)
                if event=='expand_action':
                    record.update(children_before=len(n.children),effective_limit=sealed_child_limit(n,task.state.config),
                                  stage_capacity=capacity(n) if hasattr(n,'sealed_edges') else None,
                                  stage_slots_used=used_slots(n) if hasattr(n,'sealed_edges') else None)
                else:
                    record['q_denominator_source']='integer_phase_edge_traversals'
                    for c in record['candidates']:
                        c['edge_phase_visits']=n.edge_phase_visits.get(c['action'],0)
                        c['edge_total_visits']=n.edge_total_visits.get(c['action'],0)
                        c['sealed']=c['action'] in getattr(n,'sealed_edges',set())
                        assert c['edge_phase_visits']==c['edge_value_visits']
            if event=='terminal_observation':stage_exact[record['label']]+=1
            if event=='real_visit_backpropagation':
                for e in record['edges']:
                    if e['parent_id']==task.state.root.creation_index:stage_root[e['action']]+=1
            stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
            if event=='restart':resets.write(json.dumps(record,ensure_ascii=False,default=str)+'\n');resets.flush()
        def close_stage(reason):
            tree=snapshot_v11(task.state)
            stages.append({'phase':phase,'start_iteration':stage_start+1,'end_iteration':iteration,
                           'length':iteration-stage_start,'end_reason':reason,'nodes_start':stage_node_count,
                           'nodes_end':tree['nodes'],'graph_depth_end':tree['graph_depth'],
                           'event_counts':dict(stage_counts),'root_edge_distribution':distribution(stage_root),
                           'terminal_distribution':dict(stage_exact)})
        set_trace_sink(sink)
        try:
            with sealed_policy():
                while iteration-last_new<300 and (cap is None or iteration<cap):
                    before=set(shared.discovered_exact_classes)
                    task.state.stop_reason=None
                    task.iteration_limit=max(task.iteration_limit,last_new+300)
                    fresh=task.step()
                    iteration=task.state.iterations_completed
                    new=sorted(shared.discovered_exact_classes-before)
                    assert new==sorted(d.class_id for d in fresh)
                    if new:
                        for d in fresh:
                            record={'iteration':iteration,'class':d.class_id,'phase':phase,'discovery':asdict(d)}
                            discoveries.append(record)
                            emit('new_class_discovery',**record,global_classes=sorted(shared.discovered_exact_classes))
                        close_stage('new_class')
                        prior=snapshot_v11(task.state)
                        reset_sealed_phase(task.state)
                        after=snapshot_v11(task.state)
                        assert prior['nodes']==after['nodes'] and prior['edges']==after['edges']
                        assert all(e['q']==0 for n in after['node_statistics'] for e in n['edges'])
                        emit('restart',iteration=iteration,classes=new,before=prior,after=after,
                             affected_nodes=after['nodes'],immediate_slots=6,clear_Q=True,total_visits_preserved=True,
                             scope='all_unique_retained_nodes',next_phase=phase+1)
                        last_new=iteration;phase+=1;stage_start=iteration;stage_node_count=after['nodes']
                        stage_counts.clear();stage_root.clear();stage_exact.clear()
                        (output/'checkpoint.pkl').write_bytes(dump_task(task))
                        write_json(output/'progress.json',{'iteration':iteration,'last_new':last_new,
                                   'restart_count':phase,'discoveries':discoveries,'stages':stages})
                    if iteration%25==0 or new:
                        stream.flush()
                        print(json.dumps({'iteration':iteration,'stale':iteration-last_new,'new':new,
                                          'restarts':phase,'nodes':len(task.state.nodes)}),flush=True)
                stop_reason='300_consecutive_iterations_without_new_class' if iteration-last_new==300 else 'smoke_cap'
                if iteration>stage_start:close_stage(stop_reason)
        finally:
            set_trace_sink(None)
    task.state.stop_reason=stop_reason
    tree=snapshot_v11(task.state)
    root=next(n for n in tree['node_statistics'] if n['id']==tree['root_id'])
    assert root['total_visits']==iteration
    assert root['phase_visits']==iteration-last_new
    assert sum(s['length'] for s in stages)==iteration
    assert len({d['class'] for d in discoveries})==len(discoveries)
    assert len(stages)==phase+1 or (cap is not None and iteration==last_new)
    (output/'final_snapshot.pkl').write_bytes(dump_task(task))
    payload={'meta':meta,'result':task.result().to_dict(),'iterations':iteration,'stop_reason':stop_reason,
             'stale_iterations':iteration-last_new,'last_new_iteration':last_new,'restart_count':phase,
             'discoveries':discoveries,'global_classes':sorted(shared.discovered_exact_classes),
             'stages':stages,'event_counts':dict(counts),'tree':tree,'elapsed_seconds':time.monotonic()-started,
             'verification':{'root_total_matches_iterations':True,'root_phase_matches_stale':True,
                             'stage_lengths_sum_to_iterations':True,'unique_global_discoveries':True}}
    write_json(output/'result.json',payload)
    lines=['# v11 正式实验' if cap is None else '# v11 冒烟实验','',
           f"停止原因：{stop_reason}；共{iteration}轮，重启{phase}次，终点连续{iteration-last_new}轮无新发现。",'',
           f"新class：`{[(d['class'],d['iteration']) for d in discoveries]}`。",'',
           f"节点{tree['nodes']}，父边{tree['edges']}，图深度{tree['graph_depth']}；特殊selection共{counts['sealed_selection']}次。",'',
           f"Exact/terminal分布：`{dict(task.state.encountered)}`。",'',
           '| 阶段 | 起止轮次 | 长度 | 节点数 | 深度 | expansion | 特殊selection | 普通selection | 根最大占比 |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in stages:
        e=s['event_counts'];lines.append(f"| {s['phase']} | {s['start_iteration']}–{s['end_iteration']} | {s['length']} | {s['nodes_end']} | {s['graph_depth_end']} | {e.get('expand_action',0)} | {e.get('sealed_selection',0)} | {e.get('ucb_choice',0)} | {s['root_edge_distribution']['max_share']:.1%} |")
    lines+=['','配置在config.json，源码与环境指纹在environment.json。所有决策及首次发现、重启在decisions.jsonl；restart_events.jsonl单独保留重启前后完整统计。阶段终点与精确分类分布在result.json，checkpoint.pkl保留最近一次重启后的完整状态，final_snapshot.pkl保存终点。',
            '', '本次是单seed正式运行，没有匹配的无重启正式对照，不能仅凭发现数量推断总体能力提升。原v4.4预发现集合与Q分母不同；此前ABC/D仅60轮单次重启短测，也不构成相同预算对照。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(output),'iterations':iteration,'restarts':phase,'new_classes':[d['class'] for d in discoveries]}),flush=True)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'runs/class8_v11')
    p.add_argument('--cap',type=int,help='Smoke-test cap only; omit for formal run')
    a=p.parse_args();run(a.output,cap=a.cap)
