"""Add D to the existing matched snapshot without rerunning or overwriting ABC."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import json
from audit_environment import HERE, ROOT, snapshot
from mcts.rzq_.phase_restart import load_task, tree_snapshot
from mcts.rzq_.sealed_restart import reset_sealed_phase, sealed_policy, nodes, used_slots, capacity
from mcts.rzq_.run_class8_phase_restart_paired import run_steps, write_json
from mcts.rzq_.analyze_phase_restart_paired import summarize, graph_depth, compare_windows


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    blob=(base/'snapshot.pkl').read_bytes()
    digest=hashlib.sha256(blob).hexdigest()
    assert digest==json.loads((base/'snapshot_manifest.json').read_text())['sha256']
    task=load_task(blob)
    original=json.loads((base/'C_start.json').read_text())
    assert tree_snapshot(task.state)==original
    env=snapshot()
    env['rzq_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    write_json(output/'environment.json',env)
    cfg=json.loads((base/'config.json').read_text())
    cfg.update(branch='D',snapshot_source=str(base/'snapshot.pkl'),snapshot_sha256=digest,
               rule='expand while possible; otherwise exploration-only sealed selection; both consume stage slots',
               special_selection_descends_into_child=True,normal_selection_candidates='all_children',
               no_eligible_sealed_edge='normal_selection')
    write_json(output/'config.json',cfg)
    write_json(output/'D_start.json',tree_snapshot(task.state))
    restart=reset_sealed_phase(task.state)
    write_json(output/'D_restart.json',dict(restart,event='restart',iteration=100,branch='D'))
    with (output/'restart_events.jsonl').open('w') as f:f.write(json.dumps(dict(restart,event='restart',iteration=100,branch='D'))+'\n')
    with sealed_policy():
        trace=run_steps(task,60,output/'D','D')
    tree=tree_snapshot(task.state)
    write_json(output/'seal_end.json',[{'id':n.creation_index,'sealed':sorted(n.sealed_edges),
               'special_used':n.special_slots_used,'slots_used':used_slots(n),'capacity':capacity(n)}
               for n in nodes(task.state) if hasattr(n,'sealed_edges')])
    old=json.loads((base/'analysis.json').read_text())
    windows={str(n):summarize(trace,tree,original,1,n) for n in (10,30,60)}
    special=[e for e in trace if e['event']=='sealed_selection']
    ctrace=[json.loads(x) for x in (base/'C/decisions.jsonl').read_text().splitlines()]
    strip=lambda xs:[{k:v for k,v in e.items() if k!='branch'} for e in xs]
    equal=json.loads(json.dumps(strip(trace)))==strip(ctrace)
    result={'snapshot_sha256':digest,'iterations':60,'nodes':tree['nodes'],'depth':graph_depth(tree),
            'special_selection_count':len(special),'special_selections':special,
            'trace_identical_to_C_except_branch':equal,'windows':windows,
            'comparisons':{a:{str(n):compare_windows(old['arms'][a]['windows'][str(n)],windows[str(n)]) for n in (10,30,60)} for a in 'ABC'}}
    write_json(output/'analysis.json',result)
    assert task.state.root.visits==160 and task.state.root.phase_visits==60
    assert sum(e['event']=='real_visit_backpropagation' for e in trace)==60
    write_json(output/'verification.json',{'completed':True,'same_snapshot':True,'root_total':160,'root_phase':60,
               'trace_identical_to_C_except_branch':equal,'special_selection_count':len(special)})
    lines=['# D封印selection配对短测','',
           '从原100轮快照恢复，仅新增D的60轮；A/B/C数据保持不动。D保留C的即时6个名额；动作全部展开且有名额时，仅在封印旧父边中按探索项选择，每次消耗1个名额并解封。名额不足时正常selection；无封印候选时也正常selection。','',
           f'特殊selection触发 **{len(special)}次**。D与C的完整trace（去除分支标签）完全一致：**{equal}**。','',
           '| 分支 | 根最大访问占比 | 新根边 | 新class | 旧快照边遍历 |','|---|---:|---:|---|---:|']
    for a in 'ABCD':
        w=windows['60'] if a=='D' else old['arms'][a]['windows']['60']
        lines.append(f"| {a} | {w['root_actual']['max_share']:.1%} | {w['root_expansions']} | {[(d['class'],d['relative_iteration']) for d in w['new_classes']]} | {w['edge_traversals'].get('snapshot_edge',0)} |")
    lines+=['', '名额=新增父边数+特殊selection次数，容量=max(6,K(阶段visits))。封印只属于父边，普通selection也会解封本阶段首次遍历的旧父边。特殊selection沿选中的子边继续树搜索；不会额外强制rollout。','',
            '若特殊selection没有触发，本次D不能验证该分支对真实搜索的改善；不能把与C相同的发现归因于新增机制。完整数值在analysis.json，事件在D/decisions.jsonl。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('windows','comparisons','special_selections')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=HERE/'runs/class8_phase_restart_paired_100_60_v2')
    p.add_argument('--output',type=Path,default=HERE/'runs/class8_phase_restart_paired_D_60')
    a=p.parse_args();run(a.base,a.output)

