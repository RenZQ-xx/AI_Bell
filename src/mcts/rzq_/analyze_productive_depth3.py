"""Inspect actual depth-three states, visit chronology and outgoing flow."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

BASE=Path(__file__).resolve().parent/'runs'
OUT=BASE/'v4_v4_1_depth3_productive'

def analyze(v):
    closer=None
    if v not in ('v4','v4_1'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
        from mcts.queue_search import QueueSearchConfig,build_class_scorer
        from mcts.rzq_.compatibility import ClassCompatibilityIndex
        from mcts.rzq_.rollout_scorer import RZQRolloutScorer
        scorer,blocks=build_class_scorer(QueueSearchConfig(),8)
        closer=RZQRolloutScorer(scorer,ClassCompatibilityIndex.build(blocks),())
    events=defaultdict(list)
    for line in (BASE/f'class8_{v}_300_trace'/'decisions.jsonl').open():
        e=json.loads(line);events[e['iteration']].append(e)
    groups=defaultdict(list)
    for it,es in events.items():
        visits=[e for e in es if e['event']=='node_visit']
        actions=[e['chosen_action'] if e['event']=='ucb_choice' else e['action'] for e in es if e['event'] in ('ucb_choice','expand_action')]
        term=next(e for e in es if e['event']=='terminal_observation')['label']
        if len(actions)<3:continue
        if len(visits)>3:
            selected=tuple(visits[3]['selected_blocks'])
            recorded=visits[3]['visits']
        else:
            # Verified by the companion reconstruction: closure adds no blocks at depths <= 4.
            selected=tuple(sorted(set(visits[-1]['selected_blocks'])|{actions[2]}))
            if closer is not None:
                key,_=closer.flat_closure(tuple(int(i in selected) for i in range(40)))
                selected=tuple(i for i,x in enumerate(key) if x)
            recorded=None
        groups[selected].append(dict(iteration=it,path=actions,label=term,
            exact=term.startswith('exact:'),child=actions[3] if len(actions)>3 else None,
            recorded_visits=recorded,
            outgoing_kind=next((e['event'] for e in es if e['event'] in ('ucb_choice','expand_action') and (e.get('node_path',e.get('parent_path'))==list(visits[3]['path']) if len(visits)>3 else False)),None)))
    result=[]
    for key,rows in groups.items():
        for ordinal,row in enumerate(rows):
            if row['recorded_visits'] is not None:
                assert row['recorded_visits']==ordinal, (v,key,ordinal,row)
        cls=Counter(r['label'] for r in rows if r['exact'])
        children=Counter(r['child'] for r in rows if r['child'] is not None)
        total=sum(children.values())
        ordinals=[i+1 for i,r in enumerate(rows) if r['exact']]
        seen=set();second=None
        for i,r in enumerate(rows,1):
            if r['exact']:seen.add(r['label'])
            if len(seen)>=2 and second is None:second=i
        result.append(dict(selected=list(key),paths=[list(p) for p in sorted({tuple(r['path'][:3]) for r in rows})],
            visits=len(rows),exact=sum(cls.values()),classes=cls,second_class_visit=second,
            child_visits=dict(children),child_count=len(children),descents=total,
            max_child_visits=max(children.values(),default=0),
            child_top1=max(children.values(),default=0)/total if total else None,
            child_hhi=sum((n/total)**2 for n in children.values()) if total else None,
            repeat_child_visits=total-len(children),exact_visit_ordinals=ordinals,rows=rows))
    return sorted(result,key=lambda r:(-len(r['classes']),-r['visits'],-r['exact']))

def main():
    OUT.mkdir(exist_ok=True)
    data={v:analyze(v) for v in ('v4','v4_1')}
    (OUT/'data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# 第三层多类别产出节点：访问与后续分支','',
           '按实际已选 block 集合合并不同到达路径。访问含首次 expansion 到达。子分支流量含后续 expansion 和 UCB selection；节点创建当轮直接 rollout，不计入任何树子分支。已由同深度 compat 重建确认前三层 closure 不增加隐式 block。','']
    lines += ['原先按动作路径统计，v4 有4个多类别分支、最大8次访问；按实际节点统计，v4 有5个多类别节点、最大9次访问。所有 node_visit 记录的历史 visits 已与逐次累计数量核对一致。','']
    for v,nodes in data.items():
        lines += [f'## {v}：全部第三层节点','','| 状态 | 路径 | visits | exact | 类别及次数 | 第几次访问出现第二类 | 子动作:访问 |','|---|---|---:|---:|---|---:|---|']
        for n in nodes:
            lines.append(f"| {n['selected']} | {n['paths']} | {n['visits']} | {n['exact']} | {dict(n['classes'])} | {n['second_class_visit']} | {n['child_visits']} |")
        lines+=['',f'## {v}：多类别节点逐次访问（v4.1 列出重复 exact 节点）','']
        for n in nodes:
            if not (len(n['classes'])>1 or (v=='v4_1' and n['exact']>1)):continue
            lines += [f"### {n['selected']}",'','| 第几次访问 | iteration | 树路径 | 终点 |','|---:|---:|---|---|']
            for i,r in enumerate(n['rows'],1):lines.append(f"| {i} | {r['iteration']} | {r['path']} | {r['label']} |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for v,nodes in data.items():
        print(v,'state_visit_hist',dict(Counter(n['visits'] for n in nodes)))
        print('first3',dict(visits=sum(min(3,n['visits']) for n in nodes),exact=sum(r['exact'] for n in nodes for r in n['rows'][:3]),multi=sum(len({r['label'] for r in n['rows'][:3] if r['exact']})>1 for n in nodes)))
        print('productive',json.dumps([n for n in nodes if len(n['classes'])>1 or n['exact']>1]))
        print('visit_bins',json.dumps({vis:dict(nodes=sum(n['visits']==vis for n in nodes),multi=sum(n['visits']==vis and len(n['classes'])>1 for n in nodes),exact=sum(n['exact'] for n in nodes if n['visits']==vis)) for vis in sorted({n['visits'] for n in nodes})}))

if __name__=='__main__':main()
