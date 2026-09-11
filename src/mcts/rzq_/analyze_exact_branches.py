"""Attribute terminal outcomes to actual tree prefixes, never rollout prefixes."""
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE=Path(__file__).resolve().parent/'runs'
OUT=BASE/'v4_v4_1_exact_branches'

def summary(rows):
    exact=[r for r in rows if r['class_id'] is not None]
    counts=Counter(r['class_id'] for r in exact)
    new=sorted({r['class_id'] for r in exact if r['new_to_initial']})
    return dict(visits=len(rows),exact=len(exact),classes=dict(sorted(counts.items())),new_classes=new,
                exact_rate=len(exact)/len(rows) if rows else 0,
                mean_tree_depth=sum(r['tree_depth'] for r in rows)/len(rows) if rows else 0,
                mean_ucb_depth=sum(r['ucb_depth'] for r in rows)/len(rows) if rows else 0,
                mean_exact_tree_depth=sum(r['tree_depth'] for r in exact)/len(exact) if exact else None,
                invalid=dict(Counter(r['label'] for r in rows if r['class_id'] is None)))

def analyze(v):
    folder=BASE/f'class8_{v}_300_trace'
    result=json.loads((folder/'result.json').read_text())
    pre=set(result['meta']['pre_discovered_class_ids'])
    by_it=defaultdict(list)
    for line in (folder/'decisions.jsonl').open():
        e=json.loads(line); by_it[e['iteration']].append(e)
    rows=[]
    for it,events in sorted(by_it.items()):
        tree=[e['chosen_action'] if e['event']=='ucb_choice' else e['action'] for e in events if e['event'] in ('ucb_choice','expand_action')]
        ucb=[e for e in events if e['event']=='ucb_choice']
        exp=[e for e in events if e['event']=='expand_action']
        visits=[e for e in events if e['event']=='node_visit']
        terminal=[e for e in events if e['event']=='terminal_observation']
        assert len(terminal)==1
        label=terminal[0]['label']
        cid=int(label.split('class')[1]) if label.startswith('exact:class') else None
        rolls=[e for e in events if e['event']=='rollout_choice']
        rows.append(dict(iteration=it,tree_path=tree,tree_depth=len(tree),ucb_depth=len(ucb),
                         rollout_length=len(rolls), expansion_parent_rank=visits[-1]['rank'],
                         rollout_start_rank=rolls[0]['current_rank'] if rolls else None,
                         expansion_count=len(exp),class_id=cid,label=label,new_to_initial=cid is not None and cid not in pre))
    assert Counter(r['label'] for r in rows)==Counter(result['result']['encountered_label_counts'])
    branches={}
    for depth in (1,2,3):
        groups=defaultdict(list)
        for r in rows:
            if r['tree_depth']>=depth: groups[tuple(r['tree_path'][:depth])].append(r)
        branches[depth]=[dict(path=list(p),**summary(rs),exact_iterations=[r['iteration'] for r in rs if r['class_id'] is not None],
                              stops_here=summary([r for r in rs if r['tree_depth']==depth]),
                              goes_deeper=summary([r for r in rs if r['tree_depth']>depth])) for p,rs in sorted(groups.items())]
    return dict(total=summary(rows),by_depth={d:summary([r for r in rows if r['tree_depth']==d]) for d in sorted({r['tree_depth'] for r in rows})},
                branches=branches,rows=rows)

def classes(c): return ', '.join(f'{k}×{v}' for k,v in c.items()) or '—'

def main():
    OUT.mkdir(exist_ok=True)
    data={v:analyze(v) for v in ('v4','v4_1')}
    (OUT/'data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# v4 与 v4.1：exact 产出的树分支分布','',
           '分支按本轮 trace 实际 UCB 动作 + expansion 动作定义；不把 rollout 动作充当树分支。深度为 expansion 后树路径边数，通常等于 UCB 次数 + 1。闭包隐式加入的 block 不计为额外树边。',
           '只进入第一步便 rollout 的命中，仅计入第一步表；进入第二步才计入第二步表，以此类推。同一深度内互斥，跨深度包含关系，不可累加。classes 格式为 class编号×命中次数。新类别以旧实验预发现集合为准。','',
           '## 树深度与 exact','', '| 版本 | 树深度 | 迭代数 | exact | 命中率 | 类别及次数 |','|---|---:|---:|---:|---:|---|']
    for v,r in data.items():
        for d,s in r['by_depth'].items(): lines.append(f"| {v} | {d} | {s['visits']} | {s['exact']} | {s['exact_rate']:.1%} | {classes(s['classes'])} |")
    for v,r in data.items():
        lines+=['',f'## {v}','']
        for d,bs in r['branches'].items():
            lines+= [f'### 第 {d} 步树分支','', '| 路径 | 访问 | exact | 命中率 | 类别及次数 | 新类别 | 在本层结束的exact | 更深处的exact | 命中iteration |', '|---|---:|---:|---:|---|---|---:|---:|---|']
            for b in sorted(bs,key=lambda b:(-b['exact'],-len(b['classes']),-b['visits'],b['path'])):
                lines.append(f"| `{b['path']}` | {b['visits']} | {b['exact']} | {b['exact_rate']:.1%} | {classes(b['classes'])} | {b['new_classes']} | {b['stops_here']['exact']} | {b['goes_deeper']['exact']} | {b['exact_iterations']} |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for v,r in data.items():
        print(v,'total',json.dumps(r['total']))
        print('depth',json.dumps(r['by_depth']))
        print('root',json.dumps(r['branches'][1]))
        print('depth2_top',json.dumps(sorted(r['branches'][2],key=lambda b:-b['exact'])[:8]))
        print('depth3_top',json.dumps(sorted(r['branches'][3],key=lambda b:-b['exact'])[:8]))

if __name__=='__main__': main()
