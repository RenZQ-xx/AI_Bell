"""Compare actual edge traversals (selection plus expansion) in saved traces."""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'runs'

def stats(counts):
    vals = sorted((v for v in counts.values() if v), reverse=True)
    total = sum(vals)
    p = [v / total for v in vals] if total else []
    entropy = -sum(x * math.log(x) for x in p)
    return dict(total=total, branches=len(vals), top1=vals[0]/total if total else 0,
                top3=sum(vals[:3])/total if total else 0,
                effective_branches=math.exp(entropy),
                normalized_entropy=entropy/math.log(len(vals)) if len(vals)>1 else 0,
                hhi=sum(x*x for x in p), counts=dict(sorted(counts.items(), key=lambda kv:-kv[1])))

def analyze(version):
    path = BASE / f'class8_{version}_300_trace' / 'decisions.jsonl'
    events = defaultdict(list)
    for line in path.open():
        e = json.loads(line); events[e['iteration']].append(e)
    depths = defaultdict(Counter)
    edges = defaultdict(Counter)
    phases = Counter()
    root_phases = Counter()
    root_history = {}
    root_expansions = []
    ratios, root_ratios = [], []
    visit_paths = Counter()
    for it, es in sorted(events.items()):
        state = None
        depth = 0
        actual_path = []
        for e in es:
            if e['event'] == 'node_visit':
                state = tuple(e['selected_blocks'])
            if e['event'] not in ('ucb_choice','expand_action'):
                continue
            action = e['chosen_action'] if e['event']=='ucb_choice' else e['action']
            assert state is not None
            edges[state][action] += 1
            actual_path.append(action)
            depths[depth+1][str(actual_path)] += 1
            if e['event']=='ucb_choice':
                phase=e.get('selection_phase','ucb'); phases[phase]+=1
                if depth==0: root_phases[phase]+=1
                chosen=next(c for c in e['candidates'] if c['action']==action)
                q=abs(chosen['exploitation']); u=chosen['exploration']
                item=dict(iteration=it, depth=depth, phase=phase, q=q, exploration=u,
                          chosen_below_max_q=chosen['exploitation'] < max(c['exploitation'] for c in e['candidates'])-1e-10,
                          exploration_ge_q=u>=q, ratio=u/q if q>0 else None)
                ratios.append(item)
                if depth==0: root_ratios.append(item)
            elif depth==0:
                root_expansions.append(dict(iteration=it,action=action))
            depth+=1
        visit_paths[depth]+=1
        if it in (50,100,150,200,250,300): root_history[it]=stats(edges[()])
    assert sum(edges[()].values())==300
    root=stats(edges[()])
    def score_summary(rows):
        return dict(n=len(rows), median_q=statistics.median(r['q'] for r in rows),
                    median_exploration=statistics.median(r['exploration'] for r in rows),
                    exploration_ge_q=sum(r['exploration_ge_q'] for r in rows),
                    chosen_below_max_q=sum(r['chosen_below_max_q'] for r in rows),
                    median_exploration_q_ratio=statistics.median(r['ratio'] for r in rows if r['ratio'] is not None)) if rows else {}
    return dict(root=root,root_history=root_history,root_expansions=root_expansions,
                phases=phases,root_phases=root_phases,depths={d:stats(c) for d,c in depths.items()},
                iteration_tree_edges=visit_paths,
                selection_score_summary=score_summary(ratios),root_selection_score_summary=score_summary(root_ratios),
                normal_selection_score_summary=score_summary([r for r in ratios if r['phase']!='minimum_visits']),
                root_normal_score_summary=score_summary([r for r in root_ratios if r['phase']!='minimum_visits']),
                parents={str(s):stats(c) for s,c in edges.items()}, source=str(path))

if __name__=='__main__':
    result={v:analyze(v) for v in ('v4','v4_1')}
    out=BASE/'v4_v4_1_visit_distribution.json'
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for v,r in result.items():
        print(v, json.dumps({k:r[k] for k in ('root','root_phases','phases','root_normal_score_summary','normal_selection_score_summary','iteration_tree_edges')}))
        print('depths',json.dumps({d:{k:v for k,v in s.items() if k!='counts'} for d,s in r['depths'].items()}))
        print('history',json.dumps({d:{k:s[k] for k in ('branches','top1','effective_branches','hhi')} for d,s in r['root_history'].items()}))
