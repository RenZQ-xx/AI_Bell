"""Reconstruct saved tree states and compare compatibility under a fixed index."""
import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from baseline.orbit_blocks import add_block
from mcts.queue_search import QueueSearchConfig,build_class_scorer
from mcts.rzq_.compatibility import ClassCompatibilityIndex
from mcts.rzq_.rollout_scorer import RZQRolloutScorer

BASE=Path(__file__).resolve().parent/'runs'
OUT=BASE/'v4_v4_1_node_compat'

def mean(xs): return statistics.mean(xs) if xs else 0

def summarize(rows):
    if not rows: return {}
    return dict(n=len(rows),unique_states=len({tuple(r['selected']) for r in rows}),
        mean_rank=mean([r['rank'] for r in rows]),mean_blocks=mean([len(r['selected']) for r in rows]),
        mean_classes=mean([r['nclasses'] for r in rows]),median_classes=statistics.median(r['nclasses'] for r in rows),
        min_classes=min(r['nclasses'] for r in rows),max_classes=max(r['nclasses'] for r in rows),
        mean_masks=mean([r['masks'] for r in rows]),median_masks=statistics.median(r['masks'] for r in rows),
        mean_new_classes=mean([r['nnew'] for r in rows]),mean_new_masks=mean([r['new_masks'] for r in rows]),
        zero_compat=sum(r['nclasses']==0 for r in rows),zero_new_compat=sum(r['nnew']==0 for r in rows),
        class44_share=mean([r['compat'].get(44,0)/r['masks'] if r['masks'] else 0 for r in rows]),
        exact=sum(r['exact'] for r in rows),exact_rate=mean([r['exact'] for r in rows]),
        mean_masks_by_class={c:mean([r['compat'].get(c,0) for r in rows]) for c in range(1,47)},
        availability_by_class={c:mean([r['compat'].get(c,0)>0 for r in rows]) for c in range(1,47)})

def unique(rows):
    return list({tuple(r['selected']):r for r in reversed(rows)}.values())

def main():
    OUT.mkdir(exist_ok=True)
    base,blocks=build_class_scorer(QueueSearchConfig(),8)
    index=ClassCompatibilityIndex.build(blocks)
    pre=set(json.loads((BASE/'class8_v4_300_trace/result.json').read_text())['meta']['pre_discovered_class_ids'])
    scorer=RZQRolloutScorer(base,index,pre)
    cache={}
    def state_data(key):
        if key not in cache:
            counts=index.nonzero_counts(key)
            cache[key]=dict(selected=[i for i,x in enumerate(key) if x],rank=base.affine_rank(key),compat=counts,
                nclasses=len(counts),masks=sum(counts.values()),nnew=sum(c not in pre for c in counts),
                new_masks=sum(n for c,n in counts.items() if c not in pre))
        return cache[key]
    all_data={}
    for version in ('v4','v4_1'):
        events=defaultdict(list)
        for line in (BASE/f'class8_{version}_300_trace'/'decisions.jsonl').open():
            e=json.loads(line);events[e['iteration']].append(e)
        starts=[]; reaches=[]
        for it,es in events.items():
            terminal=next(e for e in es if e['event']=='terminal_observation')
            exact=terminal['label'].startswith('exact:class')
            visits=[e for e in es if e['event']=='node_visit']
            for depth,e in enumerate(visits):
                if depth==0: continue
                key=tuple(int(i in e['selected_blocks']) for i in range(len(blocks)))
                reaches.append(dict(iteration=it,depth=depth,exact=exact,**state_data(key)))
            parent=visits[-1]
            expansion=next(e for e in es if e['event']=='expand_action')
            key=add_block(tuple(int(i in parent['selected_blocks']) for i in range(len(blocks))),expansion['action'])
            raw=key
            if version=='v4_1': key,_=scorer.flat_closure(key)
            assert index.counts(raw)==index.counts(key), 'closure changed compatible supporting facets'
            row=dict(iteration=it,depth=len(visits),exact=exact,label=terminal['label'],**state_data(key))
            rolls=[e for e in es if e['event']=='rollout_choice']
            if rolls: assert row['rank']==rolls[0]['current_rank']
            starts.append(row); reaches.append(row)
        s={}
        for d in range(1,6):
            rr=[r for r in reaches if r['depth']==d]
            st=[r for r in starts if r['depth']==d]
            s[d]=dict(reached_weighted=summarize(rr),reached_unique=summarize(unique(rr)),
                      expanded=summarize(st),expanded_unique=summarize(unique(st)))
        bins={}
        for lo,hi in ((0,99),(100,199),(200,299),(300,399),(400,99999)):
            rr=[r for r in starts if lo<=r['masks']<=hi]
            bins[f'{lo}-{hi}']=summarize(rr)
        all_data[version]=dict(by_depth=s,mask_bins=bins,expanded_rows=starts,reached_rows=reaches,
              by_rank={rank:summarize([r for r in starts if r['rank']==rank]) for rank in sorted({r['rank'] for r in starts})},
              outcome={out:summarize([r for r in starts if r['exact']==flag]) for out,flag in [('exact',True),('invalid',False)]})
    (OUT/'data.json').write_text(json.dumps(all_data,ensure_ascii=False,indent=2)+'\n')
    lines=['# v4 / v4.1 同深度节点 compatibility 比较','',
       '所有节点统一使用 class8 原分块和同一参考库的 class1–46 masks；新类别以旧实验初始预发现集合为准，避免动态发现集合不同造成口径偏差。compat 数为包含当前所有已选 blocks 的兼容 facet masks 数，不是终点成功概率。',
       '展开节点：本轮 expansion 后、rollout 开始前的实际状态。v4.1 按代码执行 flat closure；已验证 300 个展开状态闭包前后的 compat 完全一致，以及恢复 rank 与 rollout trace 一致。到达节点含 UCB 到达与首次 expansion，按实际树边深度分组，不计 rollout。',
       '去重按实际已选 block 集合，不按对称等价去重。独立节点摘要中的 exact 只保留该状态首次出现那轮的结果，不解释为节点总体成功率；判断输出率以展开记录/访问加权结果为准。','']
    for mode,title in [('expanded','展开节点（每轮一次）'),('reached_unique','同层独立节点'),('reached_weighted','同层节点按实际访问加权')]:
        lines += [f'## {title}','','| 版本 | 深度 | 记录数 | 独立状态 | 平均rank | 兼容类别均值 | 兼容mask均值 | 新类别均值 | 新mask均值 | 零兼容 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for v,data in all_data.items():
            for d,s in data['by_depth'].items():
                r=s[mode]
                if not r: continue
                lines.append(f"| {v} | {d} | {r['n']} | {r['unique_states']} | {r['mean_rank']:.2f} | {r['mean_classes']:.2f} | {r['mean_masks']:.2f} | {r['mean_new_classes']:.2f} | {r['mean_new_masks']:.2f} | {r['zero_compat']} |")
    lines+=['','## 各 class 的展开节点平均 compat mask 数（前三层）','','| class | v4 d1 | v4.1 d1 | v4 d2 | v4.1 d2 | v4 d3 | v4.1 d3 |','|---|---:|---:|---:|---:|---:|---:|']
    for c in range(1,47):
        vals=[all_data[v]['by_depth'][d]['expanded']['mean_masks_by_class'][c] for d in (1,2,3) for v in ('v4','v4_1')]
        if any(vals):lines.append('| '+str(c)+' | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for v,data in all_data.items():
        print(v)
        for d,s in data['by_depth'].items():
            for mode in ('expanded','reached_unique','reached_weighted'):
                r=s[mode]
                if r:print(d,mode,json.dumps({k:x for k,x in r.items() if k not in ('mean_masks_by_class','availability_by_class')}))
        print('rank',json.dumps({rank:{k:r[k] for k in ('n','mean_classes','mean_masks','exact_rate')} for rank,r in data['by_rank'].items()}))

if __name__=='__main__':main()
