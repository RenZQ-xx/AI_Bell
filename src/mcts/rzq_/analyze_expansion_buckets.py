"""Compare compatibility-prior and uniform expansion outcomes in v4.3/v4.4."""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE=Path(__file__).resolve().parent/'runs'
OUT=BASE/'v4_3_v4_4_bucket_quality'

def fisher_two_sided(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    def prob(x): return math.comb(c1,x)*math.comb(n-c1,r1-x)/math.comb(n,r1)
    lo=max(0,r1-(n-c1));hi=min(r1,c1);p0=prob(a)
    return min(1,sum(prob(x) for x in range(lo,hi+1) if prob(x)<=p0+1e-15))

def wilson(k,n,z=1.96):
    if not n:return [0,0]
    p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]

def summary(rows,pre):
    labels=Counter(r['label'] for r in rows); exact=[r for r in rows if r['class_id']]
    classes=Counter(r['class_id'] for r in exact)
    return dict(n=len(rows),exact=len(exact),exact_rate=len(exact)/len(rows) if rows else 0,
        exact_ci95=wilson(len(exact),len(rows)),classes=dict(sorted(classes.items())),
        class_count=len(classes),new_classes=sorted(set(classes)-pre),non44=sum(c!=44 for c in classes.elements()),
        boundary=labels['invalid:boundary'],non_coplanar=labels['invalid:non_coplanar'],
        mean_parent_rank=sum(r['rank'] for r in rows)/len(rows) if rows else 0,
        mean_depth=sum(r['depth'] for r in rows)/len(rows) if rows else 0,
        mean_node_expansion_ordinal=sum(r['ordinal'] for r in rows)/len(rows) if rows else 0)

def load(v):
    folder=BASE/f'class8_{v}_300_trace';result=json.loads((folder/'result.json').read_text())
    pre=set(result['meta']['pre_discovered_class_ids']); events=defaultdict(list)
    for line in (folder/'decisions.jsonl').open():
        e=json.loads(line);events[e['iteration']].append(e)
    ordinals=Counter();rows=[]
    for it,es in sorted(events.items()):
        choice=next(e for e in es if e['event']=='progressive_choice')
        terminal=next(e for e in es if e['event']=='terminal_observation')
        visits=[e for e in es if e['event']=='node_visit']; parent=tuple(visits[-1]['selected_blocks'])
        ordinal=ordinals[parent];ordinals[parent]+=1
        cid=int(terminal['label'].split('class')[1]) if terminal['label'].startswith('exact:class') else None
        rows.append(dict(iteration=it,bucket=choice['bucket'],kind=choice['decision_kind'],
            group='prior' if choice['bucket']==0 else 'uniform',label=terminal['label'],class_id=cid,
            rank=choice['node_rank'],depth=len(visits),ordinal=ordinal,parent=list(parent),action=choice['chosen_action']))
    assert len(rows)==300
    groups={g:summary([r for r in rows if r['group']==g],pre) for g in ('prior','uniform')}
    a=groups['prior']['exact'];b=groups['prior']['n']-a;c=groups['uniform']['exact'];d=groups['uniform']['n']-c
    groups['comparison']=dict(risk_difference=groups['prior']['exact_rate']-groups['uniform']['exact_rate'],
        risk_ratio=groups['prior']['exact_rate']/groups['uniform']['exact_rate'],fisher_p=fisher_two_sided(a,b,c,d),
        non44_prior_rate=groups['prior']['non44']/groups['prior']['n'],
        non44_uniform_rate=groups['uniform']['non44']/groups['uniform']['n'],
        non44_fisher_p=fisher_two_sided(groups['prior']['non44'],groups['prior']['n']-groups['prior']['non44'],
                                       groups['uniform']['non44'],groups['uniform']['n']-groups['uniform']['non44']))
    strata={}
    for rank in sorted({r['rank'] for r in rows}):
        rr=[r for r in rows if r['rank']==rank]
        if any(r['group']=='prior' for r in rr) and any(r['group']=='uniform' for r in rr):
            strata[f'rank{rank}']={g:summary([r for r in rr if r['group']==g],pre) for g in ('prior','uniform')}
    # Exact paired comparison at the same parent: bucket0's first expansion vs bucket1's second expansion.
    parents=defaultdict(dict)
    for r in rows:
        if r['ordinal'] in (0,1):parents[tuple(r['parent'])][r['ordinal']]=r
    pairs=[(x[0],x[1]) for x in parents.values() if 0 in x and 1 in x and x[0]['group']=='prior' and x[1]['group']=='uniform']
    discord=Counter((a['class_id'] is not None,b['class_id'] is not None) for a,b in pairs)
    # two-sided exact McNemar/binomial on discordant pairs
    x=discord[(True,False)];y=discord[(False,True)];m=x+y
    paired_p=min(1,2*sum(math.comb(m,k) for k in range(0,min(x,y)+1))/(2**m)) if m else 1
    paired=dict(n=len(pairs),prior_exact=sum(a['class_id'] is not None for a,b in pairs),
        uniform_exact=sum(b['class_id'] is not None for a,b in pairs),discordant_prior_only=x,
        discordant_uniform_only=y,mcnemar_exact_p=paired_p)
    return dict(rows=rows,groups=groups,rank_strata=strata,paired_first_second=paired)

def main():
    OUT.mkdir(exist_ok=True);data={v:load(v) for v in ('v4_3','v4_4')}
    (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# v4.3 / v4.4 expansion bucket结果质量','',
      '每轮只发生一次expansion，因此将该轮terminal结果归因于该次bucket。prior=bucket0；uniform=v4.3的bucket1及v4.4的bucket1–5。该归因包含后续rollout影响。','',
      '|版本|bucket组|样本|exact|exact率（95% CI）|类别数|非44 exact|新类别|boundary|non-coplanar|平均父rank|平均树深度|平均节点扩展序号|',
      '|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for v,d in data.items():
        for g in ('prior','uniform'):
            s=d['groups'][g];ci=s['exact_ci95']
            lines.append(f"|{v}|{g}|{s['n']}|{s['exact']}|{s['exact_rate']:.1%} ({ci[0]:.1%}–{ci[1]:.1%})|{s['class_count']}|{s['non44']}|{s['new_classes']}|{s['boundary']}|{s['non_coplanar']}|{s['mean_parent_rank']:.2f}|{s['mean_depth']:.2f}|{s['mean_node_expansion_ordinal']:.2f}|")
        c=d['groups']['comparison'];lines += ['',f"{v}: prior-uniform exact率差 `{c['risk_difference']:.1%}`，比值 `{c['risk_ratio']:.2f}`，Fisher双侧 `p={c['fisher_p']:.4f}`。",f"非class44 exact率为 `{c['non44_prior_rate']:.1%}` vs `{c['non44_uniform_rate']:.1%}`，Fisher双侧 `p={c['non44_fisher_p']:.4f}`。",'']
        p=d['paired_first_second'];lines.append(f"同一父节点首个prior与第二个uniform配对：{p['n']}对，exact为{p['prior_exact']} vs {p['uniform_exact']}；仅prior成功{p['discordant_prior_only']}、仅uniform成功{p['discordant_uniform_only']}，McNemar精确p={p['mcnemar_exact_p']:.4f}。")
        lines += ['','类别分布：',f"- prior: `{d['groups']['prior']['classes']}`",f"- uniform: `{d['groups']['uniform']['classes']}`",'']
    lines += ['## 解释边界','',
      'bucket并非随机处理：bucket0通常是节点第一次扩展，而uniform发生得更晚；两组父rank、树深度及节点扩展序号不同。Fisher检验只描述未分层汇总差异，不能证明prior本身造成差异。配对比较控制父节点，但仍把第一次与第二次扩展混在处理差异中。',
      'v4.4改变bucket数量也改变progressive widening初期K(s)，因此v4.3与v4.4之间的总结果不能只归因于prior/random比例。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
