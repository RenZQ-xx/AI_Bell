"""Summarize v4/v4.1/v4.2 results and validate the v4.2 ablation trace."""
import argparse
import math
import json
from collections import Counter
from pathlib import Path

from analyze_v4_visit_distribution import analyze as visits
from analyze_productive_depth3 import analyze as depth3

BASE=Path(__file__).resolve().parent/'runs'

def fisher_two_sided(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    def probability(x):
        return math.comb(c1,x)*math.comb(n-c1,r1-x)/math.comb(n,r1)
    lo=max(0,r1-(n-c1)); hi=min(r1,c1); observed=probability(a)
    return min(1.0,sum(probability(x) for x in range(lo,hi+1) if probability(x)<=observed+1e-15))

def main(include_v43=False, include_v44=False, include_v45=False, include_v46=False):
    versions=('v4','v4_1','v4_2','v4_3','v4_4','v4_5','v4_6') if include_v46 else ('v4','v4_1','v4_2','v4_3','v4_4','v4_5') if include_v45 else ('v4','v4_1','v4_2','v4_3','v4_4') if include_v44 else ('v4','v4_1','v4_2','v4_3') if include_v43 else ('v4','v4_1','v4_2')
    stem='_'.join(versions)+'_comparison'
    data={}
    for v in versions:
        folder=BASE/f'class8_{v}_300_trace'
        payload=json.loads((folder/'result.json').read_text())
        r=payload['result'];meta=payload['meta']
        counts=r['encountered_label_counts']
        exact=sum(n for label,n in counts.items() if label.startswith('exact:'))
        dist=visits(v)
        nodes=depth3(v)
        data[v]=dict(seed=meta['seed'],iterations=r['iterations_completed'],pre=meta['pre_discovered_class_ids'],
            exact=exact,classes=r['exact_class_ids'],new_classes=sorted(set(r['exact_class_ids'])-set(meta['pre_discovered_class_ids'])),
            discoveries=[dict(class_id=d['class_id'],iteration=d['iteration']) for d in r['exact_discoveries']],
            counts=counts,nodes=r['nodes_created'],root=dist['root'],phases=dist['phases'],
            mean_tree_depth=sum(int(d)*n for d,n in dist['iteration_tree_edges'].items())/r['iterations_completed'],
            depth3_states=len(nodes),depth3_max_visits=max(n['visits'] for n in nodes),
            depth3_multi=sum(len(n['classes'])>1 for n in nodes),
            depth3_visit_hist=dict(Counter(n['visits'] for n in nodes)),
            depth3_productive=[n for n in nodes if len(n['classes'])>1])
        if v in ('v4_2','v4_3','v4_4','v4_5','v4_6'):
            es=[json.loads(line) for line in (folder/'decisions.jsonl').open()]
            ucbs=[e for e in es if e['event']=='ucb_choice']
            phase='ucb' if v in ('v4_3','v4_4','v4_5','v4_6') else 'normalized_ucb'
            assert ucbs and all(e['min_action_visits']==1 and e['selection_phase']==phase for e in ucbs)
            if v in ('v4_3','v4_4','v4_5','v4_6'):
                assert all(math.isclose(c['exploitation'],c['edge_survival_q'],abs_tol=1e-12) for e in ucbs for c in e['candidates'])
            assert all(c['ucb_visit_source']=='real_node_visits' for e in ucbs for c in e['candidates'])
            expansions=[e for e in es if e['event']=='progressive_choice']
            assert len(expansions)==300 and len(expansions[0]['candidate_actions'])==40
            allowed_buckets=range(6) if v in ('v4_4','v4_5','v4_6') else (0,1)
            assert all(e['bucket'] in allowed_buckets and e['decision_kind'] in ('compatibility_richness_prior','uniform_random') for e in expansions)
            if v=='v4_4':
                assert all((e['bucket']==0)==(e['decision_kind']=='compatibility_richness_prior') for e in expansions)
            if v=='v4_5':
                assert all(e['decision_kind']=='uniform_random' for e in expansions)
                assert all(len(set(round(float(p),14) for p in e['candidate_priors'].values()))==1 for e in expansions)
            if v=='v4_6':
                assert all((e['bucket']%2==0)==(e['decision_kind']=='compatibility_richness_prior') for e in expansions)
            assert not any(e['event'] in ('symmetric_node_reuse','symmetric_node_merge') for e in es)
            assert Counter(e['label'] for e in es if e['event']=='terminal_observation')==Counter(counts)
    assert all(d['seed']==data['v4']['seed'] and d['pre']==data['v4']['pre'] and d['iterations']==300 for d in data.values())
    envs=[json.loads((BASE/f'class8_{v}_300_trace'/'environment.json').read_text()) for v in ('v4_1','v4_2')]
    assert envs[0]['sha256']==envs[1]['sha256'], 'shared source/data changed between v4.1 and v4.2'
    old=json.loads((BASE/'class8_v4_1_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
    new=json.loads((BASE/'class8_v4_2_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
    assert {k:v for k,v in old.items() if k!='min_action_visits'}=={k:v for k,v in new.items() if k!='min_action_visits'}
    if include_v43:
        env3=json.loads((BASE/'class8_v4_3_300_trace'/'environment.json').read_text())
        assert env3['sha256']==envs[1]['sha256']
        config3=json.loads((BASE/'class8_v4_3_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
        excluded={'ucb_selection','normalized_q','equal_q_normalized_value'}
        assert {k:v for k,v in new.items() if k not in excluded}=={k:v for k,v in config3.items() if k not in excluded}
    if include_v44:
        config4=json.loads((BASE/'class8_v4_4_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
        assert config4['expansion_bucket_count']==6
        assert config4['min_action_visits']==1 and config4['normalized_q'] is None
    if include_v45:
        config5=json.loads((BASE/'class8_v4_5_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
        assert config5['expansion_bucket_count']==6 and config5['expansion_prior'] is None
        assert config5['min_action_visits']==1 and config5['normalized_q'] is None
    if include_v46:
        config6=json.loads((BASE/'class8_v4_6_300_trace'/'result.json').read_text())['meta']['rzq_scoring']
        assert config6['expansion_bucket_count']==6 and config6['expansion_prior']=='compatibility_richness'
        assert config6['min_action_visits']==1 and config6['normalized_q'] is None
    (BASE/(stem+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    header='| 指标 | '+' | '.join(v.replace('_','.') for v in versions)+' |'
    separator='|---|'+'---:|'*len(versions)
    lines=['# '+' / '.join(v.replace('_','.') for v in versions)+' 对照','',
        '相同 class8 分块、seed=20288754、旧预发现集合、300 iterations。v4.2 唯一改动是将 v4.1 的最低 child.visits 从5改为1；未启用重启机制。','',
        header,separator]
    if include_v43:
        lines.insert(3,'v4.3 相对 v4.2 仅关闭父边 Q 归一化，保留最低访问1和真实节点访问探索项。')
    if include_v44:
        lines.insert(4,'v4.4 相对 v4.3 仅将 compatibility-richness expansion bucket 数从2恢复为6。')
    if include_v45:
        lines.insert(5,'v4.5 相对 v4.4 仅将六个 expansion bucket 全部改为均匀随机。')
    if include_v46:
        lines.insert(6,'v4.6 保留六bucket容量，并按prior/random逐次交替。')
    metrics=[('exact次数',lambda d:d['exact']),('exact类别数',lambda d:len(d['classes'])),('新类别',lambda d:d['new_classes']),
        ('class44次数',lambda d:d['counts'].get('exact:class44',0)),('非class44 exact',lambda d:d['exact']-d['counts'].get('exact:class44',0)),
        ('boundary invalid',lambda d:d['counts'].get('invalid:boundary',0)),('non-coplanar invalid',lambda d:d['counts'].get('invalid:non_coplanar',0)),
        ('节点数',lambda d:d['nodes']),('根动作数',lambda d:d['root']['branches']),('根top3占比',lambda d:f"{d['root']['top3']:.1%}"),
        ('根有效分支数',lambda d:f"{d['root']['effective_branches']:.2f}"),('最低访问保障selection次数',lambda d:d['phases'].get('minimum_visits',0)),
        ('平均树深度',lambda d:f"{d['mean_tree_depth']:.3f}"),('第三层独立状态数',lambda d:d['depth3_states']),
        ('第三层节点最大visits',lambda d:d['depth3_max_visits']),('第三层多类别节点数',lambda d:d['depth3_multi'])]
    for label,fn in metrics:lines.append('| '+label+' | '+' | '.join(str(fn(d)) for d in data.values())+' |')
    lines+=['','## Exact class 分布','',header.replace('指标','class'),separator]
    for c in sorted(set().union(*(d['classes'] for d in data.values()))):
        lines.append('| '+str(c)+' | '+' | '.join(str(d['counts'].get(f'exact:class{c}',0)) for d in data.values())+' |')
    for v,d in data.items():
        lines+=['',f'## {v} 新类别首次发现','']
        for hit in d['discoveries']:
            if hit['class_id'] in d['new_classes']:lines.append(f"- class{hit['class_id']}：iteration {hit['iteration']}")
    if include_v45:
        v44=data['v4_4'];v45=data['v4_5']
        exact_p=fisher_two_sided(v44['exact'],300-v44['exact'],v45['exact'],300-v45['exact'])
        v44_non44=v44['exact']-v44['counts'].get('exact:class44',0)
        v45_non44=v45['exact']-v45['counts'].get('exact:class44',0)
        non44_p=fisher_two_sided(v44_non44,300-v44_non44,v45_non44,300-v45_non44)
        lines+=['','## v4.4 / v4.5 显著性','',
            f"总exact率：{v44['exact']/300:.1%} vs {v45['exact']/300:.1%}，Fisher双侧p={exact_p:.4f}。",
            f"非class44 exact率：{v44_non44/300:.1%} vs {v45_non44/300:.1%}，Fisher双侧p={non44_p:.4f}。"]
    if include_v46:
        lines+=['','## v4 / v4.4 / v4.5 / v4.6 显著性','',
            '|对比|总exact率|Fisher p|非44 exact率|Fisher p|','|---|---:|---:|---:|---:|']
        for left in ('v4','v4_4','v4_5'):
            a=data[left]; b=data['v4_6']; an=a['exact']-a['counts'].get('exact:class44',0); bn=b['exact']-b['counts'].get('exact:class44',0)
            lines.append(f"|{left.replace('_','.')} vs v4.6|{a['exact']/300:.1%} vs {b['exact']/300:.1%}|{fisher_two_sided(a['exact'],300-a['exact'],b['exact'],300-b['exact']):.4f}|{an/300:.1%} vs {bn/300:.1%}|{fisher_two_sided(an,300-an,bn,300-bn):.4f}|")
    lines+=['','## 验证','',
        'v4.1/v4.2 环境记录中的公共源码和输入数据指纹一致，结果配置中只有 min_action_visits 不同。v4.2 trace 全部 selection 使用 normalized_ucb 和真实节点访问数；minimum_visits 阶段0次；根候选40个；expansion为双bucket；没有对称复用或合并事件；终点计数与result.json一致。',
        '第三层按实际状态合并不同到达路径，访问含首次扩展；不是按路径前缀分别统计。完整逐次记录见比较JSON。单seed仅支持本次消融观察，不能保证跨seed稳定。']
    if include_v43:
        lines+=['v4.2/v4.3 公共源码和输入指纹一致；v4.3 全部 selection 使用原始 UCB 阶段，逐个候选验证 exploitation == edge_survival_q；最低访问保障触发0次。']
    if include_v44:
        lines+=['v4.4 trace 已验证 bucket 范围为0–5，bucket0使用compatibility-richness prior，其余bucket均匀随机；selection使用原始父边Q，最低访问保障触发0次。']
    if include_v45:
        lines+=['v4.5 trace 已验证 bucket 范围为0–5，全部decision_kind为uniform_random且每次候选概率相等；selection使用原始父边Q，最低访问保障触发0次。']
    if include_v46:
        lines+=['v4.6 trace 已验证 bucket 0/2/4使用compatibility-richness prior，1/3/5均匀随机；selection使用原始父边Q，最低访问保障触发0次。']
    (BASE/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-v43',action='store_true')
    parser.add_argument('--include-v44',action='store_true')
    parser.add_argument('--include-v45',action='store_true')
    parser.add_argument('--include-v46',action='store_true')
    args=parser.parse_args()
    main(args.include_v43 or args.include_v44 or args.include_v45 or args.include_v46,args.include_v44 or args.include_v45 or args.include_v46,args.include_v45 or args.include_v46,args.include_v46)
