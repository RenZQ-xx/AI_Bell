"""Audit surviving classes 9/11/25 along v11's final 300 actual paths."""
from __future__ import annotations
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import statistics
from audit_environment import HERE
from mcts.rzq_.phase_restart import load_task

TARGETS=(9,11,25)
P=HERE/'runs/class8_v11'
OUT=P/'remaining_compat_last300'


def main():
    OUT.mkdir(exist_ok=False)
    result=json.loads((P/'result.json').read_text())
    task=load_task((P/'final_snapshot.pkl').read_bytes())
    scorer=task.state.scorer;index=scorer.compatibility
    ns={n.creation_index:n for n in task.state.nodes.values()}
    masks={c:index.masks_by_class[c] for c in TARGETS}
    @lru_cache(None)
    def compat(key):
        selected=frozenset(i for i,v in enumerate(key) if v)
        return tuple(sum(selected<=m for m in masks[c]) for c in TARGETS)
    def counts(key):return dict(zip(TARGETS,compat(tuple(key))))
    @lru_cache(None)
    def health(key):
        selected=frozenset(i for i,v in enumerate(key) if v)
        remaining=set(range(len(key)))-selected
        data={}
        for c in TARGETS:
            viable=[m for m in masks[c] if selected<=m]
            safe=set().union(*(m-selected for m in viable)) if viable else set()
            data[c]={'masks':len(viable),'root_fraction':len(viable)/len(masks[c]),
                     'safe_next_actions':len(safe),'remaining_actions':len(remaining),
                     'safe_fraction':len(safe)/len(remaining) if remaining else None,
                     'safe_action_ids':sorted(safe)}
        return data
    def node_info(node):
        h=health(tuple(node.key))
        return {'id':node.creation_index,'stored_path':node.path,'rank':node.rank,
                'selected_blocks':[i for i,v in enumerate(node.key) if v],
                'compat':counts(node.key),'health':h,
                'final_unexpanded_safe_actions':{c:sorted(set(h[c]['safe_action_ids'])-set(node.children)) for c in TARGETS}}
    lo=result['last_new_iteration']+1;hi=result['iterations']
    assert hi-lo+1==300
    grouped=defaultdict(list)
    for line in (P/'decisions.jsonl').open():
        e=json.loads(line)
        if lo<=e.get('iteration',0)<=hi:grouped[e['iteration']].append(e)
    assert len(grouped)==300
    depth_rows=defaultdict(list);edge_groups=defaultdict(Counter);node_visits=Counter();ucb_visits=Counter()
    root_edge_visits=Counter();d2_edge_visits=Counter();iteration_rows=[]
    selection_summary=defaultdict(Counter);selection_losses=[]
    rollout_losses={c:Counter() for c in TARGETS};rollout_eliminations={c:[] for c in TARGETS}
    rollout_losses['any']=Counter();rollout_eliminations['any']=[]
    @lru_cache(None)
    def union_count(key):
        return sum(compat(key))
    def safe_count(key,actions,c):
        j=TARGETS.index(c) if c!='any' else None
        safe=[]
        for a in actions:
            k=list(key);k[a]=1;cs=compat(tuple(k))
            if (sum(cs) if c=='any' else cs[j])>0:safe.append(a)
        return safe
    for it,es in grouped.items():
        bp=next(e for e in es if e['event']=='real_visit_backpropagation')
        route=bp['edges'];ids=[result['tree']['root_id']]+[e['child_id'] for e in route]
        selected_by=[]
        for e in es:
            if e['event'] in ('ucb_choice','sealed_selection'):
                selected_by.append(e['event'])
                parent=ns[e['parent_id']]
                child=parent.children[e['chosen_action']]
                ucb_visits[parent.creation_index]+=1
                pc=counts(parent.key);cc=counts(child.key)
                for c in TARGETS:
                    if pc[c]>0:
                        s=selection_summary[(e['event'],c)];s['parent_viable']+=1
                        valid=[a['action'] for a in e['candidates'] if counts(parent.children[a['action']].key)[c]>0]
                        s['eligible_viable_exists']+=bool(valid)
                        s['chosen_viable']+=cc[c]>0
                        if cc[c]==0:
                            s['target_eliminated']+=1
                            selection_losses.append({'iteration':it,'event':e['event'],'parent_id':parent.creation_index,
                                    'chosen_action':e['chosen_action'],'class':c,'viable_eligible_actions':valid})
        for d,nid in enumerate(ids):
            n=ns[nid];node_visits[nid]+=1
            depth_rows[d].append({'iteration':it,'id':nid,**node_info(n)})
        root_edge_visits[route[0]['action']]+=1
        if len(route)>1:d2_edge_visits[(route[0]['action'],route[1]['action'],route[1]['child_id'])]+=1
        start=ns[ids[-1]]
        depth_rows['rollout_start'].append({'iteration':it,**node_info(start)})
        key=tuple(start.key);initial=counts(key);lost={}
        for step,e in enumerate([e for e in es if e['event']=='rollout_choice'],1):
            before=counts(key)
            after_key=list(key);after_key[e['chosen_action']]=1;after_key=tuple(after_key)
            after=counts(after_key)
            assert all(after[c]<=before[c] for c in TARGETS)
            for c in (*TARGETS,'any'):
                n=sum(before.values()) if c=='any' else before[c]
                m=sum(after.values()) if c=='any' else after[c]
                if n==0:continue
                st=rollout_losses[c];st['viable_steps']+=1
                sample=safe_count(key,e['candidate_actions'],c)
                pool=safe_count(key,e['pool'],c)
                st['sample_has_safe']+=bool(sample);st['pool_has_safe']+=bool(pool)
                st['chosen_safe']+=m>0
                if not m:
                    reason='sample_omitted_safe_actions' if not sample else 'top_k_removed_safe_actions' if not pool else 'softmax_chose_incompatible'
                    st[reason]+=1
                    rec={'iteration':it,'rollout_step':step,'before_counts':before,'chosen_action':e['chosen_action'],
                         'safe_sample':sample,'safe_pool':pool,'reason':reason}
                    rollout_eliminations[c].append(rec);lost[c]=step
            key=after_key
        label=next(e['label'] for e in es if e['event']=='terminal_observation')
        iteration_rows.append({'iteration':it,'root_action':route[0]['action'],'node_ids':ids,
                               'rollout_start_compat':initial,'rollout_final_compat':counts(key),'first_loss_rollout_step':lost,'label':label})
    def summarize(rows):
        data={'visits':len(rows),'unique_nodes':len({r['id'] for r in rows}),'targets':{}}
        for c in TARGETS:
            vals=[r['compat'][c] for r in rows]
            data['targets'][c]={'positive_visits':sum(v>0 for v in vals),'zero_visits':sum(v==0 for v in vals),
                                'mean_masks':statistics.mean(vals) if vals else 0,
                                'median_masks':statistics.median(vals) if vals else 0,'min_masks':min(vals,default=0),'max_masks':max(vals,default=0),
                                'mean_root_fraction':statistics.mean(vals)/len(masks[c]) if vals else 0,
                                'mean_safe_action_fraction':statistics.mean(r['health'][c]['safe_fraction'] or 0 for r in rows) if rows else 0}
        data['all_targets_zero']=sum(not any(r['compat'].values()) for r in rows)
        return data
    by_depth={d:summarize(rows) for d,rows in depth_rows.items()}
    root=task.state.root
    root_rows=[{'action':a,'last300_visits':root_edge_visits[a],**node_info(n)} for a,n in root.children.items()]
    root_rows.sort(key=lambda r:(-r['last300_visits'],r['action']))
    d2_rows=[{'root_action':a,'action':b,'last300_visits':v,**node_info(ns[nid])}
             for (a,b,nid),v in d2_edge_visits.most_common()]
    ucb_rows=[{'last300_selection_calls':v,'last300_node_visits':node_visits[nid],**node_info(ns[nid])}
              for nid,v in ucb_visits.most_common()]
    # Compare against the uniform distribution over all already-expanded root actions.
    root_uniform=summarize([node_info(n) for n in root.children.values()])
    full=index.counts(root.key);known=set(result['global_classes'])
    report={'iterations':[lo,hi],'global_missing':sorted(set(range(1,47))-known),
            'root_compatible_missing':{c:n for c,n in full.items() if n and c not in known},
            'root_counts_all_classes':full,'depth_summary':by_depth,'root_uniform_reference':root_uniform,
            'root_children':root_rows,'depth2_branches':d2_rows,'ucb_parents':ucb_rows,
            'selection_summary':{f'{event}/class{c}':dict(v) for (event,c),v in selection_summary.items()},
            'selection_eliminations':selection_losses,
            'rollout_summary':{c:dict(v) for c,v in rollout_losses.items()},
            'rollout_eliminations':rollout_eliminations,'iterations_detail':iteration_rows}
    (OUT/'data.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# v11 最后300轮：class9/11/25 compatibility','',
           f'窗口：{lo}–{hi}。按真实树边路径定义根为深度0、根子节点为深度1、孙节点为深度2；不把rollout动作算作树层级。完全相同闭包状态按节点ID计数，分支访问按实际父边记录。','',
           f"当前class8根节点可兼容而未发现的恰好为 `{report['root_compatible_missing']}`。全46类未发现集合是 `{report['global_missing']}`，其他未发现类在该分块根节点compat为0。",'',
           'compat表示支持集合包含当前全部已选block的参考facet mask数量，不是命中概率。正数表示仍有参考解可达；后续安全动作比例衡量有多少未选block可保留该class至少一个mask，也不是rollout实际抽样概率。','',
           '| 实际到达层级 | 访问数 | class9正compat访问/均值 | class11正compat访问/均值 | class25正compat访问/均值 | 三者全零 |',
           '|---|---:|---:|---:|---:|---:|']
    for d,r in by_depth.items():
        cells=[f"{r['targets'][c]['positive_visits']}/{r['targets'][c]['mean_masks']:.2f}" for c in TARGETS]
        lines.append(f"| {d} | {r['visits']} | {' | '.join(cells)} | {r['all_targets_zero']} |")
    lines+=['','## 根下全部子边','', '| 动作 | 节点 | 后300轮访问 | rank | compat9 | compat11 | compat25 | 安全下一步动作数9/11/25 |','|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in root_rows:
        lines.append(f"| {r['action']} | {r['id']} | {r['last300_visits']} | {r['rank']} | {r['compat'][9]} | {r['compat'][11]} | {r['compat'][25]} | {'/'.join(str(r['health'][c]['safe_next_actions']) for c in TARGETS)} |")
    lines+=['','## 深度2访问最多的20个分支','', '| 根动作→第二动作 | 节点 | 访问 | rank | compat9 | compat11 | compat25 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in d2_rows[:20]:lines.append(f"| {r['root_action']}→{r['action']} | {r['id']} | {r['last300_visits']} | {r['rank']} | {r['compat'][9]} | {r['compat'][11]} | {r['compat'][25]} |")
    lines+=['','## selection丢失兼容性','', '| 类型/class | 父节点仍兼容次数 | 候选含兼容边 | 选中仍兼容 | 选后归零 |','|---|---:|---:|---:|---:|']
    for k,r in report['selection_summary'].items():lines.append(f"| {k} | {r['parent_viable']} | {r['eligible_viable_exists']} | {r['chosen_viable']} | {r.get('target_eliminated',0)} |")
    lines+=['','## rollout丢失兼容性的第一步原因','', '| class | 兼容起点轮数 | 中位首次丢失步数 | 采样未含安全动作 | top-k剔除安全动作 | 概率抽样选了不兼容动作 |','|---|---:|---:|---:|---:|---:|']
    for c in (*TARGETS,'any'):
        rr=rollout_losses[c];lost=rollout_eliminations[c]
        start=sum((sum(r['rollout_start_compat'].values()) if c=='any' else r['rollout_start_compat'][c])>0 for r in iteration_rows)
        lines.append(f"| {c} | {start} | {statistics.median([r['rollout_step'] for r in lost]) if lost else '—'} | {rr['sample_omitted_safe_actions']} | {rr['top_k_removed_safe_actions']} | {rr['softmax_chose_incompatible']} |")
    lines+=['','每类只计从正compat降到零的首次步骤；any表示三类兼容mask的并集变空。三个class的丢失统计不能相加作为轮数。完整节点health、安全动作列表、未展开安全动作、UCB父节点和每轮路径见data.json。','']
    conclusion=['## 结论','',
      '这次数据不支持“最后300轮主要访问的浅层UCB节点对9/11/25兼容性都不好”的解释。所有40条根子边对三者均非零，实际访问的18个根子节点也均非零；深度2仍兼容三者的访问比例分别为97%、97.7%、99%。访问加权mask数与根子边均匀参考接近，没有明显集中在低compat根分支。','',
      '这里“浅层尚有余量”只表示保留多个参考mask和大量可继续保留目标的一步动作，不表示多步成功概率高。实际根子节点安全下一步比例约90.9%/96.6%/98.7%，深度2约80.9%/89.7%/94.4%；深度3/4开始明显收窄，不能将浅层正compat直接当作整条rollout健壮。','',
      '主要丢失位置在rollout：295/300轮起点仍兼容至少一类，最终全部归零，中位首次全丢失在rollout第6步。273次全丢失发生时top-k池仍有保留目标的动作，却抽中了不兼容动作；22次由top-k把兼容动作全部排除。没有一次全丢失是因为最初候选采样未包含兼容动作。','',
      '因此后续若排查未命中的原因，应重点检查rollout为何无法连续保留这些目标的compat，而不能仅归因于浅层UCB访问了不可能通往它们的状态。本报告仅分析已有trace，没有修改搜索或追加实验。','']
    lines[2:2]=conclusion
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['root_compatible_missing','depth_summary','root_uniform_reference','selection_summary','rollout_summary']},ensure_ascii=False))

if __name__=='__main__':main()

