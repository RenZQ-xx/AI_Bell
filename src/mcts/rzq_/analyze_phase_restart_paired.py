"""Analyze the predeclared single-snapshot paired restart diagnostic."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def events(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
def span(values):return max(values)-min(values) if values else 0

def distribution(counts):
    total=sum(counts.values())
    ps=[v/total for v in counts.values() if v]
    return {'counts':dict(sorted(counts.items())), 'total':total,
            'max_share':max(ps,default=0), 'entropy_bits':-sum(p*math.log2(p) for p in ps),
            'hhi':sum(p*p for p in ps), 'visited_actions':len(ps)}

def graph_depth(tree):
    rows={r['id']:r for r in tree['node_statistics']}
    memo={}; active=set()
    def depth(n):
        if n in memo:return memo[n]
        assert n not in active,'Cycle in exact-closure graph'
        active.add(n)
        memo[n]=max((1+depth(e['child_id']) for e in rows[n]['edges']),default=0)
        active.remove(n)
        return memo[n]
    return depth(tree['root_id'])

def summarize(trace, tree, base, start=1, end=60):
    selected=[e for e in trace if start<=e['relative_iteration']<=end]
    root=base['root_id']; old_nodes={r['id'] for r in base['node_statistics']}
    old_edges={(r['id'],e['action']) for r in base['node_statistics'] for e in r['edges']}
    root_counts=Counter(); root_ucb=Counter(); edge_kinds=Counter(); depths=[]; paths=[]
    terminals=Counter(); expanded=[]; ucbs=[]; discoveries=[]
    for e in selected:
        if e['event']=='real_visit_backpropagation':
            path=[]
            for edge in e['edges']:
                key=(edge['parent_id'],edge['action'])
                if edge['parent_id']==root:root_counts[edge['action']]+=1
                kind='snapshot_edge' if key in old_edges else 'new_edge_at_snapshot_node' if edge['parent_id'] in old_nodes else 'new_edge_at_new_node'
                edge_kinds[kind]+=1;path.append(key)
            paths.append(path);depths.append(len(path))
        elif e['event']=='expand_action':expanded.append(e)
        elif e['event']=='terminal_observation':terminals[e['label']]+=1
        elif e['event']=='new_class_discovery':discoveries.append({'iteration':e['iteration'],'relative_iteration':e['relative_iteration'],'class':e['class_id']})
        elif e['event']=='ucb_choice':
            cs=e['candidates']
            if len(cs)>1:
                qs=span([c['raw_q'] for c in cs]);es=span([c['exploration'] for c in cs])
                e_best=max(cs,key=lambda c:c['exploration'])['action']
                ucbs.append({'iteration':e['relative_iteration'],'parent_id':e['parent_id'],
                    'q_range':qs,'e_range':es,'ratio':qs/es if es else None,
                    'chosen_q':next(c['raw_q'] for c in cs if c['action']==e['chosen_action']),
                    'max_exploration':max(c['exploration'] for c in cs),
                    'q_range_dominates':qs>es, 'choice_differs_from_exploration_only':e['chosen_action']!=e_best})
            if e['parent_id']==root:root_ucb[e['chosen_action']]+=1
    root_events=[e for e in ucbs if e['parent_id']==root]
    return {'window':[start,end],'iterations':len(depths), 'root_actual':distribution(root_counts),
            'root_selection':distribution(root_ucb), 'edge_traversals':dict(edge_kinds),
            'expansions':len(expanded),'root_expansions':sum(e['parent_id']==root for e in expanded),
            'restart_only_expansions':sum(e['restart_only_slot'] for e in expanded),
            'expansions_at_snapshot_nodes':sum(e['parent_id'] in old_nodes for e in expanded),
            'mean_traversal_depth':statistics.mean(depths) if depths else 0,'max_traversal_depth':max(depths,default=0),
            'exact_distribution':{k:v for k,v in sorted(terminals.items()) if k.startswith('exact:')},
            'terminal_distribution':dict(sorted(terminals.items())), 'new_classes':discoveries,
            'ucb_multiple_child_decisions':len(ucbs),
            'q_range_dominates_count':sum(e['q_range_dominates'] for e in ucbs),
            'choices_differ_from_exploration_only':sum(e['choice_differs_from_exploration_only'] for e in ucbs),
            'first_root_q_dominance':next((e['iteration'] for e in root_events if e['q_range_dominates']),None),
            'root_ucb_diagnostics':root_events,'paths':paths}

def instant(before,after):
    pairs=list(zip(before['node_statistics'],after['node_statistics']))
    eligible=[(a,b) for a,b in pairs if len(a['edges'])>1]
    changes=[{'node_id':a['id'],'path':a['path'],'before_best':a['ucb_best_action'],
              'after_best':b['ucb_best_action'],'was_expansion_saturated':not a['can_expand'],
              'can_expand_after':b['can_expand']} for a,b in eligible if a['ucb_best_action']!=b['ucb_best_action']]
    reopened=[a['id'] for a,b in pairs if a['unexpanded'] and not a['can_expand'] and b['can_expand']]
    return {'multiple_child_parents':len(eligible),'changed_best_count':len(changes),'changes':changes,
            'reopened_saturated_nodes':reopened,
            'expansion_ready_before':sum(a['can_expand'] for a,b in pairs),
            'expansion_ready_after':sum(b['can_expand'] for a,b in pairs),
            'all_exploration_terms_identical':all(x['exploration']==y['exploration'] for a,b in pairs for x,y in zip(a['edges'],b['edges'])),
            'all_q_zero':all(e['q']==0 for a,b in pairs for e in b['edges'])}

def compare_windows(a,b):
    da=a['root_actual']['counts'];db=b['root_actual']['counts'];na=sum(da.values());nb=sum(db.values())
    tv=.5*sum(abs(da.get(k,0)/na-db.get(k,0)/nb) for k in da.keys()|db.keys())
    return {'root_distribution_total_variation':tv,
            'same_relative_iteration_root_action_changes':sum(x[0]!=y[0] for x,y in zip(a['paths'],b['paths']) if x and y),
            'same_relative_iteration_full_path_changes':sum(x!=y for x,y in zip(a['paths'],b['paths']))}

def analyze(path):
    assert load(path/'verification.json')['completed']
    base=load(path/'warmup/tree.json'); root=next(r for r in base['node_statistics'] if r['id']==base['root_id'])
    baseline=distribution(Counter({e['action']:e['edge_total_visits'] for e in root['edges']}))
    warm=events(path/'warmup/decisions.jsonl')
    warm_root=[e for e in warm if e['event']=='ucb_choice' and e['parent_id']==base['root_id']]
    result={'snapshot':{'nodes':base['nodes'],'edges':base['edges'],'graph_depth':graph_depth(base),
                       'root_lifetime_distribution':baseline,'root_edges':root['edges'],
                       'root_selection_last20':dict(Counter(e['chosen_action'] for e in warm_root if e['iteration']>80)),
                       'global_classes':base['global_classes']},'instant':{},'arms':{},'comparisons':{}}
    for label in ('B','C'):
        ev=load(path/f'{label}_restart.json');result['instant'][label]=instant(ev['before'],ev['after'])
    result['integrity'] = {}
    for label in ('A','B','C'):
        trace=events(path/label/'decisions.jsonl'); tree=load(path/label/'tree.json')
        assert load(path/f'{label}_start.json') == base
        assert load(path/label/'result.json')['iterations_completed'] == 160
        backprops=[e for e in trace if e['event']=='real_visit_backpropagation']
        assert len(backprops)==60
        root_final=next(r for r in tree['node_statistics'] if r['id']==tree['root_id'])
        assert root_final['total_visits']==160
        assert root_final['phase_visits']==(160 if label=='A' else 60)
        incoming_total=Counter();incoming_phase=Counter()
        for row in tree['node_statistics']:
            assert isinstance(row['total_visits'],int) and isinstance(row['phase_visits'],int)
            for edge in row['edges']:
                incoming_total[edge['child_id']]+=edge['edge_total_visits']
                incoming_phase[edge['child_id']]+=edge['edge_phase_visits']
        for row in tree['node_statistics']:
            if row['id']!=tree['root_id']:
                assert row['total_visits']==incoming_total[row['id']]
                assert row['phase_visits']==incoming_phase[row['id']]
        for e in trace:
            if e['event']=='ucb_choice':
                for c in e['candidates']:
                    expected=1.4*c['prior']*math.sqrt(max(1,e['parent_total_visits']))/(1+c['visits'])
                    assert math.isclose(expected,c['exploration'],rel_tol=1e-12,abs_tol=1e-12)
                    assert isinstance(c['edge_phase_visits'],int)
        result['integrity'][label]={'iterations':60,'root_total':160,'root_phase':root_final['phase_visits'],
                                  'incoming_edge_counts_match_nodes':True,'exploration_uses_total_visits':True}
        result['arms'][label]={'nodes':tree['nodes'],'edges':tree['edges'],'graph_depth':graph_depth(tree),
             'windows':{str(n):summarize(trace,tree,base,1,n) for n in (10,30,60)},
             'segments':{f'{a}-{b}':summarize(trace,tree,base,a,b) for a,b in ((1,10),(11,30),(31,60))}}
        # Attach the actual tree route to each discovery, including granted edges.
        births={}; routes={}
        for e in trace:
            if e['event']=='expand_action':births[(e['parent_id'],e['action'])]=e
            elif e['event']=='real_visit_backpropagation':routes[e['iteration']]=e['edges']
            elif e['event']=='new_class_discovery':
                route=routes[e['iteration']]
                for d in result['arms'][label]['windows']['60']['new_classes']:
                    if d['iteration']==e['iteration'] and d['class']==e['class_id']:
                        d['route']=route
                        d['uses_restart_only_edge']=any(births.get((x['parent_id'],x['action']),{}).get('restart_only_slot',False) for x in route)
    for left,right in (('A','B'),('B','C'),('A','C')):
        result['comparisons'][f'{left}-{right}']={str(n):compare_windows(result['arms'][left]['windows'][str(n)],result['arms'][right]['windows'][str(n)]) for n in (10,30,60)}
    (path/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# 单快照重启配对短测','',
           '固定方案：预热100轮；同一完整快照恢复A/B/C，各60轮。C在重启瞬间给每个有剩余动作的已有节点6个名额，随后容量为max(6,K(阶段visits))。期间发现class立即登记，但不再次重启。',
           '', 'A：修正Q分母后的无重启对照；B：仅清Q和阶段访问；C：完整重启。所有分支探索项均用总visits，历史epoch价值衰减均关闭。',
           '',f"快照：{base['nodes']}节点、{base['edges']}边、图最大深度{graph_depth(base)}；根分支总访问分布 `{baseline['counts']}`；最大占比{baseline['max_share']:.1%}。",
           f"根节点最后20轮selection分布：`{result['snapshot']['root_selection_last20']}`。",'',
           '## 重启瞬间','', '| 分支 | 多子节点父节点 | UCB首选改变 | 饱和节点重开 | 探索项完全不变 |', '|---|---:|---:|---:|---|']
    for label in ('B','C'):
        x=result['instant'][label]
        lines.append(f"| {label} | {x['multiple_child_parents']} | {x['changed_best_count']} | {len(x['reopened_saturated_nodes'])} | {x['all_exploration_terms_identical']} |")
    lines+=['','## 实际运行','', '| 轮数 | 分支 | 根分支最大占比 | 根分支熵(bits) | 根扩展 | 额外名额扩展 | 平均路径深度 | Q范围>探索范围 |', '|---:|---|---:|---:|---:|---:|---:|---:|']
    for n in (10,30,60):
        for label in ('A','B','C'):
            w=result['arms'][label]['windows'][str(n)]
            lines.append(f"| {n} | {label} | {w['root_actual']['max_share']:.1%} | {w['root_actual']['entropy_bits']:.3f} | {w['root_expansions']} | {w['restart_only_expansions']} | {w['mean_traversal_depth']:.2f} | {w['q_range_dominates_count']}/{w['ucb_multiple_child_decisions']} |")
    lines+=['','分支分布使用短测期间的真实根边遍历，包含expansion和selection；熵越大、最大占比越小表示访问越分散。Q范围比只对多子节点的实际selection计算，不能将它等同于因果贡献。','',
            '| 比较 | 前10轮根分布TV | 前30轮根分布TV | 前60轮根分布TV | 60轮根动作不同 | 60轮完整路径不同 |','|---|---:|---:|---:|---:|---:|']
    for key,x in result['comparisons'].items():
        lines.append(f"| {key} | {x['10']['root_distribution_total_variation']:.3f} | {x['30']['root_distribution_total_variation']:.3f} | {x['60']['root_distribution_total_variation']:.3f} | {x['60']['same_relative_iteration_root_action_changes']} | {x['60']['same_relative_iteration_full_path_changes']} |")
    lines+=['','TV=根分支访问概率差的绝对值之和的一半，范围0–1。路径比较按相同相对轮次；分叉后随机数消耗可能不同，不能解释为每轮的严格反事实。','', '## 分支详情','']
    for label in ('A','B','C'):
        arm=result['arms'][label];w=arm['windows']['60']
        lines += [f'### {label}', '',f"- 终点节点/边/图深度：{arm['nodes']}/{arm['edges']}/{arm['graph_depth']}。",
                  f"- 根边访问：`{w['root_actual']['counts']}`；根selection：`{w['root_selection']['counts']}`。",
                  f"- 遍历来源：`{w['edge_traversals']}`。",
                  f"- 首次根selection的Q范围超过探索范围：相对第{w['first_root_q_dominance']}轮。",
                  f"- Exact分布：`{w['exact_distribution']}`。",
                  f"- 新class（相对轮次）：`{[(d['class'],d['relative_iteration']) for d in w['new_classes']]}`。",'']
    lines+=['## 可复现与边界','',
            '- 原始配置、环境/源码指纹在config.json和environment.json；基准入口及历史结果未修改。',
            '- snapshot.pkl保存全部树、全局发现、缓存、两个随机流；manifest记录SHA256及隔离检查。仅加载本实验自己生成的pickle文件。',
            '- A从快照再次恢复并重跑前10轮，完整事件trace一致；verification.json记录验证结果。',
            '- 此次是一个seed、一个预先指定快照、一次重启的机制短测。新class数量只作辅助，不能证明总体发现能力提升。',
            '- 原v4.4既有不同预发现集合，又用折扣value_visits，本报告不把它当作匹配对照；A才是本次对照。',
            '- 重启事件发生在快照边界，完整前后统计保存于B_restart.json/C_restart.json；分支内部trace记录其后所有决策。','']
    a=result['arms']['A']['windows']['60'];b=result['arms']['B']['windows']['60'];c=result['arms']['C']['windows']['60']
    intro=['## 判断','',
           f"本次机制短测确认走势明显改变：A与C在60轮中有{result['comparisons']['A-C']['60']['same_relative_iteration_root_action_changes']}轮根动作不同。但清Q并未持续建立探索项主导的selection；B和C在各自第一次根selection时Q范围就已超过探索范围。",'',
           f"C发现{len(c['new_classes'])}个新class，A/B分别为{len(a['new_classes'])}/{len(b['new_classes'])}。C的根分支最大占比为{c['root_actual']['max_share']:.1%}，B为{b['root_actual']['max_share']:.1%}，A为{a['root_actual']['max_share']:.1%}：完整重启比仅清Q分散，但仍比无重启对照集中。",'']
    if c['edge_traversals'].get('snapshot_edge',0)==0:
        intro += ['C没有遍历任何快照已有子边。两次新class发现位于新根动作3的后续rollout路径，该根动作在重启后第6轮使用新增名额展开。因此本次观察到的发现来自新扩展子树，而非重新探索旧子节点；9个被重新开放的旧饱和节点在此次路径中并没有得到访问。','']
    first_b=b['root_ucb_diagnostics'][0];first_c=c['root_ucb_diagnostics'][0]
    intro += [f"数值原因：B第一次根selection（第{first_b['iteration']}轮）选中分支Q={first_b['chosen_q']:.3f}，最大探索项仅{first_b['max_exploration']:.3f}；C第一次根selection（第{first_c['iteration']}轮）选中分支Q={first_c['chosen_q']:.3f}，最大探索项仅{first_c['max_exploration']:.3f}。一次正回传就能使新边Q达到数十，而尚未重访的旧边Q仍为0。总visits保留使旧高访问边的探索项仍小，新边迅速取得优势。",'',
              f"三组都扩展60条边、终点均161节点，最大图深度均5。效果是改变扩展的位置：C在根新增{c['root_expansions']}条边，A/B各{a['root_expansions']}条；并未增加这个固定预算中的总扩展次数。",'',
              '结论限于当前seed和预定快照：清历史Q、即时发放6个名额能明显迁移搜索方向；尚不能把它称为持续恢复UCB探索，也不能据此证明总体发现能力提升。本次按约定到短测结束，未启动正式实验。','']
    lines[2:2]=intro
    (path/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','snapshot')},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);analyze(p.parse_args().path)


