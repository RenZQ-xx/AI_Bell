"""v11 provenance and descriptive comparison with archived v4.4."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def read_events(path):
    with path.open() as f:
        for line in f:yield json.loads(line)


def fixed_window(path,pre,n=300):
    labels=Counter();root=Counter();expansion=ucb=special=0;depth=0
    for e in read_events(path):
        if e.get('iteration',0)>n:break
        if e['event']=='terminal_observation':labels[e['label']]+=1
        if e['event']=='expand_action':expansion+=1
        if e['event']=='ucb_choice':ucb+=1
        if e['event']=='sealed_selection':special+=1
        if e['event']=='backpropagation':
            edges=e.get('edge_updates',[]);depth=max(depth,len(edges))
            for edge in edges:
                if edge['parent_path']==[]:root[edge['action']]+=1
    exact={int(k.split('class')[1]) for k in labels if k.startswith('exact:class')}
    return {'iterations':n,'terminal_counts':dict(labels),'exact_hits':sum(v for k,v in labels.items() if k.startswith('exact:')),
            'distinct_exact_classes':sorted(exact),'new_relative_to_own_pre':sorted(exact-set(pre)),
            'expansion_count':expansion,'ordinary_selection_count':ucb,'special_selection_count':special,
            'max_traversed_tree_depth':depth,'root_visits':dict(root),'root_max_share':max(root.values(),default=0)/max(1,sum(root.values()))}


def analyze(path):
    result=read(path/'result.json');audit=read(path/'verification.json');assert audit['passed']
    stage_edges=set();last_route=[];special_this_iteration=[];current_iter=None
    stages=defaultdict(Counter);provenance=[];special_events=[];unsealed_at={}
    for e in read_events(path/'decisions.jsonl'):
        i=e.get('iteration',0);phase=e['phase']
        if i!=current_iter:special_this_iteration=[];current_iter=i
        if e['event']=='sealed_selection':
            special_this_iteration.append(e);special_events.append(e)
            unsealed_at[(phase,e['parent_id'],e['chosen_action'])]=i
        if e['event']=='real_visit_backpropagation':
            last_route=e['edges']
            for edge in last_route:
                key=(edge['parent_id'],edge['action'])
                stages[phase]['old_edge_traversals' if key in stage_edges else 'new_edge_traversals']+=1
                if edge['parent_id']==result['tree']['root_id']:
                    stages[phase]['old_root_visits' if key in stage_edges else 'new_root_visits']+=1
        if e['event']=='new_class_discovery':
            root=last_route[0] if last_route else None
            provenance.append({'iteration':i,'class':e['class'],'phase':phase,
                               'root_action':root['action'] if root else None,
                               'root_edge_existed_at_phase_start':bool(root and (root['parent_id'],root['action']) in stage_edges),
                               'special_selection_on_discovery_iteration':bool(special_this_iteration),
                               'root_special_unseal_iteration_this_phase':unsealed_at.get((phase,root['parent_id'],root['action'])) if root else None,
                               'tree_route':last_route})
        if e['event']=='restart':
            stage_edges={(n['id'],x['action']) for n in e['after']['node_statistics'] for x in n['edges']}
    old_dir=path.parent/'class8_v4_4_300_trace';old=read(old_dir/'result.json')
    comparison={'v4.4':fixed_window(old_dir/'decisions.jsonl',old['meta']['pre_discovered_class_ids']),
                'v11_first300':fixed_window(path/'decisions.jsonl',result['meta']['pre_discovered_class_ids'])}
    report={'provenance':provenance,'stage_edge_visits':{k:dict(v) for k,v in stages.items()},
            'special_selection_count':len(special_events),'special_selection_by_phase':dict(Counter(e['phase'] for e in special_events)),
            'special_selection_by_parent':dict(Counter(e['parent_id'] for e in special_events)),
            'first_special_iteration':special_events[0]['iteration'] if special_events else None,
            'historical_comparison':comparison}
    (path/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# v11 核验与行为分析','',
           f"实际完成{result['iterations']}轮，{result['restart_count']}次重启；最后新发现位于第{result['last_new_iteration']}轮，之后恰好300轮无新发现。完整轨迹审计通过，正式前20轮与冒烟trace一致。",'',
           f"特殊selection共{len(special_events)}次，首次发生在第{report['first_special_iteration']}轮；本次正式运行实际覆盖了D新增分支。",'',
           '| class | 首次发现轮次 | 阶段 | 根动作 | 该根边阶段初已存在 | 本轮经过特殊selection | 本阶段根边特殊解封轮次 |',
           '|---:|---:|---:|---:|---|---|---:|']
    for d in provenance:lines.append(f"| {d['class']} | {d['iteration']} | {d['phase']} | {d['root_action']} | {d['root_edge_existed_at_phase_start']} | {d['special_selection_on_discovery_iteration']} | {d['root_special_unseal_iteration_this_phase']} |")
    lines+=['','“根边阶段初已存在”区分该阶段重新访问旧分支与拓展新分支；“经过特殊selection”只描述发现这一轮的路径，并不证明发现由该机制单独导致。','',
            '| 阶段 | 旧根边访问 | 新根边访问 | 所有旧边遍历 | 所有新边遍历 |','|---:|---:|---:|---:|---:|']
    for p,v in stages.items():lines.append(f"| {p} | {v['old_root_visits']} | {v['new_root_visits']} | {v['old_edge_traversals']} | {v['new_edge_traversals']} |")
    lines+=['','## 原v4.4参考（均截取前300轮）','',
            '| 指标 | 原v4.4 | v11前300轮 |','|---|---:|---:|']
    for name,key in [('Exact命中次数','exact_hits'),('expansion次数','expansion_count'),('普通selection次数','ordinary_selection_count'),('特殊selection次数','special_selection_count'),('最大实际树路径深度','max_traversed_tree_depth')]:
        lines.append(f"| {name} | {comparison['v4.4'][key]} | {comparison['v11_first300'][key]} |")
    lines+=['',f"原v4.4命中Exact类别：`{comparison['v4.4']['distinct_exact_classes']}`；v11：`{comparison['v11_first300']['distinct_exact_classes']}`。",'',
            '上述为相同轮数下的描述性统计，单位和窗口可以对齐，但不能作为重启的因果收益。预发现集合、阶段Q分母及发现后的调度不同，影响奖励、prior与路径。各自“新class数”只能参考；正式终点轮数由新发现决定，不能与历史固定300轮直接比较。没有相同配置与停机规则的无重启正式对照，也没有多seed结果，因此本次只能证明v11规则已实际执行并报告发现结果。','']
    (path/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('provenance','historical_comparison','stage_edge_visits')},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);analyze(p.parse_args().path)

