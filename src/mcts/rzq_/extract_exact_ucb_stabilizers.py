"""Extract saved hits and setwise stabilizers, without rerunning MCTS."""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from baseline.orbit_blocks import build_state_group_permutations
from mcts.queue_search import QueueSearchConfig, build_class_scorer
from mcts.rzq_.symmetry_quotient import build_partition_block_maps, parent_stabilizer_block_maps

TARGET = {7, 8, 9, 10, 11, 12, 15, 19, 20, 22}
BASE = Path(__file__).resolve().parent / 'runs'
OUT = BASE / 'v1_v3_v4_exact_ucb_stabilizers'

def cycles(p):
    seen, parts = set(), []
    for i in range(len(p)):
        if i in seen:
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j); c.append(j); j = p[j]
        if len(c) > 1:
            parts.append('(' + ' '.join(map(str, c)) + ')')
    return ''.join(parts) or 'e'

def main():
    OUT.mkdir(exist_ok=True)
    _, blocks = build_class_scorer(QueueSearchConfig(), 8)
    assert len(blocks) == 40
    group = build_state_group_permutations()
    maps = build_partition_block_maps(blocks)
    bindex = {frozenset(b): i for i, b in enumerate(blocks)}
    lifts = defaultdict(list)
    for gi, g in enumerate(group):
        images = [bindex.get(frozenset(g[v] for v in b)) for b in blocks]
        if None not in images:
            lifts[tuple(images)].append(gi)
    nodes, hits, sources = {}, [], {}
    def node_record(selected):
        key = tuple(sorted(selected))
        if key in nodes:
            return nodes[key]
        vertices = sorted({v for b in key for v in blocks[b]})
        vset = set(vertices)
        full = [i for i, g in enumerate(group) if {g[v] for v in vertices} == vset]
        smaps = parent_stabilizer_block_maps(tuple(int(i in key) for i in range(40)), maps)
        mids = [maps.index(m) for m in smaps]
        partition = sorted(i for m in smaps for i in lifts[m])
        assert set(partition) <= set(full)
        record = dict(id=f'N{len(nodes):03}', selected_blocks=list(key), selected_vertices=vertices,
                      block_stabilizer_order=len(mids), block_element_ids=mids,
                      partition_vertex_stabilizer_order=len(partition), partition_vertex_element_ids=partition,
                      full_bell_stabilizer_order=len(full), full_bell_element_ids=full)
        nodes[key] = record
        return record
    for version in ('v1', 'v3', 'v4'):
        run = BASE / f'class8_{version}_300_trace'
        env = json.loads((run / 'environment.json').read_text())
        for f in ('src/baseline/orbit_blocks.py', 'src/baseline/bell322.py', 'data/facet_classes_322_examples.txt'):
            assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == env['sha256'][f], f
        events = defaultdict(list)
        trace = run / 'decisions.jsonl'
        sources[version] = dict(path=str(trace.relative_to(ROOT)), sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        for line_no, line in enumerate(trace.open(), 1):
            e = json.loads(line); e['source_line'] = line_no
            events[e['iteration']].append(e)
        counts = Counter()
        for it, es in sorted(events.items()):
            ts = [e for e in es if e['event'] == 'terminal_observation']
            if not ts or ts[-1]['label'] not in {f'exact:class{c}' for c in TARGET}:
                continue
            terminal = ts[-1]; cid = int(terminal['label'].split('class')[1]); counts[cid] += 1
            visits = [e for e in es if e['event'] == 'node_visit']
            ucbs = [e for e in es if e['event'] == 'ucb_choice']
            expansions = [e for e in es if e['event'] == 'expand_action']
            rollouts = [e for e in es if e['event'] == 'rollout_choice']
            assert len(visits) == len(ucbs) + 1
            route = []
            for k, visit in enumerate(visits):
                n = node_record(visit['selected_blocks'])
                u = ucbs[k] if k < len(ucbs) else None
                if u:
                    assert visit['path'] == u['node_path']
                route.append(dict(node_id=n['id'], path=visit['path'], rank=visit['rank'],
                                  source_line=visit['source_line'], ucb_choice=u))
            for a, b in zip(rollouts, rollouts[1:]):
                assert a['current_path'] + [a['chosen_action']] == b['current_path']
            if rollouts:
                assert rollouts[-1]['current_path'] + [rollouts[-1]['chosen_action']] == terminal['path']
            hits.append(dict(version=version, iteration=it, class_id=cid, ucb_actions=[u['chosen_action'] for u in ucbs],
                             tree_nodes=route, expansions=expansions, rollout_actions=[r['chosen_action'] for r in rollouts],
                             rollout_events=rollouts, terminal=terminal,
                             symmetry_events=[e for e in es if e['event'] == 'symmetric_node_reuse']))
        result = json.loads((run / 'result.json').read_text())['result']
        expected = {c: result['encountered_label_counts'].get(f'exact:class{c}', 0) for c in TARGET}
        assert all(counts[c] == n for c, n in expected.items())
    catalog = dict(blocks=blocks, full_group_order=len(group), partition_group_order=sum(map(len, lifts.values())),
                   effective_block_group_order=len(maps),
                   block_elements=[dict(id=i, permutation=m, cycles=cycles(m), full_bell_lifts=lifts[m]) for i,m in enumerate(maps)],
                   full_bell_elements=[dict(id=i, permutation=g, cycles=cycles(g)) for i,g in enumerate(group)])
    (OUT / 'group_elements.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'trajectories.json').write_text(json.dumps(dict(target_classes=sorted(TARGET), sources=sources, nodes=list(nodes.values()), hits=hits), ensure_ascii=False, indent=2)+'\n')
    lines = ['# v1 / v3 / v4 指定 exact class 的轨迹与稳定子群', '',
             '筛选集合：`{7,8,9,10,11,12,15,19,20,22}`。保留全部命中，包括重复 class。所有编号均从 0 开始。', '',
             '稳定子群按已选 block 集合的**集合稳定子**计算，允许群元素交换已选 block，不要求逐点固定。',
             f'完整 Bell 群 G 阶数 {len(group)}；保持分区的顶点群 P 阶数 {sum(map(len,lifts.values()))}；其有效 block 置换群 B 阶数 {len(maps)}。',
             '节点表同时给出 B 内稳定子、P 内稳定子和完整 G 内稳定子的阶数。B 的作用可能有核，不能把 block 群阶数直接当作顶点群阶数。', '',
             '实际访问状态取自 node_visit.selected_blocks；path 是节点记录的路径。v3 含闭包和对称节点复用，不能简单累加 UCB action 来代替实际状态。',
             '节点链含根及最后一个 UCB 到达的扩展父节点；最后节点没有 UCB 选择时记为“扩展”。expansion 与 rollout 分开记录。v3 的 minimum_visits 选择也保留在 UCB 阶段。', '',
             '完整机器可读轨迹与原始行号见 [trajectories.json](trajectories.json)；全部具体群元素（置换与循环记号）见 [group_elements.json](group_elements.json)。', '',
             '## 命中统计', '', '| class | v1 | v3 | v4 |','|---|---:|---:|---:|']
    for c in sorted(TARGET):
        lines.append('| '+str(c)+' | '+' | '.join(str(sum(h['version']==v and h['class_id']==c for h in hits)) for v in ('v1','v3','v4'))+' |')
    lookup = {n['id']: n for n in nodes.values()}
    for version in ('v1','v3','v4'):
        lines += ['', f'## {version}', '']
        for h in [h for h in hits if h['version']==version]:
            lines += [f"### iteration {h['iteration']} → class{h['class_id']}", '',
                      f"- UCB actions：`{h['ucb_actions']}`",
                      f"- expansion：`{[e['action'] for e in h['expansions']]}`",
                      f"- rollout：`{h['rollout_actions']}`",
                      f"- terminal path：`{h['terminal']['path']}`", '',
                      '| 节点 | 实际已选 blocks | 记录 path | rank | 下一步 | B 稳定子阶 | P 稳定子阶 | G 稳定子阶 |',
                      '|---|---|---|---:|---|---:|---:|---:|']
            for r in h['tree_nodes']:
                n=lookup[r['node_id']]; u=r['ucb_choice']
                step = f"UCB {u['chosen_action']} ({u.get('selection_phase','ucb')})" if u else '扩展'
                lines.append(f"| [{n['id']}](#{n['id'].lower()}) | `{n['selected_blocks']}` | `{r['path']}` | {r['rank']} | {step} | {n['block_stabilizer_order']} | {n['partition_vertex_stabilizer_order']} | {n['full_bell_stabilizer_order']} |")
            lines += ['']
    lines += ['## 节点稳定子群元素', '', 'B_i 表示 group_elements.json 中 block_elements[id=i]；G_i 表示 full_bell_elements[id=i]。以下列出每个节点的全部 B 元素；P、G 的全部元素 ID 收录在 trajectories.json 的对应节点中。', '']
    for n in nodes.values():
        lines += [f"### {n['id']}", '', f"已选 blocks：`{n['selected_blocks']}`。", '',
                  f"B 稳定子 = `{{{', '.join('B'+str(i) for i in n['block_element_ids'])}}}`", '']
        for i in n['block_element_ids']:
            lines += [f"- B{i} = `{cycles(maps[i])}`"]
        lines += ['', f"P 稳定子元素：`{n['partition_vertex_element_ids']}`", '',
                  ('G 稳定子为完整 G（全部 3072 个元素）。' if len(n['full_bell_element_ids'])==len(group) else f"G 稳定子元素：`{n['full_bell_element_ids']}`"), '']
    lines += ['## 核查', '', '已校验：筛选命中数与各 result.json 完全一致；UCB 选择与节点访问一一对应；rollout 路径连续且与终点一致；每个 P 稳定子包含于对应 G 稳定子。',
              '构造 block 分区依赖的 orbit_blocks.py、bell322.py 与输入 examples 数据 SHA-256 均与三次实验环境记录一致。未重新运行搜索。']
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(hits=len(hits), unique_nodes=len(nodes), versions=dict(Counter(h['version'] for h in hits)), group_orders=[len(group),sum(map(len,lifts.values())),len(maps)], output=str(OUT)), indent=2))

if __name__ == '__main__':
    main()
