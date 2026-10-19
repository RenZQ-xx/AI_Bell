# v23: 定向 pattern 层级与停滞后 Filler 再升温

## 目标

v22 同时启用了几何节点合并和分层 pattern policy。固定 seed 的消融显示：

- 仅启用几何合并，i3000 为 44/46；
- 仅启用分层 policy，i3000 为 45/46；
- 两者同时启用，i3000 为 43/46。

几何节点合并会把具有不同合法动作与先验的 pattern context 写进同一个 policy
node，破坏 MCTS 状态的 Markov 性；它与 hierarchy 的交互是 v22 回退的主要来源。
v23 因此保留 hierarchy，但恢复 v21 的 context-sensitive node key，不启用
`geometric_transpositions`。

## 定向 partition 关系

v22 只检查 target 是否细化 current，把反方向的真包含也记成 `intersection`。
v23 使用有向 partition certificate：

- `continue`：两个 pattern 相同；
- `refine`：target 的每个 orbit block 都落在 current 的一个 block 中；
- `coarsen`：current 细化 target；
- `intersection`：双方均不存在 partition 包含；
- `identity`：显式回到 singleton pattern。

例如从 identity singleton pattern 进入 pair pattern 是 `coarsen`，不再被误记为
`intersection`。Progressive widening 使用固定的 relation quota 轮转，然后才在
对应 relation 内依据 pattern value、transition value、coherence 和大小 bucket
选择动作。该判定只读取已注册的 orbit partition，不读取 class 样例。

## Relation UCB

知识库新增 `(role, relation)` 在线统计。Filler 与 Corrector 分别记录每种 relation
的访问次数、平均回报、rank 变化、有效终端数和新 class 数。动作先验由固定的小额
结构 bonus 与带 optimism 的 relation value 共同组成；单次极端 novelty reward 在
进入 relation prior 前会被截断，避免一个偶然发现永久垄断 relation lane。

旧入口默认不更新 relation stats，因而 v20/v21/v22 的选择行为保持隔离。

## 双停滞 Filler 再升温

v21 在覆盖接近完成时会把自由 Filler 比例压到约 10%，这有利于 Corrector 兑现
已知 frontier，但也可能缺少构造新源 facet basin 的预算。v23 只有同时满足以下条件
才把自由 Filler 比例临时提高到 35%：

1. 距离最后一次新 class 已经过 512 次 simulation；
2. 距离 Corrector 首次到达新 canonical target 已经过 256 次 simulation；
3. 当前实际终端覆盖已经达到 adaptive replay 阈值。

首次实验曾把“新 source-target pair”也当作 Corrector frontier 进展。旧 target 间的
组合 pair 一直增长到 i5888，使升温完全不触发。最终实现只让首次新 target 重置
productive 时钟；pair 进展仍单独记录，不能给 Corrector 无限续命。升温不改变
rollout 动作空间，也不使用缺失 class 的身份。

## 入口与隔离

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.hierarchical_search --iterations 6000
```

该入口启用 v21 ridge/adaptive Corrector、hierarchical policy、directed relation、
relation UCB 和 Filler reheat，但明确不启用几何 policy-node 合并。直接运行
`mcts.group_game`、`ridge_corrector_search` 和 `scalable_search` 仍分别对应
v20、v21 和 v22。

## 固定 seed 测试

环境为 `PYTHONHASHSEED=0`、BLAS/OMP 单线程、seed 3222026、初始 known classes
为空、无中间快照。

| 运行 | 完成迭代 | 覆盖 | class 3 | class 24 | class 41 | 缺失 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| v21 i3000 | 3000 | 44/46 | 2257 | 未发现 | 已发现 | 24, 34 | 157.8994 s |
| v22 i3000 | 3000 | 43/46 | 1778 | 未发现 | 未发现 | 24, 34, 41 | - |
| v23 i6000 | 6000 | 45/46 | 2576 | 2341 | 2282 | 34 | 339.7004 s |

v23 在 i3000 已达到 45/46，说明关闭几何 policy-node 合并、补齐 `coarsen` 后，
早期覆盖恢复且 class 24/41 不再丢失。最后新增为 class 9（i2977），随后到 i6000
仍未发现 class 34。

严格 target-progress 版本在 i3489 启动升温，产生 2478 个自由 Filler slot 和
4214 个唯一终端 support；把 pair 也视为 progress 的消融没有触发升温，只有
1851 个自由 slot 和 3889 个唯一终端 support。升温将样本覆盖增加 8.36%，但耗时
由 311.5489 秒增至 339.7004 秒，仍未改变 class 覆盖。因此它是有效的探索机制，
尚不是 class 34 的收口解。

完整结果为：

- `runs/single_tree_group_game_i6000_v23_deterministic.json`
- `runs/single_tree_group_game_i6000_v23_pair_progress_ablation.json`

核心回归测试为 104 passed。

## 已定位的下一瓶颈

class 34 在 v21/v22 都通过 `OBS-03cf50faaf05fc30` 的 orbit 填充进入源 facet，再由
一次 rewrite 和 external exit 命中。但这个 ID 由 canonical support 生成，而其
`blocks` 来自首次发现时的原始坐标朝向。v21、v22、v23 对同一 ID 实际保存了三个
不同的 orbit partition；`register_pattern(setdefault)` 会丢弃后来出现的其他朝向。

因此 v23 增加自由 Filler 后仍只能重复当前保留的 embedding，无法学习同一
canonical symmetry family 的其他具体作用。下一步应把知识拆成两层：canonical
`PatternFamily` 共享统计，concrete `PatternEmbedding` 保存坐标相关 blocks 和动作；
MCTS 节点继续保持 context-sensitive，不重新引入 v22 的错误 policy-node 合并。
