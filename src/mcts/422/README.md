# Bell 4-2-2 Single-Tree Filler-Corrector MCTS

> 当前从 3-2-2 v25 迁移的隔离版本入口为
> `python -m mcts.422.v25_search`。完整设计与迁移边界见
> [`ALGORITHM_V25.md`](ALGORITHM_V25.md)。原有 v1/v2 入口保持不变。

本目录把 `mcts/group_game` v20 的核心思想迁移到 Bell (4,2,2)，并与
3-2-2 实现完全隔离。迁移版本标识为：

```text
bell422_single_tree_symmetry_filler_corrector_v1
```

## 问题规格

- 4 个 party，每个 party 有 2 个 input、2 个二值 output；
- 256 个 deterministic assignments；
- 80 个非空 correlator 坐标；
- 完整 Bell relabeling group 的阶为 98,304；
- facet 的 tight support 必须具有 affine rank 79。

`problem.py` 固化了 `422_handoff/01_422_definitions/bell422.py` 的坐标顺序和
群作用约定，因此搜索输出的 H-row 可以直接和 handoff 数据核对。

## 先验边界

搜索阶段只允许使用群论结构：

```text
02_complete_nodes/complete_nodes.csv
02_complete_nodes/nodes/*/orbit_blocks.csv
```

`atlas.py` 明确没有读取 `nodes/*/facets.csv` 的接口。索引里的 `facet_count`
字段也不会被读取或参与排序。`03_global_facets/global_facet_classes.csv` 只在候选
先通过 rank 79、supporting hyperplane 和 tight-support 闭包验证以后，用于给结果
命名；它不影响 action、prior、reward 或预算。

因此初始 `discovered` 是空集。identity pattern 只是 256 个 singleton 的划分，
不默认计入任何 facet class。

## 一棵树中的状态与动作

一个 `GameNode` 的状态为：

```text
(support_word, role, corrections_used, context_pattern_id, source_facet_word)
```

`support_word` 是一个 Python 256-bit 整数。节点在持久树中按完整状态去重，避免
把相同点集在不同 Corrector 深度或 pattern 上下文中的价值错误合并。

动作有四种结构：

- `add`：Filler 加入一个 orbit block 或若干 orbit 的 union；
- `remove`：Corrector 删除当前 facet 中的一个或多个 orbit 部分；
- `rewrite`：先删除，再从当前或相邻 pattern 加入另一块；
- `stop`：模型保留此类型，当前 v1 由预算和无动作状态隐式终止。

## Filler

Filler 不固定 block 大小。721 个 handoff subgroup pattern 的 orbit partition 采用
lazy 读取，只有 progressive widening 实际打开一个 pattern 时才加载对应 CSV。

每个节点同时进行两层 widening：

```text
K_pattern(N) = k0 + alpha * (N + 1)^beta
K_action(N)  = k0 + alpha * (N + 1)^beta
```

pattern 初次进入时按 `C2 x C2 / C4 / C6 / S3` 交错轮转；有在线经验以后，
`OnlineKnowledge` 的新 class 率、reward 和 UCB optimism 决定同一窗口内的顺序。
block action 再按 `fine / single / union / broad` 配额轮转。这保留了 rare block
组合获得后代验证机会的 progressive widening 含义。

在 rank 78 时，Filler 枚举一个剩余 block；在 rank 77 时枚举两个 block 的组合。
每次 tail 调用最多检查 `tail_checks` 个候选，且只有达到 rank 79 才调用完整 facet
验证。该 look-ahead 只查询当前枚举出口，不使用未发现 class 的支持集。

## Corrector

每次发现一个 facet 后，Corrector 从该次发现的 tight support 在线学习局部对称性：

1. 检查 15 个标准生成元，并用可复现随机游走采样更多 stabilizer 元素；
2. 对采样到的 stabilizer 生成有界 subgroup closure；
3. 由该 subgroup 在 256 个顶点上的 orbit 自动建立 `ONLINE422-*` pattern；
4. 在当前 pattern、在线 pattern 和结构邻居中轮转删除与 rewrite。

Corrector 的删除范围包括单 orbit、多 orbit 和 orbit 与 support 的交集，不要求固定
block 大小。rewrite 可以跨到生成元 token 和 orbit-count 意义下的相邻 pattern。
这些新 pattern 来自本次搜索结果，不来自 handoff facet 样例。

## Rollout 与 Reward

rollout portfolio 为 8 次一周期：4 次 coherent、2 次 neighbor、1 次 mixed、
1 次 broad。每种策略使用由 `(seed, iteration, depth, policy)` 派生的独立 RNG，
但共享 rank cache、terminal cache、discovered 集合和结构知识库。

主要 reward：

- 全局首次出现的 canonical facet class：`+10`；
- 已发现 class：从 `+0.10` 起按命中频率衰减；
- rank 增长：每级 `+0.035`；
- 无效 rank-79 terminal：`-1.25`；
- dead end：`-0.60`；
- Corrector 删除按顶点数收取小成本。

一旦发现新 class，knowledge epoch 递增。PUCT 优先读取当前 epoch 的 edge value，
旧 epoch 只保留折扣后的结构经验，从而让“新 class”奖励保持动态全局语义。

## 终端分类

`geometry.py` 依次执行：

1. affine rank 79 检查；
2. SVD 求候选超平面并检查 256 个点是否位于同一侧；
3. 将法向量恢复为 primitive integer H-row；
4. 以精确整数重新计算完整 tight support；
5. 在完整 98,304 元 relabeling orbit 中取字典序最大 H-row；
6. 最后才查询只读 class catalog。

rank、terminal 和 action generation 都有缓存；action cache 与 atlas block cache 有
上限，适合长预算运行。

## 运行

仓库根目录下：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.422 `
  --iterations 300 `
  --seed 4222026 `
  --output src\mcts\422\runs\bell422_group_game_i300.json
```

不使用答案 catalog 的盲搜审计：

```powershell
.\.venv\Scripts\python.exe -m mcts.422 --iterations 10 --no-catalog
```

此时发现仍按全群 canonical H-row 去重，只把名称写成 `BFS422-NEW-*`。
`--no-canonicalize` 仅适合快速 smoke，不应作为正式覆盖率实验。

## 当前迁移范围

v1 已完成单树、Filler、Corrector、progressive widening、rollout portfolio、动态
全局 novelty reward、在线 pattern 学习和完整终端分类的迁移。4-2-2 有 34,836
个 catalog classes，不能把 3-2-2 的 i15000 预算直接解释为“全覆盖”；后续实验应
先依据 discovery curve 调整 pattern racing 和 tail 配额，而不是把已有 facet
support 反向注入搜索。

## v2: 可扩展对称性状态

独立入口 `python -m mcts.422.scalable_search` 启用共享的高维设计：

- 节点按 support/role/correction/source 合并，pattern 作为动作上下文集合保存；
- orbit partition 将 pattern 切换分成 continue/refine/intersection/identity；
- rank-78 retained face 使用通用二维法锥 ridge pivot；
- action cache 不再复制完整 reward 状态，rollout 每步只打开 2 个 pattern、每个
  Filler/Corrector pool 默认物化 16/24 个候选；
- rank-band 与 pattern-relation 首次出现获得小额结构 novelty，重复 basin 逐步衰减。

这些开关不会读取 facet 样例或未发现 class。v1 入口保持不变，便于在相同 handoff
数据上对照候选生成量、终端验证次数与墙钟时间。

## v25: 独立 Embedding 单树版本

`v25_search.py` 保留 3-2-2 v25 的核心结构，并替换为 4-2-2 的 256 点、80 维、
目标 affine rank 79 几何：

- 一次 simulation 只在持久树中扩展一条新边，之后使用瞬时 rollout；
- pattern、pattern 内 block、节点 child 三层 progressive widening；
- relation、block-size 和 action-lane 公平配额；
- 同一在线 pattern family 最多保留 4 个 concrete embeddings，但各 embedding
  独立学习价值；
- discovery epoch PUCT 与停滞 edge recycle；
- Filler 最多合并 2 个 orbit，Corrector 最多删除 4 个 orbit；
- rank 77/78 tail enumeration 与 rank 78 ridge pivot；
- Corrector counterfactual credit、facet replay 与 endpoint novelty；
- 非终端 leaf heuristic 关闭，普通 leaf value 为 0。

运行一个带完整终端分类的 probe：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.422.v25_search `
  --iterations 100 `
  --seed 4222025 `
  --output src\mcts\422\runs\bell422_group_game_i100_v25.json
```

该入口不默认把 identity singleton pattern 对应为任何已知 class，初始
`discovered` 始终为空。34,836 个 catalog class 只在候选通过精确 facet 验证以后
负责命名，不参与 pattern/action 选择。
