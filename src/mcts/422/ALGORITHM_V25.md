# Bell 4-2-2 Filler-Corrector MCTS v25

本文记录 `mcts.422.v25_search`。它是 3-2-2 v25 的高维迁移版，与既有 4-2-2
v1/v2 隔离，便于进行同数据、同预算对照。

## 1. 问题与先验边界

4-2-2 场景包含 256 个 deterministic vertices 和 80 个 correlator 坐标。一个
facet 的完整 tight support 必须达到 affine rank 79。搜索状态使用一个 Python
256-bit 整数记录 support。

搜索允许读取：

- `complete_nodes.csv` 中的 subgroup 元数据；
- `nodes/*/orbit_blocks.csv` 中由群作用诱导的 vertex partition。

搜索不读取 `nodes/*/facets.csv`。`global_facet_classes.csv` 仅在候选已经通过 rank、
supporting hyperplane 和 tight-support closure 验证后用于命名 class。因此：

- identity pattern 只是 256 个 singleton，不代表已发现某个 class；
- 未发现 class 的 support 不会进入 action、prior、reward 或预算；
- 初始全局 `discovered` 为空。

## 2. 单树状态与生命周期

持久节点键为：

```text
(support_word, role, corrections_used, context_pattern_id, source_facet_word)
```

`role` 为 `filler`、`corrector` 或 `done`。`context_pattern_id` 保留产生当前几何状态
的具体 embedding，`source_facet_word` 保留 Corrector 跳跃的出发 facet。

每次 simulation：

1. 从 root 使用 epoch-aware PUCT 选择已有边；
2. progressive widening 只向持久树安装一条新边；
3. 新边之后创建瞬时 `GameNode` 做 rollout，不污染持久树；
4. 到达 exact facet、invalid terminal、dead end 或深度上限后回传；
5. 持久边更新普通统计和当前 discovery epoch 统计。

所以节点总量近似受 simulation 数约束，而 rollout 可以自由试探更深路径。

## 3. 群论 Pattern 与 Embedding

初始 atlas 包含 handoff 的 721 个 subgroup partitions 和一个 identity partition。
每个 partition 的 block 是 subgroup 在 256 个 vertices 上的 orbit。

当搜索发现 facet 时，在线结构学习会：

1. 检查 15 个标准 relabeling generators；
2. 通过可复现随机游走采样保持该 support 的 stabilizer 元素；
3. 对保持元素求有界 subgroup closure；
4. 从 subgroup 作用生成新的 orbit partition；
5. 将 partition 注册为该 facet family 的 concrete embedding。

同一 family 最多保留 4 个 concrete embeddings。family 只负责容量控制和
`reembed` 关系识别，不共享 value；每个 embedding 独立累计 pattern、transition
和 context 统计。这保留了 v25 消融后最重要的结论：不同具体嵌入不能因为抽象群
结构相同就被错误地赋予相同价值。

## 4. Directed Pattern Relations

pattern 切换按有向关系区分：

- `continue`：继续使用同一 concrete partition；
- `reembed`：同一在线 family 的另一个 embedding；
- `refine`：目标 partition 更细；
- `coarsen`：目标 partition 更粗；
- `intersection`：无包含关系，但可提供新的交叉切分；
- `identity`：回到 singleton 粒度。

关系公平周期为：

```text
continue, reembed, coarsen, continue, refine, reembed,
intersection, identity, coarsen, refine, intersection
```

关系既参与候选配额，也独立学习 role-specific value。它不是静态偏好表。

## 5. 三层 Progressive Widening

一个节点不会一次物化全部 722+ patterns 和全部 block actions，而是分三层开放：

```text
K_pattern(N) = floor(k0_p + alpha_p * max(1,N)^beta_p)
K_block(N)   = floor(k0_b + alpha_b * max(1,N/P)^beta_b)
K_child(N)   = floor(k0_c + alpha_c * max(1,N)^beta_c) + epoch reserve
```

其中 `P` 是已展开 pattern 数。三层分别控制：

- 当前节点可竞争的 pattern 数；
- 每个 pattern 可物化的 block action 数；
- 节点持久 child 总数。

pattern 层还轮转 singleton、pair、small、medium、large 近似 size buckets；action 层
分别轮转 Filler 与 Corrector lane。这样 rare partition 和 rare block 组合会随着节点
访问增长逐步获得验证机会，而不是被一次性的 top-k 截断。

发现新 class 后 knowledge epoch 增加。PUCT 的 Q 优先读取当前 epoch 统计；旧边保留
结构记忆，但不能长期垄断新的全局 novelty 目标。epoch 变化或长期停滞时，可回收
访问充分、无 discovery hit 且均值不佳的边，为新 pattern 留出 child 容量。

## 6. Filler

Filler 的目标是把低 rank support 构造成 rank-79 facet 候选。

动作 lane：

```text
fine, single, single, union, fine, single, union, broad
```

- `fine` 优先最小 orbit；
- `single` 加入一个 orbit-support difference；
- `union` 加入两个 orbit；
- `broad` 从大 block 一侧取样，但一次仍最多合并两个 orbit。

动作先做廉价 shape/coherence/online-value 排序，只对受限候选调用 rank oracle。
rank 增长不是终极目标，只是一项较弱的过渡奖励。

在 rank 77 和 78，Filler 启用 tail enumeration：分别枚举两块和一块补全，达到
rank 79 后立即做完整 facet 验证。rank 78 且 support 是已知 source facet 的
codimension-1 retained face 时，还会通过二维 normal cone 枚举 ridge 的相邻 facet
出口。两者都只验证现场生成的出口，不查询未发现 class 样例。

## 7. Corrector

Corrector 在发现一个 exact facet 后继续工作，不把该 facet 简单视为 episode 终点。
动作 lane 为：

```text
remove_small, rewrite, remove_broad,
remove_small, rewrite, remove_broad
```

能力包括：

- 删除一个 orbit 与当前 support 的交集；
- `remove_broad` 轮转删除 2、3、4 个 orbit；
- 在同一 pattern 中 remove 后 add；
- 用相邻 pattern 做 cross-pattern rewrite；
- 从 rank-78 ridge state 重新补全到邻接 facet；
- 通过已发现 facet 的最短 witness 做 adaptive replay。

Corrector 不只接收最终回报。执行 remove/rewrite 时记录出发 facet 的
counterfactual utility；到达下一个 terminal 后，用“结果 utility - 出发 utility”
加 basin novelty 和动作成本形成专属 credit。新 class 的 `+10` 会按折扣距离在路径
上的 Corrector contributors 之间归一化分配，避免每次删除都重复获得完整奖励。

随着已发现 basin 增加，自由 Filler 比例从 0.75 衰减到最低 0.10，更多 simulation
用于 facet replay 和 Corrector 跳跃。这里的 scale 是启动调度尺度，不是用
`discovered / 34836` 假装全局覆盖率。

## 8. Rollout Portfolio

8 次一周期：

```text
mixed, mixed, coherent, mixed, mixed, fine, mixed, mixed
```

即 6/8 mixed、1/8 coherent、1/8 fine。三种 policy 使用独立 RNG，但共享：

- 全局 discovered 集合；
- rank、facet 和 action structure cache；
- pattern、transition、relation 在线统计；
- 在线 stabilizer patterns。

`coherent` 高概率沿当前 context partition 延伸；`fine` 保留 identity 及小 block；
`mixed` 在 continue/reembed/refine/coarsen/intersection 间保持探索。rollout 节点是
瞬时对象，但经过的 action 仍会更新共享结构知识。

## 9. Reward

终端 reward：

- 全局首次 canonical class：`+10`；
- 已发现 class：从 `+0.10` 开始，并按 class 命中频率衰减；
- 首次 canonical support：`+0.35`；
- 重复 support：按频率惩罚，最大 `0.25`；
- invalid rank-79 terminal：`-1.50`；
- dead end：`-0.75`。

过渡 reward：

- Filler rank gain：每级 `+0.055`；
- 无 rank gain 的 add：`-0.025`；
- remove：按删除点数及 rank loss 计成本；
- rewrite：按 remove/add 总点数及 rank loss 计成本。

Corrector 从一个 facet 跳到另一个 facet 时还可获得三层 endpoint novelty：新
`(source,target,pattern,depth)` context、新有向 facet pair、新 target basin。

普通非终端 leaf value 为 `0`。v25 不启用非终端 leaf heuristic，也不启用 Filler
reheat，避免启发式势函数在高维下压过真实终端证据。

## 10. 4-2-2 迁移调整

保持不变的是游戏结构、树生命周期、widening 层级、relation lanes、独立 embedding
价值、Corrector credit、rollout 比例和全局 novelty 语义。按 4-2-2 几何调整的是：

- support 从 64 位扩展到 256 位；
- target rank 从 25 改为 79；
- tail rank 改为 77/78；
- Corrector 最低 rank 默认 60；
- max depth 默认 112；
- support 随机种子使用完整 32-byte 散列；
- class catalog 规模记录为 34,836，但不作为搜索覆盖率分母或先验。

高维搜索的 wall time 主要由 rank-79 facet verification、98,304 元 canonicalization
和在线 stabilizer closure 决定，因此 3-2-2 的 iteration 数不能直接换算为相同的
运行时间或覆盖率。

## 11. 运行与输出

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m mcts.422.v25_search `
  --iterations 100 `
  --seed 4222025 `
  --output src\mcts\422\runs\bell422_group_game_i100_v25.json
```

长预算可以设置优雅截断并保存中间快照：

```powershell
.\.venv\Scripts\python.exe -m mcts.422.v25_search `
  --iterations 15000 `
  --max-wall-hours 6 `
  --checkpoint 10000 src\mcts\422\runs\bell422_group_game_i10000_v25.json `
  --output src\mcts\422\runs\bell422_group_game_i15000_v25.json
```

达到 wall-time 上限时，当前 iteration 完成后会正常写出部分结果，并在 `runtime`
中记录 `termination_reason=wall_time_limit` 与 `truncated=true`。

输出记录完整 config、发现时间和 action path、节点/边数量、每种 relation/action/
rollout 计数、Corrector credit、endpoint graph、online pattern families、rank/facet
cache 和真实 wall time，可用于后续预算与覆盖率分析。
