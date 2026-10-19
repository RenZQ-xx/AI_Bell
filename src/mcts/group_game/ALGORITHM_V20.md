# Filler-Corrector MCTS 算法说明

本文档总结 `src/mcts/group_game` 当前 v20 实现。算法版本标识为：

```text
deterministic_fair_growth_bridge_v20
```

它面向 Bell 3-2-2 多胞体的 facet 搜索：从 64 个确定性顶点中构造候选顶点集，
寻找 affine rank 为 25 的有效 facet，并通过实际终端分类统计本次运行发现的
46 个等价类。

## 1. 目标与设计原则

在 3-2-2 场景中，Bell 表示空间维数为 26。一个满维 facet 需要其 tight support
达到 affine rank 25。搜索状态可以写为 64 位 bitset：

\[
S\subseteq\{0,1,\ldots,63\}.
\]

算法需要解决两个互相冲突的问题：

1. 直接按顶点搜索的分支数过大。
2. 固定一种群轨道划分又会遗漏只能由其他 block 结构构造的 class。

v20 采用以下原则：

- 只有一棵共享 MCTS，不为 class、pattern 或起始点建立独立搜索树。
- Filler 和 Corrector 是同一状态机中的两种角色，可以在一条路径中反复切换。
- 群论知识负责构造 action 和 prior，不直接给出目标 class 的顶点集合。
- 未发现 class 的 support、pattern 和示例不参与动作生成。
- class 查询只在实际到达 rank-25 终端后进行。
- 几何 look-ahead 可以验证候选出口，但只有实际执行出口才算发现和获得奖励。
- 新 class 使用全局 discovered 集合，首次命中在整个 run 中只奖励一次。

## 2. 总体架构

```text
                             GroupKnowledgeBase
                     pattern / context / transition stats
                                      |
                                      v
root(empty support) -> Filler -> rank 17..24 -> rank 25 Corrector
       ^                  |                              |
       |                  |                              +-> validate/classify
       |                  |                              |
       |                  +------ add orbit blocks       +-> stop
       |                                                 |
       +---------- remove/rewrite -> lower-rank Filler <-+
                                                         |
                                                         +-> facet repair
                                                              T -> R -> R union E

              terminal reward / counterfactual credit / novelty
                                      |
                                      v
                    MCTS edge stats and shared knowledge
```


## 3. 状态、节点与动作

### 3.1 搜索状态

`SymmetryState` 包含：

```python
SymmetryState(
    support_word,         # 当前选中的顶点集合
    role,                 # filler / corrector / done
    corrections_used,     # 当前路径已执行的修正数
    context_pattern_id,   # 当前群论 pattern 上下文
    repair_source_word,   # 两步 facet repair 的源 facet
)
```

状态键为：

\[
(S, role, corrections, context\_pattern, repair\_source).
\]

完全相同的键共享 `GameNode`。因此对外称为单树，数据结构上更接近带状态合并的
有向搜索图。相同 support 在不同 role、pattern 上下文或修正深度下仍是不同节点。

### 3.2 节点统计

`GameNode` 保存：

- `visits` 和 `value_sum`；
- 当前持久边 `edges`；
- 已展开、耗尽的 pattern；
- 每个 pattern 已展开的动作数；
- 被 rank、重复 child 或约束拒绝的 action；
- relation、action lane 和 size bucket 的轮转游标；
- discovery epoch 和停滞回收状态。

### 3.3 宏动作

`MacroAction` 有四种：

| kind | 执行者 | 状态变换 |
| --- | --- | --- |
| `add` | Filler | \(S' = S\cup A\) |
| `remove` | Corrector | \(S' = S\setminus R\) |
| `rewrite` | Corrector | \(S'=(S\setminus R)\cup A\) |
| `stop` | Corrector | role 变为 `done` |

一个 action 同时记录 pattern、orbit 下标、实际增删的 bitset、block 大小、arity、
来源和源 facet，使普通群轨道动作与几何 repair 可以进入相同的 MCTS 边接口。

### 3.4 边统计

`EdgeStats` 保存历史统计和当前 discovery epoch 统计：

```text
visits / value_sum
epoch / epoch_visits / epoch_value_sum
prior
rank_before / rank_after
discovery_hits
```

历史统计用于知识积累；当前 epoch 统计用于 PUCT，减少已经发现的 class 对后续搜索
的长期支配。

## 4. 完整搜索流程

### 4.1 初始化

根节点为：

```text
support = empty
role = filler
corrections_used = 0
context = identity singleton pattern
```

初始激活 identity pattern 和 atlas 中的极小超群根 pattern。更高层 pattern 先注册
但不全部激活，后续由实际终端的 stabilizer 分析逐步加入。

### 4.2 每个 simulation

一次 `run_iteration` 按以下顺序执行：

1. 设置本轮 rollout policy 和 context/global repair policy。
2. 根据运行中保存的 facet witness 决定是否进行 replay。
3. 从 root 沿持久树下降。
4. 每次进入 Corrector 节点，立即验证当前 rank-25 support。
5. 优先执行合法 replay action，否则尝试 progressive widening。
6. 若不能扩展新边，则使用 epoch-aware PUCT 选择已有边。
7. 一次 simulation 最多增加一条普通持久边。
8. 扩展后从新 child 开始临时 rollout。
9. 到达 `done`、dead end 或最大深度 72 后计算叶子值。
10. 为每个 Corrector 修正解析前后终端效用差。
11. 沿路径折扣回传，更新持久树和共享知识库。

运行只在以下条件停止：

- 达到配置的总 iterations；
- 目标集合中的所有 class 都已经由实际终端命中。

局部停滞不会直接停止整个 run。

## 5. Progressive Widening 与树策略

v20 使用三层 progressive widening，而不是一次生成固定 top-k action。

### 5.1 Pattern 数量

\[
K_p(s)=\left\lfloor 3+1.15N(s)^{0.45}\right\rfloor.
\]

它控制当前节点允许尝试多少种群论 pattern。

### 5.2 每个 pattern 的 block 数量

令 \(P_s\) 为已展开 pattern 数：

\[
K_b(s)=\left\lfloor
1+1.25\left(\frac{N(s)}{\max(1,P_s)}\right)^{0.5}
\right\rfloor.
\]

它控制每个 pattern 可以逐步释放多少 block action。

### 5.3 节点总 child 数量

\[
K_a(s)=\min\left(
K_{hard},
\left\lfloor4+2N(s)^{0.5}+8E\right\rfloor
\right),
\]

其中 \(E\) 是 discovery epoch：

- Filler hard cap 为 192；
- Corrector hard cap 为 128。

每次新 class 发现都会增加 epoch reserve，使访问充分的旧节点能够继续暴露新 action。

### 5.4 PUCT

节点不能继续 widening 时，选择：

\[
a^*=\arg\max_a\left[
Q_E(s,a)+cP(s,a)\frac{\sqrt{N(s)+1}}{1+N_E(s,a)}
\right],
\]

其中：

- \(c=1.35\)；
- \(Q_E\) 和 \(N_E\) 只取当前 discovery epoch；
- \(P(s,a)\) 来自 pattern、context、transition、coherence、rank 变化和动作规模；
- 未访问的当前 epoch 边令 \(Q_E=0\)。

发现新 class 后只重置边的 epoch-local novelty 价值，不删除历史结构经验。

### 5.5 低价值边回收

新 discovery epoch 或长期停滞时，可以回收最多 16 条满足以下条件的旧边：

- 至少访问 8 次；
- 历史均值不大于 0；
- 从未通向新 class；
- 同一 pattern 至少保留一条边。

回收后最多开放 4 个新 pattern。节点对象仍保留，因此结果中的 `node_count` 可能
大于当前 `edge_count + 1`。

## 6. Filler 模块

### 6.1 功能

Filler 负责从当前 support 出发加入群轨道 block，使 affine rank 接近 25：

\[
S' = S\cup A.
\]

它不是固定 pattern 的填充器，而是可以在 identity、极小超群、在线 stabilizer
pattern 及其邻接 pattern 之间切换。

### 6.2 Block 构造

对于 pattern 的 orbit partition：

\[
\mathcal B_P=\{B_1,B_2,\ldots,B_k\},
\]

一个 Filler action 可以加入一个或两个 orbit 的并：

\[
A=B_{i_1}\cup B_{i_2}.
\]

每个 pattern 最多生成 32 个 union 候选。生成器混合连续 orbit 组合和由 pattern
专属 RNG 产生的随机组合，避免候选池只包含一种排列顺序。

### 6.3 Action lanes

持久树 widening 按以下周期轮转：

```text
fine, single, single, union,
fine, single, union, broad
```

- `fine`：单 orbit 且有效 block 大小不超过 2；
- `single`：任意单 orbit；
- `union`：两个 orbit 的并；
- `broad`：有效 block 大小至少为 5。

这使 singleton、pair、普通 orbit union 和大 block 都有确定性的暴露份额。

### 6.4 Pattern relation

Filler 将目标 pattern 与当前 context 的关系分为：

```text
continue -> neighbor -> switch
```

neighbor 即：两个 pattern 之间存在一个已知的、一步可解释的 subgroup 引入、扩张或 stabilizer 包含关系。代码上的 relation 判断如下：
```python
if P == Q:
    relation = "continue"
elif Q in pattern_neighbors[P]:
    relation = "neighbor"
else:
    relation = "switch"
```

持久动作生成按：

```text
continue, neighbor, continue, switch, neighbor, switch
```

轮转。`continue` 保持构造连贯，`neighbor` 使用 subgroup atlas 或在线分析得到的
相邻结构，`switch` 保证远距离 pattern 仍可进入树。

### 6.5 Filler prior

Filler action prior 综合：

- pattern 的历史收益和新 class 命中率；
- 相同结构、rank 区间、block 大小和 arity 的 context 统计；
- 当前 pattern 到目标 pattern 的 transition 统计；
- action 后 support 与目标 pattern 的 coherence；
- `continue/neighbor/switch` 关系奖励；
- affine rank 增量；
- 超出当前 rank 需求的大 block 惩罚；
- rank 18 以上的 supportability 几何弱先验。

supportability 只判断当前点集接近可支撑超平面的程度，不读取目标 class。

### 6.6 状态转换

```text
new rank < 25  -> 保持 Filler
new rank = 25  -> 切换到 Corrector
new rank > 25  -> 拒绝 action
```

处于两步 facet repair 的 Filler 还必须加入源 facet 外部的顶点，不能删除后只补回
原 facet 内部。

## 7. Corrector 模块

Corrector 由局部群轨道删改器和 facet-relative 几何修复器组成。

### 7.1 进入 Corrector

任何 Filler 或 Corrector action 到达 rank 25 后，状态进入 `corrector`。在生成下一步
动作前，算法先执行完整终端验证：

- 计算真实 tight support；
- 验证是否为 facet；
- 对实际终端查询 class；
- 更新 discovered、facet witness 和在线 stabilizer 知识；
- 如果有效，将它注册到 `FacetCorrectorBank`。

### 7.2 局部动作

Corrector 支持：

1. `remove`：删除当前 pattern 下一个或多个已有 orbit；
2. `same_rewrite`：在同一 pattern 内删除一组 orbit 并加入另一组；
3. `cross_rewrite`：从当前 context pattern 删除，按相邻 pattern 加入；
4. `stop`：结束路径。

每个删除动作最多联合 4 个 orbit。每个 pattern 的候选上限为：

```text
remove union: 64
same rewrite: 48
cross rewrite: 24
```

Corrector lane 周期为：

```text
remove_small, rewrite, remove_broad,
remove_small, rewrite, remove_broad
```

### 7.3 Rank 与修正次数

```text
17 <= new rank < 25 -> 切换回 Filler
new rank = 25       -> 保持 Corrector
new rank < 17       -> 拒绝
new rank > 25       -> 拒绝
```

基础修正上限为 4。当当前 facet 仍有未完成几何 frontier 或未执行 context 时，允许
延长到 8。若同一 canonical facet 在一条 simulation 中出现至少 3 次且不存在开放
context，则停止继续修正，避免循环。

### 7.4 FacetAnchor

每个已验证 facet \(T\) 建立一个 `FacetAnchor`，保存：

- tight support、normal 和 offset；
- 在 Bell 对称群中的实际 stabilizer \(H\)；
- identity、完整 \(H\) 和在线循环子群诱导的 orbit partitions；
- 已发现 retained face；
- repair plans 和 external exits；
- ridge、orbit-face、lower-face completion 的进度。

在线 embedded pattern 只由当前真实 facet 的 stabilizer 构造，最多保留 12 种不同
partition，不从某个未发现 class 样例反推 block 划分。

### 7.5 几何保留面生成

Corrector 在源 facet 内寻找：

\[
R\subset T,\qquad17\le rank(R)\le24.
\]

有四条候选通道：

1. **Invariant LP**：在 identity 或 stabilizer subgroup 的不变子空间内随机求解 LP；
2. **Ridge enumeration**：对适用的小 facet 系统枚举 rank-24 ridge，最多 4096 个；
3. **Orbit-face enumeration**：删除 1 到 3 个 orbit，最多检查 256 个候选；
4. **Lower-face completion**：对 rank 17..23 保留面进行约束 LP，每个 canonical job
   最多 6 次尝试。

所有候选必须通过 affine rank 和完整 facet 几何验证。

### 7.6 RepairPlan 与两步执行

一个已验证计划为：

```python
RepairPlan(
    source_word=T,
    retained_word=R,
    pattern_id=P,
    exit_words=(E1, E2, ...),
)
```

其中每个出口满足：

\[
T'=R\cup E
\]

是有效 facet，且 \(E\) 包含源 facet 外部顶点。

计划不会直接把搜索传送到 \(T'\)，而是显式执行：

```text
Corrector: T --facet_repartition/rewrite--> R
Filler:    R --facet_external_exit/add---> R union E
```

只有第二步实际进入 rank 25 并执行终端分类后，才可能发现 class。

### 7.7 对称修复缓存

`SymmetryRepairCache` 以 canonical facet 为键缓存：

- retained face；
- external exits；
- embedded pattern；
- subgroup generators。

当对称等价的 raw facet 出现时，缓存通过群置换把计划运输到当前坐标方向。缓存不存
class 标签和完整 reward，也不会仅因 look-ahead 验证成功就计入发现。

## 8. Reward 设计

### 8.1 普通折扣回报

\[
G_t=r_t+0.985G_{t+1}.
\]

Filler 边学习普通折扣回报。Corrector 的 `remove/rewrite` 边使用独立反事实学习值，
避免把此前 Filler 的构造质量错误归因给修正动作。

### 8.2 Filler rank 奖励

对于 `add`：

\[
r_{add}=0.055\,[rank(s')-rank(s)]
-0.025\,\mathbf 1[rank(s')\le rank(s)].
\]

它提供终端前的稠密信号，同时惩罚只增加冗余点而不增加 rank 的动作。

### 8.3 Corrector 动作成本

删除：

\[
r_{remove}=-(0.012|R|+0.035\max(0,rank(s)-rank(s'))).
\]

重写：

\[
r_{rewrite}=-(0.008(|R|+|A|)
+0.035\max(0,rank(s)-rank(s'))).
\]

大范围删改仍可探索，但需要由更好的下游 facet 抵消成本。

### 8.4 终端奖励

令 \(n_c\) 为该 class 此前的命中次数，\(n_s\) 为该 canonical support 此前的
命中次数。

| 终端 | 基础 reward |
| --- | ---: |
| 本次运行首次命中的全局新 class | `10.0` |
| 已发现 exact class | \(0.10-\min(0.25,0.18\ln(1+n_c))\) |
| 有效但未分类 facet | `0.10` |
| invalid | `-1.50` |

Support novelty 再叠加：

```text
首次有效 canonical support: +0.35
重复有效 support: -min(0.25, 0.12 * ln(1 + n_s))
```

新 class 奖励对所有 class 相同，不按 class 编号或已知稀有程度加权。

### 8.5 Corrector 反事实效用

Corrector 用独立终端 utility 比较修正前后状态。exact class 的 utility 为：

\[
U_{exact}=0.25-\min(0.20,0.04\ln(n_c+2)).
\]

未分类有效 facet 为 `0.10`，invalid 为 `-1.50`。修正 credit 为：

\[
C=U_{after}-U_{before}+B_{basin}+C_{transition},
\]

其中当 canonical support 改变时：

\[
B_{basin}=0.02.
\]

若修正后没有到达另一个终端，则使用 dead-end 或叶子估值作为保守 fallback，不会
因为无终点游走而获得正向改善。

### 8.6 新 class 的 Corrector credit

当路径发现新 class 时，路径中所有实际改变终端的 Corrector 动作共享总计 10 分。
对 Corrector step \(i\)：

\[
w_i=0.985^{d_i},\qquad
C_i^{new}=10\frac{w_i}{\sum_jw_j}.
\]

距离新终端更近的修正权重更大，但所有 Corrector novelty credit 的总和严格为 10。
Filler 仍通过普通终端回传学习如何构造到该 class。

### 8.7 几何 novelty

真实执行非自身 `facet_external_exit` 时：

| 首次执行事件 | reward |
| --- | ---: |
| 新 `(source,target,pattern,corrections)` context | `0.20` |
| 新 canonical source-target pair | `0.35` |
| 新 canonical target | `0.45` |

一次执行最多得到 `1.0`。该奖励加入 external Filler action 的普通回报，同时作为
geometric credit 归给前一个 Corrector。self endpoint 不获得 novelty reward。

### 8.8 叶子和死路

未终止叶子的启发式值为：

\[
V_{leaf}=0.018\,rank-0.006\max(0,|S|-30).
\]

`dead_end` 为 `-0.75`，`done` 为 `0`。该估值鼓励接近 rank 25，并轻度惩罚包含
大量仿射冗余顶点的 support。

### 8.9 实际回传目标

对 `remove/rewrite` 且存在修正基线的边：

\[
V_C=C+C^{new}+C^{geometry}.
\]

其他边使用普通 \(G_t\)。持久边只更新一次，但树内和 rollout 内的 `add`、
`remove`、`rewrite` 都会更新共享知识库。

## 9. Rollout 设计

### 9.1 持久树与临时 rollout

一次 simulation 在持久树中选择到第一个新扩展边后停止扩展，后续使用临时
`GameNode` 和 `EdgeStats` 完成 rollout。临时节点不写入全局树，但其 action/outcome
会训练 `GroupKnowledgeBase`。

最大路径深度为 72。原 seed 的 46/46 run 中：

```text
平均持久树前缀 3.274 步
平均 rollout       7.645 步
平均总动作深度    10.919 步
depth-limit hits        0
```

### 9.2 Policy portfolio

rollout 使用三个独立 RNG，按固定 8 次周期轮换：

```text
mixed, mixed, coherent, mixed,
mixed, fine, mixed, mixed
```

比例为：

- `mixed`：6/8；
- `coherent`：1/8；
- `fine`：1/8。

独立 RNG 避免一种 policy 的随机调用数量改变其他 policy 的随机序列。

### 9.3 Filler rollout lanes

- `fine` policy 始终使用 `fine` lane，并强制保留 identity pattern；
- `coherent` policy 使用 `single` lane；
- `mixed` policy 以 50% 选择 `single`、35% 选择 `union`、15% 选择 `broad`。

`coherent` 模式若存在当前 context pattern 的合法动作，85% 概率只在这些动作中选取。

### 9.4 Corrector rollout lanes

Corrector 不区分 fine/coherent/mixed 的 lane 概率：

```text
remove_small: 35%
remove_broad: 30%
rewrite:      35%
```

基础修正上限内，Corrector 以 88% 概率继续，以 12% 概率 stop。超过基础上限但仍有
开放 frontier 时进入扩展修正阶段，最多到 8 次。

在普通 remove/rewrite 前，rollout 优先尝试几何 facet repair：

- 开放 facet 的基础尝试概率为 0.75；
- 已全局饱和且无新 context 时，有效保留概率降为 0.12；
- context policy 下存在未执行精确 context 时优先保留该动作。

### 9.5 Pattern 候选池

每个 rollout step 最多保留 8 个 pattern。候选来源包括：

- 当前 context pattern；
- 最多 3 个相邻 pattern；
- 当前 role 下评分最高的 3 个 active pattern；
- 最多 3 个随机 active pattern；
- fine policy 下额外加入 identity。

排序结合 policy 偏好、pattern score、coherence 和 continue/neighbor/switch 关系。

### 9.6 Action 候选池

对于每个 pattern：

1. 按当前 lane 惰性生成动作；
2. 保留 shape score 最高的一项；
3. 若有其他候选，再随机保留一项；
4. 最多 materialize 12 个动作并执行真实 rank 约束检查。

选择包含两个独立的 18% 随机份额：

- 18% 概率先打乱粗候选池，影响进入前 12 的动作；
- 18% 概率从 materialized 候选中随机选取；
- 其余情况选择 `prior + 0.08 * random_noise` 最大者。

rollout 不使用 UCB，因为临时边没有访问统计；它使用在线 prior 和受控随机性。

### 9.7 多终端轨迹

一条 rollout 可以多次经过 rank-25 facet：

```text
Filler -> facet A -> Corrector
       -> Filler -> facet B -> Corrector
       -> Filler -> facet C -> Corrector -> stop
```

每次进入 Corrector 都立即验证，因此普通回报可以包含多个终端结果；每个
Corrector 修改则单独比较相邻终端的前后效用。

## 10. 在线群论知识库

`GroupKnowledgeBase` 保存三类 `OnlineStats`：

1. `(role, pattern)`；
2. `(role, structure, rank_band, size_bucket, arity)`；
3. `(role, current_pattern, target_pattern)`。

Pattern score 为：

\[
S(P)=\overline R(P)
+2\frac{new\_classes(P)}{visits(P)}
+0.45\sqrt{\frac{\ln(total+2)}{visits(P)}}.
\]

未访问 pattern 的初值为 `1.25`，保证新 pattern 有较高的初始探索机会。
Transition score 使用较弱的 `0.30` optimism。

Action prior 截断在 `[0.05, 3.0]`。它只是 PUCT 和 rollout 的结构先验，不直接
计入 reward。

首次观察有效 canonical support 时，系统分析其真实 stabilizer：

- 注册 `OBS-*` pattern；
- 激活匹配的 atlas root 和 level-2 pattern；
- 建立 pattern 邻接；
- 每个 class 最多进行 2 次较重的 pattern 分析。

class id 仅用于终端统计和全局新颖性，不用于选择某个未发现 class 的示例 pattern。

## 11. Witness、Replay 与 Frontier

### 11.1 Facet witness

每个 canonical facet 保存代价最小的实际到达路径，代价优先比较：

1. `remove/rewrite` 次数；
2. 总动作数。

同时按 `(canonical facet, corrections_used)` 保存深度见证，为长 Corrector 链提供
精确上下文 replay。

### 11.2 Replay 周期

- 每 4 个 simulation 尝试一次 facet witness replay；
- 每两个 replay 中有一个优先兑现 frontier，因此通常每 8 次出现一次；
- replay 仍需要逐动作通过当前状态和 rank 检查；
- 一次 simulation 仍最多增加一条持久边。

### 11.3 Context frontier

连续 500 iterations 无新 class 后启用 context frontier。lane 周期为：

```text
endpoint, endpoint, endpoint, bridge
```

- `endpoint`：执行尚未在精确 pattern/depth context 中兑现的几何出口；
- `bridge`：允许已执行 source-target pair 在新的 pattern 或修正深度中作为桥梁复访。

它不查询 pair 对应哪个 class。

### 11.4 v20 确定性几何增长

v20 的 geometry growth 只把以下事件视为 productive：

- 新 canonical target；
- 新的非自身 source-target pair。

仅生成新 `RepairPlan` 不再重置 empty streak。每个来源先获得 2 次 warm-up；每
4 次保留一次最少服务来源 probe。普通来源最多领先当前服务 floor 8 次；曾产生
新 target 的来源额外获得 16 次上限。

墙钟时间只记录诊断，不参与选择。这样同 seed 的逻辑路径不再因机器负载或某次
LP 耗时波动而变化。

`YieldFrontier` 的 productivity 和 MCTS reward 是两套不同信号：前者决定有限几何
工作先服务哪个来源，后者只有在真实 action 执行后才更新树。

## 12. 缓存与性能设计

当前主要缓存包括：

- scorer 的共享结构评分缓存；
- 跨路径共享的终端验证缓存；
- `(role, pattern)` 的 orbit-union template cache；
- action candidate LRU，最多 6000 项；
- coherence 等轻量 LRU，最多 30000 项；
- support 到 scorer key 的转换缓存；
- canonical symmetry repair cache；
- endpoint canonicalization 和 source-target graph。

缓存结构或验证结果，但不缓存会绕过全局 discovered 状态的完整 reward。新 class
是否为首次命中始终在实际终端处动态判断。

输出采用临时文件加原子替换。完整快照较大，原 seed 每 500 iterations 写快照使
总耗时从 323.7520 秒增加到 358.5630 秒，因此性能实验默认 `snapshot_interval=0`。

## 13. 当前实验结果

固定 `PYTHONHASHSEED=0` 和 BLAS/OMP 单线程，`known_classes=[]`：

| seed | iterations completed | 覆盖 | 缺失 | elapsed |
| ---: | ---: | ---: | --- | ---: |
| 3222026 | 6712 | 46/46 | 无 | 323.7520 s |
| 673963275 | 6824 | 46/46 | 无 | 341.3601 s |
| 1706386020 | 15000 | 45/46 | 3 | 1114.1361 s |
| 210656494 | 15000 | 44/46 | 3, 34 | 1005.6555 s |

四个 seed 中两个全覆盖，平均覆盖 45.25。样本量很小，不能把 50% 解释为稳定
成功率，但可以确认：

- v20 具备在单树中实际命中全部 46 类的能力；
- `i15000` 不是跨 seed 的全覆盖保证；
- class 3 是当前最明显的跨 seed 长尾，class 34 次之；
- 两个未完成 run 在最后发现后继续运行 8704 和 8384 次仍无新增，单纯增加同一
  seed 的预算可能不如停滞后更换 seed；
- 约 6700 iterations 的全覆盖 run 有约 6600 个持久节点，完整 i15000 run 有约
  14500 个持久节点；临时 rollout 节点不计入该数量。

完整结果见 `runs/v20_multiseed_i15000_comparison.json` 和 `runs/record.md`。

## 14. 可复现实验命令

在仓库根目录运行：

```powershell
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 15000 `
  --seed 3222026 `
  --snapshot-interval 0 `
  --output src\mcts\group_game\runs\single_tree_group_game_i15000_v20.json
```

## 15. 当前局限与后续方向

### 15.1 Seed 方差

不同 seed 会改变：

- orbit union 的随机候选；
- rollout pattern 和 action；
- invariant LP 方向；
- 在线出现的 facet、embedded pattern 和 repair chain。

这会同时影响覆盖率、节点数和单 iteration 成本。当前结果支持增加多 seed 评估，
不支持只用一个成功 seed 判断稳定性。

### 15.2 长尾不是简单预算不足

失败 run 在约 i6300/i6600 后进入超过 8000 iterations 的无新发现尾部。它们仍
生成节点和执行几何工作，但没有形成缺失 class 所需的完整 action chain。下一步
更值得研究：

- 基于在线停滞的 seed restart 或 tree diversification；
- 不依赖 class 编号的 frontier coverage 指标；
- 提高 Corrector 长链组合的信用分配质量；
- 减少重复 endpoint/context 执行；
- 在不引入目标样例的前提下扩大可验证 block rewrite 结构。

### 15.3 有界枚举不是完备证明

Ridge、orbit-face 和 lower-face completion 都有预算上限。`finished` 可能表示当前
有限候选流耗尽或额度用完，不表示完整 facet 邻接图已经穷尽。因此未找到某个 class
不能解释为其在当前起点下不可达。

### 15.4 知识库仍存在非平稳性

Discovery epoch 重置边的局部价值，但 pattern/context/transition 的历史均值仍跨
epoch 累积。早期常见 class 的统计可能继续影响后期 prior。后续可以考虑保留结构
统计，同时为 novelty-sensitive 统计增加衰减或 epoch 分层。

## 16. 一句话总结

v20 是一棵由群论 pattern 动态生成增删改动作、由 Filler 构造 facet、由 Corrector
在实际 stabilizer 和公共面几何下换面、通过全局新 class 奖励与反事实 credit 学习，
并使用 progressive widening、epoch-aware PUCT、多策略 rollout、对称缓存和确定性
frontier 调度来扩大覆盖的单树 MCTS。
