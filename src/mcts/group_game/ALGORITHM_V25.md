# Filler-Corrector MCTS v25 算法说明

本文档总结 `src/mcts/group_game` 当前推荐版本 v25。正式入口为：

```text
python -m mcts.group_game.lean_embedding_search
```

结果文件中的算法版本标识为：

```text
independent_embedding_relation_mcts_v25
```

v25 面向 Bell 3-2-2 多胞体的 facet 搜索。它把群论 pattern、Filler、Corrector、
几何邻接和在线学习放入一棵共享 MCTS 中，以实际终端命中覆盖 46 个等价 class。
当前版本是在 v24 concrete embedding 表示之上，经 progressive widening、reward
和多 seed 消融后得到的精简配置。

## 1. 问题定义与设计原则

3-2-2 问题包含 64 个确定性顶点，Bell 表示空间维数为 26。搜索支撑集为：

\[
S\subseteq\{0,1,\ldots,63\}.
\]

一个候选边界需要达到 affine rank 25，随后必须通过完整的 supporting-facet 几何
验证。算法要同时处理直接逐点搜索分支过大、固定 orbit partition 覆盖不足、常见
class 重复吸走预算，以及 Filler/Corrector credit assignment 不同等问题。

v25 遵循以下原则：

- 只有一棵持久 MCTS，不为 class、pattern 或起始点建立独立树。
- Filler 和 Corrector 是同一状态机中的两个合作角色。
- 群论知识生成动作、描述上下文并提供结构先验，不直接给出目标 class 的 support。
- 未发现 class 的代表样例、Compatibility Bank 和 target bank 不参与动作选择。
- class 查询只发生在实际到达 rank-25 候选并进行终端验证时。
- 几何 look-ahead 可以验证出口，但只有实际执行出口才算发现。
- 全局新 class 在整个 run 中只获得一次 `10.0` 的 class novelty 奖励。
- identity singleton 起点只是动作划分，不会默认把 class 1 计入结果。

## 2. v25 配置边界

当前入口默认启用：

- adaptive Corrector replay；
- rank-24 ridge pivot 与 eager ridge focus；
- hierarchical pattern policy；
- `continue/reembed/coarsen/refine/intersection/identity` 定向关系；
- Filler 与 Corrector 分角色的 relation value 学习；
- concrete pattern embedding 收集；
- 每个 family 最多 4 个 embedding；
- 每个已发现 class 最多 4 次 stabilizer embedding 分析；
- 多层 progressive widening、bucket quota 和 discovery epoch 回收；
- endpoint/context frontier 与对称 repair cache。

当前入口明确关闭：

- geometric transposition 节点合并；
- embedding family value sharing；
- 停滞触发的 Filler reheat；
- 非终端 leaf rank heuristic。

因此同一 family 的不同坐标 embedding 都能进入动作空间，但各自独立学习价值。

## 3. 总体架构

```text
                         GroupKnowledgeBase
                 pattern / context / transition / relation
                                  |
                                  v
root(empty support) -> Filler -> rank 17..24 -> rank 25 Corrector
       ^                  |                              |
       |                  | add orbit / union            +-> validate/classify
       |                  |                              |
       |                  +------------------------------+-> stop
       |                                                 |
       +------- remove/rewrite -> lower-rank Filler <----+
                                                         |
                                                         +-> facet repair
                                                              T -> R -> R union E

        terminal reward / counterfactual credit / endpoint novelty
                                  |
                                  v
                  MCTS edge statistics and online knowledge
```

Filler 从空支撑集或 Corrector 留下的低 rank 面构造 rank-25 候选。Corrector 验证
当前 facet、删除或重写 block，并在已验证 facet 的保留面上寻找相邻出口。二者共享
发现新 class 的合作目标，但边的学习目标不同。

## 4. 状态、节点与动作

### 4.1 SymmetryState

```python
SymmetryState(
    support_word,         # 当前支撑集，64-bit bitset
    role,                 # filler / corrector / done
    corrections_used,     # 本路径已执行的 remove/rewrite 数
    context_pattern_id,   # 当前 concrete pattern 上下文
    repair_source_word,   # 两步 facet repair 的源 facet
)
```

节点键为：

\[
(S, role, corrections, context\_pattern, repair\_source).
\]

完全相同的键共享 `GameNode`，所以实现是带状态复用的有向搜索图。v25 没有启用
geometric transposition，context 仍属于节点身份；同一 support 在不同 pattern
语境下是不同节点。

### 4.2 GameNode 生命周期

`GameNode` 保存访问值、持久边、已展开/耗尽 pattern、各 pattern 动作数、拒绝和
退休 action、child 状态集合、bucket/relation/action lane 游标及 discovery epoch。

节点首次由合法持久边到达时创建；以后相同状态键复用。一个 simulation 最多新增
一条普通持久边。扩展后的路径由临时 `GameNode` 完成 rollout，临时节点不进入全局
树，但其动作结果仍训练在线知识库。边被回收时节点对象和结构统计不会删除。

### 4.3 MacroAction

| kind | 角色 | 状态变换 | 用途 |
| --- | --- | --- | --- |
| `add` | Filler | \(S'=S\cup A\) | 加入 orbit 或 orbit union |
| `remove` | Corrector | \(S'=S\setminus R\) | 删除当前 pattern 下的 block |
| `rewrite` | Corrector | \(S'=(S\setminus R)\cup A\) | 同 pattern 或跨 pattern 替换 |
| `stop` | Corrector | `role=done` | 结束轨迹 |

动作还记录 concrete pattern、orbit 下标、实际增删 bitset、block 大小、arity、来源、
remove/add pattern 和源 facet，使普通群轨道动作与几何 repair 共用一种边表示。

### 4.4 EdgeStats

每条持久边同时保存：

```text
visits / value_sum
epoch / epoch_visits / epoch_value_sum
prior
rank_before / rank_after
discovery_hits
```

历史统计保留长期结构经验；PUCT 只读取当前 discovery epoch 的局部 Q 和访问数，
避免旧的新类奖励永久支配选择。

## 5. 一次 Simulation 的流程

一次 `run_iteration` 执行：

1. 按 iteration 选择 rollout policy 和 Corrector saturation policy。
2. 根据覆盖率、周期和 frontier 决定是否 replay 已有 facet witness。
3. 从 root 沿持久树下降。
4. 每次进入 Corrector 节点，先验证当前 rank-25 候选并计算终端效用。
5. 优先执行合法 replay，否则尝试 progressive widening。
6. 不能扩展新边时，使用 epoch-aware PUCT 选择已有边。
7. 遇到本轮第一条新持久边后停止树扩展。
8. 从新 child 开始 transient rollout，直到 `done`、dead end 或深度 72。
9. 为每个 Corrector 动作解析修改前后首个终端的反事实效用差。
10. 以 `0.985` 折扣回传，并更新 pattern、context、transition 和 relation 统计。

运行在达到 iteration 上限或 46 个目标 class 都被实际终端命中时停止。局部停滞只会
触发回收和 frontier 调度，不会直接结束全局运行。

## 6. Progressive Widening 与树策略

### 6.1 三层 widening

节点访问数为 \(N(s)\) 时，允许展开的 pattern 数为：

\[
K_p(s)=\left\lfloor3+1.15N(s)^{0.45}\right\rfloor.
\]

令 \(P_s\) 为已展开 pattern 数，每个 pattern 的动作额度为：

\[
K_b(s)=\left\lfloor
1+1.25\left(\frac{N(s)}{\max(1,P_s)}\right)^{0.5}
\right\rfloor.
\]

节点 child 总额度为：

\[
K_a(s)=\min\left(
K_{hard},
\left\lfloor4+2N(s)^{0.5}+8E\right\rfloor
\right),
\]

其中 \(E\) 为 discovery epoch；Filler 的硬上限为 192，Corrector 为 128。每发现
一个新 class，旧节点都会获得额外 child reserve。

### 6.2 Relation quota

v25 的 relation lane 周期为：

```text
continue, reembed, coarsen, continue, refine, reembed,
intersection, identity, coarsen, refine, intersection
```

- `continue`：沿用当前 concrete embedding；
- `reembed`：切换到同一 canonical family 的另一坐标划分；
- `refine`：目标 partition 是当前 partition 的细化；
- `coarsen`：目标 partition 是当前 partition 的粗化；
- `intersection`：不存在单向细化关系；
- `identity`：显式回到 singleton partition。

除 `reembed` lane 外，候选优先覆盖尚未展开的 family，并对每个 family 先选一个
代表，避免同 family embedding 占满 widening 名额。

### 6.3 Size bucket 与 action lane

Pattern 按主导 block 大小轮转：

```text
singleton, pair, small(3-4), medium(5-8), large(9+)
```

Filler action lane：

```text
fine, single, single, union, fine, single, union, broad
```

Corrector action lane：

```text
remove_small, rewrite, remove_broad,
remove_small, rewrite, remove_broad
```

### 6.4 Epoch-aware PUCT

节点不能继续 widening 时选择：

\[
a^*=\arg\max_a\left[
Q_E(s,a)+1.35P(s,a)\frac{\sqrt{N(s)+1}}{1+N_E(s,a)}
\right].
\]

`Q_E` 和 `N_E` 只使用当前 epoch；尚未在当前 epoch 访问的边令 `Q_E=0`。极小随机
量仅用于同分破同。

### 6.5 低价值边回收

在新 discovery epoch，或全局 500 次 simulation 无新 class 且节点自上次回收后又
访问至少 128 次时，饱和节点可以回收旧边。候选必须至少访问 8 次、历史均值不大于
0、没有 discovery hit、不是 `stop`，并且同 pattern 删除后仍保留入口。

每次最多回收 16 条边，并开放最多 4 个此前未展开的 pattern。节点和共享知识不会
删除，退休 action 也不会被立即重新展开。

## 7. Filler 模块

### 7.1 功能与 pattern 来源

Filler 执行 \(S'=S\cup A\)，负责从空集启动、把低 rank 支撑集推到 rank 25，以及
完成 Corrector 提出的外部出口。可用 pattern 来自：

1. identity singleton partition；
2. 24 个与 facet 无关的极小超群代表；
3. 由实际命中 facet 激活的二级 subgroup pattern；
4. 实际终端 tight support 在线产生的 stabilizer concrete embeddings；
5. Corrector 在具体 facet 稳定子中发布的 embedded subgroup partition。

一条路径可以切换 pattern，但每个动作本身始终是某个真实 partition 下的完整 orbit
或 orbit union。未发现 class 的样例不会进入 pattern 集。

### 7.2 Orbit-union 动作

设 pattern 的 orbit partition 为 \(\mathcal B_P=\{B_1,\ldots,B_k\}\)。Filler 最多
联合两个 orbit：

\[
A=B_i\quad\text{或}\quad A=B_i\cup B_j.
\]

每个 pattern 最多生成 32 个 union 候选。生成器混合循环连续组合与从
`(seed, role, pattern_id)` 派生的独立随机组合，既可复现，又不固定为低编号顺序。

### 7.3 Filler prior

v25 的原始 action prior 为：

\[
\begin{aligned}
P_{raw}={}&0.85+0.18V_{pattern}+0.12V_{context}
+0.10V_{transition}\\
&+0.42C_{coherence}+B_{relation}
+0.12V_{relation}+G_{rank}-L_{oversize}.
\end{aligned}
\]

最终截断到 `[0.05, 3.0]`。pattern 统计具体 embedding 的均值、新 class 命中率和
optimism；context 按 role、结构、rank band、size bucket、arity 聚合；transition
学习 concrete pattern 间的转换；coherence 衡量当前 support 对目标 partition 的
破块程度。Filler 的几何项为 `0.12*max(rank_delta,-2)`，过大 block 会受到轻微惩罚。

rank 18 到 24 的非终端 support 还可使用 class-independent supportability 先验。
它只测量形成 supporting hyperplane 的几何阻碍，不查询目标 class。

### 7.4 状态转换

```text
new rank < 25  -> 保持 Filler
new rank = 25  -> 进入 Corrector
new rank > 25  -> 拒绝动作
```

若正在完成两步 facet repair，最终候选必须包含源 facet 外的顶点；只补回原 facet
内部点的闭环会被拒绝。

## 8. Corrector 模块

Corrector 包含通用群轨道删改和以真实 facet 为锚点的几何修复两层。

### 8.1 进入与终端验证

任何动作到达 rank 25 后进入 Corrector。生成下一步前，系统执行完整 facet 验证、
计算 tight support、对有效终端查询 class、更新 discovered/witness，并为有效 facet
建立 `FacetAnchor`。终端查询不会在候选到达前指导选哪些顶点或 block。

### 8.2 通用删改

Corrector 支持：

- `remove`：删除一个或最多四个已有 orbit 的并；
- `same_rewrite`：在同一 pattern 下删除并补入另一组 orbit；
- `cross_rewrite`：按当前 context 删除，按另一个 pattern 补入；
- `stop`：结束当前轨迹。

每个 pattern 的有界候选数为：

```text
remove union: 64
same rewrite: 48
cross rewrite: 24
```

删除或 rewrite 后只允许 rank 17 到 25。rank 17 是局部修复保留面的下限，不是最终
facet rank，也不表示 Corrector 只能处理 rank 17。

基础修正上限为 4；若仍有几何 frontier 或未执行 context，可延长到 8。同一
simulation 内同一 canonical facet 出现至少 3 次且无开放 context 时强制停止。

### 8.3 FacetAnchor 与稳定子

每个在线验证 facet \(T\) 建立 `FacetAnchor`，保存 raw tight support、normal、offset、
实际稳定子 \(H=Stab(T)\)、embedded partitions、生成元、不变子空间投影、retained
faces、repair plans、external exits 和 frontier 进度。

候选 partition 包括 identity、完整 \(H\) 和从真实稳定子中抽取的循环子群。循环闭包
阶数上限为 6，每个 anchor 最多保留 12 种不同 partition；不会用未发现 class 的
已知 pattern 反推划分。

### 8.4 Retained face 通道

Corrector 搜索：

\[
R\subset T,\qquad17\le rank(R)\le24.
\]

当前有四类在线通道：

1. **Invariant LP**：在 identity 或真实 stabilizer 子群的不变方向上求支撑面；每个
   anchor 最多 48 次本地尝试，identity 与非平凡子群交替。
2. **Systematic ridge enumeration**：当
   \(\binom{|T|}{25}\le32768\) 时以 Gale/nullspace 条件枚举 rank-24 ridge；每批
   最多检查 256 个候选。
3. **Orbit-face enumeration**：对不适用小 facet ridge 流的源，按 embedded partition
   删除 1 到 3 个完整 orbit；每源最多 256 个候选，每批 8 个。
4. **Lower-face completion**：对 rank 17 到 23 的保留面做约束 LP；每个 canonical
   `(source, retained, pattern)` 最多 6 次。

所有候选都必须通过 affine rank、源 facet 支撑面条件和完整目标 facet 几何验证。

### 8.5 Rank-24 ridge pivot

对 rank-24 ridge，`adjacent_facet_words` 在该 ridge 的二维法向商空间中寻找相邻
supporting ray，形成“已知 facet -> 共同 ridge -> 相邻 facet”的局部跳跃，不需要
事先知道相邻 class 的替换规则。

覆盖达到 50% 后，每次发现新 class 可对当前 facet 给予最多 48 个 focused ridge
batch，并在 eager 模式下立即派发出口。pivot 无结果时仍回退到源 facet 外的 singleton
几何检查。

### 8.6 RepairPlan 与两步执行

```python
RepairPlan(
    source_word=T,
    retained_word=R,
    pattern_id=P,
    exit_words=(E1, E2, ...),
)
```

计划满足 \(T'=R\cup E\) 是有效 facet，且 \(E\) 含源 facet 外部顶点。计划不会把
搜索直接传送到目标，而是显式执行：

```text
Corrector: T --facet_repartition/rewrite--> R
Filler:    R --facet_external_exit/add---> R union E
```

只有第二步实际进入终端验证后，目标才可能计为新发现。这将 look-ahead 与 MCTS
真实路径严格分开。

### 8.7 Repair cache 与 endpoint graph

`SymmetryRepairCache` 在 canonical facet 坐标下保存 retained face、external exit、
embedded pattern 和 subgroup generator。对称等价 raw facet 出现时，通过真实群置换
和共轭把计划运输到当前坐标。

缓存不保存 class 标签或完整 reward。`EndpointGraph` 只记录已通过几何验证的
canonical `(source,target)`；实际执行 external exit 后才增加使用次数。规划或
look-ahead 本身不算发现。

## 9. Reward 与 Credit Assignment

### 9.1 普通回报

\[
G_t=r_t+0.985G_{t+1}.
\]

Filler 边学习普通折扣回报。Corrector 的 `remove/rewrite` 边学习独立反事实值，避免
把此前 Filler 的构造质量错误归因给后续删改。

### 9.2 Transition shaping

Filler add：

\[
r_{add}=0.055[rank(s')-rank(s)]
-0.025\mathbf 1[rank(s')\le rank(s)].
\]

Corrector remove：

\[
r_{remove}=-(0.012|R|+0.035\max(0,rank(s)-rank(s'))).
\]

Corrector rewrite：

\[
r_{rewrite}=-(0.008(|R|+|A|)
+0.035\max(0,rank(s)-rank(s'))).
\]

这些成本不禁止大修改，但要求其由更好的后继终端抵消。

### 9.3 终端 reward

令 \(n_c\) 为该 class 此前命中次数，\(n_s\) 为 canonical tight support 此前命中
次数。

| 终端类型 | 基础 reward |
| --- | ---: |
| 全局首次命中的 class | `10.0` |
| 已发现 exact class | \(0.10-\min(0.25,0.18\ln(1+n_c))\) |
| 有效但未分类 facet | `0.10` |
| invalid / non-facet | `-1.50` |

有效 support 再叠加：

```text
首次 canonical support: +0.35
重复 canonical support: -min(0.25, 0.12 * ln(1 + n_s))
```

因此首次新 class 且首次 support 的终端项通常为 `10.35`。奖励不按 class 编号或
预先知道的稀有程度加权。重复 exact facet 的总终端 reward 最低约 `-0.40`，仍优于
invalid 的 `-1.50`。

### 9.4 Corrector 反事实效用

Exact class 的 Corrector 语义效用为：

\[
U_{exact}=0.25-\min(0.20,0.04\ln(n_c+2)).
\]

有效未分类 facet 为 `0.10`，invalid 为 `-1.50`。一次删改的 credit 为：

\[
C=U_{after}-U_{before}+B_{basin}+r_{transition},
\]

canonical support 改变时 \(B_{basin}=0.02\)。若没有到达后继终端，则使用保守
fallback，并保证无终点游走不会被当作语义改善。

### 9.5 Corrector 的新类 credit

轨迹发现新 class 时，发现之前所有有效 Corrector 动作共享总计 `10.0`：

\[
w_i=0.985^{d_i},\qquad
C_i^{new}=10\frac{w_i}{\sum_jw_j}.
\]

距离新终端更近的修正权重略高，总 credit 严格为 10。Filler 仍通过普通终端回传
学习如何构造到该 class。

### 9.6 Endpoint novelty

真实执行非自身 `facet_external_exit` 时，首次事件获得：

| 事件 | reward |
| --- | ---: |
| 新 `(source,target,pattern,corrections)` context | `0.20` |
| 新 canonical source-target pair | `0.35` |
| 新 canonical target | `0.45` |

一次最多 `1.0`。该值加入 external Filler 的普通回报，也作为 geometric credit 归给
前一个 Corrector。self endpoint 不获得 novelty。

### 9.7 Leaf、dead end 与最终学习值

v25 关闭非终端 leaf shaping：`done=0`，普通非终端 leaf 为 `0`，dead end 为
`-0.75`。旧公式 `0.018*rank - 0.006*max(0, |S|-30)` 只保留作历史复现。

对具有反事实基线的 `remove/rewrite` 边：

\[
V_C=C+C^{new}+C^{geometry}.
\]

其他边学习普通 \(G_t\)。树内和 rollout 内执行的动作都会更新共享结构知识。

## 10. Rollout 设计

### 10.1 Policy portfolio

rollout 以 8 次 simulation 为周期：

```text
mixed, mixed, coherent, mixed,
mixed, fine, mixed, mixed
```

即 `mixed` 6/8、`coherent` 1/8、`fine` 1/8。三种 policy 使用从主 seed 独立派生的
RNG，避免一种 policy 的随机调用数量扰动其他 policy。

### 10.2 Filler rollout

- `fine`：只选有效大小不超过 2 的单 orbit，并保留 identity；
- `coherent`：使用 single lane；有当前 context 合法动作时以 85% 概率继续；
- `mixed`：50% single、35% union、15% broad。

### 10.3 Corrector rollout

```text
remove_small: 35%
remove_broad: 30%
rewrite:      35%
```

基础修正上限内，以 88% 概率继续、12% 概率 stop。若有开放 frontier，可延长到最多
8 次修正。普通删改前以基础概率 `0.75` 尝试几何 repair；已饱和且无新 context 时
保留概率降为 `0.12`。

### 10.4 Pattern 与 action 候选池

每步最多保留 8 个 pattern，来源包括当前 context、最多 3 个邻接 pattern、最多 2 个
同 family sibling、每个 family 一个代表后的 top 3、最多 3 个随机 active pattern，
以及 fine policy 的 identity。

对每个 pattern 按 lane 惰性生成动作，保留 shape score 最高的一项和一个随机代表；
全局最多 materialize 12 个动作做真实 rank 检查。通常选择 `prior + 0.08*random`
最高项，同时保留 18% 随机选择概率。

### 10.5 多终端轨迹

轨迹每次进入 Corrector 都验证当前 rank-25 support，因此一次 simulation 可以经过
多个 facet 并发现多个 class。终端 reward 只向其之前的动作传播，之后执行的修正
不会错误领取前一个新 class 的 credit。

### 10.6 当前深度统计

固定 seed 3222026 的 v25 全覆盖运行：

```text
iterations                 2507
tree policy steps         15913
rollout steps             14052
平均持久树前缀             6.347
平均 rollout               5.605
平均总动作深度            11.952
depth-limit hits               0
dead ends                      0
```

## 11. 在线群论知识库

### 11.1 Family 与 embedding

在线终端 pattern 使用两层标识：

```text
family:    OBS-{canonical_support}
embedding: OBS-{canonical_support}-E{partition_digest}
```

family 表示 canonical symmetry family；embedding digest 只由当前原始坐标中排序后的
orbit bit words 计算，表示真正可以生成动作的 partition。每个 family 最多 4 个
embedding，每个已发现 class 最多分析 4 个实际朝向。

### 11.2 独立价值语义

v25 设置 `share_embedding_family_values=false`。family 只用于 embedding 上限、
family-first 公平候选、`reembed` relation 和诊断；pattern value 与 transition value
完全按 concrete embedding 更新。这样不会把一个朝向的成功错误推广到其他坐标划分。

### 11.3 在线统计与 discovery epoch

知识库按角色记录：

- `(role, concrete_pattern)`；
- `(role, structure, rank_band, size_bucket, arity)`；
- `(role, current_pattern, target_pattern)`；
- `(role, relation)`；
- canonical support 的 stabilizer、极小超群分解和 embeddings。

未访问 pattern/transition/relation 的初始 score 分别为 `1.25/0.35/0.20`。每次首次
发现 class 都开始新的 discovery epoch；节点、边、pattern 和缓存不清空，只重新
估计当前 epoch 的边价值并开放新的 child reserve。

## 12. Witness、Replay 与 Frontier

### 12.1 Witness

每个已验证 canonical facet 保存一条从 root 到达它的合法路径，优先修正次数更少，
其次路径更短；还按 `corrections_used` 保存 depth witness。Replay 始终从同一 root
重放，不创建独立 anchor tree。

### 12.2 自适应 Corrector 比重

每 4 次 simulation 至少一次周期 replay。启用 adaptive replay 后，自由 Filler 比例：

\[
r_{free}=\begin{cases}
0.75, & coverage<0.5,\\
0.10+0.65(1-coverage)^2, & coverage\ge0.5.
\end{cases}
\]

覆盖提高后，更多预算转向已发现 facet 间的 ridge/repartition 跳跃。v25 不启用 Filler
reheat，因此后期不会因停滞把自由比例强制抬回 35%。

### 12.3 Frontier 调度

`frontier_replay_stride=2`，周期 replay 会定期兑现未执行出口。当前使用
`yield_frontier=True`、`fair_frontier=False`；`ridge/orbit/completion/delivery`
保留固定 probe 份额，其余机会按新 target、新 pair 和服务次数调度。墙钟耗时只作
诊断，不进入选择分数，保持固定 seed 的逻辑可复现性。

距上次新 class 至少 500 次后，context frontier 检查尚未执行的
`(source,target,pattern,corrections)`，lane 为三次 `endpoint` 加一次 `bridge`。
候选仍需由合法 witness 重建并实际执行。

## 13. 缓存与性能设计

当前主要缓存包括 affine rank、orbit-union template、candidate action LRU、pattern
block/coherence LRU、共享 scorer 结构、terminal validation、canonical repair、
endpoint graph、completion frontier 和 pattern relation cache。

重 candidate cache 上限为 6000，轻 coherence cache 上限为 30000。候选先做廉价
shape 排序，每次 widening 最多对 6 个候选做真实 rank materialization；rollout 最多
检查 12 个动作。完整 reward 不进入缓存，动态新类和频率 reward 在命中时重新计算。

运行 JSON 保存 discovery 时间和完整 action path，以及树、回收、Corrector 反事实、
endpoint novelty、facet frontier 和 embedding 统计，可用于后验审计。

## 14. 当前实验结果

### 14.1 主 seed

环境为 `PYTHONHASHSEED=0`、BLAS/OMP 单线程、seed `3222026`、初始 known class 为空。
i3000 上限下在全覆盖时提前停止：

| 完成迭代 | 覆盖 | 耗时 | 节点 | 边 | 唯一终端 support | 最后发现 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2507 | 46/46 | 163.8385 s | 2432 | 2451 | 1874 | class 41 |

尾部发现时间：class 24 为 i1694，class 34 为 i1835，class 33 为 i2473，class 41
为 i2507。class 1 在 i1163 被实际终端命中。结果文件为
`runs/single_tree_group_game_i3000_v25_no_leaf_value.json`。

### 14.2 三个额外 seed

| seed | 完成迭代 | class 3 | class 24 | class 34 | class 41 | 最后发现 | 耗时 | 节点 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 673963275 | 4507 | 1297 | 4507 | 3208 | 791 | class 24 | 256.4864 s | 4417 |
| 1706386020 | 4386 | 1441 | 4386 | 548 | 4009 | class 24 | 236.1897 s | 4270 |
| 210656494 | 3154 | 2154 | 3061 | 3154 | 1442 | class 34 | 195.6599 s | 3080 |

三组均为 46/46，平均完成点 i4015.7，最晚 i4507，平均耗时 229.4453 秒，且均未
触发 depth limit。结构化汇总为 `runs/v25_multiseed_i10000_comparison.json`。

v20 在相同三个额外 seed 上分别为 46/46、45/46、44/46。对共同全覆盖的 seed
673963275，v25 从 i6824 降到 i4507，耗时从 341.3601 秒降到 256.4864 秒。
当前数据支持跨 seed 稳定性改善，但不构成数学完备性证明。

## 15. 消融结论

### 15.1 v24 到 v25

| 配置 | 完成迭代 | 覆盖 | 耗时 |
| --- | ---: | ---: | ---: |
| v24 full | 5770 | 46/46 | 341.3231 s |
| 去 family value sharing | 2534 | 46/46 | 165.5026 s |
| 再关闭 Filler reheat | 2507 | 46/46 | 164.0679 s |
| lean + embedding cap 2 | 4408 | 46/46 | 259.2728 s |
| lean + 关闭 relation UCB | 6000 | 45/46 | 361.7244 s |

所以 v25 保留 concrete embeddings、embedding relation、relation UCB 和 cap 4，
移除 family value sharing 与 Filler reheat。

### 15.2 Widening 与 reward

i3000 消融中，只保留 child widening 为 41/46，关闭 size bucket 为 45/46，关闭
action lane 为 26/46，关闭 epoch reserve/recycle 为 21/46。关闭 transition、support、
frequency、Corrector novelty 或 endpoint novelty 也都至少漏一类。因此这些机制保留。

唯一无损项为非终端 leaf heuristic。关闭后固定 seed 的发现序列、节点、边和终端
计数不变，耗时从 166.1970 秒降到 163.8385 秒，故 v25 默认关闭。完整证据见
`V24_ABLATION.md` 和 `POLICY_REWARD_ABLATION.md`。

## 16. 可复现实验命令

在 `E:\pycode\AI_Bell` 下运行：

```powershell
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& .\.venv\Scripts\python.exe -m mcts.group_game.lean_embedding_search `
  --iterations 10000 `
  --seed 3222026
```

入口自动设置：

```text
ridge_max_candidates             32768
ridge_batch_size                   256
ridge_focus_batches                 48
max_pattern_embeddings_per_family    4
max_pattern_analyses_per_class        4
```

默认输出为
`src/mcts/group_game/runs/single_tree_group_game_i{iterations}_v25.json`。

## 17. 当前边界与迁移注意事项

1. 46/46 是对当前 3-2-2 catalog 的实验覆盖，不是 facet 枚举完备性证明。
2. ridge、orbit face 和 lower-face completion 都有预算；frontier 耗尽只表示当前配置
   下没有待处理工作。
3. class 标签用于终端发现、动态 reward、停止条件和每 class 分析预算，不用于未到达
   终端前的动作构造。
4. 当前 support helper 固定为 64-bit；迁移到 4-2-2 时需换成任意宽度表示和相应 rank。
5. pattern refinement、family/embedding 分离、Corrector 反事实值与 repair 接口是维数
   无关设计；3-2-2 的 atlas 数量、rank 25 和枚举阈值不能原样照搬。
6. 增加 iteration 能降低部分 seed 方差，但当前结果说明调度和 credit 设计比单纯把
   预算提高到数万更重要。

## 18. 源码导航

| 文件 | 职责 |
| --- | --- |
| `lean_embedding_search.py` | v25 正式入口和推荐开关 |
| `search.py` | MCTS、widening、reward、rollout、replay 与结果记录 |
| `model.py` | 状态、动作、边和节点模型 |
| `knowledge.py` | concrete pattern/context/transition/relation 在线统计 |
| `facet_corrector.py` | FacetAnchor、保留面、repair plan 和出口发布 |
| `ridge_enumeration.py` | 有界、可续跑的 rank-24 ridge 枚举 |
| `ridge_pivot.py` | 相邻 facet pivot 接口 |
| `orbit_faces.py` | 大 facet 的 orbit 删除候选流 |
| `completion_frontier.py` | rank 17 到 23 保留面的 LP 续跑 |
| `repair_cache.py` | canonical 修复缓存和对称运输 |
| `endpoint_graph.py` | 已验证 canonical facet 邻接关系 |
| `yield_frontier.py` | class-independent frontier 调度 |
| `POLICY_REWARD_ABLATION.md` | widening 与 reward 消融证据 |

## 19. 总结

v25 是一棵以 concrete 群轨道 embedding 为动作语言、由 Filler 构造 facet、由
Corrector 沿在线发现的保留面和 ridge 跳跃，并通过 discovery epoch 与反事实 credit
持续重分配预算的合作式 MCTS。它不依赖未发现 class 样例，并在当前四个 seed 实验
中都完成了 3-2-2 的 46/46 覆盖。
