# Single-tree symmetry Filler-Corrector MCTS

这个目录记录一个与旧 `interrupt_search.py`、`subgroup_interrupt_search.py`
隔离的新算法。它不是在多棵固定 pattern 树之间做调度，而是把群论结构直接放入
一棵 MCTS 的状态和动作空间。

## 核心表示

- 状态是 `(64-bit vertex support, role, corrections_used, symmetry_context)`，不绑定
  固定 block partition。`symmetry_context` 记录最近采用或在线发现的 subgroup
  pattern，使同一个 support 在不同群论构造语境下可以学习不同的后续动作价值。
- Filler 动作是 `(subgroup pattern, one-or-more orbits)`。一次动作可以加入一个
  轨道，也可以加入同一子群下的少量轨道并集。
- Corrector 是同一棵树中的另一类节点。rank 25 状态既可以停止，也可以删除由
  subgroup orbit/orbit union 定义的宏块，或者原子地执行
  `remove orbit union + add orbit union` 的 rewrite，再返回 Filler。rewrite 既可
  使用同一 pattern，也可从当前 context 删除、由群关系图中的相邻 pattern 补入。
- 删除次数写入状态，因此 Corrector 不会产生无限的 add/remove 环。

这意味着同一个节点可以同时扩展 singleton、pair、C3/C4/C6 orbit，以及搜索过程
实际发现的 stabilizer orbit。pattern 不再等于一棵独立搜索树。

## 可学习部分

搜索使用两层 progressive widening：

1. 根据节点访问次数逐渐增加可用 subgroup pattern 数量。
2. 对已经打开的 pattern，逐渐增加 orbit/orbit-union 宏动作数量。

pattern 和动作先验由在线 contextual-UCB 统计更新。上下文包含 Filler/Corrector
角色、当前 symmetry context、context 间的群关系、pattern 结构、当前 rank 区间、
block 大小和轨道并集阶数。新 class 的全局 reward 为 10；发现后进入新的
discovery epoch，旧 edge 的 novelty Q 值不再直接支配选择，但历史结构统计仍被保留。

每个 simulation 最多向树中持久化一个新 edge。完成 selection/expansion 后，剩余
路径使用不写入树的 transient rollout，因此逻辑迭代数与树扩展数保持同一量级，
不会再把一整条随机 rollout 当作许多次 MCTS expansion。节点 child 上限同时依赖
访问次数和 discovery epoch；发现新 class 后旧节点会重新获得 widening 空间。

动作 widening 使用显式 quota 轮转 singleton、普通 orbit union 和 broad union；
Corrector 则轮转 small removal、1--4 orbit broad removal 与 rewrite，避免低编号或
小宏块候选长期占满 child 槽位。

v4 的 transient rollout 以八次 simulation 为一轮：一次 fine、一次 coherent、
六次 mixed（v3 为 3/3/2）。fine 只扩展有效大小不超过 2 的单轨道，coherent 在可行时以 0.85
概率继续当前 pattern，mixed 保留跨 pattern 的宏块探索。各策略使用独立 RNG，
共享同一棵树、discovered 集合和结构缓存。候选池同时保留高分与随机代表。

饱和节点在新 discovery epoch 到来且仍有未展开 pattern 时，最多回收 16 个已有
至少 8 次访问、平均回报非正的冗余 edge，每个 pattern 至少保留一个入口。
回收动作记入 retired 集合，避免立刻重新展开；已有状态缓存保留。

v4 还允许在连续 500 次 simulation 无新 class、且节点距上次回收至少经历 128 次
访问时执行相同回收。每次回收最多优先打开 4 个新的 pattern；参与过全局新类发现
的树边永久免于回收。所有触发原因、停滞长度、回收数量和新 pattern 都写入
`recycle_events`，epoch 与停滞两类计数分别报告。该机制不会提前结束总预算。

可用 `--stagnation-patience 0` 关闭停滞触发，或用 `--rollout-policy-cycle` 显式指定
策略轮转次序进行拆分实验；配置和策略配额都会写入结果文件。

已发现 class 的 reward 为负，并随该 class 和 canonical terminal basin 的重复次数
继续下降。因此一条曾经得到 novelty reward 的路径不会永久吸引搜索。

候选宏动作还使用与 class 身份无关的 affine-rank 和 supportability 几何量作为
先验。它们判断一个 prefix 是否仍可能落在 supporting hyperplane 上，不查询这个
hyperplane 最终属于哪个 class。

Filler 与 Corrector 是合作式 alternating game：二者共享“发现全局新 class”的
效用。Corrector 学习的是从重复或无效终端 basin 逃离后产生新结果的反事实价值，
而不是采用会主动破坏搜索目标的零和效用。

Corrector 的 edge Q 不直接使用整条 simulation reward，而使用：

`修改后首个终端语义效用 - 修改前 basin 的语义效用 - 修改成本 + 延迟 novelty 份额`

首次发现全局新 class 时，将总计 10 的 Corrector novelty credit 按距离折扣归一化，
分配给当前路径中发生在发现之前的所有 Corrector。首个后继终端只计算语义改进，
不再重复领取完整的 10；Filler 的普通终端 reward 仍使用全局新 class 奖励。
全局搜索对已知 class 的基本奖励是 -2，并附加频率惩罚；Corrector 对有效 exact
边界的语义基本值独立设为 0.25，避免强搜索惩罚反过来鼓励把有效边界破坏为 invalid。

这里的语义效用只比较 exact class、invalid boundary、non-facet 等终端性质，不把
“第一次见到某个 invalid support”的 basin novelty 当成改进。新 basin 只保留很小
的探索 bonus，避免 invalid-to-invalid 的删除或 rewrite 获得虚假的正反馈。

终端发现只向发生在它之前的动作传播。这样在一次 rollout 中先发现 class、随后才
执行的删除动作不会错误获得该 class 的发现 credit。

## 群论知识库

初始知识只有与 facet 无关的 Bell 322 群 atlas：

- identity singleton partition；
- 24 个极小超群代表；
- 极小超群到二级超群的结构边，但二级 pattern 初始不激活。

只有搜索真正走到 rank 25 并完成终端验证后，才会：

- 对有效 facet 计算该 tight support 的 stabilizer 和 orbit partition；
- 记录它的极小超群分解和二级 subgroup 匹配；
- 将观测到的 stabilizer pattern 加入 Filler/Corrector 动作库；
- 沿已观测到的极小超群关系激活有限数量的二级 pattern。

未发现 class 的代表 support、Compatibility Bank 和 class 特定 target bank 均不参与
动作生成。class 数据只用于 rank 25 后验证与报告。

## v6: 持久复访与自适应修正

- Filler 对已知有效 facet 的基础奖励改为 0.10，class 和 support 重复惩罚分别
  封顶 0.25，保证重复有效终端的单次 reward 不低于 -0.40，优于 invalid 的 -1.50。
  新 invalid support 不再领取 novelty bonus；真正全局新 class 的奖励仍为 10。
- 每个在线 canonical facet 保存一条合法 witness，优先更少修正、其次更短路径。
  每四次 simulation 分配一次公平复访，从同一个 root 沿 witness 重放，逐轮新增
  最多一条持久边，随后照常 rollout 和回传。不是独立 anchor 树或另一搜索进程。
  满容量时只回收无历史 discovery credit 的边，并单独记录回收数量。
- Corrector 基础上限仍为 4。若当前是有效 facet、本条路径未重复到达它、且仍有
  可生成或未试的补全出口，允许延长到 8；三次回到同一 canonical facet 时即使未到
  基础上限也停止。树选择和 rollout 都遵守停止规则。
- 低 rank 补全失败后，对非 identity 的 K 额外尝试一次 identity 细化补全，保持 R
  不变。仍需通过原 facet 外部、rank 25、supporting facet 的验证，不使用 class 目标。
- `repair_cache.py` 在 canonical 坐标下缓存已验证的 R、出口、partition 和生成元。
  通过群元素和共轭运输到当前坐标后复用；缓存没有 class 标签或 reward。
  本地几何生成与缓存获取交替，保留每个原始 anchor 的 48 次本地尝试预算。
  出口使用次数也在 canonical 坐标下共享，优先尝试未用出口，保留随机探索份额。
- 大 JSON 使用临时文件加替换写出。不要用文本补丁编辑大型运行 JSON；统计重算
  应通过结构化 JSON 解析/序列化生成，并保留原始配置与时间。

新参数为 `--extended-max-corrections`（默认 8）和 `--facet-replay-interval`
（默认 4，0 关闭复访）。`tree.facet_replay` 报告复访、持久化与扩展修正工作量；
`facet_corrector.statistics` 报告运输与细化补全计数。

## v7: 共享的系统 Ridge 枚举

`ridge_enumeration.py` 对在线验证过的 facet T 使用仿射依赖空间（Gale nullspace），
逐步枚举能留下 25 个独立顶点的补集。补集上的一维、同号 slack 暴露一个 rank-24
保留面；零 slack 的额外顶点也保留，因此可以处理非 simplex 和退化 ridge。
每个保留面再枚举 T 外的 singleton，仍通过完整几何验证才发布补全动作。

- 只有候选总数 `C(|T|, 25) <= 4096` 时启用，所以当前覆盖 26--28 顶点的 facet。
  这是通用工作量门槛，不查询 class 编号或未发现的样例。26 顶点 simplex 的全部
  26 个 ridge 都能被枚举；完成枚举不等于树已实际走过所有出口。
- 每次最多检查 32 个补集、最多发布一个计划，每四次请求尝试一次；本地随机 LP
  耗尽后仍可以继续枚举。原有群分块 LP、奖励、复访和 Corrector 深度保持不变。
- 进度按 canonical facet 共享，保留面通过真实群元素运输回当前坐标；已验证计划
  继续进入 v6 的几何缓存。没有增加独立树，也不直接把 look-ahead 出口计为发现。
- `--ridge-max-candidates 0` 关闭新增通道；`--ridge-batch-size` 控制单次工作量。
  `facet_corrector.systematic_ridges` 区分不适用、尚未完成与枚举完成，记录检查、
  拒绝和重复数量；`statistics.systematic_*` 记录验证与发布开销。

同 seed 的 i15000 对照：v6 为 43/46、739.43 秒，v7 为 45/46、749.02 秒，
新增 class 4、18，无旧结果丢失，仅缺 class 3。14 个适用 facet 的系统枚举尚未
完成；结果不是局部出口穷尽或多 seed 稳定性证明。详见 `runs/record.md` 和
`runs/v6_v7_i15000_comparison.json`。

## v8: 有界批次与 Frontier 出口兑现

- 系统枚举遇到重复计划、后验 rank 不合格或无外部出口时，在剩余批次额度内
  继续推进；每次仍最多检查 32 个组合、发布一个计划。Gale 子矩阵使用绝对秩
  容差，避免 pyramid 顶点处理论零行的舍入噪声被当作真实维度。
- canonical 缓存维护未使用的 `(R, exit)` 集合，规划时不消耗，真正执行补全后
  才移除。几何计划跨朝向运输时会合并同一 R/partition 的已验证出口。
- 默认每两次原有复访，一次保留旧策略，一次公平选择有待检查组合或未使用出口
  的在线 facet。后者推进一次有界枚举，再按计划轮转选取一个未使用出口，将
  Corrector/Filler 两步接在合法 witness 后，仍从同一个 root 执行、验证与回传。
- 不在规划时查询终端 class，不提前计入发现；总深度、Corrector 上限和路径重复
  停止仍生效。每次 simulation 最多新增一条持久边，余下步骤为暂态 rollout。
- `--frontier-replay-stride 2` 是默认轮转间隔，0 关闭新增复访通道。
  `tree.facet_replay.frontier_*` 区分规划与实际执行；
  `facet_corrector.untried_exit_counts` 记录每个 canonical facet 尚未使用的出口数。

同 seed i15000 实验：v8 为 44/46、710.42 秒，较 v7 快 5.15%，但遗漏 class 3、24，
覆盖比 v7 少一类。因此 v8 保留为独立实验，v7 仍为覆盖基线；不把推进量提升等同于
最终覆盖提升。详见 `runs/v7_v8_i15000_comparison.json` 和 `runs/record.md`。

## v9: 完整 Facet 终点去重

`endpoint_graph.py` 仅根据已经通过几何验证的补全计算完整 tight support，再用
Bell 对称群计算 canonical support。它不加载 class 标签，也不把 look-ahead
终点加入 discovered 集合或在线 pattern 知识库。

- 强制复访以 `(canonical source, canonical target)` 为单位，而非 `(R, added)`。
  同一关系实际执行后，其余等价补全不再占用强制复访；与源对称等价的终点也不
  使用此名额。不同源到同一终点仍是不同关系。
- 普通 MCTS、LP 修复和随机探索保留所有合法动作，不删除整个搜索空间中的
  等价路径。没有待兑现的独立终点时，复访回到原策略，未完成枚举仍可继续推进。
- 终点在验证时缓存，运输计划共用映射。只有已发布计划进入待执行图，避免重复
  或废弃候选造成不可执行的调度任务；规划不消耗关系，实际执行才记账。
- `--no-endpoint-dedup` 关闭此修改。结合 `--frontier-replay-stride 0`，可对 v8
  的数值/有界批次修改做无强制复访的消融；这不等于完整回退到 v7。
- `facet_corrector.endpoint_graph` 保存动作数、独立源/终点关系、重复执行和
  自身等价跳转统计。图中几何终点数与实际发现 class 数是两个不同指标。

同 seed i15000：仅数值/批次的消融为 44 类、764.03 秒，完整 v9 为 43 类、
764.87 秒（缺 3、24、34），均未超过 v7 的 45 类。v9 在 128.13 秒找齐其最终
43 类，但尾部没有增益，仍保留为实验版本。终点图仅覆盖已发布计划，不能据此
排除发布前丢弃的候选中存在新出口；详见 `runs/v9_i15000_matrix.json` 和运行记录。

## v10: 本地出口合并与缓存增量传播

- 本地发布遇到相同 `(R, pattern)` 时，按集合合并新增的已验证出口，不再直接
  丢弃整个计划；完全相同的重复发布不改变计划、缓存或待执行集合。
- 新出口同步更新 `anchor.plans`、Filler 出口表、canonical 共享缓存和终点图。
  缓存读取也检查新增出口，允许已有本地计划接收其他坐标副本的增量。
- 已执行的旧出口不会因重复发布重新获得未用状态。只缓存结构和几何结果，
  不缓存完整 reward，不提前分类或计入 class 发现。
- `publication_audit` 报告已验证候选与已发布动作之间的差集；
  `local_merge_events` 记录本地新增出口及发生的 iteration（最多 2000 条）。
  `merged_local_*`、`merged_cached_exits` 区分本地发布和缓存运输的合并工作量。
- 不改变 v9 的种子、预算、终点去重、修复深度、枚举范围或奖励，用 i15000
  单独验证发布链路修正的效果。旧 v7/v9 结果和源码快照继续保留。

同 seed i15000：v10 为 44/46、765.20 秒，遗漏 3、36；相较 v9 找回 24、34，
但未找到 36，净增一类且耗时基本相同。18962 个已验证候选动作全部发布；46 个
本地新增出口均指向当时已经发现的类别，不能把覆盖变化解释为直接补回新类出口。
v7 仍为本框架的覆盖基线（45/46、749.02 秒）。回归 113 项通过，完整诊断见
`runs/v9_v10_i15000_comparison.json`、`runs/v7_v10_i15000_comparison.json`。

## v11: 低 Rank 保留面有界续跑

本轮只检验低 rank 面的多次补全，不同时扩大大 facet 的保留面生成范围。
`completion_frontier.py` 将已生成的 rank 17--23 保留面变为可续跑工作，仍处于
同一棵 MCTS 内，不创建独立搜索树。

- 按共享缓存坐标中的 `(canonical source, retained, embedded pattern)` 记账，
  包括初次尝试在内最多 6 次 LP；同一 key 的不同坐标副本不重置预算。
  此处不声称已经对整个 pair 的 stabilizer orbit 做最小化。
- 普通生成每四次请求给予一次续跑机会，随机生成额度耗尽后也能续跑；现有
  frontier 复访在系统枚举没有产出时推进该队列。每批最多 2 次 LP，按源内部
  round-robin 轮换，达到上限即移出待处理队列。
- 首次补全沿用原 RNG，续跑使用从几何 key 派生的独立 RNG。继续保留实际子群
  的不变空间约束和 supporting facet 验证，新增出口走 v10 合并/运输链路。
- 不根据 class 编号或未发现样例选面，不将 look-ahead 计为发现。所有旧 reward、
  修复深度、rank 下限和小 facet 系统枚举参数不变。
- `--lower-face-max-attempts 0` 关闭此机制；`lower_face_frontier.jobs` 记录每个 key
  的 LP 次数、续跑批次、有效出口数与是否耗尽，`completion_continuation_*`
  区分新增工作量。预算耗尽不是该保留面已经被完整枚举的证明。

同 seed i15000：v11 为 **45/46、698.71 秒**，只缺 class 3；相较 v10 多 class 36，
没有丢失其他类，耗时减少 8.69%。与 v7 覆盖相同、耗时减少 6.72%。在 i9696、
452.99 秒达到最终 45 类，之后 5304 次仍无新增，不能据此认为总预算越多越好。
1234 个补全任务中 1233 个耗尽预算，613 个没有有效外部出口。118 项测试通过；
关闭续跑的 i200 发现路径和树统计与 v10 一致。完整对照见
`runs/v10_v11_i15000_comparison.json`、`runs/v7_v11_i15000_comparison.json`。
这是单 seed 的改善，尚未验证多 seed 稳定性；旧基线继续保留。

## v12: 大 Facet 的 Orbit 删除候选

`orbit_faces.py` 仅使用在线已发现 facet 及其真实 embedded 子群划分，不加载
未发现 class 样例。对不适用现有小 facet ridge 枚举的 canonical 源，注册一个
共享候选流；不同坐标副本不重置进度。

- 在 identity、稳定子及已有 cyclic 子群划分之间，轮转枚举删除 1--3 个完整
  orbit 的组合。每种 pattern 的 orbit 顺序由独立几何 seed 打乱；不预生成
  全部组合。每个源最多 256 个候选，每批最多 8 个，遇到一个有效保留面即返回。
- 保留面必须满足原有 rank 下限至 24，并通过 Gale 依赖的正 slack 检查，确保
  它确实是源 facet 的支撑面。多维 slack 核使用小型可行性 LP，而非只检查 rank。
- rank 24 的面交给现有外部 orbit 补全，低 rank 面交给 v11 有界补全与续跑；
  最终出口依然进行完整 supporting facet 验证，再通过 v10 发布/运输链路进入树。
- 原随机面生成和小 facet 枚举保留。新通道不将保留面写入旧 `seen_faces`，
  避免仅因新通道看过某种划分就屏蔽旧生成策略；它会改变有限预算内的树轨迹。
- `--orbit-face-max-candidates 0` 关闭。`orbit_face_frontiers` 记录各候选流的
  进度、rank/支撑面拒绝原因及接受面；`orbit_face_publication_events` 记录发布
  的迭代、源、保留面、pattern 和出口。`finished` 可能仅表示额度耗尽，只有
  `exhaustive` 才表示该有限 1--3 orbit 组合流被全部检查；两者都不是全空间穷尽。

同 seed i15000：v12 为 **45/46、681.43 秒**，仍缺 class 3。与 v11 相比覆盖集合
不变，总耗时减少 2.47%；在 i7304、342.47 秒达到最终 45 类，比 v11 提前 110.52
秒，但之后 7696 次、338.96 秒无新增。新通道接受 1445 个支撑面，留下 1399 个
未耗尽低 rank 补全任务，小 facet 枚举只完成 8/14，提示需要平衡生成与补全预算。
122 项测试及独立几何审计通过，关闭开关的 i200 与 v11 路径/树统计一致。
详见 `runs/v11_v12_i15000_comparison.json`、`runs/v7_v12_i15000_comparison.json`。
这是单 seed 实验，不把候选增多、较早达到已有覆盖解释为找全或稳定加速的证明。

## v13: 通道配额与补全积压反馈

`frontier_scheduler.py` 将强制复访改为先选工作类型、再在该类型内公平选 canonical
facet。普通 `grow` 使用每个 canonical 源共享的本地轮转进度；不同坐标副本不
重置相位。仍在同一棵 MCTS 内，不启动额外搜索树或并行任务。

- 正常 8 个机会中：ridge 枚举 2、已有补全 3、orbit 生成 2、已有出口执行 1。
  某个可用补全源的剩余 LP 名额达到 24 时：ridge 2、补全 4、orbit 1、出口 1。
  orbit 不会因积压被永久关闭；空通道跳过，份额不是 CPU 时间百分比。
- 每次调度最多推进一个有界工作批次，不再在一次机会内依次尝试三种通道，
  也不因本次没得到计划就让另一种几何工作立即抢占份额。普通借用缓存与随机
  面生成保留原有触发条件；通道配额并不是整个算法全部计算的硬配额。
- 补全积压按共享队列中尚未执行的 LP 名额统计，不按重复 plan 数或 class 标签。
  已验证出口有独立交付机会，几何规划不提前发现或消耗它。
- `frontier_scheduling` 记录选择次数、各源服务计数和积压模式；
  `scheduled_*_dispatches/new_pairs/plans` 区分选中、实际调度、新几何关系和计划。
  同一源/终点的新关系也可能属于已发现 class，本轮没有按这些收益动态学习权重。
- `--no-fair-frontier` 关闭此修改。其余候选范围、奖励、总迭代、修复深度和
  v11/v12 单任务预算保持不变，使用同 seed i15000 做对照。

同 seed i15000：v13 为 **44/46、722.07 秒**，缺 3、34；比 v12 少一类且慢 5.96%，
属于负向实验。虽然未完成补全任务从 1399 降至 279，但 orbit 候选检查从 7250
降至 3405，小 facet 枚举完成数仍为 8/14。3134 次补全调度只产生 19 种新源/终点
关系，不能把队列变短等同于搜索进步。128 项测试通过，关闭开关的 i200 与 v12
路径及几何任务一致。继续以 v12 为效果基线时请传入 `--no-fair-frontier`；当前
当时默认配置仍启用 v13，旧源码和结果均保留，尚未把本次失败解释为某个缺失类的专属问题。
详见 `runs/v12_v13_i15000_comparison.json` 和 `runs/v13_frontier_diagnostics.json`。

## v15: 候选缓存与饱和出口门控

v15 保留 v14 的稀疏几何收益调度，同时针对 profile 中的两个主要问题做隔离修改：

- `MacroAction.action_id` 按不可变动作缓存；`(support, pattern)` coherence 使用
  30000 项 LRU；动态动作结构按 `(role, support, context, pattern)` 使用 6000 项
  LRU。学习中的 pattern/action value 不进入缓存，因此 discovery epoch 重评分不失效。
- 同一 pattern 的动作排序只比较会随动作变化的尺寸和 orbit-union 项；pattern
  value、关系和 coherence 在这个集合内均为常数。rewrite 对相同删除大小复用
  addition 排序。这些是等价计算优化，不缩小候选集合。
- 若一个已观察 facet 源已经产生过出口、且没有尚未执行的非自身 canonical
  终点，普通树/rollout 再次进入 facet repartition 的总概率从 0.75 降至 0.12。
  未生成出口或仍有待执行终点的源保持原概率，强制 frontier replay 不受此门控；
  规划仍不提前分类或发现。门控释放的机会进入原有通用 Corrector 动作。
- `--no-endpoint-saturation-gate` 只关闭策略门控、保留 v14；
  `--no-yield-frontier` / `--no-fair-frontier` 同时关闭门控并恢复 v12 CLI 行为。
  关闭 endpoint dedup 时门控也自动关闭。

同为 i500 的 cProfile 条件下，关闭门控以隔离纯计算优化，搜索内计时由
58.7062 秒降至 47.1909 秒（19.61%），函数调用由 1.759 亿降至 1.202 亿。
这是 profiler 下的单次局部测量，不等于完整 i15000 的稳定加速。正式覆盖与
耗时结果见 `runs/record.md`。

正式同 seed i15000 为 **45/46、883.07 秒**，缺 class 34。相对 v14 新增
class 3，但慢 24.62%；相对 v12 是相同覆盖、以 class 3 替换 class 34，慢
29.59%。class 3 在 i4796 / 257.92 秒由本次运行已观察到的 class 45 稳定子群
宏动作和三轮在线 facet 修复到达，没有 class 3 样例。门控将 v14 的 27081 次
外部执行降至 10626，重复关系 26690 降至 10227，自身跳转 11456 降至 2109，
但通用候选生成增多，完整耗时反而增加。当前全局 source-target 饱和判据对长
修正链过粗：旧 v12 的 class 34 路径需要 7 次连续修正，已执行关系在不同路径
上下文中仍可能是必要桥梁。因此 v15 是有覆盖价值但效率回退的实验，不替代
v12 基线。详见 `runs/v15_diagnostics.json` 和两个 v15 comparison 文件。

## v16: Lazy Lane 与上下文桥接门控

v16 仍使用一棵 MCTS 树，不读取未发现 class 的 support、pattern 或标签来生成
动作。它分别修正 v15 的候选构造成本和过粗的全局出口饱和判据。

- progressive widening 先确定当前 `fine/single/union/broad/remove/rewrite` lane，
  `candidate_actions_for_pattern` 只实例化该动作族。只有在排除节点已展开、已拒绝
  和已回收动作后 lane 为空，才惰性构造完整候选作为旧语义 fallback；因此不会
  把“本 lane 已用完”误判为整个 pattern 耗尽。lane 进入有界 LRU cache key，
  `lane_fallbacks` 单独记录回退次数。
- 一个已验证外部出口只有在树或 rollout 真正执行时才计数。上下文键为
  `(canonical_source, canonical_target, repair_pattern_id, corrections_used)`；同一
  source-target 在新的 repair pattern 或修正深度中仍是开放桥梁。只有当前可选
  repartition 的所有出口上下文都执行过，才应用原 0.12 饱和概率。
- 上下文开放时优先未执行上下文，再以全局 source-target 次数作为弱 tie-break。
  `endpoint_contexts` 输出每个键的执行次数和修正深度分布，并显式声明
  `class_labels_used=false`。关闭 `--no-endpoint-saturation-gate` 后不使用上下文
  排序；关闭 yield 与门控的隔离模式继续逐项复现 v12/v14 搜索语义。

i200 同配置 smoke 与 v15 均发现 class 43、44，耗时 18.5423 -> 13.6779 秒，
快 26.23%。进一步同时关闭 yield 与门控时，v16 与保存的 v14-disabled 结果在
节点、边、tree/rollout 步数、动作计数和全部 facet 几何统计上完全一致，耗时
17.0452 -> 13.2050 秒，快 22.53%。i500 关闭门控的 cProfile 中，候选生成累计
时间由 9.878 -> 4.553 秒，总搜索内计时由 47.1909 -> 41.1776 秒；这是隔离的
单次性能证据，不代表默认门控的最终覆盖或完整 i15000 耗时。

正式同 seed i15000 为 **44/46、757.3270 秒**，缺 class 3、34。相对 v15
快 14.24%，但少 class 3；相对 v14 覆盖相同、慢 6.87%；相对 v12 少 class 34、
慢 11.14%。上下文门将外部执行由 v15 的 10626 恢复到 15045，发现路径上的
外部步骤由 108 恢复到 152，且实际命中深度 7/8 的 class，说明长链不再被全局
饱和直接截断。但 10408 次执行仍是完整 context 重复，class 3/34 的最终 target
均未进入 endpoint graph；问题已推进为有限 Corrector 候选和树内上下文调度没有
串起完整链。v16 的 lazy lane 可保留作为等价性能优化，当前 context gate 仍是
覆盖负向实验，不替代 v12 基线。完整数据见 `runs/v16_diagnostics.json` 及
`runs/v12_v16_i15000_comparison.json`、`runs/v14_v16_i15000_comparison.json`、
`runs/v15_v16_i15000_comparison.json`。

## v17--v20: 深度见证、桥接复访与确定性公平增长

v17 将在线发现的 facet 见证按 `(canonical facet, corrections_used)` 保存，并在
同一棵树内交替使用 context/global 饱和策略。真实执行首次产生新 endpoint
context、source-target pair 或 canonical target 时才给小额几何 novelty reward，
并把 credit 回传到前一 Corrector；长期停滞后可从精确深度见证复建路径。v17
i15000 为 44/46，缺 class 3、34。

v18 增加 context geometry growth：只从本次运行已观察的深度见证中选择 facet，
继续其 ridge/orbit/completion 几何枚举；选择信号只来自在线生成的新 target、pair
和 plan，不查询 class 标签。v18 得到 45/46，缺 class 3。v19 又加入
`endpoint, endpoint, endpoint, bridge` 复访周期，使已执行 pair 能在新的
pattern/depth 上作为桥梁重放；但按实测秒数归一化会让廉价空工作或一次性局部
收益长期占优，正式结果回落到 44/46。

v20 将两个自适应层都改成确定性 dispatch 收益：墙钟时间只记录诊断，不参与
选择。context growth 只有新增 canonical target 或非自身 pair 才算 productive，
单纯生成 plan 不再重置空 streak；每个来源先公平 warm-up，普通来源最多领先
当前服务下限 8 次，曾生成新 target 的来源额外获得 16 次上限。所有规则均只用
当前运行产生的几何结构，不使用未发现 class 的 support、pattern 或标签。

固定 `PYTHONHASHSEED=0`、BLAS/OMP 单线程、seed 3222026、初始已知集合为空的
i15000 在 **i6712 提前完成 46/46**。class 24 在 i5720，最后的 class 3 在
i6712；后者由 4 次在线 facet 修正、3 种运行中导出的 embedded pattern 到达。
无快照耗时 323.7520 秒；`--snapshot-interval 500` 的严格复跑为 358.5630 秒，
46 条发现的 iteration 和完整 action path 逐项相同，快照 I/O 增加 34.8110 秒。
源码见 `runs/source_v20.zip`，正式结果与审计见
`runs/single_tree_group_game_i15000_v20_snapshot500.json` 和
`runs/v19_v20_i15000_comparison.json`。group-game 与保留的串行 interrupt/pair/
Filler-Corrector 相关回归共 152 项通过；全仓库收集仍受环境缺少 `mpmath` 阻断。

## v14: 稀疏几何收益调度

从 v12 的局部工作顺序出发，默认不使用 v13 的积压配额。每 4 次强制 frontier
复访仅 1 次使用 `yield_frontier.py` 选择源与通道，其余保留 v12 的源轮转和
ridge/orbit/completion 顺序。自适应选择中，每 4 次有 1 次按通道轮转探测；
探测内优先较少获得自适应服务的 canonical 源，空通道跳过。

- 在原有几何批次外计时，不增加额外 LP。局部 grow 与 legacy replay 的真实
  工作也参与估计；不把无可用工作时的空调用当成廉价成功。
- 收益 = 新 canonical 终点数 + 0.1 * 新非自身源/终点关系数。此前已在出口图或
  已观察 facet 池中的终点不再获得第一项奖励。同一个新终点的不同源关系
  只分别贡献第二项；重复 plan 别名和自身跳转不贡献收益。
- 每个 canonical 源/通道的收益与秒数使用 0.2 EMA，新源使用通道先验。
  利用分数为 `(EMA收益 + 0.05/sqrt(1+观察次数+自适应服务次数)) /
  max(EMA秒数, 0.0005)`。这里的先验仅来自本次运行的在线统计。
- delivery 不生成几何，保留定期探测，并继续在所有几何工作后尝试交付。
  收益评估不分类、不提前发现、不消耗出口，不改变终端全局 novelty reward。
- `yield_scheduling` 保存启用状态、逐源计量和逐次选择理由。计量只覆盖这三类
  几何工作，不是完整运行时间。测得的时间会影响选择，因此即使 seed 相同，
  不同硬件或负载也不保证路径逐位一致；单次结果不代表稳定加速。
- 默认启用 v14；`--no-yield-frontier` 或兼容的 `--no-fair-frontier` 恢复 v12。
  `--fair-frontier` 显式启用 v13。关闭 endpoint dedup 时也关闭收益调度。

同 seed i15000：**44/46、708.62 秒**，缺 3、34。比 v13 快 1.86%，但比 v12
慢 3.99% 且少一类，仍是负向的最终覆盖对照，不替代 v12 效果基线。i5000 时
42 类高于 v12 的 37 类，但最后发现发生于 i6552 / 309.44 秒，此后 399.18 秒
没有新增。464 次自适应选择含 116 次探测；三类受计量工作总计 28.65 秒，只占
完整耗时 4.04%，不能据此推断剩余耗时具体在哪个模块。下一步需要全程热点
分析和生成候选/重复轨迹层面的改进，而不是继续把预算转向便宜通道。
137 项测试与全部发现路径审计通过；关闭收益调度的 i200 与 v12 路径、树及
几何工作一致。详见 `runs/record.md`、`runs/v12_v14_i15000_comparison.json`、
`runs/v13_v14_i15000_comparison.json`、`runs/v14_yield_diagnostics.json`。

## 运行

### v5: 完整 Facet 引导的结构化 Corrector

`facet_corrector.py` 只接收搜索过程中已验证的完整 tight support、法向量与偏移。
不加载任何 class 样例；初始已知 class 编号也不提供几何信息。

- 计算当前坐标下真实的 `H = Stab(T)`，保存实际生成元。有限采样 `K <= H` 的
  cyclic 子群，连同 identity 和 H 的 orbit partition，最多保留 12 种划分。
  这些不是 atlas 共轭代表的直接套用；缓存按原始 T 区分坐标朝向。
- 在 K 不变线性函数空间内，对 T 的极多胞体解随机线性规划，得到由完整 K-orbit
  组成的保留面 R。每个 T 最多 48 次尝试，identity 与其他 K 交替分配尝试。
- rank 24 的 R 枚举原 facet 外的 K-orbit 并验证 supporting facet；rank 17--23
  的 R 最多执行两次受约束极多胞体 LP，几何验证多 orbit 补全。没有有效外部出口
  的候选不进入这条修正通道。这是有限 look-ahead，不是完整 facet 枚举。
- Corrector 将当前选点 S 重组为 R，可以删除也可以补入 T 中先前未选的点。
  Filler 接收 K、原始 T 和已验证的补全动作，禁止在 T 内重新封闭到 rank 25。
  `repair_source_word` 进入转置表 key，不同源 facet 的修复任务不会混为一个状态。
- 默认 75% 的机会尝试此通道，其余保留原有通用删改策略。所有动作仍位于同一棵
  MCTS 树，每次 simulation 最多增加一条持久边；并没有添加独立搜索树或调度器。

模式选择使用共享知识库中的在线价值，终端类别只在实际命中后分类。新增的
`facet_corrector` 记录几何尝试、拒绝原因、实际子群生成元和已验证出口数量；
`corrector_novelty_credit_events` 记录每次发现的多步 credit 分配。

局限：当前每个 facet 的子群/LP 尝试均有上限，不枚举全部子群，也未对不同共轭
朝向做缓存搬运；一次有效外部补全仍可能属于已发现 class。几何逃离不等于新类发现。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 10000 `
  --max-corrections 4 `
  --min-corrector-rank 17 `
  --output src\mcts\group_game\runs\single_tree_group_game_i10000_observed_only.json
```

默认已知集合为空。identity 的 64 个 singleton 只是搜索动作的起始划分，
不表示已经发现 class 1。只有实际终端分类命中才计入 `discovered_class_ids`、
覆盖率和全目标完成判据。

显式传入 `--known-classes 1` 只会抑制 class 1 的全局新类奖励，不读取其 support，
也不会把未命中的 class 1 计入覆盖。`reward_known_class_ids` 单独记录奖励去重集合。
历史 v5 对照运行曾显式使用 `--known-classes 1`，复现原配置的命令保留在 runs/record.md。

主要输出包括发现时间线、每条发现路径中的 pattern/block、单树节点统计、Filler
与 Corrector 的在线价值统计，以及由实际终端逐步形成的 symmetry knowledge base。
