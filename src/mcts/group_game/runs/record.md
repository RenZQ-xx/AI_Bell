# Single-tree group game runs

本目录只记录 `mcts.group_game` 的结果。该算法在一棵增广 MCTS 树内共同表示
Filler 和 Corrector，并把 subgroup orbit 作为动态宏动作；它不复用旧版多树
scheduler 的运行记录。

## Knowledge boundary

- 动作生成不读取任何 class representative。
- 初始结构知识来自 facet-free subgroup atlas。
- stabilizer pattern、极小超群分解和 level-2 pattern 激活，只能来自本次搜索已经
  到达并验证的终端 support。

后续正式运行在此追加命令、配置、耗时、发现 class 和首次发现 iteration。

## 2026-09-07 architecture probes

### Real scorer smoke, i2

- 文件：`single_tree_smoke_i2.json`
- 搜索耗时：`0.0781 s`，不含 atlas/scorer 初始化。
- 结果：48 个节点，进入 Filler 和 Corrector 两种节点；未发现新 class。
- 用途：验证真实 rank、终端验证和 JSON 输出链路。

### Real scorer quality probe, i300

- 文件：`single_tree_probe_i300.json`
- 配置：class 1 初始只记为已知，不读取其 support；`iterations=300`，
  `max_corrections=2`，开启 level-2 terminal analysis。
- 搜索耗时：`7.5955 s`，不含初始化。
- class 44 在 iteration 31、`0.920 s` 左右首次发现。
- 单树规模：3552 nodes；知识库 active pattern 从 25 增长到 89。
- 覆盖：`{1, 44}`。这次只用于确认“单树混合 pattern + 在线知识回流”的实现
  语义，尚未达到旧 subgroup interrupt search 的覆盖率。

中间诊断文件 `single_tree_smoke_i20.json` 和 `single_tree_probe_i100.json` 分别用于
检查终端到达率以及首次 class 44 发现的稳定性。

## 2026-09-07 counterfactual Corrector, class 1 i500

- 文件：`single_tree_group_game_class1_i500_cf.json`
- 配置：`iterations=500`，seed `3222026`，class 1 只作为初始 discovered 标记，
  `max_corrections=2`，开启 level-2 在线分析。
- 搜索耗时：`16.4705 s`，不含 atlas/scorer 初始化。
- class 44 在 iteration 109、`4.4158 s` 首次发现，发现路径使用了 2 次
  Corrector。
- 关键删除：从 `invalid:boundary` 删除 identity pattern 下两个 singleton 的并集
  `0x00000000000c0000`，随后到达 class 44；反事实 credit 为 `11.8742`。
- 单树规模：9078 nodes / 9086 edges；active pattern 由 25 增长到 89。
- 终端：1194 次 `invalid:boundary`，263 次 class 44，共 1158 个 canonical/raw
  terminal basin。
- Corrector 结果：742 improved、210 neutral、4 worse、1 new-class。
- 覆盖仍为 `{1, 44}`。频率惩罚减少了 class 44 的重复次数，但有效终端比例仍是
  下一阶段的主要瓶颈。

## 2026-09-08 counterfactual Corrector, class 1 i10000

- 文件：`single_tree_group_game_class1_i10000_cf.json`
- 配置与 i500 相同，仅将逻辑迭代提高到 `10000`，并每 500 次保存快照。
- 搜索耗时：`203.7611 s`，约 `49.1 iterations/s`，不含 atlas/scorer 初始化。
- class 44 仍是唯一新 class，在 iteration 109、`4.9053 s` 首次发现，经过 2 次
  Corrector；iteration 110--10000 没有覆盖提升。
- 单树规模：119356 nodes / 119777 edges，平均每次 simulation 新增约 11.9 个
  node。
- 终端：20948 次 `invalid:boundary`、6605 次 class 44；有效终端率约 24%，但
  所有有效终端均落在同一 class。unique terminal supports 为 20404。
- Corrector：12696 improved、4591 neutral、265 worse、1 new-class。最先记录的
  2000 条诊断中，1520 条 improved 实际是 `invalid -> invalid`。
- active pattern 为 89，但根节点受 48-child hard cap 限制，只包含 12 个 pattern；
  48 个根动作全部是二轨道并集，其中 40 个 block size 为 4。
- Filler 的 atlas action visits 中，`BFS322-C2-09` 单个 pattern 占约 56%，说明一次
  class 44 路径上的重复 pattern credit 仍造成长期偏置。
- 结论：增加到与旧 subgroup search 接近的逻辑预算并不能修复覆盖。下一版应先改
  真正的单节点 expansion + 非入树 rollout，再处理 epoch-aware child reopening、
  action arity quota、symmetry-context coherence 和 Corrector semantic credit。

## 2026-09-08 contextual single-expansion Corrector v2, class 1 i10000

- 文件：`single_tree_group_game_class1_i10000_v2.json`
- 命令：`.\.venv\Scripts\python.exe -m mcts.group_game --iterations 10000 --seed 3222026 --known-classes 1 --max-depth 72 --max-corrections 3 --min-corrector-rank 18 --snapshot-interval 500 --output src/mcts/group_game/runs/single_tree_group_game_class1_i10000_v2.json`
- revision：`contextual_single_expansion_rewrite_corrector_v2`。
- 搜索计时：`493.6757 s`，不含 atlas/scorer 初始化及最终 JSON 写出；中间快照写出
  包含在计时中。10000 次 simulation，9680 nodes / 10000 edges，77102 个临时
  rollout steps。根节点有 192 个 child、50 个 pattern，单轨道/并集为 94/98。
- 覆盖 `{1,27,29,38,42,43,44,45,46}`，其中 1 仅为初始已知标记；搜索新增 8 类。
- active pattern 为 206，其中 8 个在线 stabilizer pattern；根节点只容纳其中 4 个。
- 21790 次 invalid boundary，11140 次 exact，21576 个 unique terminal support。
  class 44/43 分别重复命中 8676/1989 次，占全部 exact 的 95.74%。

| Class | Iteration | Seconds | Corrector 次数 |
| --- | ---: | ---: | ---: |
| 44 | 1 | 0.109 | 0 |
| 43 | 14 | 3.715 | 3 |
| 38 | 809 | 50.429 | 0 |
| 46 | 1193 | 71.457 | 1 |
| 29 | 1276 | 77.008 | 2 |
| 45 | 1440 | 84.972 | 3 |
| 42 | 3381 | 178.262 | 1 |
| 27 | 5830 | 293.417 | 0 |

- 5/8 条首次发现路径经过 Corrector，全部使用 remove。class 43 的最后一次删除
  为 4-orbit/5-point；class 29 的两次删除均为 4-orbit，说明扩大删除阶数产生了
  实际新出口。rewrite 共执行 7923 次，但未出现在首次发现路径中。
- 最后 4170 次 iteration 没有发现，耗时约 `200.259 s`。所有首次发现路径只有
  4--17 个动作；这是偏好短宏块路径的证据，但不能据此推断所有长路径都没有探索。
- Corrector 全量结果：1550 worse、1103 improved、20281 neutral、5 new_class。
  详细 diagnostics 只保留最先 2000 条，不能当作全程无偏样本。
- 相较 v1：新增覆盖从 1 类增至 8 类、持久节点从 119356 降至 9680；搜索时间
  从 203.7611 增至 493.6757 秒。候选生成与临时 rollout 的新增工作使 wall time
  上升；较小的树不等于更短的搜索时间。
- 后续诊断发现 v2 的 Corrector 新类事件虽被记录，反事实回传仍采用普通 exact
  语义值，未正确使用 10 分的新类奖励。此问题在 v3 修复，v2 JSON 保留原始结果。

## 2026-09-08 v3 post-run adjustments

- 每八次 simulation 固定执行 3 fine / 3 coherent / 2 mixed，使用独立 RNG，
  共享单树与知识库；fine 保留大小不超过 2 的单轨道动作，coherent 在可行时以
  0.85 概率延续 context。候选池保留一个高分和一个随机代表。
- 已知类搜索基本 reward 调整为 -2，频率系数 0.18。Corrector 的 exact 语义值
  独立设为 0.25、频率系数 0.04；首次发现严格回传 10 分。Corrector 知识统计
  只使用它后面首个终端的结果，与反事实 reward 的归因区间一致。
- Corrector 最多 4 次，最低 rank 17；rewrite 删除大小分层轮转，并允许当前
  context 删除和相邻 pattern 补入，记录两端 pattern 身份。
- 保留 192/128 的 Filler/Corrector child 上限；新 epoch 对已满载且尚有未展开
  pattern 的节点最多回收 16 个重复 pattern 下的负回报 edge，每个 pattern 保留
  至少一个入口。退役 action 不再立即重新展开，状态缓存保留。
- 新增全程 Corrector 操作分类统计，弥补前 2000 条 detailed diagnostics 的限制。
- 功能短测 `single_tree_v3_probe_i200.json`：class 44 在 i1、class 43 在 i25；
  fine/coherent/mixed 为 75/75/50。该短测与回归测试有并行时间重叠，26.4666 秒
  只作链路验证记录，不用于性能结论。相关测试 84 passed。

## 2026-09-08 portfolio and cross-rewrite v3, class 1 i10000

- 文件：`single_tree_group_game_class1_i10000_v3.json`
- 命令：`.\.venv\Scripts\python.exe -m mcts.group_game --iterations 10000 --seed 3222026 --known-classes 1 --max-depth 72 --max-corrections 4 --min-corrector-rank 17 --snapshot-interval 500 --output src/mcts/group_game/runs/single_tree_group_game_class1_i10000_v3.json`
- revision：`portfolio_cross_rewrite_epoch_recycling_v3`。解释器为仓库根目录的 `.venv`。
- 完整执行 10000 次，搜索计时 `988.9949 s`（16 分 28.995 秒）；计时口径与 v2
  相同，不含初始化和最终 JSON 写出。全量运行期间未修改搜索实现。
- 新增 `{35,43,44,46}`；含初始已知标记 1，共 5/46，缺少 41 类。class 1 未在
  本轮被重新搜索命中，不应将初始标记计为本轮新发现。

| Class | Iteration | Seconds | Simulation 策略 | Corrector 次数 | 路径动作数 |
| --- | ---: | ---: | --- | ---: | ---: |
| 44 | 1 | 0.151 | fine | 0 | 19 |
| 43 | 25 | 5.619 | fine | 1 | 21 |
| 46 | 4477 | 463.521 | coherent | 0 | 6 |
| 35 | 4835 | 495.216 | mixed | 2 | 3 |

策略列表示该次 simulation 分配的 rollout 策略，路径还包含共享 tree policy 的
动作，因此不能仅凭该列认定某个策略对发现具有独立因果贡献。

### Class 35 的实际路径

1. 加入从已发现 class 46 学到的 `OBS-35ac5c3aa3c5ca53` stabilizer pattern 下
   一个 32-point orbit，support 为 `0x3a5cac3553cac5a3`。
2. 使用 atlas 的 `BFS322-H-0106`（C2 x C2），删除
   `0x1008800100000000` 并补入 `0x0000000008100810`，4 点换 4 点。
3. 在同一 pattern 下删除 `0x0000000001800180` 并补入
   `0x1008800100000000`，再次 4 点换 4 点，终端验证得到 class 35。

这是在线群结构知识回流与 rewrite 共同产生的新出口，不读取未发现 class 35 的
样例。四条发现路径均已按位回放，与输出 selected support 完全一致。

### 工作量与 Corrector

- 9790 nodes / 10000 edges，10000 次持久 expansion；25508 个 tree-policy steps，
  138391 个 transient rollout steps。没有 depth-limit 截断，3 次 dead end。
- fine/coherent/mixed 的 simulation 配额为 3750/3750/2500。
- active pattern 174 个，其中在线 stabilizer 4 个；根节点 192 个 edge、61 个
  pattern，包含 3 个在线 stabilizer。
- 终端 exact 8863 次（class 44: 6690，43: 1663，46: 506，35: 4），invalid
  boundary 30431 次；unique terminal supports 30238。44/43 仍占 exact 的约 94.25%。
- Corrector 全量结果：1612 worse、25612 neutral、2071 improved、2 new_class。
- 按操作分组：remove 18899 次，贡献 1 次 new_class；同 pattern rewrite 9467 次，
  贡献 1 次 new_class；跨相邻 pattern rewrite 931 次，36 次 improved、0 次 new_class。
  这些是全量已解析的 Corrector 结果，包含已展开 tree edge 被再次选择的执行。
- 最后 5165 次没有新发现，耗时 `493.7785 s`，约占总时间一半。
- epoch 回收实际触发 0 次。最后一次发现发生时根节点还未满载，之后才达到
  192-child 上限；此后没有新 epoch，因此此次运行没有验证槽位回收对覆盖的收益。

### 与 v2 对照及下一步判断

| 指标 | v2 i10000 | v3 i10000 |
| --- | ---: | ---: |
| 搜索时间（秒） | 493.6757 | 988.9949 |
| 本轮新发现 | 8 | 4 |
| 含初始 class 1 的覆盖 | 9/46 | 5/46 |
| 持久节点 | 9680 | 9790 |
| 临时 rollout steps | 77102 | 138391 |

- 新增 v2 未发现的 class 35，但丢失 v2 的 27、29、38、42、45。时间变为
  `2.0033` 倍，临时 rollout 工作量变为 `1.7949` 倍，本配置整体未优于 v2。
- fine 确实产生较长的首次发现路径，却未带来新的类别；固定 3/8 fine 配额是否
  值得保留，需要独立消融。跨 pattern rewrite 扩大了可执行动作，但目前无新类收益。
- 回收条件需要检验“访问停滞且仍有未展开 pattern”的非先验触发方式，避免只有
  已经发现新 class 才能打开新入口。
- 建议下一次以 v2 为覆盖参照，先单独检验 Corrector 新类 credit 修复和有收益的
  broad rewrite，再单独测试 portfolio 与停滞回收。此次同时修改多个因素，不能
  把覆盖下降归因于其中某一项；不建议直接继续提高总 iteration。

## 2026-09-09 stagnation frontier recycling v4, class 1 i10000

- 文件：`single_tree_group_game_class1_i10000_v4.json`
- 命令：`.\.venv\Scripts\python.exe -m mcts.group_game --iterations 10000 --seed 3222026 --known-classes 1 --max-depth 72 --max-corrections 4 --min-corrector-rank 17 --snapshot-interval 500 --output src/mcts/group_game/runs/single_tree_group_game_class1_i10000_v4.json`
- revision：`stagnation_frontier_recycling_mixed_portfolio_v4`。
- 本次保留 v3 的 reward、Corrector 范围和 rewrite；修改为 6 mixed / 1 coherent /
  1 fine 的八次轮转，并增加停滞回收。各策略 RNG 独立，总 iteration 不提前缩减。
- 停滞条件：连续 500 次 iteration 无新类、节点饱和且仍有未展开 pattern、距离
  该节点上次回收至少 128 次访问。最多回收 16 条经过至少 8 次访问、平均 reward
  非正的冗余 edge，每个 pattern 保留至少一个入口，并优先打开最多 4 个新 pattern。
- 新增 `EdgeStats.discovery_hits`，参与过新类发现的 tree edge 永久免于回收。
  `recycle_events` 记录触发原因、节点 support/context、迭代、时间、停滞长度、
  回收数量和 fresh pattern。
- 可通过 `--stagnation-patience 0` 或 `--rollout-policy-cycle ...` 单独控制后续
  消融因素。本次没有执行多 seed 或独立消融，不能分离两项修改各自的因果效应。

### 验证与结果

- 相关测试 `87 passed in 8.79s`。包括无新 epoch 时触发、500 次 patience、128 次
  本地访问冷却、历史发现路径保护，以及新入口实际获得扩展的测试。
- 真实 scorer 短测 `single_tree_v4_smoke_i50.json` 完成，搜索计时 7.4561 秒。
- 完整执行 10000 次，搜索计时 `829.7770 s`（13 分 49.777 秒），不含初始化和
  最终 JSON 写出，计时口径与 v2/v3 相同。运行期间未修改搜索实现。
- 搜索新增 `{23,43,44,46}`。含预设已知 class 1 为 5/46，仍缺 41 类；class 1
  本轮未重新命中。四条发现路径按位回放均与 selected support 一致。

| Class | Iteration | Seconds | Simulation 策略 | Corrector 次数 | 路径动作数 |
| --- | ---: | ---: | --- | ---: | ---: |
| 44 | 1 | 0.125 | mixed | 1 | 12 |
| 43 | 424 | 43.659 | mixed | 3 | 14 |
| 23 | 2860 | 246.422 | mixed | 3 | 9 |
| 46 | 8595 | 716.053 | coherent | 2 | 9 |

策略列标记该 simulation 分配的 rollout 策略，包含共享 tree-policy 路径的发现
不能据此独立归因于该策略。

### 回收与工作量

- 第一次停滞回收发生在 i924、85.358 秒，恰好距 class 43 的发现 500 次 iteration。
- 全程 56 次回收事件：55 次 stagnation 回收 357 条 edge，1 次 discovery epoch
  回收 16 条；总计 373 条。
- 根节点最终有 192 条 edge，覆盖全部 172 个活跃 pattern，4 条根边具有历史
  discovery credit 并被保留。i6000 时当时的全部 148 个活跃 pattern 已进入根节点。
- 9846 nodes，10000 次持久 expansion，9627 条保留 edge；满足
  `9627 + 357 + 16 = 10000`。退役边对应的状态缓存仍保留，node 数不等于可达边数。
- tree-policy steps 23111，transient rollout steps 115076；mixed/coherent/fine
  simulation 数为 7500/1250/1250；2 次 dead end，0 次 depth-limit 截断。
- exact 7489 次（44: 6580，43: 720，23: 1，46: 188），invalid boundary 31543 次，
  exact 比例约 19.19%，unique terminal support 为 31393。
- Corrector：4 new_class、2015 improved、1342 worse、25673 neutral。
  直接新类 credit 分配到 3 次 remove 和 1 次同 pattern rewrite；跨 pattern rewrite
  有 44 次 improved，但没有直接新类 credit。

### Class 23 与剩余瓶颈

- i2844 的根节点停滞回收打开 `BFS322-H-0076`；i2860 的 class 23 路径从这个
  pattern 的 4-point add 出发。其后混合若干 atlas pattern，到达终端后依次执行
  两次 `BFS322-C2-13` 的 1-point-to-1-point rewrite、一次 `BFS322-C2-07` 的
  4-point remove，最后由 `BFS322-H-0110` 加入 5 点，验证得到 class 23。
- 因此新回收入口实际进入了发现路径；这不是以 class 23 样例预先构造的路径。
  当前 Corrector 只把新类 credit 给紧邻发现的 remove，前两次 rewrite 的延迟
  贡献仍值得通过多步、归一化 credit assignment 单独检验。
- 最长无新类间隔仍有 5735 次 iteration（class 23 到 46），耗时 469.6303 秒；
  最后一次发现后的尾部为 1405 次、113.7244 秒。末尾缩短不表示整体长尾已解决。

| 指标 | v2 | v3 | v4 |
| --- | ---: | ---: | ---: |
| 搜索秒数 | 493.6757 | 988.9949 | 829.7770 |
| 新发现 class 数 | 8 | 4 | 4 |
| 临时 rollout steps | 77102 | 138391 | 115076 |

- 相比 v3 节省 159.2179 秒（16.10%），临时 rollout steps 减少 16.85%；新增数
  持平，增加 class 23、未复现 class 35。相比 v2 仍然更慢且覆盖更少。
- 根层已包含全部活跃 pattern，单纯扩大根层入口不能充分解释或解决剩余覆盖问题。
  下一步应优先检验 Filler 的 prefix 可行性/终端补全质量，以及多次 Corrector
  的延迟发现归因；继续提高总预算或单纯增加根 child 上限缺少本轮数据支持。

## 2026-09-09: v5 Facet-relative Subgroup Corrector, i10000

### 实现与边界

- 新模块 `../facet_corrector.py` 接收搜索已经验证的完整 tight support T，而不是
  仅按当前 basis S 删块。计算当前坐标下真实 H=Stab(T)，从 H 的实际 cyclic
  子群、H 本身和 identity 构造至多 12 种 K-orbit partition，保存生成元。
- 在 K 不变函数空间对 T 的极多胞体做随机 LP，构造完整 K-orbit 的保留面 R。
  每个原始 T 最多 48 次尝试。rank 24 枚举外部 orbit；rank 17--23 最多两次
  受约束 LP 做多 orbit 补全，均经几何 facet 验证。失败候选只计诊断，不授予动作。
- Corrector 重组 S 为 R，允许删除 S 的点，也允许补入 T 中此前未选的点。
  Filler 接收实际 K、源 T 和验证过的补全；必须加入 T 外的点，不能在 T 内重建
  相同 facet。源 T 纳入转置表 key。默认 75% 机会尝试此通道，其余保留通用删改。
- 每个全局新 class 向此前 Corrector 分配总计 10 的距离折扣归一化 credit。
  紧邻终端只计算语义改进，不重复领取完整 novelty；较早的修正也能收到延迟 credit。
- 所有修改隔离在 group_game 与对应测试，旧 serial subgroup interrupt 未修改。
  仍为一棵 MCTS 树，每个 simulation 最多一条持久 expansion。
- 未读取任何未发现 class 样例，也没有指定 rare class。元数据中的
  `facet_examples_used_for_action_generation=false` 指未加载外部 class 样例；
  `online_validated_facets_used_for_repairs=true` 明确说明使用在线已验证 facet。
  子群缓存使用原始坐标下 T，不把共轭代表错误套用到当前顶点；尚未实现共轭缓存搬运。

### 配置与验证

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 10000 --known-classes 1 `
  --max-corrections 4 --min-corrector-rank 17 --snapshot-interval 250 `
  --output src\mcts\group_game\runs\single_tree_group_game_class1_i10000_v5.json
```

- seed=3222026，mixed/coherent/fine 配额 7500/1250/1250，其他原有参数保持 v4。
  class 1 仅为初始已知编号，不加载其 support；本轮也未实际重新命中 class 1。
- 使用项目根目录 `.venv`，SciPy 1.16.3 为现有锁定环境依赖，未安装新包。
- 相关 91 项测试通过，完整运行前 8.88 秒，运行后复验 9.01 秒。
  新测试覆盖真实嵌入生成元、orbit 完整性、外部补全有效性、源 facet 转置隔离、
  原 facet 内封闭的拒绝，以及跨 Corrector 归一化 credit。
- 完整运行期间未修改算法代码。20 条发现路径按位回放均与 selected support 一致；
  其中 32 次结构化补全均独立通过几何验证，保留面均被记录的实际 K 生成元固定。
- 20 个 novelty 分配事件总 credit=200，每个事件严格为 10，没有给后续动作倒发奖励。

### 结果与发现时间

搜索计时 **791.5119 秒（13 分 11.512 秒）**，不含初始化及最终 JSON 写出，
与 v4 计时口径一致。完成 10000 次 iteration，实际发现 20 类，覆盖 **20/46**。
原报告将预设 class 1 算作覆盖的 21/46 不成立，已于统计复核时更正。

| Class | Iteration | Seconds |
| --- | ---: | ---: |
| 44 | 1 | 0.117 |
| 46 | 56 | 7.691 |
| 29 | 56 | 9.867 |
| 43 | 79 | 11.988 |
| 28 | 153 | 18.381 |
| 38 | 257 | 26.310 |
| 30 | 464 | 40.837 |
| 42 | 614 | 50.684 |
| 25 | 988 | 79.704 |
| 45 | 1031 | 83.497 |
| 23 | 1176 | 95.276 |
| 41 | 1221 | 98.901 |
| 33 | 1620 | 127.285 |
| 16 | 1620 | 127.427 |
| 35 | 1639 | 128.886 |
| 27 | 2691 | 211.726 |
| 17 | 3497 | 275.410 |
| 40 | 3829 | 302.625 |
| 32 | 5461 | 434.798 |
| 8 | 7298 | 579.872 |

仍缺 `{1,2,3,4,5,6,7,9,10,11,12,13,14,15,18,19,20,21,22,24,26,31,34,36,37,39}`。

### 结构化跳转的实际贡献

- 20 类中除 class 44 外的 19 类，首次命中的最后动作均为 `facet_external_exit`。
  其中 14 条首次发现路径的最后一次结构化修正采用非 identity 子群。
- 发现路径中的结构化保留面 rank 分布为 24:25 次、23:2 次、21:4 次、19:1 次。
  class 38 由 order-6 子群下 rank 19 保留面补全；class 25 由 rank 23 补全；
  class 33 由 order-4 子群下 rank 21 补全，同一 simulation 随后得到 class 16。
  因而不只是删除原 basis 的一两个点后重新填回同一超平面。
- 286 个在线 facet anchor，2800 个按 anchor 统计的嵌入 partition（含跨 anchor
  重复），3654 次保留面 LP，1420 次低 rank 补全 LP，20977 次 ridge 出口几何检查。
- 缓存 1977 个有效修正方案、14164 个出口动作。1272 个方案来自 identity，705 个
  来自其他 K；93 个方案是低于 rank 24 的保留面。实际重组与外部补全各执行 4162 次。
  出口数量不代表不同 facet/class 数量，同一个 facet 可由多个出口点表示。
- 主要拒绝/去重：1054 次重复保留面、617 个无有效外部出口的候选、6 个 rank 越界；
  低 rank 补全中 677 次停留源 facet 或未保住 R、612 次非 facet、4 次 LP 失败。
  另有 22 次所选 K 的有效不变函数空间为空。

### 与上一版单树对照

| 指标 | v4 | v5 |
| --- | ---: | ---: |
| Iterations | 10000 | 10000 |
| 搜索秒数 | 829.7770 | 791.5119 |
| 新发现 class 数 | 4 | 20 |
| 本轮实际命中覆盖（已纠正统计口径） | 4/46 | 20/46 |
| Tree-policy steps | 23111 | 22228 |
| Transient rollout steps | 115076 | 117141 |

- 本次比 v4 快 38.2650 秒（4.61%），覆盖增加且保留 v4 的全部四类。
  这是单 seed、多项联动修改的对照，不足以单独归因于某项优化，也不保证普遍加速。
- 9739 nodes、10000 次持久扩展、9613 条保留边；回收 40+347=387 条，满足
  `9613+387=10000`。2 次 dead end，0 次 depth-limit 截断。
- exact 8459 / 总终端 39294（21.53%），invalid 30835；根出发的大量无效 rollout
  仍是主要浪费。结构化外部跳转也仍会落到已知 class，几何逃离不等于 class novelty。
- 最后一次发现后仍运行 2702 次、211.6396 秒。当前尚未找到全部 class，也未达到
  旧 serial subgroup interrupt 的全覆盖。后续宜先检验已验证 facet 的复访分配、
  低 rank 非 facet 补全的拒绝原因及对称等价缓存复用，不能仅由本轮推断加预算即可找全。

### 统计纠正与预算复核

- singleton pattern 不表示发现 class 1。默认 `known_classes` 改为空；即使显式
  指定已知编号，覆盖与全目标停止判据也只使用本轮正次数终端命中。奖励去重集合
  与本轮命中集合分开报告。上文历史 v2--v4 中含初始编号的集合大小不应视为覆盖率。
- v5 JSON 只更正统计字段并增加 `coverage_corrected_after_run=true`；原运行配置、
  轨迹、发现时间、奖励与耗时保持不变；没有假定旧运行使用空集合。
- 实际覆盖随预算变化：i1000=9，i2000=15，i3000=16，i4000=18，i5000=18，
  i6000=19，i7500=20，i10000=20。增加预算仍曾带来发现，但边际收益明显下降。
- 286 个原始坐标 facet anchor 中只有 28 个达到 48 次局部尝试上限，192 个少于
  8 次尝试，总共使用 3676 次尝试，配置上尚有 10052 次未使用。原始 anchor 之间
  可能对称等价，这些数字不能解释为相同数量的独立搜索方向。
- 下一步优先考虑同一树内有效 facet 的持久保留/复访、按实际出口新颖性分配
  局部尝试，以及低 rank 补全的退化面细化。单纯增大总 iterations 或统一增大
  每个 anchor 的上限，均不能保证把预算送到有价值的分支。
- 另一个待验证问题是 Filler 的基础奖励排序：known valid 为 -2，invalid 为 -1.5，
  且 known 的频率惩罚会继续增大。Corrector 独立语义效用虽避免直接奖励破坏 facet，
  但 Filler 的累计回传仍可能压低已知有效且可继续修复的分支。建议与成功 rollout
  路径的持久保留一起做下一版对照；本次只修正统计，没有改动这些奖励值。
- 更正大 JSON 的过程中，后半段结构明细发生损坏。已保留损坏文本备份，并使用
  原参数的 `single_tree_group_game_class1_i10000_v5_recovery.json` 做确定性恢复。
  复跑完成 10000 次，耗时 801.7184 秒；全部非计时发现轨迹、增长/回收事件、
  Corrector 诊断、奖励分配、最终计数和 facet 统计均与原文件完整部分一致。
  原文件保留原始 791.5119 秒及各事件时间，仅用验证一致的复跑结构恢复损坏部分，
  并写入恢复来源元数据。完整 JSON 与 286 个 anchor 数量复核通过。这不是加预算实验。

## 2026-09-09: v6 持久复访、奖励排序与自适应 Corrector

### 修改与实验约定

- 已知有效终端基础 reward=0.10；class/support 重复惩罚各封顶 0.25，使其单次
  reward 下界为 -0.40，高于 invalid 的 -1.50。invalid 不再领取新 support bonus。
  全局新类奖励仍为 10，默认已知集合为空，覆盖只计实际命中。
- 对每个在线 canonical facet 保存合法 witness，优先更少修正、其次更短路径。
  每四轮进行一次公平复访，从原 root 重放，在同一棵树中每轮最多固化一条边。
  不建立第二棵树，不启动并行搜索；回收仅选择没有历史 discovery credit 的边。
- Corrector 基础上限 4，硬上限 8。延长必须有有效且未在本路径重复的 facet 及
  未试/可生成 frontier；第三次回到同一 canonical facet 即停止。树选择也检查
  动态停止，已有 repair edge 不得绕过该约束。
- 低 rank 补全失败后，保持 R 并额外尝试一次 identity 细化，仍做完整几何验证。
  对称缓存仅运输已验证保留面、出口、partition 与生成元，不存储 reward/class 标签。
  本地生成与缓存获取交替；每个原始坐标 anchor 的本地尝试上限仍为 48。
- `search.py`、`facet_corrector.py`、新 `repair_cache.py` 及对应测试保持隔离。
  旧 serial subgroup interrupt 未修改。源码快照为 `source_v6.zip`。
- 测试 96 passed in 9.14s。真实 i200 短测 18.2205 秒、4 类；随后补上树选择
  的动态停止检查，再开始以下完整实验。完整实验之间不调整算法。
- 完整实验使用 seed=3222026、snapshot_interval=500、known_classes=空，除总预算
  外配置相同。分别从初始状态执行 i10000 和 i15000，核对非计时发现路径前缀。
  事先约定：若 i10000--i15000 有新类且未全覆盖，则执行 i20000；否则不追加。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 10000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i10000_v6.json
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v6.json
```

### i10000 已完成

- 10000 iterations，**481.6377 秒（8 分 1.638 秒）**，实际发现 **38/46**。
  相比 v5 的 791.5119 秒、20/46，时间减少 39.15%，且包含 v5 的全部实际发现。
- 缺少 `{3,4,10,11,18,21,26,36}`。class 1 在 i3216、156.768 秒实际命中，并非
  来自预设标记。最后发现 class 22：i7444、354.598 秒，尾部 2556 次/127.0395 秒。
- 38 条发现路径回放通过，99 次结构化外部补全分别验证通过；35 个 class 的最终
  命中动作是外部补全，2 个为 atlas 动作，1 个为在线 stabilizer 动作。
- class 9：i252、20.683 秒，从已发现 class 46 的 Stab(192) pattern 加入 32 点，
  对其完整 tight support 做 identity 划分下的通用修正，再加入一个外部点命中。
  共三个动作、一次 Corrector；没有读取 class 9 样例或配置专属 pattern。
- class 7/28/1/14/39 的发现分别用了 5/6/5/5/6 次修正。延长预算进入了实际成功
  路径，但这是联动修改的单 seed 实验，不能据此单独估计某一修改的因果贡献。
- 2500 次公平复访，177 条 witness 边逐步持久化，1693 次延长修正。9719 nodes，
  9999 次持久扩展、9558 条保留边；回收 156+234+51=441 条，满足 9558+441=9999。
- Corrector 持久节点 1114（v5 为 340）；tree-policy steps=31620，rollout steps=79160。
  exact 29959/37379=80.15%，invalid 7420；v5 exact 比例为 21.53%。
- 生成 3105 个有效方案，运输复用 6721 个方案、34235 个出口；外部补全执行 16634 次。
  868 次 identity 细化尝试中 675 次成功；数字是操作/方案计数，不是不同 class 数。
- 详细时间线和检查统计见 `v6_i10000_audit.json`；原始 JSON 不作文本补丁编辑。

### 后续预算对照

- i15000 完成：**739.4335 秒（12 分 19.433 秒），43/46**，缺 `{3,4,18}`。
  与 i10000 相比新增 `{10,11,21,26,36}`，总耗时增加 257.7958 秒。
- 相同配置、seed 下，前 10000 次的完整非计时发现路径及 Corrector 诊断一致；
  两轮全部发现路径、生成元不变性、外部补全几何和 edge accounting 均复核通过。
- 额外预算区间的发现：class 10 在 i11884，11 在 i11896，21 在 i11940，26 在
  i12772，36 在 i12992。最后仍有 2008 次无新发现，但后 5000 次确有五类增益。
- 满足事先约定的追加条件，因此启动独立的同配置 i20000；不是改变 seed 或策略。
  算法源码 SHA256 核对未变化。完整对照数据见 `v6_budget_comparison.json`。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 20000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i20000_v6.json
```

i20000 已完成，最终预算结论见下。

第三轮到 i15000 时，已将该快照与完整 i15000 结果比较：除总预算配置和计时字段
外，所有已报告字段完全一致，包括树统计、根边价值、知识库、anchor 明细及诊断。

第三轮首次尝试在 i15500 写快照时遇到 Windows `PermissionError/WinError 5`，
临时快照已完整生成，但替换目标文件失败并终止进程。该尝试耗时至少 768.8222 秒，
仅作中断记录，不当作完整 i20000 结果。保留文件为
`single_tree_group_game_v6_interrupted_i15000.json` 和
`single_tree_group_game_v6_interrupted_i15500.json`。

已只修改 I/O：替换失败做退避重试，周期快照持续失败时警告并继续搜索；最终写出
仍要求成功。增加两项回归测试，98 passed in 9.84s。搜索逻辑、参数和随机 seed
均未改变，补丁源码另存 `source_v6_io_retry.zip`。独立重跑 i20000 已完成，输出为
`single_tree_group_game_i20000_v6_retry.json`；其完成时间与中断尝试开销分别记录。

### 完整预算结论

| 总预算 | 实际覆盖 | 搜索秒数 | 相比上一预算新增 |
| ---: | ---: | ---: | --- |
| 10000 | 38/46 | 481.6377 | -- |
| 15000 | 43/46 | 739.4335 | 10,11,21,26,36 |
| 20000 | 43/46 | 1009.7553 | 无 |

- i15000 的增量有实际收益，符合追加 i20000 的预设条件；但 i15000--i20000
  没有增加覆盖，总耗时增加 270.3218 秒。因此本 seed 下不支持继续单纯提高总预算。
- 三轮仍未找到 `{3,4,18}`。所有已找到的 43 类在 i12992 之前出现；i20000 最后
  7008 次、375.9543 秒没有新发现。这个结果不是剩余 class 不可达的证明。
- 两组发现路径前缀比较均通过；全量路径回放、实际子群生成元不变性、120 次外部
  补全几何验证以及单树 edge accounting 全部通过。诊断前缀比较针对已记录的
  Corrector diagnostics（上限 2000 条），不代表记录了所有 rollout 的完整轨迹。
- i20000：19310 nodes，19994 次持久扩展，19347 条保留边，回收 161+348+138=647
  条，满足 19347+647=19994。5000 次公平复访、290 条 witness 边固化、3711 次
  延长修正；13 次 dead end，0 次 depth-limit 截断。
- 生成 4322 个有效方案，运输复用 13520 个方案、63085 个出口，实际外部补全
  33072 次。相比 i15000 仍进行了大量几何工作，却没有新 class 增益，说明本轮
  后期的主要问题不是“完全没有计算”，而是新增计算未产生覆盖提升。
- 修复 I/O 后源码 SHA256 再次核对未变化；最终回归测试 98 passed in 8.96s。
  三次成功搜索计时合计 2230.8265 秒；中断尝试另耗至少 768.8222 秒，不能把两者
  混作单次 i20000 的算法耗时。
- 建议后续方法对照先采用 i15000；本 seed 的 i13000 已足以包含这次实际找到的
  43 类，但未做多 seed 实验，不能保证该预算在其他 seed 下取得相同覆盖。
  下一步应优先改善局部出口覆盖，而非直接追加 i25000/i30000。

最终机器可读对照、全部 class 发现时间和验证统计：`v6_budget_comparison.json`。

## v7：共享系统 ridge 枚举，固定 i15000 对照

后续方法对照固定为 i15000。本轮只修改隔离的 `group_game`，不修改旧串行
`subgroup_interrupt_search.py`。源码快照为 `source_v7.zip`；运行结束逐字节检查
所有算法 Python 文件与启动时快照一致。

### 修改与配置

- 新增 `ridge_enumeration.py`：对在线验证 facet 的仿射依赖空间作有界、可恢复
  的 ridge 候选枚举。候选总量 `C(|T|,25) <= 4096` 才启用，目前对应 26--28 个
  顶点；每批最多检查 32 个补集。这里按计算量筛选，不按 class 编号筛选。
- canonical facet 共享枚举进度，群运输回当前坐标；保留面的真实 rank 必须为
  24，每个外部 singleton 补全都必须通过 supporting-facet 验证。
- 每四次请求尝试一次，随机 LP 本地预算耗尽后仍可推进枚举；复用原有计划缓存、
  动作选择与回传，不增加第二棵树、不将 look-ahead 出口直接计入发现。
- 新参数 `ridge_max_candidates=4096`、`ridge_batch_size=32`。其余共有配置经
  机器比较完全相同：seed=3222026，known_classes 为空，Corrector 基础/扩展
  上限 4/8，rank 下限 17，复访周期 4，snapshot_interval=500。
- `--ridge-max-candidates 0` 可以关闭本轮新增通道。默认 coverage 仍只统计实际
  终端命中，class 1 的 singleton 起始 pattern 不等于已经发现 class 1。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v7.json
```

### 完整结果

| 方法 | 预算 | 实际覆盖 | 搜索秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v6 基线 | 15000 | 43/46 | 739.4335 | 3,4,18 |
| v7 共享 ridge | 15000 | 45/46 | 749.0155 | 3 |

- v7 完整执行 15000 次，耗时 **12 分 29.016 秒**。增加 9.5821 秒（1.30%），
  新增 class 4、18，没有丢失 v6 的任何 class。单 seed 单次计时不能解释为稳定的
  1.30% 性能差异；计时口径与 v6 相同，为搜索内部计时，不含初始化及最终写出。
- class 4：**i11441，558.9031 秒**。从已验证 class 45 的 32 点 facet 出发，
  identity 修复删除 6 点，保留 26 点、rank 24，再补入一个外部点命中 class 4。
- class 18：**i11960，587.4734 秒**。沿上述合法 witness 到 class 4，再删除两点，
  保留 25 点、rank 24，补入一个外部点，实际命中 26 点的 class 18。
- class 1 在 i5404、258.3786 秒实际命中。最后发现为 class 18，尾部仍有
  3040 次、161.5422 秒没有新发现。没有额外追加 i20000。
- i10000 时为 42 类（v6 为 38）；i12000 为 45 类（v6 为 41）。早期并非全程
  占优，例如 i4000 时 v7 为 28、v6 为 37。完整逐千步比较在机器可读文件中。

### 验证与工作量

- 新增 simplex 全部 26 个 ridge、退化 cube、独立 ConvexHull 对照、组合数上限、
  参数校验测试；另验证对称朝向共享进度，且 LP 预算为零时仍可推进系统枚举。
  最终回归 **104 passed in 9.48s**。i200 smoke 另耗 19.4477 秒，不混入正式计时。
- 全部 45 条发现路径回放、实际生成元不变性、173 个外部补全步骤的几何验证、
  credit 归一化及单树 edge accounting 均通过。持久树 14562 个节点。
- 新通道执行 298 批，检查 1189 个补集，发布 112 个有效计划、1860 个验证通过
  的出口；几何验证 4150 次，rank 后验拒绝 120 次，已有计划跳过 58 次。
  这些是计划/动作数量，不是不同 class 数，也不是全部真实 ridge 的数量。
- 总计生成 3919 个计划，运输复用 9880 个计划，执行 26759 次结构化外部补全；
  3750 次公平复访，227 条 witness 边持久化，4471 次延长修正。

### 解释与下一步

- 新增机制改变了后续随机调用、树路径和学习价值，不能把两个新 class 都直接
  归因为“新增枚举器产生了最后一个动作”。class 4 的直接来源是 32 点的 class 45，
  不满足新增枚举门槛，仍依赖保留的 LP/缓存修复通道。本轮只支持整体方法对照。
- 14 个在线 facet 满足系统枚举门槛，但全部尚未完成。三个 simplex 的进度为
  21/26、18/26、18/26；这些对应类别是运行后依据终端结果标注，不进入搜索生成。
  其余候选进度也完整记录，不能将本次称为已穷尽全部局部出口。
- 下一步优先提高现有 frontier 的兑现率：让重复/后验 rank 不合格候选在同一
  有界批次内继续推进，并给未完成枚举及未使用的已验证出口稳定的轮转机会。
  仍应遵守单树渐进扩展、真实几何校验和无未发现 class 样例的约束，不针对 class 3
  指定 pattern 或修改 reward。先继续固定 i15000 对照，不直接扩大总预算。

原始结果：`single_tree_group_game_i15000_v7.json`。
完整逐类发现时间、增益/遗漏、路径审计和枚举进度：`v6_v7_i15000_comparison.json`。
对照脚本：`scripts/compare_group_game_versions.py`。

## v8：有界批次推进与未用出口复访，i15000

本轮继续隔离修改 `group_game`，旧串行 subgroup 方法和 v7 原始结果保持不变。
启动时源码/测试快照为 `source_v8.zip`，正式运行结束确认所有算法 Python 文件
与快照逐字节一致。

### 修改

- 枚举请求在总计 32 个组合的限额内跳过重复、rank 不合格及无出口候选，找到
  一个有效计划才返回；仍为每次最多发布一个计划，不会一次展开全部候选。
- Gale 子矩阵使用绝对秩容差，过滤理论零行的舍入噪声。保留独立的真实 rank 和
  完整 supporting-facet 验证，未放宽几何正确性要求。
- canonical 几何缓存维护未使用的 `(retained, added)` 集合；候选规划不消耗，
  只有实际执行补全才移除。同一保留面/partition 的运输计划合并已验证出口。
- 原有每 4 次 simulation 一次的复访频率不变；默认其中一半公平轮转到尚有
  枚举工作或未用出口的在线 facet，推进一次有界枚举，按计划轮转选未用出口，
  把 Corrector/Filler 两步接在原 witness 后。另一半保持原复访策略。
- 所有动作仍从同一个 root 执行，规划时不查 class、不计发现、不回传 reward。
  新增步骤仍受动态 Corrector 停止和路径深度限制，每次最多一条持久扩展。
- 新参数 `frontier_replay_stride=2`；0 关闭新增复访通道，但不撤销批次和数值修改。
  两版全部共有配置机器核对相同，包括 seed=3222026、空 known_classes、修复
  4/8 上限、rank 下限 17、i15000 总预算及 500 次快照间隔。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v8.json
```

### 完整对照

| 方法 | 完成迭代 | 实际覆盖 | 搜索秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v7 | 15000 | 45/46 | 749.0155 | 3 |
| v8 | 15000 | 44/46 | 710.4203 | 3,24 |

- v8 耗时 **11 分 50.420 秒**，减少 38.5952 秒（5.15%），但未找到 v7 的 class 24，
  没有新增 class。因此这是枚举推进效率改善、最终覆盖下降的实验，不能用速度
  改善掩盖覆盖回退；v7 仍保留为覆盖基线。单 seed 不能分离各项修改的因果贡献。
- class 18：i3408、148.7920 秒；class 4：i3424、149.6732 秒，均比 v7 更早。
  class 1：i6976、315.6012 秒，仍为实际终端命中，而非默认计入。
- 最后发现 class 36：i10250、478.6427 秒；尾部 4750 次、231.7776 秒无新发现。
  v7 的对应尾部为 3040 次、161.5422 秒，新版尾部没有改善。
- i4000 覆盖 35 对 28，i10000 为 43 对 42，i11000 为 44 对 42；但 v7 在
  i12000 达到 45 类，v8 后期未再增加。不得仅凭中间检查点认定新版更优。

### 推进与验证

- 542 个系统枚举批次检查 8392 个组合（v7 为 1189），生成 342 个计划、5203 个
  验证通过的出口；真实几何检查 12640 次。后验 rank 拒绝为 0（v7 为 120），
  已有计划跳过 50 次；数字是工作量，不是不同 class 数。
- 13 个适用在线 facet 中，7 个完成系统枚举（v7：14 个适用、0 个完成）；三个
  simplex 的全部 26 个候选均完成。其余未完成进度完整记录，不能称为全搜索穷尽。
- 3750 次总复访中，1875 次使用新增通道，1873 次成功规划修复/补全，1873 对
  全部实际执行；2 次没有可用出口后回到原 witness。全程未使用 class 特定样例。
- 计划缓存最后仍有 11013 个未用 `(R, exit)`，它们不是 11013 个不同邻接 facet。
  仅用这个计数衡量剩余探索价值有局限，不能据此直接决定继续追加全局预算。
- 总计 LP 尝试 6237（v7：6773），生成 3997 个计划，运输 10947 个计划，实际
  结构化补全 28008 次。新增复访将 2044 条 witness 边持久化，v7 为 227 条；
  这是树内预算分配的明显改变，不是纯粹的缓存加速。
- 持久树 14498 个节点，14999 次扩展，14400 条保留边；回收 241+264+94=599
  条，满足 14400+599=14999。完整路径、125 个外部补全的几何验证、生成元
  不变性、credit 归一化、规划/实际执行计数和单树边数审计全部通过。
- 最终回归 **108 passed in 9.80s**，含批次继续推进、零 Gale 行、规划不提前发现
  或消耗出口、实际执行记账及动态停止测试。前置 i200 smoke 另耗 15.7288 秒，
  不计入正式 i15000 时间。本轮未追加其他总预算实验。

### 后续方向

保留 v7 覆盖基线及本次独立结果。下一步更适合先对补全得到的完整 canonical facet
记录/去重，检查不同 `(R, added)` 是否重复落入同一几何终端，再减少这类重复出口
占用的轮转预算；同时将数值/批次修正与强制出口复访拆开做消融。这里尚未证明
重复几何出口就是 class 24 遗漏的原因，也不应为 class 3 或 24 添加专属策略。

原始结果：`single_tree_group_game_i15000_v8.json`。
完整时间线、覆盖检查点、枚举进度和审计：`v7_v8_i15000_comparison.json`。

## v9：完整终点去重与消融，固定 i15000

继续在 `group_game` 内隔离实现，不修改旧 subgroup 串行方法。新增
`endpoint_graph.py`，两轮串行测试均使用 seed=3222026、空 known_classes、
修复 4/8 上限、rank 下限 17 和 500 次快照间隔。源码快照 `source_v9.zip`，
两轮结束确认全部算法 Python 文件与启动快照一致。

### 修改和边界

- 对通过验证的补全，用 normal/offset 计算完整 tight support，再用 Bell 群
  计算 canonical support。缓存不加载 class 标签，不把 look-ahead 当实际发现，
  也不将未实际走到的终点加入在线 pattern 库。
- 强制复访按 `(canonical source, canonical target)` 去重。相同源/终点关系
  实际执行后，不再为其他等价补全分配强制复访；源自身的等价终点也不占该名额。
  不同源到同一终点仍是不同关系，规划时不消耗关系。
- 普通 MCTS、随机探索和 LP 修复保留全部原有合法动作，只改变强制复访调度。
  同一 canonical facet 不一定代表相同有限预算搜索机会，所以本轮不是无损压缩
  的证明，仍需要用覆盖结果检验。
- 只有已成功发布的计划进入待执行终点图，避免废弃/重复计划留下无法执行的
  任务。终点映射随共享几何缓存跨坐标复用，不重复分类，不修改 reward。
- 新参数 `--no-endpoint-dedup` 可关闭此机制。第一轮同时关闭强制复访，保留
  v8 数值修正和有界批次，用来检查仅保留这些修改的表现；第二轮启用完整 v9。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --frontier-replay-stride 0 --no-endpoint-dedup --output src\mcts\group_game\runs\single_tree_group_game_i15000_v9_batch_only.json
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v9.json
```

### 完整结果

| 版本/实验 | 完成迭代 | 实际覆盖 | 搜索秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v7 覆盖基线 | 15000 | 45/46 | 749.0155 | 3 |
| v8 原始出口复访 | 15000 | 44/46 | 710.4203 | 3,24 |
| 仅数值/批次，无强制复访 | 15000 | 44/46 | 764.0268 | 3,24 |
| v9 完整终点去重 | 15000 | 43/46 | 764.8700 | 3,24,34 |

- 第一轮耗时 12 分 44.027 秒；最后在 i12132、609.3831 秒发现 class 13，
  尾部 2868 次、154.6437 秒没有新发现。仅关闭强制复访并未恢复 v7 的覆盖。
- 第二轮耗时 12 分 44.870 秒；在 **i2584、128.1338 秒已找到最终全部 43 类**，
  后面 12416 次、636.7363 秒没有新增。class 1 在 i2128、106.3949 秒实际命中。
- 完整 v9 比 v8 多耗 54.4497 秒（7.66%），还少 class 34；比本轮消融多耗
  0.8432 秒，仍少 class 34。早期发现快不代表最终覆盖好，v7 继续保留为覆盖基线。
  不对这些单 seed、单次计时作稳定性或独立因果贡献保证。
- 两轮成功搜索合计 1528.8969 秒；前置 i200 smoke 另为 19.7106 秒，不混入正式
  计时。本轮没有追加其他预算或第三轮搜索。

### 几何和调度诊断

- 完整 v9 的 18308 个已发布动作 key，只对应 357 种独立源/终点关系；实际补全
  26890 次，其中重复关系 26533 次（98.67%），源自身等价跳转 10883 次。
  该重复率包含普通 MCTS，不能拿它直接估计强制复访去重的加速比例。
- 357 种已发布关系全部执行，待执行非自身关系为 0。终点图中的 43 个 canonical
  终点与本轮实际发现集合逐一相符，不只是数量相同；没有已发布但未走到的新终点。
- 3750 次总复访中，740 次进入新通道：154 对修复/补全被规划并全部执行；586 次
  推进枚举但没有新的独立终点，回到原 witness。其余复访按原策略处理。
- 系统枚举检查 18213 个组合，生成 392 个计划、6087 个出口；13 个适用 facet
  全部完成（最晚在 i8500 快照已完成）。只覆盖候选量门槛内的 26--28 点 facet，
  不代表完整 Corrector 空间穷尽。消融只完成 1 个枚举器、检查 3444 个组合。
- 175 个发现路径外部补全步骤、全部 43 条发现路径、生成元不变性、credit、
  终点映射与执行计数均审计通过。14996 次扩展，14675 条保留边，回收
  106+196+19=321 条，满足 14675+321=14996；持久节点 14688。
- 最终回归 **111 passed in 9.56s**，新增同终点别名、不同源的独立性、规划不
  消耗、映射冲突拒绝、完整几何及群运输一致性测试。

### 解释限制与后续方向

- 终点图只统计成功发布的候选。底层通过几何验证、却被 `duplicate_plan` 拒绝
  的候选没有完整终点报告，本轮计数为 645。因此不能由“图中没有未发现终点”
  推断所有底层 look-ahead 都从未生成过缺失类。
- 代码检查发现：本地 `_publish_plan` 对重复 `(R, pattern)` 直接返回，而运输
  `_import_plan` 会合并新增出口。下一步应先审计并合并本地重复计划中的新增
  有效出口，防止验证工作在发布阶段丢失；这尚未被证明是遗漏的具体原因。
- 旧 v7 中 class 24、34 的成功路径均有 8 次修复，其最后一次修复的源 facet
  分别有 29、34 个顶点，超出本轮系统枚举范围。之后可以面向所有较大 facet，
  利用已观测 stabilizer 的 quotient/分块结构生成更多保留面，而不针对任何
  class 编号设计 pattern。当前结果不支持继续单纯扩大总预算或加强终点去重。

完整机器可读矩阵：`v9_i15000_matrix.json`。
分项对照：`v7_v9_batch_only_i15000_comparison.json`、`v8_v9_i15000_comparison.json`、
`v9_ablation_i15000_comparison.json`。原始结果与源码快照均保留。

## v10：本地出口无损合并与缓存增量传播，固定 i15000

2026-09-10。继续仅修改隔离的 `group_game`，不改旧 subgroup 串行方法。
修订标识 `lossless_local_and_shared_exit_merge_v10`。seed=3222026、空
known_classes、修复上限 4/8、最低 rank 17、快照间隔 500；与 v9 的全部配置
相同，仅修正出口发布/读取链路并增加诊断。启动前冻结 `source_v10.zip`，运行
结束核对所有算法 Python 文件与快照一致。

### 修正内容

- `_publish_plan` 对相同 `(retained, pattern)` 合并新增的已验证出口，同时更新
  本地计划、Filler 出口表、canonical 共享缓存和终点图。完全重复的发布不改变
  状态，已执行出口不会重新变为未执行。
- `_borrow` 不再直接跳过已有本地计划，而是检查缓存版本是否包含新增出口；
  通过原有 `_import_plan` 将增量合并。共享缓存保持追加式，不破坏旧读取游标。
- `publication_audit` 记录已验证候选与已发布动作的差集；`local_merge_events`
  记录新增出口和迭代节点。仅缓存结构/几何，不缓存完整 reward，不提前分类、
  发现或兑现 novelty。没有使用未发现 class 的样例指导生成。
- 保持 `seen_faces` 去重、系统枚举候选上限和随机 LP 尝试上限不变；本次修正
  不代表所有可能的保留面或外部补全都已枚举。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v10.json
```

### 完整结果

| 版本 | 迭代 | 实际覆盖 | 搜索秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v7 单树覆盖基线 | 15000 | 45/46 | 749.0155 | 3 |
| v9 | 15000 | 43/46 | 764.8700 | 3,24,34 |
| v10 | 15000 | 44/46 | 765.1971 | 3,36 |

- 正式运行耗时 **12 分 45.197 秒**，相较 v9 增加 0.3271 秒（0.043%），视为
  基本相同；找回 class 24、34，但漏掉 36，净增一类。相较 v7 仍少 class 36，
  慢 16.1816 秒（2.16%），尚未超过单树覆盖基线。单 seed 不保证跨种子稳定提升。
- class 18：i896、50.6291 秒；class 1：i2176、108.9475 秒，均为实际终端
  命中，class 1 不默认计入结果。
- class 34：i8176、389.6651 秒；最后一个 class 24：i11316、562.1263 秒。
  此后 3684 次迭代、203.0708 秒没有新发现。
- i3000 覆盖 39 对 v9 的 43；i9000 为 43 对 43，i12000 才为 44 对 43。
  修正没有同时改善早期覆盖；不能只引用最终净增一类来描述所有预算区间。
- 前置 i200 smoke 为 18.7166 秒、4 类，单独记录，不混入正式运行时间。

### 发布、几何与路径审计

- 本地重复发布 653 次：607 次完全重复，46 次合并共 46 个新增出口；缓存运输
  另合并 1015 个本地缺失出口。后者包含不同坐标副本，不是 1015 个独立终点。
- 18962 个已验证 canonical 动作 key 全部发布，未发布差集为 **0**。
  46 条合并事件逐一验证 retained 属于原 facet、包含外部新增点、补全为有效
  supporting facet；事件数量未触及 2000 条日志上限。
- 事后用本轮实际发现的 support 标记合并终点：46 个本地新增出口仅落到
  class 23、42、43、44，全部早于合并至少 280 次迭代已经发现。标签只用于
  离线分析。因此不能声称此次合并直接挽救了未发现类别的终点；缓存增量也改变
  了后续动作可见性和随机轨迹，尚未分离各项贡献。
- 已发布动作对应 368 种独立源/终点关系，全部实际执行，待执行非自身关系 0。
  图中的 44 个 canonical 终点集合与本轮实际发现集合逐项相等；没有已验证、
  已发布却未实际走到的额外几何终点。
- 外部补全执行 27415 次，其中重复源/终点关系 27047 次（98.66%），自身等价
  跳转 10617 次。重复率包括普通 MCTS，不能直接换算成强制复访的可节省时间。
- 14 个适用的在线 facet 系统枚举全部完成：997 个批次、21489 个候选，403 个
  计划、6227 个出口。仅覆盖当前 26--28 点/组合上限范围，不是全搜索空间穷尽。
- 3750 次复访中，821 次进入 frontier 通道；157 对修复/补全被规划并实际执行，
  664 次没有独立终点而返回原 witness。全部 44 条发现路径、163 个外部补全步骤、
  生成元不变性及 novelty credit 审计通过。
- 持久树 14654 个节点，14998 次扩展、14596 条保留边，回收 132+231+39=402，
  满足 14596+402=14998，不超过总迭代数。最终回归 **113 passed in 10.30s**，
  包含开启/关闭终点去重时的本地合并、远端已有计划增量接收、幂等发布及旧出口
  不重新入队测试。

### 后续方向

本次修复解决真实的发布一致性问题，但没有证明它是覆盖不足的主因。当前几何
候选终点均已实际访问，优先方向应是通用地增加候选保留面的多样性，而非继续
提高相同出口的复访频率或单纯增加全局预算。

v10 的 class 34 路径包含从 32 点、34 点源 facet 的 rank-24 修复；class 24
路径包含从 32 点源 facet 保留 rank-21 面，再从 29 点源 facet 做 rank-24
修复。这些源超出小 facet 系统枚举范围。后续可面向所有已发现大 facet，利用
在线 stabilizer 的 orbit/quotient 结构生成可续跑的保留面候选，并为低 rank
保留面保留多次补全机会；不能根据缺失 class 编号定制 pattern 或目标样例。
这里是下一步建议，本轮未实施额外策略，也未追加其他总预算实验。

原始结果：`single_tree_group_game_i15000_v10.json`。
完整时间线、覆盖检查点、合并终点事后标记和路径审计：
`v9_v10_i15000_comparison.json`、`v7_v10_i15000_comparison.json`。

## v11：低 Rank 保留面有界续跑，固定 i15000

2026-09-10。本轮先隔离检验低 rank 面的多次补全，没有同时扩大大 facet 的
保留面生成范围。新增 `completion_frontier.py`，修订标识
`bounded_shared_lower_face_continuation_v11`。仍为一棵 MCTS，不增加独立树或
并行调度，不修改旧 subgroup 串行方法。源码冻结于 `source_v11.zip`，完整运行
结束确认所有算法 Python 文件与启动快照相同。

### 实现和边界

- 旧逻辑对 rank <24 的保留面通常只尝试两次受约束 LP；即使没有有效补全，
  之后再次生成该面也会被 `seen_faces` 跳过。本轮不删除该去重，而是将已生成
  低 rank 面的补全工作单独登记，允许有界续跑。
- 按共享缓存坐标的 `(canonical source, retained, embedded pattern)` 共享
  最多 6 次 LP，包括初次尝试；不同坐标副本不能重置预算。这里没有进一步对
  source/retained/pattern 的整个 stabilizer orbit 做最小化，不宣称完全去重。
- 每个续跑批次最多 2 次 LP，源内部 round-robin 轮换，额度用完移出队列。
  普通 `grow` 每四次请求给予一次续跑机会，随机面生成额度耗尽后也可续跑；
  原有 frontier 复访在小 facet 系统枚举没有产出时推进补全队列。
- 首次补全仍使用原 anchor RNG；续跑使用由 seed 和 canonical 几何 key 派生
  的独立 RNG。保留实际子群不变空间、原 rank 下限及 supporting facet 验证。
  有效出口通过 v10 的本地合并和共享缓存运输，不提前分类、奖励或计为发现。
- `lower_face_frontier.jobs` 记录共享 key、LP 次数、续跑批次、有效出口 key 数
  和是否用完预算；`completion_continuation_*` 记录续跑工作量。不能把一个
  几何出口当成一个新 class，也不能把额度耗尽当成保留面已经穷举完毕。
- 新配置 `lower_face_max_attempts=6`，设为 0 可关闭。其余与 v10 完全相同：
  i15000、seed=3222026、空 known_classes、修复 4/8、最低 rank 17、快照 500。
  不载入未发现 class 的样例，不为任何缺失编号添加专用策略。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v11.json
```

### 结果与时间线

| 版本 | 迭代 | 实际覆盖 | 搜索秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v7 单树覆盖基线 | 15000 | 45/46 | 749.0155 | 3 |
| v10 | 15000 | 44/46 | 765.1971 | 3,36 |
| v11 | 15000 | 45/46 | 698.7079 | 3 |

- 正式运行耗时 **11 分 38.708 秒**。相较 v10 多 class 36、没有丢失其他类别，
  减少 66.4892 秒（8.69%）；与 v7 覆盖集合完全相同，减少 50.3076 秒（6.72%）。
  这是同 seed 单次运行的结果，未验证多 seed 稳定性。
- class 34：i1984、99.2772 秒；class 36：i2960、143.4552 秒；class 18：
  i3328、159.2903 秒。class 6：i5824、273.8427 秒；class 1：i5872、275.7208 秒。
  class 1 是实际终端命中，不是默认已发现信息。
- class 33：i8376、389.9464 秒；最后 class 24：**i9696、452.9887 秒**。
  达到 45 类比 v7 的 i11960、587.4734 秒提前 2264 次迭代、134.4847 秒。
  这是同一长运行的发现时间，不是另一次较小总预算实验的计时。
- 此后 5304 次迭代、245.7192 秒没有新增。无新发现尾部仍存在，甚至长于
  v10 的 203.0708 秒；不能将总体速度改善解释为已解决长尾。
- 早期并非全面改善：i2000 为 27 对 v10 的 34，i4000 为 37 对 40，i5000
  才达到同样的 40 类；i6000 为 42 对 40，i10000 为 45 对 43。

### 几何工作量与审计

- 共登记 1234 个 canonical 补全任务，1233 个耗尽 6 次预算，最后一个尝试 5 次。
  总 LP 7403 次，等于逐任务计数之和；3013 个续跑批次执行其中的 5478 次 LP。
  所有 key 唯一、保留面属于源 facet、rank 在 17--23、单 key 不超过 6 次的
  审计均通过。未完成任务的几何 key 完整保存在原始 JSON。
- 613 个任务没有得到有效外部出口，500 个得到多个不同出口 key；这些数字
  不是不同 class 数。续跑触发 708 次新增/合并计划发布，本地总计合并 612 个
  出口，缓存运输另合并 2892 个本地缺失出口。
- 19296 个已验证候选动作全部发布，未发布差集为 0；终点图的 45 个 canonical
  终点集合与实际发现集合逐一相等，没有已经验证却未兑现的新终点。
- 373 种独立源/终点关系全部执行，待执行非自身关系为 0。外部补全执行 27759
  次，其中重复关系 27386 次，自身等价跳转 11205 次，重复工作仍然很多。
- 14 个适用小 facet 的系统枚举完成 11 个，总计 20176 个候选，产生 398 个
  计划、6139 个出口。其余三个进度为 2590/3276、3157/3276、2768/3276，不能
  称为枚举完成。续跑也会改变原 frontier 内部的预算分配。
- 3750 次复访中，1855 次进入 frontier，155 对修复/补全被规划并实际执行，
  1700 次没有独立终点。全部 45 条发现路径、151 个外部补全步骤、生成元
  不变性及 novelty credit 审计通过。路径中的保留面 rank 分布为 24:139、
  23:11、19:1，比 v10 的低 rank 步骤更多，但不构成单独的因果证明。
- class 36 的最后一次结构化修复从 36 点源 facet 保留 28 点、rank 23 的面，
  再补全到新终端。源终端事后分类为 class 39；这一标签未用于选择保留面。
  本轮没有逐计划的初次/续跑来源标记，不能断言该特定出口就是续跑首次生成。
- 持久树 14333 个节点、15000 次扩展、14296 条保留边，回收 208+325+171=704，
  满足 14296+704=15000。仍然每次 simulation 最多新增一条持久边。
- 相较 v10，低 rank LP 从 3723 增至 7403，但保留面 LP 从 7004 降至 6592，
  几何出口检查从 66536 降至 63958，rollout 步数从 120063 降至 111298。
  耗时改善伴随轨迹/工作量变化，不能解释为 LP 内核加速，也未做分项 profiler
  来确定各部分节省的独立贡献。

### 回归和后续方向

- 最终 **118 passed in 9.63s**，新增预算共享、有限轮转、独立 RNG、跨坐标
  增量发布、有效几何、规划不消耗出口和关闭开关测试。
- 前置 i200 smoke：16.8502 秒、2 类；关闭机制的 i200：17.8344 秒、4 类。
  关闭后发现路径/迭代、完整树统计及几何统计均与 v10 i200 一致，且没有补全
  任务，核对保存于 `v11_disabled_smoke_verification.json`。两次 smoke 独立
  计时，不混入正式 i15000。本轮仅进行一轮完整 i15000。
- v11 可作为下一轮实验对照，v7/v10 的结果与源码快照仍保留。下一步优先考虑
  本轮尚未实现的“大 facet 在线 stabilizer/orbit 引导的新保留面生成”，以及
  根据实际几何出口收益分配续跑名额，不应为 class 3 加专用样例或策略。
  大量任务没有外部出口、后期已有终点全部执行，当前证据不支持盲目增加相同
  LP 的次数或全局总预算；也不能由有限失败次数证明没有可达的新 class。

原始结果：`single_tree_group_game_i15000_v11.json`。
完整时间线、检查点、任务审计及路径对照：`v10_v11_i15000_comparison.json`、
`v7_v11_i15000_comparison.json`。新增代码与测试包含于 `source_v11.zip`。

## v12：大 Facet 对称 Orbit 删除候选，固定 i15000

2026-09-10。新增 `orbit_faces.py`，修订标识
`bounded_symmetry_orbit_face_frontier_v12`。继续隔离在 `group_game`，不修改旧
subgroup 串行方法。冻结源码 `source_v12.zip`，运行结束所有算法文件与启动快照
一致。保持 v11 的 seed=3222026、空 known_classes、rank 下限 17、修复上限 4/8、
低 rank 补全每 key 6 次、快照间隔 500；新增 orbit 候选 256/源、8/批。

### 候选生成与几何边界

- 对不适用现有小 facet ridge 枚举的已观测 canonical 源，创建一个共享候选流。
  当前默认参数下对应大于 28 点的 facet；不同对称坐标副本共享进度，不重置额度。
- 仅使用该实际源 facet 的 identity、稳定子与已采样 cyclic 子群划分，轮转枚举
  删除 1--3 个完整 orbit。每种 pattern 的 orbit 顺序由独立几何 seed 打乱，
  不构造所有组合列表，不依据 class 编号或未发现样例选块。
- 候选首先检查保留面 rank 在 17--24，再用源 facet 的 Gale 依赖检查存在一个
  在保留点上为零、在所有删除 orbit 上严格为正的 affine slack。多维核使用
  小型可行性 LP，不能仅凭 rank 就将任意子集视为支撑面。
- 通过检查的 rank-24 面调用原外部 orbit 枚举，低 rank 面调用 v11 的有限 LP
  补全及续跑。最终 supporting facet 验证、动态 reward、实际终端发现、v10
  合并发布/运输逻辑不变，不提前把 look-ahead 计入发现。
- 原随机面生成和小 facet 枚举保留，新通道不会把面写入旧 `seen_faces`；同一
  保留面在其他 pattern 下的旧探索机会不因该标记被提前删除。新 pattern 的
  运输和候选发布仍会改变有限预算内的搜索轨迹，不是无损加速证明。
- 普通生成沿用每四次请求的扩展机会，原 frontier 复访也可推进新通道；在该
  机会中，小 facet 枚举、新 orbit 面、已有低 rank 补全按顺序尝试。这个优先级
  会改变各通道的实际工作量，是本轮需要诊断的预算分配变化。
- `orbit_face_frontiers` 保存候选数、rank/支撑面拒绝原因、已接受面及 pattern
  的源内 blocks；`orbit_face_publication_events` 保存发布迭代、源、保留面、
  rank、pattern 和出口。`finished` 可能只是触及 256 次上限，`exhaustive` 仅
  表示这个有限 1--3 orbit 组合流全部检查，不代表全部子群/保留面已穷举。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v12.json
```

### 完整结果

| 版本 | 迭代 | 实际覆盖 | 总秒数 | 达到最终覆盖秒数 | 遗漏 |
| --- | ---: | ---: | ---: | ---: | --- |
| v7 | 15000 | 45/46 | 749.0155 | 587.4734 | 3 |
| v11 | 15000 | 45/46 | 698.7079 | 452.9887 | 3 |
| v12 | 15000 | 45/46 | 681.4297 | 342.4655 | 3 |

- 正式运行 **11 分 21.430 秒**，与 v11 覆盖集合完全相同，总耗时减少 17.2782
  秒（2.47%）；较 v7 减少 67.5859 秒（9.02%）。仅为单 seed、单次计时结果，
  未证明多 seed 稳定改善，也没有找到 class 3。
- 最后 class 13：**i7304、342.4655 秒**；达到 45 类较 v11 提前 2392 次迭代、
  110.5232 秒（24.40%）。此后 7696 次、338.9642 秒没有新增，比 v11 的无新增
  尾部多 93.2449 秒。最后发现时间不等于本轮完整运行时间。
- class 36：i2592、122.2478 秒；class 24：i3360、157.5659 秒；class 1：
  i5892、275.6837 秒；class 6：i5956、278.6135 秒。class 1 仍为实际命中，
  不是默认已发现信息。
- 早期覆盖下降明显：i2000 为 19 对 v11 的 27，i4000 为 25 对 37；i6000
  才达到 43 对 42，i8000 为 45 对 43。不能把本轮概括为所有预算下都更好。

### 候选收益与积压

- 31 个大 facet 候选流共检查 7250 个组合，23 个触及额度，8 个仍未完成；
  没有任何一个流被完整穷举。接受 1445 个支撑面，拒绝原因分别为 rank 过低
  116、rank 过高 2769、不是支撑面 2920，合计正好 7250。支撑性 LP 1217 次。
- 接受面的 rank 分布：17:32、18:29、19:85、20:41、21:123、22:248、23:467、
  24:420；其中 1025 个低于 rank 24，说明新通道确实扩大了低 rank 面的候选来源。
  不同 pattern 下的同一面仍可能重复，1445 不是互不等价面或 class 的数量。
- 新通道发布 1243 次计划，发布事件均完整记录，未触及 2000 条日志上限。
  事后只用本轮实际发现的 support 标记其终点，没有未知终点；15 个类别的首次
  发现迭代存在指向该类的 orbit 计划发布。同迭代内的先后没有单独时间戳，
  不据此宣称严格因果归因，也不把离线标签用于在线搜索。
- 最后发现之后仍有 527 次 orbit 计划发布，涉及 572 条事件内去重的目标记录，
  全部指向此前已发现的类别。增加有效几何计划并未突破后期覆盖。
- 低 rank 补全任务从 v11 的 1234 增至 2404，仅 1005 个用完额度，留下
  **1399 个未完成任务、4890 次剩余 LP 名额**。共执行 9534 次补全 LP，其中
  5148 次来自 2737 个续跑批次；1001 个任务尚无有效外部出口，715 个有多个。
- 小 facet 系统枚举完成 8/14（v11 为 11/14），候选检查从 20176 降至 11493。
  这表明新候选生成增加了后续工作，并改变/挤占了其他通道预算；不能把现有
  积压当成已经尝试失败的证据，也不能直接断言追加预算一定找到缺失类。

### 一致性与验证

- 21072 个已验证 canonical 动作全部发布，未发布差集 0；终点图的 45 个
  canonical 终点集合与本轮实际发现集合逐项相同。
- 已发布 409 种源/终点关系，408 种实际执行；唯一未执行关系是自身等价跳转，
  待执行非自身关系为 0。外部补全执行 28830 次，其中重复关系 28422 次
  （98.58%）、自身等价跳转 11105 次，重复工作仍然很多。
- 3750 次复访中 1875 次进入 frontier，205 对修复/补全被规划并实际执行；
  1670 次没有独立终点。候选发布与实际执行严格分开记账。
- 1445 个接受面全部用独立的原坐标 affine slack LP 再次验证支撑性，同时
  检查 rank、完整 orbit 保留、删除 orbit 数量、候选预算和计数守恒。1243 条
  发布记录的出口及全部 45 条发现路径、153 个外部补全步骤均通过几何审计。
- 发现路径中的修复保留 rank 为 24:124、23:14、22:9、17:6。只说明有效路径
  使用了更多不同 rank 的保留面，不证明某一 rank 本身是覆盖提升的原因。
- 持久树 14250 个节点，14996 次扩展、14289 条保留边，回收 215+365+127=707，
  满足 14289+707=14996。单棵树、每轮最多一条持久扩展的约束仍成立。
- 最终回归 **122 passed in 9.70s**，新增支撑性独立验证、分批不改变枚举序列、
  共享进度、群运输、预算上限、小 facet 跳过和关闭开关测试。
- 前置 i200 smoke：17.2086 秒；关闭 `--orbit-face-max-candidates 0` 的 i200：
  16.7343 秒。关闭后的发现路径/迭代、树统计、几何统计和低 rank 任务均与
  v11 i200 相同，保存于 `v12_disabled_smoke_verification.json`。两次 smoke
  独立计时，不混入正式 i15000；本轮只进行一轮完整 i15000。

### 下一步

保留 v11/v12 两个独立对照。优先为“小 facet 枚举 / 大 facet 候选生成 / 已有低
rank 补全”设置可比较的最低处理份额，并根据在线独立几何终点收益限制持续生成
低收益候选，避免新面生产快于补全处理。改进依据应是已验证几何、实际新发现和
队列积压，而不是缺失 class 编号或样例。当前结果支持进一步调整预算分配，但
不支持把更多候选、更多总 iteration 或这次 2.47% 的计时差当作找全保证。

原始结果：`single_tree_group_game_i15000_v12.json`；源码：`source_v12.zip`。
完整时间线、覆盖检查点、候选/出口审计及发布终点离线分析：
`v11_v12_i15000_comparison.json`、`v7_v12_i15000_comparison.json`。

## v13：通道公平调度与积压反馈，固定 i15000

2026-09-10。新增 `frontier_scheduler.py`，修订标识
`fair_backpressure_frontier_scheduling_v13`。继续隔离在 `group_game`，不改旧
subgroup 串行算法。源码 `source_v13.zip` 在运行前冻结，结束后算法文件与快照
一致。本轮正式配置仍为 i15000、seed=3222026、空 known_classes、修复上限 4/8、
最低 rank 17、每补全任务 6 次 LP、orbit 256/源和 8/批、快照 500。

### 修改内容

- 强制复访不再先按 facet 总次数选源，而是先轮转选择工作类型，再按该类型的
  已服务次数公平选 canonical facet。普通 `grow` 在每个 canonical 源内部轮转，
  对称坐标副本共享相位和计数。仍是一棵 MCTS，没有增加独立树或并行任务。
- 正常循环的 8 个机会：ridge 枚举 2、已有补全 3、orbit 生成 2、出口交付 1。
  当某个可用补全源至少积压 24 次 LP 时，改为 ridge 2、补全 4、orbit 1、交付 1。
  空通道跳过；orbit 至少保留一个槽位，积压不会将它永久关闭。
- 每次调度最多推进一个有界工作批次，不再在一次机会中依次尝试三种几何工作。
  借用缓存和随机保留面生成仍保留原触发条件。份额是机会数，不是 CPU 时间，
  也不是整个算法全部运算的硬配额；本轮同时改变了顺序和一次机会的工作上限。
- 积压直接计算 canonical 共享队列中尚未执行的 LP 次数，而非 plan 数或 class
  标签。调度只使用在线工作状态，不使用未发现案例；几何验证、终端发现和奖励
  规则保持原样。新关系工作量仅作为诊断，本轮没有按它在线学习调度权重。
- `frontier_scheduling` 保存选择次数、分源计数、积压模式和峰值；
  `scheduled_*_dispatches/new_pairs/plans` 分别记录实际调度、新几何关系与返回
  计划。选择不等于实际执行，已验证出口的交付也单独保留机会。
- 新配置 `fair_frontier=True`、`frontier_high_water=24`；传入
  `--no-fair-frontier` 可关闭本轮修改，复现 v12 行为。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game --iterations 15000 --snapshot-interval 500 --output src\mcts\group_game\runs\single_tree_group_game_i15000_v13.json
```

### 完整结果：负向实验

| 版本 | 迭代 | 实际覆盖 | 总秒数 | 遗漏 |
| --- | ---: | ---: | ---: | --- |
| v11 | 15000 | 45/46 | 698.7079 | 3 |
| v12 | 15000 | 45/46 | 681.4297 | 3 |
| v13 | 15000 | 44/46 | 722.0707 | 3,34 |

- 正式运行 **12 分 2.071 秒**。相较 v12 少 class 34，没有新增类别，慢
  40.6410 秒（5.96%）。本轮不能认定为搜索效果上的改进，v12 继续作为效果基线；
  当前代码保留 v13 实验，后续要运行 v12 路径需显式传 `--no-fair-frontier`。
- 最后发现 class 36：i8864、418.4588 秒；后续 6136 次、303.6119 秒没有新增。
  尾部虽然比 v12 短约 35 秒，但最终少一类，不能把它解释为更好的停止策略。
- class 6：i1705、85.9871 秒；class 1：i3072、146.2895 秒；class 24：
  i3556、167.8108 秒；class 13：i6660、313.7598 秒。class 1 依旧是实际终端
  命中，不是初始化时计入的已知信息。
- i2000 覆盖为 23 对 v12 的 19，i4000 为 36 对 25；但 i6000 已变为 40 对 43，
  i8000 为 43 对 45，最终为 44 对 45。早期领先没有转化为最终覆盖提升。

### 积压改善与探索代价

| 指标 | v12 | v13 |
| --- | ---: | ---: |
| 低 rank 任务总数 | 2404 | 1541 |
| 已耗尽预算的任务 | 1005 | 1262 |
| 未完成任务 | 1399 | 279 |
| 剩余 LP 名额 | 4890 | 848 |
| 全部补全 LP | 9534 | 8398 |
| 其中续跑 LP | 5148 | 5820 |
| orbit 候选检查 | 7250 | 3405 |
| orbit 计划发布 | 1243 | 484 |
| 小 facet 枚举检查 | 11493 | 11306 |
| 小 facet 枚举完成 | 8/14 | 8/14 |

- 未完成任务减少 80.06%、剩余 LP 名额减少 82.66%。这既来自更多任务完成，
  也来自新增任务减少，不能简单称为补全吞吐提高。小 facet 完成数没有增加。
- 实际强制复访选择：ridge 479、补全 1084、orbit 275、交付 37，合计 1875。
  其中 1855 次处于积压模式，占 **98.93%**；最大单源积压为 135 次 LP。
  因为采用“任一可用源超过阈值”，该模式近乎常态，而非只缓解短期突发积压。
- 普通 grow 的调度为 ridge 143、补全 2050、orbit 522；942 次处于积压模式。
  空通道跳过，所以实际全局比例不等于初始循环的名义比例。

| 通道 | 实际批次/调度 | 新源/终点关系 |
| --- | ---: | ---: |
| ridge | 622 | 94 |
| 已有补全 | 3134 | 19 |
| orbit | 797 | 64 |
| 已有出口交付 | 37 | 0 |

上述“新关系”是几何 source/target 的首次登记，不是新 class；各通道一批的成本
不同，不能直接当作每秒收益。交付本来就不生成关系，0 不是交付无用的证据。
但这些数据说明大量补全机会没有带来新的几何关系，而 orbit 工作减少明显；
“队列更长就分配更多预算”不是探索价值的可靠替代。这尚不能单独证明 class 34
遗漏的原因，也未分离通道配额、积压反馈和单批限制的各自因果贡献。

### 几何与调度审计

- 19221 个已验证 canonical 动作全部发布，未发布差集为 0。377 种源/终点
  关系全部实际执行，待执行非自身关系为 0；图中 44 个 canonical 终点与实际
  发现集合逐项相同，没有已经验证却未走到的额外类别终点。
- 外部补全实际执行 25364 次，其中重复关系 24987 次、自身等价跳转 11339 次。
  本轮未对不同通道做 wall-clock profiler，不能将耗时增加精确归因到某个操作。
- 大 facet 候选流 30 个，仅 5 个触及额度，没有穷举完成的流。3405 个组合中
  接受 582 个面；rank 过低 55、过高 1600、支撑检查未通过 1168，计数守恒。
  415 次支撑性 LP、484 条新通道发布记录及其出口均通过独立几何审计。
- 1541 个补全任务的尝试总和为 8398，单 key 不超过 6；续跑批次为 3134。
  673 个任务没有有效外部出口，620 个存在多个出口 key，不能将其当作新类数量。
- 3750 次总复访中，1875 次进入 frontier；150 对修复/补全被规划并实际执行，
  1725 次没有可用独立终点。各源选择计数与通道总计相符，grow 的选择/调度相符，
  replay 实际调度不超过选择次数，本次均通过审计。
- 全部 44 条发现路径、122 个外部补全步骤、生成元不变性和 credit 验证通过。
  发现路径中保留面 rank 为 24:113、22:2、21:7。
- 持久树 14456 个节点，14996 次扩展、14371 条保留边，回收 264+271+90=625，
  满足 14371+625=14996；每个 simulation 最多新增一条持久边的约束未改变。
- 最终回归 **128 passed in 9.59s**，新增正常/积压配额、通道内公平、空通道
  跳过、orbit 不饥饿、canonical 相位共享、单次只调度一个批次以及启用/关闭
  新调度时规划不提前发现/消耗出口的测试。
- 前置 i200 smoke：17.8097 秒、4 类；关闭新调度的 i200：16.9103 秒、2 类。
  关闭后的发现路径/迭代、树统计、几何统计、补全任务和 orbit 前沿均与 v12
  i200 一致，记录于 `v13_disabled_smoke_verification.json`。两次 smoke 单独
  计时，不混入正式 i15000。本轮只进行一轮完整 i15000。

### 后续方向

保留负向结果，不用更短的队列掩盖覆盖回退。下一步应从 v12 的效果基线出发，
将补全优先级与在线新增几何关系、实际全局新发现及测得的计算成本结合，而非
仅根据剩余名额；同时保留可验证的探索最低份额。当前阈值使全局几乎一直处于
积压模式，应检查按源反馈或更平滑的份额调整，而不是盲目进一步提高补全比例。
这些均为下一步建议，本轮未追加其他策略实验，也不使用缺失 class 的样例调参。

原始结果：`single_tree_group_game_i15000_v13.json`；源码：`source_v13.zip`。
完整对照、时间线、候选/出口及调度审计：`v12_v13_i15000_comparison.json`。
通道诊断：`v13_frontier_diagnostics.json`；关闭验证：`v13_disabled_smoke_verification.json`。

## v14: 稀疏几何收益 / 成本调度（i15000）

### 修改范围和假设

本轮从 v12 的局部工作顺序出发，隔离修改 `group_game`，未修改旧串行
`subgroup_interrupt_search.py`。不使用缺失 class 样例，也不按编号配置路径。
起始仍为 identity singleton，已知 class 集合为空，class 1 必须实际命中。

- 新增 `yield_frontier.py`。每 4 次强制 frontier replay 仅 1 次使用在线收益
  选择源/通道，其余继续 v12 轮转与 ridge/orbit/completion 顺序。
- 自适应选择每 4 次保留 1 次通道轮转探测；探测内按 canonical 源的自适应
  服务计数选择，空通道跳过。它不是所有源的 CPU 时间公平保证。
- 原有几何工作也参与计量，不增加额外 LP。收益为新 canonical 几何终点数
  加 0.1 倍新非自身源/终点关系数。已在图中或已观察 facet 池中的终点不再
  获得第一项奖励；别名计划和自身跳转无收益，不使用 class 标签。
- 按源/通道保存收益、实际耗时的 0.2 EMA，新源使用本次运行的通道统计。
  分数加入随观察/服务次数衰减的不确定性项，耗时分母最低 0.0005 秒。
- delivery 保留探测机会，且继续在几何工作后交付。规划、计量不提前发现
  或消耗出口，终端 reward 与单树扩展约束不变。
- 默认配置运行 v14；`--no-yield-frontier` 或 `--no-fair-frontier` 恢复 v12，
  `--fair-frontier` 显式运行 v13。旧版源码与运行文件全部保留。

测得的 wall time 会影响选择，因此同 seed 在不同机器负载下不保证路径一致。
本轮只有一次完整 i15000，不把单次轨迹差异解释成稳定或因果性的优化效果。

### 命令和计时

使用项目根目录 `.venv`，运行前冻结 `source_v14.zip`。运行期间没有修改算法。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 15000 --snapshot-interval 500 `
  --output src/mcts/group_game/runs/single_tree_group_game_i15000_v14.json
```

seed=3222026，修复次数普通上限 4 / 扩展上限 8、最小保留 rank 17，其余搜索
预算与 v12/v13 一致。完整搜索计时不包含此前 smoke、pytest 或后续审计。

| 版本 | 实际发现 | 缺失 | 完整耗时 | 最后发现 | 无新类尾段 |
| --- | --- | --- | --- | --- | --- |
| v12 | 45/46 | 3 | 681.4297 s | class 13，i7304 / 342.4655 s | 7696 iter / 338.9642 s |
| v13 | 44/46 | 3、34 | 722.0707 s | class 36，i8864 / 418.4588 s | 6136 iter / 303.6119 s |
| v14 | 44/46 | 3、34 | 708.6161 s | class 13，i6552 / 309.4352 s | 8448 iter / 399.1809 s |

相较 v12：少 class 34、慢 27.1864 秒（3.99%）；相较 v13：覆盖相同、快
13.4546 秒（1.86%）。前半程较快不等于最终覆盖更好，仍以 v12 为效果基线。
v12 的 class 34 在 i2312 / 109.0495 秒发现，本轮没有命中。

| iteration | v12 | v14 |
| --- | --- | --- |
| 1000 | 12 | 15 |
| 2000 | 19 | 28 |
| 3000 | 23 | 31 |
| 4000 | 25 | 39 |
| 5000 | 37 | 42 |
| 6000 | 43 | 43 |
| 7000 | 44 | 44 |
| 8000--15000 | 45 | 44 |

部分实际发现节点：class 6 为 i733 / 40.8211 s，class 1 为 i900 / 49.2014 s，
class 9 为 i4040 / 189.8955 s，class 10 为 i4472 / 210.3769 s，class 18 为
i4688 / 221.9216 s，class 24 为 i5732 / 272.4850 s。

### 调度诊断

1858 次 frontier replay 中有 464 次自适应选择：348 次利用、116 次探测。
选择通道计数为 ridge 273、orbit 87、completion 93、delivery 11。
下表计量包括 legacy grow/replay，不等于只由 464 次选择产生的收益。

| 通道 | 计量批次 | 实测秒数 | 新 canonical 终点 | 新非自身源/终点关系 |
| --- | --- | --- | --- | --- |
| ridge | 732 | 3.5756 | 12 | 114 |
| orbit | 1752 | 10.7757 | 15 | 105 |
| completion | 2744 | 14.2940 | 2 | 7 |

- 受计量工作合计 28.6453 秒，占完整耗时 4.04%。这里不含其他几何工作、
  MCTS 主体和新增计量接口自身开销，不能当成全程序 profile。
- 小 facet 枚举检查 13556 个候选，比 v12 的 11493 更多，完成 9/14 个
  合格前沿；但这没有带来新增 class。orbit 检查 6920 个候选、1397 个保留面、
  1205 次发布，30 个源中 22 个用完候选预算，没有完整穷尽的证明。
- 补全任务 2285 个，1042 个预算耗尽，仍有 1243 个未完成、4432 个 LP 名额。
  总补全 LP 9278 次，其中 continuation 5150 次 / 2744 个批次。不能把
  排空这个队列本身作为优化目标，其新增非自身关系只有 7 种。
- 终点图有 21089 个已发布动作键、391 种源/终点关系，全部实际执行过；
  44 种 canonical 终点、0 个 pending nonself。未发现出口没有被缓存漏掉。
- 外部执行 27081 次，重复已执行关系 26690 次，其中自身跳转 11456 次。
  新几何关系仍可能只连接已知类别，这个中间收益不足以解决末段停滞。

### 验证与结论

- 44 条发现路径、109 个外部补全步骤及 credit、生成元不变性均通过审计。
  路径保留面 rank 分布：24:91、23:8、22:3、21:2、19:1、17:4。
- 持久树 14273 个节点，14998 次扩展；14301 条保留边加回收
  173+417+107=697，恰为 14998，保持每次 simulation 最多一条持久扩展。
- 全部候选发布、补全任务预算、orbit 保留面的支持性和子群不变性通过检查。
  逐次自适应选择的数量、探测节奏、收益/耗时有效性也通过审计。
- 最终回归 **137 passed in 10.91s**。新增测得成本归一化、历史收益衰减、
  周期探测、空通道、零耗时下限、非自身/非别名计分、规划不消耗和模式开关测试。
- i200 smoke 为 12.8459 秒 / 1 类；关闭收益调度的 i200 为 17.0452 秒 / 2 类。
  关闭后发现路径/迭代、树、几何统计、补全和 orbit 与 v12 i200 全部一致。
  这两次 smoke 不混入正式计时。最终核对冻结的 21 个代码/测试文件均未变化。

保留这次负向结果，不继续按缺失 class 调整倾向。下一步应先完整 profile，
定位未计量的主要成本；结合已执行源/终点关系，减少重复低收益的修复轨迹，
并在停滞时改善通用候选生成的多样性。当前仅移动少量复访预算，未改变候选
生成器的覆盖边界，不能依靠更高的收益调度比例保证找全。这些是后续方向，
本轮没有追加其他策略或增加预算实验。

原始结果：`single_tree_group_game_i15000_v14.json`；源码：`source_v14.zip`。
完整对照：`v12_v14_i15000_comparison.json`、`v13_v14_i15000_comparison.json`。
计量诊断：`v14_yield_diagnostics.json`；关闭验证：`v14_disabled_smoke_verification.json`。

## v15: 候选缓存与饱和出口门控（i15000）

### 修改与隔离验证

本轮延续 v14 单树框架，不使用未发现 class 样例或 class-specific 参数。
新增两组修改：

- 等价性能优化：缓存不可变 `MacroAction.action_id`；以 30000 项 LRU 缓存
  `(support, pattern)` coherence，以 6000 项 LRU 缓存
  `(role, support, context, pattern)` 动作结构。动态在线 value 不入缓存。
  同一 pattern 内排序只计算变化的尺寸/orbit 项，rewrite 中相同删除尺寸复用
  addition 排序；候选集合、progressive widening 和 rank 检查上限不变。
- 策略修改：一个 facet 源存在已验证关系且没有未执行非自身终点时，普通
  tree/rollout 进入 facet repartition 的总概率从 0.75 降至 0.12。未生成出口
  或仍有 pending nonself 的源维持 0.75；强制 frontier replay 不受影响。
  `--no-endpoint-saturation-gate` 可隔离关闭此策略，恢复 v14 选择。

i500 cProfile 中关闭门控，只测等价优化：搜索内计时从 v14 的 58.7062 秒降至
47.1909 秒，快 19.61%；函数调用由 175915690 降至 120239131。profile 总开销
与正式运行不可直接比较。这说明缓存/排序修改有效，但不保证门控后全程变快。

定向 i1000 成对短跑中，关闭门控为 11 类 / 45.0241 秒，开启为 18 类 /
54.0924 秒。开启时外部执行 1298→601，重复关系 1263→531，自身跳转
932→155，canonical 关系 35→72；释放出的预算进入了更多但更贵的通用动作。
由于 v14 按实测时间选 frontier，同 seed 也存在负载敏感性，短跑仅作机制诊断。

### 正式结果

冻结 `source_v15.zip` 后运行：

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 15000 --snapshot-interval 500 `
  --output src/mcts/group_game/runs/single_tree_group_game_i15000_v15.json
```

seed=3222026、初始已知集合为空、修复上限 4/8、min rank 17，其他预算与
v14 相同。结果 **45/46**，缺 **class 34**，总耗时 **883.0739 秒**。
最后发现 class 24：i7864 / 435.6255 秒；尾段 7136 iterations / 447.4484 秒。

| 版本 | 覆盖 | 缺失 | 总耗时 | 相对 v15 |
| --- | --- | --- | --- | --- |
| v12 | 45/46 | 3 | 681.4297 s | v15 慢 201.6442 s / 29.59% |
| v14 | 44/46 | 3、34 | 708.6161 s | v15 慢 174.4578 s / 24.62% |
| v15 | 45/46 | 34 | 883.0739 s | - |

v15 相对 v14 增加 class 3、没有丢掉 v14 已发现类；相对 v12 则是以 class 3
替换 class 34，总覆盖没有提高。前期节点为 i1000:18、i2000:28、i3000:34、
i4000:38、i5000:41、i6000:42、i7000:44、i8000:45，此后不再新增。

class 3 在 i4796 / 257.9157 秒命中，support 为 `0x153530acca53aac8`，使用
3 次修正。路径先选 atlas pair `BFS322-C2-15`，再选运行中发现的 class 45
稳定子群 `OBS-33cc5a5aa5a5cc33` 的 30 点 orbit，最后经过三组在线验证的
identity facet repartition + singleton exit。没有读取或构造 class 3 样例。

### 重复、成本与审计

- 15837 次饱和修复机会中跳过 13302 次，保留 2535 次；另有 826 次开放源
  修复机会。外部执行 10626 次、重复关系 10227 次、自身跳转 2109 次。
  相对 v12 分别下降 63.14%、64.02%、81.01%。
- canonical 源/终点关系 399 种、终点 45 种、最终 pending nonself 为 0。
  12738 个唯一终端 support，比 v12 多 33.31%；总终端观察 54441 次。
- coherence 缓存 1306815 hits / 1889988 misses；候选结构缓存 10007 hits /
  674941 misses。候选缓存命中很低，说明大部分成本来自真正不同的路径状态，
  不是简单扩大缓存即可解决。
- systematic 检查 16324 个候选；orbit 检查 6663 个、发布 1140 次；补全 LP
  7846 次，2104 个任务中 763 个用完预算，仍有 1341 个任务、4778 个 LP 名额。
- 45 条发现路径与 108 个外部补全步骤通过几何审计。保留面 rank 分布：
  24:90、23:3、22:5、19:2、18:3、17:5。20043 个验证动作键全部发布。
- 持久树 14574 节点、14993 次扩展、14429 条保留边；回收
  182+318+64=564，满足 14429+564=14993，每次 simulation 至多一个持久扩展。
- 最终 **140 passed in 9.47s**。冻结后的 22 个源码/README/测试文件将在文档
  追加后按代码和测试部分核对；i200 smoke 与两次 profile/短跑不计入正式耗时。

门控找到了 class 3，证明将饱和出口预算释放给通用 Filler-Corrector 有价值；
但“canonical source-target 执行过”不是足够精细的无效重复判据。v12 的 class 34
路径包含 7 次连续修正，同一已执行关系在不同前缀、context pattern 或修正深度
仍可能是必要桥梁。下一版应将复访额度按 `(source, target, context/depth)` 计量，
同时保留全局去重作为弱先验，而不是继续降低 0.12。还应让 candidate generation
按 progressive widening lane 真正惰性产生，减少门控释放预算后的 67 万次结构生成。

原始结果：`single_tree_group_game_i15000_v15.json`；源码：`source_v15.zip`。
对照：`v14_v15_i15000_comparison.json`、`v12_v15_i15000_comparison.json`。
摘要：`v15_diagnostics.json`；profile：`v14_i500_profile.pstats`、
`v15_i500_profile.pstats`；smoke：`v14_v15_smoke_comparison.json`。

## v16: Lazy Lane 与上下文桥接门控（i15000）

### 修改与隔离验证

本轮仍是一棵 MCTS 树，不读取未发现 class 的 support、pattern 或标签。v16 对
v15 的两个问题分别做了通用修改：

- 动作生成先取当前 progressive-widening lane，只实例化该 lane 的 add/remove/
  rewrite 候选。排除节点已展开、已拒绝、已回收动作后 lane 为空时，才构造完整
  pool 作为 fallback。开发中的第一版曾在排除前 fallback，会把已用完的 lane
  误判为 pattern 耗尽；定向测试发现并修复，新增回归覆盖此边界。
- 外部出口实际执行时记录
  `(canonical source, canonical target, repair pattern, corrections_used)`。同一个
  source-target 在新 pattern 或新修正深度下仍视为开放；只有当前 repartition
  的所有出口 context 都执行过，才使用 0.12 饱和概率。全局 pair 次数仅作为
  context 之后的弱排序项。关闭 gate 后不查询 context 排序。
- `tree.endpoint_contexts` 保存执行明细和深度分布，且声明
  `class_labels_used=false`。审计要求 context 总执行数与 endpoint graph 的实际
  mapped execution 完全一致；look-ahead 仍不算发现。

i200 默认 smoke 与 v15 均只发现 43、44，耗时 **18.5423 -> 13.6779 秒**，
快 26.23%。同时关闭 yield 和 gate 后，与保存的 v14-disabled 在节点/边、
tree/rollout 步数、动作计数及全部 facet 几何统计上逐项相同，耗时
**17.0452 -> 13.2050 秒**，快 22.53%，证明 lazy lane 没有缩小动作语义。

i500 关闭 gate 的相同 cProfile 条件下，两版均只发现 43、44；
`candidate_actions_for_pattern` 累计时间 **9.8779 -> 4.5526 秒**，下降 53.91%，
搜索内计时 **47.1909 -> 41.1776 秒**，下降 12.74%。v14 的 wall-time yield
调度会因计时微差改变少量轨迹，因此另用上述同时关闭 yield 的结果做严格等价
验证。正式运行前相关回归为 **144 passed in 9.92s**；全仓库 pytest 另有
`ncpol2sdpa`、`mpmath` 两个未安装依赖导致收集失败，与本模块无关。

### 正式命令与结果

运行前冻结 22 个源码、README 与测试文件为 `source_v16.zip`，SHA-256 为
`C5E1C197F65481E315576DE7E9F6B7A9840C5B475C17BBA5F429B0641E625D60`。运行期间
没有修改算法或测试文件：

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 15000 --snapshot-interval 500 `
  --output src\mcts\group_game\runs\single_tree_group_game_i15000_v16.json
```

seed=3222026、初始已知集合为空、修正普通/扩展上限 4/8、min rank 17，其余
配置与 v15 相同。结果 **44/46**，缺 **class 3、34**，耗时 **757.3270 秒**。
最后新增 class 24：i6896 / 335.9736 秒；尾段 8104 iterations / 421.3534 秒。

| 版本 | 覆盖 | 缺失 | 总耗时 | v16 相对变化 |
| --- | --- | --- | --- | --- |
| v12 | 45/46 | 3 | 681.4297 s | 少 1 类，慢 75.8973 s / 11.14% |
| v14 | 44/46 | 3、34 | 708.6161 s | 覆盖相同，慢 48.7109 s / 6.87% |
| v15 | 45/46 | 34 | 883.0739 s | 少 class 3，快 125.7469 s / 14.24% |
| v16 | 44/46 | 3、34 | 757.3270 s | - |

| iteration | v12 | v15 | v16 |
| --- | --- | --- | --- |
| 1000 | 12 | 18 | 13 |
| 2000 | 19 | 28 | 19 |
| 3000 | 23 | 34 | 32 |
| 4000 | 25 | 38 | 39 |
| 5000 | 37 | 41 | 41 |
| 6000 | 43 | 42 | 43 |
| 7000 | 44 | 44 | 44 |
| 8000 | 45 | 45 | 44 |
| 10000--15000 | 45 | 45 | 44 |

本轮 class 9/10 在 i2784/i2856 命中；class 39、6 分别通过 7 次修正命中，
class 2、13 通过 8 次修正命中。44 个发现中修正深度 7/8 各有 2 个，说明新门
确实保留了深链，而不是退回短路径搜索。class 1 仍只在实际终端 i3836 命中后
计入覆盖，并未因 identity singleton 起点而默认加入。

### 机制诊断与结论

- v16 外部执行 15045 次、发现路径包含 152 个已验证外部步骤；v15 分别为
  10626/108，v12 的发现路径为 153。4637 个完整 context 覆盖修正深度 1--8；
  4291 次机会属于全局 pair 已饱和但 context 仍开放，证明该误杀已被解除。
- 仍有 10408/15045（69.18%）次执行重复同一完整 context。全局 pair 重复
  14636 次，自身跳转 2886 次。门控恢复了长链机会，但没有公平保障开放 context
  在同一条 simulation 前缀中连续获得后续 Corrector 预算。
- endpoint graph 有 409 种 source-target pair，全部执行、pending nonself=0，
  canonical target 44 种。12596 个唯一终端 support、57032 次终端观察；候选 miss
  611032，比 v15 少 9.47%，coherence miss 少 10.00%。lazy lane 的性能收益成立。
- 44 条发现路径、152 个外部步骤和 context/endpoint 对账均通过审计。发现路径
  保留面 rank 分布为 24:130、23:10、22:6、21:4、19:2。20775 个验证动作键
  全部发布，无 unpublished target。持久树 14531 节点、14319 条边、14996 次
  扩展；回收 203+347+127=677，满足 14319+677=14996。

后验对照只用于解释结果，不进入搜索。v15 class 3 的三段旧外部链在 v16 中仅
第一段以相同 context 执行（91 次），后两种 pair 没有生成；v12 class 34 的七段
旧链中前三个相同 context 分别执行 8/1/2 次，第四个 pair 全局执行过 3 次但所需
pattern/depth context 没执行，后三种 pair 未生成。两个最终 canonical target 都
不在 v16 endpoint graph，因此不存在“已生成但 pending 未执行”的缓存错误。

结论：v16 的 lazy lane 是可保留的等价性能优化；context gate 将问题从全局饱和
误杀推进到了有限几何候选和树内链式调度，但本次单 seed 最终覆盖回退，不能替代
v12。下一步更通用的方向是给首次 context 几何跃迁小额 Corrector intrinsic credit，
按精确开放 context 公平复访，并让长期没有带来新 support/context 的重复执行衰减；
不能依据 class 3/34 样例硬编码路径或 pattern。

原始结果：`single_tree_group_game_i15000_v16.json`；源码：`source_v16.zip`。
诊断：`v16_diagnostics.json`；正式对照：`v12_v16_i15000_comparison.json`、
`v14_v16_i15000_comparison.json`、`v15_v16_i15000_comparison.json`。
隔离结果：`v14_v16_disabled_smoke_comparison.json`、
`v15_v16_i500_profile_comparison.json`；profile：`v16_i500_profile.pstats`。

## v17--v20: 确定性公平几何增长首次完成 46/46

### 失败链与通用修正

v17 引入按 `(canonical facet, corrections_used)` 保存的深度见证、context/global
双修复门、真实执行后的几何 novelty credit，以及停滞后的 context frontier
复访。i15000 为 **44/46、1088.4179 秒**，缺 3、34；249 次 context episode
没有触发 growth，暴露出按单个见证计数的 grow cadence 过稀。

v18 改为全局 context geometry growth racing，从本次运行见证继续 facet 的
有界枚举。结果 **45/46、1041.2202 秒**，仅缺 class 3；但某个廉价来源得到
199 次服务。v19 加入 endpoint/bridge lane，并让空工作收益衰减，结果
**44/46、1089.9686 秒**，缺 3、34；一个早期产生 target/plan 的来源仍获得
409 次服务。以上后验 class 路径检查只用于诊断，没有写回动作生成或选择。

v20 作如下与 class 无关的修正：

- `YieldFrontier` 和 context growth 均取消按实测秒数归一化；耗时保留在输出中，
  但 `measured_time_affects_choices=false`。
- growth 的 productive 定义收紧为新增 canonical target 或非自身 endpoint pair；
  plan 只计诊断，不再重置 empty streak。
- 每个在线来源先获得 2 次 warm-up。利用阶段的服务次数受硬上限约束：普通来源
  不超过当前 floor+8；曾产生新 target 的来源可到 floor+24。每 4 次仍保留一次
  最少服务来源 probe，避免历史收益永久垄断。
- 保留 v19 的 3:1 endpoint/bridge context 轮转、精确 pattern/depth 复访、全局
  discovered reward 和一棵 MCTS 树；没有 class-specific target 或旧样例。

group-game 核心回归为 **97 passed in 6.05s**；包含保留的串行 interrupt、pair、
Filler-Corrector 方法在内的最终相关回归为 **152 passed in 9.96s**，另有定向的
cheap-empty、plan-only-empty 和 hard service-lead 测试。全仓库测试收集仍因
虚拟环境缺少既有依赖 `mpmath` 而在 `test_group_valid.py` 阻断。运行前冻结源码
与测试为 `source_v20.zip`，SHA-256：
`11118357ABEDA6BD424A139FDDCC24898B666D829C910006D415F38E7DB794F7`。

### 探针与正式运行

固定环境：`PYTHONHASHSEED=0`，`OMP_NUM_THREADS=1`，
`OPENBLAS_NUM_THREADS=1`，`MKL_NUM_THREADS=1`，虚拟环境为仓库根目录 `.venv`。

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game `
  --iterations 15000 --snapshot-interval 500 `
  --output src/mcts/group_game/runs/single_tree_group_game_i15000_v20_snapshot500.json
```

正常阈值 i5000 探针为 **44/46、215.6141 秒**，缺 3、24；相同种子下 v19
i5000 为 40/46，v20 恢复了 1、6、13、34。正式控制运行在 **i6712** 找齐
**46/46** 后提前停止，总耗时 **358.5630 秒**，最后两个稀有命中为：

| class | iteration | elapsed（snapshot=500） | corrections | rollout |
| --- | ---: | ---: | ---: | --- |
| 24 | 5720 | 290.2616 s | 3 | mixed |
| 3 | 6712 | 358.5219 s | 4 | mixed |

覆盖检查点为 i1000:13、i2000:22、i3000:38、i4000:42、i5000:44、
i6000:45、i6712:46。class 1 在 i2784 实际终端命中后才计入覆盖；
`known_classes_at_start=[]`，没有因 singleton identity 起点而默认包含。

无快照的同版本运行也在 i6712 完成 46/46，耗时 **323.7520 秒**。两次运行的
46 个发现 iteration 和完整 action path 逐项一致；每 500 次序列化完整状态带来
34.8110 秒（10.75%）额外 I/O，但不影响确定性调度。因此相对 v19 的总耗时差
不能全部解释为算法加速；可严格归因的结论是 v20 在更少的 6712 次逻辑迭代内
补齐了 v19 缺失的 class 3、34，且没有丢失任何 class。

### 搜索与审计统计

- context geometry growth 共 129 次，覆盖 45 个 canonical 来源；服务范围
  2--19，没有再次出现 199/409 次局部垄断。产生 1 个新 canonical target、
  3 个新 pair、34 个新 plan，总计仅 0.5353 秒。
- endpoint graph 含 341 个 canonical source-target pair、46 个 look-ahead
  canonical target，340 个 pair 已实际执行，pending nonself 为 0。7179 次执行
  中 6839 次为 pair 重复、1998 次为 self pair；2455 个精确 context 与 graph
  mapped execution 完全对账。
- 1678 次 replay 中 217 次 frontier episode；32 次 context episode 包含
  24 次 endpoint 与 8 次 bridge，符合 3:1 周期。树为 6568 节点、6427 条边，
  6712 次 simulation 各至多加入一条持久边。
- `compare_group_game_versions.py` 已兼容 context 计划计数和新版 replay 调度，
  对 v19/v20 完整结果的 facet、endpoint、context、publication 与发现路径审计
  通过。对照文件为 `v19_v20_i15000_comparison.json`。

结论：在固定 seed 的当前 3-2-2 目标集上，v20 是首个单树、无未发现 class
样例引导且实际终端命中全部 46 类的版本。已验证预算上界从 15000 降为 6712；
后续若继续研究，应优先做多 seed 稳定性与最小预算分布，不再为单个 class 添加
专用路径。

## v20: 三个额外随机 seed 的 i15000 稳定性测试

保持 v20 代码与正式配置不变，仅将 seed 改为随机抽取的 `673963275`、
`1706386020`、`210656494`。固定 `PYTHONHASHSEED=0`，BLAS/OMP 单线程，
初始已知集合为空，关闭中间快照以排除大型 JSON 序列化开销。每组均独立启动，
不共享 discovered 集合、MCTS 树或修复缓存。

| seed | 完成迭代 | 覆盖 | 缺失 class | 最后发现 | 搜索耗时 |
| ---: | ---: | ---: | --- | --- | ---: |
| 673963275 | 6824 | 46/46 | 无 | class 34, i6824 | 341.3601 s |
| 1706386020 | 15000 | 45/46 | 3 | class 36, i6296 | 1114.1361 s |
| 210656494 | 15000 | 44/46 | 3, 34 | class 24, i6616 | 1005.6555 s |

三个新增 seed 中 1 次全覆盖，覆盖数为 46、45、44，平均 45 类；总运行时间
2461.1516 秒（41 分 1.1516 秒）。两个未完成 run 在最后发现后仍分别执行
8704 和 8384 次迭代，没有获得新 class。缺失频率为 class 3 两次、class 34
一次，说明 v20 已有较强覆盖能力，但 `i15000` 还不是跨 seed 的全覆盖保证。

连同原 seed `3222026` 的 46/46 结果，当前四个 seed 有 2 个全覆盖，平均覆盖
45.25 类。样本量仍小，不能将 50% 当作稳定成功率；可以确定的是 class 3
仍是最主要的跨 seed 长尾，class 34 次之。原始结果分别为
`single_tree_group_game_i15000_v20_seed673963275.json`、
`single_tree_group_game_i15000_v20_seed1706386020.json`、
`single_tree_group_game_i15000_v20_seed210656494.json`，汇总为
`v20_multiseed_i15000_comparison.json`。
