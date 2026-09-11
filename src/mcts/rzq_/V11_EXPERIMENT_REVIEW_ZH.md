# v11 原始实验设计与结果（中文审阅稿）

状态：待用户检验。范围严格限定为原始v11从第1轮到第1394轮的正式实验。包括它原本第1095–1394轮无新发现阶段；不包括后300轮重新计算、v11.1耐心延长、动态decline修复、反事实成功轨迹或bridge二元约束。
## 证据与版本边界

正式配置：[config.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/config.json)；结果：[result.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/result.json)；环境：[environment.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/environment.json)；原审计：[verification.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/verification.json)。
源码与结果链接固定到已上传提交 a4093cedbdf626a572187dd069a8884dca9029df。历史行为保留真实缺陷，不能把后续修正解释为原v11已经具备。
## 目标与初始条件

以v4.4为基础，采用D方案：首次全局新class触发重启，保留图结构及累计visits，清空阶段Q，通过新扩展名额和封印旧边的特殊selection恢复探索机会。单独运行一棵class8保留图，不启动其他class的栈式搜索。
预发现集合共9类，不含class8：

```json
[44,43,23,38,41,46,45,35,39]
```

| 配置 | 实际设置 |
|---|---|
| 搜索对象 | class8 原 orbit block 分块，40个block；search_index=29，parent_search_index=27 |
| closure / 状态复用 | flat closure开启；禁止对称等价合并；完全相同闭包状态复用 |
| 动作集合 | 全部未选普通block，不做对称动作商集缩减 |
| expansion | 6个bucket轮转；0按compatibility-richness prior抽样，1–5均匀抽样 |
| prior | 2·log(1+兼容类数)+非零类log(1+mask数)的均值，经温度1.0转为prior；发现相关prior/cache刷新与rollout decline参考集合是不同机制 |
| selection | 最低child总visits=1；父边survival Q不做min-max归一化；探索项使用真实父子节点总visits，c=1.4 |
| rollout | 温度0.85；top-k=4；边界同分均匀无放回选择 |
| 候选评分预算 | 常规8；productive阶段16；每第8次rollout可扩大到24；受剩余动作数限制 |
| 回传 | survival权重1，novelty权重0；discount=0.97 |
| 深度限制 | max_depth=60；终点rank判定另行控制，不能把该上限当实际树深度 |

## 父边Q与D重启的实际实现

Q修正为本阶段父边survival累计值 / 本阶段父边整数真实遍历次数；分母由edge_phase_visits提供，不用旧value_visit权重，也不用共享child的visits代替父边计数。无阶段访问时Q=0。发现epoch的旧价值折扣被设为1.0，避免破坏真实阶段分子/分母。
探索项仍为1.4·prior(action)·sqrt(max(1,parent总visits))/(1+child总visits)。重启不重置这些总visits，因此探索项本身并不会因重启而增大；变化来自清空Q以及扩展/特殊selection资格。
每次首次发现全局新Exact class，回调立即加入全局集合；本轮rollout和backpropagation完成后，同轮发现合并触发一次重启。发现轮归旧阶段，新阶段从下一轮开始。
重启对所有唯一保留节点执行一次：清空阶段node/父边访问、value累计量和Q；保留图、总visits、父边总遍历次数、bucket轮转指针及桶累计数；封印此前访问过的父边。旧阶段信息归档到trace和阶段统计。
每个已有节点立即获得6个名额；后续累计容量为max(6,K(本阶段visits))，已用名额=重启后新增父边数+特殊selection次数。初始搜索不额外重启，第一次新class之后才执行D。
有名额且有未展开动作：正常expand并继续rollout。有名额但所有动作已展开：从封印旧边中仅按探索项选择，消耗1个名额并解封该父边，然后沿原树逻辑继续；这不是强制立即从该child rollout。
名额用完时普通selection。若没有可选封印边，也回退普通selection。普通访问同样解封对应父边。完全相同闭包子节点被多父边复用时，复用不会重置共享节点或再发名额；父边访问与封印独立。
## 预算与停机

初始耐心300轮，每次新class使连续无新发现计数归零。没有额外全局总轮数上限；独立入口覆盖task原早停/预算，直到连续300轮无新发现。故1394不是预设预算，而是此次运行结果。

## rollout评分的历史真实行为

非终点分数为rank_gain项 + flat项 + supportability项 + decline。rank增益为正时给5，否则−5；flat项为−log(1+flat_capacity)，support项为−log(1+closer_side)。几何/closure与终点分类由原scorer完成。
decline参考集合只在scorer构造时从预发现集合与根兼容类的交集中确定。它不会自动随着后续发现增加；不能把动态节点compat缓存或动态expansion prior误认为rollout decline也已动态更新。
设A为参考类中父compat非零的类别。普通下降率D为各类相对下降率的算术平均；normal=(D/0.2)·exp(1−D/0.2)，D=0时为0。A非空且每类父compat≤1时进入bridge，历史实现额外加5E，E为参考mask消除比例。也就是bridge仍采用normal+5E，没有强制下降的动作资格约束。
Exact终点使用原静态终点回调：本实验新类命中记录为100分，已知类命中记录为10分，无效终点为−10分；终点分覆盖过程评分，不再叠加decline。compat是参考mask包含关系计数，不是命中概率。

**原v11的重要历史限制：** decline参考集合固定为`[35,38,39,43,44,45,46]`，没有随发现与重启扩大。bridge也仍是normal+5E，未实施“有下降才可选”的二元约束。本总结有意保留这两个事实。

## 终点结果

| 指标 | 原v11结果 |
|---|---:|
| 总iterations / root visits | 1394 / 1394 |
| 相对初始集合新class数 | 16 |
| 全局已发现类数 | 25 |
| 重启 / 阶段数 | 16 / 17 |
| 最后新发现iteration | 1094（class10） |
| 终点连续无新发现 | 300 |
| 唯一节点 / 父边 | 1292 / 1394 |
| 最大图深度（根为0） | 4 |
| expansion / 相同闭包复用事件 | 1394 / 103 |
| 普通 / 特殊selection | 2161 / 109 |
| rollout动作步数 | 21214 |
| Exact命中 / 不同Exact类数 | 374 / 18 |

最终全局集合：`[7, 8, 10, 12, 15, 19, 20, 22, 23, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46]`。
在当前class8根分块下，仍兼容而未发现的类别是9、11、25。这不等于全部46类只剩这三类未发现；其他未发现类可能在当前根分块下compat为0。

## 首次新发现时间线与阶段长度

| class | 首次iteration | 本次结束的阶段长度 |
|---:|---:|---:|
| 29 | 5 | 5 |
| 15 | 8 | 3 |
| 8 | 32 | 24 |
| 12 | 72 | 40 |
| 20 | 77 | 5 |
| 42 | 80 | 3 |
| 28 | 128 | 48 |
| 7 | 138 | 10 |
| 34 | 228 | 90 |
| 30 | 311 | 83 |
| 40 | 515 | 204 |
| 19 | 573 | 58 |
| 22 | 660 | 87 |
| 32 | 866 | 206 |
| 31 | 1054 | 188 |
| 10 | 1094 | 40 |
| 无新发现，停止 | 1394 | 300 |

上述17个阶段长度合计1394，16次新发现对应16次重启。

## Exact与无效结果分布

| Exact class / 无效类型 | 次数 |
|---|---:|
| exact:class10 | 1 |
| exact:class12 | 1 |
| exact:class15 | 22 |
| exact:class19 | 2 |
| exact:class20 | 9 |
| exact:class22 | 1 |
| exact:class28 | 15 |
| exact:class29 | 136 |
| exact:class30 | 1 |
| exact:class31 | 1 |
| exact:class32 | 1 |
| exact:class34 | 2 |
| exact:class40 | 2 |
| exact:class42 | 12 |
| exact:class43 | 20 |
| exact:class44 | 136 |
| exact:class7 | 3 |
| exact:class8 | 9 |
| invalid:boundary | 271 |
| invalid:non_coplanar | 749 |

## 各阶段树规模与访问分布

| 阶段 | iteration区间 | 长度 | 终点节点 | 图深度 | 根最大分支占比 |
|---:|---|---:|---:|---:|---:|
| 0 | 1–5 | 5 | 6 | 1 | 20.00% |
| 1 | 6–8 | 3 | 9 | 1 | 33.33% |
| 2 | 9–32 | 24 | 33 | 3 | 37.50% |
| 3 | 33–72 | 40 | 73 | 3 | 35.00% |
| 4 | 73–77 | 5 | 78 | 3 | 20.00% |
| 5 | 78–80 | 3 | 81 | 3 | 33.33% |
| 6 | 81–128 | 48 | 129 | 3 | 41.67% |
| 7 | 129–138 | 10 | 139 | 3 | 40.00% |
| 8 | 139–228 | 90 | 228 | 4 | 30.00% |
| 9 | 229–311 | 83 | 307 | 4 | 24.10% |
| 10 | 312–515 | 204 | 499 | 4 | 18.63% |
| 11 | 516–573 | 58 | 556 | 4 | 43.10% |
| 12 | 574–660 | 87 | 638 | 4 | 32.18% |
| 13 | 661–866 | 206 | 830 | 4 | 24.76% |
| 14 | 867–1054 | 188 | 999 | 4 | 13.30% |
| 15 | 1055–1094 | 40 | 1030 | 4 | 37.50% |
| 16 | 1095–1394 | 300 | 1292 | 4 | 16.33% |

最后300轮访问18个根动作，最大占比16.33%。真实根父边访问分布：

| 根action | 访问数 |
|---:|---:|
| 15 | 49 |
| 18 | 31 |
| 20 | 27 |
| 19 | 26 |
| 16 | 25 |
| 13 | 24 |
| 1 | 21 |
| 26 | 19 |
| 28 | 19 |
| 5 | 15 |
| 2 | 13 |
| 27 | 12 |
| 32 | 10 |
| 8 | 3 |
| 34 | 2 |
| 39 | 2 |
| 11 | 1 |
| 31 | 1 |

## 行为解释与可比性

前8次新发现经由该阶段新建根边；后8次经由阶段初已有根边，这些旧根边此前在该阶段被特殊selection解封。没有一次新发现恰好发生在执行特殊selection的同一轮。因此数据支持“后期重新访问旧根分支参与了发现”，但不能把每次发现都单独归因于特殊selection。详细证据见[ANALYSIS.md](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/ANALYSIS.md)。
与原v4.4同取前300轮：Exact命中62 vs 78，expansion均300，普通selection539 vs 364，特殊selection0 vs 21，最大实际树路径深度均4。可以作为对齐窗口的行为统计，不能视为单一重启变量的因果收益，因为初始发现集合和Q分母也变化。
不能直接比较v4.4的300轮新增2类与v11的1394轮新增16类来宣称提升；尚无同初始集合、同Q定义、同预算/停机规则的完整无重启对照，也没有多seed统计。

## 可复现性与结果文件

base seed=20260502；task seed=20288754；主RNG初始seed=20289763；rollout同分RNG seed=2167772401。Python3.12.3，WSL2 Linux。正式命令：

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/run_class8_v11.py \
  --output src/mcts/rzq_/runs/class8_v11_review_replay
```

本总结没有执行重跑。入口拒绝覆盖已存在目录；--cap只用于另目录冒烟。原正式前20轮与冒烟一致；终点审计检查计数、父边访问、全局发现及重启事件顺序。
checkpoint.pkl保存最后一次重启后状态，即第1094轮；final_snapshot.pkl保存1394轮终点，含图、共享状态和RNG。两者不能混用。GitHub中大文件保存为无损gzip，见[ARCHIVE.md](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/ARCHIVE.md)和[SHA256SUMS](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/SHA256SUMS)。
完整trace：[decisions.jsonl.gz](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/decisions.jsonl.gz)；重启事件：[restart_events.jsonl](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/restart_events.jsonl)；终点快照：[final_snapshot.pkl.gz](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v11/final_snapshot.pkl.gz)。

## 相关Python文件与跳转

| Python 文件 | 核验职责 |
|---|---|
| [run_class8_v11.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v11.py) | 原正式入口：预发现集合、发现驱动重启、连续300轮停机、日志与快照。 |
| [run_class8_v4_4.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v4_4.py) | 继承的v4.4任务工厂。 |
| [phase_restart.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/phase_restart.py) | 真实阶段父边计数、Q适配、统计清零、可序列化scorer与快照读写。 |
| [sealed_restart.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/sealed_restart.py) | D方案：即时6名额、封印父边、特殊selection与名额消耗。 |
| [search.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/search.py) | 渐进扩展K(s)、bucket调度、父边UCB/PUCT、rollout候选采样与同分top-k、回传。 |
| [interrupt_search.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/interrupt_search.py) | task.step、发现回调、真实RNG状态、全局发现缓存和epoch处理。 |
| [node_manager.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/node_manager.py) | flat闭包节点管理、相同闭包状态复用、父边关联。 |
| [rollout_scorer.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/rollout_scorer.py) | 过程评分、固定decline参考集合、normal与历史5E bridge奖励。 |
| [compatibility.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/compatibility.py) | class→参考facet mask兼容计数。 |
| [audit_environment.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/audit_environment.py) | 环境、依赖和源码/数据指纹记录。 |
| [verify_v11.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/verify_v11.py) | 原正式trace和终点计数审计。 |
| [analyze_v11.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/analyze_v11.py) | 新旧根边发现来源、阶段统计、原v4.4参考比较。 |
| [test_phase_restart.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/test_phase_restart.py) | 阶段统计及Q计数相关测试。 |
| [test_sealed_restart.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/test_sealed_restart.py) | 特殊selection、共享名额与父边隔离等测试。 |
