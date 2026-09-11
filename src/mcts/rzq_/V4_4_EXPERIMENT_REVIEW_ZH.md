# v4.4 原始实验设计与结果（中文审阅稿）

状态：待用户检验。本文件整理原始class8_v4_4_300_trace的300轮结果；没有重跑实验、修改源码或结果，也不包括后续重启变体。
## 证据与版本边界

结果来源：[result.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v4_4_300_trace/result.json)；原始逐步记录：[decisions.jsonl](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v4_4_300_trace/decisions.jsonl)；环境：[environment.json](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/runs/class8_v4_4_300_trace/environment.json)。
下面源码链接均固定到已上传的提交 a4093cedbdf626a572187dd069a8884dca9029df，便于本地审阅后在GitHub仍能跳转到同一版本。运行行为以入口、实际配置和trace共同核对，不仅依赖历史meta描述。
## 目标与初始条件

v4.4在v4.3基础上只把expansion bucket数量从2恢复为6，考察prior/随机扩展配比及早期扩展配额变化。它没有D重启机制。保留相同class8分块与其余v4.3设置。
预发现集合共29类（包含class8和class25）：

```json
[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
```

固定调用300轮task.step，档案证实实际完成300轮。它不是“连续300轮无新发现”的耐心实验；新类出现不会触发D重启或重新计300轮。

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

## Q、探索项与渐进扩展

历史父边Q = edge_survival_sums[action] / edge_value_visits[action]。分母是价值统计计数，并不是后续v11修正的整数阶段父边遍历次数。发现epoch变化时，价值累计量与value_visits按0.25折扣；真实node.visits不折扣。此点属于历史实现事实，不能在总结中写成已经修正。
普通selection的探索项为1.4·prior(action)·sqrt(max(1,parent.visits))/(1+child.visits)，并非经典log形式UCB。最低访问保障使用child.visits，阈值1，通常新扩展后已满足。
原K(s)的基础部分为floor(max(1,1+sqrt(max(1,visits))))，还取与bucket配额上限的较大者。bucket_quota=1，每桶初始保障配额与visits+1共同限制；不是在搜索开始直接固定发放6个无条件名额。

## rollout评分的历史真实行为

非终点分数为rank_gain项 + flat项 + supportability项 + decline。rank增益为正时给5，否则−5；flat项为−log(1+flat_capacity)，support项为−log(1+closer_side)。几何/closure与终点分类由原scorer完成。
decline参考集合只在scorer构造时从预发现集合与根兼容类的交集中确定。它不会自动随着后续发现增加；不能把动态节点compat缓存或动态expansion prior误认为rollout decline也已动态更新。
设A为参考类中父compat非零的类别。普通下降率D为各类相对下降率的算术平均；normal=(D/0.2)·exp(1−D/0.2)，D=0时为0。A非空且每类父compat≤1时进入bridge，历史实现额外加5E，E为参考mask消除比例。也就是bridge仍采用normal+5E，没有强制下降的动作资格约束。
Exact终点使用原静态终点回调：本实验新类命中记录为100分，已知类命中记录为10分，无效终点为−10分；终点分覆盖过程评分，不再叠加decline。compat是参考mask包含关系计数，不是命中概率。

本版本固定decline参考集合为22类：`[7,8,12,15,20,22,25,28,29,30,31,32,34,35,38,39,40,42,43,44,45,46]`。

## 原始结果

| 指标 | 结果 |
|---|---:|
| 实际iterations / root visits | 300 / 300 |
| 唯一节点数 | 296 |
| expansion / 相同闭包复用事件 | 300 / 5 |
| 最大实际树边路径深度（根为0） | 4 |
| 普通selection次数 | 539 |
| rollout动作步数 | 4508 |
| Exact命中 / 不同Exact类数 | 62 / 10 |
| 相对预发现集合新增类数 | 2 |
| D重启次数 | 0 |
| 根访问最多分支占比 | 21.00% |

“不同Exact类数10”不等于“新类数10”。相对预发现集合仅新增：

| 新class | 首次iteration | 分数 |
|---:|---:|---:|
| 19 | 125 | 100.0 |
| 10 | 218 | 100.0 |

新类为class19（125轮）、class10（218轮），最终全局集合为原29类并入这两类，共31类。class25虽被命中，但已在预发现集合中。

| Exact class / 无效类型 | 次数 |
|---|---:|
| exact:class10 | 1 |
| exact:class15 | 1 |
| exact:class19 | 1 |
| exact:class20 | 1 |
| exact:class25 | 1 |
| exact:class28 | 1 |
| exact:class29 | 1 |
| exact:class43 | 14 |
| exact:class44 | 40 |
| exact:class45 | 1 |
| invalid:boundary | 79 |
| invalid:non_coplanar | 159 |

根分支按真实父边回传统计（action为零基block编号）：

| 根action | 访问数 |
|---:|---:|
| 3 | 63 |
| 11 | 33 |
| 23 | 32 |
| 9 | 30 |
| 37 | 29 |
| 28 | 27 |
| 7 | 22 |
| 15 | 21 |
| 8 | 16 |
| 38 | 11 |
| 22 | 8 |
| 6 | 2 |
| 39 | 1 |
| 30 | 1 |
| 31 | 1 |
| 25 | 1 |
| 32 | 1 |
| 33 | 1 |

## 复现方式与局限

base seed=20260502，task seed=20288754；选择/rollout主RNG初始seed=20289763，rollout同分RNG seed=2167772401。记录环境为Python3.12.3、WSL2 Linux；详细依赖和源码指纹以environment.json为准。
从仓库根目录使用原虚拟环境运行到一个新目录，避免覆盖档案：

```bash
src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/run_class8_v4_4.py --iterations 300 --output src/mcts/rzq_/runs/class8_v4_4_review_replay
```

本总结只进行了代码/数据读取与统计核验，没有执行上述重跑命令。v4.4与v11的预发现集合、Q分母、重启规则和总预算不同，因此不能直接用各自终点的新类数比较算法优劣。相同前300轮可作描述性对照，也不能单独归因于重启。

## 相关Python文件与跳转

| Python 文件 | 核验职责 |
|---|---|
| [run_class8_v4_4.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v4_4.py) | 独立入口；bucket数量设为6。 |
| [run_class8_v4_3.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v4_3.py) | 继承层：关闭Q的min-max归一化。 |
| [run_class8_v4_2.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v4_2.py) | 继承层：最低child访问阈值降为1。 |
| [run_class8_v4_1.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v4_1.py) | 继承层：全部普通动作、禁止对称节点复用、保留完全相同闭包复用。 |
| [run_class8_v3.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v3.py) | 上游工厂与compatibility-richness配置。 |
| [run_class8_round1.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_round1.py) | RZQ评分器基础工厂。 |
| [run_class8_v2.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/run_class8_v2.py) | 共享固定轮数执行器、预发现集合注入与trace输出。 |
| [search.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/search.py) | 渐进扩展K(s)、bucket调度、父边UCB/PUCT、rollout候选采样与同分top-k、回传。 |
| [interrupt_search.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/interrupt_search.py) | task.step、发现回调、真实RNG状态、全局发现缓存和epoch处理。 |
| [node_manager.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/node_manager.py) | flat闭包节点管理、相同闭包状态复用、父边关联。 |
| [rollout_scorer.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/rollout_scorer.py) | 过程评分、固定decline参考集合、normal与历史5E bridge奖励。 |
| [compatibility.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/compatibility.py) | class→参考facet mask兼容计数。 |
| [audit_environment.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/audit_environment.py) | 环境、依赖和源码/数据指纹记录。 |
| [compare_v4_variants.py](https://github.com/RenZQ-xx/AI_Bell/blob/a4093cedbdf626a572187dd069a8884dca9029df/src/mcts/rzq_/compare_v4_variants.py) | 历史v4系列横向比较入口。 |
