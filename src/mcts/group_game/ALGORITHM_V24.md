# v24: Pattern Family 与 Concrete Embedding

## 问题

v23 的后验审计发现，同一个 canonical stabilizer pattern ID 在不同运行中可能保存
不同的 raw-coordinate orbit blocks。例如 `OBS-03cf50faaf05fc30` 的 ID 来自
canonical support，但 blocks 来自搜索首次命中的具体坐标朝向。旧实现同时存在：

1. `_learn_terminal_pattern` 按 canonical support 只分析一次；
2. `register_pattern` 对相同 ID 使用 `setdefault`。

因此后来实际命中的共轭朝向会被忽略。MCTS 学到的不是完整的 stabilizer family，
而是随机保留下来的第一个 embedding。

## 两层表示

v24 将在线观测 pattern 拆为两层：

- `PatternFamily`：ID 仍为 `OBS-{canonical_support}`，表示同一个 canonical
  symmetry family，并共享在线价值统计；
- `PatternEmbedding`：ID 为 `OBS-{canonical_support}-E{partition_digest}`，保存
  当前 64 个顶点坐标中的具体 orbit partition、候选动作和 transition 统计。

digest 只由排序后的 orbit bit words 计算。同一 partition 重复出现时不会新增动作，
不同 partition 不再互相覆盖。该过程只分析搜索实际到达并完成终端验证的 support，
不会枚举 3072 个群元素，也不读取未发现 class 的代表样例。

## 有界在线收集

启用 v24 后，同一 canonical family 最多保留 4 个 embedding，每个已验证 class 的
stabilizer 分析也最多 4 次。重复 raw support、重复 partition 和达到上限的 support
分别跳过并计数。这样 action space 随搜索经验增长，但保持有限。

旧版本只在 canonical support 首次出现时学习 pattern。v24 对重复 canonical support
仍检查 raw support；仅当这个具体朝向从未处理过且预算未耗尽时才运行新的 stabilizer
分析。终端 reward、discovered 集合和 class 完成条件没有改变。

## 共享学习与公平 widening

每个 embedding 保留自己的 pattern/transition stats；同一 family 另有共享的
family stats 和 family-transition stats。未访问的新 embedding 继承 family value，
但获得小额 embedding novelty optimism；使用后由 70% family value 和 30% concrete
value 共同评分。

分层关系新增 `reembed`：current 与 target 的 concrete ID 不同，但 canonical family
相同。Progressive widening 使用独立 `reembed` quota。其他 relation lane 首先在
尚未展开的 family 中选一个代表，避免一个 family 的多个 embedding 挤掉其他群结构；
rollout 同样保留当前 family 的 sibling embedding，并在全局 top pool 中每个 family
只放一个代表。

MCTS 节点仍使用 v21/v23 的 context-sensitive key，不重新启用 v22 的几何 policy-node
合并。Filler/Corrector 可以在一条路径中使用不同 concrete embedding，但每次动作
仍是某一个真实 subgroup partition 下的 orbit 或 orbit union。

## 入口

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.embedding_search --iterations 6000
```

入口继承 v23 的 directed hierarchy、relation UCB、adaptive Corrector、ridge pivot
和双停滞 Filler reheat，并额外启用 embedding 收集。v20--v23 的入口和默认开关
保持不变。

## 固定 seed 结果

环境：`PYTHONHASHSEED=0`、BLAS/OMP 单线程、seed 3222026、初始 known classes 为空、
无中间快照。

| 版本 | 完成迭代 | 覆盖 | class 3 | class 34 | class 24 | 耗时 | 节点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v21 | 5579 | 46/46 | 2257 | 5579 | 3650 | 294.8124 s | 5455 |
| v23 | 6000 | 45/46 | 2576 | 未发现 | 2341 | 339.7004 s | 5855 |
| v24 | 5770 | 46/46 | 2576 | 4269 | 5770 | 341.3231 s | 5543 |

v24 在 i5770 提前完成 46/46，最后发现 class 24；class 34 在 i4269 命中。class 1
在 i2466 实际终端命中后才加入覆盖，起始 singleton pattern 没有被计作已知 class。

本次共执行 167 次新朝向分析，注册 124 个不同 embedding，43 次分析得到重复
partition。46 个 family 的 embedding 数为：15 个 family 达到 4，13 个为 3，
7 个为 2，11 个为 1，未超过硬上限。78 个非首 embedding 全部获得过动作访问，
累计 5320 次；14 个非首 embedding 动作出现在新 class 的发现路径中，最后的
class 24 也由非首 embedding 启动。

class 34 本次路径使用的是两个 family 的首 embedding，因此不能把它的提前发现
单独归因于某个新增朝向；可以严格确认的是，修复表示后改变了全局 family/action
调度，并且新增 embedding 对多个发现及最终 class 24 有直接贡献。

v24 比 v23 少执行 230 次 simulation 并补齐 class 34，但因为额外 stabilizer 分析，
墙钟时间仍多 1.6227 秒。相对 v21，多 191 次 simulation、慢 46.5107 秒。当前结果
证明了单 seed 的覆盖恢复，不构成跨 seed 稳定性结论。

完整结果：`runs/single_tree_group_game_i6000_v24_deterministic.json`。核心回归测试
为 108 passed，family/embedding 上限、唯一 ID、实际 class 1 命中和 46/46 输出均
通过结构审计。

## 后续方向

- 对多个 seed 评估 1--4 个 embedding 的覆盖收益分布；
- 将 expensive stabilizer analysis 改为 family 内按停滞触发，而非尽早用满预算；
- 根据 family 的 concrete embedding 边际收益自适应分配上限；
- 在 4-2-2 中沿用同一 family/embedding 接口，但使用任意宽度 partition digest，
  不引入 64-bit 假设。
