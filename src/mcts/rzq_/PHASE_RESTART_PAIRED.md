# 阶段重启配对短测

本实现对应用户确认的阶段重启方案，取代早期折扣型试做。早期 `run_class8_v4_4_restart.py` 及其输出仅为已中止的旧方案，不用于本次结论。历史v4.4入口、公共search模块和原实验结果未被本次修改。

## 已确认的行为

父边Q为本阶段survival累计值除以本阶段这条父边的整数真实遍历次数，分母为0时Q为0。闭包复用子节点可从多个父边抵达，不能用共享子节点visits作为父边Q分母。

节点保留总visits和阶段visits；父边也保留总遍历和阶段遍历。重启只清阶段计数与价值累计量，总访问不变。UCB仍用v4.4公式：

`Q_edge + 1.4 * prior * sqrt(max(1, parent.total_visits)) / (1 + child.total_visits)`。

完整重启时，保存节点当前子边数为阶段基数B。节点立即获6个新名额，此后条件为：

`len(children) - B < max(6, original_K(phase_visits))`，且有未展开动作。

这是累计阶段容量：不是每次选中都再送6个，也不是6加K。旧边保留；未用名额在下次重启时不叠加。新阶段的子边扩展仍沿用既有六bucket轮转（1个prior、5个均匀随机），不重置bucket指针和累计bucket统计。重启后新创建的节点先沿用原始渐进扩展，直至其经历下一次重启。

为落实“阶段累计值/阶段真实访问”，所有分支关闭原discovery epoch的0.25价值折扣，但保留该epoch机制的发现集合、动作分数和缓存刷新。否则新发现会改变阶段累计值的历史权重。旧value_visits字段为公共实现兼容仍存在，但selection适配器明确读取独立整数edge_phase_visits；测试故意使旧分母错误，仍能选择正确的子边。

## 固定短测

- v4.4 class8 orbit partition、flat closure、关闭对称复用、开启相同闭包复用，六bucket compatibility-richness prior、最小总child.visits=1、raw Q、rollout同分均匀无放回等保持。
- 预发现集合为 `[44,43,23,38,41,46,45,35,39]`，不预先登记class8。
- 基准seed=20260502；search_index=29，派生task seed=20288754。选择随机流初始seed=20289763；同分随机流初始seed=2167772401。
- 100轮预热、不重启。固定第100轮，不按收益挑快照。保留task.step的productive状态切换，由外层仅覆盖停机预算，使所有分支真正完成固定轮数。
- A从完整快照恢复，不重启；B清Q和阶段访问，但扩展继续按原总访问规则；C完整重启（立即6个名额）。各60轮。
- 短测期间新class立即加入该分支的全局集合，但不再触发重启，不延长预算。
- A再从相同快照重跑前10轮，逐条验证事件trace一致。

## 快照与隔离

snapshot.pkl包含树、共享闭包节点的对象关系、全局发现集合、缓存、task状态和两个随机流。发现回调使用可序列化的绑定函数，恢复后指向该分支自己的global state。弱引用缓存保存当前条目，恢复后重建WeakValueDictionary，缓存对象别名关系由pickle保持。每次分支都从同一字节串重新加载，绝不跨分支共享可变缓存。

源码在独立phase_restart.py中，用上下文适配器绑定interrupt_search的selection/backpropagation/widening函数，退出后恢复。该适配器只用于独立单进程实验，不支持在同一解释器里并发运行其他搜索。

## 运行

在WSL工作区根目录：

```bash
PYTHONPATH=src:src/mcts/rzq_ src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/test_phase_restart.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/run_class8_phase_restart_paired.py \
  --output src/mcts/rzq_/runs/class8_phase_restart_paired_100_60_v2
src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/analyze_phase_restart_paired.py \
  src/mcts/rzq_/runs/class8_phase_restart_paired_100_60_v2
```

输出目录若已存在会拒绝覆盖。环境记录Python、numpy/BLAS、线程/hash设置、配置和源码指纹。首次尝试在100轮保存时遇到弱引用序列化错误，旧目录保留；v2只修复快照保存，预热轨迹另作一致性验证。

## 指标和解释边界

检查重启瞬间UCB首选变化、Q清零、总visits/探索项不变、饱和节点重开；检查前10/30/60轮和分段窗口的根分布、熵、路径变化、真实遍历来源、扩展额度使用、图深度以及Exact分布。额外名额扩展指该次扩展时，节点按其当前总visits计算的原K已饱和；并不等于它在任何无重启轨迹下都不可能发生。

A→B检验清除Q历史；B→C检验开放阶段扩展。分叉后随机调用会不同，逐轮路径不同不能全归因于某一个公式项。一个seed、一个快照只能说明局部机制效果，不能证明总体新class发现能力提高。本次不运行连续300轮无新class的正式实验。

完整重启事件JSONL：`restart_events.jsonl`（B/C各一次，含完整前后统计）。`verification.json`还记录了两次预热trace一致及父边/节点计数一致性。
