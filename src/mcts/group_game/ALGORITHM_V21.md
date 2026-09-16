# Group-Game MCTS v21

v21 的目标是在同一棵 MCTS 树内加强后期 Corrector，使搜索从已发现 facet 的
真实 rank-24 ridge 系统地跳向邻接 facet，同时减少覆盖已经较高时的自由 Filler
抽样。它是 v20 的可选扩展，不修改默认 `mcts.group_game` 行为。

## 入口与边界

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.ridge_corrector_search `
  --iterations 15000 --seed 3222026 --snapshot-interval 0
```

- 单 MCTS 树；每次 simulation 最多增加一条持久边。
- 初始已知 class 集合为空，singleton identity pattern 不等于发现 class 1。
- Filler、Corrector 和调度器不接收未发现 class 的 support、pattern 或标签。
- ridge look-ahead 只验证几何出口；class 仅在 rollout 实际命中终端后分类。
- `search.py` 中新增开关默认关闭，旧 v20 命令和调度保持不变。

## Rank-24 Ridge Pivot

设已验证 facet 为 `T`，其仿射维数为 25；Corrector 枚举保留面 `R subset T`，
且 `rank(R)=24`。包含 `R` 的仿射超平面构成二维法向空间。`ridge_pivot.py` 将所有
64 个顶点的 supporting halfspace 投影到该二维空间，枚举所得凸锥的极射线。每条
有效极射线对应一个包含 `R` 的 supporting facet，因此可直接得到 `T` 之外的
邻接 facet，而无需猜测应补入哪个顶点或预先知道目标 class。

每个候选仍经过 `validate_facet_support` 验证。若数值退化或没有直接出口，则回退
到原 v20 的 singleton outside-vertex 检查，避免丢失旧路径。经对称 canonical 化
后的 RepairPlan、endpoint pair 和出口继续写入原共享缓存。

## 后期 Corrector 调度

覆盖率使用本次运行中实际命中的 target class 计算。低于 50% 时保持 v20 的
75% 自由 Filler、25% 周期 Corrector replay。达到 50% 后，自由比例为：

```text
free_ratio = 0.10 + (0.75 - 0.10) * (1 - coverage)^2
```

其余配额通过确定性余额累加器转为 Corrector replay。每次后期新 class 命中，
其在线 facet 获得 48 个 ridge batch；v21 每批检查 256 个候选，单 facet 最多
32768 个候选。生成新 canonical source-target pair 后，该来源的 delivery 获得
临时优先级，确保几何结果进入 rollout，而不只是留在缓存中。

## Reward 与树语义

v21 不改变终端全局 novelty reward，也不把 ridge look-ahead 当作发现。直接 pivot
生成的 target 和 source-target pair 只沿用 v20 的小额几何 novelty credit；真正
的新 class 奖励必须等实际路径到达 rank-25 终端并首次命中。Corrector 的 rewrite
和 Filler 的 external-exit add 都是普通 MacroAction，进入原 progressive-widening
候选池、rollout 和回传流程，持久节点仍是 GameNode。

## 固定 Seed 结果

seed 3222026、`PYTHONHASHSEED=0`、BLAS/OMP 单线程、无快照：

| 版本 | 全覆盖迭代 | 耗时 | 节点 | class 3 | class 24 | class 34 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v20 | 6712 | 323.7520 s | 6568 | 6712 | 5720 | 3512 |
| v21 | 5579 | 294.8124 s | 5455 | 2257 | 3650 | 5579 |

v21 的全覆盖迭代减少 16.88%，耗时减少 8.94%。这是一组固定 seed 的对照，不代表
跨 seed 成功率；后续仍需复用 v20 的多 seed 方案检验 class 3/34 的稳定收口。
