# Progressive Widening 与 Reward 消融

本文在 v25 lean embedding 配置上继续拆分 progressive widening 与 reward。目标是
寻找可以删除的机制，同时保持 3-2-2 的 46-class 覆盖和运行效率。

## 实验设置

- `PYTHONHASHSEED=0`，OpenBLAS/MKL/OMP 单线程；
- seed `3222026`，最多 3000 次 simulation；
- v25 lean 的 concrete embedding、relation UCB、ridge 和 adaptive Corrector 保持开启；
- family value sharing、Filler reheat 保持关闭；
- 每次只关闭一项，控制组在 i2507 提前完成 46/46。

控制组与上一轮
`single_tree_group_game_i6000_v24_no_family_value_no_reheat.json` 的发现序列、节点、
边、tree/rollout steps 和 unique supports 完全一致，说明新增实验开关默认透明。

## Progressive Widening

| 配置 | 完成迭代 | 覆盖 | 缺失 | 耗时 | 节点 |
| --- | ---: | ---: | --- | ---: | ---: |
| v25 full | 2507 | 46/46 | - | 166.1970 s | 2432 |
| child-only widening | 3000 | 41/46 | 13, 33, 34, 36, 41 | 200.6413 s | 2960 |
| pattern beta 0.45 -> 0.50 | 3000 | 45/46 | 24 | 178.5484 s | 2937 |
| no size bucket | 3000 | 45/46 | 34 | 182.6508 s | 2934 |
| no action lane | 3000 | 26/46 | 20 classes | 205.0618 s | 2879 |
| uniform relation lanes | 3000 | 45/46 | 34 | 184.9474 s | 2912 |
| no epoch reserve/recycle | 3000 | 21/46 | 25 classes | 170.3819 s | 2860 |
| no child recycle | 3000 | 45/46 | 34 | 180.4974 s | 2931 |

结论：当前活跃的 widening 层不能按组删除。pattern/block/action 三层控制的是不同
粒度；action lane 对覆盖影响最大。size bucket 与 relation 重复配额都保护了 class
34 的尾部路径。discovery epoch reserve 是后期扩容的主要来源，child recycle 虽只
发生 6 次、回收 48 条边，但关闭后也未在 i3000 收口。

因此保留：

- pattern、block、action 三层 progressive limit；
- block-size bucket 轮转；
- Filler/Corrector action lane；
- weighted relation lane 与 relation UCB；
- discovery-epoch child reserve 和低价值边回收；
- pattern beta `0.45`，不强行统一为平方根。

可以直接删除两个不参与生产路径的遗留项：`_matches_action_lane` 只被测试当作
过滤 oracle，`GameNode.action_bucket_cursor` 从未被读取。两者已移除，不改变搜索。

## Reward 整组消融

| 配置 | 完成迭代 | 覆盖 | 缺失 | 耗时 | 节点 |
| --- | ---: | ---: | --- | ---: | ---: |
| v25 full | 2507 | 46/46 | - | 166.1970 s | 2432 |
| no transition shaping | 3000 | 45/46 | 24 | 182.5637 s | 2924 |
| no terminal support shaping | 3000 | 44/46 | 24, 34 | 191.3982 s | 2951 |
| no terminal frequency shaping | 3000 | 45/46 | 34 | 190.5761 s | 2907 |
| no Corrector new-class credit | 3000 | 42/46 | 24, 33, 34, 41 | 193.4908 s | 2882 |
| no endpoint novelty credit | 3000 | 45/46 | 24 | 184.4688 s | 2915 |
| no leaf value | **2507** | **46/46** | - | **163.8385 s** | **2432** |

`no leaf value` 与控制组的发现序列、tree/rollout steps、terminal counts、root value
和 unique supports 逐项相同。两个运行均为 `depth_limit_hits=0`、`dead_ends=0`。
进一步扫描 110 个历史 `single_tree_group_game_*.json`，没有任何结果出现非零
`depth_limit_hits`；dead end 会直接使用 `dead_end_reward`，也不会调用 leaf value。

因此 `0.018 * rank - 0.006 * size_excess` 在现有 3-2-2 工作流中是不可达 reward。
`lean_embedding_search` 现已默认关闭该项：

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.lean_embedding_search `
  --iterations 3000 `
  --seed 3222026
```

共享 core 暂时保留该开关和实现，因为旧版本需要可复现，且 4-2-2 的更深路径可能真正命中 depth limit；
不能仅凭 3-2-2 结果把这一结论外推到高维问题。

## Reward 细粒度消融

| 配置 | 覆盖 | 缺失 | 耗时 |
| --- | ---: | --- | ---: |
| no Filler transition reward | 45/46 | 34 | 174.6338 s |
| no Corrector transition cost | 41/46 | 9, 10, 22, 30, 31 | 181.1978 s |
| no endpoint context credit | 44/46 | 24, 34 | 195.6353 s |
| no endpoint pair credit | 45/46 | 34 | 194.7640 s |
| no endpoint target credit | 44/46 | 24, 34 | 218.7154 s |
| no novel-support bonus | 44/46 | 24, 41 | 196.8322 s |
| no repeated-support penalty | 44/46 | 24, 34 | 202.3971 s |
| no known-class frequency penalty | 44/46 | 31, 34 | 236.0782 s |
| no Corrector exact-frequency penalty | 44/46 | 24, 34 | 209.0742 s |

细分后仍未找到第二个无损项：

- Filler rank gain 与 Corrector 修改成本都提供独立方向；
- endpoint 的 context、pair、target 分别奖励上下文、邻接边和全局终点 novelty；
- support 的首次 bonus 与重复 penalty 不能互相替代；
- 普通 terminal 与 Corrector counterfactual 的 class-frequency penalty 都有作用；
- Corrector new-class credit 不是终端 10 分的简单重复，它负责把 credit 直接分配给
  产生 basin 跃迁的 remove/rewrite。

## 判定边界

这些结果说明相应机制在固定 seed、i3000 的当前调度中不能无损删除，并不构成数学
上的必要性证明。多个变体只漏 class 24 或 34，增加到 i6000 后可能仍会完成；但它们
在控制组已经完成的预算上同时更慢、覆盖更低，不符合当前“覆盖优先且提高效率”的
简化标准。若未来更看重代码体积而允许更长运行，可以再对这些 45/46 变体做多 seed
和 i6000 比较。

所有变体由 `policy_reward_ablation_search.py --variant ...` 复现，完整 JSON 位于
`runs/single_tree_group_game_i3000_v25_*.json`。

## 三个额外 seed 的 i10000 验证

最新 `lean_embedding_search` 默认关闭 leaf shaping。沿用历史 v20 的三个困难 seed，
上限设为 i10000，三组均提前完成 46/46：

| seed | 完成迭代 | 最后发现 | 耗时 | 节点 |
| ---: | ---: | --- | ---: | ---: |
| 673963275 | 4507 | class 24 | 256.4864 s | 4417 |
| 1706386020 | 4386 | class 24 | 236.1897 s | 4270 |
| 210656494 | 3154 | class 34 | 195.6599 s | 3080 |

平均完成点 i4015.7，最晚 i4507；三组 `depth_limit_hits` 均为 0。历史 v20 在相同
seed 上分别为 46/46、45/46、44/46，其中后两组运行到 i15000 仍有遗漏。结果说明
v25 的覆盖改善能跨越这组三个既有困难 seed，而不是只存在于 seed 3222026。
结构化汇总见 `runs/v25_multiseed_i10000_comparison.json`。
