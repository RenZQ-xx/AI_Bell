# v24 机制消融与 v25 精简配置

本文记录 canonical family / concrete embedding 版本的机制消融。目标不是只比较
单个开关，而是识别模块之间的补偿关系，并据此形成一个覆盖率优先、结构更简单的
固定配置。所有完整实验均使用相同环境：

- `PYTHONHASHSEED=0`，OpenBLAS/MKL/OMP 单线程；
- seed `3222026`，从 64 个 singleton 的 identity pattern 开始；
- 最多 6000 次 simulation，发现 46 个 class 后立即停止；
- ridge candidate/batch/focus 分别为 32768/256/48；
- 除明确消融项外，其他搜索参数保持 v24 一致。

## 实验矩阵

| 配置 | 完成迭代 | 覆盖 | 缺失 | 耗时 | 节点 | unique supports |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| v24 full | 5770 | 46/46 | - | 341.3231 s | 5543 | 3850 |
| no family value | 2534 | 46/46 | - | 165.5026 s | 2456 | 1914 |
| no embedding policy | 4890 | 46/46 | - | 317.3409 s | 4706 | 3411 |
| no relation UCB | 5792 | 46/46 | - | 361.0175 s | 5551 | 3968 |
| no reheat | 6000 | 45/46 | 34 | 339.7467 s | 5783 | 3856 |
| no family value + no embedding policy | 5617 | 46/46 | - | 344.1115 s | 5452 | 3949 |
| no family value + no reheat | **2507** | **46/46** | - | **164.0679 s** | **2432** | **1874** |
| lean + no relation UCB | 6000 | 45/46 | 34 | 361.7244 s | 5834 | 4230 |
| lean + embedding cap2 | 4408 | 46/46 | - | 259.2728 s | 4301 | 2974 |

`lean` 表示关闭 family value sharing 和 Filler reheat，同时保留 concrete
embedding、embedding relation policy、relation UCB 和 embedding cap4。

## 可以简化的机制

### 1. 删除 family value sharing

这是最稳定、收益最大的简化。关闭后完成迭代从 5770 降为 2534，耗时从
341.3231 秒降为 165.5026 秒。canonical family 仍作为分组和动作配额使用，但
不同 concrete embedding 不再共享累计 value；每个朝向依靠自己的实际访问结果
学习价值。

共享 value 的问题是信用扩散过强：一个朝向上的成功会抬高整个 family，尚未验证
的 sibling embedding 因此获得虚高先验，并持续挤占 progressive widening 和
rollout 预算。family 是几何等价关系，不等于在当前 raw support 和路径上下文中
具有相同的有限预算收益。

### 2. 在关闭 family sharing 后删除 Filler reheat

只从原 v24 关闭 reheat 会停在 45/46，因此单因素上它像是必要机制。但先移除
family value sharing 后，reheat 不再承担纠正错误信用扩散的任务：进一步关闭
reheat 后仍在 i2507 完成 46/46，并比保留 reheat 少 27 次 simulation、快
1.4347 秒。

因此 reheat 是原配置中的补偿机制，不是 lean 配置的必要组成。删除后也同时移除
两个 patience 时钟、升温比例和后期 phase 切换，使 Filler 调度更容易解释。

## 应当保留的机制

### Concrete embedding 与 embedding relation policy

v23 只保留首次 embedding，在 i6000 为 45/46、缺 class 34。v24 的多个 concrete
embedding 修复了这个表示缺陷。lean 配置中再关闭 embedding policy，完成点从
i2507 后移到 i5617；说明 `reembed` lane、family-first 选择和 sibling 去重仍在
有效控制不同朝向的探索。它们不应与 family value sharing 一起删除。

### Relation UCB

原 v24 单独关闭 relation UCB 仍能完成，但迭代和耗时均略差。更关键的是，在 lean
配置中关闭它会在 i6000 停于 45/46、缺 class 34。独立 embedding 不共享 value
后，relation UCB 是跨 `continue/refine/intersection/coarsen/reembed/identity`
分配预算的主要在线反馈，应当保留。

### embedding cap4

cap2 仍能完成 46/46，且会减少 embedding 分析与存储；但需要 i4408 和
259.2728 秒，比 cap4 lean 分别增加 75.83% 和 58.03%。因此 cap4 是覆盖率和速度
优先的默认值，cap2 只适合作为高维问题中的内存受限选项。

## 推荐配置

v25 独立入口为：

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.lean_embedding_search `
  --iterations 6000 `
  --seed 3222026
```

它相对 v24 full：

- 保留 concrete embedding、embedding relation policy、relation UCB、ridge 与
  adaptive Corrector；
- 关闭 family value sharing；
- 关闭 adaptive Filler reheat；
- 关闭在 3-2-2 搜索中未触发的非终端 leaf heuristic；
- 保持每个 family/class 最多 4 个 embedding 分析。

最佳组合的证据文件是
`runs/single_tree_group_game_i6000_v24_no_family_value_no_reheat.json`。该文件是在
消融入口下生成，所以 revision 仍标记为 v24；除后来确认无行为影响的 leaf
heuristic 外，其有效开关与 v25 入口一致。
原 `embedding_search` 入口不变，仍可复现 v24 full。

## 结论边界

这些实验完成了固定 seed 下的结构筛选，足以支持隔离出 lean 版本，但还不能证明
所有 seed 上都严格更优。正式删除旧实现前，应至少再做 3 个 seed 的 paired run，
重点观察 class 34、24、33、41 的尾部发现率。现阶段保留旧开关和 v24 入口，避免
把单 seed 的性能结论变成不可逆的代码删除。
