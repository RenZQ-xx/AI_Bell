# Bell 4-2-2 Group Game Runs

本目录只记录 `mcts.422` 的隔离实验。搜索使用 subgroup orbit blocks；
`nodes/*/facets.csv` 不参与搜索。

## 2026-09-14 migration smoke

- `smoke_i1.json`：关闭 catalog/canonicalization 的 CLI 启动验证，1 iteration，
  2 个持久节点，0.05 秒。
- `smoke_i10_catalog_tail24.json`：有界 tail、正式 catalog 分类，i10 用时
  5.72 秒；在 iteration 1、rollout depth 22 发现
  `BFS422-F-b6b3c8b38259`（catalog index 34835），最终持久树 11 个节点。
  本次共做 763 次 tail 候选检查，其中 112 次进入完整终端几何验证。

正式对照应使用默认 `tail_checks=24` 并保留 canonicalization/catalog。

## 2026-09-14 i15000 baseline

命令：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.422 --iterations 15000 --seed 4222026 `
  --progress-interval 25 `
  --output src\mcts\422\runs\bell422_group_game_i15000.json
```

- 完成 15,000 次逻辑 iteration，总耗时 7,356.60 秒，即约 2 小时 2 分
  36.6 秒。
- 最终发现 2 / 34,836 个 catalog classes：
  `BFS422-F-b6b3c8b38259` 和 `BFS422-F-4d5952852bdb`。
- `BFS422-F-b6b3c8b38259`：iteration 1、1.77 秒、rollout depth 22，
  由 pattern `BFS422-H-0650` 的 rank-tail enumeration 发现，tight support
  含 240 个顶点，catalog index 34835。
- `BFS422-F-4d5952852bdb`：iteration 9733、4,697.15 秒、rollout depth 11，
  由 pattern `BFS422-H-1193` 的 rank-tail enumeration 发现，tight support
  含 224 个顶点，catalog index 34829。
- 终态命中频率分别为 28,638 和 6；第二个 class 之前连续停滞 9,732 次，
  之后到结束又停滞 5,267 次。
- 持久树含 14,983 个节点、16,824 条边；15,000 次 iteration 几乎都在创建
  新节点，transposition/复访共享很弱。
- Filler 共生成 68,528,242 个候选；action cache miss 1,284,408 次、hit
  26,139 次，并发生 1,264,408 次 eviction，说明当前 cache key 粒度几乎不能
  复用 rollout 候选。
- rank cache miss 1,972,555 次、hit 385,054 次；完整终端验证 208,819 次。
  rank-tail 检查 1,526,416 个候选，其中 204,778 个进入完整几何验证，成功
  9,639 次，但绝大多数落回第一个 class。
- 在线观察 258 个 facet supports，生成 148 个不同 online stabilizer patterns；
  pattern 数量增加没有转化为相应的 class 多样性。

结论：增加到 i15000 确实在长尾中找到第二个 class，但当前 v1 的主要瓶颈不是
预算上限，而是候选复用率低、持久树几乎每次只横向增加一个叶子，以及 tail
成功结果高度集中于同一 class。后续应先修正树的扩展/复访比例、按 canonical
terminal class 对 tail 出口降权，并将候选缓存拆成与 support 无关的 pattern
结构模板和轻量 state mask 两层，再扩大预算。

## 2026-09-14 scalable v2

隔离入口 `python -m mcts.422.scalable_search` 启用几何 transposition、分层
partition policy、rank-78 ridge pivot、context-free action cache 和较小的 lazy
候选窗口。搜索先验边界与 v1 相同，仍不读取 `nodes/*/facets.csv`。

相同 seed、catalog/canonicalization 开启的 i10 公平 smoke：

| 版本 | 覆盖 | 耗时 | Filler 候选 | terminal 验证 | tail 几何检查 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 1 | 5.7184 s | 80,247 | 119 | 112 |
| v2 | 1 | 4.1819 s | 15,996 | 40 | 37 |

v2 将 Filler 候选物化减少 80.07%，terminal 验证减少 66.39%，耗时减少
26.87%。结果为 `bell422_scalable_i10_v2_catalog.json`。

i100 v2 完成 100 次、101 个节点、109 条边，耗时 34.9573 秒；仍只发现
`BFS422-F-b6b3c8b38259`。共生成 145,942 个 Filler 候选、执行 438 次 terminal
验证，说明新缓存和窗口可以持续运行，但这个预算不足以证明覆盖提升。结果为
`bell422_scalable_i100_v2.json`。

## 2026-09-16 v25 migration i100

入口：`python -m mcts.422.v25_search`。seed 为 4222025，catalog 与完整群
canonicalization 开启，结果为 `bell422_group_game_i100_v25.json`。

- 完成 100 次 iteration，用时 470.0649 秒；持久树 94 个节点、100 条边，root
  visits 为 100，符合每次 simulation 最多增加一条持久边的约束。
- 共发现 2 个 class：`BFS422-F-b6b3c8b38259` 在 i1、4.3188 秒发现，support
  size 240；`BFS422-F-4d5952852bdb` 在 i66、365.5282 秒发现，support size 224。
- 两条发现路径都是纯 Filler 路径，长度分别为 18 和 19；末步均为 rank-77 的
  两块 tail enumeration。第二类末步使用 `BFS422-H-1073` 的两个 blocks，合计
  加入 4 个 vertices。
- 本次树中仍执行了 3 条持久 remove、1 条持久 rewrite，并完成 5 次 ridge pivot，
  但它们没有直接出现在两条 discovery path 中，不能把第二类归因给 Corrector。
- 共 2,408 个 rollout steps、251 个 tree-policy steps；8 次触发 max depth 112。
  rank cache miss 21,422 次，terminal validation 1,215 次。
- Filler 物化 341,323 个候选，Corrector 物化 20,086 个候选；tail 检查 2,635
  次，其中 363 次进入几何验证并成功 22 次。
- 在线生成 5 个 concrete stabilizer patterns；knowledge epoch 为 2，且
  `share_embedding_family_values=false`。

对照意义：旧 v1 i15000 中第二类到 i9733 才出现，v2 i100 只找到第一类；v25
在 i66 即找到第二类，短预算覆盖明显改善。但 v25 i100 比 v2 i100 的 34.9573 秒
慢约 13.45 倍，主要来自更深 rollout、更多候选物化、terminal validation 和在线
stabilizer/ridge 工作。后续优化应保留多 pattern/tail 的覆盖能力，同时压缩 rollout
候选池并减少重复 exact-terminal 验证。

## 2026-09-16 v25 paired i10000 / i15000

使用 seed 4222025 执行一条确定性的 i15000 轨迹，并在 i10000 原子保存完整快照。
这与同 seed 独立 i10000 的搜索前缀相同，避免把前 10,000 次昂贵的 4-2-2
canonicalization 重算一遍。运行设置六小时 wall-time 上限，最终正常完成，未截断。

| 预算 | 覆盖 | 耗时 | 节点 / 边 | rollout steps | terminal validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| i10000 | 2 | 11,484.72 s | 8,402 / 9,776 | 209,522 | 100,459 |
| i15000 | 2 | 16,572.48 s | 12,440 / 14,501 | 313,418 | 149,741 |

总耗时分别为约 3 小时 11 分 24.72 秒和 4 小时 36 分 12.48 秒。两类均在早期
发现：`BFS422-F-b6b3c8b38259` 在 i1，`BFS422-F-4d5952852bdb` 在 i66。
i66 之后到 i15000 没有第三类。

i10000 到 i15000 的增量为：

- 5,087.75 秒、4,038 个节点、4,725 条边；
- 103,896 个 rollout steps；
- 13,571,430 个 Filler candidates、1,112,217 个 Corrector candidates；
- 168,957 次 tail 检查、13,777 次 tail geometry checks、913 次 tail success；
- 32 次 ridge pivot calls、158 次 ridge success；
- 49,282 次 terminal validation 和 953,852 次 rank cache miss；
- 新 class 为 0，unique canonical terminals 仍为 2。

i15000 累计有 1,585 条持久 remove 和 307 条 rewrite，ridge success 647 次，说明
Corrector 不是没有运行；问题是这些修改仍只到达同两个 terminal basins。两个预算的
endpoint pair count 都只有 1，也是 basin 跳跃缺少多样性的直接信号。

性能方面，i15000 共物化 39,862,855 个 Filler candidates，candidate cache miss
1,423,713 次并 eviction 1,403,713 次，缓存命中仅 31,976 次。额外预算主要转化为
重复候选生成、rank 和终端验证，没有转化为覆盖。因此当前配置已经表现出明显的
action-distribution / basin-diversity 瓶颈，继续单纯增加 iteration 的收益很低。

结果文件为 `bell422_group_game_i10000_v25.json`、
`bell422_group_game_i15000_v25.json`，结构化差分为
`bell422_v25_i10000_i15000_comparison.json`。
