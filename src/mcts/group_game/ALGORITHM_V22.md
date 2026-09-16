# v22: 可迁移的分层对称性 MCTS

## 目标

v21 把 `context_pattern_id` 写入节点 key，因此相同 support 经不同 pattern 到达时会
形成多个节点。与此同时，一条路径可以逐步换用不同 pattern；此时 pattern 实际上
只是上一动作的标签，既不能表达残余对称性，也会重复学习同一个几何状态。

v22 将两件事分开：

- 几何状态决定 MCTS 节点身份；
- pattern 是生成 Filler/Corrector 动作的群论坐标系，并由节点保存其可用集合。

这套设计位于共享目录 `mcts/symmetry_game`，不依赖 64 个顶点或 26 维坐标，可由
3-2-2 和 4-2-2 共同使用。v20/v21 的默认入口和旧节点 key 保持不变。

## 几何转置节点

启用 `geometric_transpositions` 后，节点 key 为：

```text
(support_word, role, corrections_used, repair_source_word)
```

`context_pattern_id` 不再制造重复节点。每个 `GameNode` 使用
`context_pattern_ids` 保存所有到达该几何状态时见过的 pattern。新路径命中已有节点
时只合并上下文，沿用同一组 visits、edge values 与终端经验。

`repair_source_word` 仍在 key 中，因为同一 retained face 属于不同源 facet 时，
Corrector 的合法外部出口不同，不能合并。

## 分层 pattern 策略

`PatternRelationIndex` 将每个 subgroup 的 vertex-orbit partition 注册为位集划分，
并把候选 pattern 相对节点上下文分为：

- `continue`：继续使用已有 pattern；
- `refine`：候选 partition 的每个 block 都包含在当前某个 block 内；
- `intersection`：两个 partition 交叠，需要降低到更细的共同残余对称性；
- `identity`：退回 singleton 坐标系。

MCTS progressive widening 以固定公平周期打开这些通道，再在通道内部结合在线
pattern value、transition value、support coherence 和 block-size bucket 选动作。
因此跨 pattern 不再是无解释的任意切换，而是显式的 continue/refine/intersection
决策；不同关系的长期收益仍由搜索自行学习。

这是可扩展的 partition certificate，不是完整子群格计算。若 pattern 来自子群
`H` 与 `K`，共同细化可作为 `H intersection K` 作用的廉价候选坐标证书，但代码
不会声称仅凭 orbit partition 已恢复精确的交子群。

## Filler

Filler 在一个几何节点上可以读取全部已合并上下文，并按上述关系逐步打开新的
pattern。动作仍是一个 pattern 下的单 orbit 或有限 orbit union；rank 检查后才能
进入树。候选池和 reward 不读取未发现 class 的 support。

相同 support 经不同填充顺序到达时共享价值，但各 pattern 的 action id 保持不同，
所以 MCTS 仍能分别学习“在哪一种群坐标中填哪一块”更有效。

## Corrector

Corrector 除保留 v21 的在线 facet repair 和 rank-24 ridge pivot 外，还获得两项
与几何节点合并配套的能力：

1. rewrite 候选可从节点已经见过的多个 pattern 中选择删除坐标，并在目标 pattern
   中加入 block；候选缓存 key 包含完整上下文集合，新增上下文后会重新生成。
2. retained face 的外部出口不再只查询首次到达节点的 pattern，而是查询该节点全部
   context 对应的已验证出口，避免 transposition 合并后丢失 Corrector 动作。

终端 class 仍只在 rank-25 supporting facet 实际到达后分类。ridge look-ahead、
endpoint novelty 与全局新 class reward 的边界和 v21 相同。

## 高维共享层

`mcts/symmetry_game` 提供：

- 任意 Python 整数宽度的几何节点 key；
- 任意顶点数的 orbit-partition relation/coherence 缓存；
- 任意环境维数的 ridge 二维法锥 pivot。

4-2-2 v2 使用 256-bit support、rank-78 ridge 和 rank-79 facet，不再含 3-2-2 的
64-bit 或 rank-24 常量。4-2-2 另外使用 context-free action cache、较小的 rollout
pattern window 和 lazy atlas block 加载，避免 721 个 pattern 在每个节点全部物化。

## 入口

3-2-2 v22：

```powershell
.\.venv\Scripts\python.exe -m mcts.group_game.scalable_search --iterations 15000
```

4-2-2 v2：

```powershell
.\.venv\Scripts\python.exe -m mcts.422.scalable_search --iterations 300
```

直接运行 `mcts.group_game` 仍是 v20；`ridge_corrector_search` 仍是 v21。这样新的
群论状态语义可以独立评估，不改变已有覆盖基线。

## 当前限制

- 几何 transposition 合并的是等价 support，不等于把所有到达历史解释为同一个
  精确残余子群；`context_pattern_ids` 是可用动作坐标的并集。
- partition relation 不能替代显式的 subgroup inclusion/intersection 证明。以后若
  4-2-2 的生成元 closure 成本可控，可以在同一接口下升级为精确证书。
- 3-2-2 固定 seed 的首次正式测试保持 46/46，但最后 class 的随机收口仍可能晚于
  v21。高维价值主要来自路径去重、lazy 生成和统一几何接口，不能由单 seed 推断
  覆盖率优势。

## 固定 seed 结果

固定 `PYTHONHASHSEED=0`、BLAS/OMP 单线程、seed 3222026、初始 discovered 为空，
无快照 i15000 在 **i11876** 提前完成 46/46，耗时 **707.4991 秒**，树包含
11398 个节点、11484 条边和 123 个被合并的 pattern 上下文。class 3、24、34
分别在 i1778、i4100、i11876 发现。

这证明几何 transposition 与分层 pattern 策略没有破坏完整覆盖，但当前 3-2-2
效率低于 v21 的 i5579/294.8124 秒。主要长尾是 class 34，而不是 rank/ridge
验证失败。因此 v22 目前是高维迁移与状态语义实验分支，不替换 v21 的 3-2-2
性能基线。完整结果为 `runs/single_tree_group_game_i15000_v22_deterministic.json`。

一次加入独立 subgroup-neighbor lane 的对照在 i10000 仍为 45/46、缺 class 34，
耗时 630.9342 秒，已撤回；快照保留为
`runs/single_tree_group_game_i10000_v22_neighbor_rejected.json`。
