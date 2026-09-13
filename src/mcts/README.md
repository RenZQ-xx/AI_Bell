# MCTS 入口说明

这是 `src/mcts` 的入口文档。当前目录提供的是基于 Bell 3-2-2 orbit block 的 Monte Carlo Tree Search 版本，用来替代原来的 beam search 试探流程。

## 目录内容

```text
src/mcts/
  __init__.py                  # 包级导出
  search.py                    # MCTS 树搜索和可暂停 session
  interrupt_search.py          # 单线程栈式中断搜索
  subgroup_patterns.py         # 纯群论 atlas、stabilizer 分解与 pattern 重建
  subgroup_interrupt_search.py # class 中断 + subgroup 渐进队列统一搜索
  filler_corrector_search.py   # Filler + Corrector 搜索
  run_mcts_probe.py            # 基础命令行入口
  README.md                    # 本说明
```

## 如何运行

先从仓库根目录进入，再在 PowerShell 中设置 `PYTHONPATH` 并运行入口：

```powershell
Set-Location E:\pycode\AI_Bell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.run_mcts_probe
```

这会使用默认参数运行一组 class1 / pattern0 的 probe，并把结果写到：

```text
src/mcts/runs/modular_mcts_seed20260502_probe4.json
```

## 常用参数

`run_mcts_probe.py` 先从 `data/facet_classes_322_examples.txt` 里读代表行，再构造 orbit block，然后执行 MCTS。最常用的参数如下：

- `--row-class`：代表类编号，默认 `1`
- `--rep-index`：该类下的代表行编号，默认 `1`
- `--pattern-index`：orbit pattern 编号，默认 `0`
- `--rare-target-classes`：希望重点追踪的稀有类
- `--target-classes`：普通目标类集合，默认 `1..46`
- `--seeds`：随机种子列表
- `--restarts-per-seed`：每个种子的重启次数
- `--iterations`：MCTS 总迭代次数
- `--max-depth`：rollout 最大深度
- `--exploration-constant`：UCB 探索系数
- `--discount`：rollout 折扣因子
- `--prior-temperature`：扩展阶段候选动作的先验温度
- `--rollout-temperature`：rollout 阶段采样温度
- `--expansion-candidate-pool`：扩展时保留的候选动作数
- `--rollout-candidate-pool`：仿真时保留的候选动作数
- `--terminal-scoring-mode`：终局评分模式，默认 `static`

## 示例命令

跑一个和 baseline 风格接近的 class1 probe：

```powershell
Set-Location E:\pycode\AI_Bell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.run_mcts_probe --row-class 1 --rep-index 1 --pattern-index 0 --seeds 20260502 --restarts-per-seed 4 --iterations 2000 --max-depth 60 --exploration-constant 1.4 --discount 0.97 --output src/mcts/runs/class1_mcts.json
```

如果你只想做快速冒烟测试，可以把 `--iterations` 降到 50 或 100。

## Filler-Corrector

`filler_corrector_search.py` 会同时评估 no-op 与多个一块/两块删除候选。
各候选从相同树快照和 RNG 状态开始，先进行交错 pilot，再把预算分给
finalist 和胜者。所有 task 共享一个 contextual-UCB Corrector，但其输入
只包含结构、终端出口统计和在线搜索回报，不查询未发现 class 的案例兼容性。

输出中的 `corrector.events[*].arm_evaluations` 记录每个候选的计算量、命中、
单位计算收益和相对 no-op 优势，`corrector.shared_policy` 记录跨 task 学习状态。

## 输出文件

输出是一个 JSON，结构和 baseline 的 probe 一致，主要看这几个字段：

- `meta`：运行参数
- `runs`：每个 restart 的详细结果
- `summary.label_counts`：最终最好终局的标签统计
- `summary.encountered_label_counts`：搜索过程中遇到的终局标签统计
- `summary.opened_rare_target_classes`：最终打开的稀有类
- `summary.encountered_rare_target_classes`：实际遇到过的稀有类

每个 `run` 中常见字段：

- `best`
- `terminal_bests`
- `encountered_label_counts`
- `iterations_completed`
- `nodes_created`
- `root_visits`

## Subgroup Pattern 统一搜索

`subgroup_interrupt_search.py` 把旧 pair bridge 与 class-rooted interrupt
放到同一个调度器中。它读取 `data/subgroup_search_atlas_322.json` 中的纯群论
两层 DAG：恒等群之上有 24 个极小超群代表（23 个 `C2`、1 个 `C3`），
其中恰有 18 个 `C2` 在 64 个点上诱导 32 个二元 block，也就是旧 pair
involution search 的 18 个 pattern。

每次发现新 class 后，程序从该次终端的法向量重建完整 tight support，再在线
计算 stabilizer。完整 stabilizer 中的 `C2/C3` 子群会按群共轭类分解到一级
pattern；低阶组合则匹配二级 `C2 x C2/C4/C6/S3` 节点。这里不会读取
`class -> pattern` 表，也不会使用未发现 class 的案例选择 subgroup。

一级 subgroup task 先运行固定 50 次 racing warm-up；调度器一次只保留一棵
未判定树，完成后再启动下一棵，以便及时释放淘汰树的缓存。warm-up 中到达过
任意 cross-class 的树晋级并继续使用完整预算，其余树立即淘汰。晋级的一级 task
只有在实际发现全局新 class 后，才沿 growth edge 渐进加入少量二级 task。

恒等分区由 4 棵独立 seed 的 MCTS 树组成；四棵树共享 discovered 集合、结构
缓存和终端验证缓存，但不共享节点统计。对 64 个 singleton block 的树，rollout
到达 affine rank 23 后会枚举所有二点尾部，并只在 rank 25 终端执行分类。每棵
树的枚举前缀数由 `--rank23-tail-max-prefixes` 限制；非 singleton subgroup 不会
触发该枚举。class task 仍保持 LIFO 中断优先。

串行入口默认把一个 rank-23 尾部拆成每批 512 个 pair 候选。若一批未完成，程序
会原子挂起整次 MCTS iteration，并立即续跑同一 frame；完成前不会公开部分发现、
回传 reward、增加节点 visit 或消耗 iteration。这样既保留未分批版本的 RNG、候选
顺序和搜索轨迹，也让每次 `task.step()` 有界。跨树完成的尾部仍写入共享终端缓存。
可用 `--rank23-tail-candidates-per-step` 调整批大小，或用
`--no-rank23-tail-active-service` 恢复非原子分批行为。

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m mcts.subgroup_interrupt_search `
  --initial-class-id 1 `
  --iterations 300 `
  --root-iterations 300 `
  --level2-iterations 300 `
  --subgroup-warmup-iterations 50 `
  --identity-tree-count 4 `
  --rank23-tail-max-prefixes 24 `
  --rank23-tail-candidates-per-step 512 `
  --output src/mcts/runs/subgroup_interrupt_class1_i300_racing4_rank23tail.json
```

输出中的 `discovery_timeline[*].pattern_analysis` 记录完整 stabilizer 阶、轨道
大小、一级分解、多重度和二级匹配。例如 class 44 的完整 stabilizer 阶为
48、轨道为 `24+24+8+8`；四阶 `{e,g,h,f}` 是其中一个可用于搜索的低阶
子群，而不是完整 stabilizer。

精简 atlas 由外部 `322_` 项目的纯群论产物生成，并预存群共轭 canonical
partition 以降低启动时间：

```powershell
.\.venv\Scripts\python.exe scripts/export_subgroup_search_atlas.py
```

## 代码入口

如果你要在别的脚本里复用 MCTS，可以直接导入：

```python
from mcts.search import MCTSConfig, run_mcts_search
```

然后把 `ExpansionScorer` 和 `pattern.orbits` 传进去即可。

## 和 baseline 的关系

- `src/baseline/run_strict_log_probe.py` 仍然是旧的 beam search 入口
- `src/mcts/run_mcts_probe.py` 是新的 MCTS 入口
- 两者共享同一套 `baseline` 的几何、验证和评分模块

如果后续要做实验对比，建议保留两个入口分别跑，不要混用同一个输出文件。
