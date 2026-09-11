# MCTS 入口说明

这是 `src/mcts` 的入口文档。当前目录提供的是基于 Bell 3-2-2 orbit block 的 Monte Carlo Tree Search 版本，用来替代原来的 beam search 试探流程。

## 目录内容

```text
src/mcts/
  __init__.py         # 包级导出
  search.py           # MCTS 树搜索实现
  run_mcts_probe.py   # 命令行入口
  README.md           # 本说明
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
