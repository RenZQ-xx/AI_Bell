# rzq_：基于 MCTS 的实验与修改

本目录用于保存 rzq 的实验实现、配置与复现说明。已建立独立 uv 环境；当前操作说明见下方，其后的“推荐方案”保留为最初设计参考。

## 当前环境与运行入口

已固定 Python 3.12.3、NumPy 2.4.0，使用独立 `.venv`，不继承父环境的 site-packages。适用范围是当前 MCTS 追踪/回放代码，不是整个 AI_Bell 的所有功能。NumPy 2.4.0 已被 PyPI 撤回（兼容性问题），这里为复现父环境而有意精确锁定；未来升级需单独比较结果。

在 Ubuntu / WSL 的 Bash 中，从本目录执行：

```bash
unset UV_PROJECT_ENVIRONMENT
uv sync --locked
uv run --locked python reproduce_class8.py
```

入口自动定位仓库 `src`，调用目录外的 `mcts.trace_class8_current_300`，将输出重定向到本目录 `runs/class8_current_300_trace/`，不修改原实验文件。它保存 `result.json`、`decisions.jsonl`、`README.md` 的原始及换行归一化 SHA-256；仅将 CRLF 转成 LF 后比较，内容不一致会以非零状态退出。每次运行保存环境和源码/数据指纹。已有输出时会拒绝覆盖，重新运行前请移走旧输出目录；只重新验收已有输出可运行 `uv run --locked python reproduce_class8.py --compare-only`。

完整的新追踪文件、运行环境快照和比较摘要保存在被忽略的 `runs/`；它们是本机验证产物，不随源码提交。

注意：需要完整仓库和现有未提交的追踪代码、参考结果。仅 checkout 当前 HEAD 不能重建这些未提交内容。提交实验时必须包含这些依赖，指纹记录不能代替源码本身。

## Exact-class compatibility

`compatibility.py` 从 `baseline.reference_classes` 的已知 facet 数据构建兼容索引，不依赖被 Git 忽略的 `src/experiments/`。对每个已知 facet，它把64顶点 tight-support 投影到当前 orbit-block partition；只要某个 block 被部分包含，该 facet 就不属于这个搜索空间。对于状态 `key`，`compat_class_c` 是满足“已选 blocks 是 facet block mask 的子集”的 class-c masks 数。

查询 class8 pattern0 的空状态：

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src uv run --project src/mcts/rzq_ --locked \
  python -m mcts.rzq_.inspect_compatibility
```

查询选择 blocks 0、7、10 后的状态：

```bash
PYTHONPATH=src uv run --project src/mcts/rzq_ --locked \
  python -m mcts.rzq_.inspect_compatibility --selected-blocks 0 7 10
```

运行独立回归检查：

```bash
PYTHONPATH=src uv run --project src/mcts/rzq_ --locked \
  python -m unittest mcts.rzq_.test_compatibility
```

## 推荐方案

将本目录作为独立 uv 项目：自己的 `pyproject.toml`、`uv.lock`、`.python-version` 和 `.venv/`；通过仓库的 `src` 导入已有 `mcts` 和 `baseline` 源码。复现交付单位是整个 AI_Bell 仓库的确定 Git 提交，而不是单独复制本目录。

父项目和子项目的 `.venv` 可以并存。解释器使用哪个环境，决定第三方依赖来自哪里；`PYTHONPATH` 决定从哪里找到仓库源码。子环境不会自动继承父环境安装的包：调用外部源码所需的第三方依赖，也必须声明在子项目中。

虚拟环境只隔离 Python 依赖，不隔离操作系统、驱动、系统库或文件访问。若实验需要更严格的系统一致性，可以后续增加 Docker；GPU 驱动等仍需单独记录。

## 建议布局

```text
src/mcts/rzq_/
  README.md
  pyproject.toml       # 实验及调用链的第三方依赖、必要的 uv sources/index
  uv.lock             # 提交到 Git，固定依赖解析结果
  .python-version     # 固定实际验证过的 Python 补丁版本
  .venv/              # 本机重建，不提交
  __init__.py
  run_experiment.py   # 实验入口，复用 mcts/baseline
  configs/            # 完整参数，包括随机种子
  scripts/            # 环境检查、运行、结果比较
  expected/           # 小型参考结果、评价指标和允许误差
  runs/              # 本地原始输出；后续添加忽略规则
```

根目录已有 `.venv/` 忽略规则，也覆盖此处的 `.venv/`。其他条目是建议，尚未创建。

## 环境初始化原则

1. 创建独立项目时可使用 `uv init --bare --no-workspace`，避免自动修改父项目为 workspace。当前根配置没有声明 workspace。uv workspace 共享锁文件和环境，不适合这里要求的独立依赖实验。
2. 子项目采用应用模式，无需构建安装自身；可以显式设置 `[tool.uv] package = false`。用 `PYTHONPATH` 暴露仓库源码，无需先改造根项目打包配置。
3. 初次对照实验优先匹配根 `uv.lock` 中实际使用的版本，再有意修改依赖。单纯复制根 `pyproject.toml` 的版本范围并重新解析，可能得到不同版本。
4. 不默认复制全部重型依赖：按实验实际调用链确定需求。但若需要根项目的自定义来源（例如 Git 版 ncpol2sdpa、PyTorch 专用索引），必须在独立项目中明确配置，不能假设父项目的 sources 自动适用。
5. 维护者执行 `uv lock` 并提交锁文件；复现者执行 `uv sync --locked`。锁文件不匹配时应报错，不应在复现过程中自动更新依赖。
6. 如果以后将 AI_Bell 正式配置为可安装包，可以考虑 editable path dependency；目前不将它作为前提。可编辑依赖仍依赖仓库实际源码，不能代替 Git 提交固定。

## 复现命令约定（环境配置完成后）

统一在 Ubuntu/WSL 内运行以下 Bash 命令。以下是未来环境建立后的命令模板，目前不能直接作为已验证的复现入口。

```bash
# 在 AI_Bell 仓库内执行；先 checkout 要复现的完整实验提交
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 防止外部设置将 uv 项目环境重定向到其他位置
unset UV_PROJECT_ENVIRONMENT
uv sync --project "$ROOT/src/mcts/rzq_" --locked

# 工作目录固定为仓库根目录，兼容现有相对数据/输出路径
PYTHONPATH="$ROOT/src" PYTHONHASHSEED=0 \
uv run --project "$ROOT/src/mcts/rzq_" --locked \
  python -m mcts.run_mcts_probe \
  --seeds 20260502 --restarts-per-seed 1 --iterations 50 \
  --output "$ROOT/src/mcts/rzq_/runs/baseline_smoke.json"
```

显式指定 `--project`，不要使用父环境的 `python` 或 `uv run --active`。未来自己的入口使用 `python -m mcts.rzq_.run_experiment`。入口和配置应使用仓库相对路径或基于 `__file__` 定位，避免 `/home/ap809` 等个人绝对路径。

## 一次实验必须记录什么

- **代码**：YangBingquan 上游基准提交 SHA、包含实验和所有外部源码修改的实验提交 SHA；正式参考结果应来自干净工作树。子目录外的改动同样属于实验依赖。
- **环境**：提交 pyproject、uv.lock、Python 版本；记录 uv 版本、操作系统、CPU/GPU，以及实际用到的 CUDA、系统库、外部二进制版本和构建方式。需要 MOSEK 等工具时说明许可证配置方式，不提交许可证。
- **输入**：数据文件相对路径与 SHA-256；未被 Git 跟踪的数据给出可访问下载位置或生成命令。
- **执行**：完整命令、全部参数、随机种子、重复次数、停止条件和线程设置。分别固定实际使用的 Python/NumPy/PyTorch 随机源；固定一个 seed 不保证所有随机源或并行过程都确定。
- **验收**：保存小型参考结果和比较命令。说明是要求逐项一致，还是在明确容差内复现指标；跨硬件浮点计算和耗时不应默认逐位一致。算法效果比较使用相同输入、预算和多组 seeds。

交付前用一个干净 checkout 和新环境走完文档中的命令，确认没有依赖本机未提交源码、缓存或数据。不要把“依赖安装成功”当成“结果已经复现”。

## 与上游协作

建议从确认过的上游基准提交建立个人分支，例如 `codex/rzq-mcts`。新增实现尽量放本目录，通过函数复用、包装或明确的扩展接口调用原实现。必须调整共享代码时，保持改动小且在提交说明中解释原因；单纯放进子目录不意味着运行时独立。

创建本说明时，当前分支为 `mcts`，HEAD 为 `e5b06684113ddd92fd23ae291f59ee5fd9d38a6f`，工作树已有共享 MCTS 文件修改及未跟踪实验文件。因此该 HEAD 不能单独代表当前本机运行的源码状态，也尚未确认为 YangBingquan 的对照基准。

## 官方参考

- [uv workspace：共享管理及自动加入行为](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [项目配置与虚拟环境](https://docs.astral.sh/uv/concepts/projects/config/)
- [依赖来源与 editable path dependency](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [锁文件和环境同步](https://docs.astral.sh/uv/concepts/projects/sync/)
