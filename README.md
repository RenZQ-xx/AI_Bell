# Project AI Bell 
AI搜索bell不等式，详见[**文档资料**](./docs/index.md)
## 🛠️ 安装与环境配置 (Installation)

本项目使用 **[uv](https://github.com/astral-sh/uv)** 进行极速依赖管理。请确保已安装 uv，然后按照以下步骤操作：

### 1. 初始化环境
```bash
# 1. 克隆项目后进入目录
cd AI_Bell

# 2. 创建并同步环境 (自动安装 Python 3.12 及所有依赖)
uv sync
```

### 2. 激活虚拟环境
根据你的操作系统选择对应的命令：

*   **Windows (PowerShell/CMD)**:
    ```powershell
    .venv\Scripts\activate
    ```
*   **macOS / Linux**:
    ```bash
    source .venv/bin/activate
    ```

---

## 🔑 MOSEK 求解器配置 (至关重要)

本项目核心算法依赖 **MOSEK** 优化求解器。
> ⚠️ **注意**：`uv sync` 虽然安装了 Python 库，但你必须配置独立的 **许可证文件 (`mosek.lic`)**，否则程序会报错 `MOSEK error 1001`。

### 第一步：获取许可证
*   🎓 **学术用户 (学生/教职)**: [申请免费学术许可证](https://www.mosek.com/products/academic-licenses/) (需使用 .edu 邮箱，秒发)
*   💼 **商业/试用用户**: [申请 30 天免费试用](https://www.mosek.com/products/trial/)

### 第二步：放置许可证文件
收到 `mosek.lic` 后，请将其放入**系统用户主目录**下的 `mosek` 文件夹中：

| 操作系统 | 放置路径 | 备注 |
| :--- | :--- | :--- |
| **Windows** | `C:\Users\你的用户名\mosek\mosek.lic` | 如果文件夹不存在请新建 |
| **macOS/Linux** | `$HOME/mosek/mosek.lic` | 即 `~/mosek/mosek.lic` |

*(可选高级配置: 你也可以通过设置环境变量 `MOSEKLM_LICENSE_FILE` 指定自定义路径)*

### 第三步：验证安装
配置完成后，在项目根目录运行以下命令验证：
```bash
python -c "import mosek; env = mosek.Env(); print('✅ MOSEK 许可证验证成功！')"
```

---

## 🤝 远程协作指南 (Collaboration Guidelines)

为了维护代码库的稳定性，请团队成员严格遵守以下流程。

### 1. 分支策略 (Branching Strategy)
我们采用 GitHub Flow 模式。
*   🛡️ **main 分支受保护**：禁止直接 Push，必须通过 Pull Request (PR) 合并。
*   🌿 **分支命名规范**：
    *   `feat/xxx` : 新功能 (例如 `feat/login-page`)
    *   `fix/xxx` : Bug 修复 (例如 `fix/api-error`)
    *   `docs/xxx` : 文档修改
    *   `refactor/xxx` : 代码重构

### 2. 开发工作流 (Workflow)

1.  **同步主分支** (每日开工第一步):
    ```bash
    git checkout main
    git pull origin main
    ```
2.  **创建新分支**:
    ```bash
    git checkout -b feat/your-feature
    ```
3.  **开发与依赖变更**:
    *   如果需要增删包，**严禁**手动修改 `pyproject.toml`。
    *   请使用 `uv` 命令：
        ```bash
        uv add pandas          # 添加依赖
        uv remove numpy        # 移除依赖
        uv add --dev pytest    # 添加开发工具
        ```
4.  **提交代码与 PR**:
    *   `git push origin feat/your-feature`
    *   在 GitHub 页面发起 Pull Request 请求合并入 `main`。

### 3. 依赖与冲突管理 (重要)

*   **🔒 关于 `uv.lock`**:
    *   `uv.lock` 必须提交到 Git，它保证了所有人环境完全一致。
*   **🔄 拉取代码后**:
    *   如果同事更新了依赖，你拉取代码后必须运行 `uv sync` 来更新你的本地环境。
*   **💥 遇到 Lock 文件冲突**:
    1.  先解决 `pyproject.toml` 和代码文件的冲突。
    2.  **不要手动合并 `uv.lock`**。
    3.  直接运行 `uv lock` 重新生成锁文件。
    4.  提交变更。

## 📂 大文件与数据管理 (Data & Large Files)

Git 仓库**不适合**存储大文件。GitHub 对单文件超过 50MB 会警告，超过 100MB 会直接拒绝。为了保持仓库轻量化，请严格遵守以下规则：

### 1. 🛑 严禁提交的文件
在执行 `git add` 或 commit 之前，请务必确认**不包含**以下内容：

*   ❌ **虚拟环境目录**: `.venv/`, `env/` (这是最常见的错误，请确保已被 `.gitignore` 忽略)
*   ❌ **临时/中间数据**: `output/`, `temp_data/`, `*.csv` (除非是极小的测试集)
*   ❌ **IDE 配置文件**: `.idea/`, `.vscode/` (个人配置不应上传)
*   ❌ **编译产物**: `__pycache__/`, `build/`, `dist/`

> **提示**: 如果 PyCharm 提示文件过大，请右键该文件选择 **"Rollback"**，然后将其加入 `.gitignore`。

### 2. 📦 必要大文件处理 (Git LFS)
如果必须提交必要的 AI 模型 (`.pt`, `.onnx`) 或二进制数据：

**请务必使用 Git LFS (Large File Storage) 进行管理：**

1.  **安装 LFS** (仅需运行一次):
    ```bash
    git lfs install
    ```
2.  **追踪大文件类型**:
    ```bash
    # 例如追踪所有 .pt 模型文件
    git lfs track "*.pt"
    ```
3.  **提交配置**:
    这会生成一个 `.gitattributes` 文件，必须将其提交到 Git：
    ```bash
    git add .gitattributes
    ```

**注意**: 对于超过 100MB 的超大文件，建议上传至网盘或 NAS，并在文档中提供下载链接，**不要**强行存入 Git。




