# Differentiable Convex Optimization Layers 



## 问题背景

在深度学习与物理学的交叉研究中，我们面临的核心矛盾是：神经网络擅长**搜索**，但无法直接感知**硬性物理约束**。

1. **归纳偏置的缺失**：传统的神经网络难以处理复杂的几何约束（如量子关联矩矩阵的半正定性 $\Gamma \succeq 0$）。
2. **“黑盒”优化的困境**：传统求解器（如 SCS, Mosek）的迭代过程不可微。这意味着梯度无法穿透物理约束边界，导致我们无法通过端到端的方式优化定义问题的参数（如 Bell 不等式的 $s$）。
3. **解决方案**：利用**隐函数微分**技术，将不可导的优化迭代过程转化为可导的线性微分方程，实现"Optimization as a Layer"。论文实现了一个算法库`cvxpylayers` , 可以直接调用完成该功能



## 方法

### 1. 优化问题的形式化定义
为了实现通用的微分，`cvxpylayers` 要求将所有凸优化问题转化为**标准锥规划（Canonical Cone Program）**形式：

$$\text{minimize} \quad c^T x$$
$$\text{subject to} \quad Ax = b$$
$$Gx + s = h, \quad s \in \mathcal{K}$$

* **优化变量 $x \in \mathbb{R}^n$**：是求解器最终给出的答案。
* **参数 $\theta$**：是该层接收的输入。它由 $\{c, A, b, G, h\}$ 组成，定义了问题的目标和边界。
* **凸锥 $\mathcal{K}$**：定义了不等式约束的性质。对于 **SDP（半正定规划）**，$\mathcal{K}$ 即为正定锥，要求 $h - Gx$ 对应的矩阵必须半正定。

### 2. 变量向量的来源
* **控制参数 $\theta$**：由神经网络产生，决定了优化问题的“形状”。
* **解向量 $z$**：由求解器算出，不仅包含原变量 $x$，还包含对偶变量（乘子） $\nu$ 和 $\lambda$。
  $$z = (x, \nu, \lambda)$$
  其中 $\nu$ 对应等式约束 $Ax=b$，$\lambda$ 对应锥约束 $Gx+s=h$。

### 3. KKT条件 $F(z, \theta) = 0$ 
$F(z, \theta) = 0$ 被称为**残差映射（Residual Map）**。它基于 **KKT 最优性条件**，是判定 $z$ 是否为针对参数 $\theta$ 的最优解的唯一准则。其具体方程组定义如下：



$$
F(z, \theta) = \begin{bmatrix} 
c + A^T \nu + G^T \lambda & \text{—— (1) 平稳性 (Stationarity)} \\
Ax - b & \text{—— (2) 原问题可行性 (Primal Feasibility)} \\
\Pi_{\mathcal{K}^*}(Gx - h + \lambda) - \lambda & \text{—— (3) 互补松弛性 (Complementary Slackness)}
\end{bmatrix} = \mathbf{0}
$$

* **方程 (1)**：规定了目标函数的拉力与约束的反作用力必须抵消。
* **方程 (3)**：这是处理 **SDP** 的核心。通过对偶锥投影算子 $\Pi_{\mathcal{K}^*}$，强制要求变量必须落在正定空间内。

### 4. 理论推导：方程组如何求导？
由于在最优解处 $F(z, \theta) = 0$ 恒成立

$$\frac{\partial F}{\partial z} \mathrm{d}z + \frac{\partial F}{\partial \theta} \mathrm{d}\theta = 0$$

整理得到**雅可比矩阵（导数）**：
$$\frac{\mathrm{d}z}{\mathrm{d}\theta} = - \left[ \frac{\partial F}{\partial z} \right]^{-1} \frac{\partial F}{\partial \theta}$$

* **$\frac{\partial F}{\partial z}$ (系统刚度矩阵)**：记录了方程组里每一个式子对每一个变量分量的偏导数。它捕捉了优化问题局部的几何曲率。
* **$\frac{\partial F}{\partial \theta}$ (外部冲击矩阵)**：记录了神经网络通过修改 $\theta$ 带来的直接扰动。

通过求解这个线性系统，梯度得以从解 $z$ 成功回传到参数 $\theta$。

---



## 与寻找最优 Bell 不等式的联系

将此理论应用于寻找最优 Bell 不等式向量 $s$ 的任务中，其逻辑闭环如下：

### 1. 物理问题与形式化参数的映射
* **$c \leftarrow s$**：神经网络生成的 Bell 不等式系数 $s$ 映射为目标系数 $c$。这决定了我们想让量子关联向哪个方向“违背”。
* **$\mathcal{K} \leftarrow$ 正定锥**：利用 NPA Hierarchy 构造矩矩阵 $\Gamma$。方程组 $F$ 中的投影项 $\Pi$ 强制要求 $\Gamma \succeq 0$。这保证了产出的关联 $v$ 一定具有量子实现。

### 2. 梯度引导的边界搜索
* **穿透边界**：当神经网络通过 Loss 函数尝试提升违背值时，反向传播会触发。
* **感知几何**：导数 $\frac{\mathrm{d}z}{\mathrm{d}\theta}$ 实际上告诉了神经网络：**“如果你想增加量子违背，你应该如何微调 $s$ 的分量，使得最优量子点 $v^*$ 能够沿着正定锥的弯曲边界滑向更远的地方。”**

### 3. 核心优势
* **解析级精度**：不同于随机搜索，该梯度是基于 KKT 平衡状态的解析求导，搜索效率极高。
* **自动演化**：通过多次迭代，神经网络可以自动学习到量子多胞形（Quantum Set）的精细边缘结构，从而发现具有最大量子违背的新型 Bell 不等式。
