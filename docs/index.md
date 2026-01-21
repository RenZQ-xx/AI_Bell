#  项目文档索引 (Documentation Index)

欢迎查阅 **AI Bell** 的文档库。

---

##  重要参考
1. 问题背景，见[**RMP 86**](./Bell nonlocality.pdf)
2. Q(s)，C(s)的[**计算方法**](https://mixed-polonium-133.notion.site/413d65c7cf194658994fabaf22aa9db6?source=copy_link)
3. [**2-2-2场景**](https://mixed-polonium-133.notion.site/2-2-2-2e8ec9f8b932806fb86ccd3bbb9d5947?source=copy_link)的定义和解





---

##  2026-1-5 会议纪要

文献分享

| 文档名称                              |                                                      原文                                                      |  讲解   | 
|:----------------------------------|:------------------------------------------------------------------------------------------------------------:|:-----:|
| [**并行符号回归**](./2026-1-5-rzq.pptx) |[**Nat Comput Sci (2025)**](./Discovering physical laws with parallel symbolic enumeration.pdf)     |  Ren  | 
| [**Diffusion组合优化**]()             | [**arXiv:2406.01661**](./A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization.pdf) | Jiang | 
| [**RL组合优化**]()                    |         [**arXiv:1611.09940**](./Neural Combinatorial Optimization with Reinforcement Learning.pdf)          | Yang  | 
下一步计划：
1. 补充Q(s)计算方法，重新整理问题定义
2. 用遍历方法先找出2-2-2场景的CHSH不等式


---
##  2026-1-9 会议纪要

补充了[**计算方法**](https://mixed-polonium-133.notion.site/413d65c7cf194658994fabaf22aa9db6?source=copy_link)
文档，讨论问题定义

下一步计划：
1. 聚焦问题：Q(s)的拟合
2. 在2-2-2场景中，用强化学习的思路训练网络 

   input：向量s. 

    output: 算符A,B,..
---
##  2026-1-13 会议纪要

文献分享，确定计划

| 文档名称                                                                                                                                               |                                                         原文                                                         | 讲解  | 
|:---------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------:|:---:|
| [**可微凸优化**](./Differentiable Convex Optimization Layers note.md)                                                                                                                   |  [**arXiv:1910.12430**](./Differentiable Convex Optimization Layers.pdf)   | Li  | 
| [**机器学习判别非定域性**](https://mixed-polonium-133.notion.site/Machine-learning-non-local-correlations-2e3ec9f8b93280e8b19cc7799e048da4?source=copy_link) |                       [**arXiv:1808.07069**](./Machine learning non-local correlations.pdf)                        | Ren |
聚焦两个问题：
1. Q(s)的拟合问题
2. 给定顶点集，强化学习寻找最佳顶点组合

---
## 2026-1-20 会议纪要 

补充了2-2-2场景的思路，并讨论初步结果

- 思路：

用强化学习，定义动作为顶点群变换操作，状态为当前所取顶点组合。详见[**2-2-2场景**](https://mixed-polonium-133.notion.site/2-2-2-2e8ec9f8b932806fb86ccd3bbb9d5947?source=copy_link)


 - 结果与问题：

用DQN找到了5个CHSH不等式，但缺少停止信号，无法在最优处收敛

 - 下一步计划：
1. 暂时不考虑Q(s)的拟合
   (因为3-2-2和2-2-2的groundtruth求解都在ms量级)
2. 设计更好的action，让2-2-2可以在CHSH不等式处停下或收敛