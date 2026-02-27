
## 0. 基本信息

- `尝试ID`: 01
- `标题`: MLP拟合Q(s)
- `日期`: 2025.12.25
- `负责人`: `Ren`
- `状态`: `archived`

## 1. 尝试分类（核心标识）

- `问题场景`: `222` 
- `方法类型`: `MLP拟合` 
- `目标类型`: `量子边界拟合`

## 2. 目标与假设

- `研究问题`: 在222场景下，用神经网络拟合Q(s)
- `核心假设`: 222作为已知的小场景，状态空间小，数据量充足
- `成功标准（可量化）`: 
  - 指标1: 预测值与NPA求解值的残差
  - 指标2: 
  - 指标3:

## 3. 代码与数据定位

- `主入口脚本`: [../../../src/experiments/222/mlp_Qs/main.py](../../../src/experiments/222/mlp_Qs/main.py)

- `关键模块文件`:
  - [../../../src/experiments/222/mlp_Qs/data_processing.py](../../../src/experiments/222/mlp_Qs/data_processing.py)
  - [../../../src/experiments/222/mlp_Qs/model.py](../../../src/experiments/222/mlp_Qs/model.py)
  - [../../../src/experiments/222/mlp_Qs/predict.py](../../../src/experiments/222/mlp_Qs/predict.py)
  - [../../../src/experiments/222/mlp_Qs/data_generator.py](../../../src/experiments/222/mlp_Qs/data_generator.py)
- `依赖工具/求解器`: `MOSEK`, `PyTorch`
- `数据来源`: [../../../data/data_222.npz](../../../data/data_222.npz) 由 [../../../src/experiments/222/mlp_Qs/data_generator.py](../../../src/experiments/222/mlp_Qs/data_generator.py) 生成
- `输出产物`: 
  - 模型/权重: ../args
  - 日志: 
  - 图表: 
  - 中间结果: 

## 4. 实验配置

- `硬件环境`: CPU
- `软件环境`: Python
- `随机种子`:
- `关键超参数`:
  - 
  - 
- `训练/求解预算`: 如 episode 数、时长、迭代次数

## 5. 实验流程（可复现）

1. `准备`:
2. `运行命令`: [主脚本](../../../src/experiments/222/mlp_Qs/main.py)
3. `结果收集`: 
4. `结果分析`: 拟合准确度符合预期

## 6. 结果记录

- `主要结果（定量）`:
- `对比基线`:
- `典型可视化/样例`:
- `是否达到成功标准`: `是` 

## 7. 问题与风险

- `当前问题`: 1.针对数据量充足的222小场景好用，对数据量不充足的更大场景还有用吗？2.NPA求解器的速度在4-2-2场景下也尚且能够接受(0.1s per step),那对于Q(s)的拟合还有必要吗？
- `失败模式`:
- `潜在原因`:
- `风险等级`: 

## 8. 结论与下一步

- `结论（一句话）`: 仅作尝试，不继续深入
- `建议保留内容`:
- `建议归档内容`: 仅py脚本
- `下一步动作（最多3条）`:
  1. 
  2. 
  3. 

## 9. 重构映射（给 step2 用）

- `建议归属目录`: `experiments/222/mlp_Qs/`
- `是否需要保留兼容入口`: `否`
- `重命名建议`:
  - [../../../src/experiments/222/mlp_Qs/data_processing.py](../../../src/experiments/222/mlp_Qs/data_processing.py)
  - [../../../src/experiments/222/mlp_Qs/model.py](../../../src/experiments/222/mlp_Qs/model.py)
  - [../../../src/experiments/222/mlp_Qs/predict.py](../../../src/experiments/222/mlp_Qs/predict.py)
  - [../../../src/experiments/222/mlp_Qs/data_generator.py](../../../src/experiments/222/mlp_Qs/data_generator.py)

- `测试需求`: 无
  - 
  - 

---

## 附：当前项目可先登记的尝试清单（待你补全）

1. `MLP 拟合 222 量子边界`
2. `DQN 学习 222 场景边界搜索策略`
3. `DQN 学习 322 场景边界搜索策略`
4. `使用 lrs 枚举 322 全部边界`
