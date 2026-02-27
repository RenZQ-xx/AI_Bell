

## 0. 基本信息

- `尝试ID`: 05
- `标题`: lrs遍历322场景所有边界
- `日期`: 2026-02-06
- `负责人`: `Ren`
- `状态`:  `archived`
- `优先级`: 

## 1. 尝试分类（核心标识）

- `问题场景`: `322` 
- `方法类型`: `LRS枚举` 
- `目标类型`: `全边界枚举`

## 2. 目标与假设

- `研究问题`: 在322场景下，借助lrs枚举所有边界
- `核心假设`:
- `成功标准（可量化）`:
  - 指标1:
  - 指标2:
  - 指标3:

## 3. 代码与数据定位

- `主入口脚本`: [../../../src/experiments/322/lrs/Solver_lrs.py](../../../src/experiments/322/lrs/Solver_lrs.py)
- `关键模块文件`:
  - [../../../src/experiments/322/lrs/gen_polytope_ext.py](../../../src/experiments/322/lrs/gen_polytope_ext.py)
  - 
- `依赖工具/求解器`: `lrs`
- `数据来源`: 由[../../../src/experiments/322/lrs/gen_polytope_ext.py](../../../src/experiments/322/lrs/gen_polytope_ext.py)生成的[../../../data/polytope.ext](../../../data/polytope.ext)
- `输出产物`: 
  - 文件: [../../../data/facets.txt](../../../data/facets.txt)
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
2. `运行命令`:
3. `结果收集`:
4. `结果分析`:

## 6. 结果记录

- `主要结果（定量）`: 成功求出所有边界
- `对比基线`:
- `典型可视化/样例`:
- `是否达到成功标准`: `部分达到`

## 7. 问题与风险

- `当前问题`:
- `失败模式`:
- `潜在原因`:
- `风险等级`: 

## 8. 结论与下一步

- `结论（一句话）`: 成功求出所有边界，还需进一步做归类
- `建议保留内容`:
- `建议归档内容`:
- `下一步动作（最多3条）`:
  1. 归类
  2. 
  3. 

## 9. 重构映射（给 step2 用）

- `建议归属目录`: `experiments/322/lrs/`
- `是否需要保留兼容入口`:  `否`
- `重命名建议`:
  - [../../../src/experiments/322/lrs/Solver_lrs.py](../../../src/experiments/322/lrs/Solver_lrs.py)
  - [../../../src/experiments/322/lrs/gen_polytope_ext.py](../../../src/experiments/322/lrs/gen_polytope_ext.py)

- `测试需求`:
  - 
  - 

---

## 附：当前项目可先登记的尝试清单（待你补全）

1. `MLP 拟合 222 量子边界`
2. `DQN 学习 222 场景边界搜索策略`
3. `DQN 学习 322 场景边界搜索策略`
4. `使用 lrs 枚举 322 全部边界`
