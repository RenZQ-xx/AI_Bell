# D：封印父边的特殊selection

独立入口：run_class8_sealed_restart_paired.py；策略模块：sealed_restart.py。

每次完整重启清空本阶段父边Q与真实遍历计数、节点阶段visits，保存已有子边数B，并将过去访问过的父边封印。总visits、树结构及原bucket轮转保留。

- 阶段容量为max(6,K(阶段visits))，重启时立即6个名额。
- 已用名额为len(children)-B+特殊selection次数；新边即便复用旧闭包节点也消耗一个名额。
- 有名额、有未展开动作：原有expansion与rollout流程。
- 有名额、动作已经全部展开：只在封印父边中按探索项选择；既有高Q不参与竞争。选中后立即解封并消耗一个名额，沿子边继续原有树搜索。
- 名额不足：在所有子边中恢复普通selection。如果普通selection访问封印边，该边也解封，但不额外计作特殊selection。
- 若仍有名额但已无封印候选，正常selection，避免空候选或循环。
- 封印属于父边；共享闭包节点的一次访问不会替别的父节点解封。

与C相比唯一策略变化是上述特殊selection。D仍从既有的第100轮snapshot.pkl恢复，固定60轮，不因新class再次重启或延长，不重跑/覆盖ABC。

```bash
PYTHONPATH=src:src/mcts/rzq_ src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/test_sealed_restart.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  src/mcts/rzq_/.venv/bin/python src/mcts/rzq_/run_class8_sealed_restart_paired.py
```

输出为runs/class8_phase_restart_paired_D_60/；记录原快照SHA256、seed、配置、环境和源码指纹。特殊选择事件为sealed_selection，记录候选探索项、排除的动作、选择动作、前后名额使用量和剩余封印边。seal_end.json记录终点封印状态。

针对性测试覆盖：六个旧子边依次获得机会且不被前一条的大Q阻断；新扩展边与特殊selection共用名额；共享子节点不串联解封其他父边。测试不是新class发现能力实验。如果D的真实轨迹没有触发该条件，只能报告未覆盖，不能把C式扩展带来的结果归因于特殊selection。
