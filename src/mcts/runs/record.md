# MCTS Runs

和 Class 7 在同一个 pattern 下 的 Class 11 和 Class 22 没有被搜索到，进一步扩大搜索范围有可能能找到这两个解。
以下记录了`interrupt_search.py` 在 $\text{iteration} = 1000$时的工作栈，栈顶元素即为当前搜索状态：

| Search Index | Start Class id | Find Class | Stack Trace | Notes | Results |
| --- | --- | --- | --- | --- | --- |
| 1 | 7 | 44 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 44 搜索 | |
| 2 | 44 | ~ | <44, 7> | Class 44 搜索完成，回溯到 Class 7 | "exact:class44": 94, "invalid:non_coplanar": 906 |
| 1 | 7 | 43 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 43 搜索 | |
| 3 | 43 | ~ | <43, 7> | Class 43 搜索完成，回溯到 Class 7 | "exact:class43": 401 |
| 1 | 7 | 29 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 29 搜索 | |
| 4 | 29 | 42 | <29, 7> | 保存 Class 29 搜索状态，下一步开始从 Class 42 搜索 | |
| 5 | 42 | 46 | <42, 29, 7> | 保存 Class 42 搜索状态，下一步开始从 Class 46 搜索 | |
| 6 | 46 | ~ | <46, 42, 29, 7> | Class 46 搜索完成，回溯到 Class 42 | "exact:class46": 401 |
| 5 | 42 | ~ | <42, 29, 7> | Class 42 搜索完成，回溯到 Class 29 | "exact:class42": 401, "exact:class46": 115 |
| 4 | 29 | 38 | <29, 7> | 保存 Class 29 搜索状态，下一步开始从 Class 38 搜索 | |
| 7 | 38 | ~ | <38, 29, 7> | Class 38 搜索完成，回溯到 Class 29 | "exact:class38": 160, "exact:class44": 105, "exact:class46": 347, "invalid:non_coplanar": 388 |
| 4 | 29 | ~ | <29, 7> | Class 29 搜索完成，回溯到 Class 7 | "exact:class28": 33, "exact:class29": 372, "exact:class30": 11, "exact:class31": 3, "exact:class38": 8, "exact:class42": 116, "exact:class44": 227, "exact:class46": 208, "invalid:non_coplanar": 22 |
| 1 | 7 | 35 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 35 搜索 | |
| 8 | 35 | ~ | <35, 7> | Class 35 搜索完成，回溯到 Class 7 | "exact:class35": 401, "exact:class45": 98, "exact:class46": 51 |
| 1 | 7 | 25 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 25 搜索 | |
| 9 | 25 | 40 | <25, 7> | 保存 Class 25 搜索状态，下一步开始从 Class 40 搜索 | |
| 10 | 40 | ~ | <40, 25, 7> | Class 40 搜索完成，回溯到 Class 25 | "exact:class40": 401 |
| 9 | 25 | ~ | <25, 7> | Class 25 搜索完成，回溯到 Class 7 | "exact:class25": 401, "exact:class40": 132, "invalid:boundary": 398, "invalid:non_coplanar": 45 |
| 1 | 7 | 39 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 39 搜索 | |
| 11 | 39 | ~ | <39, 7> | Class 39 搜索完成，回溯到 Class 7 | "exact:class39": 401 |
| 1 | 7 | 34 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 34 搜索 | |
| 12 | 34 | ~ | <34, 7> | Class 34 搜索完成，回溯到 Class 7 | "exact:class34": 401, "invalid:boundary": 189, "invalid:non_coplanar": 24 |
| 1 | 7 | ~ | <7> | Class 7 搜索完成 | "exact:class10": 1, "exact:class12": 1, "exact:class15": 7, "exact:class19": 4, "exact:class20": 19, "exact:class25": 4, "exact:class28": 12, "exact:class29": 155, "exact:class30": 1, "exact:class31": 1, "exact:class32": 5, "exact:class34": 1, "exact:class35": 103, "exact:class39": 2, "exact:class40": 1, "exact:class42": 42, "exact:class43": 255, "exact:class44": 342, "exact:class45": 9, "exact:class46": 17, "exact:class7": 13, "exact:class8": 2, "exact:class9": 1 |
## 2026-07-06 interrupt_search adaptive i200

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_adaptive.json`
- `src/mcts/runs/interrupt_search_probe_i200_adaptive.runs.json`

Settings:

- Initial class: 7
- Pattern index: 0
- Base iterations: 200
- Max depth: 26
- Adaptive sink stop: min 50 iterations, min 20 terminal hits, self-or-invalid ratio >= 0.9
- Adaptive budget cap used in this run: 3 * iterations = 600

Classes still not found in this run: 9, 10, 11, 22. These four classes are in the same pattern family as class 7, so the next run should keep more budget on class 7 and its non-terminal descendants.

Stack trace, top of stack first:

| Search Index | Start Class id | Find / Interrupt | Stack Trace | Stop / Budget | Results |
| --- | --- | --- | --- | --- | --- |
| 1 | 7 | 44 | <7> | suspend class 7, push class 44 | |
| 2 | 44 | ~ | <44, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class44": 6, "invalid:non_coplanar": 44` |
| 1 | 7 | 29 | <7> | resume class 7, push class 29 | |
| 3 | 29 | 46 | <29, 7> | suspend class 29, push class 46 | |
| 4 | 46 | ~ | <46, 29, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class46": 50` |
| 3 | 29 | 42 | <29, 7> | resume class 29, push class 42 | |
| 5 | 42 | ~ | <42, 29, 7> | iterations_exhausted, 300 / 300 | `"exact:class42": 233, "exact:class46": 54, "invalid:boundary": 11, "invalid:non_coplanar": 2` |
| 3 | 29 | 38 | <29, 7> | resume class 29, push class 38 | |
| 6 | 38 | ~ | <38, 29, 7> | iterations_exhausted, 400 / 400 | `"exact:class38": 69, "exact:class44": 41, "exact:class46": 131, "invalid:boundary": 16, "invalid:non_coplanar": 143` |
| 3 | 29 | ~ | <29, 7> | iterations_exhausted, 600 / 600 | `"exact:class28": 26, "exact:class29": 191, "exact:class30": 15, "exact:class31": 2, "exact:class38": 5, "exact:class42": 82, "exact:class44": 165, "exact:class46": 75, "invalid:non_coplanar": 39` |
| 1 | 7 | 35 | <7> | resume class 7, push class 35 | |
| 7 | 35 | ~ | <35, 7> | exact_frequency_threshold, 363 / 400 | `"exact:class35": 241, "exact:class45": 47, "exact:class46": 46, "invalid:boundary": 7, "invalid:non_coplanar": 22` |
| 1 | 7 | 43 | <7> | resume class 7, push class 43 | |
| 8 | 43 | ~ | <43, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class43": 50` |
| 1 | 7 | 39 | <7> | resume class 7, push class 39 | |
| 9 | 39 | ~ | <39, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class39": 36, "invalid:boundary": 2, "invalid:non_coplanar": 12` |
| 1 | 7 | 25 | <7> | resume class 7, push class 25 | |
| 10 | 25 | 40 | <25, 7> | suspend class 25, push class 40 | |
| 11 | 40 | ~ | <40, 25, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class40": 42, "invalid:non_coplanar": 8` |
| 10 | 25 | ~ | <25, 7> | iterations_exhausted, 300 / 300 | `"exact:class25": 101, "exact:class40": 36, "invalid:boundary": 131, "invalid:non_coplanar": 32` |
| 1 | 7 | 34 | <7> | resume class 7, push class 34 | |
| 12 | 34 | ~ | <34, 7> | self_or_invalid_sink, 50 / 200 | `"exact:class34": 27, "invalid:boundary": 18, "invalid:non_coplanar": 5` |
| 1 | 7 | ~ | <7> | iterations_exhausted, 600 / 600 | `"exact:class12": 1, "exact:class15": 5, "exact:class19": 1, "exact:class20": 12, "exact:class25": 2, "exact:class28": 5, "exact:class29": 114, "exact:class30": 1, "exact:class32": 2, "exact:class34": 2, "exact:class35": 53, "exact:class39": 1, "exact:class40": 1, "exact:class42": 21, "exact:class43": 128, "exact:class44": 233, "exact:class45": 2, "exact:class46": 7, "exact:class7": 7, "exact:class8": 1` |

## 2026-07-29 interrupt_search cached reward i200

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_cached_reward.json`
- `src/mcts/runs/interrupt_search_probe_i200_cached_reward.runs.json`

Settings:

- Initial class: 7
- Pattern index: 0
- Base iterations: 200
- Max depth: 26
- Target and rare classes: 1 through 46
- Adaptive budget cap: 5 * iterations = 1000

Timing:

- Started: 2026-07-29 14:24:33.587
- Finished: 2026-07-29 14:58:39.410
- Stopwatch elapsed: 2045.816 seconds (00:34:05.816)
- Previous strict-global-reward i200 elapsed: 1370.867 seconds (00:22:50.867)
- Full i200 regression: +674.949 seconds (+49.2%)

Results:

- Stop reason: `stack_empty`
- Completed tasks: 12 / 12
- Total iterations: 3600
- Total nodes: 2057
- Exact class coverage: 23
- Found classes: 7, 8, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Classes 9, 10, and 11 remain missing from the class-7 pattern family.
- Class 22 was found at global iteration 3331, class-7 local iteration 731, depth 17.

Performance note:

The cached design reduced the fixed i10 probe from 104.538 seconds to 51.863 seconds, but the complete i200 run was slower than the previous strict-global-reward run. Coverage, total iterations, discovery timing for class 22, and almost all per-task node counts were unchanged. The short probe improvement therefore does not generalize to the full search and should not be treated as an i200 speedup.

## 2026-08-07 interrupt_search lazy structure cache i200

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_lazy_structure_cache.json`
- `src/mcts/runs/interrupt_search_probe_i200_lazy_structure_cache.runs.json`

Settings:

- Initial class: 7
- Pattern index: 0
- Base iterations: 200
- Max depth: 26
- Target and rare classes: 1 through 46
- Adaptive budget cap: 5 * iterations = 1000
- Lazy widening score batch: 4
- Rollout score batch: 8

Timing:

- Started: 2026-08-07 10:10:00.210
- Finished: 2026-08-07 10:13:49.495
- Stopwatch elapsed: 229.283 seconds (00:03:49.283)
- Previous cached-reward i200 elapsed: 2045.816 seconds (00:34:05.816)
- Wall-clock reduction: 1816.533 seconds (-88.8%, 8.92x faster)

Workload and normalized timing:

- Completed tasks: 9 / 9, compared with 12 / 12 previously
- Total iterations: 2739, compared with 3600 previously
- Total nodes: 1938, compared with 2057 previously
- Current elapsed per iteration: 0.083710 seconds
- Previous elapsed per iteration: 0.568282 seconds
- Iteration-normalized speedup: 6.79x
- The 8.92x wall-clock speedup is not a pure implementation comparison because lazy widening changed the search trajectory and reduced both task count and total iterations.

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 17
- Found classes: 7, 8, 11, 15, 20, 28, 29, 30, 31, 34, 35, 38, 42, 43, 44, 45, 46
- Classes 9, 10, and 22 remain missing from the class-7 pattern family.
- Class 11 was found at global iteration 2691, class-7 local iteration 492, depth 17.

## 2026-08-07 interrupt_search lazy i200 without frequency stop

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_lazy_no_frequency_stop.json`
- `src/mcts/runs/interrupt_search_probe_i200_lazy_no_frequency_stop.runs.json`

Settings:

- Same settings and seed as the preceding lazy structure-cache i200 run
- Initial class: 7
- Base iterations: 200
- Max depth: 26
- Adaptive budget cap: 5 * iterations = 1000
- `exact_stop_ratio`: 1.0, which disables `exact_frequency_threshold`
- The 50-iteration `self_or_invalid_sink` check remained enabled

Timing:

- Started: 2026-08-07 10:32:54.034
- Finished: 2026-08-07 10:38:58.835
- Stopwatch elapsed: 364.799 seconds (00:06:04.799)
- Added time versus the early-stop lazy run: 135.516 seconds
- Previous cached-reward i200 elapsed: 2045.816 seconds
- Wall-clock speedup versus cached reward: 5.61x

Workload:

- Completed tasks: 11 / 11
- Total iterations: 3300
- Total nodes: 2469
- Elapsed per iteration: 0.110545 seconds
- Iteration-normalized speedup versus cached reward: 5.14x
- No task stopped with `exact_frequency_threshold`
- Class 7 stopped with `iterations_exhausted` at 1000 / 1000 iterations

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 22
- Found classes: 7, 8, 10, 11, 12, 15, 20, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Classes 9, 19, 22, and 25 remain missing from this run; among the specifically tracked class-7 pattern classes, 9 and 22 are still missing.
- Relative to the early-stop lazy run, the additional search recovered classes 10, 12, 32, 39, and 40 without losing any previously found class.
- Late class-7 discoveries after the old stop point at iteration 540: class 40 at 584, class 39 at 642, class 45 at 677, class 12 at 808, class 10 at 812, and class 32 at 821.
- Relative to the old 23-class cached-reward run, this run additionally found classes 10 and 11 but did not find classes 19, 22, and 25.

## 2026-08-07 interrupt_search productive-weighted i200

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted.json`
- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted.runs.json`

Settings:

- Initial class: 7
- Base iterations: 200
- Max depth: 26
- Productive budget: base + 100 * distinct cross classes, capped at 1000
- Widening score batch: 4 + floor(sqrt(node visits)), capped at 16
- Productive rollout score batch: 16
- Broad rollout: 24 candidates every 8 actual rollouts
- Productive novelty patience: 400 iterations
- The original 50-iteration non-prior sink test remained enabled for non-productive tasks

Timing:

- Started: 2026-08-07 11:07:24.369
- Finished: 2026-08-07 11:18:42.567
- Stopwatch elapsed: 678.195 seconds (00:11:18.195)
- Elapsed per iteration: 0.215986 seconds
- Per-iteration cost versus the fixed-8 rollout run: +95.4%
- Previous cached-reward i200 elapsed: 2045.816 seconds
- Wall-clock speedup versus cached reward: 3.02x
- Iteration-normalized speedup versus cached reward: 2.63x

Workload and budget allocation:

- Completed tasks: 12 / 12
- Total iterations: 3140
- Total nodes: 2333
- Class 7: 22 cross classes, 1000 / 1000 iterations
- Class 35: 2 cross classes, 400 / 400 iterations
- Class 38: 2 cross classes, 400 / 400 iterations
- Class 42: 1 cross class, 300 / 300 iterations
- Class 25: 1 cross class, 300 / 300 iterations
- Class 29: 7 cross classes, budget 900; stopped at 440 by productive novelty patience
- Productive tasks stopped by sink or exact-frequency rules: 0

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 24
- Found classes: 7, 8, 9, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- This is a strict superset of the old 23-class cached-reward result, adding class 9 without losing any old class.
- Class 9 was found at global iteration 2253, class-7 local iteration 113, depth 17.
- Class 22 was found at global iteration 2500, class-7 local iteration 360, depth 17.
- Among the specifically tracked class-7 pattern classes 9, 10, 11, and 22, this run found 9 and 22; classes 10 and 11 remain missing.

## 2026-08-07 interrupt_search portfolio-survival i200

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_portfolio_survival.json`
- `src/mcts/runs/interrupt_search_probe_i200_portfolio_survival.runs.json`

Settings:

- Initial class: 7
- Base iterations: 200; adaptive cap: 1000
- Max depth: 26
- Progressive widening uses 7 buckets, including undiscovered-class compatibility survival
- Rollout portfolio per 8 actual rollouts: 4 narrow batch-8/top-4, 3 medium batch-16/top-4, 1 broad batch-24/top-8 plus one low-aggression tail action
- Narrow, medium, and broad use independent RNG streams
- Productive novelty patience: 400 iterations

Timing:

- Started: 2026-08-07 12:03:35.296
- Finished: 2026-08-07 12:12:35.168
- Stopwatch elapsed: 539.872 seconds (00:08:59.872)
- Elapsed per iteration: 0.170253 seconds
- Versus productive-weighted i200: 20.4% lower wall time and 21.2% lower time per iteration
- Versus cached-reward i200: 3.79x wall-clock speedup

Workload and portfolio:

- Completed tasks: 12 / 12
- Total iterations: 3171
- Total nodes: 2365
- Actual rollouts: 2454 = 1234 narrow, 917 medium, 303 broad
- Class 7: 16 cross classes, 1000 / 1000 iterations; exact portfolio split 500 / 375 / 125
- Class 29: 7 cross classes; stopped at 471 by productive novelty patience
- Stop reasons: 6 self-or-invalid sinks, 5 iteration exhaustion, 1 productive novelty patience

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 21
- Found classes: 7, 8, 15, 19, 20, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Missing from the known 26-class union: 9, 10, 11, 12, and 22
- Class 10 and class 11 had zero terminal encounters
- Relative to productive-weighted i200, this run added no class and lost classes 9, 12, and 22; coverage fell from 24 to 21 despite similar total iterations
- The portfolio schedule and undiscovered-survival bucket improved runtime for this trajectory, but this single seed did not improve rare-class coverage

## 2026-08-07 productive-weighted i200 deterministic rerun

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted_rerun.json`
- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted_rerun.runs.json`

Timing:

- Started: 2026-08-07 12:27:00.834
- Finished: 2026-08-07 12:38:19.268
- Stopwatch elapsed: 678.434 seconds (00:11:18.434)
- Historical productive-weighted elapsed: 678.195 seconds
- Timing difference: 0.239 seconds (0.035%)

Reproduction checks:

- Exact class coverage: 24
- Found classes: 7, 8, 9, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Completed tasks: 12; total iterations: 3140; total nodes: 2333
- Discovery timeline, interrupt events, per-task runs, and aggregate summary are identical after JSON normalization to the historical 24-class result
- Class 9 reproduced at global iteration 2253, class-7 local iteration 113, depth 17
- Class 22 reproduced at global iteration 2500, class-7 local iteration 360, depth 17
- Classes 10 and 11 remain missing

## 2026-08-07 interrupt_search class1 i200 cap2000

Result files:

- `src/mcts/runs/interrupt_search_class1_i200_cap2000.json`
- `src/mcts/runs/interrupt_search_class1_i200_cap2000.runs.json`

Settings:

- Initial class: 1
- Base iterations: 200
- Max depth: 26
- Adaptive extra iterations: 100 per distinct cross class
- Run-level adaptive max multiplier: 10, giving a per-task hard cap of 2000
- Productive novelty patience: 400 iterations
- The code default multiplier remains unchanged at 5

Timing and workload:

- Started: 2026-08-07 12:47:14.873
- Finished: 2026-08-07 13:34:49.260
- Stopwatch elapsed: 2854.387 seconds (00:47:34.387)
- Completed tasks: 15 / 15
- Total iterations: 4721
- Total nodes: 3915
- Elapsed per iteration: 0.604615 seconds
- Stop reasons: 7 self-or-invalid sinks, 4 iteration exhaustion, 4 productive novelty patience

Budget use:

- Class 1: 20 cross classes, limit 2000; stopped at 1104 after its last local discovery at 704
- Class 15: 24 cross classes, limit 2000; stopped at 1004 after its last local discovery at 604
- Class 23: 8 cross classes, limit 1000; stopped at 448 by novelty patience
- No exact class was discovered at a local iteration greater than 1000
- Relative to a 1000 hard cap on this same deterministic trajectory, the larger cap consumed 108 additional no-discovery iterations and did not add coverage

Results:

- Stop reason: `stack_empty`
- Exact terminal coverage: 36
- Found classes: 2, 4, 5, 7, 8, 9, 10, 12, 14, 15, 17, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
- Missing class IDs: 1, 3, 6, 11, 13, 16, 18, 24, 26, 33
- The starting class 1 is not counted automatically and had zero exact terminal encounters
- Class 10 was first found by the class-15 task at global iteration 2864, local iteration 196, depth 17
- Class 11 had zero terminal encounters

## 2026-08-07 productive-weighted i200 post-rollback rerun

Result files:

- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted_post_rollback.json`
- `src/mcts/runs/interrupt_search_probe_i200_productive_weighted_post_rollback.runs.json`

Timing and workload:

- Started: 2026-08-07 20:13:35.543
- Finished: 2026-08-07 20:38:26.000
- Stopwatch elapsed: 1490.453 seconds (00:24:50.453)
- Completed tasks: 12 / 12
- Total iterations: 3140
- Total nodes: 2333
- Elapsed per iteration: 0.474666 seconds
- Stop reasons: 6 self-or-invalid sinks, 5 iteration exhaustion, 1 productive novelty patience

Results and reproduction checks:

- Stop reason: `stack_empty`
- Exact class coverage: 24
- Found classes: 7, 8, 9, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Metadata, started/finished class IDs, interrupt events, discovery timeline, summary, exact counts, and all per-task runs are identical after JSON normalization to `interrupt_search_probe_i200_productive_weighted_rerun`
- Class 9 reproduced at global iteration 2253, class-7 local iteration 113
- Class 22 reproduced at global iteration 2500, class-7 local iteration 360
- Class 8 reproduced at global iteration 2580, class-7 local iteration 440
- Historical rerun elapsed: 678.434 seconds; this run was 812.019 seconds slower and took 2.197x as long despite identical work and trajectory

## 2026-08-09 paper-inspired Filler-Corrector class7 i200

Result files:

- `src/mcts/runs/interrupt_search_filler_corrector_i200.json`
- `src/mcts/runs/interrupt_search_filler_corrector_i200.runs.json`

Method:

- Existing interruptible MCTS is the Filler; all tasks retain the shared global discovered set, structural cache, and terminal validation cache
- Corrector protects a leading rank-8 block core, removes one non-core block from an exact terminal state, and requires the corrected prefix to have rank 23 or 24
- Candidate lookahead rotates across source classes, evaluates at most 12 corrected prefixes, and prioritizes one-step exits to globally undiscovered classes
- Each remove action gets an independently seeded 48-iteration Filler repack window; the original tree and RNG state are restored afterward
- Productive tasks permit up to three correction windows, while an otherwise stopping sink task permits one

Timing and workload:

- Elapsed: 640.665 seconds (00:10:40.665)
- Completed tasks: 12 / 12
- Total iterations: 3332; total nodes: 2097
- Stop reasons: 6 self-or-invalid sinks, 5 iteration exhaustion, 1 productive novelty patience
- Baseline productive-weighted rerun: 678.434 seconds, 3140 iterations, and 2333 nodes
- This run used 192 more iterations and 236 fewer nodes; it was 37.769 seconds (5.57%) faster than that baseline run, although prior identical-trajectory reruns show substantial wall-time variance

Corrector transitions:

- 15 remove-and-repack transitions across tasks 7, 25, 29, 34, 35, 38, 40, 42, 43, and 44
- 720 iterations were spent in Corrector repack windows; four sink-task windows account for the 192-iteration increase over baseline
- 9 / 15 transitions produced a larger terminal support; mean size delta was +2.267 vertices
- No transition discovered a globally new class, and all selected candidates had `entrance_rare_count=0`
- Class 7 corrections triggered at local iterations 230, 385, and 616

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 24, identical to the productive-weighted baseline
- Found classes: 7, 8, 9, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Added classes relative to baseline: none; lost classes: none
- Classes 10 and 11 remain missing
- Class 22 moved from class-7 local iteration 360 to 456, and class 8 moved from 440 to 536, exactly reflecting the first two 48-iteration correction windows before those discoveries

## 2026-08-09 paper-inspired Filler-Corrector class1 i200 cap2000

Result files:

- `src/mcts/runs/interrupt_search_filler_corrector_class1_i200_cap2000.json`
- `src/mcts/runs/interrupt_search_filler_corrector_class1_i200_cap2000.runs.json`

Settings:

- Initial class: 1
- Base iterations: 200
- Max depth: 26
- Adaptive max multiplier: 10, giving a per-task hard cap of 2000
- Corrector settings are unchanged from the class7 experiment: protected rank 8, corrected rank 23/24, one removed block, 12-candidate rotating shortlist, and 48 Filler iterations per repack window
- Baseline comparison: `interrupt_search_class1_i200_cap2000.json`

Timing and workload:

- Elapsed: 4571.636 seconds (01:16:11.636)
- Completed tasks: 17 / 17
- Total iterations: 6322; total nodes: 4692
- Elapsed per iteration: 0.723131 seconds
- Stop reasons: 7 self-or-invalid sinks, 5 iteration exhaustion, 5 productive novelty patience
- Historical class1 baseline: 2854.387 seconds, 15 tasks, 4721 iterations, and 3915 nodes
- This run used 1601 more iterations and 777 more nodes, and took 1717.249 seconds longer (1.602x baseline)
- The observed Python worker working set reached approximately 6.0 GB during the final Corrector validation phase

Corrector transitions:

- 25 remove-and-repack transitions across tasks 1, 13, 15, 23, 25, 27, 29, 34, 35, 38, 40, 42, 43, and 44
- 11 / 25 transitions produced a larger terminal support; mean size delta was +1.72 vertices
- No transition directly discovered a globally new class, and every selected candidate had `entrance_rare_count=0`
- Class 1 corrections triggered at local iterations 344, 647, and 775
- The corrected side searches share tree nodes with the resumed main search, so their visit/value updates changed later main-tree selection even though the correction windows themselves found no new class

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 41, up from the baseline's 36
- Found classes: 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
- Added relative to baseline: 3, 13, 16, 24, and 33
- Lost relative to baseline: none
- Missing class IDs: 1, 6, 11, 18, and 26
- Class 3 was found by class 1 at local iteration 567
- Class 13 was found by class 1 at local iteration 1239; its new pattern task found class 16 at local iteration 7, class 33 at 10, and class 24 at 115
- Class 4 was recovered by class 1 at local iteration 1565
- Class 10 was reproduced by class 15 at local iteration 196, identical to the historical baseline's local discovery point
- Class 11 had zero terminal encounters
- Class 1 stopped at local iteration 1965 after its last new local class at iteration 1565; class 13 stopped at 515 after its last new class at 115

## 2026-08-13 class7 i200 partition-and-basin task identity

Result files:

- `src/mcts/runs/interrupt_search_class7_i200_partition_basin.json`
- `src/mcts/runs/interrupt_search_class7_i200_partition_basin.runs.json`

Method:

- Replaced block-size-multiset task deduplication with `task_identity = (partition_key, basin_key)`
- `partition_key` is the group-canonical block partition; `basin_key` is the group-canonical full representative support
- Classes with the same partition but different basins now receive independent MCTS nodes, visits, Q values, RNG streams, and budgets
- Scorers with the same ordered partition and scorer configuration share reward-independent geometry caches; terminal validation remains shared by the 64-bit support key
- All search parameters otherwise match the historical productive-weighted class7 i200 run

Timing and workload:

- Elapsed: 12228.348 seconds (03:23:48.348)
- Completed tasks: 26 / 26; old shape-only baseline: 12 / 12
- Total iterations: 14211; total nodes: 12945
- Elapsed per iteration: 0.860485 seconds
- Stop reasons: 13 iteration exhaustion, 6 productive novelty patience, 7 self-or-invalid sinks
- Historical 24-class baseline: 678.434 seconds, 3140 iterations, 2333 nodes, and 0.216062 seconds per iteration
- This run used 4.526x as many iterations and 5.549x as many nodes, took 18.024x as long, and was 3.983x slower per iteration
- The Python worker working set was observed at approximately 12.1 GB near the end of the run; retained independent trees plus per-partition structural caches created substantial memory pressure

Identity behavior:

- 13 discoveries opened a unique partition and 12 opened a new basin under an already-started partition
- Classes 28, 31, 30, 15, 20, 19, 12, 8, 10, 22, 11, and 9 were no longer suppressed merely because their block-size shape had already appeared
- Different basins keep independent search statistics while identical ordered partitions can reuse geometry calculations

Results:

- Stop reason: `stack_empty`
- Exact class coverage: 26, up from the old baseline's 24
- Found classes: 7, 8, 9, 10, 11, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46
- Added relative to the 24-class baseline: 10 and 11; lost classes: none
- Class 10 was found by the class-8 task at global iteration 4868, local iteration 383, depth 17
- Class 10 found class 22 at global iteration 5808, local iteration 940; the independent class-22 basin then found class 11 after 8 local iterations at global iteration 5816, depth 17
- Class 11 subsequently used its full 1000-iteration productive budget and found no additional globally new class
- The coverage improvement validates the partition-plus-basin identity, but the runtime and memory growth show that this policy needs global cache eviction and basin-aware budget scheduling before becoming the default large search

## 2026-08-24 pair-involution class1 i200 / i500

Result files:

- `src/mcts/runs/pair_involution_search_class1_i200.json`
- `src/mcts/runs/pair_involution_search_class1_i200.runs.json`
- `src/mcts/runs/pair_involution_search_class1_i500.json`
- `src/mcts/runs/pair_involution_search_class1_i500.runs.json`

Method:

- Added the independent `src/mcts/pair_involution_search.py` entry point
- The 261 fixed-point-free involutions in the Bell symmetry group reduce to 18 conjugacy-inequivalent pair partitions; every partition owns an independent MCTS tree and RNG stream
- Every terminal contains exactly 13 two-vertex blocks, hence exactly 26 vertices
- A depth-k prefix survives only when its affine rank is the maximum `2k - 1`; rank-21 and rank-23 prefixes use cached exhaustive two-block and one-block tails
- Rollouts use full-orbit, undiscovered-class compatibility without hard-coding class 18; trees share only the global discovered set and reference-support terminal cache
- Corrector proposals remove one or two selected pairs from an exact terminal and reuse the rank-23/rank-21 exhaustive tail; each tree permits at most 24 events
- Class 1 is the initial global discovery seed. The pair trees are a specialized companion search, not a replacement for the old class-derived MCTS

Structural check:

- Exactly one of the 18 patterns, pattern 3, can represent a known 26-point facet support
- Pattern 3 represents 64 symmetry images of class 18; the other 17 patterns represent no known 26-point class support
- Every class-18 target prefix follows affine ranks `1, 3, 5, ..., 21, 23, 25`
- A rank-21 combinatorial tail has `C(21, 2) = 210` possible completions before rank pruning, and a rank-23 tail has 20

Timing and workload:

| run | elapsed | scheduled tree iterations | nodes | full-rank terminals | exact hits | terminal-cache hit rate | Corrector events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| i200 | 40.774 s | 3,600 | 3,572 | 211,634 | 297 | 50.18% | 24 (12 remove-1, 12 remove-2) |
| i500 | 98.542 s | 9,000 | 8,823 | 524,180 | 540 | 50.24% | 24 (12 remove-1, 12 remove-2) |

Discovery timeline and coverage:

- Both runs discovered class 18 in round 1 at global tree iteration 3, which is pattern 3's first visit
- Class-18 wall time was 0.998 seconds in i200 and 1.042 seconds in i500, including setup
- Standalone coverage was `{1, 18}` in both runs: class 1 is the seed and class 18 is the sole search discovery
- The old class1 i200 baseline covered 41 classes and missed 6, 9, 13, 18, and 24; union with the pair i200 result covers 42 classes and leaves 6, 9, 13, and 24
- The old class1 i500 baseline covered 45 classes and missed only 18; union with the pair i500 result covers all 46 classes
- Increasing from i200 to i500 produced no new pair-search class because the exact representability bank already proves that class 18 is the only known 26-point class available under these 18 pair partitions

## 2026-08-24 pair-involution class1 i200 without target guidance

Result files:

- `src/mcts/runs/pair_involution_search_class1_i200_blind.json`
- `src/mcts/runs/pair_involution_search_class1_i200_blind.runs.json`

Configuration and isolation check:

- Added `--disable-target-guidance`; `target_guidance_enabled=false` and `target_bank_usage=diagnostics_only` are recorded in metadata
- Expansion actions are sampled uniformly from full-rank survivors
- UCB receives no compatibility novelty prior, rollouts never select a known target mask, and Corrector proposals do not use missing-class compatibility
- Exact support lookup occurs only after a 13-block rank-25 terminal has been constructed
- A regression test replaces the target bank with methods that raise on access and confirms that blind MCTS runs without calling them

Timing and workload:

- Elapsed: 39.540 seconds
- 18 trees x 200 iterations = 3,600 scheduled tree iterations
- 209,936 full-rank terminal evaluations, all classified as non-reference supports
- 3,200 unique rank-21 tail prefixes and 35,716 unique rank-23 tail prefixes
- Shared terminal cache: 104,974 hits and 104,962 misses
- Pattern 3 evaluated 16,288 full-rank terminals and had zero exact hits
- Corrector events: 0, because blind search produced no exact terminal source from which to remove blocks

Results:

- Initial known seed: class 1
- Newly discovered classes: none
- Class 18 was not found
- Standalone coverage remained `{1}`
- Union with the old class1 i200 baseline remains 41 / 46, missing classes 6, 9, 13, 18, and 24
- This contrasts with target-guided i200, which found class 18 on pattern 3's first visit; full-rank pruning and tail exhaustion alone are therefore insufficient at this budget

## 2026-08-25 unbiased pair-terminal bridge + interrupt class1 i200

Result files:

- `src/mcts/runs/pair_interrupt_search_class1_i200.json`
- `src/mcts/runs/pair_interrupt_search_class1_i200.pair.json`
- `src/mcts/runs/pair_interrupt_search_class1_i200.pair.runs.json`
- `src/mcts/runs/pair_interrupt_search_class1_i200.interrupt.json`
- `src/mcts/runs/pair_interrupt_search_class1_i200.interrupt.runs.json`

Method:

- Added the independent `src/mcts/pair_interrupt_search.py` entry point
- Phase 1 runs all 18 unbiased pair-pattern trees with 200 iterations per tree; class supports are queried only after constructing a complete 13-pair rank-25 terminal
- Phase 2 injects every genuinely new pair-terminal class as a synthetic discovery at the class-1 interrupt root; the resulting class task is pushed before class 1 consumes an interrupt iteration
- Only the observed complete support crosses the bridge. Pair masks, class-18 target masks, and post-search `PairTargetBank` representability diagnostics never enter the interrupt scorer
- The discovered support is independently validated once and seeds the shared interrupt terminal cache; all interrupt tasks share global discovery, rare-target state, terminal validation, and structure caches
- Class 18 was therefore opened as a normal interrupt child task rather than merely unioned into the final coverage set

Timing and workload:

- Pair phase: 137.239 seconds; bridge validation/setup: 0.712 seconds; interrupt phase: 3084.715 seconds
- Full logical elapsed time: 3222.666 seconds (00:53:42.666)
- The completed pair checkpoint was reused after correcting the bridge's integer-support-to-index conversion; the resumed process wall time was 3085.430 seconds
- Pair workload: 18 x 200 = 3,600 scheduled tree iterations
- Interrupt workload: 44 completed tasks and 8,688 local iterations; minimum 50, maximum 536, mean 197.455 iterations per task
- Stop reasons: 33 productive global-novelty patience and 11 self-or-invalid sink
- Shared interrupt terminal cache: 25,000 retained entries, 3,484 hits, 1,325,655 misses, and 1,300,656 evictions

Bridge behavior and discoveries:

- The pair phase found class 18 from pattern 3 in round 63, global pair iteration 1119, at 42.551 seconds
- The class-18 interrupt task directly found classes 37, 46, 35, 36, 39, and 26; class 35 then found class 45
- A later chain `7 -> 22 -> 19 -> 9 -> 11` found class 9 at 640.590 seconds and class 11 at 664.900 seconds
- After returning to class 19, class 10 was found at 852.530 seconds
- Class 1 found class 16, which found class 33 and class 24; class 24 then found class 13
- Late class-1 discoveries at local iterations 336 and the class-4 child added classes 4 and 5

Results:

- Final exact coverage: 44 / 46
- Found classes: 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, and 46
- Missing class IDs: 3 and 6
- The historical class1 i200 interrupt baseline covered 41 classes. This run added classes 9, 13, 18, and 24 relative to it, but did not reproduce baseline class 3, for a net gain of three classes
- A simple union of the historical baseline with standalone pair class 18 covers 42 classes; executing class 18 as an interrupt task and sharing its downstream discoveries raised the independent hybrid run to 44

## 2026-08-25 unbiased pair-terminal bridge + interrupt class1 i300

Result files:

- `src/mcts/runs/pair_interrupt_search_class1_i300.json`
- `src/mcts/runs/pair_interrupt_search_class1_i300.pair.json`
- `src/mcts/runs/pair_interrupt_search_class1_i300.pair.runs.json`
- `src/mcts/runs/pair_interrupt_search_class1_i300.interrupt.json`
- `src/mcts/runs/pair_interrupt_search_class1_i300.interrupt.runs.json`

Configuration:

- Pair phase: 18 trees with 300 iterations per tree
- Interrupt phase: base 300 iterations per task, adaptive maximum 900, productive patience 200, and frontier extension 50
- All target-guidance and bridge-isolation rules are unchanged from the hybrid i200 run
- The run used a fresh pair phase and a fresh interrupt phase; no checkpoint was reused

Timing and workload:

- Pair phase: 235.361 seconds; bridge setup: 0.001 seconds; interrupt phase: 4725.626 seconds
- Full elapsed time: 4960.987 seconds (01:22:40.987)
- Full coverage was first reached at 4230.066 seconds; the remaining 730.921 seconds completed the open class-3, class-4, and class-1 tasks under the configured patience rules
- Pair workload: 18 x 300 = 5,400 scheduled tree iterations
- Interrupt workload: 46 completed tasks and 11,957 local iterations; minimum 50, maximum 640, mean 259.935 iterations per task
- Stop reasons: 35 productive global-novelty patience and 11 self-or-invalid sink
- Relative to hybrid i200, elapsed time increased by 1738.321 seconds and interrupt iterations increased by 3,269

Discovery path to full coverage:

- The pair phase again found class 18 from pattern 3 in round 63, global pair iteration 1119, at 42.229 seconds
- The first 42 interrupt discoveries reproduced the hybrid i200 trajectory through class 2, leaving classes 3 and 6 missing
- Class 1 found class 4 at local iteration 336 and 3195.516 seconds
- Class 4 found class 5 at local iteration 19 and 3215.269 seconds
- The class-5 task used all 300 base iterations without finding a globally new class
- After class 5 returned, class 4 found class 6 at local iteration 284, global interrupt iteration 10802, and 3770.133 seconds
- The class-6 task used all 300 iterations without finding class 3
- After class 6 returned, class 4 remained productive and found class 3 at local iteration 440, global interrupt iteration 11258, and 4230.066 seconds
- Class 4 stopped at local iteration 640, exactly 200 iterations after its final global discovery; class 1 stopped at iteration 536

Results:

- Final exact coverage: 46 / 46
- Missing class IDs: none
- The i300 run added classes 3 and 6 relative to hybrid i200
- This run shows that base 300 is sufficient for this deterministic hybrid seed and task order: finding class 6 before the base limit made class 4 permanently productive, allowing its adaptive budget to reach the later class-3 discovery at iteration 440
- Raising every pair tree from 200 to 300 added 98.122 seconds but did not change the pair discovery set; a split configuration with pair 200 and interrupt 300 should preserve the useful interrupt budget while avoiding that extra pair work
