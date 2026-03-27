# FacetExpansionV1 Status Log

Last updated: 2026-03-20

This log summarizes the new `FacetExpansionV1` line under `src/experiments/322/FacetExpansionV1`, which is intentionally separate from `src/experiments/322/DIffUCO`.

## 1. Problem Reframing

The original DIffUCO-style objective is good at finding one valid low-energy facet, but it is structurally mismatched with the real goal:

- not `generate one good facet`
- but `given known facet(s), expand to other valid facet classes`

So this line reformulates the task as a conditional expander:

- input: a known exact facet seed
- output: another valid candidate
- training: unsupervised / weakly supervised
- evaluation: exact facet-class coverage growth

The key shift is:

- from unconditional `P(facet)`
- to conditional expansion `P(next facet | seed, archive)`

## 2. Main Components Built Today

Core files now in place:

- `seed_bank.py`
  - reconstructs seed masks from `data/facet_classes_322_examples.txt`
- `editor_model.py`
  - conditional edit proposal model
- `train.py`
  - current main training line for the conditional expander
- `inference.py`
  - rollout inference with stochastic sampling and novelty-aware selection
- `eval_multiseed.py`
  - multi-seed evaluation for the standard rollout setting
- `long_horizon_controller.py`
  - explicit multi-branch controller for long-horizon rollout
- `infer_longhorizon.py`
  - long-horizon controlled inference
- `eval_multiseed_longhorizon.py`
  - multi-seed evaluation for long-horizon inference
- `train_longhorizon.py`
  - first training-side prototype with controller-in-the-loop rollout

## 3. Best Standard Baseline So Far

Current strongest non-long-horizon baseline:

- run directory:
  - `src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10`

This run uses:

- novelty-aware frontier selection
- repeat cap in frontier retention
- stochastic inference
- `temperature = 1.2`
- rollout expansion

Main result:

- multi-seed summary:
  - `src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/eval_multiseed_t12_prefern/multiseed_summary.json`
- `16 / 16` seeds produce non-`44` classes
- union of matched classes:
  - `1, 2, 3, 4, 5, 6, 14, 16, 20, 23, 27, 29, 42, 43, 44, 45`
- class coverage:
  - `16 / 46 = 34.8%`

Interpretation:

- the expander definitely learns useful multi-basin behavior
- but standard rollout still tends to find most new classes early, then shrink back

## 4. What Long-Horizon Inference Added

The most important new result from today is that explicit search control helps.

### 4.1 Single-seed smoke

File:

- `src/experiments/322/FacetExpansionV1/tmp_longhorizon_seed0.json`

Using:

- checkpoint:
  - `src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/facet_expander_v1.pt`
- long-horizon controller
- rounds = 6
- frontier size = 4
- stochastic sampling

Result for `seed 0`:

- matched classes:
  - `1, 8, 16, 23, 27, 43, 44`
- canonical count:
  - `7`
- frontier size stayed alive into later rounds:
  - roughly `2 -> 2 -> 3 -> 4 -> 4 -> 4`

This was the first clear sign that later rounds could still open new classes.

### 4.2 Multi-seed long-horizon evaluation

Combined summary:

- `src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/eval_longhorizon_t12_combined/multiseed_summary.json`

Main result:

- `16 / 16` seeds still produce non-`44` classes
- union of matched classes:
  - `1, 2, 3, 4, 5, 6, 8, 14, 16, 20, 23, 27, 29, 35, 40, 42, 43, 44, 45`
- class coverage:
  - `19 / 46 = 41.3%`

So relative to the previous best baseline:

- before long-horizon controller:
  - `16 / 46`
- after long-horizon controller:
  - `19 / 46`

New classes opened in this setting:

- `8, 35, 40`

Additional useful signal:

- `8 / 16` seeds still discovered new exact classes at round `>= 4`

This is important because it means:

- the improvement is not just "more diversity in round 1"
- it is partially a true long-horizon effect

## 5. What Still Blocks Long-Horizon Discovery

Even with the long-horizon controller, the system is not yet a fully stable long-range discoverer.

Current bottlenecks:

1. Dominant canonical still overwhelms counts.
   - In the long-horizon summary, the canonical corresponding to `class 44` still dominates heavily.

2. Many branches still converge into a small set of familiar basins.
   - Most commonly `44`, then `23` and `43`.

3. The proposal model itself was not trained under a truly multi-branch controller distribution.
   - Current long-horizon gains come mostly from inference-side control.

So the present state is:

- long-horizon inference helps
- long-horizon training has not yet caught up

## 6. Training-Side Long-Horizon Prototype

New prototype:

- `src/experiments/322/FacetExpansionV1/train_longhorizon.py`

Goal:

- keep the current proposal model family
- but train it inside a multi-branch controller rollout loop

### 6.1 Smoke result

Smoke output:

- `src/experiments/322/FacetExpansionV1/tmp_longhorizon_train_smoke8`

What happened:

- the script runs and trains stably
- by epoch 2 onward it starts producing exact candidates
- but train-time `seen_exact_classes` is almost entirely `44`
- train-time `frontier_size` still collapses to `1`

So this prototype is:

- runnable
- not useless
- but not yet solving the long-horizon training problem

### 6.2 Inference from the smoke checkpoint

File:

- `src/experiments/322/FacetExpansionV1/tmp_longhorizon_train_smoke8/infer_seed0.json`

Result:

- `seed 0` matched:
  - `1, 6, 8, 23, 43, 44`

Interpretation:

- even this weak training-side prototype already gives a usable proposal
- but it does not yet beat:
  - old training + new long-horizon controller inference

## 7. Current Best Practical Recommendation

If starting a new thread right now, the best working recipe is:

1. Treat `run_prefern_repeatcap_10` as the strongest training checkpoint.
2. Use `infer_longhorizon.py` rather than the older rollout inference when the goal is class expansion.
3. Use:
   - rounds `= 6`
   - frontier size `= 4`
   - temperature `= 1.2`
   - stochastic sampling
4. Use the long-horizon multi-seed summary as the current best evidence of progress.

Most important current artifact:

- `src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/eval_longhorizon_t12_combined/multiseed_summary.json`

## 8. Recommended Next Step

Most promising next step:

- improve `train_longhorizon.py` so that frontier evolution does not depend only on exact candidates

Reason:

- early training is too sparse
- if frontier updates rely only on exact matches, training easily collapses into a single safe branch

Most natural minimal fix:

- allow a small number of high-scoring non-exact supporting candidates to act as bridge states during training rollout

In other words:

- keep exact matching as the main archive/evaluation standard
- but do not require exactness for every training-time frontier transition

## 9. One-Sentence Summary

Today established that `FacetExpansionV1` is no longer just a short-horizon expander: with an explicit long-horizon controller, coverage improved from `16/46` to `19/46`, and later-round discovery became real on a nontrivial fraction of seeds, but training-side long-horizon dynamics still lag behind inference-side control.
