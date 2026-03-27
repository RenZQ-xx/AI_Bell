# Geometric-DiffUCO 3-2-2 Status Log

This note summarizes the main experiment history, what was tried, what was learned, what remains in the codebase, and what still blocks progress.

## Goal

Search for facet classes of the 3-2-2 Bell polytope with a DIffUCO-style model.

Current practical target:

- generate valid supporting facets reliably
- expand coverage beyond the current dominant `class 44`

## Known Data / Evaluation Setup

- Ground-truth facets: `data/facets_322.txt`
- Class examples and class sizes: `data/facet_classes_322_examples.txt`
- Inference pipeline:
  - sample soft masks
  - round to hard masks
  - validate supporting-face geometry
  - recover integer Bell row
  - match against the reference database

Important observation:

- repeated hits are usually different exact facets inside the same symmetry class, not the identical facet repeated
- `class 44` has size `64`, so its dominance is not explained by orbit size alone

## Main Findings So Far

1. The model can reliably produce valid exact facets, but mostly from `class 44`.
2. Simply increasing sample count does not naturally broaden class coverage.
3. Many "unknown" outputs are not noise; they are lower-dimensional supporting faces.
4. The core bottleneck is not only geometric validity. It is mode preference in trajectory space.

## Tried Approaches

### 1. Soft covariance facet penalty

Idea:

- use covariance spectrum of the soft selected point cloud
- encourage one collapsed direction and discourage further degeneracy

Outcome:

- useful
- this became the current retained geometric surrogate

Conclusion:

- keep

### 2. Near-hard rank surrogate

Idea:

- sharpen probabilities with a sigmoid gate
- compute covariance on near-hard weights
- use the second-smallest eigenvalue as a facet-dimension surrogate

Outcome:

- training was numerically stable after tuning
- but the surrogate often stayed inactive or mismatched the final hard candidate

Conclusion:

- too much complexity for too little gain
- removed from the main code

### 3. Tight-face / near-plane surrogate

Idea:

- restrict the rank surrogate to near-hard points close to the selected support plane

Outcome:

- geometrically better motivated than plain near-hard
- but too brittle
- tight-face cardinality often collapsed
- spectral terms became numerically fragile

Conclusion:

- removed from the main code

### 4. Soft-only rank ratio ("A" scheme)

Idea:

- use
  - smallest eigenvalue to enforce hyperplanarity
  - ratio `lambda_26 / (lambda_25 + eps)` to discourage lower-dimensional collapse

Outcome:

- stable
- interpretable
- successfully turned the facet-dimension term into an active training signal
- but did not solve class coverage by itself

Conclusion:

- keep
- this is the current geometric backbone

### 5. Batch diversity reward

Idea:

- encourage pairwise L1 distance between batch members

Outcome:

- weak effect
- did not reliably break `class 44` dominance

Conclusion:

- removed

### 6. Fixed `class 44` avoidance mask

Idea:

- penalize final masks that are too similar to a known `class 44` representative

Outcome:

- occasionally reduced frequency of the dominant mode
- but did not produce stable new-class discovery

Conclusion:

- removed

### 7. Stage-wise coverage memory

Idea:

- after validation, collect matched classes and representative masks
- penalize future masks that are similar to high-frequency known classes

Outcome:

- interesting signal: during training, `class 43` appeared once in memory
- but final inference still stayed dominated by `class 44`
- extra mechanism complexity was not justified yet

Conclusion:

- removed from the main code
- conceptually still interesting, but not part of the clean baseline

### 8. Orbit-aware forward noise

Idea:

- replace plain Bernoulli transition noise with a group-structured kernel based on symmetry generators

Outcome:

- pure orbit kernel was too rigid for the current learned reverse model
- mixed Bernoulli-orbit kernel was trainable but did not improve coverage
- often hurt exact facet quality

Conclusion:

- removed from the main code

### 9. Explicit trajectory model and reverse-KL restructuring

Idea:

- align the Bell implementation with the original DIffUCO structure:
  - explicit `p(X_t | X_{t-1})`
  - explicit terminal `p(X_0) proportional to exp(-E(X_0)/T)`
  - explicit reverse trajectory `q_theta(X_{t-1} | X_t)`

Outcome:

- valuable
- clarified the connection to the original DIffUCO code
- made the trajectory objective explicit

Conclusion:

- keep

### 10. True REINFORCE-style reverse-KL gradients

Idea:

- move from a heuristic trajectory loss to actual score-function gradient contributions for noise and terminal energy

Outcome:

- theoretically cleaner
- variance is significantly higher
- short training became less stable geometrically
- did not improve class coverage yet

Conclusion:

- keep for now as the explicit trajectory baseline
- but treat as high-variance and unfinished

### 11. Lower-variance relaxed trajectory objective

Idea:

- keep the explicit Bernoulli `p/q` trajectory bookkeeping
- optimize the relaxed entropy + transition + terminal-energy objective directly
- keep the REINFORCE correction only as an optional weighted add-on

Outcome:

- now the default code path
- intended to reduce optimization variance without discarding the explicit trajectory decomposition

Conclusion:

- keep as the default baseline
- use nonzero `reinforce_weight` only for targeted comparisons

## Current Retained Code Path

The codebase has been cleaned back to the following main line:

### Geometry

File: `src/experiments/322/DIffUCO/energy.py`

Retained pieces:

- supporting-hyperplane geometric energy
- soft-only rank-ratio facet surrogate

## 2026-03-19 Update

This session focused on understanding why the experiment still collapses to `class 44`, and on recording several negative-but-informative interventions.

### A. Rounding and inference-side findings

- The old "conditional expectation" implementation was not a true CE procedure. It behaved more like:
  - seed with the top `min_cardinality` points
  - greedily add extra points if hard energy does not worsen
- `inference.py` was updated to a more faithful sequential CE-style rounding rule:
  - iterate variables in probability order
  - compare `X_i = 0` vs `X_i = 1`
  - keep the better hard-energy branch
- Result:
  - exact-facet count increased
  - but coverage collapsed even more strongly to `class 44`
- Interpretation:
  - the dominance of `class 44` is not only a bug in the old greedy rounding
  - under the current relaxed energy, CE itself also prefers the wider `class 44` basin

### B. Soft vs hard geometry diagnosis

- `class 43` is not obviously worse as a final hard facet.
- In successful hard matches, `class 43` can even have cleaner plane fit than many `class 44` examples.
- The real difference appears earlier:
  - `class 43` is a narrower basin in soft-probability space
  - `class 44` is easier to round into a valid hard facet
- Boundary decomposition shows:
  - `support` is usually not the issue
  - `inactive` is usually tiny
  - the main separating term is `active`
- Practical interpretation:
  - `class 43` is more fragile to soft active-set ambiguity
  - `class 44` is tolerated by the current active-term design

### C. Active-term experiments

Several variants were tested against the current CE inference baseline.

Observed pattern:

- `active_probability_power = 2.0` was the only tested change that preserved a `class 43` hit under the current CE pipeline.
- `gamma = 1.5` increased exact matches, but collapsed entirely to `class 44`.
- Huber-style active loss reduced exact facet quality too much.
- `plane_topk` promoted some exploration / unknown orbits, but did not recover `class 43`.
- `active_topk` behaved close to baseline.

Conclusion:

- sharpening the active weights is the most promising single-term change seen so far
- but this only helps narrow-basin survival
- it does not solve the broader coverage problem

### D. Memory / novelty / reranking attempts

The following ideas were tested and did not solve the main issue:

- mask-space memory repulsion
- plane-space memory repulsion
- density-based plane novelty
- inference-time plane-space reranking for diversity

Common failure mode:

- these methods sometimes reduce repeated exact visits to the dominant pattern
- but they do not create a stable path into new exact classes
- in larger candidate pools, the exact-match candidates often still belong only to `class 44`

Conclusion:

- the problem is upstream of final ranking
- the main bottleneck is still the training energy landscape and mode dynamics

### E. Cardinality shaping

- The previous explicit reward for larger supporting sets was removed.
- Cardinality now only penalizes falling below `min_cardinality`.
- This did not recover `class 43`.
- In one test it even increased exact `class 44` hits.

Conclusion:

- the old size reward was a bias
- but not the main cause of collapse

### F. Set-level objectives tried this session

The main open hypothesis was:

- a single-sample objective may be fundamentally too weak
- maybe diversity must be optimized at the set / batch level

Three families were tested.

#### 1. Batch plane coverage bonus

Implementation:

- reward batch members whose soft plane embedding is far from the nearest other high-quality sample

Outcome:

- training stayed stable
- exact output remained entirely `class 44`

Conclusion:

- too weak
- mainly spreads samples inside the dominant basin

#### 2. DPP / log-det plane diversity

Implementation:

- build a quality-gated plane kernel across the batch
- use a log-det style diversity gain as a set bonus

Outcome:

- the objective is much stronger than nearest-neighbor coverage
- it nudged outputs toward extra orbit-like regions
- but training became numerically unstable very quickly
- even when isolated from other auxiliary terms, runs typically stopped with non-finite gradients around epoch 5
- best checkpoints still produced exact matches dominated by `class 44`

Important note:

- one run produced a second canonical orbit / unknown cluster
- but still no new exact class beyond `44`

Conclusion:

- strong set-level diversity is directionally interesting
- but the current full-batch DPP formulation is too aggressive for this training setup

#### 3. Elite population objective

Implementation:

- select a small elite set of high-quality soft terminal samples in each batch
- optimize:
  - low elite energy
  - plus diversity across elite plane embeddings

Outcome:

- this behaved like a more explicit population-style objective than the previous batch bonuses
- however, it was also unstable
- both tested weights (`0.1`, `0.01`) stopped around epoch 5 with non-finite gradients
- saved checkpoints still produced only `class 44` exact matches, with lower exact-facet count than the CE baseline

Conclusion:

- the population idea is conceptually reasonable
- but the current direct elite objective is too forceful and not yet useful

### G. Current interpretation after this session

The evidence now points to the following picture:

1. `class 44` is not merely a post-processing artifact.
2. It is a genuinely wide and stable basin under the current relaxed energy and CE-style decoding.
3. `class 43` and likely other classes behave like narrower basins that are easy to erase with extra regularization.
4. Weak diversity regularizers do not open new exact classes.
5. Strong set-level objectives do push harder, but in the current formulation they destabilize training before they create useful new coverage.

### H. Recommended restart point for the next session

If work resumes in a new thread, the most relevant open directions are:

- design a more stable multi-head or latent-mode architecture
- or redesign population / set-level training so that:
  - elite selection is detached or softened
  - gradients do not explode
  - different mode channels can specialize without all collapsing to `class 44`

The main takeaway to carry forward is:

- the remaining problem is no longer "can the model produce valid facets?"
- it is "how to represent and optimize multiple narrow basins without immediately collapsing into the broad `class 44` basin"
- hard-mask validation helper

### Trajectory model

Files:

- `src/experiments/322/DIffUCO/noise.py`
- `src/experiments/322/DIffUCO/diffusion.py`
- `src/experiments/322/DIffUCO/train.py`

Retained pieces:

- Bernoulli forward noise
- explicit trajectory bookkeeping
- explicit terminal Boltzmann factor
- reverse-KL style loss decomposition
- lower-variance relaxed objective as the default loss
- REINFORCE-style trajectory terms still present as an optional correction

### Inference and coverage reporting

Files:

- `src/experiments/322/DIffUCO/inference.py`
- `src/experiments/322/DIffUCO/coverage_tracker.py`

Retained pieces:

- rounding
- geometric validation
- integer row recovery
- exact class matching
- coverage summary reporting

## Current Main Problems

### 1. Strong mode preference

The model still heavily prefers `class 44`.

This appears to be a trajectory / basin issue, not just a final-energy issue.

### 2. Coverage is poor

Even when exact facet generation is good, class coverage remains low.

### 3. Reverse-KL is high variance

The explicit REINFORCE-style trajectory objective is cleaner theoretically, but noisier than the earlier relaxed heuristics.

### 4. Exploration mechanisms tested so far were not strong enough

Multiple anti-collapse ideas changed behavior slightly, but none yielded robust multi-class discovery.

## Recommended Baseline Going Forward

If starting a fresh thread, assume this baseline:

1. Keep the clean code path only.
2. Use `energy.py` with the soft-only rank ratio.
3. Use the explicit Bernoulli trajectory model.
4. Treat `class 44` dominance as the current main open problem.
5. Avoid reintroducing old branches unless there is a specific new hypothesis.

## Recommended Next Questions

The most useful next-step questions are now:

1. Can a multi-head or latent-mode architecture prevent all probability mass from collapsing into the broad `class 44` basin?
2. Can a population / set-level objective be redesigned so that:
   - elite selection is detached or softened
   - gradients stay finite
   - specialization is encouraged without destroying exact-facet quality?
3. Is there a quotient-aware objective that helps, but is stable enough to coexist with the current relaxed geometric backbone?

## Useful Reference Outputs Still Kept

The following result folders are still useful as baselines or historical references:

- `src/experiments/322/DIffUCO/run_cpu_probe_soft_ratio`
- `src/experiments/322/DIffUCO/run_cpu_probe_soft_ratio_30`
- `src/experiments/322/DIffUCO/run_cpu_probe_reversekl_5`
- `src/experiments/322/DIffUCO/run_cpu_probe_reversekl_reinforce_5`

## Sanity Check

After cleanup, a smoke run still succeeds with the current retained code path.

## Historical Orbit-Aware Note

The following orbit-aware steps were implemented earlier and are still useful to remember:

1. Canonicalization of recovered integer Bell rows under the 3-2-2 symmetry group.
2. Inference and coverage summaries keyed by canonical orbit representative in addition to class id.
3. Orbit-aware reward shaping during training:
   - exact-match novelty bonus by class visit count
   - unknown-face novelty bonus by canonical-row visit count
   - orbit-size penalty to reduce the natural advantage of large or easy-to-hit orbits
4. A first hard-terminal orbit-aware objective:
   - hard terminal `log p_0` depends on canonical orbit information
   - element-level terminal probability includes orbit-size normalization
   - the relaxed geometric backbone is still kept for stability

What remains true from that stage:

- the machinery works end-to-end
- the reward activates during training
- short schedules still remain dominated by `class 44`

Important correction after the 2026-03-19 experiments:

- it is no longer enough to say "the next natural step is a soft orbit surrogate"
- several stronger interventions were tried after that point:
  - active-term reshaping
  - mask / plane memory repulsion
  - density novelty
  - inference reranking
  - batch coverage bonus
  - DPP/log-det set objective
  - elite population objective
- none of them solved exact-class coverage
- the stronger set-level objectives were interesting, but became unstable before opening a reliable path to new exact classes

So the current restart recommendation is not simply:

- "continue from hard-terminal orbit-aware reward to soft orbit surrogate"

Instead it is:

- treat orbit-aware soft surrogates as one historical direction that was explored
- and prioritize architecture-level anti-collapse ideas such as multi-head / latent-mode designs, or a more stable population-style training scheme
