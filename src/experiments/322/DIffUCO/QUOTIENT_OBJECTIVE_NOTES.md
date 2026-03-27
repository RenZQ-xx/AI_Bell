# Quotient-Space / Orbit-Aware Objective Notes

This note records the current practical implementation path toward quotient-space training and the next deeper objective still under study.

## Implemented Stages

### 1. Canonicalization

- integer Bell rows now expose a canonical orbit representative under the 3-2-2 symmetry group
- exact matches carry:
  - `canonical_integer_row`
  - `canonical_key`
  - `orbit_size`
  - `orbit_unique_rows`
- unknown supporting faces are also grouped by canonicalized recovered rows

### 2. Canonical / Class Statistics

- inference summaries now report:
  - class hit counts
  - canonical orbit hit counts
  - unknown clusters grouped by canonical row
- coverage reports now include canonical orbit counts in addition to class coverage

### 3. Orbit-Aware Reward Shaping

Current training implementation keeps the low-variance relaxed trajectory objective as the base loss and adds an extra score-function correction:

`L_total = L_relaxed + lambda_orbit * L_orbit_reward`

where:

- exact matched classes receive a novelty bonus based on inverse square-root visit count
- unknown supporting faces receive a smaller novelty bonus based on canonical-row visit count
- a log orbit-size penalty reduces the value of large orbits that are easy to revisit

This is not yet a true quotient-space ELBO. It is a practical reward-shaping layer on top of the element-space trajectory objective.

### 4. First Orbit-Aware Terminal Objective

The training path now also supports a first hard-terminal orbit-aware objective.

Instead of using only:

`log p_0(x) = -E_geom(x) / T`

the hard terminal score is replaced by an orbit-aware version built from:

- geometric terminal energy
- canonical orbit representative
- orbit size
- class / canonical novelty memory

At the element level this follows the practical rule:

`log p_0(x) = -E_orbit([x]) / T - log orbit_size([x])`

This is still mixed with the relaxed geometric backbone for stability, but the terminal `log p_0` used by the trajectory bookkeeping is now orbit-aware.

## Short Experimental Result

Short run:

- `run_orbit_baseline_6`
- `run_orbit_reward_6`

Observed behavior:

- the orbit-aware reward is active and produces nonzero correction terms during training
- canonical/class bookkeeping works and stays aligned with the known `class 44` dominance pattern
- short training still collapses to `class 44`
- the orbit-aware run produced many canonical unknown supporting-face orbits at inference time
- the baseline checkpoint developed an inference-time slowdown / stall during rounding-validation, which needs a separate follow-up

Interpretation:

- quotient-aware bookkeeping is now real, not just conceptual
- reward shaping alone is not yet strong enough to break the dominant basin on a short schedule
- the unknown canonical-orbit diversity signal suggests the idea is interacting with exploration, but not yet converting that exploration into new exact classes

## Next Deeper Objective Under Study

The current implementation now has a first hard-terminal orbit-aware objective, but it still keeps the relaxed element-space backbone for stability. A deeper quotient-space objective would push more of the terminal and relaxed loss fully into orbit space.

### Proposed Direction

Instead of only using:

`p(x_0) proportional to exp(-E_geom(x_0) / T)`

consider an orbit-level target:

`p([x_0]) proportional to exp(-E_orbit([x_0]) / T)`

with:

`E_orbit([x]) = E_geom(x_canon) + beta * log orbit_size([x]) - gamma * novelty([x])`

where:

- `[x]` is the orbit of `x`
- `x_canon` is its canonical representative
- `log orbit_size` compensates for large or easy-to-hit orbits
- `novelty([x])` can come from coverage memory or an online occupancy estimate

### Practical Approximation

The next deeper implementation is:

1. sample hard terminal masks as usual
2. canonicalize the recovered row
3. estimate an orbit-level terminal score
4. define a soft orbit surrogate so the relaxed terminal term is also orbit-aware, not only the hard terminal bookkeeping

This would move the code from "orbit-aware hard terminal objective" to a more complete quotient-space training objective.

## Open Risks

- canonicalization is cheap enough for training, but exact inference can still become slow on difficult checkpoints because rounding remains element-space
- novelty bonuses may increase unknown supporting-face diversity without improving exact class discovery
- orbit-size correction helps debias large classes, but it may also reduce exact-facet quality if applied too aggressively
