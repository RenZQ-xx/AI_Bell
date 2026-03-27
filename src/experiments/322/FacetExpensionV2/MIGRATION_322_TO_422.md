# 322 To 422 Migration Notes

## Why Direct Transition Transfer Fails

The recent supervised transition experiment on `322` proves that strong supervision can learn
`source class -> target class` mappings inside the small labeled `322` world.

It does **not** prove that the learned mapping is suitable for `422`.

The main reasons are structural:

- `322` and `422` do not share the same input dimensionality.
- `322` and `422` do not share the same output mask size.
- `322` has a small closed class vocabulary (`46` known classes with `3` examples each).
- `422` is an open discovery problem with no comparable labeled class bank.

Therefore, what transfers cannot be "target mask prediction".

## Geometry Difference

- `322` point tensor shape: `64 x 26`
- `422` point tensor shape: `256 x 80`

So both the number of states and the feature dimension change.

This means a flat MLP that maps a `322` support mask to another `322` support mask is not a
meaningful warm start for `422`.

## Symmetry Graph Difference

- `322` symmetry graph:
  - nodes: `64`
  - directed edges: `576`
  - edge types / generators: `11`

- `422` symmetry graph:
  - nodes: `256`
  - directed edges: `3136`
  - edge types / generators: `15`

The important part is that the *form* is similar:

- both use binary-state symmetry graphs
- both use typed generators
- both naturally admit graph/set encoders

So graph-based message passing is a plausible transfer vehicle.

## Label Difference

`322`:

- small labeled class bank
- supervision can define exact class-conditioned transitions

`422`:

- no comparable class bank
- final objective is discovery of previously unknown inequalities

So `322` should be used to learn a **search prior**, not a final answer map.

## What Should Transfer

The following are plausible transfer targets:

1. Candidate scoring:
   - given current state `x` and candidate `x'`, estimate whether `x'` is a good escape step
2. Feasibility-aware ranking:
   - prefer candidates that move away from the current basin without exploding energy
3. Structure-aware encoder:
   - graph/set representation of `(points, mask, candidate)` that is not tied to a fixed class vocabulary

## What Should Not Transfer

The following should be treated as `322`-only:

1. Full-mask transition decoder:
   - `source class + target class -> target mask`
2. Anything that depends on the `46` known `322` class ids at inference time
3. Any scorer that assumes a fixed node count or fixed point feature dimension

## Recommended Migration Strategy

### Stage 1: Learn Search Priors On 322

Train a graph-based escape scorer on `322` with labels built from:

- positive examples:
  - transitions that reach a target class while remaining exact / feasible
- hard negatives:
  - candidates that collapse back to the dominant basin
  - candidates that move away but become high-energy / infeasible

Training target:

- pairwise ranking or binary preference over candidates
- not target-mask reconstruction

### Stage 2: Reuse The Encoder On 422

On `422`, reuse only:

- the graph/set encoder initialization
- the scorer head initialization, if input semantics match

Then adapt using `422`-native signals:

- energy
- feasibility
- novelty
- hard-negative mining
- rollout survival

## Immediate Design Constraint

Any new model intended for `422` transfer should satisfy both:

1. It consumes graph-structured inputs and supports variable node counts.
2. Its output is a candidate score or local edit preference, not a fixed-length target mask tied to `322`.
