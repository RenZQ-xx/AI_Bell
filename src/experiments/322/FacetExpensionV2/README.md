# FacetExpensionV2

This experiment resets the objective around the real bottleneck observed in `FacetExpansionV1`:

- `class 44` is the only class that is stably reachable from DIffUCO-style search
- our true goal is not "retain non-44 seeds"
- our true goal is "start from class 44 and escape into unknown non-44 classes"

V2 therefore changes the problem framing.

## Core Hypothesis

`class 44` behaves like a dominant attractor basin.

Because of that, local conditional editing and heuristic reranking are likely insufficient on their own.
V2 explores a more disruptive search design:

- reverse-search "escape corridors" from known non-44 classes back toward the `44` basin boundary
- jump proposals that can move toward corridor prototypes instead of relying only on local edits
- `44-only` evaluation as the primary metric

## Primary Metrics

V2 should be judged first by `44-only` escape metrics, not by mixed-seed non-44 retention.

- `escape_rate_from_44`
- `unique_escaped_classes_from_44`
- `first_escape_round`
- `late_round_escape_count`

## Initial Plan

1. Build a corridor inventory from the existing seed bank.
2. Identify all `class 44` seeds and all candidate non-44 target classes.
3. Export metadata that can drive reverse-search and jump-proposal experiments.
4. Implement a first corridor-aware proposal mechanism in a separate V2 script set.

## First Script

Generate the V2 seed/corridor inventory:

```bash
python src/experiments/322/FacetExpensionV2/build_escape_corridor_inventory.py \
  --device cpu \
  --output src/experiments/322/FacetExpensionV2/outputs/corridor_inventory.json
```

This inventory is intentionally simple: it creates the bookkeeping needed to start V2 around `44 -> non44` escape, instead of reusing the V1 mixed-seed protocol.
