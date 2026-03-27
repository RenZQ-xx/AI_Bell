# FacetExpansionV1

This experiment is a first unsupervised `seed -> new candidate` prototype for 3-2-2.

Core idea:

- start from a known valid facet mask
- learn a stochastic edit operator instead of a noise-reversal process
- optimize for:
  - low relaxed geometric energy
  - nontrivial distance from the seed
  - repulsion from the current archive of discovered masks

This is intentionally separate from `src/experiments/322/DIffUCO`.

Train:

```bash
python src/experiments/322/FacetExpansionV1/train.py --device cpu --epochs 100
```

Inference:

```bash
python src/experiments/322/FacetExpansionV1/inference.py \
  --checkpoint src/experiments/322/FacetExpansionV1/outputs/facet_expander_v1.pt \
  --seed-index 0 \
  --num-samples 32
```

Multi-seed evaluation:

```bash
python src/experiments/322/FacetExpansionV1/eval_multiseed.py \
  --checkpoint src/experiments/322/FacetExpansionV1/run_archivectx_20/facet_expander_v1.pt \
  --stochastic \
  --prefer-novel \
  --temperature 1.2 \
  --rounds 6 \
  --num-samples 16 \
  --seed-count 16 \
  --output-dir src/experiments/322/FacetExpansionV1/run_archivectx_20/eval_multiseed_t12_prefern
```

Long-horizon controlled inference:

```bash
python src/experiments/322/FacetExpansionV1/infer_longhorizon.py \
  --checkpoint src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/facet_expander_v1.pt \
  --seed-index 0 \
  --rounds 6 \
  --frontier-size 4 \
  --num-samples 16 \
  --temperature 1.2 \
  --output src/experiments/322/FacetExpansionV1/tmp_longhorizon_seed0.json
```

Long-horizon multi-seed evaluation:

```bash
python src/experiments/322/FacetExpansionV1/eval_multiseed_longhorizon.py \
  --checkpoint src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/facet_expander_v1.pt \
  --seed-count 16 \
  --rounds 6 \
  --frontier-size 4 \
  --num-samples 16 \
  --temperature 1.2 \
  --output-dir src/experiments/322/FacetExpansionV1/run_prefern_repeatcap_10/eval_longhorizon_t12
```

Experimental long-horizon training:

```bash
python src/experiments/322/FacetExpansionV1/train_longhorizon.py \
  --device cpu \
  --epochs 20 \
  --rollout-steps 4 \
  --frontier-size 4 \
  --samples-per-branch 6 \
  --output-dir src/experiments/322/FacetExpansionV1/outputs_longhorizon
```

Stability sweep:

```bash
python src/experiments/322/FacetExpansionV1/sweep_stability.py \
  --device cpu \
  --epochs 12 \
  --seed-count 8 \
  --output-dir src/experiments/322/FacetExpansionV1/sweeps/stability_v1
```
