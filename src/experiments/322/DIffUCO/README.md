# Geometric-DiffUCO for 3-2-2

This experiment adapts the DIffUCO idea to the 64-vertex, 26-dimensional 3-2-2 Bell polytope search problem.

Core choices:

- Vertices are the fixed `Points_322()` construction already used in `AI_Bell`.
- The graph is not fully connected. It is built from the same symmetry generators used in `src/experiments/322/nauty/graph_define.py`.
- The backbone is a time-conditioned GAT that updates only along symmetry edges.
- The terminal geometric energy uses a soft-only rank-ratio surrogate, `lambda_26 / (lambda_25 + eps)`, to prefer exactly one collapsed direction.
- Training follows an explicit Bernoulli trajectory model with terminal `p(X_0) proportional to exp(-E(X_0)/T)` and reverse-KL style loss terms.

Facet-dimension term:

- `plane`: drives the smallest covariance eigenvalue down so selected points lie on a hyperplane.
- `facet_dim`: uses the soft covariance spectrum directly and penalizes the ratio `lambda_26 / (lambda_25 + eps)`, which favors exactly one collapsed direction while keeping the neighboring direction non-degenerate.
- `support / active / inactive`: choose the better support orientation, pull selected points onto the support plane, and push unselected points into the interior.

Training:

```bash
python src/experiments/322/DIffUCO/train.py --device cpu --epochs 400 --batch-size 32
```

The default training objective is now the lower-variance relaxed trajectory loss. To re-enable the score-function correction for comparison, pass a nonzero `--reinforce-weight` such as `0.1` or `1.0`.

Inference:

```bash
python src/experiments/322/DIffUCO/inference.py \
  --checkpoint src/experiments/322/DIffUCO/outputs/geometric_diffuco_322.pt \
  --num-samples 256
```

Outputs:

- Checkpoints are written under `src/experiments/322/DIffUCO/outputs/`.
- Inference returns integerized exact matches, filtered known classes, and clustered unknown supporting faces.
