# FacetExpensionV2 Status Log

## 2026-03-23

Experiment initialized.

Problem statement:

- V1 mixed-seed evaluation overstated progress for the real task.
- Under the corrected `44-only` protocol, both the old strongest run and the best V1 scorer remain at `0/3` escape from `class 44`.

V2 starts from that corrected baseline.

Immediate goals:

1. Treat `44 -> non44` escape as the main objective.
2. Build an explicit corridor inventory from known seeds.
3. Use that inventory to support reverse-search and jump-proposal experiments.

Current state:

- directory created
- V2 README created
- initial corridor inventory builder added

Progress since initialization:

- Added reference-guided geometry probes:
  - `select_corridor_prototypes.py`
  - `build_jump_proposal_bank.py`
- Added the first true unsupervised sampler around the `44` basin:
  - `sample_44_boundary_states.py`
- Sampled `384` nearby states from the three `class 44` seeds and wrote:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/boundary_state_cloud_v1.jsonl`
- Key result from the unsupervised state cloud:
  - `129` exact states
  - `0` non-`44` exact states
  - strongest boundary-like states are still tiny `hamming=1..3` perturbations near the `44` basin edge

Reference-guided jump check:

- Added `infer_jump_escape.py` and `eval_jump_escape.py`
- Using the reference-guided jump bank derived from known non-`44` seeds, `44-only` escape becomes `3/3`
- This is treated only as an oracle-style feasibility probe, not a valid mainline method

Unsupervised boundary-prototype jump check:

- Added:
  - `cluster_boundary_states.py`
  - `build_unsupervised_jump_bank.py`
- Built unsupervised boundary prototypes from the sampled state cloud:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/boundary_prototypes_v1.json`
- Built the corresponding unsupervised jump bank:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/jump_proposal_bank_unsupervised_v1.json`
- Evaluated both exact-only and bridge-enabled unsupervised jumps:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_unsupervised_v1/multiseed_summary.json`
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_unsupervised_v1_bridge/multiseed_summary.json`
- Result: both remain `0/3` escape from `class 44`

Current interpretation:

- Structured jump moves can break the `44` attractor if we give the search oracle-like non-`44` targets.
- Purely unsupervised local boundary prototypes are not enough; they remain too close to the `44` basin and do not open an escape corridor.

Next step:

- move from "nearest boundary prototypes" to more aggressive, diversity-seeking or anti-`44` proposals that are not tied to known non-`44` seeds but are also not confined to tiny local edits

Migration-friendly escape scorer baseline:

- Added:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/graph_escape_scorer.py`
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/train_graph_escape_scorer.py`
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/eval_graph_escape_scorer.py`
- Mainline change:
  - stop predicting full target masks for transfer
  - instead learn a structure-aware `score(source -> candidate)` objective on `322`
  - current lightweight baseline uses the `SetEscapeScorer`, which keeps variable-size point-set inputs and avoids fixed class-decoder assumptions
- Efficiency fixes:
  - cached exact-seed candidate records instead of re-validating them for every source
  - added `--eval-every`
  - normalized `/home/...` CLI paths to `\\wsl$\\Ubuntu\\...` under Windows so outputs write back into the repo
- Holdout run:
  - output dir:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/graph_escape_scorer_holdout_set_v1`
  - training summary:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/graph_escape_scorer_holdout_set_v1/train_summary.json`
  - evaluation:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/graph_escape_scorer_holdout_set_v1/eval.json`
- Result:
  - `num_samples = 19596`
  - `val_accuracy = 97.90%`
  - `val_positive_accuracy = 99.52%`
  - `val_negative_accuracy = 57.49%`
  - `top1_escape_rate = 0.9928`
  - `top1_escape_rate_source44 = 1.0`
- Interpretation:
  - the scorer is already very good at ranking exact candidates from "escape vs non-escape" on held-out examples
  - for all three `source_class=44` seeds, the top-ranked held-out exact candidate is a non-`44` escape
  - this is the first V2 result that looks genuinely transferable to `422`, because it learns a candidate-quality prior instead of a fixed-dimension class-to-class decoder

Escape-scorer-in-the-loop check on real `44-only` search:

- Integrated the V2 `escape scorer` into:
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/infer_jump_escape.py`
  - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/eval_jump_escape.py`
- Added:
  - scorer loading for both `graph` and `set` architectures
  - candidate-level `escape_score`
  - `--escape-scorer`, `--escape-mode`, `--escape-weight`
  - path normalization for Windows/WSL mixed execution
- Evaluated on real `44-only` jump search with bridge-enabled proposals:
  - anti-44 bank + scorer:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_anti44_v1_bridge_escape_scorer/multiseed_summary.json`
  - unsupervised boundary bank + scorer:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_unsupervised_v1_bridge_escape_scorer/multiseed_summary.json`
- Result:
  - both remain `0/3` escape from `class 44`
- Diagnostic:
  - scorer changes frontier dynamics, but the search still produces no exact non-`44` candidates
  - example:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_anti44_v1_bridge_escape_scorer/seed_131.json`
  - `frontier_size` grows to `2 -> 3`, yet every round still only matches `class 44`
  - highest scorer-ranked candidates in this real search are still exact `44` states with very low escape scores
- Interpretation:
  - the current scorer is useful as an exact-candidate prior
  - but real `44-only` rollout is still bottlenecked earlier: it does not generate reachable non-`44` exact candidates for the scorer to rescue
  - mainline priority should shift from "better rerank" to "better proposal generation / projection from bridge to feasible non-44 states"

Thread-handoff summary for the next session:

What was tried today:

- Corrected the evaluation target:
  - moved the main question from mixed-seed "non-44 retention" to true `44-only escape`
  - confirmed that the relevant `class 44` seeds are only `129,130,131`
  - under the corrected protocol, old V1 strongest and all V1 scorer variants are still `0/3` escape

- Tested jump/corridor ideas in V2:
  - reference-guided jump bank from known non-`44` targets can reach `3/3` escape
  - but this is oracle-like and not a valid mainline method
  - unsupervised boundary-state cloud, boundary prototypes, and bridge-enabled unsupervised jump banks all remain `0/3`
  - anti-`44` jump proposals also remain `0/3`, although they can increase frontier size to `2` or `3`

- Tested supervised class-to-class transition on `322`:
  - full-mask supervised transition can reach near-perfect / perfect `322` fit
  - this was useful only as a proof that supervision can learn transitions
  - it is not a good transfer object for `422` because it is too close to a fixed-size lookup / decoder

- Clarified `322 -> 422` migration:
  - `322` and `422` differ in point-set size, feature dimension, and graph size
  - transferable object should be a structure-aware candidate scorer / search prior, not a class-conditioned full-mask decoder
  - wrote this down in:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/MIGRATION_322_TO_422.md`

- Built the first migration-friendly supervised prior:
  - added:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/graph_escape_scorer.py`
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/train_graph_escape_scorer.py`
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/eval_graph_escape_scorer.py`
  - current best lightweight baseline is the `SetEscapeScorer`
  - held-out result:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/graph_escape_scorer_holdout_set_v1/eval.json`
  - key metrics:
    - `val_accuracy = 97.90%`
    - `top1_escape_rate = 0.9928`
    - `top1_escape_rate_source44 = 1.0`
  - interpretation:
    - on a static exact-candidate set, the scorer can rank escape candidates very well

- Closed the loop by inserting the escape scorer back into real `44-only` jump search:
  - integrated scorer support into:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/infer_jump_escape.py`
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/eval_jump_escape.py`
  - evaluated:
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_anti44_v1_bridge_escape_scorer/multiseed_summary.json`
    - `/home/dministrator/workspace/AI_Bell/src/experiments/322/FacetExpensionV2/outputs/eval_jump_escape_unsupervised_v1_bridge_escape_scorer/multiseed_summary.json`
  - result:
    - still `0/3` escape
  - diagnosis:
    - scorer changes frontier dynamics, but the search is not generating reachable non-`44` exact candidates
    - so reranking alone cannot fix the real bottleneck

Current best interpretation:

- The problem is no longer "can we tell a good escape candidate from a bad one?" in an oracle/static sense.
- The bottleneck is proposal generation:
  - current local edits and unsupervised / anti-`44` jumps do not produce feasible candidates that actually land in a non-`44` exact basin.

Mainline next steps to preserve:

1. `bridge-to-feasible projection`
   - take anti-`44` or bridge states and learn / design a mechanism that projects them back toward the feasible low-energy manifold
   - goal:
     - generate reachable non-`44` exact candidates, not just wider bridge frontiers

2. Reverse-collapse analysis from `non-44 -> 44`
   - explicitly study how known non-`44` states fall back into `44`
   - use this reverse process to identify the key decision points / support changes that control basin collapse
   - potential use:
     - define anti-collapse supervision
     - identify which edits must be preserved or avoided when searching outward from `44`
     - build more targeted escape proposals around the "decision boundary" rather than around generic novelty
