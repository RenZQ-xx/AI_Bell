# Static Orbit-Block Baseline

This is the clean baseline for the Bell 3-2-2 orbit-block search line.

Read this file first.  The older baseline notes were merged here so that a new
collaborator can understand the problem, run the code, and avoid repeating the
script-drift mistakes that broke earlier reproductions.

## What We Are Solving

Bell 3-2-2 has 64 deterministic vertices.  The known facets are grouped into 46
exact classes.  For a fixed facet class representative, we take its stabilizer
orbit partition, start from the empty support, and repeatedly add one orbit
block.  The question is:

```text
Given one source pattern, which exact facet classes can the search encounter?
```

Important terms:

- `static baseline`: terminal scoring is fixed by configured target/rare sets.
- `zero-start add-block`: search starts from the all-zero block key and only adds new blocks.
- `pattern experiment`: a row is a stabilizer orbit pattern, not necessarily one unique class.  For example, `28/29/30/31 pattern0` is one shared pattern family.
- `encountered hit`: the search generated an exact terminal candidate.  The main summary table uses encountered hits, not only final best labels.
- `rare class`: a class given high terminal reward in that run.  This affects search preference but is not identical to the representable range.

## Directory Map

Core baseline code:

```text
src/baseline/
  bell322.py                  # 64 x 26 Bell 3-2-2 deterministic point table
  reference_classes.py        # data/facets_322.txt and representative rows -> exact class reference
  geometry.py                 # affine rank, hyperplane fit, supporting facet checks
  facet_validator.py          # terminal support -> exact:classXX or invalid label
  orbit_blocks.py             # stabilizer orbit patterns and block-level states
  supportability.py           # low-rank supportability heuristic
  scorer.py                   # phase-aware score; static terminal scoring by default
  search.py                   # zero-start add-block beam search
  run_strict_log_probe.py     # main modular entry point
  legacy_phase_aware_search.py # historical snapshot only; see below
  checks/validate_geometry.py # data/classifier sanity check
  runs/README.md              # experiment result log
  summary/                    # static baseline summary HTML and generator
```

Repository-level data:

```text
data/
  facets_322.txt
  facet_classes_322.tsv
  facet_classes_322_examples.txt
  polytope_322.ext
  class46x46_grouped_min_occupancy_matrix.html
  class46x46_grouped_min_occupancy_matrix_summary.json
  class46x46_matrix_summary.json
  block_only_reachable_classes_summary.json
```

Data-generation scripts:

```text
data_generation/
  generate_polytope_322_ext.py
  run_lrs_facets_322.py
  classify_facets.py
  check_facet_mean_parallel.py
  generate_class46x46_grouped_min_occupancy_matrix.py
  derive_block_only_reachable_classes.py
```

The final shareable project may omit `src/experiments`.  The modular baseline
must therefore use `data/` and `data_generation/`, not hidden files from
`src/experiments`.

## How One Probe Works

For a run such as `class13 pattern0 static`:

1. `run_strict_log_probe.py` reads a representative row from `data/facet_classes_322_examples.txt`.
2. `support_mask_from_row()` converts the inequality row into a 64-bit tight-support mask.
3. `build_orbit_patterns_from_support()` computes the stabilizer orbit partition.
4. Search state is a block-level tuple such as `(0, 0, 1, 0, ...)`.
5. `search.py` starts from all zeros and adds one block at a time.
6. `scorer.py` ranks children using rank growth, flatness, supportability, rank-24 entrance, and terminal labels.
7. `facet_validator.py` validates rank-25 terminal supports and maps them to exact classes.
8. The output JSON records encountered labels, final best labels, and per-restart terminal bests.

Key JSON fields:

```text
summary.encountered_label_counts
summary.label_counts
summary.encountered_rare_target_classes
runs[*].terminal_bests
```

## Main Commands

Run the geometry/classifier sanity check:

```bash
cd /home/ap809/workspace/AI_Bell
PYTHONPATH=src .venv/bin/python -m baseline.checks.validate_geometry
```

Expected high-level result:

```text
passed: true
points_shape: [64, 26]
classes_seen: 46
example_rows_checked: 138
exact_matches: 138
```

Run a class1 probe4:

```bash
cd /home/ap809/workspace/AI_Bell
PYTHONPATH=src .venv/bin/python -m baseline.run_strict_log_probe \
  --row-class 1 \
  --rep-index 1 \
  --pattern-index 0 \
  --rare-target-classes 1 2 3 4 5 6 \
  --target-classes $(seq 1 46) \
  --seeds 20260502 \
  --restarts-per-seed 4 \
  --candidate-pool 16 \
  --temperature 0.85 \
  --max-blocks 60 \
  --rank24-entrance-exists-weight 6.0 \
  --output src/baseline/runs/my_probe4.json
```

Regenerate the static baseline summary table:

```bash
PYTHONPATH=src .venv/bin/python src/baseline/summary/generate_static_baseline_matrix.py
```

Open:

```text
src/baseline/summary/static_baseline_pattern_matrix.html
```

## Current Reproduction Status

The result log lives in:

```text
src/baseline/runs/README.md
```

The most important restored class1 results are:

```text
modular_strict_log_seed20260502_probe4_cf.json
  final labels: exact:1, exact:24, exact:29, exact:2
  encountered rare classes: 1,2,5,6
  class44: exact:44 = 79 / 1036 terminal encounters

modular_strict_log_seed20260501_20260502_probe8_cf.json
  final labels: exact:5, exact:4, exact:43, exact:29,
                exact:1, exact:24, exact:29, exact:2
  encountered rare classes: 1,2,3,4,5,6
```

The full static pattern reproduction summary is:

```text
src/baseline/summary/static_baseline_pattern_matrix.html
```

Table semantics:

- Rows are merged representative pattern experiments.
- Columns are the 46 classes in the same grouped order as the strict 46x46 matrix.
- Green cells are exact encountered hits.
- Pink cells are representable/target classes that were missed.
- Blank cells are outside that row's target range.
- A number is the exact encountered count.
- `*` marks classes configured as rare in that run.

Important reading anchors:

- `class1 pattern0 all46 focus1-6`: restores rare encountered coverage for 1-6.
- `class7 pattern0 broad no43/44`: target includes 43/44, but rare excludes 43/44.
- `class13 pattern0 static`: static hits 6/7 and misses class34; the old dynamic-terminal result hit 7/7.
- `class18 pattern0 static`: static and old dynamic both hit 7/8 and miss class36.

## Static vs Dynamic Terminal Scoring

The default baseline is static:

```text
ScorerConfig.terminal_scoring_mode = "static"
```

Static terminal scoring:

```text
boundary invalid: -20 - 2 * closer_side
rare target:      100
class44:          -5
target class:     10
other exact:      1
invalid:          -10
```

Dynamic terminal scoring exists only as an explicit compatibility interface for
old dynamic-terminal experiments.  Do not switch the default to dynamic and do
not tune the score to chase one row.  The class13 case is the cautionary
example: the old dynamic-terminal experiment found class34, while the current
static baseline does not.

## Legacy Snapshot And Provenance

`src/baseline/legacy_phase_aware_search.py` is not the teaching entry point and
not the future development target.  Its job is to preserve a restored historical
snapshot that anchors the May-3 strict-log static reproduction.

Why it exists:

1. The old target run was:

   ```text
   src/experiments/322/OrbitStabilizerSearch/runs/
   phase_aware_expansion_class1_pattern0_all46_focus1_6_probe8.json
   ```

2. That run was created on 2026-05-01 14:14, but the original source file was
   later recreated/modified around 2026-05-09.  There is no tracked copy of the
   exact 2026-05-01 script.

3. The recovery work happened in:

   ```text
   src/experiments/322/OrbitStabilizerSearch/analysis_tmp/
   historical_20260501_replay/
   ```

   Key forensic files there include:

   ```text
   README.md
   replay_focus1_6_probe8.py
   focus1_6_probe8_replay_report.json
   strict_log_seed20260502_probe4.json
   saved_*_path_diagnostic_*.json/csv
   ```

4. That replay verified that the saved terminal supports in the old JSON still
   classify as their claimed exact labels.  The investigation then restored the
   May-3 strict-log settings: static terminal scoring, nonpositive rank-gain
   penalty, child-rank flat capacity, key-dependent supportability, augmented
   directions, bucket-stochastic diversity, and rank24 entrance.

5. `legacy_phase_aware_search.py` is the copied-out historical reproduction
   snapshot from that recovery line.  It still imports old experimental modules
   from `src/experiments/322/...`, so it will not be runnable in a final package
   that omits `src/experiments`.

The key preserved legacy artifact is:

```text
src/baseline/runs/strict_log_seed20260502_probe4.json
```

It reproduced the second half of the old probe8:

```text
final labels:
  exact:1, exact:24, exact:29, exact:2

encountered rare classes:
  1,2,5,6

encountered counts:
  exact:1 = 30
  exact:2 = 20
  exact:5 = 68
  exact:6 = 12
```

The modular code in `run_strict_log_probe.py`, `scorer.py`, and `search.py`
then became the clean baseline.  After restoring the `child_flat_p50` structural
bucket, the modular probe4 matched the legacy final labels and rare encountered
counts.

Practical rule:

```text
Use legacy_phase_aware_search.py for provenance only.
Use run_strict_log_probe.py for new runs and collaboration.
```

## Lessons And Guardrails

Do not infer old behavior from current exploratory defaults.  Earlier failures
came from treating the later `phase_aware_expansion_search.py` as if it were the
old script plus harmless options.  It was not.

Major drift sources included:

```text
terminal scoring changed from static rare targets to dynamic discovery
rank_gain <= 0 penalty was removed
flat_capacity default changed
rank24 entrance default changed
beam diversity default changed
supportability settings changed
supportability augmented directions were accidentally disabled in diagnostics
```

Important guardrails:

- Do not casually change score terms.
- Keep static terminal scoring as the default.
- Separate expressibility, encountered, and final/opened.
- Do not read one pattern row as 46 independent class experiments.
- Add new interfaces cautiously and document defaults in `runs/README.md`.
- New experiments should write new JSON files, not overwrite baseline artifacts.
- Use WSL `.venv`; avoid Windows Python on `\\wsl$` paths for heavy probes.

## Final Package Checklist

If the uploaded version omits `src/experiments`, keep:

```text
src/baseline/
data/
data_generation/
```

The modular baseline is self-contained with those directories.  The only
remaining hard dependency on `src/experiments` is the historical snapshot:

```text
src/baseline/legacy_phase_aware_search.py
```

For a clean final package, either exclude that file or keep it clearly marked as
historical provenance only.  The runnable baseline does not depend on it.

