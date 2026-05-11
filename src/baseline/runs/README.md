Baseline run artifacts for the restored strict-log phase-aware search.

## strict_log_seed20260502_probe4.json

This is the restored May-3 static score reproduction for the second half of the
historical class1/pattern0 focus1-6 probe.

Settings:

```text
script: src/baseline/legacy_phase_aware_search.py
row_class = 1
rep_index = 1
pattern_index = 0
seed = 20260502
restarts_per_seed = 4
beam_width = 64
candidate_pool = 16
samples_per_state = 4
temperature = 0.85
max_blocks = 60
rare_target_classes = 1,2,3,4,5,6
target_classes = 1..46
rank24_entrance_exists_weight = 6.0
```

Result summary:

```text
final labels:
  exact:1, exact:24, exact:29, exact:2

encountered rare target classes:
  1,2,5,6

encountered counts:
  exact:1 = 30
  exact:2 = 20
  exact:5 = 68
  exact:6 = 12

class44:
  exact:44 = 79 / 1036 terminal encounters
```

The final labels match the second half of the old
`phase_aware_expansion_class1_pattern0_all46_focus1_6_probe8.json` run.

## modular_strict_log_seed20260502_probe4.json

This is the older modular probe4 run before restoring `child_flat_p50` in the
beam-diversity structural bucket:

```text
script: src/baseline/run_strict_log_probe.py
```

Result summary:

```text
final labels:
  exact:1, exact:19, exact:23, exact:2

encountered rare target classes:
  1,2,5,6

encountered counts:
  exact:1 = 28
  exact:2 = 20
  exact:5 = 68
  exact:6 = 27

class44:
  exact:44 = 96 / 1036 terminal encounters
```

The modular baseline has recovered the important rare encountered coverage, but
its final labels are not yet bit-for-bit identical to the legacy snapshot.  The
remaining difference is likely in the beam-diversity structural bucket and other
search-retention details, not in the main strict-log scoring channel.

## modular_strict_log_seed20260502_probe4_cf.json

This is the same probe after restoring the legacy `child_flat_p50` bucket
dimension in the modular baseline.

Result summary:

```text
final labels:
  exact:1, exact:24, exact:29, exact:2

encountered rare target classes:
  1,2,5,6

encountered counts:
  exact:1 = 30
  exact:2 = 20
  exact:5 = 68
  exact:6 = 12

class44:
  exact:44 = 79 / 1036 terminal encounters
```

This matches the legacy probe4 final labels and rare encountered counts.

## modular_strict_log_seed20260501_20260502_probe8_cf.json

This is the full modular 8-restart probe after restoring `child_flat_p50` in
the structural bucket.  It uses seeds `20260501` and `20260502`, with four
restarts per seed.

Result summary:

```text
final labels:
  exact:5, exact:4, exact:43, exact:29,
  exact:1, exact:24, exact:29, exact:2

encountered rare target classes:
  1,2,3,4,5,6

encountered counts:
  exact:1 = 30
  exact:2 = 67
  exact:3 = 38
  exact:4 = 11
  exact:5 = 107
  exact:6 = 12

class44:
  exact:44 = 162 / 2071 terminal encounters
```

This matches the historical probe8 summary for class1-6 encountered coverage
and final label counts.

## modular_strict_log_class23_pattern0_probe16.json

This applies the restored modular strict-log baseline to the class23/pattern0
rare-tail setting, without doing any score ablation.  The purpose is to compare
the current baseline behavior against the old `ablations_class23_score` full
condition, not to tune the score.

Settings:

```text
row_class = 23
rep_index = 1
pattern_index = 0
seeds = 20260429,20260430
restarts_per_seed = 8
beam_width = 64
candidate_pool = 12
samples_per_state = 4
temperature = 0.75
max_blocks = 24
rare_target_classes = 35,38,39,41,45,46
target_classes = 23,35,38,39,41,43,44,45,46
rank24_entrance_exists_weight = 4.0
```

Result summary:

```text
final labels:
  exact:35 = 6
  exact:38 = 3
  exact:39 = 1
  exact:45 = 3
  exact:46 = 3

encountered rare target classes:
  35,38,39,41,45,46

encountered counts:
  exact:35 = 419
  exact:38 = 150
  exact:39 = 257
  exact:41 = 21
  exact:45 = 478
  exact:46 = 184

terminal totals:
  exact = 3733
  invalid = 2472
  invalid_rate = 39.84%
```

Comparison to the old `ablations_class23_score_probe16_summary.md` full row:

```text
final rare coverage:
  current = 5/6
  old full = 5/6

encountered rare coverage:
  current = 6/6
  old full = 6/6

final label counts:
  current = exact:35 x6, exact:38 x3, exact:39 x1, exact:45 x3, exact:46 x3
  old full = exact:35 x6, exact:38 x3, exact:39 x1, exact:45 x3, exact:46 x3

class41 encountered:
  current = 21
  old full = 24
```

The class23 result confirms that the restored baseline preserves the useful
class23 rare-tail behavior without changing the score.

## modular_strict_log_class7_pattern0_broad_no43_44_probe16.json

This applies the restored modular strict-log baseline to class7/pattern0 with
the broad class7-compatible rare set, excluding the strong 43/44 basins from
rare reward while keeping them as target classes.  This is the direct modular
reproduction target for the old
`phase_aware_expansion_class7_pattern0_supportability_gate_probe16.json`.

Result summary:

```text
final labels:
  exact:28 = 5
  exact:29 = 3
  exact:42 = 8

encountered rare target classes:
  7,8,12,15,19,20,25,28,29,30,31,32,35,42,45,46

encountered rare coverage:
  16 / 24

class43/class44:
  exact:43 = 528
  exact:44 = 70
```

This matches the old class7 supportability-gate probe16 result exactly on final
labels, rare encountered coverage, and class43/class44 counts.

## modular_strict_log_class7_pattern0_low_7_22_probe16.json

This is a class1-style focused class7 probe.  The rare set is restricted to the
lower-numbered class7-compatible classes:

```text
7,8,9,10,11,12,15,19,20,22
```

The target set remains all 26 class7-compatible classes.

Result summary:

```text
final labels:
  exact:7 = 3
  exact:8 = 1
  exact:12 = 1
  exact:15 = 4
  exact:19 = 4
  exact:20 = 2
  exact:42 = 1

encountered rare target classes:
  7,8,12,15,19,20

missing rare target classes:
  9,10,11,22

encountered rare coverage:
  6 / 10

class43/class44:
  exact:43 = 888
  exact:44 = 502
```

This shows a clear difficulty split inside the lower-numbered class7-compatible
classes: rewarding them increases hits for 7/8/12/15/19/20, but 9/10/11/22
remain absent under the restored baseline.

## Class13 Pattern0 Notes

Class13/pattern0 has two separate records that must not be collapsed.

The first record is the restored modular strict-log baseline with static
terminal scoring:

```text
file:
  src/baseline/runs/modular_strict_log_class13_pattern0_probe16.json

row_class = 13
rep_index = 1
pattern_index = 0
seeds = 20260506,20260507
restarts_per_seed = 8
candidate_pool = 12
max_blocks = 64
rare_target_classes = 13,16,24,25,33,34,40
target_classes = 13,16,24,25,33,34,40
terminal_scoring_mode = static
```

Result summary:

```text
final labels:
  exact:13 = 7
  exact:16 = 9

encountered rare target classes:
  13,16,24,25,33,40

missing rare target class:
  34

encountered counts:
  exact:13 = 188
  exact:16 = 1085
  exact:24 = 137
  exact:25 = 1088
  exact:33 = 32
  exact:34 = 0
  exact:40 = 1126

encountered rare coverage:
  6 / 7
```

The second record is the original experimental dynamic-terminal result:

```text
file:
  src/experiments/322/OrbitStabilizerSearch/runs/
  phase_aware_expansion_class13_pattern0_dynamic_terminal_probe16.json

terminal_scoring_mode = dynamic
dynamic_new_class_score = 100
dynamic_known_class_score = 10
dynamic_frequent_class_score = -5
dynamic_frequent_class_threshold = 16
```

Original dynamic result summary:

```text
final labels:
  exact:13 = 3
  exact:16 = 8
  exact:24 = 2
  exact:25 = 1
  exact:33 = 2

encountered rare target classes:
  13,16,24,25,33,34,40

encountered counts:
  exact:13 = 52
  exact:16 = 155
  exact:24 = 35
  exact:25 = 103
  exact:33 = 25
  exact:34 = 59
  exact:40 = 126

encountered rare coverage:
  7 / 7
```

Do not describe the dynamic result as restored by the current static baseline.
The current baseline result is 6/7 encountered and misses class34; the original
dynamic result is 7/7 encountered.

## Dynamic Terminal Interface Policy

The modular runner exposes `--terminal-scoring-mode dynamic` only as an explicit
compatibility interface for old dynamic-terminal experiments.  It must remain
off by default.

Use this interface carefully:

```text
default baseline:
  terminal_scoring_mode = static

dynamic compatibility:
  terminal_scoring_mode = dynamic
  requires a named old JSON target and a separate output file
```

Do not tune the score or switch the default to chase a dynamic result.  Before
claiming a dynamic reproduction, compare against the old JSON meta and, when
possible, replay saved paths or encountered examples.  This rule is here to
avoid repeating the earlier script-drift mistake where later workflow changes
were mistaken for the historical baseline.

## modular_strict_log_class14_pattern0_probe16.json

This applies the restored modular strict-log baseline to class14/pattern0 using
the target set from the original dynamic-terminal experiment, but keeps terminal
scoring static.

Settings:

```text
row_class = 14
rep_index = 1
pattern_index = 0
seeds = 20260506,20260507
restarts_per_seed = 8
candidate_pool = 12
max_blocks = 64
terminal_scoring_mode = static
rare_target_classes = 14,17,21,26,27,32,33,36,37,39,43,45
target_classes = 14,17,21,26,27,32,33,36,37,39,43,45
```

Result summary:

```text
final labels:
  exact:27 = 2
  exact:43 = 14

encountered rare target classes:
  14,17,21,26,27,32,33,36,37,39,43,45

encountered counts:
  exact:14 = 197
  exact:17 = 143
  exact:21 = 77
  exact:26 = 3
  exact:27 = 122
  exact:32 = 40
  exact:33 = 7
  exact:36 = 8
  exact:37 = 62
  exact:39 = 18
  exact:43 = 3352
  exact:45 = 185

encountered rare coverage:
  12 / 12
```

The original dynamic-terminal class14 experiment also encountered all 12 target
classes, but opened a more diverse final set.  Under static terminal scoring,
the restored baseline still gets full encountered coverage while final
selection is dominated by class43.

## modular_strict_log_class28_pattern0_probe16.json

This applies the restored modular strict-log baseline to the class28/pattern0
family, using the target set from the original
`phase_aware_expansion_class28_pattern0_dynamic_terminal_probe16.json`, but
keeping terminal scoring static.

Note that classes 28, 29, 30, and 31 share the same pattern0 orbit partition.
Older hunt artifacts sometimes name this family as `class29_pattern0`; the
phase-aware probe16 artifact names it `class28_pattern0`.

Settings:

```text
row_class = 28
rep_index = 1
pattern_index = 0
seeds = 20260506,20260507
restarts_per_seed = 8
candidate_pool = 12
max_blocks = 64
terminal_scoring_mode = static
rare_target_classes = 28,29,30,31,38,42,44,46
target_classes = 28,29,30,31,38,42,44,46
```

Result summary:

```text
final labels:
  exact:29 = 5
  exact:42 = 2
  exact:44 = 9

encountered rare target classes:
  28,29,30,31,38,42,44,46

encountered counts:
  exact:28 = 87
  exact:29 = 200
  exact:30 = 1
  exact:31 = 26
  exact:38 = 2
  exact:42 = 63
  exact:44 = 3619
  exact:46 = 97

encountered rare coverage:
  8 / 8
```

The original dynamic-terminal class28 experiment also encountered all 8 target
classes and opened a more diverse final set.  Under static terminal scoring,
the restored baseline still gets full encountered coverage, with final
selection more strongly dominated by class44.

## Additional Static Pattern0 Reproductions

The following runs apply the restored static baseline to the remaining
phase-aware pattern0 targets from the old dynamic-terminal experiments.  Each
run uses the old experiment's `row_class`, `rep_index`, `pattern_index`,
`rare_target_classes`, `target_classes`, seeds, restart count, candidate pool,
and max-block setting, but keeps:

```text
terminal_scoring_mode = static
```

Output files follow this naming rule:

```text
src/baseline/runs/modular_strict_log_classXX_pattern0_probe16.json
```

Comparison summary:

| class | target classes | old dynamic encountered | static encountered | static final labels |
|---:|---|---|---|---|
| 18 | 18,26,35,36,37,39,45,46 | 7/8: 18,26,35,37,39,45,46 | 7/8: 18,26,35,37,39,45,46 | exact:18x3, exact:35x1, exact:37x9, exact:45x2, exact:46x1 |
| 26 | 26,36,37,39 | 4/4: 26,36,37,39 | 4/4: 26,36,37,39 | exact:26x5, exact:36x11 |
| 27 | 27,45 | 2/2: 27,45 | 2/2: 27,45 | exact:27x9, exact:45x7 |
| 32 | 32,45 | 2/2: 32,45 | 2/2: 32,45 | exact:32x12, exact:45x4 |
| 35 | 35,45,46 | 3/3: 35,45,46 | 3/3: 35,45,46 | exact:35x12, exact:45x4 |
| 36 | 36 | 1/1: 36 | 1/1: 36 | exact:36x16 |
| 37 | 37 | 1/1: 37 | 1/1: 37 | exact:37x16 |
| 40 | 40 | 1/1: 40 | 1/1: 40 | exact:40x16 |
| 38 | 38,44,46 | 3/3: 38,44,46 | 3/3: 38,44,46 | exact:38x9, exact:46x7 |
| 39 | 39 | 1/1: 39 | 1/1: 39 | exact:39x16 |
| 42 | 42,46 | 2/2: 42,46 | 2/2: 42,46 | exact:42x10, exact:46x6 |
| 43 | 43 | 1/1: 43 | 1/1: 43 | exact:43x16 |
| 41 | 41 | 1/1: 41 | 1/1: 41 | exact:41x16 |
| 44 | 44 | 1/1: 44 | 1/1: 44 | exact:44x16 |
| 45 | 45 | 1/1: 45 | 1/1: 45 | exact:45x16 |
| 46 | 46 | 1/1: 46 | 1/1: 46 | exact:46x16 |
| 25 | 25,40 | 2/2: 25,40 | 2/2: 25,40 | exact:25x11, exact:40x5 |
| 33 | 33 | 1/1: 33 | 1/1: 33 | exact:33x16 |
| 34 | 34 | 1/1: 34 | 1/1: 34 | exact:34x16 |

Main reading:

```text
static baseline preserves encountered coverage for this batch.
```

For class18, both the old dynamic run and the static baseline hit 7/8 targets
and miss class36.  For all other rows in this batch, the static baseline hits
the same target coverage as the old dynamic result.  Final labels can differ
because dynamic terminal scoring was designed to diversify opened classes,
whereas the restored static baseline rewards each rare target uniformly.
