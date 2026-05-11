# Data Generation

This directory contains scripts that generate or derive repository-level data
artifacts.  Generated artifacts should be written to `data/`, not to
`src/experiments`, because the final shareable project may omit the historical
experiment archive.

## Core 3-2-2 Facet Data

Generate the lrslib V-representation:

```bash
python3 data_generation/generate_polytope_322_ext.py
```

Generate the facet H-representation with `mplrs`:

```bash
python3 data_generation/run_lrs_facets_322.py --processes 4
```

This requires `external/lrslib-073a/mplrs` and `mpirun`. It reads:

```text
data/polytope_322.ext
```

and writes:

```text
data/facets_322.txt
```

Classify facets into symmetry classes and write the class table plus examples:

```bash
python3 data_generation/classify_facets.py
```

Outputs:

```text
data/facet_classes_322.tsv
data/facet_classes_322_examples.txt
```

Optional diagnostic:

```bash
python3 data_generation/check_facet_mean_parallel.py
```

Output:

```text
data/facet_classes_322_mean_parallel_check.txt
```

## 46x46 Orbit-Occupancy Matrix

Canonical data artifacts:

```text
data/class46x46_grouped_min_occupancy_matrix.html
data/class46x46_grouped_min_occupancy_matrix_summary.json
data/class46x46_matrix_summary.json
```

Regenerate the grouped HTML from the machine-readable summary:

```bash
python3 data_generation/generate_class46x46_grouped_min_occupancy_matrix.py
```

Derive block-only reachable classes:

```bash
python3 data_generation/derive_block_only_reachable_classes.py
```
