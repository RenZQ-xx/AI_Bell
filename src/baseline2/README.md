# baseline2

Clean CandidateFamily demo line for Bell 3-2-2 round0.

This package intentionally contains only the CandidateFamily search path:

- no `old_score`
- no proxy ranker
- no suppressed-compat runtime configuration
- no shared historical runner

Run from the repository root:

```bash
PYTHONPATH=src python -m baseline2 --max-steps 5 --cap 108
```

The default pattern is class 7, rep 1, pattern 0. Other legacy class
representative patterns can be selected without enabling any historical
ranker:

```bash
PYTHONPATH=src python -m baseline2 --row-class 8 --rep-index 1 --pattern-index 0 --max-steps 3
```

The search expands rank-gain action orbits into globally canonical
`CandidateFamily` children, demotes `i_A` polluted families, then orders by
supportability and flat capacity.

Full support representations are not materialized during online search by
default. Enable them only for diagnostics:

```bash
PYTHONPATH=src python -m baseline2 --materialize-support-representations
```
