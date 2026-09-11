"""Print compat_class_* values for one orbit-block state."""

from __future__ import annotations

import argparse
import json

from baseline.orbit_blocks import build_orbit_patterns_from_support, empty_key
from baseline.reference_classes import parse_example_rows, support_mask_from_row

from .compatibility import ClassCompatibilityIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-class", type=int, default=8)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--selected-blocks", type=int, nargs="*", default=[])
    parser.add_argument("--classes", type=int, nargs="*", default=list(range(1, 47)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = parse_example_rows()
    support = support_mask_from_row(examples[args.pattern_class][args.rep_index])
    patterns = build_orbit_patterns_from_support(
        support,
        class_id=args.pattern_class,
        rep_index=args.rep_index,
        max_patterns=args.pattern_index + 1,
    )
    pattern = patterns[args.pattern_index]
    key = list(empty_key(len(pattern.orbits)))
    for block in args.selected_blocks:
        if block < 0 or block >= len(key):
            raise IndexError(f"selected block {block} is out of range")
        key[block] = 1
    index = ClassCompatibilityIndex.build(pattern.orbits, args.classes)
    counts = index.counts(key)
    print(json.dumps({
        "pattern_class": args.pattern_class,
        "rep_index": args.rep_index,
        "pattern_index": args.pattern_index,
        "block_count": len(pattern.orbits),
        "selected_blocks": sorted(set(args.selected_blocks)),
        "compat_counts": {f"compat_class_{class_id}": count for class_id, count in counts.items()},
        "compat_active_classes": sum(count > 0 for count in counts.values()),
        "compat_total_masks": sum(counts.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
