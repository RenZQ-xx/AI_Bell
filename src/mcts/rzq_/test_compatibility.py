"""Regression checks for the extracted compatibility calculation."""

from __future__ import annotations

import unittest

from baseline.orbit_blocks import build_orbit_patterns_from_support, empty_key
from baseline.reference_classes import parse_example_rows, support_mask_from_row

from .compatibility import ClassCompatibilityIndex, project_support_to_blocks


class CompatibilityTests(unittest.TestCase):
    def test_partial_block_support_is_incompatible(self) -> None:
        blocks = ((0, 1), (2, 3))
        self.assertEqual(project_support_to_blocks((1, 1, 0, 0), blocks), frozenset({0}))
        self.assertIsNone(project_support_to_blocks((1, 0, 0, 0), blocks))

    def test_class8_known_state_counts(self) -> None:
        support = support_mask_from_row(parse_example_rows()[8][1])
        pattern = build_orbit_patterns_from_support(support, class_id=8, max_patterns=1)[0]
        index = ClassCompatibilityIndex.build(pattern.orbits)
        key = list(empty_key(len(pattern.orbits)))
        for block in (0, 7, 10):
            key[block] = 1
        counts = index.counts(key)
        self.assertEqual((counts[7], counts[8], counts[9]), (6, 12, 4))
        self.assertEqual(sum(count > 0 for count in counts.values()), 24)
        self.assertEqual(sum(counts.values()), 168)


if __name__ == "__main__":
    unittest.main()
