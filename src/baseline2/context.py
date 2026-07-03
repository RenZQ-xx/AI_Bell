from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baseline2.config import GeometryConfig
from baseline2.core.clean_symmetry import build_partition_block_maps
from baseline2.core.geometry_oracle import GeometryOracle
from baseline2.core.supportability_oracle import SupportabilityOracle
from baseline2.primitives.bell322 import generate_bell322_points
from baseline2.primitives.orbit_blocks import build_orbit_patterns_from_support
from baseline2.primitives.reference_classes import parse_example_rows, support_mask_from_row
from baseline2.primitives.supportability import SupportabilityAnalyzer


@dataclass(frozen=True)
class CandidateFamilyContext:
    blocks: list[tuple[int, ...]]
    block_maps: list[tuple[int, ...]]
    geometry: GeometryOracle
    points: Any
    config: GeometryConfig


def build_context(
    *,
    row_class: int = 7,
    rep_index: int = 1,
    pattern_index: int = 0,
) -> CandidateFamilyContext:
    examples = parse_example_rows()
    support = support_mask_from_row(examples[int(row_class)][int(rep_index)])
    pattern = build_orbit_patterns_from_support(
        support,
        class_id=int(row_class),
        rep_index=int(rep_index),
        max_patterns=int(pattern_index) + 1,
    )[int(pattern_index)]
    blocks = [tuple(int(vertex) for vertex in block) for block in pattern.orbits]
    block_maps = [tuple(int(value) for value in row) for row in build_partition_block_maps(blocks)]
    points = generate_bell322_points()
    config = GeometryConfig()
    supportability = SupportabilityAnalyzer(
        points=points,
        rank_tol=config.rank_tol,
        support_tol=config.support_tol,
    )
    supportability_oracle = SupportabilityOracle.from_parts(
        supportability=supportability,
        config=config,
    )
    geometry = GeometryOracle.from_parts(
        blocks=blocks,
        points=points,
        config=config,
        rank_cache={},
        vertices_cache={},
        flat_cache={},
        affine_hull_cache={},
        incremental_affine_hull_enabled=True,
        supportability_oracle=supportability_oracle,
    )
    return CandidateFamilyContext(
        blocks=blocks,
        block_maps=block_maps,
        geometry=geometry,
        points=points,
        config=config,
    )


def build_class7_context() -> CandidateFamilyContext:
    return build_context(row_class=7, rep_index=1, pattern_index=0)
