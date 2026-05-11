from __future__ import annotations

"""Legacy strict-log phase-aware search baseline.

This is the restored May-3 static phase-aware search driver, copied out of the
exploratory OrbitStabilizerSearch tree so it can serve as a clean baseline
artifact.  It fixes one orbit partition, starts from the empty block mask, and
repeatedly adds one orbit block.  Intermediate states are ranked by heuristic
geometric scores; terminal states are still certified by the exact
DIffUCO/classifier stack.

The three most important ideas are:

1. add-only construction: state = selected orbit blocks, action = add one block;
2. phase-aware scoring: early rank growth, mid-rank supportability, late exits;
3. exact accounting: every exact hit is recorded as encountered/final evidence.
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    BASELINE_DIR = Path(__file__).resolve().parent
    SRC_DIR = BASELINE_DIR.parent
    PROJECT_ROOT = SRC_DIR.parent
    OSS_ROOT = SRC_DIR / "experiments" / "322" / "OrbitStabilizerSearch"
    AI_GUIDED_DIR = OSS_ROOT / "ai_guided"
    ROOT_322 = OSS_ROOT.parent
    CORE_DIR = OSS_ROOT / "core"
    STRICT_DIR = ROOT_322 / "StrictStabilizerSearch"
    DIFFUCO_DIR = ROOT_322 / "DIffUCO"
    LRS_DIR = ROOT_322 / "lrs"
    sys.path[:0] = [
        str(AI_GUIDED_DIR),
        str(CORE_DIR),
        str(STRICT_DIR),
        str(DIFFUCO_DIR),
        str(LRS_DIR),
        str(ROOT_322),
    ]
    from classify_facets import _parse_hrep_rows
    from energy_factory import build_energy
    from DIffUCO.facet_reference import build_reference_database
    from DIffUCO.geometry import generate_points_322
    from DIffUCO.inference import classify_hard_mask
    from search_utils import classify_label
    from stabilizer_orbit_patterns import build_class_orbit_patterns, point_rows, support_mask_from_row
    from summarize_lrs_pattern_block_masks import block_mask_for_support, row_class_map
else:
    from ...lrs.classify_facets import _parse_hrep_rows
    from ..core.energy_factory import build_energy
    from ..core.search_utils import classify_label
    from ...DIffUCO.facet_reference import build_reference_database
    from ...DIffUCO.geometry import generate_points_322
    from ...DIffUCO.inference import classify_hard_mask
    from ...StrictStabilizerSearch.stabilizer_orbit_patterns import (
        build_class_orbit_patterns,
        point_rows,
        support_mask_from_row,
    )
    from .summarize_lrs_pattern_block_masks import block_mask_for_support, row_class_map


# A block-level 0/1 tuple.  It is used both for live search states and for
# block-constant supports imported from known LRS facets.
BlockKey = Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Add-only phase-aware expansion search over class23 orbit blocks."
    )
    parser.add_argument("--facets", type=Path, default=root / "data" / "facets_322.txt")
    parser.add_argument("--row-class", type=int, default=23)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=[35, 38, 39, 41, 45, 46])
    parser.add_argument("--target-classes", type=int, nargs="*", default=[23, 35, 38, 39, 41, 43, 44, 45, 46])
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260429, 20260430, 20260501, 20260502])
    parser.add_argument("--restarts-per-seed", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=12)
    parser.add_argument("--shuffle-candidate-ties", choices=["on", "off"], default="off")
    parser.add_argument("--samples-per-state", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max-blocks", type=int, default=24)
    parser.add_argument("--rank-tol", type=float, default=1e-6)
    parser.add_argument("--support-tol", type=float, default=1e-6)
    parser.add_argument("--phase-b-start-rank", type=int, default=22)
    parser.add_argument("--phase-boundary-mode", choices=["fit_shift", "supportability", "supportability_gate", "energy_delta", "none"], default="supportability_gate")
    parser.add_argument("--supportability-direction-samples", type=int, default=512)
    parser.add_argument("--supportability-seed", type=int, default=20260430)
    parser.add_argument(
        "--supportability-direction-bank",
        choices=["legacy_rng", "key_dependent", "global_cached"],
        default="key_dependent",
    )
    parser.add_argument(
        "--supportability-augment-directions",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--supportability-selection",
        choices=["min_shift", "min_penalty"],
        default="min_shift",
    )
    parser.add_argument(
        "--supportability-schedule",
        choices=["fixed", "adaptive_a", "adaptive_b", "late20", "late22", "late23"],
        default="fixed",
    )
    parser.add_argument("--supportability-start-rank", type=int, default=0)
    parser.add_argument("--supportability-gate-weight", type=float, default=6.0)
    parser.add_argument("--supportability-gate-closer-weight", type=float, default=1.0)
    parser.add_argument("--beam-diversity-slots", type=int, default=16)
    parser.add_argument("--beam-diversity-mode", choices=["bucket_release", "bucket_stochastic"], default="bucket_stochastic")
    parser.add_argument("--beam-diversity-near-multiple", type=int, default=4)
    parser.add_argument("--beam-diversity-rank-start", type=int, default=17)
    parser.add_argument("--beam-diversity-rank-end", type=int, default=21)
    parser.add_argument("--beam-diversity-temperature", type=float, default=0.25)
    parser.add_argument("--beam-diversity-late-slots", type=int, default=0)
    parser.add_argument("--beam-diversity-late-rank-start", type=int, default=22)
    parser.add_argument("--beam-diversity-late-rank-end", type=int, default=23)
    parser.add_argument("--rank24-entrance-weight", type=float, default=1.0)
    parser.add_argument("--rank24-entrance-exists-weight", type=float, default=4.0)
    parser.add_argument("--rank24-class44-entrance-weight", type=float, default=1.0)
    parser.add_argument("--rank24-invalid-entrance-weight", type=float, default=1.0)
    parser.add_argument("--rank-gain-weight-scale", type=float, default=1.0)
    parser.add_argument("--nonpositive-rank-gain-penalty", type=float, default=8.0)
    parser.add_argument("--flat-penalty-weight-scale", type=float, default=1.0)
    parser.add_argument("--flat-capacity-method", choices=["child_rank", "affine_hull"], default="child_rank")
    parser.add_argument("--terminal-scoring-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--invalid-rank-terminal-score", type=float, default=-100.0)
    parser.add_argument("--dynamic-new-class-score", type=float, default=100.0)
    parser.add_argument("--dynamic-known-class-score", type=float, default=10.0)
    parser.add_argument("--dynamic-frequent-class-score", type=float, default=-5.0)
    parser.add_argument("--dynamic-frequent-class-threshold", type=int, default=16)
    parser.add_argument("--class44-terminal-score", type=float, default=-5.0)
    parser.add_argument("--initial-discovered-classes", type=int, nargs="*", default=[])
    parser.add_argument("--initial-frequent-classes", type=int, nargs="*", default=[])
    parser.add_argument("--discovered-repulsion-weight", type=float, default=0.0)
    parser.add_argument("--discovered-repulsion-rank-start", type=int, default=21)
    parser.add_argument("--discovered-repulsion-min-blocks", type=int, default=6)
    parser.add_argument("--discovered-repulsion-power", type=float, default=2.0)
    parser.add_argument("--discovered-repulsion-mixed-alpha", type=float, default=2.0)
    parser.add_argument("--discovered-repulsion-require-supportable", action="store_true")
    parser.add_argument("--discovered-repulsion-supportable-closer-max", type=float, default=0.0)
    parser.add_argument("--discovered-repulsion-classes", type=int, nargs="*", default=[23, 35, 38, 39, 43, 44, 45, 46])
    parser.add_argument("--class44-compat-target-class", type=int, default=44)
    parser.add_argument("--class44-compat-rank-start", type=int, default=23)
    parser.add_argument("--class44-compat-rank-end", type=int, default=24)
    parser.add_argument("--class44-compat-zero-bonus", type=float, default=0.0)
    parser.add_argument("--class44-compat-escape-bonus", type=float, default=0.0)
    parser.add_argument("--class44-compat-zero-slots", type=int, default=0)
    parser.add_argument("--class44-compat-positive-slots", type=int, default=0)
    parser.add_argument("--class44-compat-near-multiple", type=int, default=4)
    parser.add_argument("--secondary-compat-target-class", type=int, default=43)
    parser.add_argument("--secondary-compat-zero-within-primary-zero-slots", type=int, default=0)
    parser.add_argument("--secondary-compat-zero-slots", type=int, default=0)
    parser.add_argument("--secondary-compat-positive-slots", type=int, default=0)
    parser.add_argument("--secondary-compat-rank-start", type=int, default=21)
    parser.add_argument("--secondary-compat-rank-end", type=int, default=24)
    parser.add_argument("--joint-compat-rank-start", type=int, default=20)
    parser.add_argument("--joint-compat-rank-end", type=int, default=24)
    parser.add_argument("--joint-compat-00-slots", type=int, default=0)
    parser.add_argument("--joint-compat-01-slots", type=int, default=0)
    parser.add_argument("--joint-compat-10-slots", type=int, default=0)
    parser.add_argument("--joint-compat-11-slots", type=int, default=0)
    parser.add_argument("--energy-version", choices=["old", "v2", "v22", "v23", "v24", "v24nogate", "v3"], default="v24nogate")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--discovery-patience-restarts", type=int, default=0)
    parser.add_argument("--discovery-min-restarts", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ap809/workspace/AI_Bell/src/experiments/322/OrbitStabilizerSearch/runs/phase_aware_expansion_search_class23_pattern0.json"),
    )
    return parser.parse_args()


def empty_key(block_count: int) -> BlockKey:
    return tuple(0 for _ in range(block_count))


def add_block(key: BlockKey, block_index: int) -> BlockKey:
    values = list(key)
    values[int(block_index)] = 1
    return tuple(values)


def selected_blocks(key: BlockKey) -> List[int]:
    return [idx for idx, value in enumerate(key) if int(value) == 1]


def profile_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "profile_timing", False))


def profile_add(args: argparse.Namespace, name: str, seconds: float, count: int = 1) -> None:
    if not profile_enabled(args):
        return
    timings = getattr(args, "_profile_timings", None)
    if timings is None:
        timings = {}
        setattr(args, "_profile_timings", timings)
    bucket = timings.setdefault(name, {"seconds": 0.0, "count": 0})
    bucket["seconds"] += float(seconds)
    bucket["count"] += int(count)


def effective_supportability_samples(rank: int, args: argparse.Namespace) -> int:
    base = int(getattr(args, "supportability_direction_samples", 512))
    schedule = str(getattr(args, "supportability_schedule", "fixed"))
    if schedule == "adaptive_a":
        if rank < 20:
            return min(base, 128)
        if rank < 23:
            return min(base, 256)
        return base
    if schedule == "adaptive_b":
        return min(base, 128) if rank < 22 else base
    return base


def supportability_enabled_for_rank(rank: int, args: argparse.Namespace) -> bool:
    if rank < int(getattr(args, "supportability_start_rank", 0)):
        return False
    schedule = str(getattr(args, "supportability_schedule", "fixed"))
    if schedule == "late20":
        return rank >= 20
    if schedule == "late22":
        return rank >= 22
    if schedule == "late23":
        return rank >= 23
    return True


def exact_class_from_label(label: str) -> int | None:
    if not label.startswith("exact:"):
        return None
    return int(label.split(":", 1)[1])


def collect_block_masks_local(
    *,
    facets_path: Path,
    classes: Sequence[int],
    blocks: Sequence[Sequence[int]],
) -> Dict[int, List[Dict[str, object]]]:
    rows = _parse_hrep_rows(facets_path)
    class_by_row = row_class_map(rows)
    point_cache = point_rows()
    wanted = set(int(class_id) for class_id in classes)
    by_class: Dict[int, Dict[BlockKey, Dict[str, object]]] = {class_id: {} for class_id in wanted}
    for row in rows:
        class_id = int(class_by_row[row])
        if class_id not in wanted:
            continue
        support = support_mask_from_row(row, point_cache)
        support_key = tuple(int(value) for value in support.tolist())
        block_mask, _mixed_count = block_mask_for_support(support_key, blocks)
        if block_mask is None:
            continue
        bucket = by_class[class_id]
        if block_mask not in bucket:
            bucket[block_mask] = {
                "class_id": class_id,
                "chosen_blocks": [idx for idx, value in enumerate(block_mask) if int(value) == 1],
            }
    return {
        class_id: sorted(bucket.values(), key=lambda item: item["chosen_blocks"])
        for class_id, bucket in by_class.items()
    }


class FaceRepulsion:
    def __init__(self, *, blocks: Sequence[Sequence[int]], faces: Sequence[Tuple[int, frozenset[int]]]) -> None:
        self.blocks = [tuple(int(v) for v in block) for block in blocks]
        self.faces = list(faces)
        self.cache: Dict[Tuple[BlockKey, float, float, int], Dict[str, float]] = {}

    def key_points(self, key: BlockKey) -> frozenset[int]:
        return frozenset(
            vertex
            for block_index, selected in enumerate(key)
            if int(selected) == 1
            for vertex in self.blocks[block_index]
        )

    def penalty(
        self,
        key: BlockKey,
        *,
        power: float,
        mixed_alpha: float,
        min_blocks: int,
    ) -> Dict[str, float]:
        cache_key = (key, float(power), float(mixed_alpha), int(min_blocks))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if sum(key) < int(min_blocks) or not self.faces:
            out = {
                "penalty": 0.0,
                "best_class": -1.0,
                "inside": 0.0,
                "outside": 0.0,
                "mixed_blocks": 0.0,
            }
            self.cache[cache_key] = out
            return out

        selected_points = self.key_points(key)
        selected_count = len(selected_points)
        chosen_blocks = selected_blocks(key)
        best = 0.0
        best_class = -1
        best_inside = 0
        best_outside = 0
        best_mixed = 0
        for class_id, face_points in self.faces:
            inside = len(selected_points & face_points)
            outside = selected_count - inside
            mixed = 0
            for block_index in chosen_blocks:
                hits = sum(1 for vertex in self.blocks[block_index] if vertex in face_points)
                if 0 < hits < len(self.blocks[block_index]):
                    mixed += 1
            value = (inside / max(selected_count, 1)) ** float(power)
            value /= 1.0 + float(outside) + float(mixed_alpha) * float(mixed)
            if value > best:
                best = float(value)
                best_class = int(class_id)
                best_inside = int(inside)
                best_outside = int(outside)
                best_mixed = int(mixed)

        out = {
            "penalty": float(best),
            "best_class": float(best_class),
            "inside": float(best_inside),
            "outside": float(best_outside),
            "mixed_blocks": float(best_mixed),
        }
        self.cache[cache_key] = out
        return out


def build_discovered_repulsion(
    *,
    facets_path: Path,
    classes: Sequence[int],
    blocks: Sequence[Sequence[int]],
) -> FaceRepulsion:
    masks_by_class = collect_block_masks_local(
        facets_path=facets_path,
        classes=[int(value) for value in classes],
        blocks=blocks,
    )
    faces: List[Tuple[int, frozenset[int]]] = []
    seen: set[Tuple[int, Tuple[int, ...]]] = set()
    for class_id, items in masks_by_class.items():
        for item in items:
            face_points = frozenset(
                int(vertex)
                for block_index in item["chosen_blocks"]
                for vertex in blocks[int(block_index)]
            )
            key = (int(class_id), tuple(sorted(face_points)))
            if key in seen:
                continue
            seen.add(key)
            faces.append((int(class_id), face_points))
    return FaceRepulsion(blocks=blocks, faces=faces)


class ClassMaskCompatibility:
    def __init__(self, *, masks: Sequence[Sequence[int]]) -> None:
        self.masks = [frozenset(int(value) for value in mask) for mask in masks]
        self.cache: Dict[BlockKey, int] = {}

    def compat_count(self, key: BlockKey) -> int:
        if key not in self.cache:
            selected = frozenset(selected_blocks(key))
            self.cache[key] = sum(1 for mask in self.masks if selected <= mask)
        return self.cache[key]


def build_class_mask_compatibility(
    *,
    facets_path: Path,
    class_id: int,
    blocks: Sequence[Sequence[int]],
) -> ClassMaskCompatibility:
    masks_by_class = collect_block_masks_local(
        facets_path=facets_path,
        classes=[int(class_id)],
        blocks=blocks,
    )
    masks = [item["chosen_blocks"] for item in masks_by_class.get(int(class_id), [])]
    return ClassMaskCompatibility(masks=masks)


class ExpansionScorer:
    """All geometry, scoring, and exact-label caches for one fixed pattern.

    A BlockKey is a tuple of 0/1 values over orbit blocks.  The scorer knows how
    to turn that block mask into the underlying 64-vertex support, compute cheap
    structural features, and call the expensive exact verifier only when needed.
    """

    def __init__(
        self,
        *,
        blocks: Sequence[Sequence[int]],
        points: torch.Tensor,
        energy,
        reference: Dict[str, object],
        rare_target_classes: set[int],
        target_classes: set[int],
        rank_tol: float,
        support_tol: float,
        discovered_repulsion: FaceRepulsion | None = None,
        class44_compatibility: ClassMaskCompatibility | None = None,
        secondary_compatibility: ClassMaskCompatibility | None = None,
    ) -> None:
        self.blocks = [tuple(int(v) for v in block) for block in blocks]
        self.points_tensor = points
        self.points = points.detach().cpu().numpy()
        self.block_vertex_arrays = [np.asarray(block, dtype=int) for block in self.blocks]
        self.energy = energy
        self.reference = reference
        self.rare_target_classes = set(rare_target_classes)
        self.target_classes = set(target_classes)
        self.rank_tol = float(rank_tol)
        self.support_tol = float(support_tol)
        self.rank_cache: Dict[BlockKey, int] = {}
        self.flat_cache: Dict[BlockKey, int] = {}
        self.boundary_cache: Dict[BlockKey, Dict[str, float]] = {}
        self.supportability_cache: Dict[tuple[BlockKey, int, int, str], Dict[str, float]] = {}
        self.direction_bank_cache: Dict[tuple[int, int, int], np.ndarray] = {}
        self.legacy_supportability_rngs: Dict[int, np.random.Generator] = {}
        self.label_cache: Dict[BlockKey, str] = {}
        self.rank24_entrance_cache: Dict[BlockKey, Dict[str, int]] = {}
        self.rank_gain_count_cache: Dict[BlockKey, int] = {}
        self.child_flat_p50_cache: Dict[BlockKey, float] = {}
        self.structural_bucket_cache: Dict[BlockKey, str] = {}
        self.hard_energy_cache: Dict[BlockKey, float] = {}
        self.intrinsic_points: np.ndarray | None = None
        self.intrinsic_rank: int | None = None
        self.discovered_repulsion = discovered_repulsion
        self.class44_compatibility = class44_compatibility
        self.secondary_compatibility = secondary_compatibility
        self.discovered_label_counts: Counter[str] = Counter()

    def vertex_indices(self, key: BlockKey) -> List[int]:
        """Expand selected block ids into the actual deterministic vertex ids."""
        return sorted(
            int(vertex)
            for block_index, value in enumerate(key)
            if int(value) == 1
            for vertex in self.blocks[block_index]
        )

    def affine_rank(self, key: BlockKey) -> int:
        """Affine rank of the selected support in the 26D Bell embedding."""
        if key not in self.rank_cache:
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                rank = 0
            else:
                selected = self.points[indices]
                centered = selected - selected.mean(axis=0, keepdims=True)
                rank = int(np.linalg.matrix_rank(centered, tol=self.rank_tol))
            self.rank_cache[key] = rank
        return self.rank_cache[key]

    def mask_tensor(self, key: BlockKey) -> torch.Tensor:
        """Convert a block mask to the 64-bit hard support mask used by DIffUCO."""
        mask = torch.zeros(64, dtype=torch.int64)
        for block_index in selected_blocks(key):
            for vertex in self.blocks[block_index]:
                mask[int(vertex)] = 1
        return mask

    def hard_energy(self, key: BlockKey) -> float:
        if key not in self.hard_energy_cache:
            self.hard_energy_cache[key] = float(self.energy.energy_of_hard_mask(self.mask_tensor(key)))
        return self.hard_energy_cache[key]

    def label(self, key: BlockKey) -> str:
        """Exact class label, cached because classification is expensive."""
        if key not in self.label_cache:
            self.label_cache[key] = classify_label(
                classify_hard_mask(self.mask_tensor(key), energy=self.energy, reference=self.reference)
            )
        return self.label_cache[key]

    def fit_boundary_metrics(self, key: BlockKey) -> Dict[str, float]:
        """Fast but brittle hyperplane test using one SVD-fitted normal.

        This is reliable near full rank, but at low rank the fitted normal is an
        arbitrary choice among many possible normals.  The supportability gate
        below was added because this metric over-pruned promising low-rank
        prefixes such as class41.
        """
        if key not in self.boundary_cache:
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                self.boundary_cache[key] = {
                    "closer_side": 32.0,
                    "positive": 32.0,
                    "negative": 32.0,
                    "supporting_shift": 1e6,
                }
                return self.boundary_cache[key]
            selected = self.points[indices]
            centroid = selected.mean(axis=0)
            centered = selected - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
            offset = -float(np.dot(normal, centroid))
            signed = self.points @ normal + offset
            positive = int(np.sum(signed > self.support_tol))
            negative = int(np.sum(signed < -self.support_tol))
            supporting_shift = min(max(float(signed.max()), 0.0) ** 2, max(float(-signed.min()), 0.0) ** 2)
            self.boundary_cache[key] = {
                "closer_side": float(min(positive, negative)),
                "positive": float(positive),
                "negative": float(negative),
                "supporting_shift": float(supporting_shift),
            }
        return self.boundary_cache[key]

    def intrinsic_coordinates(self) -> np.ndarray:
        if self.intrinsic_points is None:
            centered = self.points - self.points.mean(axis=0, keepdims=True)
            _, singulars, vh = np.linalg.svd(centered, full_matrices=False)
            rank = int(np.sum(singulars > self.rank_tol))
            self.intrinsic_rank = rank
            self.intrinsic_points = centered @ vh[:rank].T
        return self.intrinsic_points

    def supportability_boundary_metrics(
        self,
        key: BlockKey,
        *,
        direction_samples: int,
        seed: int,
        direction_bank: str = "key_dependent",
        augment_directions: bool = True,
        selection: str = "min_shift",
        closer_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Ask whether the current affine hull can be part of a supporting face.

        For a low-rank prefix there is a whole normal space, not one normal.  We
        sample directions in that normal space and keep the direction that needs
        the smallest shift to make all 64 points lie on one side.  Small
        supporting_shift / closer_side means the prefix is plausibly extendable.
        """
        sample_count = max(1, int(direction_samples))
        bank_mode = str(direction_bank)
        cache_key = (
            key,
            sample_count,
            int(seed),
            bank_mode,
            bool(augment_directions),
            str(selection),
            float(closer_weight),
        )
        if cache_key in self.supportability_cache:
            return self.supportability_cache[cache_key]

        coords = self.intrinsic_coordinates()
        indices = self.vertex_indices(key)
        if len(indices) <= 1:
            out = {
                "closer_side": 32.0,
                "positive": 32.0,
                "negative": 32.0,
                "supporting_shift": 1e6,
                "null_dim": float(self.intrinsic_rank or coords.shape[1]),
            }
            self.supportability_cache[cache_key] = out
            return out

        selected = coords[indices]
        centroid = selected.mean(axis=0)
        centered = selected - centroid
        _, singulars, vh = np.linalg.svd(centered, full_matrices=True)
        rank = int(np.sum(singulars > self.rank_tol))
        null_basis = vh[rank:].T
        null_dim = int(null_basis.shape[1])
        if null_dim <= 0:
            out = {
                "closer_side": 32.0,
                "positive": 32.0,
                "negative": 32.0,
                "supporting_shift": 1e6,
                "null_dim": 0.0,
            }
            self.supportability_cache[cache_key] = out
            return out

        projected = (coords - centroid) @ null_basis
        if null_dim == 1:
            directions = np.asarray([[1.0], [-1.0]], dtype=float)
        else:
            random_count = max(sample_count, 64)
            if bank_mode == "legacy_rng":
                rng = self.legacy_supportability_rngs.setdefault(int(seed), np.random.default_rng(int(seed)))
                directions = rng.normal(size=(random_count, null_dim))
            elif bank_mode == "global_cached":
                bank_key = (random_count, null_dim, int(seed))
                if bank_key not in self.direction_bank_cache:
                    rng = np.random.default_rng(int(seed) + 9176 * null_dim)
                    self.direction_bank_cache[bank_key] = rng.normal(size=(random_count, null_dim))
                directions = self.direction_bank_cache[bank_key]
            else:
                selected_code = sum((idx + 17) * (value + 1) for idx, value in enumerate(key) if int(value) == 1)
                rng = np.random.default_rng(int(seed) + 1009 * selected_code + 9176 * null_dim)
                directions = rng.normal(size=(random_count, null_dim))
            if bool(augment_directions):
                basis = np.eye(null_dim, dtype=float)
                nonzero_projected = projected[np.linalg.norm(projected, axis=1) > self.rank_tol]
                if len(nonzero_projected) > 0:
                    point_dirs = nonzero_projected / np.maximum(np.linalg.norm(nonzero_projected, axis=1, keepdims=True), 1e-12)
                    directions = np.vstack([directions, basis, -basis, point_dirs, -point_dirs])
                else:
                    directions = np.vstack([directions, basis, -basis])
            directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

        signed = projected @ directions.T
        pos_shift = np.maximum(np.max(signed, axis=0), 0.0)
        neg_shift = np.maximum(np.max(-signed, axis=0), 0.0)
        shifts = np.minimum(pos_shift, neg_shift) ** 2
        positives = np.sum(signed > self.support_tol, axis=0)
        negatives = np.sum(signed < -self.support_tol, axis=0)
        closer_sides = np.minimum(positives, negatives)
        if str(selection) == "min_penalty":
            penalties = np.log1p(shifts) + float(closer_weight) * closer_sides
            best_index = int(np.argmin(penalties))
        else:
            best_index = int(np.argmin(shifts))
        positive = int(positives[best_index])
        negative = int(negatives[best_index])
        out = {
            "closer_side": float(min(positive, negative)),
            "positive": float(positive),
            "negative": float(negative),
            "supporting_shift": float(shifts[best_index]),
            "null_dim": float(null_dim),
        }
        self.supportability_cache[cache_key] = out
        return out

    def flat_capacity_child_rank(self, key: BlockKey) -> int:
        if key not in self.flat_cache:
            rank = self.affine_rank(key)
            count = 0
            for action, selected in enumerate(key):
                if int(selected) == 1:
                    continue
                if self.affine_rank(add_block(key, action)) == rank:
                    count += 1
            self.flat_cache[key] = count
        return self.flat_cache[key]

    def flat_capacity_affine_hull(self, key: BlockKey) -> int:
        """Count unselected blocks already lying in the current affine hull.

        A high count means many actions do not increase rank.  Empirically this
        "flat capacity" is a bad basin signal, so expansion_score penalizes it.
        """
        if key not in self.flat_cache:
            rank = self.affine_rank(key)
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                count = 0
                for action, selected in enumerate(key):
                    if int(selected) == 1:
                        continue
                    if self.affine_rank(add_block(key, action)) == rank:
                        count += 1
                self.flat_cache[key] = count
                return self.flat_cache[key]
            selected = self.points[indices]
            anchor = selected[0]
            centered = selected - anchor
            _, singulars, vh = np.linalg.svd(centered, full_matrices=False)
            hull_rank = int(np.sum(singulars > self.rank_tol))
            if hull_rank <= 0:
                basis = np.zeros((self.points.shape[1], 0), dtype=float)
            else:
                basis = vh[:hull_rank].T
            count = 0
            for action, is_selected in enumerate(key):
                if int(is_selected) == 1:
                    continue
                block_points = self.points[self.block_vertex_arrays[action]]
                delta = block_points - anchor
                if basis.shape[1] > 0:
                    residual = delta - (delta @ basis) @ basis.T
                else:
                    residual = delta
                if float(np.max(np.linalg.norm(residual, axis=1))) <= 2.0 * self.rank_tol:
                    count += 1
            self.flat_cache[key] = count
        return self.flat_cache[key]

    def flat_capacity(self, key: BlockKey, args: argparse.Namespace | None = None) -> int:
        if str(getattr(args, "flat_capacity_method", "affine_hull")) == "child_rank":
            return self.flat_capacity_child_rank(key)
        return self.flat_capacity_affine_hull(key)

    @staticmethod
    def bin_value(value: float, edges: Sequence[float]) -> int:
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)

    def rank_gain_count(self, key: BlockKey) -> int:
        if key not in self.rank_gain_count_cache:
            rank = self.affine_rank(key)
            count = 0
            for action, selected in enumerate(key):
                if int(selected) == 1:
                    continue
                if self.affine_rank(add_block(key, action)) > rank:
                    count += 1
            self.rank_gain_count_cache[key] = count
        return self.rank_gain_count_cache[key]

    def class44_compat_count(self, key: BlockKey) -> int:
        if self.class44_compatibility is None:
            return -1
        return self.class44_compatibility.compat_count(key)

    def secondary_compat_count(self, key: BlockKey) -> int:
        if self.secondary_compatibility is None:
            return -1
        return self.secondary_compatibility.compat_count(key)

    def child_flat_p50(self, key: BlockKey) -> float:
        if key not in self.child_flat_p50_cache:
            rank = self.affine_rank(key)
            values: List[float] = []
            for action, selected in enumerate(key):
                if int(selected) == 1:
                    continue
                child = add_block(key, action)
                child_rank = self.affine_rank(child)
                if child_rank <= rank:
                    continue
                values.append(0.0 if child_rank >= 25 else float(self.flat_capacity(child)))
            self.child_flat_p50_cache[key] = 0.0 if not values else float(np.percentile(np.asarray(values, dtype=float), 50))
        return self.child_flat_p50_cache[key]

    def structural_bucket(self, key: BlockKey) -> str:
        """Coarse state signature used only for beam diversity release."""
        if key not in self.structural_bucket_cache:
            rank = self.affine_rank(key)
            flat = 0 if rank >= 25 else self.flat_capacity(key)
            gain_count = self.rank_gain_count(key)
            child_flat = self.child_flat_p50(key)
            self.structural_bucket_cache[key] = "|".join(
                [
                    f"r{rank}",
                    f"f{self.bin_value(flat, [0, 1, 2, 4, 8, 12])}",
                    f"g{self.bin_value(gain_count, [4, 8, 12, 16, 20, 24, 32])}",
                    f"cf{self.bin_value(child_flat, [0, 1, 2, 4, 6, 8, 12])}",
                ]
            )
        return self.structural_bucket_cache[key]

    def phase_weights(self, rank: int, args: argparse.Namespace) -> Tuple[float, float, float]:
        """Return weights for rank gain, boundary/supportability, and flatness."""
        if rank < int(getattr(args, "phase_b_start_rank", 22)):
            return 2.0, 0.2, 0.3
        if rank <= 23:
            return 1.5, 0.5, 1.0
        return 1.0, 1.5, 1.2

    def terminal_label(self, key: BlockKey) -> str:
        """Old static terminal classification used for evidence accounting.

        The May-3 score notes treat any rank >= 25 candidate as terminal: first
        test whether the fitted boundary is supporting, then classify the hard
        mask.  The later dynamic workflow split out rank-invalid terminals, but
        that is deliberately not part of this legacy reproduction path.
        """
        boundary = self.fit_boundary_metrics(key)
        if boundary["closer_side"] > 0:
            return "boundary_invalid"
        label = self.label(key)
        return label if label.startswith("exact:") else "invalid"

    def terminal_score(self, key: BlockKey, args: argparse.Namespace | None = None) -> float:
        """Score a terminal candidate using the May-3 static rare-target rule."""
        boundary = self.fit_boundary_metrics(key)
        if boundary["closer_side"] > 0:
            return -20.0 - 2.0 * boundary["closer_side"]
        label = self.label(key)
        if not label.startswith("exact:"):
            return -10.0
        class_id = exact_class_from_label(label)
        if class_id in self.rare_target_classes:
            return 100.0
        if class_id == 44:
            return float(getattr(args, "class44_terminal_score", -5.0))
        if class_id in self.target_classes:
            return 10.0
        if label.startswith("exact:"):
            return 1.0
        return -10.0

    def rank24_entrance_metrics(self, key: BlockKey, args: argparse.Namespace | None = None) -> Dict[str, int]:
        """Look one block ahead from rank 24.

        This is a late-phase outlet-quality signal: does this prefix have a
        one-step exact/new exit, or mostly invalid/frequent exits?
        """
        if key not in self.rank24_entrance_cache:
            rare = 0
            frequent = 0
            invalid = 0
            other_valid = 0
            for action, selected in enumerate(key):
                if int(selected) == 1:
                    continue
                candidate = add_block(key, action)
                if self.affine_rank(candidate) < 25:
                    continue
                label = self.terminal_label(candidate)
                if not label.startswith("exact:"):
                    invalid += 1
                    continue
                class_id = exact_class_from_label(label)
                if class_id in self.rare_target_classes:
                    rare += 1
                elif class_id == 44:
                    frequent += 1
                else:
                    other_valid += 1
            out = {
                "rare": rare,
                "class44": frequent,
                "invalid": invalid,
                "other_valid": other_valid,
            }
            self.rank24_entrance_cache[key] = out
        return self.rank24_entrance_cache[key]

    def expansion_score(self, key: BlockKey, action: int, args: argparse.Namespace) -> Dict[str, object]:
        """Score one possible add-block action from a nonterminal state.

        Nonterminal score is mostly rank_gain - flat_penalty plus optional
        supportability/repulsion/compatibility terms.  Terminal score delegates
        to terminal_score(), so exact facets are rewarded only after validation.
        """
        t_total = time.perf_counter()
        candidate = add_block(key, action)
        t0 = time.perf_counter()
        old_rank = self.affine_rank(key)
        new_rank = self.affine_rank(candidate)
        profile_add(args, "rank", time.perf_counter() - t0)
        rank_gain = new_rank - old_rank
        t0 = time.perf_counter()
        boundary = self.fit_boundary_metrics(candidate)
        profile_add(args, "fit_boundary", time.perf_counter() - t0)
        t0 = time.perf_counter()
        flat = self.flat_capacity(candidate, args) if new_rank < 25 else 0
        profile_add(args, "flat_capacity", time.perf_counter() - t0)
        entrance: Dict[str, int] | None = None
        supportability_gate: Dict[str, float] | None = None
        repulsion: Dict[str, float] | None = None
        compat_payload = {
            "enabled": 0.0,
            "old_compat": -1.0,
            "new_compat": -1.0,
            "zero_bonus": 0.0,
            "escape_bonus": 0.0,
        }

        if new_rank >= 25:
            t0 = time.perf_counter()
            score = self.terminal_score(candidate, args)
            profile_add(args, "terminal_score", time.perf_counter() - t0)
            phase = "C"
        else:
            w_rank, w_boundary, w_flat = self.phase_weights(new_rank, args)
            boundary_mode = str(getattr(args, "phase_boundary_mode", "fit_shift"))
            if boundary_mode == "energy_delta":
                boundary_score = self.hard_energy(key) - self.hard_energy(candidate)
            elif boundary_mode == "none":
                boundary_score = 0.0
            elif boundary_mode == "supportability":
                if supportability_enabled_for_rank(new_rank, args):
                    t0 = time.perf_counter()
                    boundary = self.supportability_boundary_metrics(
                        candidate,
                        direction_samples=effective_supportability_samples(new_rank, args),
                        seed=int(getattr(args, "supportability_seed", 20260430)),
                        direction_bank=str(getattr(args, "supportability_direction_bank", "key_dependent")),
                        augment_directions=str(getattr(args, "supportability_augment_directions", "on")) == "on",
                        selection=str(getattr(args, "supportability_selection", "min_shift")),
                        closer_weight=float(getattr(args, "supportability_gate_closer_weight", 1.0)),
                    )
                    profile_add(args, "supportability", time.perf_counter() - t0)
                    boundary_score = -math.log1p(boundary["supporting_shift"])
                else:
                    boundary_score = 0.0
            elif boundary_mode == "supportability_gate":
                if supportability_enabled_for_rank(new_rank, args):
                    t0 = time.perf_counter()
                    supportability_gate = self.supportability_boundary_metrics(
                        candidate,
                        direction_samples=effective_supportability_samples(new_rank, args),
                        seed=int(getattr(args, "supportability_seed", 20260430)),
                        direction_bank=str(getattr(args, "supportability_direction_bank", "key_dependent")),
                        augment_directions=str(getattr(args, "supportability_augment_directions", "on")) == "on",
                        selection=str(getattr(args, "supportability_selection", "min_shift")),
                        closer_weight=float(getattr(args, "supportability_gate_closer_weight", 1.0)),
                    )
                    profile_add(args, "supportability", time.perf_counter() - t0)
                    boundary = supportability_gate
                boundary_score = 0.0
            else:
                boundary_score = -math.log1p(boundary["supporting_shift"])
            score = (
                float(getattr(args, "rank_gain_weight_scale", 1.0)) * w_rank * float(rank_gain)
                + w_boundary * boundary_score
                - float(getattr(args, "flat_penalty_weight_scale", 1.0)) * w_flat * math.log1p(float(flat))
            )
            if rank_gain <= 0:
                score -= float(getattr(args, "nonpositive_rank_gain_penalty", 0.0))
            if supportability_gate is not None:
                shift = float(supportability_gate["supporting_shift"])
                closer_side = float(supportability_gate["closer_side"])
                if shift > self.support_tol**2 or closer_side > 0.0:
                    gate_penalty = (
                        math.log1p(shift)
                        + float(getattr(args, "supportability_gate_closer_weight", 0.25)) * closer_side
                    )
                    score -= float(getattr(args, "supportability_gate_weight", 2.0)) * gate_penalty
            if (
                self.discovered_repulsion is not None
                and float(getattr(args, "discovered_repulsion_weight", 0.0)) != 0.0
                and new_rank >= int(getattr(args, "discovered_repulsion_rank_start", 21))
            ):
                allow_repulsion = True
                if bool(getattr(args, "discovered_repulsion_require_supportable", False)):
                    if supportability_gate is None:
                        t0 = time.perf_counter()
                        supportability_gate = self.supportability_boundary_metrics(
                            candidate,
                            direction_samples=effective_supportability_samples(new_rank, args),
                            seed=int(getattr(args, "supportability_seed", 20260430)),
                            direction_bank=str(getattr(args, "supportability_direction_bank", "key_dependent")),
                            augment_directions=str(getattr(args, "supportability_augment_directions", "on")) == "on",
                            selection=str(getattr(args, "supportability_selection", "min_shift")),
                            closer_weight=float(getattr(args, "supportability_gate_closer_weight", 1.0)),
                        )
                        profile_add(args, "supportability", time.perf_counter() - t0)
                    allow_repulsion = (
                        float(supportability_gate["supporting_shift"]) <= self.support_tol**2
                        and float(supportability_gate["closer_side"])
                        <= float(getattr(args, "discovered_repulsion_supportable_closer_max", 0.0))
                    )
                if allow_repulsion:
                    repulsion = self.discovered_repulsion.penalty(
                        candidate,
                        power=float(getattr(args, "discovered_repulsion_power", 2.0)),
                        mixed_alpha=float(getattr(args, "discovered_repulsion_mixed_alpha", 2.0)),
                        min_blocks=int(getattr(args, "discovered_repulsion_min_blocks", 6)),
                    )
                    score -= float(getattr(args, "discovered_repulsion_weight", 0.0)) * float(repulsion["penalty"])
                else:
                    repulsion = {
                        "penalty": 1.0,
                        "best_class": -1.0,
                        "inside": 0.0,
                        "outside": 0.0,
                        "mixed_blocks": 0.0,
                        "blocked_by_supportability": 1.0,
                    }
                    score -= float(getattr(args, "discovered_repulsion_weight", 0.0)) * float(repulsion["penalty"])
            if new_rank == 24 and float(args.rank24_entrance_weight) != 0.0:
                t0 = time.perf_counter()
                entrance = self.rank24_entrance_metrics(candidate, args)
                profile_add(args, "rank24_entrance", time.perf_counter() - t0)
                score += float(args.rank24_entrance_weight) * (
                    float(args.rank24_entrance_exists_weight) * (1.0 if entrance["rare"] > 0 else 0.0)
                    + math.log1p(float(entrance["rare"]))
                    - float(args.rank24_class44_entrance_weight) * math.log1p(float(entrance["class44"]))
                    - float(args.rank24_invalid_entrance_weight) * math.log1p(float(entrance["invalid"]))
                )
            if (
                self.class44_compatibility is not None
                and int(getattr(args, "class44_compat_rank_start", 23)) <= new_rank <= int(getattr(args, "class44_compat_rank_end", 24))
            ):
                old_compat = self.class44_compat_count(key)
                new_compat = self.class44_compat_count(candidate)
                zero_bonus = float(getattr(args, "class44_compat_zero_bonus", 0.0)) if new_compat == 0 else 0.0
                escape_bonus = (
                    float(getattr(args, "class44_compat_escape_bonus", 0.0))
                    if old_compat > 0 and new_compat == 0
                    else 0.0
                )
                score += zero_bonus + escape_bonus
                compat_payload = {
                    "enabled": 1.0,
                    "old_compat": float(old_compat),
                    "new_compat": float(new_compat),
                    "zero_bonus": float(zero_bonus),
                    "escape_bonus": float(escape_bonus),
                }
            phase = "A" if new_rank < int(getattr(args, "phase_b_start_rank", 22)) else ("B1" if new_rank <= 23 else "B2")

        profile_add(args, "expansion_score_total", time.perf_counter() - t_total)
        return {
            "action": int(action),
            "key": candidate,
            "score": float(score),
            "phase": phase,
            "old_rank": int(old_rank),
            "new_rank": int(new_rank),
            "rank_gain": int(rank_gain),
            "flat_capacity": int(flat),
            "boundary": boundary,
            "supportability_gate": supportability_gate,
            "discovered_repulsion": repulsion,
            "rank24_entrance": entrance,
            "class44_compat": compat_payload,
        }


def sample_weighted(items: Sequence[Dict[str, object]], rng: random.Random, temperature: float) -> Dict[str, object]:
    max_score = max(float(item["score"]) for item in items)
    temp = max(float(temperature), 1e-6)
    weights = [math.exp((float(item["score"]) - max_score) / temp) for item in items]
    total = sum(weights)
    threshold = rng.random() * total if total > 0 else 0.0
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if running >= threshold:
            return item
    return items[-1]


def search_once(
    *,
    scorer: ExpansionScorer,
    block_count: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, object]:
    """Run one stochastic beam-search restart from the empty block mask."""
    root = empty_key(block_count)
    beam: List[Tuple[BlockKey, List[int]]] = [(root, [])]
    best: Dict[str, object] | None = None
    best_by_label: Dict[str, Dict[str, object]] = {}
    encountered: Counter[str] = Counter()
    encountered_examples: Dict[str, List[Dict[str, object]]] = {}
    diversity_releases: Counter[str] = Counter()
    compat_quota_releases: Counter[str] = Counter()

    for step in range(int(args.max_blocks)):
        # One outer iteration means one more selected block along every live path.
        t_step = time.perf_counter()
        expanded: Dict[BlockKey, Tuple[float, List[int], Dict[str, object]]] = {}
        for key, path in beam:
            t0 = time.perf_counter()
            if scorer.affine_rank(key) >= 25:
                profile_add(args, "beam_state_rank_check", time.perf_counter() - t0)
                continue
            profile_add(args, "beam_state_rank_check", time.perf_counter() - t0)
            t0 = time.perf_counter()
            candidates = [
                scorer.expansion_score(key, action, args)
                for action in range(block_count)
                if int(key[action]) == 0
            ]
            profile_add(args, "candidate_generation", time.perf_counter() - t0, len(candidates))
            t0 = time.perf_counter()
            if str(getattr(args, "shuffle_candidate_ties", "on")) == "on":
                # Legacy stochastic search should not let Python's stable sort
                # turn exact score ties into a hidden preference for low block ids.
                rng.shuffle(candidates)
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            pool = candidates[: max(1, min(int(args.candidate_pool), len(candidates)))]
            profile_add(args, "candidate_sort_and_pool", time.perf_counter() - t0)
            t0 = time.perf_counter()
            chosen_items: List[Dict[str, object]] = []
            available = list(pool)
            for _ in range(max(1, min(int(args.samples_per_state), len(available)))):
                chosen = sample_weighted(available, rng, float(args.temperature))
                chosen_items.append(chosen)
                available.remove(chosen)
                if not available:
                    break
            profile_add(args, "candidate_sampling", time.perf_counter() - t0, len(chosen_items))
            for item in chosen_items:
                t0 = time.perf_counter()
                candidate = tuple(int(v) for v in item["key"])
                new_path = path + [int(item["action"])]
                if scorer.affine_rank(candidate) >= 25:
                    # Terminal candidates leave the beam and become evidence.
                    label = scorer.terminal_label(candidate)
                    profile_add(args, "terminal_label_after_sampling", time.perf_counter() - t0)
                    encountered[label] += 1
                    if label.startswith("exact:"):
                        scorer.discovered_label_counts[label] += 1
                    bucket = encountered_examples.setdefault(label, [])
                    if len(bucket) < 5:
                        bucket.append({"chosen_blocks": selected_blocks(candidate), "path_actions": new_path})
                    terminal = {
                        "key": candidate,
                        "path": new_path,
                        "score": float(item["score"]),
                        "label": label,
                        "rank": scorer.affine_rank(candidate),
                        "boundary": item["boundary"],
                    }
                    if best is None or float(terminal["score"]) > float(best["score"]):
                        best = terminal
                    label_best = best_by_label.get(label)
                    if label_best is None or float(terminal["score"]) > float(label_best["score"]):
                        best_by_label[label] = terminal
                    continue
                profile_add(args, "nonterminal_after_sampling", time.perf_counter() - t0)
                previous = expanded.get(candidate)
                if previous is None or float(item["score"]) > previous[0]:
                    expanded[candidate] = (float(item["score"]), new_path, item)

        if not expanded:
            profile_add(args, "beam_step_total", time.perf_counter() - t_step)
            break
        t0 = time.perf_counter()
        ranked = sorted(expanded.items(), key=lambda pair: pair[1][0], reverse=True)
        profile_add(args, "global_rank_sort", time.perf_counter() - t0, len(ranked))
        beam_width = max(1, int(args.beam_width))
        compat_zero_slots = max(0, min(int(getattr(args, "class44_compat_zero_slots", 0)), beam_width))
        compat_positive_slots = max(0, min(int(getattr(args, "class44_compat_positive_slots", 0)), beam_width))
        compat_rank_start = int(getattr(args, "class44_compat_rank_start", 23))
        compat_rank_end = int(getattr(args, "class44_compat_rank_end", 24))
        secondary_zero_slots = max(
            0,
            min(
                int(getattr(args, "secondary_compat_zero_within_primary_zero_slots", 0)),
                compat_zero_slots,
            ),
        )
        global_secondary_zero_slots = max(
            0,
            min(int(getattr(args, "secondary_compat_zero_slots", 0)), beam_width),
        )
        global_secondary_positive_slots = max(
            0,
            min(int(getattr(args, "secondary_compat_positive_slots", 0)), beam_width),
        )
        joint_compat_slots = {
            "00": max(0, min(int(getattr(args, "joint_compat_00_slots", 0)), beam_width)),
            "01": max(0, min(int(getattr(args, "joint_compat_01_slots", 0)), beam_width)),
            "10": max(0, min(int(getattr(args, "joint_compat_10_slots", 0)), beam_width)),
            "11": max(0, min(int(getattr(args, "joint_compat_11_slots", 0)), beam_width)),
        }
        joint_compat_total_slots = sum(joint_compat_slots.values())
        joint_rank_start = int(getattr(args, "joint_compat_rank_start", 20))
        joint_rank_end = int(getattr(args, "joint_compat_rank_end", 24))
        secondary_rank_start = int(getattr(args, "secondary_compat_rank_start", 21))
        secondary_rank_end = int(getattr(args, "secondary_compat_rank_end", 24))
        compat_quota_active = (
            scorer.class44_compatibility is not None
            and (
                compat_zero_slots > 0
                or compat_positive_slots > 0
                or global_secondary_zero_slots > 0
                or global_secondary_positive_slots > 0
                or joint_compat_total_slots > 0
            )
        )
        if compat_quota_active:
            # Optional late-stage escape mechanism: reserve beam slots for
            # prefixes that have zero compatibility with common attractor faces.
            t0 = time.perf_counter()
            selected_pairs: List[Tuple[BlockKey, Tuple[float, List[int], Dict[str, object]]]] = []
            selected_keys: set[BlockKey] = set()
            near_limit = min(
                len(ranked),
                beam_width * max(1, int(getattr(args, "class44_compat_near_multiple", 4))),
            )

            def take_matching(
                *,
                want_zero: bool,
                limit: int,
                require_secondary_zero: bool = False,
            ) -> int:
                taken = 0
                for key, payload in ranked[:near_limit]:
                    if taken >= limit:
                        break
                    if key in selected_keys:
                        continue
                    rank = scorer.affine_rank(key)
                    if rank < compat_rank_start or rank > compat_rank_end:
                        continue
                    is_zero = scorer.class44_compat_count(key) == 0
                    if is_zero != want_zero:
                        continue
                    if require_secondary_zero:
                        if scorer.secondary_compatibility is None:
                            continue
                        if rank < secondary_rank_start or rank > secondary_rank_end:
                            continue
                        if scorer.secondary_compat_count(key) != 0:
                            continue
                    selected_pairs.append((key, payload))
                    selected_keys.add(key)
                    taken += 1
                return taken

            def take_secondary(*, want_zero: bool, limit: int) -> int:
                taken = 0
                if scorer.secondary_compatibility is None:
                    return taken
                for key, payload in ranked[:near_limit]:
                    if taken >= limit:
                        break
                    if key in selected_keys:
                        continue
                    rank = scorer.affine_rank(key)
                    if rank < secondary_rank_start or rank > secondary_rank_end:
                        continue
                    is_zero = scorer.secondary_compat_count(key) == 0
                    if is_zero != want_zero:
                        continue
                    selected_pairs.append((key, payload))
                    selected_keys.add(key)
                    taken += 1
                return taken

            def take_joint(bucket: str, limit: int) -> int:
                taken = 0
                if scorer.secondary_compatibility is None:
                    return taken
                want_primary_zero = bucket[0] == "0"
                want_secondary_zero = bucket[1] == "0"
                for key, payload in ranked[:near_limit]:
                    if taken >= limit:
                        break
                    if key in selected_keys:
                        continue
                    rank = scorer.affine_rank(key)
                    if rank < joint_rank_start or rank > joint_rank_end:
                        continue
                    primary_zero = scorer.class44_compat_count(key) == 0
                    if primary_zero != want_primary_zero:
                        continue
                    secondary_zero = scorer.secondary_compat_count(key) == 0
                    if secondary_zero != want_secondary_zero:
                        continue
                    selected_pairs.append((key, payload))
                    selected_keys.add(key)
                    taken += 1
                return taken

            joint_taken = {
                bucket: take_joint(bucket, limit)
                for bucket, limit in joint_compat_slots.items()
                if limit > 0
            }
            if joint_compat_total_slots > 0:
                global_secondary_zero_taken = 0
                global_secondary_positive_taken = 0
                secondary_zero_taken = 0
                zero_taken = 0
                positive_taken = 0
            else:
                global_secondary_zero_taken = take_secondary(
                    want_zero=True,
                    limit=global_secondary_zero_slots,
                )
                global_secondary_positive_taken = take_secondary(
                    want_zero=False,
                    limit=global_secondary_positive_slots,
                )
                secondary_zero_taken = take_matching(
                    want_zero=True,
                    limit=secondary_zero_slots,
                    require_secondary_zero=True,
                )
                zero_taken = secondary_zero_taken + take_matching(
                    want_zero=True,
                    limit=compat_zero_slots - secondary_zero_taken,
                )
                positive_taken = take_matching(want_zero=False, limit=compat_positive_slots)
            for key, payload in ranked:
                if key in selected_keys:
                    continue
                selected_pairs.append((key, payload))
                selected_keys.add(key)
                if len(selected_pairs) >= beam_width:
                    break
            compat_quota_releases[f"step:{step + 1}:zero"] += zero_taken
            compat_quota_releases[f"step:{step + 1}:positive"] += positive_taken
            compat_quota_releases[f"step:{step + 1}:secondary_zero"] += secondary_zero_taken
            compat_quota_releases[f"step:{step + 1}:global_secondary_zero"] += global_secondary_zero_taken
            compat_quota_releases[f"step:{step + 1}:global_secondary_positive"] += global_secondary_positive_taken
            for bucket, taken in joint_taken.items():
                compat_quota_releases[f"step:{step + 1}:joint_{bucket}"] += taken
            beam = [(key, path) for key, (_, path, _) in selected_pairs[:beam_width]]
            profile_add(args, "class44_compat_quota_select", time.perf_counter() - t0)
            profile_add(args, "beam_step_total", time.perf_counter() - t_step)
            continue
        early_slots = max(0, min(int(getattr(args, "beam_diversity_slots", 0)), beam_width))
        late_slots = max(0, min(int(getattr(args, "beam_diversity_late_slots", 0)), beam_width))
        rank_start = int(getattr(args, "beam_diversity_rank_start", 17))
        rank_end = int(getattr(args, "beam_diversity_rank_end", 21))
        late_rank_start = int(getattr(args, "beam_diversity_late_rank_start", 22))
        late_rank_end = int(getattr(args, "beam_diversity_late_rank_end", 23))
        top_ranks = [scorer.affine_rank(key) for key, _payload in ranked[:beam_width]]
        early_active = early_slots > 0 and any(rank_start <= rank <= rank_end for rank in top_ranks)
        late_active = late_slots > 0 and any(late_rank_start <= rank <= late_rank_end for rank in top_ranks)
        if early_active:
            diversity_slots = early_slots
            active_rank_start = rank_start
            active_rank_end = rank_end
        elif late_active:
            diversity_slots = late_slots
            active_rank_start = late_rank_start
            active_rank_end = late_rank_end
        else:
            diversity_slots = 0
            active_rank_start = rank_start
            active_rank_end = rank_end
        diversity_active = diversity_slots > 0
        t0 = time.perf_counter()
        if not diversity_active:
            beam = [(key, path) for key, (_, path, _) in ranked[:beam_width]]
            profile_add(args, "beam_select", time.perf_counter() - t0)
            profile_add(args, "beam_step_total", time.perf_counter() - t_step)
            continue

        score_slots = beam_width - diversity_slots
        selected_pairs = ranked[:score_slots]
        selected_keys = {key for key, _ in selected_pairs}
        selected_buckets = {scorer.structural_bucket(key) for key, _ in selected_pairs}
        near_limit = min(len(ranked), beam_width * max(1, int(getattr(args, "beam_diversity_near_multiple", 4))))
        released: List[Tuple[BlockKey, Tuple[float, List[int], Dict[str, object]]]] = []
        if str(getattr(args, "beam_diversity_mode", "bucket_release")) == "bucket_stochastic":
            # Reserve some beam slots for underrepresented structural buckets.
            buckets: Dict[str, List[Tuple[BlockKey, Tuple[float, List[int], Dict[str, object]]]]] = {}
            for key, payload in ranked[score_slots:near_limit]:
                if key in selected_keys:
                    continue
                rank = scorer.affine_rank(key)
                if rank < active_rank_start or rank > active_rank_end:
                    continue
                bucket = scorer.structural_bucket(key)
                buckets.setdefault(bucket, []).append((key, payload))
            bucket_order = sorted(
                buckets,
                key=lambda bucket: (0 if bucket not in selected_buckets else 1, -len(buckets[bucket])),
            )
            bucket_cursor = 0
            temp = max(float(getattr(args, "beam_diversity_temperature", 0.25)), 1e-6)
            while len(released) < diversity_slots and bucket_order:
                bucket = bucket_order[bucket_cursor % len(bucket_order)]
                candidates = [(key, payload) for key, payload in buckets[bucket] if key not in selected_keys]
                if candidates:
                    max_score = max(float(payload[0]) for _key, payload in candidates)
                    weights = [math.exp((float(payload[0]) - max_score) / temp) for _key, payload in candidates]
                    total = sum(weights)
                    threshold = rng.random() * total if total > 0 else 0.0
                    running = 0.0
                    chosen_index = len(candidates) - 1
                    for idx, weight in enumerate(weights):
                        running += weight
                        if running >= threshold:
                            chosen_index = idx
                            break
                    key, payload = candidates[chosen_index]
                    released.append((key, payload))
                    selected_keys.add(key)
                    selected_buckets.add(bucket)
                bucket_cursor += 1
                if bucket_cursor >= len(bucket_order) * 2 and all(
                    all(key in selected_keys for key, _payload in buckets[bucket])
                    for bucket in bucket_order
                ):
                    break
        else:
            for key, payload in ranked[score_slots:near_limit]:
                if key in selected_keys:
                    continue
                rank = scorer.affine_rank(key)
                if rank < active_rank_start or rank > active_rank_end:
                    continue
                bucket = scorer.structural_bucket(key)
                if bucket in selected_buckets:
                    continue
                released.append((key, payload))
                selected_keys.add(key)
                selected_buckets.add(bucket)
                if len(released) >= diversity_slots:
                    break

            if len(released) < diversity_slots:
                for key, payload in ranked[score_slots:near_limit]:
                    if key in selected_keys:
                        continue
                    rank = scorer.affine_rank(key)
                    if rank < active_rank_start or rank > active_rank_end:
                        continue
                    released.append((key, payload))
                    selected_keys.add(key)
                    if len(released) >= diversity_slots:
                        break

        selected_pairs = selected_pairs + released
        if len(selected_pairs) < beam_width:
            for key, payload in ranked:
                if key in selected_keys:
                    continue
                selected_pairs.append((key, payload))
                selected_keys.add(key)
                if len(selected_pairs) >= beam_width:
                    break
        diversity_releases[f"step:{step + 1}"] += len(released)
        beam = [(key, path) for key, (_, path, _) in selected_pairs[:beam_width]]
        profile_add(args, "beam_select", time.perf_counter() - t0)
        profile_add(args, "beam_step_total", time.perf_counter() - t_step)

    if best is None:
        key, path = max(
            beam,
            key=lambda item: scorer.expansion_score(
                item[0],
                next((a for a in range(block_count) if item[0][a] == 0), 0),
                args,
            )["score"],
        )
        label = scorer.terminal_label(key) if scorer.affine_rank(key) >= 25 else "nonterminal"
        best = {
            "key": key,
            "path": path,
            "score": -float("inf"),
            "label": label,
            "rank": scorer.affine_rank(key),
            "boundary": scorer.fit_boundary_metrics(key),
        }

    best_key = tuple(int(v) for v in best["key"])
    compact_terminal_bests = {
        label: {
            "chosen_blocks": selected_blocks(tuple(int(v) for v in terminal["key"])),
            "path_actions": list(terminal["path"]),
            "rank": int(terminal["rank"]),
            "score": float(terminal["score"]),
            "boundary": terminal["boundary"],
        }
        for label, terminal in sorted(best_by_label.items())
    }
    return {
        "label": best["label"],
        "chosen_blocks": selected_blocks(best_key),
        "path_actions": list(best["path"]),
        "rank": int(best["rank"]),
        "score": float(best["score"]),
        "boundary": best["boundary"],
        "encountered_label_counts": dict(sorted(encountered.items())),
        "encountered_examples": encountered_examples,
        "terminal_bests": compact_terminal_bests,
        "diversity_releases": dict(sorted(diversity_releases.items())),
        "class44_compat_quota_releases": dict(sorted(compat_quota_releases.items())),
        "profile_timings": getattr(args, "_profile_timings", {}) if profile_enabled(args) else {},
    }


def summarize(runs: Sequence[Dict[str, object]], rare_target_classes: set[int]) -> Dict[str, object]:
    """Collapse restarts into the JSON summary used by handoff reports."""
    label_counts = Counter(str(run["label"]) for run in runs)
    exact_counts: Counter[int] = Counter()
    for run in runs:
        terminal_bests = dict(run.get("terminal_bests", {}))
        labels = terminal_bests.keys() if terminal_bests else [str(run["label"])]
        exact_counts.update(
            int(str(label).split(":", 1)[1])
            for label in labels
            if str(label).startswith("exact:")
        )
    encountered: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, object]]] = {}
    diversity_releases: Counter[str] = Counter()
    for run in runs:
        encountered.update({str(k): int(v) for k, v in dict(run["encountered_label_counts"]).items()})
        diversity_releases.update({str(k): int(v) for k, v in dict(run.get("diversity_releases", {})).items()})
        for label, bucket in dict(run["encountered_examples"]).items():
            out = examples.setdefault(str(label), [])
            for item in bucket:
                if len(out) < 5:
                    out.append(item)
    rare_opened = sorted(set(exact_counts) & rare_target_classes)
    rare_encountered = sorted(
        int(label.split(":", 1)[1])
        for label in encountered
        if label.startswith("exact:") and int(label.split(":", 1)[1]) in rare_target_classes
    )
    return {
        "label_counts": dict(sorted(label_counts.items())),
        "exact_class_counts": dict(sorted(exact_counts.items())),
        "opened_rare_target_classes": rare_opened,
        "rare_target_coverage_count": len(rare_opened),
        "encountered_label_counts": dict(sorted(encountered.items())),
        "encountered_rare_target_classes": rare_encountered,
        "encountered_rare_target_coverage_count": len(rare_encountered),
        "encountered_examples": examples,
        "diversity_releases": dict(sorted(diversity_releases.items())),
    }


def summarize_profile_timings(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for run in runs:
        for name, bucket in dict(run.get("profile_timings", {})).items():
            out = totals.setdefault(str(name), {"seconds": 0.0, "count": 0})
            out["seconds"] += float(dict(bucket).get("seconds", 0.0))
            out["count"] += int(dict(bucket).get("count", 0))
    grand_total = sum(float(bucket["seconds"]) for bucket in totals.values())
    return {
        name: {
            "seconds": round(float(bucket["seconds"]), 6),
            "count": int(bucket["count"]),
            "avg_ms": round(1000.0 * float(bucket["seconds"]) / max(1, int(bucket["count"])), 6),
            "share_of_profiled_seconds": round(float(bucket["seconds"]) / grand_total, 6)
            if grand_total > 0.0
            else 0.0,
        }
        for name, bucket in sorted(totals.items(), key=lambda item: item[1]["seconds"], reverse=True)
    }


def main() -> None:
    args = parse_args()
    rare_target_classes = {int(v) for v in args.rare_target_classes}
    target_classes = {int(v) for v in args.target_classes}
    points = generate_points_322()
    energy = build_energy(points=points, version=args.energy_version)
    reference = build_reference_database()
    pattern = build_class_orbit_patterns(args.row_class, rep_index=args.rep_index)[args.pattern_index]
    blocks = [list(orbit) for orbit in pattern.orbits]
    discovered_repulsion = None
    if float(args.discovered_repulsion_weight) != 0.0:
        discovered_repulsion = build_discovered_repulsion(
            facets_path=args.facets,
            classes=args.discovered_repulsion_classes,
            blocks=blocks,
        )
    class44_compatibility = None
    if (
        float(getattr(args, "class44_compat_zero_bonus", 0.0)) != 0.0
        or float(getattr(args, "class44_compat_escape_bonus", 0.0)) != 0.0
        or int(getattr(args, "class44_compat_zero_slots", 0)) > 0
        or int(getattr(args, "class44_compat_positive_slots", 0)) > 0
        or int(getattr(args, "joint_compat_00_slots", 0)) > 0
        or int(getattr(args, "joint_compat_01_slots", 0)) > 0
        or int(getattr(args, "joint_compat_10_slots", 0)) > 0
        or int(getattr(args, "joint_compat_11_slots", 0)) > 0
    ):
        class44_compatibility = build_class_mask_compatibility(
            facets_path=args.facets,
            class_id=int(getattr(args, "class44_compat_target_class", 44)),
            blocks=blocks,
        )
    secondary_compatibility = None
    if (
        int(getattr(args, "secondary_compat_zero_within_primary_zero_slots", 0)) > 0
        or int(getattr(args, "secondary_compat_zero_slots", 0)) > 0
        or int(getattr(args, "secondary_compat_positive_slots", 0)) > 0
        or int(getattr(args, "joint_compat_00_slots", 0)) > 0
        or int(getattr(args, "joint_compat_01_slots", 0)) > 0
        or int(getattr(args, "joint_compat_10_slots", 0)) > 0
        or int(getattr(args, "joint_compat_11_slots", 0)) > 0
    ):
        secondary_compatibility = build_class_mask_compatibility(
            facets_path=args.facets,
            class_id=int(getattr(args, "secondary_compat_target_class", 43)),
            blocks=blocks,
        )
    scorer = ExpansionScorer(
        blocks=blocks,
        points=points,
        energy=energy,
        reference=reference,
        rare_target_classes=rare_target_classes,
        target_classes=target_classes,
        rank_tol=args.rank_tol,
        support_tol=args.support_tol,
        discovered_repulsion=discovered_repulsion,
        class44_compatibility=class44_compatibility,
        secondary_compatibility=secondary_compatibility,
    )
    for class_id in args.initial_discovered_classes:
        scorer.discovered_label_counts[f"exact:{int(class_id)}"] = max(
            scorer.discovered_label_counts[f"exact:{int(class_id)}"], 1
        )
    for class_id in args.initial_frequent_classes:
        scorer.discovered_label_counts[f"exact:{int(class_id)}"] = max(
            scorer.discovered_label_counts[f"exact:{int(class_id)}"],
            int(args.dynamic_frequent_class_threshold),
        )

    runs: List[Dict[str, object]] = []
    discovered_exact_labels: set[str] = set()
    restarts_without_new_discovery = 0
    stopped_early = False
    for seed in args.seeds:
        for restart_index in range(int(args.restarts_per_seed)):
            # Seeds only affect stochastic sampling; exact verification remains deterministic.
            if profile_enabled(args):
                setattr(args, "_profile_timings", {})
            rng = random.Random(int(seed) + 1009 * restart_index)
            run = search_once(scorer=scorer, block_count=len(blocks), args=args, rng=rng)
            run["seed"] = int(seed)
            run["restart_index"] = int(restart_index)
            runs.append(run)
            exact_labels = {
                str(label)
                for label in dict(run.get("encountered_label_counts", {}))
                if str(label).startswith("exact:")
            }
            new_labels = exact_labels - discovered_exact_labels
            if new_labels:
                discovered_exact_labels.update(new_labels)
                restarts_without_new_discovery = 0
            else:
                restarts_without_new_discovery += 1
            if (
                int(args.discovery_patience_restarts) > 0
                and len(runs) >= int(args.discovery_min_restarts)
                and restarts_without_new_discovery >= int(args.discovery_patience_restarts)
            ):
                stopped_early = True
                break
        if stopped_early:
            break

    payload = {
        "meta": {
            "algorithm": "phase_aware_add_only_expansion_v1",
            "row_class": args.row_class,
            "rep_index": args.rep_index,
            "pattern_index": args.pattern_index,
            "block_count": len(blocks),
            "orbit_sizes": list(pattern.orbit_sizes),
            "rare_target_classes": sorted(rare_target_classes),
            "target_classes": sorted(target_classes),
            "seeds": list(args.seeds),
            "restarts_per_seed": args.restarts_per_seed,
            "beam_width": args.beam_width,
            "candidate_pool": args.candidate_pool,
            "samples_per_state": args.samples_per_state,
            "temperature": args.temperature,
            "max_blocks": args.max_blocks,
            "phase_b_start_rank": args.phase_b_start_rank,
            "phase_boundary_mode": args.phase_boundary_mode,
            "supportability_direction_samples": args.supportability_direction_samples,
            "supportability_direction_bank": args.supportability_direction_bank,
            "supportability_augment_directions": args.supportability_augment_directions,
            "supportability_selection": args.supportability_selection,
            "supportability_schedule": args.supportability_schedule,
            "supportability_gate_weight": args.supportability_gate_weight,
            "supportability_gate_closer_weight": args.supportability_gate_closer_weight,
            "beam_diversity_slots": args.beam_diversity_slots,
            "beam_diversity_mode": args.beam_diversity_mode,
            "beam_diversity_near_multiple": args.beam_diversity_near_multiple,
            "beam_diversity_rank_start": args.beam_diversity_rank_start,
            "beam_diversity_rank_end": args.beam_diversity_rank_end,
            "beam_diversity_temperature": args.beam_diversity_temperature,
            "beam_diversity_late_slots": args.beam_diversity_late_slots,
            "beam_diversity_late_rank_start": args.beam_diversity_late_rank_start,
            "beam_diversity_late_rank_end": args.beam_diversity_late_rank_end,
            "rank24_entrance_weight": args.rank24_entrance_weight,
            "rank24_entrance_exists_weight": args.rank24_entrance_exists_weight,
            "rank24_class44_entrance_weight": args.rank24_class44_entrance_weight,
            "rank24_invalid_entrance_weight": args.rank24_invalid_entrance_weight,
            "rank_gain_weight_scale": args.rank_gain_weight_scale,
            "nonpositive_rank_gain_penalty": args.nonpositive_rank_gain_penalty,
            "flat_penalty_weight_scale": args.flat_penalty_weight_scale,
            "flat_capacity_method": args.flat_capacity_method,
            "terminal_scoring_mode": args.terminal_scoring_mode,
            "invalid_rank_terminal_score": args.invalid_rank_terminal_score,
            "dynamic_new_class_score": args.dynamic_new_class_score,
            "dynamic_known_class_score": args.dynamic_known_class_score,
            "dynamic_frequent_class_score": args.dynamic_frequent_class_score,
            "dynamic_frequent_class_threshold": args.dynamic_frequent_class_threshold,
            "initial_discovered_classes": list(args.initial_discovered_classes),
            "initial_frequent_classes": list(args.initial_frequent_classes),
            "discovered_repulsion_weight": args.discovered_repulsion_weight,
            "discovered_repulsion_rank_start": args.discovered_repulsion_rank_start,
            "discovered_repulsion_min_blocks": args.discovered_repulsion_min_blocks,
            "discovered_repulsion_power": args.discovered_repulsion_power,
            "discovered_repulsion_mixed_alpha": args.discovered_repulsion_mixed_alpha,
            "discovered_repulsion_require_supportable": args.discovered_repulsion_require_supportable,
            "discovered_repulsion_supportable_closer_max": args.discovered_repulsion_supportable_closer_max,
            "discovered_repulsion_classes": list(args.discovered_repulsion_classes),
            "discovered_repulsion_face_count": len(discovered_repulsion.faces) if discovered_repulsion else 0,
            "profile_timing": args.profile_timing,
            "discovery_patience_restarts": args.discovery_patience_restarts,
            "discovery_min_restarts": args.discovery_min_restarts,
            "stopped_early": stopped_early,
            "executed_restarts": len(runs),
        },
        "summary": summarize(runs, rare_target_classes),
        "profile_timings": summarize_profile_timings(runs) if profile_enabled(args) else {},
        "cache_sizes": {
            "rank": len(scorer.rank_cache),
            "flat": len(scorer.flat_cache),
            "boundary": len(scorer.boundary_cache),
            "label": len(scorer.label_cache),
            "rank24_entrance": len(scorer.rank24_entrance_cache),
            "rank_gain_count": len(scorer.rank_gain_count_cache),
            "child_flat_p50": len(scorer.child_flat_p50_cache),
            "structural_bucket": len(scorer.structural_bucket_cache),
            "supportability": len(scorer.supportability_cache),
            "direction_bank": len(scorer.direction_bank_cache),
            "discovered_repulsion": len(discovered_repulsion.cache) if discovered_repulsion else 0,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
