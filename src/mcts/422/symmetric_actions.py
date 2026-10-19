from __future__ import annotations

"""Reduced coefficient-space proposals for the full-space MCTS."""

import hashlib
import random
import time
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from .atlas import HandoffPatternAtlas, IDENTITY_PATTERN_ID
from .model import MacroAction, SearchState
from .problem import (
    Relabeling,
    TARGET_AFFINE_RANK,
    compose_permutations,
    named_relabelings,
    orbit_partition,
    permutation_closure,
    vertex_permutation,
    word_indices,
)
from .quotient_actions import QuotientActionGenerator


class RelaxationPatternAtlas(HandoffPatternAtlas):
    """Build proper subgroups from structural generators, without facet data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._groups: dict[str, tuple[tuple[int, ...], ...]] = {
            IDENTITY_PATTERN_ID: (tuple(range(256)),)
        }
        self._relaxations: dict[str, tuple[str, ...]] = {}
        relabelings = named_relabelings()
        for first in range(4):
            for second in range(first + 1, 4):
                permutation = list(range(4))
                permutation[first], permutation[second] = second, first
                relabelings[f"s_{'ABCD'[first]}{'ABCD'[second]}"] = Relabeling(
                    party_permutation=tuple(permutation)
                )
        self._named_permutations = {
            name: vertex_permutation(value) for name, value in relabelings.items()
        }

    def subgroup(self, pattern_id: str) -> tuple[tuple[int, ...], ...] | None:
        if pattern_id in self._groups:
            return self._groups[pattern_id]
        pattern = self.pattern(pattern_id)
        if not pattern.generator_words:
            return None
        generators = []
        for word in pattern.generator_words:
            permutation = tuple(range(256))
            # Handoff words act on a state from right to left.
            for token in reversed(word.split("*")):
                token = token.strip()
                if token in ("e", "1"):
                    continue
                if token not in self._named_permutations:
                    raise ValueError(f"unknown structural generator: {token}")
                permutation = compose_permutations(
                    permutation, self._named_permutations[token]
                )
            generators.append(permutation)
        group = permutation_closure(generators, max_order=384)
        if group is None:
            raise ValueError(f"structural subgroup exceeds closure budget: {pattern_id}")
        blocks = orbit_partition(group)
        if frozenset(map(frozenset, blocks)) != frozenset(
            map(frozenset, self.blocks(pattern_id))
        ):
            raise ValueError(f"generator/orbit mismatch: {pattern_id}")
        self._groups[pattern_id] = group
        self.stats["verified_subgroup_closures"] += 1
        return group

    def proper_subgroups(self, pattern_id: str) -> tuple[str, ...]:
        cached = self._relaxations.get(pattern_id)
        if cached is not None:
            return cached
        parent = self.subgroup(pattern_id)
        if parent is None:
            result = () if pattern_id == IDENTITY_PATTERN_ID else (IDENTITY_PATTERN_ID,)
            self._relaxations[pattern_id] = result
            return result
        parent_set = set(parent)
        groups: dict[tuple, tuple[tuple[int, ...], ...]] = {}
        for element in parent:
            group = permutation_closure((element,), max_order=len(parent))
            if group is not None and 1 < len(group) < len(parent):
                if not set(group) < parent_set:
                    raise ValueError("relaxation is not a proper subgroup")
                groups[group] = group
        ids = []
        for group in sorted(groups.values(), key=lambda value: (-len(value), value)):
            fingerprint = hashlib.sha256(repr(group).encode("ascii")).hexdigest()[:16]
            child_id = f"RELAX422-{fingerprint}"
            self.register_online_pattern(
                child_id,
                orbit_partition(group),
                structure=f"structural_subgroup_order_{len(group)}",
                family_id=child_id,
            )
            self._groups[child_id] = group
            ids.append(child_id)
        if pattern_id != IDENTITY_PATTERN_ID:
            ids.append(IDENTITY_PATTERN_ID)
        result = tuple(ids)
        self._relaxations[pattern_id] = result
        self.stats["proper_subgroup_relaxations"] += len(groups)
        return result


@dataclass(frozen=True)
class CoefficientChart:
    basis: np.ndarray
    full_slacks: np.ndarray
    representatives: tuple[int, ...]
    normalization: np.ndarray

    @classmethod
    def build(
        cls, augmented: np.ndarray, blocks: Sequence[Sequence[int]]
    ) -> "CoefficientChart":
        differences = [
            augmented[index] - augmented[block[0]]
            for block in blocks
            for index in block[1:]
        ]
        basis = (
            null_space(np.asarray(differences, dtype=float), rcond=1e-10)
            if differences
            else np.eye(augmented.shape[1])
        )
        slacks = augmented @ basis
        return cls(
            basis,
            slacks,
            tuple(int(block[0]) for block in blocks),
            slacks.sum(axis=0),
        )

    def containing(self, support_word: int) -> "CoefficientChart | None":
        if not support_word:
            return self
        selected = self.full_slacks[word_indices(support_word)]
        kernel = null_space(selected, rcond=1e-9)
        if not kernel.shape[1]:
            return None
        normalization = self.normalization @ kernel
        if np.linalg.norm(normalization) < 1e-8:
            return None
        return CoefficientChart(
            self.basis @ kernel,
            self.full_slacks @ kernel,
            self.representatives,
            normalization,
        )


class SymmetricActionGenerator(QuotientActionGenerator):
    def __init__(self, *args, conditional_cache_size: int = 8_192, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coefficient_charts: dict[str, CoefficientChart] = {}
        self._conditional_charts: OrderedDict[tuple[str, int], CoefficientChart | None] = (
            OrderedDict()
        )
        self.conditional_cache_size = int(conditional_cache_size)
        self.symmetric_stats: Counter[str] = Counter()
        self.proposal_counts: Counter[str] = Counter()
        self.proposal_seconds: Counter[str] = Counter()
        self.proposal_discoveries: Counter[str] = Counter()

    def coefficient_chart(self, pattern_id: str) -> CoefficientChart:
        chart = self.coefficient_charts.get(pattern_id)
        if chart is None:
            chart = CoefficientChart.build(self._augmented, self.atlas.blocks(pattern_id))
            self.coefficient_charts[pattern_id] = chart
            self.symmetric_stats["coefficient_charts_built"] += 1
        return chart

    def conditional_chart(
        self, pattern_id: str, support_word: int
    ) -> CoefficientChart | None:
        key = pattern_id, int(support_word)
        if key in self._conditional_charts:
            self._conditional_charts.move_to_end(key)
            self.symmetric_stats["conditional_chart_hits"] += 1
            return self._conditional_charts[key]
        chart = self.coefficient_chart(pattern_id).containing(support_word)
        self._conditional_charts[key] = chart
        while len(self._conditional_charts) > self.conditional_cache_size:
            self._conditional_charts.popitem(last=False)
            self.symmetric_stats["conditional_chart_evictions"] += 1
        self.symmetric_stats["conditional_chart_misses"] += 1
        if chart is None:
            self.symmetric_stats["incompatible_supports"] += 1
        return chart

    def supporting_face_actions(
        self,
        state: SearchState,
        pattern_id: str,
        *,
        nonce: int,
        count: int = 1,
    ) -> list[MacroAction]:
        if count <= 0 or self.rank_oracle.rank(state.support_word) >= TARGET_AFFINE_RANK:
            return []
        key = "symmetric_lp", state.support_word, pattern_id, int(nonce), int(count), state.source_facet_word
        cached = self._quotient_cache.get(key)
        if cached is not None:
            self._quotient_cache.move_to_end(key)
            self.symmetric_stats["proposal_cache_hits"] += 1
            return list(cached)
        started = time.perf_counter()
        self.proposal_counts[pattern_id] += 1
        chart = self.conditional_chart(pattern_id, state.support_word)
        if chart is None:
            self.proposal_seconds[pattern_id] += time.perf_counter() - started
            self._remember_quotient(key, ())
            return []
        token = zlib.crc32(f"{pattern_id}:{nonce}".encode("ascii"))
        support_token = int.from_bytes(
            hashlib.blake2b(state.support_word.to_bytes(32, "little"), digest_size=8).digest(),
            "little",
        )
        rng = random.Random(self.seed ^ token ^ support_token)
        blocks = self.atlas.blocks(pattern_id)
        actions = []
        seen: set[int] = set()
        attempts = max(count * 4, count + 2) if not state.support_word else max(2, count)
        for _attempt in range(attempts):
            self.symmetric_stats["lp_attempts"] += 1
            self.symmetric_stats["lp_variable_sum"] += chart.basis.shape[1]
            weights = np.empty(256, dtype=float)
            for block in blocks:
                weights[list(block)] = rng.gauss(0.0, 1.0)
            objective = self._augmented.T @ weights
            objective += np.asarray([0.05 * rng.gauss(0.0, 1.0) for _ in range(81)])
            result = linprog(
                objective @ chart.basis,
                A_ub=-chart.full_slacks[list(chart.representatives)],
                b_ub=np.zeros(len(chart.representatives)),
                A_eq=chart.normalization[np.newaxis, :],
                b_eq=np.ones(1),
                bounds=[(None, None)] * chart.basis.shape[1],
                method="highs",
            )
            if not result.success:
                self.symmetric_stats["lp_failures"] += 1
                continue
            slacks = chart.full_slacks @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            if float(np.min(slacks)) < -tolerance:
                self.symmetric_stats["negative_slack_failures"] += 1
                continue
            support = sum(1 << int(index) for index in np.flatnonzero(np.abs(slacks) <= tolerance))
            if state.support_word & ~support:
                self.symmetric_stats["lost_selected_failures"] += 1
                continue
            if support == state.source_facet_word and state.source_facet_word:
                self.symmetric_stats["source_returns"] += 1
                continue
            if support in seen:
                self.symmetric_stats["duplicate_lp_supports"] += 1
                continue
            seen.add(support)
            if self.rank_oracle.rank(support) != TARGET_AFFINE_RANK:
                self.symmetric_stats["lp_nonfacets"] += 1
                continue
            add = support & ~state.support_word
            if not add:
                continue
            actions.append(
                MacroAction(
                    "add", pattern_id, (), add,
                    "conditional_symmetric_lp" if state.support_word else "fullspace_supporting_face_lp",
                    f"{self.atlas.pattern(pattern_id).structure}:coefficient_lp",
                    add_word=add,
                )
            )
            self.symmetric_stats["lp_candidates"] += 1
            if len(actions) >= count:
                break
        self.proposal_seconds[pattern_id] += time.perf_counter() - started
        self._remember_quotient(key, tuple(actions))
        return actions

    def conditional_completion_actions(
        self, state: SearchState, pattern_id: str, *, nonce: int, relaxation_trials: int
    ) -> list[MacroAction]:
        actions = self.supporting_face_actions(state, pattern_id, nonce=nonce)
        if actions:
            return actions
        atlas: RelaxationPatternAtlas = self.atlas
        children = atlas.proper_subgroups(pattern_id)
        proper = [value for value in children if value != IDENTITY_PATTERN_ID]
        proper.sort(
            key=lambda value: (
                -self.proposal_discoveries[value] / max(0.001, self.proposal_seconds[value]),
                self.proposal_counts[value],
                -len(atlas.subgroup(value) or ()),
                value,
            )
        )
        # Reserve the last relaxation attempt for the unrestricted coefficient space.
        candidates = proper[: max(0, relaxation_trials - 1)]
        if IDENTITY_PATTERN_ID in children and relaxation_trials:
            candidates.append(IDENTITY_PATTERN_ID)
        for child_id in candidates:
            self.symmetric_stats["relaxation_attempts"] += 1
            actions = self.supporting_face_actions(state, child_id, nonce=nonce)
            if actions:
                self.symmetric_stats["relaxation_successes"] += 1
                return actions
        return []

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result["symmetric_lp"] = {
            "stats": dict(self.symmetric_stats),
            "chart_dimensions": {
                key: chart.basis.shape[1] for key, chart in self.coefficient_charts.items()
            },
            "conditional_cache_entries": len(self._conditional_charts),
            "proposal_counts": dict(self.proposal_counts),
            "proposal_seconds": dict(self.proposal_seconds),
            "proposal_discoveries": dict(self.proposal_discoveries),
        }
        return result
