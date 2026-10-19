from __future__ import annotations

"""Full-space actions proposed through subgroup quotient charts."""

import hashlib
import math
import random
import zlib
from collections import Counter, OrderedDict
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

from .atlas import HandoffPatternAtlas
from .filler_corrector import FillerCorrector, TailCompletion
from .geometry import FacetOracle, FacetResult, RankOracle
from .model import MacroAction, SearchState
from .problem import TARGET_AFFINE_RANK, word_indices
from mcts.symmetry_game.ridge import adjacent_facet_words


class QuotientActionGenerator(FillerCorrector):
    """Use orbit partitions as action charts without restricting the state.

    The MCTS state remains a full 256-vertex support. Two partitions define a
    common-refinement chart whose cells can be added, removed, or partially
    sampled. This keeps the group-theoretic structure while allowing a path to
    break one subgroup symmetry and continue in another chart.
    """

    def __init__(
        self,
        atlas: HandoffPatternAtlas,
        rank_oracle: RankOracle,
        facet_oracle: FacetOracle,
        *,
        quotient_candidate_limit: int = 16,
        quotient_secondary_limit: int = 3,
        partial_rank_checks: int = 12,
        ridge_rank_checks: int = 36,
        ridge_basis_trials: int = 4,
        ridge_basis_rank_checks: int = 320,
        refinement_cache_size: int = 2_048,
        quotient_cache_size: int = 20_000,
        ridge_exit_scan: int = 1,
        unseen_exit_first: bool = False,
        canonical_frontier_dedup: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(atlas, rank_oracle, facet_oracle, **kwargs)
        self.quotient_candidate_limit = max(1, int(quotient_candidate_limit))
        self.quotient_secondary_limit = max(1, int(quotient_secondary_limit))
        self.partial_rank_checks = max(0, int(partial_rank_checks))
        self.ridge_rank_checks = max(1, int(ridge_rank_checks))
        self.ridge_basis_trials = max(1, int(ridge_basis_trials))
        self.ridge_basis_rank_checks = max(1, int(ridge_basis_rank_checks))
        self.refinement_cache_size = max(8, int(refinement_cache_size))
        self.quotient_cache_size = max(32, int(quotient_cache_size))
        self.ridge_exit_scan = max(1, int(ridge_exit_scan))
        self.unseen_exit_first = bool(unseen_exit_first)
        self.canonical_frontier_dedup = bool(canonical_frontier_dedup)
        self._refinement_cache: OrderedDict[
            tuple[str, str], tuple[tuple[int, int, int], ...]
        ] = OrderedDict()
        self._quotient_cache: OrderedDict[tuple, tuple[MacroAction, ...]] = (
            OrderedDict()
        )
        self._ridge_consumed: dict[tuple[int, int], set[int]] = {}
        self._ridge_pairs: set[tuple[int, int]] = set()
        self._ridge_canonical_pairs: set[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = set()
        self._ridge_exhausted: set[tuple[int, int]] = set()
        self._known_terminal_keys: set[tuple[int, ...]] = set()
        self._frontier_discovery_counts: Counter[tuple[int, int]] = Counter()
        self._augmented = np.asarray(self.facet_oracle.augmented, dtype=float)
        self._slack_sum_row = self._augmented.sum(axis=0)
        self.quotient_stats: Counter[str] = Counter()

    @staticmethod
    def _dedupe_ids(pattern_ids: Iterable[str], primary: str) -> tuple[str, ...]:
        return tuple(
            pattern_id
            for pattern_id in dict.fromkeys(map(str, pattern_ids))
            if pattern_id != primary
        )

    @staticmethod
    def _terminal_key(facet: FacetResult) -> tuple[int, ...]:
        if facet.canonical_h_row is not None:
            return tuple(map(int, facet.canonical_h_row))
        if facet.h_row is not None:
            return tuple(map(int, facet.h_row))
        return (int(facet.support_word),)

    def observe_terminal(self, facet: FacetResult) -> None:
        if facet.valid:
            self._known_terminal_keys.add(self._terminal_key(facet))

    def record_frontier_discovery(
        self,
        source_facet_word: int,
        ridge_word: int,
    ) -> None:
        self._frontier_discovery_counts[
            (int(source_facet_word), int(ridge_word))
        ] += 1

    def ridge_frontier_priority(
        self,
        source_facet_word: int,
        ridge_word: int,
    ) -> float:
        """Online priority for a retained rank-78 frontier.

        Unopened and partially consumed frontiers receive a bonus. Exhausted
        frontiers without any discovery receive a penalty, while productive
        ones retain a small neutral value so their descendants remain usable.
        """

        key = int(source_facet_word), int(ridge_word)
        discoveries = self._frontier_discovery_counts[key]
        result = self._ridge_cache.get(key)
        if result is None:
            return 0.65 + min(0.20, 0.05 * discoveries)
        targets = tuple(
            target
            for target in result.facet_words
            if target != source_facet_word
        )
        consumed = self._ridge_consumed.get(key, set())
        remaining = sum(target not in consumed for target in targets)
        if remaining:
            return (
                0.25
                + min(0.45, 0.08 * math.sqrt(remaining))
                + min(0.20, 0.05 * discoveries)
            )
        if discoveries:
            return -0.05
        return -0.70

    def _common_refinement(
        self,
        primary_id: str,
        secondary_id: str,
    ) -> tuple[tuple[int, int, int], ...]:
        key = primary_id, secondary_id
        cached = self._refinement_cache.get(key)
        if cached is not None:
            self._refinement_cache.move_to_end(key)
            self.quotient_stats["refinement_cache_hits"] += 1
            return cached

        secondary_labels = [-1] * 256
        for secondary_index, block in enumerate(self.atlas.blocks(secondary_id)):
            for vertex in block:
                secondary_labels[vertex] = secondary_index
        cells: dict[tuple[int, int], int] = {}
        for primary_index, block in enumerate(self.atlas.blocks(primary_id)):
            for vertex in block:
                cell_key = primary_index, secondary_labels[vertex]
                cells[cell_key] = cells.get(cell_key, 0) | (1 << vertex)
        result = tuple(
            (primary_index, secondary_index, word)
            for (primary_index, secondary_index), word in sorted(cells.items())
            if word
        )
        self._refinement_cache[key] = result
        self._refinement_cache.move_to_end(key)
        while len(self._refinement_cache) > self.refinement_cache_size:
            self._refinement_cache.popitem(last=False)
            self.quotient_stats["refinement_cache_evictions"] += 1
        self.quotient_stats["refinement_cache_misses"] += 1
        return result

    @staticmethod
    def _word_bits(word: int) -> list[int]:
        bits: list[int] = []
        remaining = int(word)
        while remaining:
            bit = remaining & -remaining
            bits.append(bit)
            remaining ^= bit
        return bits

    def _rank_leverage_slice(
        self,
        support_word: int,
        cell_word: int,
        *,
        width: int,
        salt: str,
        budget: list[int],
    ) -> int:
        if width <= 0 or budget[0] <= 0:
            return 0
        token = int.from_bytes(
            hashlib.blake2b(salt.encode("ascii"), digest_size=8).digest(),
            "little",
        )
        bits = self._word_bits(cell_word)
        bits.sort(key=lambda bit: ((bit.bit_length() - 1) ^ token, bit))
        selected = 0
        rank = self.rank_oracle.rank(support_word)
        for bit in bits:
            if selected.bit_count() >= width or budget[0] <= 0:
                break
            budget[0] -= 1
            candidate = support_word | selected | bit
            candidate_rank = self.rank_oracle.rank(candidate)
            if candidate_rank > rank:
                selected |= bit
                rank = candidate_rank
        return selected

    def quotient_filler_actions(
        self,
        state: SearchState,
        primary_id: str,
        secondary_ids: Sequence[str],
        *,
        lane: str,
    ) -> list[MacroAction]:
        secondaries = self._dedupe_ids(secondary_ids, primary_id)[
            : self.quotient_secondary_limit
        ]
        cache_key = (
            "filler",
            state.support_word,
            primary_id,
            secondaries,
            lane,
        )
        cached = self._quotient_cache.get(cache_key)
        if cached is not None:
            self._quotient_cache.move_to_end(cache_key)
            self.quotient_stats["quotient_cache_hits"] += 1
            return list(cached)

        candidates: list[tuple[str, int, int, int]] = []
        for secondary_id in secondaries:
            for primary_index, secondary_index, cell in self._common_refinement(
                primary_id, secondary_id
            ):
                available = cell & ~state.support_word
                if available:
                    candidates.append(
                        (secondary_id, primary_index, secondary_index, available)
                    )
        candidates.sort(
            key=lambda item: (
                -item[3].bit_count() if lane == "broad" else item[3].bit_count(),
                item[0],
                item[1],
                item[2],
            )
        )
        actions: list[MacroAction] = []
        whole_limit = max(1, self.quotient_candidate_limit // 2)
        for secondary_id, primary_index, secondary_index, word in candidates[
            :whole_limit
        ]:
            actions.append(
                MacroAction(
                    "add",
                    primary_id,
                    (primary_index, secondary_index),
                    word,
                    f"quotient_intersection:{secondary_id}",
                    "quotient_refinement",
                    add_word=word,
                )
            )

        if lane in ("union", "broad"):
            window = candidates[: min(8, len(candidates))]
            for offset in range(max(0, self.quotient_candidate_limit - len(actions))):
                if len(window) < 2:
                    break
                left = window[offset % len(window)]
                right = window[(offset + 1) % len(window)]
                word = left[3] | right[3]
                actions.append(
                    MacroAction(
                        "add",
                        primary_id,
                        (left[1], left[2], right[1], right[2]),
                        word,
                        f"quotient_union:{left[0]}:{right[0]}",
                        "quotient_refinement_union",
                        add_word=word,
                    )
                )

        width = {"fine": 2, "single": 4, "union": 6, "broad": 8}.get(
            lane, 4
        )
        rank_budget = [self.partial_rank_checks]
        for secondary_id, primary_index, secondary_index, word in candidates[:4]:
            if word.bit_count() <= 1 or len(actions) >= self.quotient_candidate_limit:
                continue
            sliced = self._rank_leverage_slice(
                state.support_word,
                word,
                width=width,
                salt=(
                    f"{primary_id}:{secondary_id}:{primary_index}:"
                    f"{secondary_index}:{state.support_word}"
                ),
                budget=rank_budget,
            )
            if not sliced or sliced == word:
                continue
            actions.append(
                MacroAction(
                    "add",
                    primary_id,
                    (primary_index, secondary_index),
                    sliced,
                    f"quotient_rank_slice:{secondary_id}",
                    "quotient_partial_orbit",
                    add_word=sliced,
                )
            )
            self.quotient_stats["partial_orbit_actions"] += 1

        unique = tuple(
            list({action.action_id: action for action in actions}.values())[
                : self.quotient_candidate_limit
            ]
        )
        self._remember_quotient(cache_key, unique)
        self.quotient_stats["quotient_filler_candidates"] += len(unique)
        return list(unique)

    def supporting_face_actions(
        self,
        state: SearchState,
        pattern_id: str,
        *,
        nonce: int,
        count: int = 1,
    ) -> list[MacroAction]:
        """Sample full-space facets containing the current support.

        This is the supporting-face LP used by the reduced search, applied
        directly to the 256 original vertices. Subgroup orbits only shape the
        random objective; every returned support is already a full-space
        rank-79 candidate and therefore requires no lift step.
        """

        if count <= 0 or self.rank_oracle.rank(state.support_word) >= TARGET_AFFINE_RANK:
            return []
        cache_key = (
            "supporting_face_lp",
            state.support_word,
            pattern_id,
            int(nonce),
            int(count),
        )
        cached = self._quotient_cache.get(cache_key)
        if cached is not None:
            self._quotient_cache.move_to_end(cache_key)
            self.quotient_stats["quotient_cache_hits"] += 1
            return list(cached)

        selected = word_indices(state.support_word)
        blocks = self.atlas.blocks(pattern_id)
        equalities = [self._augmented[index] for index in selected]
        for block in blocks:
            representative = block[0]
            equalities.extend(
                self._augmented[vertex] - self._augmented[representative]
                for vertex in block[1:]
            )
        equalities.append(self._slack_sum_row)
        a_eq = np.asarray(equalities, dtype=float)
        b_eq = np.asarray([0.0] * (len(equalities) - 1) + [1.0], dtype=float)
        a_ub = -self._augmented
        b_ub = np.zeros(len(self._augmented), dtype=float)
        token = zlib.crc32(f"{pattern_id}:{nonce}".encode("ascii"))
        support_token = int.from_bytes(
            hashlib.blake2b(
                int(state.support_word).to_bytes(32, "little"),
                digest_size=8,
            ).digest(),
            "little",
        )
        rng = random.Random(self.seed ^ token ^ support_token)
        actions: list[MacroAction] = []
        seen: set[int] = set()
        for _attempt in range(max(count * 4, count + 2)):
            self.quotient_stats["supporting_face_lp_attempts"] += 1
            weights = np.empty(256, dtype=float)
            for block in blocks:
                weight = rng.gauss(0.0, 1.0)
                weights[list(block)] = weight
            objective = self._augmented.T @ weights
            objective += np.asarray(
                [
                    0.05 * rng.gauss(0.0, 1.0)
                    for _ in range(self._augmented.shape[1])
                ],
                dtype=float,
            )
            result = linprog(
                objective,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(None, None)] * self._augmented.shape[1],
                method="highs",
            )
            if not result.success:
                self.quotient_stats["supporting_face_lp_failures"] += 1
                continue
            slacks = self._augmented @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            tight = np.flatnonzero(np.abs(slacks) <= tolerance)
            support = sum(1 << int(index) for index in tight)
            if state.support_word & ~support:
                self.quotient_stats["supporting_face_lp_lost_selected"] += 1
                continue
            if support in seen or self.rank_oracle.rank(support) != TARGET_AFFINE_RANK:
                self.quotient_stats["supporting_face_lp_nonfacet"] += 1
                continue
            seen.add(support)
            add = support & ~state.support_word
            if not add:
                continue
            pattern = self.atlas.pattern(pattern_id)
            actions.append(
                MacroAction(
                    "add",
                    pattern_id,
                    (),
                    add,
                    "fullspace_supporting_face_lp",
                    f"{pattern.structure}:quotient_lp",
                    add_word=add,
                )
            )
            self.quotient_stats["supporting_face_lp_candidates"] += 1
            if len(actions) >= count:
                break
        unique = tuple({action.action_id: action for action in actions}.values())
        self._remember_quotient(cache_key, unique)
        return list(unique)

    def ridge_carve_actions(
        self,
        state: SearchState,
        primary_id: str,
        secondary_ids: Sequence[str],
    ) -> list[MacroAction]:
        if self.rank_oracle.rank(state.support_word) != TARGET_AFFINE_RANK:
            return []
        secondaries = self._dedupe_ids(secondary_ids, primary_id)[
            : self.quotient_secondary_limit
        ]
        cache_key = (
            "ridge",
            state.support_word,
            primary_id,
            secondaries,
        )
        cached = self._quotient_cache.get(cache_key)
        if cached is not None:
            self._quotient_cache.move_to_end(cache_key)
            self.quotient_stats["quotient_cache_hits"] += 1
            return list(cached)

        cells: list[tuple[str, int, int, int]] = []
        for secondary_id in secondaries:
            for primary_index, secondary_index, cell in self._common_refinement(
                primary_id, secondary_id
            ):
                inside = cell & state.support_word
                if inside:
                    cells.append(
                        (secondary_id, primary_index, secondary_index, inside)
                    )
        cells = list(
            {
                (secondary_id, primary_index, secondary_index, word)
                for secondary_id, primary_index, secondary_index, word in cells
            }
        )
        cells.sort(key=lambda item: (item[3].bit_count(), item[0], item[1], item[2]))
        window = cells[: min(16, len(cells))]
        candidates: list[tuple[tuple[int, ...], int, str]] = []
        for index, cell in enumerate(window):
            candidates.append(((cell[1], cell[2]), cell[3], cell[0]))
        for width in (2, 3, 4):
            for offset in range(len(window)):
                selected = [window[(offset + step) % len(window)] for step in range(width)]
                remove = 0
                indices: list[int] = []
                sources: list[str] = []
                for secondary_id, primary_index, secondary_index, word in selected:
                    remove |= word
                    indices.extend((primary_index, secondary_index))
                    sources.append(secondary_id)
                candidates.append((tuple(indices), remove, ":".join(sources)))

        ranked: list[tuple[int, int, MacroAction]] = []
        seen_words: set[int] = set()
        for indices, remove, source in candidates:
            if remove in seen_words or len(seen_words) >= self.ridge_rank_checks:
                continue
            seen_words.add(remove)
            rank = self.rank_oracle.rank(state.support_word & ~remove)
            self.quotient_stats["ridge_carve_rank_checks"] += 1
            if rank < TARGET_AFFINE_RANK - 3 or rank >= TARGET_AFFINE_RANK:
                continue
            exact = rank == TARGET_AFFINE_RANK - 1
            action = MacroAction(
                "remove",
                primary_id,
                indices,
                remove,
                "ridge_carve_exact" if exact else f"quotient_lower_face:{source}",
                "fullspace_ridge_frontier" if exact else "quotient_lower_face",
                remove_word=remove,
            )
            ranked.append((int(exact), rank, action))

        basis_budget = [self.ridge_basis_rank_checks]
        source_word = int(state.support_word)
        unique_cells = tuple(
            sorted(
                {word for _secondary, _primary, _index, word in cells},
                key=lambda word: (word.bit_count(), word),
            )
        )
        orderings = self._ridge_cell_orderings(
            unique_cells,
            salt=f"{primary_id}:{secondaries}:{source_word}",
        )
        for trial, ordering in enumerate(orderings[: self.ridge_basis_trials]):
            retained = self._maximal_rank78_support(
                source_word,
                ordering,
                budget=basis_budget,
            )
            if not retained or retained == source_word:
                continue
            remove = source_word & ~retained
            if remove in seen_words:
                continue
            seen_words.add(remove)
            action = MacroAction(
                "remove",
                primary_id,
                (trial,),
                remove,
                "quotient_ridge_basis",
                "fullspace_ridge_frontier",
                remove_word=remove,
            )
            ranked.append((1, TARGET_AFFINE_RANK - 1, action))
            self.quotient_stats["ridge_basis_candidates"] += 1
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                -item[2].block_size,
                item[2].action_id,
            ),
            reverse=True,
        )
        unique = tuple(
            item[2] for item in ranked[: self.quotient_candidate_limit]
        )
        self._remember_quotient(cache_key, unique)
        self.quotient_stats["ridge_carve_candidates"] += len(unique)
        self.quotient_stats["exact_ridge_carves"] += sum(
            action.source == "ridge_carve_exact" for action in unique
        )
        return list(unique)

    def _ridge_cell_orderings(
        self,
        cells: Sequence[int],
        *,
        salt: str,
    ) -> tuple[tuple[int, ...], ...]:
        if not cells:
            return ()
        digest = int.from_bytes(
            hashlib.blake2b(salt.encode("ascii"), digest_size=8).digest(),
            "little",
        )
        ascending = tuple(sorted(cells, key=lambda word: (word.bit_count(), word)))
        descending = tuple(reversed(ascending))
        hashed = tuple(
            sorted(
                cells,
                key=lambda word: (
                    int.from_bytes(
                        hashlib.blake2b(
                            (word ^ digest).to_bytes(32, "little"),
                            digest_size=8,
                        ).digest(),
                        "little",
                    ),
                    word,
                ),
            )
        )
        shifted = tuple(
            ascending[(index + digest) % len(ascending)]
            for index in range(len(ascending))
        )
        return ascending, descending, hashed, shifted

    def _maximal_rank78_support(
        self,
        source_word: int,
        cells: Sequence[int],
        *,
        budget: list[int],
    ) -> int:
        retained = 0
        retained_rank = -1
        deferred: list[int] = []
        for cell in cells:
            inside = int(cell) & source_word
            if not inside or budget[0] <= 0:
                continue
            candidate = retained | inside
            budget[0] -= 1
            rank = self.rank_oracle.rank(candidate)
            self.quotient_stats["ridge_basis_rank_checks"] += 1
            if rank <= TARGET_AFFINE_RANK - 1:
                retained = candidate
                retained_rank = rank
            else:
                deferred.append(inside)
        if retained_rank != TARGET_AFFINE_RANK - 1:
            return 0
        for cell in deferred:
            if budget[0] <= 0:
                break
            candidate = retained | cell
            budget[0] -= 1
            rank = self.rank_oracle.rank(candidate)
            self.quotient_stats["ridge_basis_rank_checks"] += 1
            if rank == TARGET_AFFINE_RANK - 1:
                retained = candidate
        return retained

    def _ridge_complete(self, state: SearchState) -> TailCompletion | None:
        key = state.source_facet_word, state.support_word
        result = self._ridge_cache.get(key)
        if result is None:
            self.stats["ridge_pivot_calls"] += 1
            result = adjacent_facet_words(
                self.facet_oracle.points,
                state.source_facet_word,
                state.support_word,
            )
            self._ridge_cache[key] = result
            self.stats["ridge_pivot_candidate_rays"] += result.candidate_rays
            self.stats["ridge_pivot_feasible_rays"] += result.feasible_rays
            self.stats["ridge_pivot_rejected_facets"] += result.rejected_facets
            if result.reason:
                self.stats[f"ridge_pivot_{result.reason}"] += 1
            else:
                self.quotient_stats["ridge_frontiers_opened"] += 1

        consumed = self._ridge_consumed.setdefault(key, set())
        source_facet = None
        source_key = None
        if self.canonical_frontier_dedup or self.unseen_exit_first:
            source_facet = self.facet_oracle.validate(state.source_facet_word)
            if source_facet.valid:
                source_key = self._terminal_key(source_facet)
        fallback: tuple[int, FacetResult, tuple[int, ...] | None] | None = None
        scanned = 0
        for target in result.facet_words:
            if target == state.source_facet_word or target in consumed:
                continue
            facet = self.facet_oracle.validate(target)
            if not facet.valid:
                consumed.add(target)
                self.stats["ridge_pivot_validation_failures"] += 1
                continue
            add = facet.support_word & ~state.support_word
            if not add:
                consumed.add(target)
                continue
            if (
                (self.canonical_frontier_dedup or self.unseen_exit_first)
                and source_key is not None
            ):
                scanned += 1
                self.quotient_stats["ridge_exit_scans"] += 1
                target_key = self._terminal_key(facet)
                canonical_pair = source_key, target_key
                if (
                    self.canonical_frontier_dedup
                    and canonical_pair in self._ridge_canonical_pairs
                ):
                    consumed.add(target)
                    self.quotient_stats[
                        "ridge_duplicate_canonical_pairs_pruned"
                    ] += 1
                    if scanned >= self.ridge_exit_scan:
                        break
                    continue
                if target_key in self._known_terminal_keys:
                    if fallback is None:
                        fallback = target, facet, target_key
                    if scanned >= self.ridge_exit_scan:
                        break
                    continue
                consumed.add(target)
                self._ridge_canonical_pairs.add(canonical_pair)
                self.quotient_stats["ridge_unseen_exits_selected"] += 1
                return self._ridge_completion(state, facet, add)
            consumed.add(target)
            return self._ridge_completion(state, facet, add)

        if fallback is not None and source_key is not None:
            target, facet, target_key = fallback
            consumed.add(target)
            self._ridge_canonical_pairs.add((source_key, target_key))
            self.quotient_stats["ridge_known_pair_exits_selected"] += 1
            add = facet.support_word & ~state.support_word
            return self._ridge_completion(state, facet, add)
        if result.facet_words and all(
            target == state.source_facet_word or target in consumed
            for target in result.facet_words
        ):
            if key not in self._ridge_exhausted:
                self._ridge_exhausted.add(key)
                self.quotient_stats["ridge_frontiers_exhausted"] += 1
        return None

    def _ridge_completion(
        self,
        state: SearchState,
        facet: FacetResult,
        add: int,
    ) -> TailCompletion:
        pair = state.source_facet_word, facet.support_word
        first_pair = pair not in self._ridge_pairs
        self._ridge_pairs.add(pair)
        self.stats["ridge_pivot_successes"] += 1
        self.quotient_stats["ridge_exits_consumed"] += 1
        self.quotient_stats["new_directed_ridge_pairs"] += int(first_pair)
        action = MacroAction(
            "add",
            state.context_pattern_id,
            (),
            add,
            "fullspace_ridge_frontier",
            "codimension_2_normal_cone",
            add_word=add,
        )
        return TailCompletion(action, facet, 1)

    def _remember_quotient(
        self,
        key: tuple,
        actions: tuple[MacroAction, ...],
    ) -> None:
        self._quotient_cache[key] = actions
        self._quotient_cache.move_to_end(key)
        while len(self._quotient_cache) > self.quotient_cache_size:
            self._quotient_cache.popitem(last=False)
            self.quotient_stats["quotient_cache_evictions"] += 1
        self.quotient_stats["quotient_cache_misses"] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            **super().to_dict(),
            "quotient_stats": dict(self.quotient_stats),
            "refinement_cache_entries": len(self._refinement_cache),
            "quotient_cache_entries": len(self._quotient_cache),
            "ridge_frontier_count": len(self._ridge_consumed),
            "directed_ridge_pair_count": len(self._ridge_pairs),
            "canonical_ridge_pair_count": len(self._ridge_canonical_pairs),
            "known_terminal_count": len(self._known_terminal_keys),
            "productive_frontier_count": len(self._frontier_discovery_counts),
            "quotient_candidate_limit": self.quotient_candidate_limit,
            "quotient_secondary_limit": self.quotient_secondary_limit,
            "partial_rank_checks": self.partial_rank_checks,
            "ridge_rank_checks": self.ridge_rank_checks,
            "ridge_basis_trials": self.ridge_basis_trials,
            "ridge_basis_rank_checks": self.ridge_basis_rank_checks,
            "ridge_exit_scan": self.ridge_exit_scan,
            "unseen_exit_first": self.unseen_exit_first,
            "canonical_frontier_dedup": self.canonical_frontier_dedup,
        }
