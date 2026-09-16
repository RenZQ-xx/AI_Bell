from __future__ import annotations

"""Group-aware action generation for both players of the 4-2-2 game."""

import hashlib
import random
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .atlas import HandoffPatternAtlas, IDENTITY_PATTERN_ID
from .geometry import FacetOracle, FacetResult, RankOracle
from .model import MacroAction, SearchState
from .problem import (
    TARGET_AFFINE_RANK,
    block_word,
    compose_permutations,
    move_word,
    orbit_partition,
    permutation_closure,
    vertex_symmetry_generators,
)
from mcts.symmetry_game.ridge import RidgePivotResult, adjacent_facet_words


FILLER_LANES = ("fine", "single", "single", "union", "fine", "broad")
CORRECTOR_LANES = ("remove_small", "rewrite", "remove_broad", "rewrite")


@dataclass(frozen=True)
class TailCompletion:
    action: MacroAction
    facet: FacetResult
    checks: int


class FillerCorrector:
    def __init__(
        self,
        atlas: HandoffPatternAtlas,
        rank_oracle: RankOracle,
        facet_oracle: FacetOracle,
        *,
        seed: int,
        filler_candidate_limit: int = 72,
        corrector_candidate_limit: int = 64,
        tail_checks: int = 96,
        stabilizer_probes: int = 192,
        candidate_cache_size: int = 20_000,
        context_free_cache_keys: bool = False,
        ridge_pivot: bool = False,
        filler_max_orbits_per_action: int = 3,
        corrector_max_orbits_per_action: int = 2,
        max_pattern_embeddings_per_family: int = 4,
    ):
        self.atlas = atlas
        self.rank_oracle = rank_oracle
        self.facet_oracle = facet_oracle
        self.seed = int(seed)
        self.filler_candidate_limit = int(filler_candidate_limit)
        self.corrector_candidate_limit = int(corrector_candidate_limit)
        self.tail_checks = int(tail_checks)
        self.stabilizer_probes = int(stabilizer_probes)
        self.candidate_cache_size = int(candidate_cache_size)
        self.context_free_cache_keys = bool(context_free_cache_keys)
        self.ridge_pivot = bool(ridge_pivot)
        self.filler_max_orbits_per_action = max(1, int(filler_max_orbits_per_action))
        self.corrector_max_orbits_per_action = max(1, int(corrector_max_orbits_per_action))
        self.max_pattern_embeddings_per_family = max(
            1, int(max_pattern_embeddings_per_family)
        )
        self._candidate_cache: OrderedDict[
            tuple[tuple[int, str, int, str, int], str, str], tuple[MacroAction, ...]
        ] = OrderedDict()
        self.facet_patterns: dict[int, tuple[str, ...]] = {}
        self.family_embeddings: dict[str, set[str]] = {}
        self.stats: Counter[str] = Counter()
        self._ridge_cache: dict[tuple[int, int], RidgePivotResult] = {}

    def _cache_state_key(self, state: SearchState) -> tuple:
        if self.context_free_cache_keys:
            return state.support_word, state.role
        return state.key

    def _rng(self, state: SearchState, pattern_id: str, lane: str) -> random.Random:
        token = zlib.crc32(f"{pattern_id}:{lane}".encode("ascii"))
        support_token = int.from_bytes(
            hashlib.blake2b(
                int(state.support_word).to_bytes(32, "little"), digest_size=8
            ).digest(),
            "little",
        )
        return random.Random(self.seed ^ token ^ support_token)

    def filler_actions(
        self,
        state: SearchState,
        pattern_id: str,
        *,
        lane: str = "single",
    ) -> list[MacroAction]:
        cache_key = (self._cache_state_key(state), pattern_id, lane)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            self._candidate_cache.move_to_end(cache_key)
            self.stats["candidate_cache_hits"] += 1
            return list(cached)
        blocks = self.atlas.blocks(pattern_id)
        pattern = self.atlas.pattern(pattern_id)
        available = [
            (index, block_word(block) & ~state.support_word)
            for index, block in enumerate(blocks)
        ]
        available = [(index, word) for index, word in available if word]
        rng = self._rng(state, pattern_id, lane)
        rng.shuffle(available)
        available.sort(key=lambda item: (item[1].bit_count(), item[0]))
        if lane == "broad":
            available.reverse()
        elif lane == "fine":
            minimum = min((word.bit_count() for _index, word in available), default=0)
            available = [item for item in available if item[1].bit_count() <= minimum + 1]

        union_arity = self.filler_max_orbits_per_action if lane == "broad" else 2
        union_budget = (
            min(24, max(1, self.filler_candidate_limit // 3))
            if lane in ("union", "broad") and len(available) > 1
            else 0
        )
        single_limit = max(0, self.filler_candidate_limit - union_budget)
        actions = [
            MacroAction("add", pattern_id, (index,), word, pattern.source, pattern.structure, add_word=word)
            for index, word in available[:single_limit]
        ]
        if lane in ("union", "broad") and len(available) > 1:
            window = available[: min(24, len(available))]
            for start in range(len(window)):
                selected = [window[(start + offset) % len(window)] for offset in range(union_arity)]
                indices = tuple(sorted(index for index, _word in selected))
                word = 0
                for _index, value in selected:
                    word |= value
                actions.append(
                    MacroAction(
                        "add",
                        pattern_id,
                        indices,
                        word,
                        pattern.source,
                        pattern.structure,
                        add_word=word,
                    )
                )
                if len(actions) >= self.filler_candidate_limit:
                    break
        unique = tuple({action.action_id: action for action in actions}.values())
        self._candidate_cache[cache_key] = unique
        self._trim_candidate_cache()
        self.stats["candidate_cache_misses"] += 1
        self.stats["filler_candidates"] += len(unique)
        return list(unique)

    def corrector_actions(
        self,
        state: SearchState,
        pattern_id: str,
        *,
        lane: str = "remove_small",
    ) -> list[MacroAction]:
        cache_key = (self._cache_state_key(state), pattern_id, lane)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            self._candidate_cache.move_to_end(cache_key)
            self.stats["candidate_cache_hits"] += 1
            return list(cached)
        pattern = self.atlas.pattern(pattern_id)
        blocks = self.atlas.blocks(pattern_id)
        inside = [(i, block_word(block) & state.support_word) for i, block in enumerate(blocks)]
        inside = [(i, word) for i, word in inside if word]
        outside = [(i, block_word(block) & ~state.support_word) for i, block in enumerate(blocks)]
        outside = [(i, word) for i, word in outside if word]
        rng = self._rng(state, pattern_id, lane)
        rng.shuffle(inside)
        rng.shuffle(outside)
        inside.sort(key=lambda item: (item[1].bit_count(), item[0]), reverse=lane == "remove_broad")
        actions: list[MacroAction] = []
        if lane.startswith("remove"):
            union_budget = (
                min(24, max(1, self.corrector_candidate_limit // 2))
                if lane == "remove_broad" and len(inside) > 1
                else 0
            )
            single_limit = max(0, self.corrector_candidate_limit - union_budget)
            for index, remove in inside[:single_limit]:
                actions.append(
                    MacroAction(
                        "remove", pattern_id, (index,), remove, pattern.source,
                        pattern.structure, remove_word=remove,
                    )
                )
            if lane == "remove_broad":
                union_limit = min(
                    union_budget, self.corrector_candidate_limit - len(actions)
                )
                widths = tuple(
                    range(
                        2,
                        min(self.corrector_max_orbits_per_action, len(inside)) + 1,
                    )
                )
                for offset in range(max(0, union_limit)):
                    if not widths:
                        break
                    width = widths[offset % len(widths)]
                    selected = [inside[(offset + index) % len(inside)] for index in range(width)]
                    remove = 0
                    for _index, word in selected:
                        remove |= word
                    actions.append(
                        MacroAction(
                            "remove", pattern_id,
                            tuple(sorted(index for index, _word in selected)), remove,
                            pattern.source, pattern.structure, remove_word=remove,
                        )
                    )
        else:
            for remove_index, remove in inside[:12]:
                for add_index, add in outside[:4]:
                    effective = remove | add
                    actions.append(
                        MacroAction(
                            "rewrite", pattern_id, (remove_index, add_index), effective,
                            "symmetry_rewrite", pattern.structure,
                            remove_word=remove, add_word=add,
                        )
                    )
                    if len(actions) >= self.corrector_candidate_limit:
                        break
                if len(actions) >= self.corrector_candidate_limit:
                    break
        unique = tuple({action.action_id: action for action in actions}.values())
        self._candidate_cache[cache_key] = unique
        self._trim_candidate_cache()
        self.stats["candidate_cache_misses"] += 1
        self.stats["corrector_candidates"] += len(unique)
        return list(unique)

    def cross_pattern_rewrites(
        self, state: SearchState, source_pattern_id: str, *, limit: int = 24
    ) -> list[MacroAction]:
        source_blocks = self.atlas.blocks(source_pattern_id)
        removals = [
            (index, block_word(block) & state.support_word)
            for index, block in enumerate(source_blocks)
        ]
        removals = sorted(
            ((index, word) for index, word in removals if word),
            key=lambda item: (item[1].bit_count(), item[0]),
        )[:8]
        actions: list[MacroAction] = []
        for target_id in self.atlas.neighbors(source_pattern_id, limit=8):
            target = self.atlas.pattern(target_id)
            additions = [
                (index, block_word(block) & ~state.support_word)
                for index, block in enumerate(self.atlas.blocks(target_id))
            ]
            additions = sorted(
                ((index, word) for index, word in additions if word),
                key=lambda item: (item[1].bit_count(), item[0]),
            )[:4]
            for remove_index, remove in removals[:3]:
                for add_index, add in additions:
                    actions.append(
                        MacroAction(
                            "rewrite", target_id, (remove_index, add_index), remove | add,
                            "cross_pattern_rewrite", target.structure,
                            remove_word=remove, add_word=add,
                        )
                    )
                    if len(actions) >= limit:
                        return actions
        return actions

    def apply(self, state: SearchState, action: MacroAction) -> tuple[int, int]:
        before = state.support_word
        if action.kind == "add":
            after = before | action.add_word
        elif action.kind == "remove":
            after = before & ~action.remove_word
        elif action.kind == "rewrite":
            after = (before & ~action.remove_word) | action.add_word
        else:
            after = before
        return after, self.rank_oracle.rank(after)

    def tail_complete(
        self,
        state: SearchState,
        pattern_ids: Sequence[str],
        *,
        max_checks: int | None = None,
    ) -> TailCompletion | None:
        rank = self.rank_oracle.rank(state.support_word)
        if rank not in (TARGET_AFFINE_RANK - 2, TARGET_AFFINE_RANK - 1):
            return None
        if (
            self.ridge_pivot
            and rank == TARGET_AFFINE_RANK - 1
            and state.source_facet_word
            and not (state.support_word & ~state.source_facet_word)
        ):
            completion = self._ridge_complete(state)
            if completion is not None:
                return completion
        budget = self.tail_checks if max_checks is None else int(max_checks)
        checks = 0
        for pattern_id in dict.fromkeys(pattern_ids):
            pattern = self.atlas.pattern(pattern_id)
            available = [
                (index, block_word(block) & ~state.support_word)
                for index, block in enumerate(self.atlas.blocks(pattern_id))
            ]
            available = [(index, word) for index, word in available if word]
            available.sort(key=lambda item: (item[1].bit_count(), item[0]))
            candidates: list[tuple[tuple[int, ...], int]] = [
                ((index,), word) for index, word in available
            ]
            if rank == TARGET_AFFINE_RANK - 2:
                candidates = []
                window = available[: min(24, len(available))]
                for first in range(len(window)):
                    for second in range(first + 1, len(window)):
                        candidates.append(
                            (
                                (window[first][0], window[second][0]),
                                window[first][1] | window[second][1],
                            )
                        )
            for indices, add in candidates:
                if checks >= budget:
                    self.stats["tail_budget_exhausted"] += 1
                    return None
                checks += 1
                self.stats["tail_candidate_checks"] += 1
                candidate = state.support_word | add
                if self.rank_oracle.rank(candidate) != TARGET_AFFINE_RANK:
                    continue
                self.stats["tail_geometry_checks"] += 1
                facet = self.facet_oracle.validate(candidate)
                if facet.valid:
                    self.stats["tail_successes"] += 1
                    action = MacroAction(
                        "add", pattern_id, indices, add, "rank_tail_enumeration",
                        pattern.structure, add_word=add,
                    )
                    return TailCompletion(action, facet, checks)
        return None

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
        for target in result.facet_words:
            if target == state.source_facet_word:
                continue
            facet = self.facet_oracle.validate(target)
            if not facet.valid:
                self.stats["ridge_pivot_validation_failures"] += 1
                continue
            add = facet.support_word & ~state.support_word
            if not add:
                continue
            self.stats["ridge_pivot_successes"] += 1
            action = MacroAction(
                "add",
                state.context_pattern_id,
                (),
                add,
                "rank_ridge_pivot",
                "codimension_2_normal_cone",
                add_word=add,
            )
            return TailCompletion(action, facet, 1)
        return None

    def _trim_candidate_cache(self) -> None:
        while len(self._candidate_cache) > self.candidate_cache_size:
            self._candidate_cache.popitem(last=False)
            self.stats["candidate_cache_evictions"] += 1

    def observe_facet(
        self,
        support_word: int,
        context_pattern_id: str,
        *,
        family_id: str = "",
    ) -> tuple[str, ...]:
        cached = self.facet_patterns.get(int(support_word))
        if cached is not None:
            return cached
        if family_id:
            embeddings = self.family_embeddings.setdefault(family_id, set())
            if len(embeddings) >= self.max_pattern_embeddings_per_family:
                self.stats["family_embedding_cap_skips"] += 1
                self.facet_patterns[int(support_word)] = tuple(
                    dict.fromkeys(
                        (*sorted(embeddings), context_pattern_id, IDENTITY_PATTERN_ID)
                    )
                )
                return self.facet_patterns[int(support_word)]
        named = vertex_symmetry_generators()
        preserving = [permutation for _name, permutation in named if move_word(support_word, permutation) == support_word]
        identity = tuple(range(256))
        support_seed = int.from_bytes(
            hashlib.blake2b(int(support_word).to_bytes(32, "little"), digest_size=8).digest(),
            "little",
        )
        rng = random.Random(self.seed ^ support_seed)
        current = identity
        for _ in range(self.stabilizer_probes):
            _name, generator = named[rng.randrange(len(named))]
            current = compose_permutations(current, generator)
            if move_word(support_word, current) == support_word:
                preserving.append(current)
        if not preserving:
            self.facet_patterns[int(support_word)] = (context_pattern_id, IDENTITY_PATTERN_ID)
            self.stats["trivial_sampled_stabilizers"] += 1
            return self.facet_patterns[int(support_word)]
        subgroup = permutation_closure(tuple(dict.fromkeys(preserving)), max_order=384)
        if subgroup is None:
            # Keep a small generated subgroup when the sampled stabilizer is large.
            subgroup = permutation_closure((preserving[0],), max_order=64) or (identity,)
        blocks = orbit_partition(subgroup)
        fingerprint = hashlib.sha256(repr(blocks).encode("ascii")).hexdigest()[:16]
        pattern_id = (
            f"{family_id}-E{fingerprint[:12]}"
            if family_id
            else f"ONLINE422-{fingerprint}"
        )
        self.atlas.register_online_pattern(
            pattern_id,
            blocks,
            structure=f"sampled_stabilizer_order_{len(subgroup)}",
            family_id=family_id,
        )
        if family_id:
            self.family_embeddings.setdefault(family_id, set()).add(pattern_id)
        self.facet_patterns[int(support_word)] = tuple(
            dict.fromkeys((pattern_id, context_pattern_id, *self.atlas.neighbors(context_pattern_id, limit=6)))
        )
        self.stats["online_stabilizer_patterns"] += 1
        self.stats[f"online_stabilizer_order_{len(subgroup)}"] += 1
        return self.facet_patterns[int(support_word)]

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": dict(self.stats),
            "candidate_cache_entries": len(self._candidate_cache),
            "ridge_cache_entries": len(self._ridge_cache),
            "context_free_cache_keys": self.context_free_cache_keys,
            "ridge_pivot": self.ridge_pivot,
            "observed_facet_pattern_count": len(self.facet_patterns),
            "pattern_family_count": len(self.family_embeddings),
            "family_embeddings": {
                family_id: sorted(pattern_ids)
                for family_id, pattern_ids in sorted(self.family_embeddings.items())
            },
            "facet_patterns": {
                f"0x{word:064x}": list(patterns)
                for word, patterns in self.facet_patterns.items()
            },
        }
