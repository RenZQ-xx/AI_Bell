from __future__ import annotations

"""Facet-relative repairs learned only from online validated terminals."""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import time

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from baseline.geometry import affine_rank, validate_facet_support
from baseline.orbit_blocks import build_state_group_permutations
from mcts.subgroup_patterns import (permutation_group_closure, subgroup_orbit_partition,
                                   subgroup_generating_set, conjugate_permutation, move_support_word)

from .knowledge import PatternSpec
from .model import block_word
from .repair_cache import SymmetryRepairCache, transported_pattern
from .ridge_enumeration import RidgeEnumerator
from .completion_frontier import CompletionFrontier
from .orbit_faces import OrbitFaceEnumerator
from .frontier_scheduler import FrontierScheduler
from .yield_frontier import YieldFrontier


@dataclass(frozen=True)
class RepairPlan:
    source_word: int
    retained_word: int
    pattern_id: str
    exit_words: tuple[int, ...]


@dataclass
class FacetAnchor:
    word: int
    normal: np.ndarray
    offset: float
    patterns: tuple[PatternSpec, ...]
    stabilizer_order: int
    rng: np.random.Generator
    attempts: int = 0
    plans: list[RepairPlan] = field(default_factory=list)
    seen_faces: set[int] = field(default_factory=set)
    projections: dict[str, np.ndarray] = field(default_factory=dict)
    generators: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    cache_cursor: int = 0
    requests: int = 0
    plan_keys: set[tuple[int, str]] = field(default_factory=set)


class FacetCorrectorBank:
    """Actual embedded H -> K partitions, retained ridges and geometric exits.

    Raw tight supports are cache keys: no conjugacy representative is silently
    reused in a different coordinate orientation. Class labels never enter here.
    """

    def __init__(self, points: np.ndarray, *, seed: int, max_attempts: int = 48, min_rank: int = 17,
                 ridge_max_candidates: int = 0, ridge_batch_size: int = 32,
                 endpoint_dedup: bool = False, lower_face_max_attempts: int = 0,
                 orbit_face_max_candidates: int = 0, orbit_face_batch_size: int = 8,
                 fair_frontier: bool = False, frontier_high_water: int = 24,
                 yield_frontier: bool = False):
        self.points = np.asarray(points, dtype=float)
        self.seed = seed
        self.max_attempts = max_attempts
        self.min_rank = min_rank
        self.anchors: dict[int, FacetAnchor] = {}
        self.exits: dict[tuple[int, int, str], tuple[int, ...]] = {}
        self.stats: Counter[str] = Counter()
        self._group: np.ndarray | None = None
        self.shared = SymmetryRepairCache()
        self.ridge_max_candidates = ridge_max_candidates
        self.ridge_batch_size = ridge_batch_size
        self.ridge_enumerators: dict[int, RidgeEnumerator] = {}
        self.frontier_plan_counts: Counter[tuple[int, int, str]] = Counter()
        self.endpoint_dedup = endpoint_dedup
        self.endpoint_canonical: dict[int, int] = {}
        self.endpoint_candidates: dict[tuple[int, int, int], int] = {}
        self.local_merge_events: list[dict[str, object]] = []
        self.current_iteration = 0
        self.completions = CompletionFrontier(seed=seed, max_attempts=lower_face_max_attempts)
        if orbit_face_max_candidates < 0 or orbit_face_batch_size <= 0:
            raise ValueError("invalid orbit face budget")
        self.orbit_face_max_candidates = orbit_face_max_candidates
        self.orbit_face_batch_size = orbit_face_batch_size
        self.orbit_faces = {}
        self.orbit_generators = {}
        self.orbit_face_publication_events = []
        self.fair_frontier = fair_frontier
        self.frontier_scheduler = FrontierScheduler(high_water=frontier_high_water)
        self.yield_frontier = yield_frontier and endpoint_dedup and not fair_frontier
        self.yield_scheduler = YieldFrontier()

    @staticmethod
    def indices(word: int) -> list[int]:
        return [i for i in range(64) if word & (1 << i)]

    def observe(self, word: int, normal: np.ndarray, offset: float) -> None:
        if word in self.anchors:
            return
        if self._group is None:
            self._group = np.asarray(build_state_group_permutations(), dtype=np.int16)
        self.shared.register(word, self._group)
        canonical = self.shared.coordinates[word][0]
        self.endpoint_canonical[word] = canonical
        if self.ridge_max_candidates and canonical not in self.ridge_enumerators:
            self.ridge_enumerators[canonical] = RidgeEnumerator(
                self.points, self.indices(canonical), max_candidates=self.ridge_max_candidates)
        membership = np.asarray([(word >> i) & 1 for i in range(64)], dtype=bool)
        stabilizer = self._group[np.all(membership[self._group] == membership, axis=1)]
        identity = tuple(range(64))
        groups = [(identity,), tuple(tuple(map(int, row)) for row in stabilizer)]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, word & 0xffffffff, word >> 32]))
        # Interleave actual cyclic subgroups with identity later. No atlas/class
        # representative is used to infer the embedding in this facet.
        for index in rng.permutation(len(stabilizer)):
            subgroup = permutation_group_closure((tuple(map(int, stabilizer[index])),), max_order=6)
            if subgroup is not None and len(subgroup) > 1:
                groups.append(subgroup)
            if len(groups) >= 32:
                break
        patterns = []
        generators = {}
        seen = set()
        for subgroup in groups:
            blocks = subgroup_orbit_partition(subgroup)
            if blocks in seen:
                continue
            seen.add(blocks)
            fingerprint = hashlib.sha256(repr(blocks).encode("ascii")).hexdigest()[:16]
            patterns.append(PatternSpec(
                pattern_id=f"EMBED-{fingerprint}", level=0,
                structure=f"embedded_order_{len(subgroup)}", blocks=blocks,
                source="online_facet_subgroup",
            ))
            generators[patterns[-1].pattern_id] = subgroup_generating_set(subgroup)
            if len(patterns) >= 12:
                break
        self.anchors[word] = FacetAnchor(
            word, np.asarray(normal).copy(), float(offset), tuple(patterns),
            len(stabilizer), rng,
        )
        self.anchors[word].generators = generators
        self.stats["anchors"] += 1
        self.stats["embedded_partitions"] += len(patterns)
        small = self.ridge_enumerators.get(canonical)
        if (self.orbit_face_max_candidates and canonical not in self.orbit_faces
                and (small is None or not small.eligible)):
            to_canonical = self.shared.coordinates[word][1]
            canonical_patterns = [transported_pattern(p, to_canonical) for p in patterns]
            self.orbit_generators[canonical] = {
                moved.pattern_id: tuple(conjugate_permutation(g, to_canonical)
                                        for g in generators[p.pattern_id])
                for p, moved in zip(patterns, canonical_patterns)
            }
            self.orbit_faces[canonical] = OrbitFaceEnumerator(
                self.points, canonical, canonical_patterns, min_rank=self.min_rank,
                max_candidates=self.orbit_face_max_candidates, seed=self.seed,
            )

    def _projection(self, anchor: FacetAnchor, pattern: PatternSpec) -> np.ndarray:
        if pattern.pattern_id not in anchor.projections:
            differences = [self.points[v] - self.points[b[0]] for b in pattern.blocks for v in b[1:]]
            invariant = null_space(np.asarray(differences).reshape(-1, self.points.shape[1]))
            tight = self.points[self.indices(anchor.word)]
            centered = tight - tight.mean(axis=0)
            _u, singular, vt = np.linalg.svd(centered @ invariant, full_matrices=False)
            keep = singular > 1e-8
            anchor.projections[pattern.pattern_id] = invariant @ vt[keep].T
        return anchor.projections[pattern.pattern_id]

    def grow(self, word: int) -> RepairPlan | None:
        anchor = self.anchors.get(word)
        if anchor is None:
            return None
        anchor.requests += 1
        if self.fair_frontier and (anchor.requests % 4 == 1 or anchor.attempts >= self.max_attempts):
            plan = self._scheduled_local_work(anchor)
            if plan is not None:
                return plan
        if not self.fair_frontier and self.ridge_max_candidates and (anchor.requests % 4 == 1 or anchor.attempts >= self.max_attempts):
            systematic = self._measured_work(anchor, "ridge")
            if systematic is not None:
                return systematic
        if not self.fair_frontier and (anchor.requests % 4 == 1 or anchor.attempts >= self.max_attempts):
            orbit_face = self._measured_work(anchor, "orbit")
            if orbit_face is not None:
                return orbit_face
            continuation = self._measured_work(anchor, "completion")
            if continuation is not None:
                return continuation
        if anchor.requests % 2 == 0 or anchor.attempts >= self.max_attempts:
            borrowed = self._borrow(anchor)
            if borrowed is not None:
                return borrowed
        if anchor.attempts >= self.max_attempts:
            return None
        attempt = anchor.attempts
        anchor.attempts += 1
        # Identity ensures unrestricted ridges remain reachable, while the other
        # half explores symmetry-preserving faces under actual subgroups of H.
        index = 0 if attempt % 2 == 0 else 1 + (attempt // 2) % max(1, len(anchor.patterns) - 1)
        pattern = anchor.patterns[min(index, len(anchor.patterns) - 1)]
        projection = self._projection(anchor, pattern)
        if projection.shape[1] == 0:
            self.stats["constant_invariant_space"] += 1
            return None
        indices = self.indices(word)
        tight = self.points[indices]
        inequalities = (tight - tight.mean(axis=0)) @ projection
        self.stats["lp_attempts"] += 1
        result = linprog(
            -anchor.rng.normal(size=projection.shape[1]), A_ub=inequalities,
            b_ub=np.ones(len(indices)), bounds=[(None, None)] * projection.shape[1],
            method="highs",
        )
        if not result.success:
            self.stats["lp_failed"] += 1
            return None
        retained = block_word(tuple(indices[i] for i in np.flatnonzero(np.abs(inequalities @ result.x - 1) <= 1e-7)))
        if retained in anchor.seen_faces:
            self.stats["duplicate_face"] += 1
            return None
        anchor.seen_faces.add(retained)
        rank = affine_rank(self.points[self.indices(retained)]) if retained else -1
        self.stats[f"retained_rank_{rank}"] += 1
        if not self.min_rank <= rank <= 24:
            self.stats["rank_out_of_range"] += 1
            return None
        return self._complete_face(anchor, retained, pattern, rank)

    def _complete_face(self, anchor, retained, pattern, rank):
        word = anchor.word
        exit_words = []
        for block in pattern.blocks if rank == 24 else ():
            outside = block_word(block) & ~word
            if not outside:
                continue
            self.stats["exit_geometry_checks"] += 1
            validation = validate_facet_support(self.points, self.indices(retained | outside))
            if validation.valid:
                exit_words.append(outside)
                self._remember_endpoint(word, retained, outside, validation)
        if rank < 24:
            exit_words = self._lower_face_exits(anchor, pattern, retained)
            if not exit_words and pattern.pattern_id != anchor.patterns[0].pattern_id:
                self.stats["refinement_attempts"] += 1
                pattern = anchor.patterns[0]
                exit_words = self._lower_face_exits(anchor, pattern, retained, attempts=1)
                if exit_words:
                    self.stats["refinement_successes"] += 1
        if not exit_words:
            self.stats["no_external_exit"] += 1
            return None
        return self._publish_plan(anchor, retained, pattern, tuple(exit_words))

    def _grow_orbit_faces(self, anchor):
        canonical, _, from_canonical = self.shared.coordinates[anchor.word]
        enumerator = self.orbit_faces.get(canonical)
        if enumerator is None or enumerator.finished:
            return None
        self.stats["orbit_face_batches"] += 1
        candidate = enumerator.advance(self.orbit_face_batch_size)
        if candidate is None:
            return None
        retained, pattern, rank = candidate
        retained = move_support_word(retained, from_canonical)
        moved_pattern = transported_pattern(pattern, from_canonical)
        if moved_pattern.pattern_id not in anchor.generators:
            anchor.patterns += (moved_pattern,)
            anchor.generators[moved_pattern.pattern_id] = tuple(
                conjugate_permutation(g, from_canonical)
                for g in self.orbit_generators[canonical][pattern.pattern_id])
        if rank == 24 and (retained, moved_pattern.pattern_id) in anchor.plan_keys:
            self.stats["orbit_face_existing_ridge"] += 1
            return None
        plan = self._complete_face(anchor, retained, moved_pattern, rank)
        if plan is not None:
            self.stats["orbit_face_publications"] += 1
            if len(self.orbit_face_publication_events) < 2000:
                self.orbit_face_publication_events.append({
                    "iteration": self.current_iteration,
                    "source_word_hex": f"0x{anchor.word:016x}",
                    "retained_word_hex": f"0x{retained:016x}",
                    "pattern_id": plan.pattern_id, "rank": rank,
                    "exit_words_hex": [f"0x{w:016x}" for w in plan.exit_words],
                })
        return plan

    def _publish_plan(self, anchor: FacetAnchor, retained: int, pattern: PatternSpec,
                      exit_words: tuple[int, ...]) -> RepairPlan | None:
        word = anchor.word
        exit_words = tuple(dict.fromkeys(exit_words))
        additions = exit_words
        if (retained, pattern.pattern_id) in anchor.plan_keys:
            self.stats["duplicate_plan"] += 1
            index = next(i for i, p in enumerate(anchor.plans)
                         if (p.retained_word, p.pattern_id) == (retained, pattern.pattern_id))
            old = anchor.plans[index]
            additions = tuple(w for w in exit_words if w not in old.exit_words)
            if not additions:
                self.stats["duplicate_plan_unchanged"] += 1
                return None
            plan = RepairPlan(word, retained, pattern.pattern_id, old.exit_words + additions)
            anchor.plans[index] = plan
            self.stats["merged_local_plans"] += 1
            self.stats["merged_local_exits"] += len(additions)
            if len(self.local_merge_events) < 2000:
                self.local_merge_events.append({
                    "iteration": self.current_iteration,
                    "source_word_hex": f"0x{word:016x}",
                    "retained_word_hex": f"0x{retained:016x}",
                    "pattern_id": pattern.pattern_id,
                    "added_exit_words_hex": [f"0x{w:016x}" for w in additions],
                    "previous_exit_count": len(old.exit_words),
                    "new_exit_count": len(plan.exit_words),
                })
        else:
            plan = RepairPlan(word, retained, pattern.pattern_id, exit_words)
            anchor.plans.append(plan)
            anchor.plan_keys.add((retained, pattern.pattern_id))
            self.stats["verified_plans"] += 1
            self.stats["identity_plans" if pattern.pattern_id == anchor.patterns[0].pattern_id else "subgroup_plans"] += 1
        self.exits[(word, retained, pattern.pattern_id)] = plan.exit_words
        self.stats["verified_exits"] += len(additions)
        self.shared.publish(word, retained, plan.exit_words, pattern, anchor.generators[pattern.pattern_id])
        if self.endpoint_dedup:
            for outside in additions:
                key = self.shared.exit_key(word, retained, outside)
                self.shared.endpoints.register(key, self.endpoint_candidates[key])
        return plan

    def _grow_systematic(self, anchor: FacetAnchor) -> RepairPlan | None:
        canonical, _, from_canonical = self.shared.coordinates[anchor.word]
        enumerator = self.ridge_enumerators.get(canonical)
        if enumerator is None or enumerator.finished:
            return None
        self.stats["systematic_batches"] += 1
        remaining = self.ridge_batch_size
        while remaining > 0 and not enumerator.finished:
            before = enumerator.examined
            canonical_retained = enumerator.advance(remaining)
            checked = enumerator.examined - before
            remaining -= checked
            self.stats["systematic_candidates"] += checked
            plan = self._check_systematic_candidate(anchor, canonical_retained, from_canonical)
            if plan is not None:
                return plan
        return None

    def _check_systematic_candidate(self, anchor, canonical_retained, from_canonical):
        if canonical_retained is None:
            return None
        retained = move_support_word(canonical_retained, from_canonical)
        pattern = anchor.patterns[0]
        if (retained, pattern.pattern_id) in anchor.plan_keys:
            self.stats["systematic_existing_plan"] += 1
            return None
        if affine_rank(self.points[self.indices(retained)]) != 24:
            self.stats["systematic_rank_rejected"] += 1
            return None
        exits = []
        for vertex in self.indices(((1 << 64) - 1) & ~anchor.word):
            outside = 1 << vertex
            self.stats["exit_geometry_checks"] += 1
            self.stats["systematic_exit_checks"] += 1
            validation = validate_facet_support(self.points, self.indices(retained | outside))
            if validation.valid:
                exits.append(outside)
                self._remember_endpoint(anchor.word, retained, outside, validation)
        if not exits:
            self.stats["systematic_no_exit"] += 1
            return None
        self.stats["systematic_plans"] += 1
        self.stats["systematic_exits"] += len(exits)
        anchor.seen_faces.add(retained)
        return self._publish_plan(anchor, retained, pattern, tuple(exits))

    def _borrow(self, anchor: FacetAnchor) -> RepairPlan | None:
        canonical, _, from_canonical = self.shared.coordinates[anchor.word]
        pool = self.shared.pools[canonical]
        while anchor.cache_cursor < len(pool):
            cached = pool[anchor.cache_cursor]
            anchor.cache_cursor += 1
            retained = move_support_word(cached.retained, from_canonical)
            pattern = transported_pattern(cached.pattern, from_canonical)
            if (retained, pattern.pattern_id) in anchor.plan_keys:
                existing = self.exits[(anchor.word, retained, pattern.pattern_id)]
                if all(move_support_word(w, from_canonical) in existing for w in cached.exits):
                    continue
            return self._import_plan(anchor, cached)
        return None

    def _import_plan(self, anchor, cached) -> RepairPlan:
        from_canonical = self.shared.coordinates[anchor.word][2]
        retained = move_support_word(cached.retained, from_canonical)
        pattern = transported_pattern(cached.pattern, from_canonical)
        transported_exits = tuple(move_support_word(w, from_canonical) for w in cached.exits)
        if (retained, pattern.pattern_id) in anchor.plan_keys:
            index = next(i for i, p in enumerate(anchor.plans)
                         if (p.retained_word, p.pattern_id) == (retained, pattern.pattern_id))
            old = anchor.plans[index]
            additions = tuple(w for w in transported_exits if w not in old.exit_words)
            if not additions:
                return old
            plan = RepairPlan(anchor.word, retained, pattern.pattern_id, old.exit_words + additions)
            anchor.plans[index] = plan
            self.exits[(anchor.word, retained, pattern.pattern_id)] = plan.exit_words
            self.stats["merged_cached_exits"] += len(additions)
            return plan
        if pattern.pattern_id not in anchor.generators:
            anchor.patterns += (pattern,)
            anchor.generators[pattern.pattern_id] = tuple(
                conjugate_permutation(g, from_canonical) for g in cached.generators)
        plan = RepairPlan(anchor.word, retained, pattern.pattern_id, transported_exits)
        anchor.plans.append(plan)
        anchor.plan_keys.add((retained, pattern.pattern_id))
        self.exits[(anchor.word, retained, pattern.pattern_id)] = plan.exit_words
        self.stats["transported_plans"] += 1
        self.stats["transported_exits"] += len(plan.exit_words)
        return plan

    def has_delivery_frontier(self, word: int) -> bool:
        if word not in self.anchors:
            return False
        canonical = self.shared.coordinates[word][0]
        enumerator = self.ridge_enumerators.get(canonical)
        pending = (self.shared.endpoints.pending[canonical] if self.endpoint_dedup
                   else self.shared.untried_exits[canonical])
        return bool((enumerator is not None and not enumerator.finished)
                    or (canonical in self.orbit_faces and not self.orbit_faces[canonical].finished)
                    or self.completions.pending(canonical)
                    or pending)

    def frontier_lanes(self, word: int) -> tuple[str, ...]:
        if word not in self.anchors:
            return ()
        canonical = self.shared.coordinates[word][0]
        lanes = []
        ridge = self.ridge_enumerators.get(canonical)
        orbit = self.orbit_faces.get(canonical)
        if ridge is not None and not ridge.finished:
            lanes.append("ridge")
        if orbit is not None and not orbit.finished:
            lanes.append("orbit")
        if self.completions.pending(canonical):
            lanes.append("completion")
        pending = (self.shared.endpoints.pending[canonical] if self.endpoint_dedup
                   else self.shared.untried_exits[canonical])
        if pending:
            lanes.append("delivery")
        return tuple(lanes)

    def choose_replay_frontier(self, sources):
        available = {}
        remaining = {}
        for canonical, raw in sources.items():
            for lane in self.frontier_lanes(raw):
                available.setdefault(lane, []).append(canonical)
            remaining[canonical] = self.completions.remaining_attempts(canonical)
        return self.frontier_scheduler.choose(available, remaining, scope=("replay", 0))

    def choose_yield_frontier(self, sources):
        available = {}
        for canonical, raw in sources.items():
            for lane in self.frontier_lanes(raw):
                available.setdefault(lane, []).append(canonical)
        return self.yield_scheduler.choose(available, iteration=self.current_iteration)

    def _measured_work(self, anchor, lane):
        workers = {"ridge": self._grow_systematic, "orbit": self._grow_orbit_faces,
                   "completion": self._grow_lower_faces}
        if lane == "delivery":
            return None
        worker = workers[lane]
        if not self.yield_frontier or lane not in self.frontier_lanes(anchor.word):
            return worker(anchor)
        before = set(self.shared.endpoints.pairs)
        observed = set(self.shared.pools)
        started = time.perf_counter()
        plan = worker(anchor)
        seconds = time.perf_counter() - started
        targets, pairs = self.yield_scheduler.novelty(before, observed, self.shared.endpoints.pairs)
        source = self.shared.coordinates[anchor.word][0]
        self.yield_scheduler.observe(source, lane, seconds, targets, pairs)
        return plan

    def _dispatch_frontier(self, anchor, lane, *, origin):
        if lane not in ("ridge", "orbit", "completion", "delivery"):
            raise ValueError("unknown frontier lane")
        self.stats[f"scheduled_{origin}_{lane}_dispatches"] += 1
        before = len(self.shared.endpoints.pairs)
        if lane == "ridge":
            plan = self._grow_systematic(anchor)
        elif lane == "orbit":
            plan = self._grow_orbit_faces(anchor)
        elif lane == "completion":
            plan = self._grow_lower_faces(anchor)
        else:
            plan = None
        self.stats[f"scheduled_{origin}_{lane}_new_pairs"] += len(self.shared.endpoints.pairs) - before
        if plan is not None:
            self.stats[f"scheduled_{origin}_{lane}_plans"] += 1
        return plan

    def _scheduled_local_work(self, anchor):
        canonical = self.shared.coordinates[anchor.word][0]
        available = {lane: (canonical,) for lane in self.frontier_lanes(anchor.word) if lane != "delivery"}
        choice = self.frontier_scheduler.choose(
            available, {canonical: self.completions.remaining_attempts(canonical)},
            scope=("grow", canonical))
        if choice is None:
            return None
        return self._dispatch_frontier(anchor, choice[0], origin="grow")

    def next_frontier_exit(self, word: int, *, lane: str | None = None) -> tuple[RepairPlan, int] | None:
        anchor = self.anchors[word]
        if self.fair_frontier:
            if lane is None:
                self._scheduled_local_work(anchor)
            else:
                self._dispatch_frontier(anchor, lane, origin="replay")
        elif self.yield_frontier and lane is not None:
            self._measured_work(anchor, lane)
        elif self._measured_work(anchor, "ridge") is None:
            if self._measured_work(anchor, "orbit") is None:
                self._measured_work(anchor, "completion")
        canonical, _, from_canonical = self.shared.coordinates[word]
        pending = self.shared.untried_exits[canonical]
        endpoint = None
        if self.endpoint_dedup:
            targets = self.shared.endpoints.pending[canonical]
            if not targets:
                self.stats["frontier_no_distinct_endpoint"] += 1
                return None
            endpoint = min(targets, key=lambda t: (self.shared.endpoints.proposals[(canonical, t)], t))
        def available(p, w):
            return ((p.retained, w) in pending and
                    (endpoint is None or self.shared.endpoints.targets.get((canonical, p.retained, w)) == endpoint))
        eligible = [p for p in self.shared.pools[canonical]
                    if any(available(p, w) for w in p.exits)]
        if not eligible:
            return None
        cached = min(eligible, key=lambda p: (self.frontier_plan_counts[(canonical, p.retained, p.pattern.pattern_id)],
                                              p.retained, p.pattern.pattern_id))
        self.frontier_plan_counts[(canonical, cached.retained, cached.pattern.pattern_id)] += 1
        outside = next(w for w in cached.exits if available(cached, w))
        if endpoint is not None:
            self.shared.endpoints.proposals[(canonical, endpoint)] += 1
        self.stats["frontier_exit_proposals"] += 1
        return self._import_plan(anchor, cached), move_support_word(outside, from_canonical)

    def has_frontier(self, word: int) -> bool:
        anchor = self.anchors.get(word)
        if anchor is None:
            return False
        canonical = self.shared.coordinates[word][0]
        enumerator = self.ridge_enumerators.get(canonical)
        return ((enumerator is not None and not enumerator.finished)
                or (canonical in self.orbit_faces and not self.orbit_faces[canonical].finished)
                or self.completions.pending(canonical)
                or anchor.attempts < self.max_attempts
                or anchor.cache_cursor < len(self.shared.pools[canonical])
                or any(self.shared.uses(word, p.retained_word, w) == 0
                       for p in anchor.plans for w in p.exit_words))

    def has_growth_frontier(self, word: int) -> bool:
        """Return whether this raw anchor can still produce or import geometry."""
        anchor = self.anchors.get(word)
        if anchor is None:
            return False
        canonical = self.shared.coordinates[word][0]
        ridge = self.ridge_enumerators.get(canonical)
        orbit = self.orbit_faces.get(canonical)
        return (
            (ridge is not None and not ridge.finished)
            or (orbit is not None and not orbit.finished)
            or self.completions.pending(canonical)
            or anchor.attempts < self.max_attempts
            or anchor.cache_cursor < len(self.shared.pools[canonical])
        )

    def _grow_lower_faces(self, anchor: FacetAnchor) -> RepairPlan | None:
        canonical = self.shared.coordinates[anchor.word][0]
        job = self.completions.next_job(canonical)
        if job is None:
            return None
        owner = self.anchors[job.source]
        pattern = next(p for p in owner.patterns if p.pattern_id == job.pattern_id)
        job.continuation_batches += 1
        self.stats["completion_continuation_batches"] += 1
        exits = self._lower_face_exits(owner, pattern, job.retained, continuation=True)
        if not exits:
            return None
        plan = self._publish_plan(owner, job.retained, pattern, tuple(exits))
        if plan is not None:
            self.stats["completion_continuation_publications"] += 1
        return plan if owner is anchor else self._borrow(anchor)

    def _lower_face_exits(self, anchor: FacetAnchor, pattern: PatternSpec, retained: int, *,
                         attempts: int = 2, continuation: bool = False) -> list[int]:
        job = None
        if self.completions.max_attempts:
            canonical, to_canonical, _ = self.shared.coordinates[anchor.word]
            moved_pattern = transported_pattern(pattern, to_canonical)
            key = (canonical, move_support_word(retained, to_canonical), moved_pattern.pattern_id)
            job = self.completions.register(key, source=anchor.word, retained=retained,
                                            pattern_id=pattern.pattern_id)
            attempts = min(attempts, self.completions.max_attempts - job.attempts)
            if attempts <= 0:
                self.stats["completion_budget_exhausted"] += 1
                return []
        differences = [self.points[v] - self.points[b[0]] for b in pattern.blocks for v in b[1:]]
        invariant = null_space(np.asarray(differences).reshape(-1, self.points.shape[1]))
        inequalities = (self.points - self.points.mean(axis=0)) @ invariant
        retained_indices = self.indices(retained)
        exits = set()
        # A vertex of this constrained polar is a candidate completion, not a
        # class-guided target. Degenerate/non-facet completions are validated out.
        for _ in range(attempts):
            self.stats["completion_lp_attempts"] += 1
            if job is not None:
                job.attempts += 1
            if continuation:
                self.stats["completion_continuation_lp_attempts"] += 1
            rng = job.rng if continuation and job is not None else anchor.rng
            result = linprog(
                -rng.normal(size=invariant.shape[1]),
                A_ub=inequalities, b_ub=np.ones(64),
                A_eq=inequalities[retained_indices], b_eq=np.ones(len(retained_indices)),
                bounds=[(None, None)] * invariant.shape[1], method="highs",
            )
            if not result.success:
                self.stats["completion_lp_failed"] += 1
                continue
            tight = block_word(tuple(map(int, np.flatnonzero(np.abs(inequalities @ result.x - 1) <= 1e-7))))
            if retained & ~tight or not (tight & ~anchor.word):
                self.stats["completion_source_or_lost_face"] += 1
                continue
            validation = validate_facet_support(self.points, self.indices(tight))
            if validation.valid:
                exits.add(tight & ~retained)
                self._remember_endpoint(anchor.word, retained, tight & ~retained, validation)
                if job is not None:
                    job.exit_keys.add(self.shared.exit_key(anchor.word, retained, tight & ~retained))
            else:
                self.stats["completion_nonfacet"] += 1
        if exits:
            self.stats["lower_face_plans"] += 1
        return sorted(exits)

    def _remember_endpoint(self, source: int, retained: int, added: int, validation) -> None:
        if not self.endpoint_dedup:
            return
        tight = block_word(tuple(map(int, np.flatnonzero(
            np.abs(self.points @ validation.normal + validation.offset) <= 1e-6))))
        if (retained | added) & ~tight:
            raise ValueError("verified completion lost selected vertices in its tight support")
        if tight not in self.endpoint_canonical:
            membership = np.asarray([(tight >> i) & 1 for i in range(64)], dtype=np.uint64)
            weights = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
            self.endpoint_canonical[tight] = int(np.min(np.sum(membership[self._group] * weights, axis=1, dtype=np.uint64)))
            self.stats["endpoint_canonicalizations"] += 1
        key = self.shared.exit_key(source, retained, added)
        self.endpoint_candidates[key] = self.endpoint_canonical[tight]

    def to_dict(self) -> dict[str, object]:
        unpublished = {key: target for key, target in self.endpoint_candidates.items()
                       if key not in self.shared.endpoints.targets}
        return {
            "statistics": dict(self.stats),
            "class_examples_loaded": False,
            "endpoint_dedup": self.endpoint_dedup,
            "endpoint_graph": self.shared.endpoints.to_dict(),
            "publication_audit": {
                "enabled": self.endpoint_dedup,
                "verified_candidate_action_keys": len(self.endpoint_candidates),
                "published_action_keys": len(self.shared.endpoints.targets),
                "unpublished_candidate_action_keys": len(unpublished),
                "unpublished_candidate_targets_hex": [f"0x{t:016x}" for t in sorted(set(unpublished.values()))],
            },
            "local_merge_events": self.local_merge_events,
            "lower_face_frontier": self.completions.to_dict(),
            "orbit_face_frontiers": {f"0x{word:016x}": e.to_dict()
                                      for word, e in self.orbit_faces.items()},
            "orbit_face_publication_events": self.orbit_face_publication_events,
            "frontier_scheduling": {"enabled": self.fair_frontier, **self.frontier_scheduler.to_dict()},
            "yield_scheduling": {"enabled": self.yield_frontier, **self.yield_scheduler.to_dict()},
            "cache_key": "raw_tight_support_in_actual_vertex_coordinates",
            "shared_canonical_facets": len(self.shared.pools),
            "shared_repair_count": sum(map(len, self.shared.pools.values())),
            "systematic_ridges": {f"0x{word:016x}": e.to_dict()
                                  for word, e in self.ridge_enumerators.items()},
            "untried_exit_counts": {f"0x{word:016x}": len(pending)
                                    for word, pending in self.shared.untried_exits.items()},
            "anchors": [{
                "tight_support_word_hex": f"0x{a.word:016x}",
                "stabilizer_order": a.stabilizer_order,
                "pattern_ids": [p.pattern_id for p in a.patterns],
                "attempts": a.attempts,
                "verified_plan_count": len(a.plans),
                "embedded_generators": a.generators,
            } for a in self.anchors.values()],
        }
