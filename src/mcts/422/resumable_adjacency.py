from __future__ import annotations

"""Checkpointed AS jobs using the existing certified geometry primitives."""

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .analyze_unmatched_symmetry import modular_rank
from .recursive_adjacency import (
    FacetCertificate, GeometryUncertain, RecursiveAdjacency, SamplingDeadline,
)
from .reduced_subgroup_search import word_from_indices


@dataclass
class AdjacencyJob:
    source: int
    chain: list[int] = field(default_factory=list)
    simplices: tuple[tuple[int, ...], ...] | None = None
    simplex_cursor: int = 0
    leaf_facets: dict[int, FacetCertificate] = field(default_factory=dict)
    leaves: tuple[FacetCertificate, ...] | None = None
    leaf_cursor: int = 0
    level: int | None = None
    ridge: int = 0
    rotations: list[dict] = field(default_factory=list)
    outputs: int = 0
    advances: int = 0
    status: str = "pending"

    @property
    def done(self):
        return self.status in ("chain_finished", "output_limit") or self.status.startswith("uncertified:")


class ResumableAdjacency(RecursiveAdjacency):
    def new_job(self, source):
        self.stats["jobs_started"] += 1
        self.stats["macro_calls"] += 1
        return AdjacencyJob(source)

    def _leaf_plan(self, job):
        word = job.chain[-1]
        if word in self.leaf_cache:
            self.stats["leaf_cache_hits"] += 1
            job.leaves = self.leaf_cache[word]
            return
        leaf = self.frame(word)
        if len(leaf.indices) > self.config.cutoff_vertices or leaf.dimension < 1:
            raise GeometryUncertain("leaf is not within enumeration cutoff")
        started = time.perf_counter()
        if leaf.dimension == 1:
            values = leaf.augmented[:, 1]
            simplices = [(int(np.argmin(values)),), (int(np.argmax(values)),)]
        elif len(leaf.indices) == leaf.dimension + 1:
            simplices = [tuple(j for j in range(len(leaf.indices)) if j != i)
                         for i in range(len(leaf.indices))]
        else:
            try:
                simplices = ConvexHull(leaf.augmented[:, 1:].astype(float)).simplices
            except QhullError as error:
                raise GeometryUncertain("Qhull leaf enumeration failed") from error
        self.seconds["leaf_hull"] += time.perf_counter() - started
        if len(simplices) > self.config.max_leaf_simplices:
            raise GeometryUncertain("leaf simplex output limit")
        job.simplices = tuple(tuple(map(int, simplex)) for simplex in simplices)

    def _enumerate_step(self, job):
        leaf = self.frame(job.chain[-1])
        simplex = job.simplices[job.simplex_cursor]
        if modular_rank(leaf.augmented[list(simplex)]) == leaf.dimension:
            selected = word_from_indices(leaf.indices[i] for i in simplex)
            certificate = self.certify(leaf.word, selected)
            job.leaf_facets[certificate.word] = certificate
            if len(job.leaf_facets) > self.config.max_leaf_facets:
                raise GeometryUncertain("leaf facet output limit")
        else:
            self.stats["degenerate_leaf_simplices_skipped"] += 1
        job.simplex_cursor += 1
        if job.simplex_cursor == len(job.simplices):
            if len(job.leaf_facets) < leaf.dimension + 1:
                raise GeometryUncertain("too few leaf supporting facets")
            job.leaves = tuple(job.leaf_facets[word] for word in sorted(job.leaf_facets))
            self.leaf_cache[leaf.word] = job.leaves
            self.stats["leaf_enumerations_finished"] += 1
            self.stats["leaf_facets_enumerated"] += len(job.leaves)
            job.simplices = None
            job.leaf_facets.clear()

    def advance(self, job, rng, *, max_outputs=2, max_steps=None, seconds=None, deadline=math.inf):
        if max_outputs < 1 or (max_steps is not None and max_steps < 1):
            raise ValueError("advance limits must be positive")
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("advance seconds must be finite and positive")
        if job.done:
            return []
        started = time.perf_counter()
        self.deadline = min(deadline, started + (seconds or self.config.max_macro_seconds))
        outputs, steps = [], 0
        job.advances += 1
        self.stats["advance_calls"] += 1
        self.stats["resumed_advances"] += int(job.advances > 1)
        try:
            while not job.done and len(outputs) < max_outputs:
                self.check_deadline()
                if max_steps is not None and steps >= max_steps:
                    break
                steps += 1
                if not job.chain:
                    source = self.certify(self.full_word, job.source)
                    if source.word != job.source:
                        raise GeometryUncertain("job source must be a closed facet")
                    job.chain = [self.full_word, source.word]
                elif job.chain[-1].bit_count() > self.config.cutoff_vertices:
                    job.chain.append(self.sample_facet(job.chain[-1], rng).word)
                elif job.leaves is None:
                    if job.simplices is None:
                        self._leaf_plan(job)
                    else:
                        self._enumerate_step(job)
                elif job.leaf_cursor >= len(job.leaves):
                    job.status = "chain_finished"
                elif job.level is None:
                    job.ridge = job.leaves[job.leaf_cursor].word
                    job.rotations = []
                    job.level = len(job.chain) - 2
                elif job.level >= 0:
                    level = job.level
                    job.ridge, proof = self.rotate(job.chain[level], job.chain[level+1], job.ridge)
                    job.rotations.append(proof)
                    job.level -= 1
                else:
                    outputs.append({
                        "destination_word_hex": hex(job.ridge),
                        "retained_word_hex": hex(job.source & job.ridge),
                        "chain_word_hex": list(map(hex, job.chain)),
                        "leaf_boundary_word_hex": hex(job.leaves[job.leaf_cursor].word),
                        "rotations": list(job.rotations),
                    })
                    job.outputs += 1
                    job.leaf_cursor += 1
                    job.level = None
                    if job.outputs >= self.config.max_outputs:
                        job.status = "output_limit"
                    elif job.leaf_cursor == len(job.leaves):
                        job.status = "chain_finished"
            if job.done:
                self.stats["jobs_finished"] += 1
                self.stats["status_" + job.status] += 1
        except SamplingDeadline:
            self.stats["paused_on_deadline"] += 1
        except GeometryUncertain as error:
            job.status = "uncertified: " + str(error)
            self.stats["jobs_uncertified"] += 1
        finally:
            self.deadline = math.inf
            self.seconds["macro_inclusive"] += time.perf_counter() - started
        self.stats["advance_steps"] += steps
        self.stats["outputs_generated"] += len(outputs)
        return outputs
