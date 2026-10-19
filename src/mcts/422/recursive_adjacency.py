from __future__ import annotations

"""Bounded recursive adjacency sampling on actual faces, without answer data."""

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from .analyze_unmatched_symmetry import modular_rank
from .geometry import _primitive_integer_row
from .reduced_subgroup_search import word_from_indices, word_indices


class GeometryUncertain(ValueError):
    pass


class SamplingDeadline(TimeoutError):
    pass


@dataclass(frozen=True)
class AdjacencyConfig:
    cutoff_vertices: int = 16
    max_leaf_simplices: int = 8192
    max_leaf_facets: int = 2048
    max_outputs: int = 32
    max_macro_seconds: float = 1.0

    def __post_init__(self):
        if not 4 <= self.cutoff_vertices <= 20:
            raise ValueError("cutoff_vertices must be between 4 and 20")
        if min(self.max_leaf_simplices, self.max_leaf_facets, self.max_outputs) < 1:
            raise ValueError("enumeration/output limits must be positive")
        if not math.isfinite(self.max_macro_seconds) or self.max_macro_seconds <= 0:
            raise ValueError("max_macro_seconds must be finite and positive")


@dataclass
class FaceFrame:
    word: int
    indices: tuple[int, ...]
    axes: tuple[int, ...]
    augmented: np.ndarray
    positions: dict[int, int]

    @property
    def dimension(self):
        return len(self.axes)


@dataclass(frozen=True)
class FacetCertificate:
    parent_word: int
    word: int
    row: tuple[int, ...]


@dataclass
class AdjacencyBatch:
    source_word: int
    chain: tuple[int, ...] = ()
    outputs: list[dict] = field(default_factory=list)
    status: str = "not_started"
    leaf_facets: int = 0
    leaf_enumeration_finished: bool = False


class RecursiveAdjacency:
    def __init__(self, points, config: AdjacencyConfig | None = None):
        raw = np.asarray(points)
        if raw.ndim != 2 or not len(raw) or not np.all(np.isfinite(raw)):
            raise ValueError("finite vertex matrix required")
        if not np.array_equal(raw, np.rint(raw)):
            raise ValueError("integer reduced coordinates required")
        self.points = raw.astype(np.int64)
        self.config = config or AdjacencyConfig()
        self.full_word = (1 << len(raw)) - 1
        self.frames: dict[int, FaceFrame] = {}
        self.certificates: dict[tuple[int, int], FacetCertificate] = {}
        self.rotation_cache: dict[tuple[int, int, int], tuple[int, dict]] = {}
        self.leaf_cache: dict[int, tuple[FacetCertificate, ...]] = {}
        self.stats: Counter = Counter()
        self.seconds: Counter = Counter()
        self.deadline = math.inf

    def check_deadline(self):
        if time.perf_counter() >= self.deadline:
            raise SamplingDeadline("adjacency time budget")

    def frame(self, word):
        if not word or word & ~self.full_word:
            raise GeometryUncertain("invalid face vertex word")
        if word in self.frames:
            return self.frames[word]
        started = time.perf_counter()
        indices = word_indices(word)
        points = self.points[list(indices)]
        differences = points[1:] - points[0]
        dimension = int(np.linalg.matrix_rank(differences.astype(float), tol=1e-8)) if len(differences) else 0
        if dimension:
            _, _, pivots = qr(differences.astype(float), pivoting=True, mode="economic")
            axes = tuple(sorted(map(int, pivots[:dimension])))
            if modular_rank(differences[:, axes]) != dimension:
                raise GeometryUncertain("intrinsic coordinate rank not certified")
        else:
            axes = ()
        augmented = np.column_stack((np.ones(len(indices), dtype=np.int64), points[:, axes]))
        result = FaceFrame(word, indices, axes, augmented, {v: i for i, v in enumerate(indices)})
        self.frames[word] = result
        self.seconds["coordinate_frames"] += time.perf_counter() - started
        return result

    def certify(self, parent_word, selected_word):
        key = parent_word, selected_word
        if key in self.certificates:
            return self.certificates[key]
        started = time.perf_counter()
        parent = self.frame(parent_word)
        if not selected_word or selected_word & ~parent_word or parent.dimension == 0:
            raise GeometryUncertain("facet is not a nonempty parent subset")
        selected = parent.augmented[[parent.positions[v] for v in word_indices(selected_word)]]
        if modular_rank(selected) != parent.dimension:
            raise GeometryUncertain("child is not a parent codimension-one face")
        _, _, vh = np.linalg.svd(selected.astype(float), full_matrices=True)
        try:
            row = _primitive_integer_row(vh[-1], parent.augmented)
        except (ValueError, OverflowError) as error:
            raise GeometryUncertain(str(error)) from error
        values = parent.augmented.astype(object) @ np.asarray(row, dtype=object)
        if any(int(v) < 0 for v in values):
            raise GeometryUncertain("facet cuts parent")
        closure = word_from_indices(parent.indices[i] for i, value in enumerate(values) if value == 0)
        if selected_word & ~closure:
            raise GeometryUncertain("integer normal lost selected vertices")
        result = FacetCertificate(parent_word, closure, row)
        self.certificates[key] = self.certificates[(parent_word, closure)] = result
        self.seconds["integer_certification"] += time.perf_counter() - started
        return result

    def sample_facet(self, parent_word, rng: random.Random):
        parent = self.frame(parent_word)
        for _ in range(5):
            self.check_deadline()
            objective = np.asarray([rng.gauss(0, 1) for _ in range(parent.dimension + 1)])
            started = time.perf_counter()
            result = linprog(
                objective, A_ub=-parent.augmented, b_ub=np.zeros(len(parent.indices)),
                A_eq=parent.augmented.sum(axis=0, keepdims=True), b_eq=[1.0],
                bounds=[(None, None)] * len(objective), method="highs",
                options={"time_limit": max(0.001, min(60.0, self.deadline - time.perf_counter()))},
            )
            self.seconds["descent_lp"] += time.perf_counter() - started
            self.stats["descent_lp_attempts"] += 1
            if not result.success:
                continue
            values = parent.augmented @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(values))))
            word = word_from_indices(parent.indices[i] for i in np.flatnonzero(np.abs(values) <= tolerance))
            try:
                return self.certify(parent_word, word)
            except GeometryUncertain:
                self.stats["uncertified_lp_faces"] += 1
        raise GeometryUncertain("no certified child facet")

    def enumerate_leaf(self, leaf_word):
        if leaf_word in self.leaf_cache:
            self.stats["leaf_cache_hits"] += 1
            return self.leaf_cache[leaf_word]
        self.check_deadline()
        leaf = self.frame(leaf_word)
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
                hull = ConvexHull(leaf.augmented[:, 1:].astype(float))
            except QhullError as error:
                raise GeometryUncertain("Qhull leaf enumeration failed") from error
            simplices = hull.simplices
        self.seconds["leaf_hull"] += time.perf_counter() - started
        if len(simplices) > self.config.max_leaf_simplices:
            raise GeometryUncertain("leaf simplex output limit")
        facets = {}
        for simplex in simplices:
            self.check_deadline()
            selected = word_from_indices(leaf.indices[int(i)] for i in simplex)
            matrix = leaf.augmented[list(map(int, simplex))]
            if modular_rank(matrix) != leaf.dimension:
                # Degenerate triangulations of nonsimplicial hulls can contain
                # zero-volume simplices; they do not define additional facets.
                self.stats["degenerate_leaf_simplices_skipped"] += 1
                continue
            certificate = self.certify(leaf_word, selected)
            facets[certificate.word] = certificate
            if len(facets) > self.config.max_leaf_facets:
                raise GeometryUncertain("leaf facet output limit")
        if len(facets) < leaf.dimension + 1:
            raise GeometryUncertain("too few leaf supporting facets")
        result = tuple(facets[word] for word in sorted(facets))
        self.leaf_cache[leaf_word] = result
        self.stats["leaf_enumerations_finished"] += 1
        self.stats["leaf_facets_enumerated"] += len(result)
        return result

    def rotate(self, parent_word, source_word, ridge_word):
        key = parent_word, source_word, ridge_word
        if key in self.rotation_cache:
            self.stats["rotation_cache_hits"] += 1
            return self.rotation_cache[key]
        self.check_deadline()
        started = time.perf_counter()
        parent = self.frame(parent_word)
        source = self.certify(parent_word, source_word)
        ridge = self.certify(source.word, ridge_word)
        if ridge.word != ridge_word:
            raise GeometryUncertain("ridge is not closed in source")
        matrix = parent.augmented.astype(float)
        selected = matrix[[parent.positions[v] for v in word_indices(ridge_word)]]
        _, singular, vh = np.linalg.svd(selected, full_matrices=True)
        rank = int(np.count_nonzero(singular > 1e-8))
        basis = vh[rank:].T
        if basis.shape[1] != 2:
            raise GeometryUncertain("ridge normal space is not two dimensional")
        normal = np.asarray(source.row, dtype=float)
        normal /= np.linalg.norm(normal)
        projected = basis - normal[:, None] * (normal @ basis)[None, :]
        direction = projected[:, int(np.argmax(np.linalg.norm(projected, axis=0)))]
        direction /= np.linalg.norm(direction)
        source_positions = [parent.positions[v] for v in word_indices(source.word)]
        values = matrix @ direction
        if float(np.sum(values[source_positions])) < 0:
            direction = -direction
            values = -values
        if np.min(values[source_positions]) < -1e-7:
            raise GeometryUncertain("rotation direction cuts source face")
        slacks = matrix @ normal
        outside = np.asarray([not (source.word >> v & 1) for v in parent.indices])
        if np.any(slacks[outside] <= 1e-10):
            raise GeometryUncertain("uncertain rotation denominator")
        parameter = float(np.max(-values[outside] / slacks[outside]))
        endpoint = values + parameter * slacks
        tolerance = 3e-7 * max(1.0, float(np.max(np.abs(endpoint))))
        selected_word = word_from_indices(parent.indices[i] for i in np.flatnonzero(np.abs(endpoint) <= tolerance))
        destination = self.certify(parent_word, selected_word)
        if destination.word == source.word or destination.word & source.word != ridge_word:
            raise GeometryUncertain("rotation does not share the certified ridge")
        proof = {"parent_word_hex": hex(parent_word), "source_word_hex": hex(source.word),
                 "ridge_word_hex": hex(ridge_word), "destination_word_hex": hex(destination.word)}
        result = destination.word, proof
        self.rotation_cache[key] = result
        self.stats["rotations_computed"] += 1
        self.seconds["rotation_inclusive"] += time.perf_counter() - started
        return result

    def sample(self, source_word, rng, *, deadline=math.inf):
        started = time.perf_counter()
        self.deadline = min(deadline, started + self.config.max_macro_seconds)
        batch = AdjacencyBatch(source_word)
        self.stats["macro_calls"] += 1
        try:
            self.check_deadline()
            source = self.certify(self.full_word, source_word)
            chain = [self.full_word, source.word]
            while chain[-1].bit_count() > self.config.cutoff_vertices:
                self.check_deadline()
                chain.append(self.sample_facet(chain[-1], rng).word)
            batch.chain = tuple(chain)
            leaves = self.enumerate_leaf(chain[-1])
            batch.leaf_enumeration_finished = True
            batch.leaf_facets = len(leaves)
            self.stats[f"chain_depth_{len(chain)-2}"] += 1
            self.stats[f"leaf_vertices_{chain[-1].bit_count()}"] += 1
            # Each leaf boundary gives an independent path of rotations back to P.
            for leaf in leaves:
                self.check_deadline()
                ridge, rotations = leaf.word, []
                for level in range(len(chain) - 2, -1, -1):
                    ridge, proof = self.rotate(chain[level], chain[level+1], ridge)
                    rotations.append(proof)
                top_ridge = source.word & ridge
                batch.outputs.append({"destination_word_hex": hex(ridge),
                                      "retained_word_hex": hex(top_ridge),
                                      "chain_word_hex": list(map(hex, chain)),
                                      "leaf_boundary_word_hex": hex(leaf.word),
                                      "rotations": rotations})
                if len(batch.outputs) >= self.config.max_outputs:
                    batch.status = "output_limit"
                    break
            else:
                batch.status = "chain_finished"
        except SamplingDeadline:
            batch.status = "time_limit"
        except GeometryUncertain as error:
            batch.status = "uncertified: " + str(error)
        finally:
            self.deadline = math.inf
            self.seconds["macro_inclusive"] += time.perf_counter() - started
        self.stats["outputs_generated"] += len(batch.outputs)
        self.stats["status_" + batch.status] += 1
        return batch

    def diagnostics(self):
        return {"statistics": dict(self.stats), "seconds": dict(self.seconds),
                "timing_scope": "inclusive nested timings; do not sum columns",
                "frames": len(self.frames), "rotation_cache": len(self.rotation_cache),
                "leaf_cache": len(self.leaf_cache),
                "leaf_backend": "SciPy Qhull (no joggling), then exact integer supporting-plane checks",
                "formal_enumeration_completeness_certificate": False,
                "finished_chain_implies_facet_adjacency_exhaustion": False}


def audit_adjacency_proof(points, proof):
    """Recheck the face lattice incidences; do not invoke the rotation algorithm."""
    geometry = RecursiveAdjacency(points)
    chain = tuple(int(word, 16) for word in proof["chain_word_hex"])
    if len(chain) < 2 or chain[0] != geometry.full_word:
        raise ValueError("invalid face-chain root")
    for parent, child in zip(chain, chain[1:]):
        if geometry.certify(parent, child).word != child:
            raise ValueError("nonclosed chain child")
    boundary = int(proof["leaf_boundary_word_hex"], 16)
    if geometry.certify(chain[-1], boundary).word != boundary:
        raise ValueError("nonclosed leaf boundary")
    rotations = proof["rotations"]
    if len(rotations) != len(chain) - 1:
        raise ValueError("incomplete upward rotation chain")
    for level, rotation in zip(range(len(chain)-2, -1, -1), rotations):
        parent = int(rotation["parent_word_hex"], 16)
        source = int(rotation["source_word_hex"], 16)
        ridge = int(rotation["ridge_word_hex"], 16)
        destination = int(rotation["destination_word_hex"], 16)
        if (parent, source, ridge) != (chain[level], chain[level+1], boundary):
            raise ValueError("rotation chain mismatch")
        if geometry.certify(source, ridge).word != ridge:
            raise ValueError("rotation ridge is not a source facet")
        if geometry.certify(parent, destination).word != destination:
            raise ValueError("rotation endpoint is not a closed supporting facet")
        if destination == source or source & destination != ridge:
            raise ValueError("rotation endpoints do not share the specified ridge")
        boundary = destination
    if boundary != int(proof["destination_word_hex"], 16):
        raise ValueError("final destination mismatch")
    if chain[1] & boundary != int(proof["retained_word_hex"], 16):
        raise ValueError("top retained ridge mismatch")
    return True
