from __future__ import annotations

"""Exact post-run relabeling stabilizers of catalog-unmatched Bell facets."""

import argparse
import json
import math
import time
from collections import Counter
from itertools import permutations, product
from pathlib import Path

import numpy as np

from .catalog_reproduction import save_json
from .problem import (
    CORRELATOR_BASIS, GROUP_ORDER, Relabeling, deterministic_vertices,
    permutation_closure, signed_coordinate_action,
)


def integer_row(row):
    if len(row) != 81 or any(not isinstance(v, (int, np.integer)) for v in row):
        raise ValueError("expected 81 exact integer coefficients")
    if sum(abs(int(v)) for v in row) > np.iinfo(np.int64).max:
        raise ValueError("row exceeds safe exact int64 slack arithmetic")
    return np.asarray(row, dtype=np.int64)


class RelabelingStabilizers:
    def __init__(self):
        self.unsigned = [
            (tuple(p), tuple(bits))
            for p in permutations(range(4)) for bits in product((0, 1), repeat=4)
        ]
        self.targets = np.asarray([
            [target for target, _ in signed_coordinate_action(Relabeling(
                party_permutation=p,
                setting_permutations=tuple((1, 0) if bit else (0, 1) for bit in bits),
            ))] for p, bits in self.unsigned
        ], dtype=np.int16)
        masks = np.asarray([sum(1 << (2*p+s) for p, s in term)
                            for term in CORRELATOR_BASIS], dtype=np.uint16)
        self.signs = np.asarray([
            [(-1 if (flip & int(mask)).bit_count() % 2 else 1) for mask in masks]
            for flip in range(256)
        ], dtype=np.int8)
        self.solution_cache = {}

    def elements(self, row):
        coefficients = integer_row(row)[1:]
        columns = tuple(map(int, np.flatnonzero(coefficients)))
        if not columns:
            raise ValueError("constant row is not a Bell facet")
        lookup = self.solution_cache.get(columns)
        if lookup is None:
            lookup = {}
            packed = np.packbits(self.signs[:, columns] < 0, axis=1, bitorder="little")
            for flip, values in enumerate(packed):
                lookup.setdefault(values.tobytes(), []).append(flip)
            if len(self.solution_cache) >= 128:
                self.solution_cache.clear()
            self.solution_cache[columns] = lookup
        moved = coefficients[self.targets]
        compatible = np.flatnonzero(np.all(np.abs(moved) == np.abs(coefficients), axis=1))
        result = []
        for unsigned in compatible:
            negative = moved[unsigned, columns] != coefficients[list(columns)]
            key = np.packbits(negative, bitorder="little").tobytes()
            result.extend((int(unsigned), flip) for flip in lookup.get(key, ()))
        if not result or GROUP_ORDER % len(result):
            raise ValueError("invalid stabilizer order")
        return tuple(result)

    def observable_permutation(self, element):
        unsigned, flip = element
        parties, swaps = self.unsigned[unsigned]
        image = []
        for signed_index in range(16):
            index = signed_index % 8
            party, setting = divmod(index, 2)
            target = 2 * parties[party] + (setting ^ swaps[party])
            negative = (signed_index >= 8) ^ bool(flip & (1 << index))
            image.append(target + 8 * int(negative))
        return tuple(image)

    def describe(self, row, elements):
        parties = {self.unsigned[u][0] for u, _ in elements}
        orbits = []
        remaining = set(range(4))
        while remaining:
            orbit = {p[min(remaining)] for p in parties}
            remaining.difference_update(orbit)
            orbits.append("".join("ABCD"[v] for v in sorted(orbit)))
        pure_party = [u for u, flip in elements if not flip and not any(self.unsigned[u][1])]
        pure_input = [u for u, flip in elements if not flip and self.unsigned[u][0] == (0, 1, 2, 3)]
        pure_output = [flip for u, flip in elements if u == 0]
        local_only = [e for e in elements if self.unsigned[e[0]][0] == (0, 1, 2, 3)]
        coefficients = integer_row(row)[1:]
        return {
            "stabilizer_order": len(elements), "equivalence_orbit_size": GROUP_ORDER // len(elements),
            "pure_party_order": len(pure_party), "pure_input_order": len(pure_input),
            "pure_output_order": len(pure_output), "local_relabeling_kernel_order": len(local_only),
            "party_projection_order": len(parties), "party_orbits": orbits,
            "party_orbit_sizes": sorted(map(len, orbits), reverse=True),
            "nonzero_term_count": int(np.count_nonzero(coefficients)),
            "terms_by_body_order": dict(Counter(len(term) for term, c in zip(CORRELATOR_BASIS, coefficients) if c)),
            "largest_absolute_coefficient": int(np.max(np.abs(coefficients))),
            "has_output_flip_symmetry": any(flip for _, flip in elements),
            "has_mixed_flip_symmetry": any(flip and u for u, flip in elements),
            "stabilizer_elements": [list(e) for e in elements],
        }


def modular_rank(matrix, prime=1_000_003):
    values = np.asarray(matrix, dtype=np.int64).copy() % prime
    rank = 0
    for column in range(values.shape[1]):
        candidates = np.flatnonzero(values[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        values[[rank, pivot]] = values[[pivot, rank]]
        values[rank] = values[rank] * pow(int(values[rank, column]), -1, prime) % prime
        factors = values[rank+1:, column].copy()
        values[rank+1:] = (values[rank+1:] - factors[:, None] * values[rank]) % prime
        rank += 1
        if rank == len(values):
            break
    return rank


def validate_facet(row):
    row = integer_row(row)
    vertices = deterministic_vertices().astype(np.int64)
    slack = row[0] + vertices @ row[1:]
    tight = vertices[slack == 0]
    if np.min(slack) < 0 or not len(tight):
        raise ValueError("inequality does not support the deterministic polytope")
    rank = modular_rank(tight[1:] - tight[0])
    if rank != 79:
        raise ValueError("tight support is not a rank-79 facet")
    if math.gcd(*map(int, row)) != 1:
        raise ValueError("row is not primitive")
    return {"vertex_count_checked": 256, "minimum_slack": int(np.min(slack)),
            "maximum_slack": int(np.max(slack)), "tight_vertex_count": len(tight),
            "affine_rank_mod_prime": rank, "prime": 1_000_003,
            "exact_rank_proof": "modular lower bound 79 plus the nonzero exact supporting normal upper bound 79"}


def permutation_order(image):
    remaining = set(range(len(image)))
    order = 1
    while remaining:
        start = min(remaining)
        current, cycle = start, []
        while current not in cycle:
            cycle.append(current)
            remaining.remove(current)
            current = image[current]
        order = math.lcm(order, len(cycle))
    return order


def correlator_label(term):
    return "".join(f"{'ABCD'[p]}_{s}" for p, s in term) or "1"


def coefficient_matrix(row):
    local = ((), ((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),),
             ((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, 0)), ((0, 1), (1, 1)))
    right = tuple(tuple((p+2, s) for p, s in term) for term in local)
    lookup = {term: int(value) for term, value in zip(CORRELATOR_BASIS, row[1:])}
    matrix = [[int(row[0]) if not (a or b) else lookup[a+b] for b in right] for a in local]
    return {"left": [correlator_label(t) for t in local],
            "right": [correlator_label(t) for t in right], "matrix": matrix,
            "orientation": "expectation(left^T matrix right) >= 0; constant included"}


def representative(entry, group):
    elements = [tuple(e) for e in entry["stabilizer_elements"]]
    identity = tuple(range(16))
    images = {group.observable_permutation(e): e for e in elements}
    candidates = sorted(images, key=lambda image: (
        sum(v != i for i, v in enumerate(image)), permutation_order(image), image))
    generated, generators = {identity}, []
    for image in candidates:
        if image in generated:
            continue
        generators.append(image)
        generated = set(permutation_closure(generators, max_order=len(elements)) or ())
        if generated == set(images):
            break
    if generated != set(images):
        raise ValueError("generator closure does not equal the complete stabilizer")
    array = np.asarray(sorted(generated), dtype=np.int16)
    central = np.ones(len(array), dtype=bool)
    for generator in generators:
        generator = np.asarray(generator, dtype=np.int16)
        central &= np.all(array[:, generator] == generator[array], axis=1)
    center_order = int(np.count_nonzero(central))
    abelian = center_order == len(elements)
    order_counts = Counter(permutation_order(image) for image in generated)
    abstract = f"order {len(elements)}"
    if len(elements) == 1:
        abstract = "trivial"
    elif abelian and set(order_counts) <= {1, 2}:
        abstract = f"C2^{len(elements).bit_length()-1}"
    elif abelian and max(order_counts) == len(elements):
        abstract = f"C{len(elements)}"
    elif len(elements) == 6 and not abelian:
        abstract = "S3"
    details = []
    for image in generators:
        u, flip = images[image]
        p, swaps = group.unsigned[u]
        details.append({"party_permutation": list(p), "input_swaps": list(swaps),
                        "output_flip_mask": flip, "order": permutation_order(image),
                        "observable_images": [
                            ("-" if image[j] >= 8 else "") + f"{'ABCD'[(image[j]%8)//2]}_{image[j]%2}"
                            for j in range(8)], "signed_permutation": list(image)})
    row = entry["canonical_h_row"]
    return {**entry, "abstract_group_when_proved": abstract,
            "is_abelian": abelian, "center_order": center_order,
            "element_order_histogram": dict(sorted(order_counts.items())),
            "generators": details, "generator_closure_verified": True,
            "facet_validation": validate_facet(row), "coefficient_matrix": coefficient_matrix(row),
            "bell_upper_bound": row[0],
            "bell_terms": [{"coefficient": -int(c), "correlator": correlator_label(t)}
                           for t, c in zip(CORRELATOR_BASIS, row[1:]) if c]}


def select_representatives(entries):
    selected = {}
    def choose(category, candidates, key):
        if candidates:
            entry = min(candidates, key=key)
            selected.setdefault(entry["class_id"], {"entry": entry, "categories": []})["categories"].append(category)
    choose("largest_stabilizer", entries, lambda e: (-e["stabilizer_order"], e["nonzero_term_count"], e["class_id"]))
    choose("pure_party_symmetry", [e for e in entries if e["pure_party_order"] > 1],
           lambda e: (-e["pure_party_order"], e["nonzero_term_count"], e["largest_absolute_coefficient"]))
    choose("mixed_relabeling", [e for e in entries if e["pure_party_order"] == 1 and e["has_mixed_flip_symmetry"]],
           lambda e: (e["nonzero_term_count"], -e["stabilizer_order"], e["largest_absolute_coefficient"]))
    choose("asymmetric_sparse", [e for e in entries if e["stabilizer_order"] == 1],
           lambda e: (e["nonzero_term_count"], e["largest_absolute_coefficient"], e["class_id"]))
    return list(selected.values())


def analyze(run_path, output):
    started = time.perf_counter()
    print("Loading saved run; no search will be started.", flush=True)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    events = run["coverage"]["discoveries"]
    selected = [{key: event.get(key) for key in ("class_id", "catalog_index", "iteration", "canonical_h_row")}
                for event in events if event.get("catalog_index") is None]
    runtime = run["runtime"]
    total_count = len(events)
    del events, run
    group = RelabelingStabilizers()
    entries = []
    vertices = deterministic_vertices().astype(np.int64)
    for index, event in enumerate(selected, 1):
        row = integer_row(event["canonical_h_row"])
        if np.min(row[0] + vertices @ row[1:]) < 0:
            raise ValueError(f"invalid canonical inequality: {event['class_id']}")
        elements = group.elements(row)
        entries.append({**event, **group.describe(row, elements)})
        if index % 1000 == 0:
            print(f"Stabilizers checked: {index}/{len(selected)}", flush=True)
    if len({tuple(e["canonical_h_row"]) for e in entries}) != len(entries):
        raise ValueError("duplicate canonical H-rows")
    representatives = [{"selection_categories": value["categories"], **representative(value["entry"], group)}
                       for value in select_representatives(entries)]
    result = {
        "run": str(run_path.resolve()), "runtime": runtime, "total_classes": total_count,
        "catalog_unmatched_count": len(entries), "catalog_matched_count": total_count-len(entries),
        "scope": "exact stabilizers of saved canonical rows; catalog is partial; unmatched does not prove literature novelty",
        "group_order": GROUP_ORDER,
        "group": "(C2^2 semidirect S2)^4 semidirect S4",
        "coordinate_convention": "80 correlators of distinct parties, outcomes +/-1; b + sum(c_I E_I) >= 0",
        "element_encoding": "[unsigned_index, flip_mask]; bit 2*party+input flips the source output before permutation",
        "unsigned_relabelings": [{"party_permutation": list(p), "input_swaps": list(bits)} for p, bits in group.unsigned],
        "stabilizer_order_histogram": dict(sorted(Counter(e["stabilizer_order"] for e in entries).items())),
        "party_projection_order_histogram": dict(sorted(Counter(e["party_projection_order"] for e in entries).items())),
        "party_orbit_size_histogram": dict(Counter("+".join(map(str, e["party_orbit_sizes"])) for e in entries)),
        "pure_party_order_histogram_in_canonical_gauge": dict(sorted(Counter(e["pure_party_order"] for e in entries).items())),
        "pure_output_order_histogram": dict(sorted(Counter(e["pure_output_order"] for e in entries).items())),
        "classes_with_nontrivial_stabilizer": sum(e["stabilizer_order"] > 1 for e in entries),
        "classes_with_pure_party_symmetry_in_canonical_gauge": sum(e["pure_party_order"] > 1 for e in entries),
        "classes_with_party_exchange_after_local_relabeling": sum(e["party_projection_order"] > 1 for e in entries),
        "classes_with_mixed_flip_symmetry": sum(e["has_mixed_flip_symmetry"] for e in entries),
        "classes_with_pure_output_flip_symmetry": sum(e["pure_output_order"] > 1 for e in entries),
        "all_256_vertex_slack_checks_passed": True,
        "canonical_gauge_warning": "pure party/input subgroups depend on the chosen labels; full stabilizer order and party projection orbit sizes do not",
        "representatives": representatives, "classes": entries,
        "analysis_seconds": time.perf_counter()-started,
    }
    save_json(result, output)
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("unsigned_relabelings", "representatives", "classes")}, indent=2), flush=True)
    print("Representatives: " + ", ".join(e["class_id"] for e in representatives), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    analyze(args.run, args.output)


if __name__ == "__main__":
    main()
