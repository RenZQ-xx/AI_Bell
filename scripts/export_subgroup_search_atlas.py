from __future__ import annotations

"""Export the facet-free Bell 322 subgroup atlas fields used by MCTS."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT.parent
    / "322_"
    / "data"
    / "generated"
    / "322"
    / "subgroup_atlas.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "subgroup_search_atlas_322.json"


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(second[index] for index in first)


def _closure(generators: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    identity = tuple(range(64))
    seen = {identity}
    stack = [identity]
    while stack:
        current = stack.pop()
        for generator in generators:
            candidate = _compose(current, generator)
            if candidate not in seen:
                seen.add(candidate)
                stack.append(candidate)
    return tuple(sorted(seen))


def _orbits(subgroup: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    unused = set(range(64))
    blocks: list[tuple[int, ...]] = []
    while unused:
        seed = min(unused)
        block = tuple(sorted({permutation[seed] for permutation in subgroup}))
        unused.difference_update(block)
        blocks.append(block)
    return tuple(sorted(blocks))


def _canonical_partition(
    blocks: tuple[tuple[int, ...], ...],
    ambient: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    best: tuple[tuple[int, ...], ...] | None = None
    for permutation in ambient:
        moved = tuple(
            sorted(
                tuple(sorted(permutation[vertex] for vertex in block))
                for block in blocks
            )
        )
        if best is None or moved < best:
            best = moved
    if best is None:
        raise ValueError("ambient group must not be empty")
    return best


def _compact_generator(generator: dict[str, object]) -> dict[str, object]:
    return {
        "word": list(generator.get("word", [])),
        "str": str(generator.get("str") or "1"),
        "vertex_permutation": list(generator["vertex_permutation"]),
    }


def _compact_node(
    node: dict[str, object],
    ambient: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    features = node["features"]
    if not isinstance(features, dict):
        raise ValueError("node features must be an object")
    generators = [
        tuple(int(value) for value in generator["vertex_permutation"])
        for generator in node["generators"]
    ]
    subgroup = _closure(generators)
    canonical_partition = _canonical_partition(_orbits(subgroup), ambient)
    return {
        "id": node["id"],
        "level": node["level"],
        "structure": node["structure"],
        "generators": [
            _compact_generator(generator)
            for generator in node["generators"]
        ],
        "features": {
            "order": features["order"],
            "semantic_family": features["semantic_family"],
            "vertex_orbit_sizes": features["vertex_orbit_sizes"],
            "projected_point_count": features["projected_point_count"],
            "fixed_polytope_dimension": features["fixed_polytope_dimension"],
        },
        "reduction_scale": node["reduction_scale"],
        "source_count": node["source_count"],
        "parent_root_ids": list(node.get("parent_root_ids", [])),
        "macro_family": node["macro_family"],
        "action_family": node["action_family"],
        "fine_type": node["fine_type"],
        "canonical_partition": [list(block) for block in canonical_partition],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    root_generators = [
        tuple(int(value) for value in generator["vertex_permutation"])
        for node in payload["nodes"]
        if int(node["level"]) == 1
        for generator in node["generators"]
    ]
    ambient = _closure(root_generators)
    if len(ambient) != 3072:
        raise RuntimeError(f"expected ambient order 3072, got {len(ambient)}")
    compact = {
        "schema_version": 1,
        "scenario": "322",
        "source": {
            "artifact": "322_/data/generated/322/subgroup_atlas.json",
            "schema_version": payload.get("schema_version"),
            "created_at": payload.get("created_at"),
            "facet_answers_used": False,
            "description": "Two-level minimal-overgroup atlas derived only from the Bell 322 relabeling action.",
        },
        "counts": payload["counts"],
        "nodes": [_compact_node(node, ambient) for node in payload["nodes"]],
        "edges": [
            {
                "id": edge["id"],
                "parent_id": edge["parent_id"],
                "child_id": edge["child_id"],
                "relation": edge["relation"],
                "extension_str": edge.get("extension_str", ""),
            }
            for edge in payload["edges"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(compact['nodes'])} nodes and {len(compact['edges'])} edges "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
