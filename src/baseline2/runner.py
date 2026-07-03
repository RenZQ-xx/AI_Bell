from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from baseline2.context import build_context
from baseline2.core.candidate_family import CandidateFamily, CandidateFamilyCache, CandidateTransition
from baseline2.core.clean_symmetry import canonical_key, induced_block_map
from baseline2.core.expansion_facts import ExpansionFacts, build_expansion_facts
from baseline2.core.facet_class_detector import KnownFacetClassDetector
from baseline2.core.facet_terminal import FacetTerminalHit, FacetTerminalOracle
from baseline2.core.state_family import representation_keys
from baseline2.primitives.orbit_blocks import empty_key, state_group_words


OUT_DIR = Path(__file__).resolve().parent / "output"
SUPPORT_REPRESENTATION_TEXT_LIMIT = 1200


def default_output_paths(*, row_class: int, rep_index: int, pattern_index: int, seed: int) -> tuple[Path, Path]:
    stem = (
        f"class{int(row_class)}_rep{int(rep_index)}_pattern{int(pattern_index)}"
        f"_seed{int(seed)}_candidate_family"
    )
    return OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.md"


def selected_blocks(key: Sequence[int]) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(key) if int(value))


def build_ia_partner(blocks: Sequence[Sequence[int]]) -> tuple[int, ...]:
    block_by_vertices = {
        frozenset(int(vertex) for vertex in block): int(index)
        for index, block in enumerate(blocks)
    }
    ia_block_map = None
    for vertex_perm, word in state_group_words():
        if tuple(word) != ("i_A",):
            continue
        ia_block_map = induced_block_map(
            vertex_perm,
            orbits=blocks,
            block_by_vertices=block_by_vertices,
        )
        break
    partner = [-1] * len(blocks)
    if ia_block_map is None:
        return tuple(partner)
    for index, mapped in enumerate(ia_block_map):
        mapped = int(mapped)
        if int(index) < mapped and int(ia_block_map[mapped]) == int(index):
            partner[int(index)] = mapped
            partner[mapped] = int(index)
    return tuple(partner)


def materialized_support_representations(
    family: CandidateFamily,
    *,
    block_maps: Sequence[Sequence[int]],
    enabled: bool,
) -> tuple[tuple[int, ...], ...] | None:
    if not bool(enabled):
        return None
    if family.support_representations is not None:
        return family.support_representations
    return representation_keys(family.canonical_support, block_maps)


def support_repr_text(representations: Sequence[Sequence[int]] | None, *, limit: int) -> str:
    if int(limit) <= 0 or representations is None:
        return ""
    rows = [" ".join(str(value) for value in selected_blocks(key)) for key in representations]
    text = " | ".join(rows)
    if len(text) > int(limit):
        return f"{text[:int(limit)]}... [truncated; representations={len(rows)}]"
    return text


def family_centroid_metrics(
    family: CandidateFamily,
    *,
    blocks: Sequence[Sequence[int]],
    points: np.ndarray,
    zero_tol: float = 1e-12,
) -> dict[str, float]:
    vertices = [
        int(vertex)
        for block_index, selected in enumerate(family.canonical_support)
        if int(selected)
        for vertex in blocks[int(block_index)]
    ]
    if not vertices:
        centroid = np.zeros(points.shape[1], dtype=np.float64)
    else:
        centroid = np.asarray(points[vertices], dtype=np.float64).mean(axis=0)
    return {
        "family_centroid_norm": float(np.linalg.norm(centroid)),
        "zero_coordinate_fraction": float(np.mean(np.abs(centroid) <= float(zero_tol))),
    }


def ia_pollution_summary(
    key: Sequence[int],
    *,
    ia_partner: Sequence[int],
) -> tuple[bool, str]:
    pairs: set[tuple[int, int]] = set()
    selected = set(selected_blocks(key))
    for block_index in selected:
        partner = int(ia_partner[int(block_index)])
        if partner >= 0 and partner in selected:
            pairs.add(tuple(sorted((int(block_index), int(partner)))))
    ordered = tuple(sorted(pairs))
    return bool(ordered), "; ".join(f"{left} {right}" for left, right in ordered)


def cached_canonical_key(
    key: Sequence[int],
    *,
    block_maps: Sequence[Sequence[int]],
    cache: CandidateFamilyCache | None,
) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in key)
    if cache is not None and normalized in cache.canonical_key_cache:
        return cache.canonical_key_cache[normalized]
    out = canonical_key(normalized, block_maps)
    if cache is not None:
        cache.canonical_key_cache[normalized] = out
    return out


def random_tie_sort(
    rows: list[dict[str, object]],
    *,
    rng: random.Random,
    use_ia_pollution_ordering: bool,
) -> list[dict[str, object]]:
    decorated = [(rng.random(), row) for row in rows]
    decorated.sort(
        key=lambda item: (
            bool(item[1]["iA_polluted"]) if bool(use_ia_pollution_ordering) else False,
            not bool(item[1]["supportable"]),
            int(item[1]["flat_capacity"]),
            item[0],
        )
    )
    return [row for _tie, row in decorated]


def row_for_child(
    *,
    step: int,
    child: CandidateFamily,
    parent: CandidateFamily,
    transition: CandidateTransition,
    parent_index: int,
    parent_support: str,
    ia_partner: Sequence[int],
    blocks: Sequence[Sequence[int]],
    points: np.ndarray,
    block_maps: Sequence[Sequence[int]],
    materialize_support_representations: bool,
    support_representation_text_limit: int,
) -> dict[str, object]:
    supportability = transition.expansion_facts.supportability
    ia_polluted, ia_pairs = ia_pollution_summary(child.canonical_support, ia_partner=ia_partner)
    centroid = family_centroid_metrics(child, blocks=blocks, points=points)
    representations = materialized_support_representations(
        child,
        block_maps=block_maps,
        enabled=bool(materialize_support_representations),
    )
    return {
        "step": int(step),
        "parent_index": int(parent_index),
        "parent_canonical_support": parent_support,
        "canonical_action": int(transition.canonical_action),
        "action_orbit": list(int(value) for value in transition.action_orbit),
        "canonical_support": " ".join(str(value) for value in child.selected_blocks),
        "rank": int(child.rank),
        "rank_gain": int(transition.expansion_facts.rank_gain),
        "supportable": bool(child.supportable),
        "flat_capacity": int(child.flat_capacity),
        "iA_polluted": bool(ia_polluted),
        "iA_pollution_pairs": ia_pairs,
        "family_centroid_norm": centroid["family_centroid_norm"],
        "zero_coordinate_fraction": centroid["zero_coordinate_fraction"],
        "closer_side": None if supportability is None else float(supportability.closer_side),
        "support_shift": None if supportability is None else float(supportability.supporting_shift),
        "stabilizer_size": int(len(child.stabilizer_maps)),
        "support_representation_count": None if representations is None else int(len(representations)),
        "all_support_representations": support_repr_text(
            representations,
            limit=int(support_representation_text_limit),
        ),
    }


def row_for_terminal_child(
    *,
    step: int,
    canonical_support: Sequence[int],
    facts: ExpansionFacts,
    canonical_action: int,
    action_orbit: Sequence[int],
    parent_index: int,
    parent_support: str,
    terminal,
    ia_partner: Sequence[int],
) -> dict[str, object]:
    key = tuple(int(value) for value in canonical_support)
    ia_polluted, ia_pairs = ia_pollution_summary(key, ia_partner=ia_partner)
    return {
        "step": int(step),
        "parent_index": int(parent_index),
        "parent_canonical_support": parent_support,
        "canonical_action": int(canonical_action),
        "action_orbit": list(int(value) for value in action_orbit),
        "canonical_support": " ".join(str(value) for value in selected_blocks(key)),
        "rank": int(facts.new_rank),
        "rank_gain": int(facts.rank_gain),
        "flat_capacity": int(facts.flat_capacity),
        "iA_polluted": bool(ia_polluted),
        "iA_pollution_pairs": ia_pairs,
        "terminal_label": str(terminal.label),
        "terminal_is_exact": bool(terminal.is_exact),
        "facet_family_id": terminal.family_id if isinstance(terminal, FacetTerminalHit) else None,
    }


def child_key_from_row(row: dict[str, object], *, block_count: int) -> tuple[int, ...]:
    selected = {int(value) for value in str(row["canonical_support"]).split()}
    return tuple(1 if index in selected else 0 for index in range(block_count))


def run_search(
    *,
    cap: int,
    max_steps: int,
    seed: int,
    row_class: int = 7,
    rep_index: int = 1,
    pattern_index: int = 0,
    use_ia_pollution_ordering: bool = True,
    materialize_support_representations: bool = False,
    support_representation_text_limit: int = SUPPORT_REPRESENTATION_TEXT_LIMIT,
) -> dict[str, object]:
    run_started = time.perf_counter()
    context = build_context(
        row_class=int(row_class),
        rep_index=int(rep_index),
        pattern_index=int(pattern_index),
    )
    blocks = context.blocks
    block_maps = context.block_maps
    oracle = context.geometry
    ia_partner = build_ia_partner(blocks)
    terminal_oracle = FacetTerminalOracle.from_parts(
        points=context.points,
        blocks=blocks,
        config=context.config,
    )
    class_detector = KnownFacetClassDetector.build()
    rng = random.Random(int(seed))
    cache = CandidateFamilyCache()
    root = CandidateFamily.from_key(
        empty_key(len(blocks)),
        oracle=oracle,
        block_maps=block_maps,
        quotient_root=True,
        cache=cache,
        materialize_actions=True,
    )
    frontier = [root]
    steps: list[dict[str, object]] = []
    for step in range(int(max_steps)):
        step_started = time.perf_counter()
        rows_by_child: dict[tuple[int, ...], dict[str, object]] = {}
        child_by_key: dict[tuple[int, ...], CandidateFamily] = {}
        terminal_rows_by_child: dict[tuple[int, ...], dict[str, object]] = {}
        input_frontier = len(frontier)
        generated_orbits = 0
        rank_gain_orbits = 0
        admitted_orbits = 0
        rank_gain_filter_bypassed_orbits = 0
        for parent_index, parent in enumerate(frontier):
            if int(parent.rank) >= 25:
                continue
            parent = parent.materialize_action_structure(oracle=oracle, cache=cache)
            parent_support = " ".join(str(value) for value in parent.selected_blocks)
            rank_gain_filter_active = bool(parent.selected_blocks)
            for orbit, is_rank_gain in zip(parent.action_orbits, parent.action_is_rank_gain):
                generated_orbits += 1
                is_rank_gain = bool(is_rank_gain)
                if not rank_gain_filter_active:
                    rank_gain_filter_bypassed_orbits += 1
                elif not is_rank_gain:
                    continue
                admitted_orbits += 1
                if is_rank_gain:
                    rank_gain_orbits += 1
                orbit = tuple(sorted(int(action) for action in orbit))
                canonical_action = max(orbit)
                facts = build_expansion_facts(oracle, parent.canonical_support, int(canonical_action))
                key = cached_canonical_key(facts.key, block_maps=block_maps, cache=cache)
                if int(facts.new_rank) >= 25:
                    current_terminal = terminal_rows_by_child.get(key)
                    if current_terminal is not None:
                        current_terminal["dedup_parent_count"] = int(current_terminal.get("dedup_parent_count", 1)) + 1
                        continue
                    terminal = terminal_oracle.evaluate_key(key, action=int(canonical_action))
                    terminal_rows_by_child[key] = row_for_terminal_child(
                        step=step,
                        canonical_support=key,
                        facts=facts,
                        canonical_action=int(canonical_action),
                        action_orbit=orbit,
                        parent_index=parent_index,
                        parent_support=parent_support,
                        terminal=terminal,
                        ia_partner=ia_partner,
                    )
                    continue
                child = CandidateFamily.from_key(
                    facts.key,
                    oracle=oracle,
                    block_maps=block_maps,
                    quotient_root=True,
                    cache=cache,
                    materialize_actions=False,
                )
                transition = CandidateTransition(
                    parent=parent,
                    action_orbit=orbit,
                    canonical_action=int(canonical_action),
                    expansion_facts=facts,
                    child=child,
                )
                row = row_for_child(
                    step=step,
                    child=child,
                    parent=parent,
                    transition=transition,
                    parent_index=parent_index,
                    parent_support=parent_support,
                    ia_partner=ia_partner,
                    blocks=blocks,
                    points=context.points,
                    block_maps=block_maps,
                    materialize_support_representations=bool(materialize_support_representations),
                    support_representation_text_limit=int(support_representation_text_limit),
                )
                existing = rows_by_child.get(key)
                if existing is None:
                    rows_by_child[key] = row
                    child_by_key[key] = child
                else:
                    existing["dedup_parent_count"] = int(existing.get("dedup_parent_count", 1)) + 1
        all_rows = list(rows_by_child.values())
        for row in all_rows:
            row.setdefault("dedup_parent_count", 1)
        terminal_rows = list(terminal_rows_by_child.values())
        for row in terminal_rows:
            row.setdefault("dedup_parent_count", 1)
        all_rows = random_tie_sort(
            all_rows,
            rng=rng,
            use_ia_pollution_ordering=bool(use_ia_pollution_ordering),
        )
        selected_rows = all_rows[: int(cap)]
        for rank, row in enumerate(all_rows, start=1):
            row["global_rank"] = int(rank)
            row["selected_for_next_frontier"] = rank <= int(cap)
        selected_child_keys = [
            child_key_from_row(row, block_count=len(blocks))
            for row in selected_rows
        ]
        frontier = [
            child_by_key[key].materialize_action_structure(oracle=oracle, cache=cache)
            for key in selected_child_keys
            if key in child_by_key
        ]
        terminal_exact = sum(1 for row in terminal_rows if bool(row["terminal_is_exact"]))
        terminal_invalid = len(terminal_rows) - terminal_exact
        rank_hist = Counter(int(row["rank"]) for row in selected_rows)
        step_elapsed = time.perf_counter() - step_started
        print(
            "step {step}: parents={parents} admitted_orbits={admitted} rank_gain_orbits={rank_gain} "
            "unique_children={children} terminals={terminals} exact={exact} "
            "invalid={invalid} selected={selected} elapsed={elapsed:.3f}s".format(
                step=step,
                parents=input_frontier,
                admitted=admitted_orbits,
                rank_gain=rank_gain_orbits,
                children=len(all_rows),
                terminals=len(terminal_rows),
                exact=terminal_exact,
                invalid=terminal_invalid,
                selected=len(selected_rows),
                elapsed=step_elapsed,
            ),
            flush=True,
        )
        steps.append(
            {
                "summary": {
                    "step": int(step),
                    "input_frontier": int(input_frontier),
                    "generated_action_orbits": int(generated_orbits),
                    "admitted_action_orbits": int(admitted_orbits),
                    "rank_gain_action_orbits": int(rank_gain_orbits),
                    "rank_gain_filter_bypassed_orbits": int(rank_gain_filter_bypassed_orbits),
                    "unique_children": int(len(all_rows)),
                    "unique_terminal_children": int(len(terminal_rows)),
                    "terminal_exact_children": int(terminal_exact),
                    "terminal_invalid_children": int(terminal_invalid),
                    "output_frontier": int(len(frontier)),
                    "selected_rank_histogram": dict(sorted(rank_hist.items())),
                    "elapsed_seconds": float(step_elapsed),
                },
                "selected": selected_rows,
                "all_children": all_rows,
                "terminals": terminal_rows,
            }
        )
        if not frontier or (not all_rows and not terminal_rows):
            break
    search_elapsed = time.perf_counter() - run_started
    facet_class_audit = class_detector.audit_registry(terminal_oracle.registry)
    total_elapsed = time.perf_counter() - run_started
    return {
        "meta": {
            "search_lane": "candidate_family",
            "search_loop": "baseline2_candidate_family",
            "ordering": [
                *(
                    ["iA_polluted=False"]
                    if bool(use_ia_pollution_ordering)
                    else ["iA_polluted recorded only; not used for ordering"]
                ),
                "supportable=True",
                "flat_capacity ascending",
                "random tie-break",
            ],
            "cap": int(cap),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "row_class": int(row_class),
            "rep_index": int(rep_index),
            "pattern_index": int(pattern_index),
            "use_ia_pollution_ordering": bool(use_ia_pollution_ordering),
            "materialize_support_representations": bool(materialize_support_representations),
            "support_representation_text_limit": int(support_representation_text_limit),
            "runtime_seconds": float(total_elapsed),
            "search_runtime_seconds": float(search_elapsed),
            "class_audit_runtime_seconds": float(total_elapsed - search_elapsed),
        },
        "steps": steps,
        "final_frontier": [
            {
                "canonical_support": " ".join(str(value) for value in family.selected_blocks),
                "rank": int(family.rank),
                "supportable": bool(family.supportable),
                "flat_capacity": int(family.flat_capacity),
            }
            for family in frontier
        ],
        "facet_families": terminal_oracle.registry.to_dict(),
        "external_facet_class_audit": facet_class_audit,
    }


def write_report(result: dict[str, object]) -> str:
    lines = ["# baseline2 CandidateFamily Search", ""]
    meta = result["meta"]
    lines.extend(
        [
            f"- pattern: class {meta['row_class']}, rep {meta['rep_index']}, pattern {meta['pattern_index']}",
            f"- seed: {meta['seed']}",
            f"- runtime_seconds: {meta['runtime_seconds']:.6f}",
            "",
        ]
    )
    lines.append("| step | parents | admitted orbits | rank-gain orbits | children | terminals | exact | invalid | selected |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step in result["steps"]:
        summary = step["summary"]
        lines.append(
            "| {step} | {input_frontier} | {admitted_action_orbits} | {rank_gain_action_orbits} | {unique_children} | {unique_terminal_children} | {terminal_exact_children} | {terminal_invalid_children} | {output_frontier} |".format(
                **summary
            )
        )
    audit = result.get("external_facet_class_audit", {})
    lines.extend(
        [
            "",
            f"- exact facet families: {audit.get('family_count', 0)}",
            f"- detected class family counts: {json.dumps(audit.get('class_family_counts', {}), ensure_ascii=False)}",
            f"- final frontier size: {len(result.get('final_frontier', []))}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean CandidateFamily baseline2 demo.")
    parser.add_argument("--cap", type=int, default=108)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--row-class", type=int, default=7)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument(
        "--support-representation-text-limit",
        type=int,
        default=SUPPORT_REPRESENTATION_TEXT_LIMIT,
        help="Maximum characters for all_support_representations when support representation diagnostics are enabled.",
    )
    parser.add_argument(
        "--materialize-support-representations",
        action="store_true",
        help="Materialize full support representations for diagnostic output.",
    )
    parser.add_argument(
        "--disable-ia-pollution-ordering",
        action="store_true",
        help="Record iA pollution diagnostics but do not use them in online ordering.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_search(
        cap=int(args.cap),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        row_class=int(args.row_class),
        rep_index=int(args.rep_index),
        pattern_index=int(args.pattern_index),
        use_ia_pollution_ordering=not bool(args.disable_ia_pollution_ordering),
        materialize_support_representations=bool(args.materialize_support_representations),
        support_representation_text_limit=int(args.support_representation_text_limit),
    )
    default_json, default_report = default_output_paths(
        row_class=int(args.row_class),
        rep_index=int(args.rep_index),
        pattern_index=int(args.pattern_index),
        seed=int(args.seed),
    )
    output_json = args.output_json or default_json
    report_md = args.report_md or default_report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    report_md.write_text(write_report(result), encoding="utf-8")
    print(output_json)
    print(report_md)


if __name__ == "__main__":
    main()
