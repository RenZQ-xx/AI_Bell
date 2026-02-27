from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from classify_facets import _parse_hrep_rows, classify_facets


GROUPS = {
    "A": [0, 1],
    "B": [2, 3],
    "C": [4, 5],
    "AB": [6, 7, 8, 9],
    "AC": [10, 11, 12, 13],
    "BC": [14, 15, 16, 17],
    "ABC": [18, 19, 20, 21, 22, 23, 24, 25],
}


def _bucket_l1(v: int) -> int:
    if v == 0:
        return 0
    if v <= 2:
        return 1
    if v <= 4:
        return 2
    if v <= 8:
        return 3
    return 4


def feature_tokens(row: tuple[int, ...]) -> frozenset[tuple]:
    b = row[0]
    c = row[1:]
    tokens: set[tuple] = set()

    tokens.add(("b_abs", abs(b)))
    tokens.add(("b_sign", 0 if b == 0 else (1 if b > 0 else -1)))

    nz_total = sum(1 for x in c if x != 0)
    tokens.add(("nz_total", nz_total))

    abs_hist = [0, 0, 0, 0, 0]  # |c|==1,2,3,4,>=5
    for x in c:
        ax = abs(x)
        if ax == 0:
            continue
        if ax >= 5:
            abs_hist[4] += 1
        else:
            abs_hist[ax - 1] += 1
    tokens.add(("abs_hist", tuple(abs_hist)))

    for gname, idxs in GROUPS.items():
        vals = [c[i] for i in idxs]
        nz = sum(1 for v in vals if v != 0)
        l1 = sum(abs(v) for v in vals)
        pos = sum(1 for v in vals if v > 0)
        neg = sum(1 for v in vals if v < 0)
        max_abs = max((abs(v) for v in vals), default=0)

        tokens.add((gname, "nz", nz))
        tokens.add((gname, "l1_bin", _bucket_l1(l1)))
        tokens.add((gname, "bal", pos - neg))
        tokens.add((gname, "max_abs", min(max_abs, 5)))

    triple = c[18:26]
    parity = tuple(1 if (abs(x) % 2 == 1) else 0 for x in triple)
    tokens.add(("ABC_parity", parity))

    return frozenset(tokens)


def build_class_lookup(rows: list[tuple[int, ...]]):
    classes, _ = classify_facets(rows, "322")
    # classes is sorted by size desc
    row_to_class: dict[tuple[int, ...], int] = {}
    class_sizes: dict[int, int] = {}
    for cid, cls in enumerate(classes, 1):
        class_sizes[cid] = cls["size"]
        for r in cls["members"]:
            row_to_class[r] = cid
    return row_to_class, class_sizes, len(classes)


def run_heuristic(
    rows: list[tuple[int, ...]],
    row_to_class: dict[tuple[int, ...], int],
    total_classes: int,
    seed: int = 0,
    max_steps: int | None = None,
):
    rnd = random.Random(seed)
    n = len(rows)

    tokens_list = [feature_tokens(r) for r in rows]
    token_freq: dict[tuple, int] = defaultdict(int)
    token_to_rows: dict[tuple, list[int]] = defaultdict(list)
    for i, toks in enumerate(tokens_list):
        for t in toks:
            token_freq[t] += 1
            token_to_rows[t].append(i)

    token_weight = {t: 1.0 / f for t, f in token_freq.items()}
    scores = [sum(token_weight[t] for t in toks) for toks in tokens_list]

    remaining = [True] * n
    remaining_count = n
    seen_tokens: set[tuple] = set()
    seen_classes: set[int] = set()
    discovered_log = []

    steps = 0
    while remaining_count > 0 and len(seen_classes) < total_classes:
        if max_steps is not None and steps >= max_steps:
            break

        # epsilon-greedy to avoid getting stuck in a small feature region
        if rnd.random() < 0.05:
            idx = rnd.randrange(n)
            while not remaining[idx]:
                idx = rnd.randrange(n)
        else:
            best_idx = -1
            best_score = -1.0
            for i in range(n):
                if not remaining[i]:
                    continue
                s = scores[i]
                if s > best_score:
                    best_score = s
                    best_idx = i
            idx = best_idx

        remaining[idx] = False
        remaining_count -= 1
        steps += 1

        row = rows[idx]
        cid = row_to_class[row]
        is_new_class = cid not in seen_classes
        seen_classes.add(cid)

        if is_new_class:
            discovered_log.append((steps, len(seen_classes), cid, row))

        new_tokens = tokens_list[idx] - seen_tokens
        if new_tokens:
            seen_tokens.update(new_tokens)
            for t in new_tokens:
                w = token_weight[t]
                for j in token_to_rows[t]:
                    if remaining[j]:
                        scores[j] -= w

    return {
        "steps": steps,
        "classes_found": len(seen_classes),
        "discovered_log": discovered_log,
    }


def run_random_baseline(
    rows: list[tuple[int, ...]],
    row_to_class: dict[tuple[int, ...], int],
    total_classes: int,
    trials: int,
    seed: int,
):
    rnd = random.Random(seed)
    n = len(rows)
    stats = []
    for _ in range(trials):
        idxs = list(range(n))
        rnd.shuffle(idxs)
        seen = set()
        step = 0
        for i in idxs:
            step += 1
            seen.add(row_to_class[rows[i]])
            if len(seen) == total_classes:
                break
        stats.append(step)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Regression on 322 facets: heuristic class discovery vs random."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to facets file. Default: data/facets_322.txt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save a text report.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    facets_path = args.input or (project_root / "data" / "facets_322.txt")
    rows = _parse_hrep_rows(facets_path)

    row_to_class, class_sizes, total_classes = build_class_lookup(rows)

    heuristic = run_heuristic(
        rows=rows,
        row_to_class=row_to_class,
        total_classes=total_classes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    random_stats = run_random_baseline(
        rows=rows,
        row_to_class=row_to_class,
        total_classes=total_classes,
        trials=args.random_trials,
        seed=args.seed + 1000,
    )

    lines = []
    lines.append(f"input: {facets_path}")
    lines.append(f"facets: {len(rows)}")
    lines.append(f"total classes: {total_classes}")
    lines.append("")
    lines.append("[Heuristic]")
    lines.append(
        f"steps_to_{total_classes}_classes: {heuristic['steps']} "
        f"(found={heuristic['classes_found']})"
    )
    lines.append("discovery_log(step, classes_found, class_id, class_size):")
    for step, ncls, cid, _row in heuristic["discovered_log"]:
        lines.append(f"{step}\t{ncls}\t{cid}\t{class_sizes[cid]}")

    lines.append("")
    lines.append("[Random Baseline]")
    lines.append(f"trials: {args.random_trials}")
    lines.append(f"mean_steps: {statistics.mean(random_stats):.2f}")
    lines.append(f"median_steps: {statistics.median(random_stats):.2f}")
    lines.append(f"min_steps: {min(random_stats)}")
    lines.append(f"max_steps: {max(random_stats)}")

    report = "\n".join(lines)
    print(report)

    if args.save_report is not None:
        out = args.save_report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        print(f"\nsaved report: {out}")


if __name__ == "__main__":
    main()
