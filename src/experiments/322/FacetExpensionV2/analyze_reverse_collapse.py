from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reverse-collapse from full non-44 seed pool toward class 44.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--target-classes", type=str, default="23,29,43")
    parser.add_argument("--per-class-limit", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_target_classes(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def jaccard(left: list[int], right: list[int]) -> float:
    inter = 0
    union = 0
    for a, b in zip(left, right):
        if a or b:
            union += 1
            if a and b:
                inter += 1
    return 1.0 if union == 0 else inter / union


def hamming(left: list[int], right: list[int]) -> int:
    return sum(int(a != b) for a, b in zip(left, right))


def diff_indices(source_mask: list[int], target_mask: list[int]) -> tuple[list[int], list[int]]:
    remove = []
    add = []
    for index, (src, tgt) in enumerate(zip(source_mask, target_mask)):
        if src == 1 and tgt == 0:
            remove.append(index)
        elif src == 0 and tgt == 1:
            add.append(index)
    return remove, add


def load_catalog(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_pairings(source_records: list[dict[str, object]], target_records: list[dict[str, object]]) -> dict[str, object]:
    remove_counter: Counter[int] = Counter()
    add_counter: Counter[int] = Counter()
    pair_summaries = []
    hammings = []
    jaccards = []

    for record in target_records:
        target_mask = record["mask"]
        best = None
        for source in source_records:
            score = hamming(source["mask"], target_mask)
            jac = jaccard(source["mask"], target_mask)
            candidate = (score, -jac, int(source["row_index"]), source)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        _, neg_jaccard, _, source = best
        remove, add = diff_indices(source["mask"], target_mask)
        remove_counter.update(remove)
        add_counter.update(add)
        hammings.append(len(remove) + len(add))
        jaccards.append(-neg_jaccard)
        pair_summaries.append(
            {
                "target_row_index": int(record["row_index"]),
                "target_support_size": int(record["support_size"]),
                "nearest_44_row_index": int(source["row_index"]),
                "nearest_44_support_size": int(source["support_size"]),
                "hamming_to_44": int(len(remove) + len(add)),
                "jaccard_to_44": float(-neg_jaccard),
                "remove_indices": remove,
                "add_indices": add,
            }
        )

    pair_summaries.sort(key=lambda item: (item["hamming_to_44"], -item["jaccard_to_44"], item["target_row_index"]))
    num_pairs = max(len(pair_summaries), 1)
    return {
        "num_pairs": len(pair_summaries),
        "mean_hamming_to_44": sum(hammings) / num_pairs,
        "mean_jaccard_to_44": sum(jaccards) / num_pairs,
        "top_remove_indices": [
            {"index": int(index), "count": int(count), "frequency": count / num_pairs}
            for index, count in remove_counter.most_common(16)
        ],
        "top_add_indices": [
            {"index": int(index), "count": int(count), "frequency": count / num_pairs}
            for index, count in add_counter.most_common(16)
        ],
        "closest_examples": pair_summaries[:12],
    }


def main() -> None:
    args = parse_args()
    target_classes = parse_target_classes(args.target_classes)
    records = load_catalog(args.catalog)

    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_class[int(record["class_id"])].append(record)

    source_records = by_class[int(args.source_class_id)]
    results = []
    for class_id in target_classes:
        target_records = by_class[int(class_id)][: args.per_class_limit]
        summary = summarize_pairings(source_records=source_records, target_records=target_records)
        results.append({"target_class_id": int(class_id), **summary})

    payload = {
        "catalog": str(args.catalog),
        "source_class_id": int(args.source_class_id),
        "source_class_size": len(source_records),
        "target_classes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
