from __future__ import annotations

"""Independently replay full witnesses and local continuation suffixes."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetOracle, RankOracle


def audit_paths(run_path: Path, *, handoff_root: Path = DEFAULT_HANDOFF_ROOT,
                payload: dict | None = None) -> dict[str, object]:
    if payload is None:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    ranks = RankOracle()
    oracle = FacetOracle(
        rank_oracle=ranks, handoff_root=handoff_root,
        use_catalog=False, canonicalize=False,
    )
    transitions: dict[tuple, tuple[int, int]] = {}
    failures: list[dict[str, object]] = []
    rank_changes: Counter[str] = Counter()
    noop_corrections = 0
    started = time.perf_counter()

    def replay(word: int, actions: Sequence[dict]) -> int:
        nonlocal noop_corrections
        for action in actions:
            kind = action["kind"]
            removed = int(action["remove_word_hex"], 16)
            added = int(action["add_word_hex"], 16)
            key = word, kind, removed, added
            before = word
            if key not in transitions:
                if kind == "remove":
                    word &= ~removed
                elif kind == "add":
                    word |= added
                elif kind == "rewrite":
                    word = (word & ~removed) | added
                rank = ranks.rank(word)
                if rank == 79:
                    facet = oracle.validate(word)
                    if not facet.valid:
                        raise ValueError("invalid rank-79 witness step")
                    word = facet.support_word
                if rank > 79:
                    raise ValueError("witness crosses affine rank 79")
                transitions[key] = word, rank
            word, rank = transitions[key]
            rank_changes[f"{kind}:{ranks.rank(before)}->{rank}"] += 1
            noop_corrections += int(kind in ("remove", "rewrite") and word == before)
        return word

    events = payload["coverage"]["discoveries"]
    for event in events:
        try:
            target = int(event["support_word_hex"], 16)
            if replay(0, event["path"]) != target:
                raise ValueError("full witness does not reach the reported support")
            origin = event.get("continuation_origin")
            local_start = int(origin["support_word_hex"], 16) if origin else 0
            local_path = event.get("local_path", event["path"])
            if replay(local_start, local_path) != target:
                raise ValueError("local suffix does not reach the reported support")
            if origin:
                prefix_length = origin["historical_path_length"]
                if replay(0, event["path"][:prefix_length]) != local_start:
                    raise ValueError("historical prefix does not reach the continuation origin")
                if event["path"][prefix_length:] != local_path:
                    raise ValueError("local path differs from the full witness suffix")
            local_corrections = sum(action["kind"] in ("remove", "rewrite") for action in local_path)
            if local_corrections > payload["config"]["extended_max_corrections"]:
                raise ValueError("local correction budget exceeded")
        except (KeyError, ValueError) as error:
            failures.append({"class_id": event["class_id"], "error": str(error)})
    return {
        "run_path": str(run_path),
        "paths_checked": len(events),
        "failure_count": len(failures),
        "failures": failures,
        "unique_transitions": len(transitions),
        "rank_change_occurrences": dict(rank_changes),
        "noop_correction_occurrences_in_full_local_and_prefix_replays": noop_corrections,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    args = parser.parse_args(argv)
    result = audit_paths(args.run_path, handoff_root=args.handoff_root)
    output = args.output or args.run_path.with_suffix(".paths.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(result["failure_count"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
