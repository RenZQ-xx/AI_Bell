from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .orbit_blocks import BlockKey, add_block, empty_key, selected_blocks, unselected_blocks
from .scorer import ExpansionScore, ExpansionScorer


@dataclass(frozen=True)
class BeamSearchConfig:
    """Minimal search controls for the baseline add-block beam search."""

    beam_width: int = 32
    candidate_pool: int = 16
    samples_per_state: int = 4
    max_blocks: int = 64
    stochastic: bool = False
    temperature: float = 1.0
    seed: int = 0
    beam_diversity_slots: int = 0
    beam_diversity_rank_start: int = 17
    beam_diversity_rank_end: int = 21
    beam_diversity_temperature: float = 0.75


@dataclass
class TerminalHit:
    """One terminal rank-25 candidate encountered during search."""

    label: str
    key: BlockKey
    path: list[int]
    score: float
    rank: int

    @property
    def chosen_blocks(self) -> list[int]:
        return selected_blocks(self.key)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "chosen_blocks": self.chosen_blocks,
            "path": list(self.path),
            "score": self.score,
            "rank": self.rank,
        }


@dataclass
class SearchResult:
    """Compact result of one zero-start add-block search run."""

    best: TerminalHit | None
    terminal_bests: dict[str, TerminalHit] = field(default_factory=dict)
    encountered_label_counts: Counter[str] = field(default_factory=Counter)
    steps_completed: int = 0
    final_beam_size: int = 0

    @property
    def exact_labels(self) -> list[str]:
        return sorted(label for label in self.terminal_bests if label.startswith("exact:"))

    def to_dict(self) -> dict[str, object]:
        return {
            "best": None if self.best is None else self.best.to_dict(),
            "terminal_bests": {
                label: hit.to_dict()
                for label, hit in sorted(self.terminal_bests.items())
            },
            "encountered_label_counts": dict(sorted(self.encountered_label_counts.items())),
            "exact_labels": self.exact_labels,
            "steps_completed": self.steps_completed,
            "final_beam_size": self.final_beam_size,
        }


def run_beam_search(
    scorer: ExpansionScorer,
    *,
    config: BeamSearchConfig | None = None,
    start_key: BlockKey | None = None,
) -> SearchResult:
    """Run one add-only beam search from the empty block state."""
    cfg = BeamSearchConfig() if config is None else config
    if cfg.beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {cfg.beam_width}")
    block_count = len(scorer.blocks)
    root = empty_key(block_count) if start_key is None else start_key
    rng = random.Random(cfg.seed)

    beam: list[tuple[BlockKey, list[int]]] = [(root, [])]
    terminal_bests: dict[str, TerminalHit] = {}
    encountered = Counter()
    best: TerminalHit | None = None
    steps_completed = 0

    for _step in range(min(cfg.max_blocks, block_count)):
        expanded: dict[BlockKey, tuple[float, list[int], ExpansionScore]] = {}
        for key, path in beam:
            if scorer.affine_rank(key) >= 25:
                continue
            candidates = [scorer.score_action(key, action) for action in unselected_blocks(key)]
            candidates.sort(key=lambda item: item.score, reverse=True)
            pool = candidates[: max(1, min(cfg.candidate_pool, len(candidates)))]
            chosen_items = sample_without_replacement(pool, rng, cfg.samples_per_state, cfg.temperature)
            for item in chosen_items:
                new_path = [*path, int(item.action)]
                if item.terminal is not None:
                    label = item.terminal.label
                    encountered[label] += 1
                    if item.terminal.is_exact:
                        scorer.discovered_label_counts[label] += 1
                    hit = TerminalHit(
                        label=label,
                        key=item.key,
                        path=new_path,
                        score=float(item.score),
                        rank=int(item.new_rank),
                    )
                    previous = terminal_bests.get(label)
                    if previous is None or hit.score > previous.score:
                        terminal_bests[label] = hit
                    if best is None or hit.score > best.score:
                        best = hit
                    continue
                previous = expanded.get(item.key)
                if previous is None or item.score > previous[0]:
                    expanded[item.key] = (item.score, new_path, item)

        if not expanded:
            break

        ranked = sorted(expanded.items(), key=lambda pair: pair[1][0], reverse=True)
        selected_pairs = select_next_beam(ranked, cfg, rng, scorer)
        beam = [(key, payload[1]) for key, payload in selected_pairs]
        steps_completed += 1
        if not beam:
            break

    return SearchResult(
        best=best,
        terminal_bests=terminal_bests,
        encountered_label_counts=encountered,
        steps_completed=steps_completed,
        final_beam_size=len(beam),
    )


def select_next_beam(
    ranked: Sequence[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]],
    config: BeamSearchConfig,
    rng: random.Random,
    scorer: ExpansionScorer | None = None,
) -> list[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]]:
    """Select the next beam either greedily or by weighted sampling."""
    limit = min(config.beam_width, len(ranked))
    diversity_slots = 0
    if scorer is not None and config.beam_diversity_slots > 0:
        top_ranks = [scorer.affine_rank(key) for key, _payload in ranked[:limit]]
        if any(config.beam_diversity_rank_start <= rank <= config.beam_diversity_rank_end for rank in top_ranks):
            diversity_slots = min(config.beam_diversity_slots, limit)
    if diversity_slots > 0 and scorer is not None:
        return select_with_bucket_diversity(ranked, config, rng, scorer, limit, diversity_slots)

    if not config.stochastic:
        return list(ranked[:limit])

    remaining = list(ranked)
    selected: list[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]] = []
    while remaining and len(selected) < limit:
        index = weighted_choice_index([payload[0] for _key, payload in remaining], rng, config.temperature)
        selected.append(remaining.pop(index))
    return selected


def select_with_bucket_diversity(
    ranked: Sequence[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]],
    config: BeamSearchConfig,
    rng: random.Random,
    scorer: ExpansionScorer,
    limit: int,
    diversity_slots: int,
) -> list[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]]:
    """Reserve beam slots for diverse structural buckets, matching legacy search."""
    score_slots = max(0, limit - diversity_slots)
    selected = list(ranked[:score_slots])
    selected_keys = {key for key, _payload in selected}
    selected_buckets = {structural_bucket(scorer, key) for key, _payload in selected}
    near_limit = min(len(ranked), limit * 4)

    buckets: dict[str, list[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]]] = {}
    for key, payload in ranked[score_slots:near_limit]:
        if key in selected_keys:
            continue
        rank = scorer.affine_rank(key)
        if rank < config.beam_diversity_rank_start or rank > config.beam_diversity_rank_end:
            continue
        bucket = structural_bucket(scorer, key)
        buckets.setdefault(bucket, []).append((key, payload))

    bucket_order = sorted(
        buckets,
        key=lambda bucket: (0 if bucket not in selected_buckets else 1, -len(buckets[bucket])),
    )
    cursor = 0
    released: list[tuple[BlockKey, tuple[float, list[int], ExpansionScore]]] = []
    while len(released) < diversity_slots and bucket_order:
        bucket = bucket_order[cursor % len(bucket_order)]
        options = [(key, payload) for key, payload in buckets[bucket] if key not in selected_keys]
        if options:
            index = weighted_choice_index(
                [payload[0] for _key, payload in options],
                rng,
                config.beam_diversity_temperature,
            )
            key, payload = options[index]
            released.append((key, payload))
            selected_keys.add(key)
            selected_buckets.add(bucket)
        cursor += 1
        if cursor >= len(bucket_order) * 2 and all(
            all(key in selected_keys for key, _payload in buckets[bucket])
            for bucket in bucket_order
        ):
            break

    selected.extend(released)
    if len(selected) < limit:
        for key, payload in ranked:
            if key in selected_keys:
                continue
            selected.append((key, payload))
            selected_keys.add(key)
            if len(selected) >= limit:
                break
    return selected[:limit]


def structural_bucket(scorer: ExpansionScorer, key: BlockKey) -> str:
    """Small structural signature used for legacy bucket diversity."""
    rank = scorer.affine_rank(key)
    flat = 0 if rank >= 25 else scorer.flat_capacity(key)
    gain_count = sum(
        1
        for action in unselected_blocks(key)
        if scorer.affine_rank(add_block(key, action)) > rank
    )
    child_flat = scorer.child_flat_p50(key)
    return "|".join(
        [
            f"r{rank}",
            f"f{bin_value(flat, [0, 1, 2, 4, 8, 12])}",
            f"g{bin_value(gain_count, [4, 8, 12, 16, 20, 24, 32])}",
            f"cf{bin_value(child_flat, [0, 1, 2, 4, 6, 8, 12])}",
        ]
    )


def bin_value(value: float, thresholds: Sequence[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if float(value) <= float(threshold):
            return index
    return len(thresholds)


def sample_without_replacement(
    items: Sequence[ExpansionScore],
    rng: random.Random,
    count: int,
    temperature: float,
) -> list[ExpansionScore]:
    """Sample scored expansion items without replacement."""
    available = list(items)
    chosen: list[ExpansionScore] = []
    for _ in range(max(1, min(int(count), len(available)))):
        index = weighted_choice_index([item.score for item in available], rng, temperature)
        chosen.append(available.pop(index))
        if not available:
            break
    return chosen


def weighted_choice_index(scores: Sequence[float], rng: random.Random, temperature: float) -> int:
    """Sample an index with softmax weights over scores."""
    if not scores:
        raise ValueError("cannot sample from an empty score list")
    temp = max(float(temperature), 1e-6)
    max_score = max(float(score) for score in scores)
    weights = [pow(2.718281828459045, (float(score) - max_score) / temp) for score in scores]
    total = sum(weights)
    if total <= 0.0:
        return len(scores) - 1
    threshold = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if running >= threshold:
            return index
    return len(scores) - 1
