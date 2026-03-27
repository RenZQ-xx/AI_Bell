from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class BranchState:
    branch_id: int
    mask: torch.Tensor
    canonical_key: str | None
    matched_classes: list[int]
    depth: int
    parent_branch_id: int | None
    history_masks: list[torch.Tensor]
    times_expanded: int = 0
    num_children_exact: int = 0
    num_children_novel: int = 0
    last_novel_round: int | None = None


@dataclass
class CandidateRecord:
    branch_id: int
    parent_branch_id: int
    mask: torch.Tensor
    canonical_key: str | None
    matched_classes: list[int]
    tier: str
    round_index: int
    exact: bool
    quality_score: float = 0.0
    tiebreak_score: float = 0.0
    source_index: int = -1


def trim_history(history_masks: list[torch.Tensor], history_limit: int | None) -> list[torch.Tensor]:
    if history_limit is None or history_limit <= 0 or len(history_masks) <= history_limit:
        return [item.detach().cpu() for item in history_masks]
    return [item.detach().cpu() for item in history_masks[-history_limit:]]


def seed_branch(mask: torch.Tensor, branch_id: int, history_limit: int | None = None) -> BranchState:
    return BranchState(
        branch_id=branch_id,
        mask=mask.detach().cpu(),
        canonical_key=None,
        matched_classes=[],
        depth=0,
        parent_branch_id=None,
        history_masks=trim_history([mask], history_limit=history_limit),
    )


class LongHorizonController:
    def __init__(
        self,
        frontier_size: int,
        max_children_per_parent: int = 2,
        max_same_class: int = 2,
        history_limit: int | None = None,
        allow_bridge_candidates: bool = False,
        max_bridge_children: int = 1,
        bridge_similarity_threshold: float = 0.85,
    ) -> None:
        self.frontier_size = frontier_size
        self.max_children_per_parent = max_children_per_parent
        self.max_same_class = max_same_class
        self.history_limit = history_limit
        self.allow_bridge_candidates = allow_bridge_candidates
        self.max_bridge_children = max_bridge_children
        self.bridge_similarity_threshold = bridge_similarity_threshold

    @staticmethod
    def mask_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        left_mask = left >= 0.5
        right_mask = right >= 0.5
        intersection = (left_mask & right_mask).sum().item()
        union = (left_mask | right_mask).sum().item()
        if union == 0:
            return 1.0
        return float(intersection / union)

    def score_candidate(
        self,
        candidate: CandidateRecord,
        seen_canonical: set[str],
        seen_classes: set[int],
    ) -> tuple[int, int, int, int, float, float, int]:
        is_new_canonical = int(candidate.canonical_key is not None and candidate.canonical_key not in seen_canonical)
        is_new_class = int(any(class_id not in seen_classes for class_id in candidate.matched_classes))
        class_bonus = len([class_id for class_id in candidate.matched_classes if class_id not in seen_classes])
        exact_bonus = 1 if candidate.exact else 0
        return (
            exact_bonus,
            is_new_canonical,
            is_new_class,
            class_bonus,
            float(candidate.quality_score),
            float(candidate.tiebreak_score),
            -candidate.parent_branch_id,
        )

    def update_frontier(
        self,
        frontier: list[BranchState],
        candidates: Iterable[CandidateRecord],
        seen_canonical: set[str],
        seen_classes: set[int],
        next_branch_id: int,
        round_index: int,
    ) -> tuple[list[BranchState], int]:
        candidate_pool = []
        for candidate in candidates:
            if candidate.exact and candidate.canonical_key is not None:
                candidate_pool.append(candidate)
                continue
            if self.allow_bridge_candidates and not candidate.exact:
                candidate_pool.append(candidate)
        sorted_candidates = sorted(
            candidate_pool,
            key=lambda candidate: self.score_candidate(candidate, seen_canonical=seen_canonical, seen_classes=seen_classes),
            reverse=True,
        )

        next_frontier: list[BranchState] = []
        parent_counts: dict[int, int] = {}
        class_counts: dict[int, int] = {}
        used_canonical: set[str] = set()
        bridge_children = 0
        branch_by_id = {branch.branch_id: branch for branch in frontier}

        for candidate in sorted_candidates:
            candidate_key = candidate.canonical_key or f"bridge:{candidate.parent_branch_id}:{candidate.round_index}:{len(next_frontier)}"
            if candidate_key in used_canonical:
                continue
            if parent_counts.get(candidate.parent_branch_id, 0) >= self.max_children_per_parent:
                continue
            if not candidate.exact and bridge_children >= self.max_bridge_children:
                continue
            if not candidate.exact and any(
                self.mask_similarity(candidate.mask, existing.mask) >= self.bridge_similarity_threshold
                for existing in next_frontier
                if not existing.canonical_key
            ):
                continue
            if candidate.matched_classes:
                primary_class = candidate.matched_classes[0]
                if class_counts.get(primary_class, 0) >= self.max_same_class:
                    continue
            parent_branch = branch_by_id.get(candidate.parent_branch_id)
            parent_history = [] if parent_branch is None else list(parent_branch.history_masks)
            child_history = trim_history(parent_history + [candidate.mask], history_limit=self.history_limit)
            next_frontier.append(
                BranchState(
                    branch_id=next_branch_id,
                    mask=candidate.mask.detach().cpu(),
                    canonical_key=candidate.canonical_key,
                    matched_classes=list(candidate.matched_classes),
                    depth=(parent_branch.depth + 1) if parent_branch is not None else 1,
                    parent_branch_id=candidate.parent_branch_id,
                    history_masks=child_history,
                    last_novel_round=round_index if candidate.canonical_key not in seen_canonical else None,
                )
            )
            next_branch_id += 1
            used_canonical.add(candidate_key)
            parent_counts[candidate.parent_branch_id] = parent_counts.get(candidate.parent_branch_id, 0) + 1
            for class_id in candidate.matched_classes:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            if not candidate.exact:
                bridge_children += 1
            if len(next_frontier) >= self.frontier_size:
                break

        if not next_frontier:
            fallback = sorted(frontier, key=lambda branch: (branch.times_expanded, branch.depth))
            next_frontier = fallback[: self.frontier_size]

        return next_frontier, next_branch_id
