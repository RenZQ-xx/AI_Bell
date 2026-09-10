"""Composable rollout scoring for RZQ MCTS experiments.

The wrapper delegates geometry, caching, and terminal classification to the
baseline ``ExpansionScorer``.  Only non-terminal rollout action values are
reassembled from explicit, independently configurable components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping

from baseline.facet_validator import FacetLabel
from baseline.orbit_blocks import BlockKey, add_block, unselected_blocks
from baseline.scorer import ExpansionScore, ExpansionScorer, exact_class_id

from .compatibility import ClassCompatibilityIndex
from .symmetry_quotient import build_partition_block_maps, representative_action_families
from .node_manager import RZQNodeManager


TerminalScoreFn = Callable[[BlockKey, FacetLabel], float]


@dataclass(frozen=True)
class RolloutScoreConfig:
    """Weights for the four non-terminal rollout score components."""

    positive_rank_gain_score: float = 5.0
    nonpositive_rank_gain_score: float = -5.0
    flat_weight: float = 1.0
    supportability_weight: float = 1.0
    decline_weight: float = 1.0
    decline_target_fraction: float = 0.2
    bridge_decline_weight: float = 5.0
    invalid_terminal_score: float = -10.0
    expansion_class_breadth_weight: float = 2.0
    expansion_mask_richness_weight: float = 1.0


@dataclass(frozen=True)
class RolloutScoreBreakdown:
    """Auditable component values for one candidate action."""

    action: int
    old_rank: int
    new_rank: int
    rank_gain: int
    flat_capacity: int
    rank_gain_score: float
    flat_score: float
    supportability_score: float
    decline_bonus: float
    decline_rate: float
    normal_decline_bonus: float
    bridge_state: bool
    bridge_elimination_rate: float
    bridge_decline_bonus: float
    total_score: float
    parent_compat_total: int
    child_compat_total: int
    parent_compat_classes: int
    child_compat_classes: int
    parent_compat_distribution: tuple[tuple[int, int], ...]
    child_compat_distribution: tuple[tuple[int, int], ...]
    decline_distribution: tuple[tuple[int, int], ...]
    terminal_label: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "old_rank": self.old_rank,
            "new_rank": self.new_rank,
            "rank_gain": self.rank_gain,
            "flat_capacity": self.flat_capacity,
            "rank_gain_score": self.rank_gain_score,
            "flat_score": self.flat_score,
            "supportability_score": self.supportability_score,
            "decline_bonus": self.decline_bonus,
            "decline_rate": self.decline_rate,
            "normal_decline_bonus": self.normal_decline_bonus,
            "bridge_state": self.bridge_state,
            "bridge_elimination_rate": self.bridge_elimination_rate,
            "bridge_decline_bonus": self.bridge_decline_bonus,
            "total_score": self.total_score,
            "parent_compat_total": self.parent_compat_total,
            "child_compat_total": self.child_compat_total,
            "parent_compat_classes": self.parent_compat_classes,
            "child_compat_classes": self.child_compat_classes,
            "parent_compat_distribution": dict(self.parent_compat_distribution),
            "child_compat_distribution": dict(self.child_compat_distribution),
            "decline_distribution": dict(self.decline_distribution),
            "terminal_label": self.terminal_label,
        }


class RZQRolloutScorer:
    """Drop-in scorer wrapper for rollout experiments.

    ``score_action`` matches the interface consumed by ``mcts.search._rollout``.
    Attributes and methods unrelated to action scoring are delegated to the
    baseline scorer, so terminal validation and geometric caches stay intact.
    """

    def __init__(
        self,
        base_scorer: ExpansionScorer,
        compatibility: ClassCompatibilityIndex,
        discovered_class_ids: Iterable[int],
        config: RolloutScoreConfig | None = None,
    ) -> None:
        if len(base_scorer.blocks) != compatibility.block_count:
            raise ValueError("base scorer and compatibility index use different block counts")
        self.base_scorer = base_scorer
        self.compatibility = compatibility
        self.rollout_score_config = config or RolloutScoreConfig()
        self.breakdown_cache: dict[tuple[BlockKey, int], RolloutScoreBreakdown] = {}
        root = tuple(0 for _ in range(compatibility.block_count))
        root_counts = compatibility.counts(root)
        discovered = {int(class_id) for class_id in discovered_class_ids}
        # This set depends only on the class-8 block partition and the known
        # class set at experiment start, so it is deliberately built once.
        self.known_compatible_classes = tuple(
            class_id
            for class_id, count in sorted(root_counts.items())
            if class_id in discovered and int(count) > 0
        )
        self._compat_distribution_cache: dict[BlockKey, dict[int, int]] = {}
        self.block_symmetry_maps = build_partition_block_maps(base_scorer.blocks)
        self._action_family_cache: dict[BlockKey, dict[int, tuple[int, ...]]] = {}
        self._flat_closure_cache: dict[BlockKey, tuple[BlockKey, tuple[int, ...]]] = {}
        self.node_manager = RZQNodeManager(self)
        self.expansion_prior_mode = "compatibility_richness"
        self._discovered_classes_provider: Callable[[], Iterable[int]] = lambda: self.known_compatible_classes
        self._discovery_epoch_provider: Callable[[], int] = lambda: 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.base_scorer, name)

    @property
    def rare_target_classes(self) -> set[int]:
        return self.base_scorer.rare_target_classes

    @rare_target_classes.setter
    def rare_target_classes(self, value: Iterable[int]) -> None:
        self.base_scorer.rare_target_classes = {int(class_id) for class_id in value}

    def terminal_score(
        self,
        key: BlockKey,
        *,
        rare_target_classes: set[int] | None = None,
        discovered_label_counts: Mapping[str, int] | None = None,
    ) -> float:
        """Apply the baseline terminal policy with uniform invalid scoring.

        Class 44 follows the same rare/target/valid path as every other exact
        class.  Every invalid result, including a hyperplane that cuts the
        polytope, receives the same score.
        """

        cfg = self.base_scorer.config
        if self.base_scorer.fit_boundary_metrics(key)["closer_side"] > 0:
            return float(self.rollout_score_config.invalid_terminal_score)
        terminal = self.base_scorer.terminal_label(key)
        if not terminal.is_exact:
            return float(self.rollout_score_config.invalid_terminal_score)
        class_id = exact_class_id(terminal.label)
        if cfg.terminal_scoring_mode == "dynamic":
            counts = (
                self.base_scorer.discovered_label_counts
                if discovered_label_counts is None
                else discovered_label_counts
            )
            count = int(counts.get(terminal.label, 0))
            if count <= 0:
                return float(cfg.dynamic_new_class_score)
            if count >= cfg.dynamic_frequent_class_threshold:
                return float(cfg.dynamic_frequent_class_score)
            return float(cfg.dynamic_known_class_score)
        active_rare = self.rare_target_classes if rare_target_classes is None else rare_target_classes
        if class_id in active_rare:
            return float(cfg.rare_terminal_score)
        if class_id in self.base_scorer.target_classes:
            return float(cfg.target_terminal_score)
        return float(cfg.valid_terminal_score)

    def rank_gain_score(self, rank_gain: int, new_rank: int) -> float:
        del new_rank
        cfg = self.rollout_score_config
        return cfg.positive_rank_gain_score if rank_gain > 0 else cfg.nonpositive_rank_gain_score

    def representative_action_families(
        self,
        key: BlockKey,
        actions: Iterable[int] | None = None,
    ) -> dict[int, tuple[int, ...]]:
        """Return minimum-action representatives under the parent stabilizer."""

        empty = tuple(0 for _ in range(len(key)))
        closed_root, _ = self.flat_closure(empty)
        use_full_group = key == closed_root
        if actions is not None:
            return representative_action_families(
                key,
                self.block_symmetry_maps,
                actions,
                use_full_group=use_full_group,
            )
        if key not in self._action_family_cache:
            self._action_family_cache[key] = representative_action_families(
                key,
                self.block_symmetry_maps,
                use_full_group=use_full_group,
            )
        return dict(self._action_family_cache[key])

    def flat_score(self, flat_capacity: int, new_rank: int) -> float:
        del new_rank
        return -self.rollout_score_config.flat_weight * math.log1p(float(flat_capacity))

    def supportability_score(self, metrics: object | None) -> float:
        if metrics is None:
            return 0.0
        cfg = self.rollout_score_config
        closer = float(getattr(metrics, "closer_side"))
        return -cfg.supportability_weight * math.log1p(closer)

    def compatibility_distribution(self, key: BlockKey) -> dict[int, int]:
        """Counts for classes known and compatible with the class-8 root."""

        if key not in self._compat_distribution_cache:
            all_counts = self.compatibility.counts(key)
            self._compat_distribution_cache[key] = {
                class_id: int(all_counts.get(class_id, 0))
                for class_id in self.known_compatible_classes
            }
        return dict(self._compat_distribution_cache[key])

    def compatibility_counts(
        self,
        key: BlockKey,
        class_ids: Iterable[int],
    ) -> dict[int, int]:
        """Return node-cache counts for the requested discovered classes."""
        requested = {int(class_id) for class_id in class_ids}
        all_counts = self.compatibility.counts(key)
        return {class_id: int(all_counts.get(class_id, 0)) for class_id in sorted(requested)}

    def configure_discovery_cache(
        self,
        class_ids_provider: Callable[[], Iterable[int]],
        epoch_provider: Callable[[], int],
    ) -> None:
        self._discovered_classes_provider = class_ids_provider
        self._discovery_epoch_provider = epoch_provider

    def refresh_node_compatibility(self, node: object) -> dict[int, int]:
        return node.refresh_compatibility(
            self,
            sorted(int(value) for value in self._discovered_classes_provider()),
            int(self._discovery_epoch_provider()),
        )

    def prepare_expansion_priors(self, node: object, temperature: float) -> None:
        """Score every canonical action using the configured prior source."""
        epoch = int(self._discovery_epoch_provider())
        actions = sorted(int(action) for action in node.action_families)
        if node.expansion_compatibility_epoch == epoch and set(node.expansion_scores) == set(actions):
            return
        discovered = sorted({int(value) for value in self._discovered_classes_provider()})
        scores: dict[int, float] = {}
        details: dict[int, dict[str, object]] = {}
        cfg = self.rollout_score_config
        for action in actions:
            if self.expansion_prior_mode == "structural_score":
                score_item = self.score_action(node.key, action)
                node.action_scores[action] = score_item
                scores[action] = float(score_item.score)
                details[action] = {
                    "prior_source": "structural_score",
                    "score": float(score_item.score),
                    "new_rank": int(score_item.new_rank),
                    "flat_capacity": int(score_item.flat_capacity),
                    "terminal": None if score_item.terminal is None else score_item.terminal.label,
                }
                continue
            if self.expansion_prior_mode != "compatibility_richness":
                raise ValueError(f"unknown expansion prior mode: {self.expansion_prior_mode}")
            child_key, closure_blocks = self.flat_closure(add_block(node.key, action))
            counts = self.compatibility_counts(child_key, discovered)
            positive = {class_id: count for class_id, count in counts.items() if count > 0}
            class_count = len(positive)
            mask_total = sum(positive.values())
            mean_log_masks = (
                sum(math.log1p(float(count)) for count in positive.values()) / class_count
                if class_count
                else 0.0
            )
            score = (
                float(cfg.expansion_class_breadth_weight) * math.log1p(float(class_count))
                + float(cfg.expansion_mask_richness_weight) * mean_log_masks
            )
            scores[action] = float(score)
            details[action] = {
                "compatible_class_count": class_count,
                "compatible_mask_total": mask_total,
                "mean_log_compatible_masks": mean_log_masks,
                "closure_blocks": list(closure_blocks),
                "compatibility_counts": positive,
            }
        if not scores:
            node.action_priors = {}
        else:
            if float(temperature) <= 0.0:
                raise ValueError("prior temperature must be positive")
            peak = max(scores.values())
            weights = {
                action: math.exp((score - peak) / float(temperature))
                for action, score in scores.items()
            }
            normalizer = sum(weights.values())
            node.action_priors = {
                action: weight / normalizer for action, weight in weights.items()
            }
            for action, child in node.children.items():
                child.prior = node.action_priors.get(action, child.prior)
        node.expansion_scores = scores
        node.expansion_score_details = details
        node.expansion_prior_source = self.expansion_prior_mode
        node.expansion_compatibility_epoch = epoch

    def prepare_legacy_action_priors(self, node: object, temperature: float) -> None:
        """Assign compatibility priors without enabling canonical two-bucket expansion."""

        epoch = int(self._discovery_epoch_provider())
        actions = sorted(int(action) for action in node.action_families)
        if (
            node.legacy_prior_override
            and node.expansion_compatibility_epoch == epoch
            and set(node.expansion_score_details) == set(actions)
        ):
            return
        discovered = sorted({int(value) for value in self._discovered_classes_provider()})
        scores: dict[int, float] = {}
        details: dict[int, dict[str, object]] = {}
        cfg = self.rollout_score_config
        for action in actions:
            child_key = add_block(node.key, action)
            counts = self.compatibility_counts(child_key, discovered)
            positive = {class_id: count for class_id, count in counts.items() if count > 0}
            class_count = len(positive)
            mean_log_masks = (
                sum(math.log1p(float(count)) for count in positive.values()) / class_count
                if class_count
                else 0.0
            )
            scores[action] = (
                float(cfg.expansion_class_breadth_weight) * math.log1p(float(class_count))
                + float(cfg.expansion_mask_richness_weight) * mean_log_masks
            )
            details[action] = {
                "prior_source": "compatibility_richness",
                "compatible_class_count": class_count,
                "compatible_mask_total": sum(positive.values()),
                "mean_log_compatible_masks": mean_log_masks,
                "compatibility_counts": positive,
            }
        if float(temperature) <= 0.0:
            raise ValueError("prior temperature must be positive")
        peak = max(scores.values()) if scores else 0.0
        weights = {action: math.exp((score - peak) / float(temperature)) for action, score in scores.items()}
        normalizer = sum(weights.values())
        node.action_priors = {
            action: weight / normalizer for action, weight in weights.items()
        } if normalizer > 0.0 else {}
        node.expansion_score_details = details
        node.expansion_prior_source = "compatibility_richness"
        node.expansion_compatibility_epoch = epoch
        node.legacy_prior_override = True

    def flat_closure(self, key: BlockKey) -> tuple[BlockKey, tuple[int, ...]]:
        """Add all blocks contained in the current affine hull, until flat is zero."""
        cached = self._flat_closure_cache.get(key)
        if cached is not None:
            return cached
        closed = tuple(int(value) for value in key)
        if not any(closed):
            result = closed, ()
            self._flat_closure_cache[key] = result
            return result
        added: list[int] = []
        while True:
            rank = self.base_scorer.affine_rank(closed)
            redundant = [
                int(action)
                for action in unselected_blocks(closed)
                if self.base_scorer.affine_rank(add_block(closed, int(action))) == rank
            ]
            if not redundant:
                break
            values = list(closed)
            for action in redundant:
                values[action] = 1
            closed = tuple(values)
            added.extend(redundant)
        result = closed, tuple(sorted(set(added)))
        self._flat_closure_cache[key] = result
        return result

    def decline_components(
        self,
        parent_counts: Mapping[int, int],
        child_counts: Mapping[int, int],
    ) -> tuple[float, float, bool, float, float]:
        """Return D, normal bonus, bridge flag, E, and bridge bonus."""

        active = tuple(class_id for class_id, count in parent_counts.items() if int(count) > 0)
        if not active:
            return 0.0, 0.0, False, 0.0, 0.0
        decline_rate = sum(
            max(0.0, float(parent_counts[class_id] - child_counts[class_id]))
            / float(parent_counts[class_id])
            for class_id in active
        ) / float(len(active))
        cfg = self.rollout_score_config
        target = float(cfg.decline_target_fraction)
        if target <= 0.0:
            raise ValueError("decline_target_fraction must be positive")
        normal_bonus = (
            float(cfg.decline_weight)
            * (decline_rate / target)
            * math.exp(1.0 - decline_rate / target)
            if decline_rate > 0.0
            else 0.0
        )
        bridge_state = max(int(value) for value in parent_counts.values()) <= 1
        parent_total = sum(int(parent_counts[class_id]) for class_id in active)
        eliminated = sum(
            max(0, int(parent_counts[class_id]) - int(child_counts[class_id]))
            for class_id in active
        )
        bridge_elimination_rate = (
            float(eliminated) / float(parent_total)
            if bridge_state and parent_total > 0
            else 0.0
        )
        bridge_bonus = float(cfg.bridge_decline_weight) * bridge_elimination_rate
        return decline_rate, normal_bonus, bridge_state, bridge_elimination_rate, bridge_bonus

    def decline_bonus(
        self,
        parent_counts: Mapping[int, int],
        child_counts: Mapping[int, int],
    ) -> float:
        _rate, normal, _bridge, _elimination, bridge = self.decline_components(
            parent_counts,
            child_counts,
        )
        return normal + bridge

    def score_action(
        self,
        key: BlockKey,
        action: int,
        *,
        terminal_score_fn: TerminalScoreFn | None = None,
    ) -> ExpansionScore:
        structure = self.base_scorer.action_structure(key, action)
        closed_key, _closure_blocks = self.flat_closure(structure.key)
        if closed_key != structure.key:
            closed_rank = self.base_scorer.affine_rank(closed_key)
            closed_terminal = self.base_scorer.terminal_label(closed_key) if closed_rank >= 25 else None
            structure = replace(
                structure,
                key=closed_key,
                new_rank=closed_rank,
                rank_gain=closed_rank - structure.old_rank,
                flat_capacity=0,
                supportability=(
                    None
                    if closed_terminal is not None
                    else self.base_scorer.supportability_metrics(closed_key)
                ),
                terminal=closed_terminal,
                phase="C" if closed_terminal is not None else structure.phase,
            )
        parent_counts = self.compatibility_distribution(key)
        child_counts = self.compatibility_distribution(structure.key)
        decline_counts = {
            class_id: max(0, parent_counts[class_id] - child_counts[class_id])
            for class_id in self.known_compatible_classes
        }
        parent_distribution = tuple(sorted(parent_counts.items()))
        child_distribution = tuple(sorted(child_counts.items()))
        decline_distribution = tuple(sorted(decline_counts.items()))
        decline_rate, normal_decline, bridge_state, bridge_rate, bridge_decline = (
            self.decline_components(parent_counts, child_counts)
        )
        if structure.terminal is not None:
            total = (
                self.terminal_score(structure.key)
                if terminal_score_fn is None
                else float(terminal_score_fn(structure.key, structure.terminal))
            )
            self.breakdown_cache[(key, int(action))] = RolloutScoreBreakdown(
                action=int(action), old_rank=structure.old_rank, new_rank=structure.new_rank,
                rank_gain=structure.rank_gain, flat_capacity=structure.flat_capacity,
                rank_gain_score=0.0, flat_score=0.0, supportability_score=0.0,
                decline_bonus=0.0, decline_rate=decline_rate,
                normal_decline_bonus=normal_decline, bridge_state=bridge_state,
                bridge_elimination_rate=bridge_rate, bridge_decline_bonus=bridge_decline,
                total_score=float(total),
                parent_compat_total=sum(parent_counts.values()),
                child_compat_total=sum(child_counts.values()),
                parent_compat_classes=sum(value > 0 for value in parent_counts.values()),
                child_compat_classes=sum(value > 0 for value in child_counts.values()),
                parent_compat_distribution=parent_distribution,
                child_compat_distribution=child_distribution,
                decline_distribution=decline_distribution,
                terminal_label=structure.terminal.label,
            )
        else:
            rank_score = self.rank_gain_score(structure.rank_gain, structure.new_rank)
            flat_score = self.flat_score(structure.flat_capacity, structure.new_rank)
            support_score = self.supportability_score(structure.supportability)
            decline = normal_decline + bridge_decline
            total = rank_score + flat_score + support_score + decline
            self.breakdown_cache[(key, int(action))] = RolloutScoreBreakdown(
                action=int(action), old_rank=structure.old_rank, new_rank=structure.new_rank,
                rank_gain=structure.rank_gain, flat_capacity=structure.flat_capacity,
                rank_gain_score=rank_score, flat_score=flat_score,
                supportability_score=support_score, decline_bonus=decline,
                decline_rate=decline_rate, normal_decline_bonus=normal_decline,
                bridge_state=bridge_state, bridge_elimination_rate=bridge_rate,
                bridge_decline_bonus=bridge_decline,
                total_score=float(total), parent_compat_total=sum(parent_counts.values()),
                child_compat_total=sum(child_counts.values()),
                parent_compat_classes=sum(value > 0 for value in parent_counts.values()),
                child_compat_classes=sum(value > 0 for value in child_counts.values()),
                parent_compat_distribution=parent_distribution,
                child_compat_distribution=child_distribution,
                decline_distribution=decline_distribution,
            )
        return ExpansionScore(
            action=structure.action,
            key=structure.key,
            score=float(total),
            phase=structure.phase,
            old_rank=structure.old_rank,
            new_rank=structure.new_rank,
            rank_gain=structure.rank_gain,
            flat_capacity=structure.flat_capacity,
            supportability=structure.supportability,
            terminal=structure.terminal,
        )

    def breakdown(self, key: BlockKey, action: int) -> RolloutScoreBreakdown:
        try:
            return self.breakdown_cache[(key, int(action))]
        except KeyError as exc:
            raise KeyError("score_action must be called before requesting its breakdown") from exc
