from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from baseline.facet_validator import FacetLabel
from baseline.orbit_blocks import BlockKey, add_block, empty_key, selected_blocks, unselected_blocks
from baseline.reference_classes import DEFAULT_EXAMPLES_PATH, parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScore, ExpansionScorer, exact_class_id
from mcts.decision_trace import emit as _trace_emit


TerminalScoreFn = Callable[[BlockKey, FacetLabel], float]
ExactHitCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class MCTSConfig:
    """Controls for the add-only MCTS baseline."""

    iterations: int = 2000
    max_depth: int = 60
    exploration_constant: float = 1.4
    discount: float = 0.97
    prior_temperature: float = 1.0
    rollout_temperature: float = 0.85
    expansion_candidate_pool: int = 16
    rollout_candidate_pool: int = 4
    widening_score_batch: int = 4
    widening_score_batch_max: int = 16
    widening_score_batch_scale: float = 1.0
    widening_score_batch_beta: float = 0.5
    rollout_score_batch: int = 8
    productive_rollout_score_batch: int = 16
    broad_rollout_score_batch: int = 24
    broad_rollout_interval: int = 8
    seed: int = 0
    selection_survival_weight: float = 1.0
    selection_novelty_weight: float = 1.0
    compatibility_examples_path: Path | None = DEFAULT_EXAMPLES_PATH
    # Progressive widening parameters
    progressive_k0: int = 1
    progressive_alpha: float = 1.0
    progressive_beta: float = 0.5
    progressive_bucket_quota: int = 1
    discovery_epoch_value_decay: float = 0.25
    ucb_use_real_node_visits: bool = False
    ucb_min_action_visits: int = 0
    ucb_normalize_edge_survival_q: bool = False
    ucb_normalization_epsilon: float = 1e-12


@dataclass(frozen=True)
class MCTSValueComponents:
    """Three-part value used by selection and backpropagation."""

    escape: float = 0.0
    survival: float = 0.0
    novelty: float = 0.0

    def total(self, *, survival_weight: float, novelty_weight: float) -> float:
        return float(self.escape + survival_weight * self.survival + novelty_weight * self.novelty)


class ClassCompatibilityBank:
    """Compatibility counts for exact classes under the current block pattern."""

    def __init__(
        self,
        *,
        blocks: Sequence[Sequence[int]],
        examples_path: Path,
        max_cache_entries: int = 20000,
    ) -> None:
        self.blocks = [tuple(int(vertex) for vertex in block) for block in blocks]
        example_rows = parse_example_rows(examples_path)
        self.masks_by_class: dict[int, tuple[int, ...]] = {}
        for class_id, rows in example_rows.items():
            masks: list[int] = []
            for row in rows.values():
                support = support_mask_from_row(row)
                support_vertices = {index for index, value in enumerate(support) if int(value) == 1}
                block_mask = sum(
                    1 << index
                    for index, block in enumerate(self.blocks)
                    if all(int(vertex) in support_vertices for vertex in block)
                )
                masks.append(block_mask)
            self.masks_by_class[int(class_id)] = tuple(masks)
        self.max_cache_entries = int(max_cache_entries)
        self._compatibility_vectors: dict[int, tuple[int, ...]] = {}

    @classmethod
    def from_scorer(cls, scorer: ExpansionScorer, examples_path: Path | None) -> ClassCompatibilityBank | None:
        if examples_path is None:
            return None
        return cls(blocks=scorer.blocks, examples_path=examples_path)

    def compat_count(self, key: BlockKey, class_id: int) -> int:
        normalized = int(class_id)
        vector = self._compatibility_vector(key)
        return vector[normalized] if 0 <= normalized < len(vector) else 0

    def total_compat_count(self, key: BlockKey, class_ids: Sequence[int]) -> int:
        vector = self._compatibility_vector(key)
        return sum(
            vector[class_id]
            for class_id in (int(value) for value in class_ids)
            if 0 <= class_id < len(vector)
        )

    def active_signature(self, key: BlockKey, class_ids: Sequence[int]) -> tuple[int, ...]:
        vector = self._compatibility_vector(key)
        return tuple(
            class_id
            for class_id in sorted(int(value) for value in class_ids)
            if 0 <= class_id < len(vector) and vector[class_id] > 0
        )

    def clear_cache(self) -> None:
        self._compatibility_vectors.clear()

    def _compatibility_vector(self, key: BlockKey) -> tuple[int, ...]:
        selected_word = sum(
            1 << index
            for index, selected in enumerate(key)
            if int(selected)
        )
        cached = self._compatibility_vectors.get(selected_word)
        if cached is not None:
            return cached

        max_class_id = max(self.masks_by_class, default=0)
        values = [0] * (max_class_id + 1)
        for class_id, masks in self.masks_by_class.items():
            values[class_id] = sum(
                1
                for mask in masks
                if selected_word & ~mask == 0
            )
        vector = tuple(values)
        self._compatibility_vectors[selected_word] = vector
        if self.max_cache_entries > 0 and len(self._compatibility_vectors) > self.max_cache_entries:
            self._compatibility_vectors.pop(next(iter(self._compatibility_vectors)))
        return vector

@dataclass
class TerminalHit:
    """One terminal candidate encountered during MCTS."""

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


@dataclass(frozen=True)
class ExactClassDiscovery:
    """First time an exact class was discovered in one MCTS run."""

    class_id: int
    label: str
    iteration: int
    depth: int
    score: float
    rank: int
    path: list[int]
    chosen_blocks: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "iteration": self.iteration,
            "depth": self.depth,
            "score": self.score,
            "rank": self.rank,
            "path": list(self.path),
            "chosen_blocks": list(self.chosen_blocks),
        }


@dataclass
class MCTSResult:
    """Compact result of one zero-start MCTS run."""

    best: TerminalHit | None
    terminal_bests: dict[str, TerminalHit] = field(default_factory=dict)
    exact_discoveries: list[ExactClassDiscovery] = field(default_factory=list)
    encountered_label_counts: Counter[str] = field(default_factory=Counter)
    iterations_completed: int = 0
    nodes_created: int = 0
    root_visits: int = 0

    @property
    def exact_labels(self) -> list[str]:
        return sorted(label for label in self.terminal_bests if label.startswith("exact:"))

    @property
    def exact_class_ids(self) -> list[int]:
        return sorted(
            class_id
            for class_id in (exact_class_id(item.label) for item in self.exact_discoveries)
            if class_id is not None
        )

    @property
    def first_exact_discovery_by_class(self) -> dict[int, ExactClassDiscovery]:
        first: dict[int, ExactClassDiscovery] = {}
        for discovery in self.exact_discoveries:
            class_id = discovery.class_id
            if class_id not in first:
                first[class_id] = discovery
        return first

    def to_dict(self) -> dict[str, object]:
        return {
            "best": None if self.best is None else self.best.to_dict(),
            "terminal_bests": {
                label: hit.to_dict()
                for label, hit in sorted(self.terminal_bests.items())
            },
            "exact_discoveries": [discovery.to_dict() for discovery in self.exact_discoveries],
            "encountered_label_counts": dict(sorted(self.encountered_label_counts.items())),
            "exact_labels": self.exact_labels,
            "exact_class_ids": self.exact_class_ids,
            "iterations_completed": self.iterations_completed,
            "nodes_created": self.nodes_created,
            "root_visits": self.root_visits,
        }


@dataclass
class MCTSNode:
    key: BlockKey
    path: list[int]
    rank: int
    parent: MCTSNode | None
    action_from_parent: int | None
    visits: int = 0
    value_visits: float = 0.0
    value_sum: float = 0.0
    escape_sum: float = 0.0
    survival_sum: float = 0.0
    novelty_sum: float = 0.0
    prior: float = 1.0
    terminal: FacetLabel | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    actions_initialized: bool = False
    unexpanded_actions: list[int] = field(default_factory=list)
    action_priors: dict[int, float] = field(default_factory=dict)
    action_scores: dict[int, ExpansionScore] = field(default_factory=dict)
    action_families: dict[int, tuple[int, ...]] = field(default_factory=dict)
    expansion_scores: dict[int, float] = field(default_factory=dict)
    expansion_score_details: dict[int, dict[str, object]] = field(default_factory=dict)
    expansion_prior_source: str = ""
    legacy_prior_override: bool = False
    expansion_compatibility_epoch: int = -1
    canonical_key: BlockKey | None = None
    creation_index: int = 0
    arrival_paths: list[tuple[int, ...]] = field(default_factory=list)
    compatibility_counts: dict[int, int] = field(default_factory=dict)
    compatibility_epoch: int = -1
    child_symmetry_maps: dict[int, tuple[int, ...]] = field(default_factory=dict)
    child_closure_blocks: dict[int, tuple[int, ...]] = field(default_factory=dict)
    edge_value_visits: dict[int, float] = field(default_factory=dict)
    edge_value_sums: dict[int, float] = field(default_factory=dict)
    edge_escape_sums: dict[int, float] = field(default_factory=dict)
    edge_survival_sums: dict[int, float] = field(default_factory=dict)
    edge_novelty_sums: dict[int, float] = field(default_factory=dict)
    selected_edge_action: int | None = None
    # rotation pointer for bucket selection when progressively expanding
    next_bucket: int = 0
    bucket_expansion_counts: list[int] = field(default_factory=lambda: [0] * 6)

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.value_visits if self.value_visits > 0 else 0.0

    @property
    def escape_q_value(self) -> float:
        return self.escape_sum / self.value_visits if self.value_visits > 0 else 0.0

    @property
    def survival_q_value(self) -> float:
        return self.survival_sum / self.value_visits if self.value_visits > 0 else 0.0

    @property
    def novelty_q_value(self) -> float:
        return self.novelty_sum / self.value_visits if self.value_visits > 0 else 0.0

    def refresh_compatibility(
        self,
        scorer: ExpansionScorer,
        class_ids: Sequence[int],
        discovery_epoch: int,
    ) -> dict[int, int]:
        """Refresh newly discovered class counts and retain the node-local cache."""
        requested = {int(class_id) for class_id in class_ids}
        missing = requested.difference(self.compatibility_counts)
        if self.compatibility_epoch != int(discovery_epoch) or missing:
            counter = getattr(scorer, "compatibility_counts", None)
            if callable(counter):
                self.compatibility_counts.update(counter(self.key, missing or requested))
            self.compatibility_epoch = int(discovery_epoch)
        return {class_id: int(self.compatibility_counts.get(class_id, 0)) for class_id in sorted(requested)}


def run_mcts_search(
    scorer: ExpansionScorer,
    *,
    config: MCTSConfig | None = None,
    start_key: BlockKey | None = None,
) -> MCTSResult:
    """Run add-only Monte Carlo tree search from the empty block state."""
    cfg = MCTSConfig() if config is None else config
    if cfg.iterations <= 0:
        raise ValueError(f"iterations must be positive, got {cfg.iterations}")

    block_count = len(scorer.blocks)
    root_key = empty_key(block_count) if start_key is None else start_key
    rng = random.Random(cfg.seed)
    rollout_tie_rng = random.Random(cfg.seed + 2_147_483_647)
    compatibility_bank = ClassCompatibilityBank.from_scorer(scorer, cfg.compatibility_examples_path)

    nodes: dict[BlockKey, MCTSNode] = {}
    root = _get_or_create_node(nodes, scorer, root_key, path=[])
    terminal_bests: dict[str, TerminalHit] = {}
    exact_discoveries: list[ExactClassDiscovery] = []
    discovered_exact_classes: set[int] = set()
    seen_signatures: Counter[tuple[int, ...]] = Counter()
    encountered = Counter()
    best: TerminalHit | None = None
    iterations_completed = 0

    def record_exact_discovery(
        terminal: FacetLabel,
        *,
        score: float,
        iteration_index: int,
        depth: int,
        key: BlockKey,
        path: Sequence[int],
        rank: int,
    ) -> None:
        class_id = exact_class_id(terminal.label)
        if class_id is None or class_id in discovered_exact_classes:
            return
        discovered_exact_classes.add(class_id)
        scorer.rare_target_classes.discard(class_id)
        exact_discoveries.append(
            ExactClassDiscovery(
                class_id=class_id,
                label=terminal.label,
                iteration=iteration_index,
                depth=depth,
                score=float(score),
                rank=int(rank),
                path=list(path),
                chosen_blocks=selected_blocks(key),
            )
        )

    def record_signature(key: BlockKey) -> None:
        if compatibility_bank is None:
            return
        signature = compatibility_bank.active_signature(key, discovered_exact_classes)
        seen_signatures[signature] += 1

    def node_components(key: BlockKey, *, survival: float = 0.0) -> MCTSValueComponents:
        if compatibility_bank is None or not discovered_exact_classes:
            return MCTSValueComponents(escape=0.0, survival=float(survival), novelty=0.0)
        active_classes = sorted(discovered_exact_classes)
        current_total = compatibility_bank.total_compat_count(key, active_classes)
        escape = 0.0 if current_total <= 0 else 1.0 / (1.0 + float(current_total))
        signature = compatibility_bank.active_signature(key, active_classes)
        novelty = 1.0 / (1.0 + float(seen_signatures[signature]))
        return MCTSValueComponents(escape=float(escape), survival=float(survival), novelty=float(novelty))

    record_signature(root.key)

    for _iteration in range(cfg.iterations):
        iterations_completed += 1
        iteration_index = iterations_completed
        path_nodes: list[MCTSNode] = [root]
        node = root

        while True:
            if node.is_terminal:
                terminal_score = float(scorer.terminal_score(node.key))
                if node.terminal is not None and node.terminal.is_exact:
                    record_exact_discovery(
                        node.terminal,
                        score=terminal_score,
                        iteration_index=iteration_index,
                        depth=len(node.path),
                        key=node.key,
                        path=node.path,
                        rank=node.rank,
                    )
                node_value = node_components(node.key, survival=terminal_score)
                best = _register_hit(
                    node,
                    terminal_score,
                    terminal_bests=terminal_bests,
                    encountered=encountered,
                    scorer=scorer,
                    best=best,
                )
                _backpropagate_components(
                    path_nodes,
                    node_value,
                    survival_weight=cfg.selection_survival_weight,
                    novelty_weight=cfg.selection_novelty_weight,
                )
                break

            if node.rank >= 25:
                terminal = scorer.terminal_label(node.key)
                terminal_score = float(scorer.terminal_score(node.key))
                if terminal.is_exact:
                    record_exact_discovery(
                        terminal,
                        score=terminal_score,
                        iteration_index=iteration_index,
                        depth=len(node.path),
                        key=node.key,
                        path=node.path,
                        rank=node.rank,
                    )
                node_value = node_components(node.key, survival=terminal_score)
                best = _register_hit(
                    node,
                    terminal_score,
                    terminal=terminal,
                    terminal_bests=terminal_bests,
                    encountered=encountered,
                    scorer=scorer,
                    best=best,
                )
                _backpropagate_components(
                    path_nodes,
                    node_value,
                    survival_weight=cfg.selection_survival_weight,
                    novelty_weight=cfg.selection_novelty_weight,
                )
                break

            if not node.actions_initialized:
                _prepare_actions(node, scorer, cfg)

            if len(node.children) < _progressive_child_limit(node, cfg) and node.unexpanded_actions:
                action = _select_progressive_action(
                    node,
                    rng,
                    scorer=scorer,
                    cfg=cfg,
                    compatibility_bank=compatibility_bank,
                    discovered_exact_classes=discovered_exact_classes,
                    seen_signatures=seen_signatures,
                )
                child_key = add_block(node.key, action)
                child_path = [*node.path, int(action)]
                child = _get_or_create_node(nodes, scorer, child_key, path=child_path, parent=node, action=action)
                child.prior = node.action_priors.get(action, child.prior)
                node.children[action] = child
                node.selected_edge_action = int(action)
                node.unexpanded_actions = [item for item in node.unexpanded_actions if item != action]
                path_nodes.append(child)

                step = node.action_scores[action]
                if child.is_terminal or child.rank >= 25:
                    terminal = child.terminal if child.terminal is not None else scorer.terminal_label(child.key)
                    terminal_score = float(step.score)
                    child_value = node_components(child.key, survival=terminal_score)
                    if terminal.is_exact:
                        record_exact_discovery(
                            terminal,
                            score=terminal_score,
                            iteration_index=iteration_index,
                            depth=len(child.path),
                            key=child.key,
                            path=child.path,
                            rank=child.rank,
                        )
                    best = _register_hit(
                        child,
                        terminal_score,
                        terminal=terminal,
                        terminal_bests=terminal_bests,
                        encountered=encountered,
                        scorer=scorer,
                        best=best,
                    )
                    record_signature(child.key)
                    _backpropagate_components(path_nodes, child_value, survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight)
                else:
                    _rollout_value, rollout_components, best = _rollout(
                        child,
                        scorer,
                        cfg,
                        rng,
                        terminal_bests,
                        encountered,
                        best,
                        record_exact_discovery,
                        compatibility_bank,
                        discovered_exact_classes,
                        seen_signatures,
                        iteration_index,
                        tie_rng=rollout_tie_rng,
                    )
                    child_components = MCTSValueComponents(
                        escape=rollout_components.escape,
                        survival=float(step.score) + rollout_components.survival,
                        novelty=rollout_components.novelty,
                    )
                    record_signature(child.key)
                    _backpropagate_components(
                        path_nodes,
                        child_components,
                        survival_weight=cfg.selection_survival_weight,
                        novelty_weight=cfg.selection_novelty_weight,
                )
                break

            _refresh_terminal_action_scores(
                node,
                scorer,
                cfg,
                terminal_score_fn=None,
            )
            next_node = _select_child(
                node,
                cfg.exploration_constant,
                rng=rng,
                survival_weight=cfg.selection_survival_weight,
                novelty_weight=cfg.selection_novelty_weight,
                use_real_node_visits=cfg.ucb_use_real_node_visits,
                min_action_visits=cfg.ucb_min_action_visits,
                normalize_edge_survival_q=cfg.ucb_normalize_edge_survival_q,
                normalization_epsilon=cfg.ucb_normalization_epsilon,
            )
            if next_node is None:
                components = _estimate_state_value(
                    node.key,
                    scorer,
                    cfg,
                    compatibility_bank=compatibility_bank,
                    discovered_exact_classes=discovered_exact_classes,
                    seen_signatures=seen_signatures,
                )
                _backpropagate_components(
                    path_nodes,
                    components,
                    survival_weight=cfg.selection_survival_weight,
                    novelty_weight=cfg.selection_novelty_weight,
                )
                break
            node = next_node
            path_nodes.append(node)

    return MCTSResult(
        best=best,
        terminal_bests=terminal_bests,
        exact_discoveries=exact_discoveries,
        encountered_label_counts=encountered,
        iterations_completed=iterations_completed,
        nodes_created=len(nodes),
        root_visits=root.visits,
    )


def _get_or_create_node(
    nodes: dict[BlockKey, MCTSNode],
    scorer: ExpansionScorer,
    key: BlockKey,
    *,
    path: list[int],
    parent: MCTSNode | None = None,
    action: int | None = None,
) -> MCTSNode:
    manager = getattr(scorer, "node_manager", None)
    if manager is not None:
        return manager.get_or_create(
            nodes,
            key,
            path=path,
            parent=parent,
            action=action,
        )
    node = nodes.get(key)
    if node is not None:
        if len(path) < len(node.path):
            node.path = list(path)
        return node

    rank = scorer.affine_rank(key)
    terminal = scorer.terminal_label(key) if rank >= 25 else None
    node = MCTSNode(
        key=key,
        path=list(path),
        rank=rank,
        parent=parent,
        action_from_parent=action,
        terminal=terminal,
    )
    nodes[key] = node
    return node


def _prepare_actions(
    node: MCTSNode,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    *,
    terminal_score_fn: TerminalScoreFn | None = None,
) -> None:
    if node.actions_initialized:
        return
    actions = unselected_blocks(node.key)
    family_builder = getattr(scorer, "representative_action_families", None)
    if callable(family_builder):
        node.action_families = dict(family_builder(node.key, actions))
        actions = sorted(node.action_families)
    else:
        node.action_families = {int(action): (int(action),) for action in actions}
    node.actions_initialized = True
    node.unexpanded_actions = list(actions)
    node.action_priors = {}
    node.action_scores = {}
    expansion_preparer = getattr(scorer, "prepare_expansion_priors", None)
    if callable(expansion_preparer):
        bucket_count = max(1, int(getattr(scorer, "expansion_bucket_count", 2)))
        node.bucket_expansion_counts = [0] * bucket_count
        node.next_bucket = 0
        expansion_preparer(node, cfg.prior_temperature)
    else:
        legacy_prior_preparer = getattr(scorer, "prepare_legacy_action_priors", None)
        if callable(legacy_prior_preparer):
            legacy_prior_preparer(node, cfg.prior_temperature)


def _progressive_child_limit(node: MCTSNode, cfg: MCTSConfig) -> int:
    """Allowed expanded children K(s) for progressive widening."""
    visits = float(max(1, node.visits))
    widening_limit = int(
        max(1, cfg.progressive_k0 + cfg.progressive_alpha * (visits ** cfg.progressive_beta))
    )
    bucket_quota = max(0, int(cfg.progressive_bucket_quota))
    if bucket_quota <= 0:
        return widening_limit

    action_count = len(node.children) + len(node.unexpanded_actions)
    quota_target = min(action_count, len(node.bucket_expansion_counts) * bucket_quota)
    quota_limit = min(quota_target, max(1, node.visits + 1))
    return max(widening_limit, quota_limit)


def _next_progressive_bucket(node: MCTSNode, cfg: MCTSConfig) -> int:
    bucket_count = len(node.bucket_expansion_counts)
    if bucket_count <= 0:
        return 0

    start = node.next_bucket % bucket_count
    quota = max(0, int(cfg.progressive_bucket_quota))
    bucket = start
    if quota > 0:
        for offset in range(bucket_count):
            candidate = (start + offset) % bucket_count
            if node.bucket_expansion_counts[candidate] < quota:
                bucket = candidate
                break

    node.bucket_expansion_counts[bucket] += 1
    node.next_bucket = (bucket + 1) % bucket_count
    return bucket


def _select_progressive_action(
    node: MCTSNode,
    rng: random.Random,
    *,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    compatibility_bank: ClassCompatibilityBank | None,
    discovered_exact_classes: set[int],
    seen_signatures: Counter[tuple[int, ...]],
    terminal_score_fn: TerminalScoreFn | None = None,
    adaptive_score_batch: bool = True,
) -> int:
    """Pick the next unexpanded action by round-robin widening buckets."""
    _refresh_terminal_action_scores(
        node,
        scorer,
        cfg,
        terminal_score_fn=terminal_score_fn,
    )
    candidates = list(node.unexpanded_actions)
    if not candidates:
        raise ValueError("cannot choose from an empty action list")

    expansion_preparer = getattr(scorer, "prepare_expansion_priors", None)
    if callable(expansion_preparer):
        expansion_preparer(node, cfg.prior_temperature)
    if node.expansion_scores:
        bucket = _next_progressive_bucket(node, cfg)
        uniform_only = bool(getattr(scorer, "uniform_expansion_buckets_only", False))
        prior_buckets = getattr(scorer, "prior_expansion_buckets", (0,))
        if bucket in prior_buckets and not uniform_only:
            weights = [max(0.0, float(node.action_priors.get(candidate, 0.0))) for candidate in candidates]
            total_weight = sum(weights)
            if total_weight <= 0.0:
                weights = [1.0 / len(candidates)] * len(candidates)
            else:
                weights = [value / total_weight for value in weights]
            decision_kind = f"{node.expansion_prior_source or 'compatibility_richness'}_prior"
        else:
            weights = [1.0 / len(candidates)] * len(candidates)
            decision_kind = "uniform_random"
        chosen_index = _weighted_choice_index(weights, rng)
        action = candidates[chosen_index]
        _ensure_action_scored(
            node,
            action,
            scorer,
            cfg,
            terminal_score_fn=terminal_score_fn,
        )
        _trace_emit(
            "progressive_choice",
            node_path=list(node.path),
            node_rank=node.rank,
            node_visits=node.visits,
            bucket=bucket,
            decision_kind=decision_kind,
            candidate_actions=list(candidates),
            representative_families={
                str(candidate): list(node.action_families.get(candidate, (candidate,)))
                for candidate in candidates
            },
            decision_values={str(k): node.expansion_scores[k] for k in candidates},
            expansion_details={str(k): node.expansion_score_details[k] for k in candidates},
            candidate_priors={str(k): weights[index] for index, k in enumerate(candidates)},
            chosen_action=action,
            chosen_score=float(node.action_scores[action].score),
            chosen_prior=float(node.action_priors.get(action, 0.0)),
        )
        return action

    expansion_preparer = getattr(scorer, "prepare_expansion_priors", None)
    if not callable(expansion_preparer):
        legacy_prior_preparer = getattr(scorer, "prepare_legacy_action_priors", None)
        if callable(legacy_prior_preparer):
            legacy_prior_preparer(node, cfg.prior_temperature)

    bucket = _next_progressive_bucket(node, cfg)
    active_classes = sorted(discovered_exact_classes)
    decision_values: dict[int, float] = {}
    decision_kind = "random"
    if bucket == 0 and node.legacy_prior_override:
        weights = [max(0.0, float(node.action_priors.get(action, 0.0))) for action in candidates]
        total_weight = sum(weights)
        weights = (
            [value / total_weight for value in weights]
            if total_weight > 0.0
            else [1.0 / len(candidates)] * len(candidates)
        )
        action = candidates[_weighted_choice_index(weights, rng)]
        decision_values = {candidate: float(node.action_priors.get(candidate, 0.0)) for candidate in candidates}
        decision_kind = f"{node.expansion_prior_source}_prior"
    elif bucket == 0:
        scored_candidates = _score_lazy_action_batch(
            node,
            candidates,
            scorer,
            cfg,
            rng,
            terminal_score_fn=terminal_score_fn,
            adaptive_score_batch=adaptive_score_batch,
        )
        decision_values = {item: float(node.action_scores[item].score) for item in scored_candidates}
        decision_kind = "structural_score"
        action = _best_action_random_tie(
            scored_candidates,
            decision_values,
            rng,
        )
    elif bucket == 1 and compatibility_bank is not None and active_classes:
        current_total = compatibility_bank.total_compat_count(node.key, active_classes)
        values = {
            action: float(
                current_total
                - compatibility_bank.total_compat_count(add_block(node.key, action), active_classes)
            )
            for action in candidates
        }
        decision_values = values
        decision_kind = "compatibility_escape_count"
        action = _best_action_random_tie(candidates, values, rng)
    elif bucket == 2 and compatibility_bank is not None and active_classes:
        current_active = {
            class_id
            for class_id in active_classes
            if compatibility_bank.compat_count(node.key, class_id) > 0
        }
        values = {
            action: float(
                sum(
                    1
                    for class_id in current_active
                    if compatibility_bank.compat_count(add_block(node.key, action), class_id) == 0
                )
            )
            for action in candidates
        }
        decision_values = values
        decision_kind = "classes_eliminated"
        action = _best_action_random_tie(candidates, values, rng)
    elif bucket == 3 and compatibility_bank is not None and active_classes:
        values = {}
        for action in candidates:
            signature = compatibility_bank.active_signature(add_block(node.key, action), active_classes)
            values[action] = 1.0 / (1.0 + float(seen_signatures[signature]))
        decision_values = values
        decision_kind = "signature_novelty"
        action = _best_action_random_tie(candidates, values, rng)
    elif bucket == 5:
        scored_candidates = _score_lazy_action_batch(
            node,
            candidates,
            scorer,
            cfg,
            rng,
            terminal_score_fn=terminal_score_fn,
            adaptive_score_batch=adaptive_score_batch,
        )
        current_total = 0
        if compatibility_bank is not None and active_classes:
            current_total = compatibility_bank.total_compat_count(node.key, active_classes)
        values = {}
        for candidate in scored_candidates:
            removed = 0.0
            if compatibility_bank is not None and active_classes:
                removed = float(
                    current_total
                    - compatibility_bank.total_compat_count(add_block(node.key, candidate), active_classes)
                )
            values[candidate] = float(node.action_scores[candidate].score) / (1.0 + max(0.0, removed))
        decision_values = values
        decision_kind = "score_over_compatibility_removal"
        action = _best_action_random_tie(scored_candidates, values, rng)
    else:
        action = rng.choice(candidates)
        decision_kind = "uniform_random"

    _ensure_action_scored(
        node,
        action,
        scorer,
        cfg,
        terminal_score_fn=terminal_score_fn,
    )
    _trace_emit(
        "progressive_choice",
        node_path=list(node.path),
        node_rank=node.rank,
        node_visits=node.visits,
        bucket=bucket,
        decision_kind=decision_kind,
        candidate_actions=list(candidates),
        representative_families={
            str(candidate): list(node.action_families.get(candidate, (candidate,)))
            for candidate in candidates
        },
        decision_values={str(k): v for k, v in decision_values.items()},
        candidate_priors={
            str(candidate): float(node.action_priors.get(candidate, 0.0))
            for candidate in candidates
        },
        expansion_details={
            str(candidate): node.expansion_score_details.get(candidate, {})
            for candidate in candidates
        },
        chosen_action=action,
        chosen_score=float(node.action_scores[action].score),
        chosen_prior=float(node.action_priors.get(action, 0.0)),
    )
    return action


def _score_lazy_action_batch(
    node: MCTSNode,
    candidates: Sequence[int],
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    rng: random.Random,
    *,
    terminal_score_fn: TerminalScoreFn | None,
    adaptive_score_batch: bool,
) -> list[int]:
    unscored = [action for action in candidates if action not in node.action_scores]
    batch_size = _widening_score_batch(
        node,
        cfg,
        adaptive=adaptive_score_batch,
    )
    if unscored:
        sampled = (
            list(unscored)
            if len(unscored) <= batch_size
            else rng.sample(unscored, batch_size)
        )
        for action in sampled:
            node.action_scores[action] = _score_action_with_terminal_reward(
                scorer,
                node.key,
                action,
                terminal_score_fn=terminal_score_fn,
            )
        _refresh_action_priors(node, cfg)
    scored_candidates = [action for action in candidates if action in node.action_scores]
    if scored_candidates:
        return scored_candidates
    action = rng.choice(list(candidates))
    _ensure_action_scored(
        node,
        action,
        scorer,
        cfg,
        terminal_score_fn=terminal_score_fn,
    )
    return [action]


def _widening_score_batch(
    node: MCTSNode,
    cfg: MCTSConfig,
    *,
    adaptive: bool,
) -> int:
    base = max(1, int(cfg.widening_score_batch))
    if not adaptive:
        return base
    maximum = max(base, int(cfg.widening_score_batch_max))
    growth = float(cfg.widening_score_batch_scale) * (
        float(max(0, node.visits)) ** float(cfg.widening_score_batch_beta)
    )
    return min(maximum, base + int(growth))


def _ensure_action_scored(
    node: MCTSNode,
    action: int,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    *,
    terminal_score_fn: TerminalScoreFn | None,
) -> ExpansionScore:
    if action not in node.action_scores:
        node.action_scores[action] = _score_action_with_terminal_reward(
            scorer,
            node.key,
            action,
            terminal_score_fn=terminal_score_fn,
        )
        _refresh_action_priors(node, cfg)
    return node.action_scores[action]


def _refresh_action_priors(node: MCTSNode, cfg: MCTSConfig) -> None:
    if node.expansion_scores or node.legacy_prior_override:
        return
    actions = list(node.action_scores)
    if not actions:
        return
    priors = _softmax(
        [node.action_scores[action].score for action in actions],
        temperature=cfg.prior_temperature,
    )
    node.action_priors = {action: prior for action, prior in zip(actions, priors)}
    for action, child in node.children.items():
        child.prior = node.action_priors.get(action, child.prior)


def _refresh_terminal_action_scores(
    node: MCTSNode,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    *,
    terminal_score_fn: TerminalScoreFn | None,
) -> None:
    """Refresh only mutable terminal rewards; geometry remains cached."""
    changed = False
    for action, score_item in list(node.action_scores.items()):
        if score_item.terminal is None:
            continue
        current_score = _resolve_terminal_score(
            scorer,
            score_item.key,
            score_item.terminal,
            terminal_score_fn=terminal_score_fn,
        )
        if current_score == float(score_item.score):
            continue
        node.action_scores[action] = replace(score_item, score=current_score)
        changed = True
    if changed:
        _refresh_action_priors(node, cfg)
    expansion_preparer = getattr(scorer, "prepare_expansion_priors", None)
    if not callable(expansion_preparer):
        legacy_prior_preparer = getattr(scorer, "prepare_legacy_action_priors", None)
        if callable(legacy_prior_preparer):
            legacy_prior_preparer(node, cfg.prior_temperature)


def _refresh_discovery_dependent_action_scores(
    node: MCTSNode,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    *,
    terminal_score_fn: TerminalScoreFn | None,
) -> None:
    """Re-score actions whose value depends on the active undiscovered classes."""
    changed = False
    for action, score_item in list(node.action_scores.items()):
        if score_item.terminal is None and score_item.new_rank != 24:
            continue
        current = _score_action_with_terminal_reward(
            scorer,
            node.key,
            action,
            terminal_score_fn=terminal_score_fn,
        )
        if current == score_item:
            continue
        node.action_scores[action] = current
        changed = True
    if changed:
        _refresh_action_priors(node, cfg)
    expansion_preparer = getattr(scorer, "prepare_expansion_priors", None)
    if not callable(expansion_preparer):
        legacy_prior_preparer = getattr(scorer, "prepare_legacy_action_priors", None)
        if callable(legacy_prior_preparer):
            legacy_prior_preparer(node, cfg.prior_temperature)


def _best_action_random_tie(
    candidates: Sequence[int],
    values: dict[int, float],
    rng: random.Random,
) -> int:
    best_value = max(values[action] for action in candidates)
    tied = [action for action in candidates if values[action] == best_value]
    return rng.choice(tied)


def _select_child(
    node: MCTSNode,
    exploration_constant: float,
    *,
    rng: random.Random | None = None,
    survival_weight: float = 1.0,
    novelty_weight: float = 1.0,
    use_real_node_visits: bool = False,
    min_action_visits: int = 0,
    normalize_edge_survival_q: bool = False,
    normalization_epsilon: float = 1e-12,
) -> MCTSNode | None:
    if not node.children:
        return None

    parent_visits = max(float(node.visits if use_real_node_visits else node.value_visits), 1.0)
    raw_survival_q: dict[int, float] = {}
    for action in node.children:
        edge_visits = float(node.edge_value_visits.get(int(action), 0.0))
        raw_survival_q[int(action)] = (
            float(node.edge_survival_sums.get(int(action), 0.0)) / edge_visits
            if edge_visits > 0.0
            else 0.0
        )
    q_min = min(raw_survival_q.values())
    q_max = max(raw_survival_q.values())
    if q_max == q_min:
        normalized_survival_q = {action: 0.5 for action in raw_survival_q}
    else:
        denominator = q_max - q_min + float(normalization_epsilon)
        normalized_survival_q = {
            action: (value - q_min) / denominator
            for action, value in raw_survival_q.items()
        }

    under_visited = [
        int(action)
        for action, child in node.children.items()
        if child.visits < int(min_action_visits)
    ]
    selection_phase = "minimum_visits" if under_visited else "normalized_ucb" if normalize_edge_survival_q else "ucb"
    forced_action: int | None = None
    if under_visited:
        lowest_visits = min(node.children[action].visits for action in under_visited)
        tied = [action for action in under_visited if node.children[action].visits == lowest_visits]
        weights = [max(0.0, float(node.action_priors.get(action, 0.0))) for action in tied]
        chooser = rng if rng is not None else random.Random(0)
        forced_action = chooser.choices(tied, weights=weights, k=1)[0] if sum(weights) > 0.0 else chooser.choice(tied)

    best_child: MCTSNode | None = None
    best_value = float("-inf")
    selected_action: int | None = None
    diagnostics: list[dict[str, object]] = []
    for action, child in node.children.items():
        edge_visits = float(node.edge_value_visits.get(int(action), 0.0))
        edge_survival_q = raw_survival_q[int(action)]
        edge_escape_q = (
            float(node.edge_escape_sums.get(int(action), 0.0)) / edge_visits
            if edge_visits > 0.0 else 0.0
        )
        edge_novelty_q = (
            float(node.edge_novelty_sums.get(int(action), 0.0)) / edge_visits
            if edge_visits > 0.0 else 0.0
        )
        prior = float(node.action_priors.get(int(action), 0.0))
        normalized_q = normalized_survival_q[int(action)]
        exploitation = (
            normalized_q
            if normalize_edge_survival_q
            else edge_escape_q + survival_weight * edge_survival_q + novelty_weight * edge_novelty_q
        )
        exploration_visits = float(child.visits) if use_real_node_visits else edge_visits
        exploration = (
            exploration_constant
            * prior
            * (parent_visits ** 0.5)
            / (1.0 + exploration_visits)
        )
        score = exploitation + exploration
        diagnostics.append({
            "action": int(action),
            "visits": child.visits,
            "value_visits": child.value_visits,
            "prior": prior,
            "edge_value_visits": edge_visits,
            "ucb_parent_visits": parent_visits,
            "ucb_child_visits": exploration_visits,
            "ucb_visit_source": "real_node_visits" if use_real_node_visits else "value_visits",
            "edge_survival_q": edge_survival_q,
            "raw_q": edge_survival_q,
            "normalized_q": normalized_q,
            "edge_escape_q": edge_escape_q,
            "edge_novelty_q": edge_novelty_q,
            "escape_q": child.escape_q_value,
            "survival_q": child.survival_q_value,
            "novelty_q": child.novelty_q_value,
            "exploitation": exploitation,
            "exploration": exploration,
            "ucb": score,
            "selection_phase": selection_phase,
        })
        if forced_action is None and score > best_value:
            best_value = score
            best_child = child
            selected_action = int(action)
    if forced_action is not None:
        selected_action = forced_action
    if selected_action is not None:
        best_child = node.children[selected_action]
        node.selected_edge_action = selected_action
    _trace_emit(
        "ucb_choice",
        node_path=list(node.path),
        parent_value_visits=node.value_visits,
        selection_phase=selection_phase,
        min_action_visits=int(min_action_visits),
        q_min=q_min,
        q_max=q_max,
        candidates=diagnostics,
        chosen_action=selected_action,
    )
    return best_child


def _rollout(
    node: MCTSNode,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    rng: random.Random,
    terminal_bests: dict[str, TerminalHit],
    encountered: Counter[str],
    best: TerminalHit | None,
    record_exact_discovery: Callable[..., None],
    compatibility_bank: ClassCompatibilityBank | None,
    discovered_exact_classes: set[int],
    seen_signatures: Counter[tuple[int, ...]],
    iteration_index: int,
    *,
    tie_rng: random.Random | None = None,
    terminal_score_fn: TerminalScoreFn | None = None,
    global_discovered_label_counts: Counter[str] | None = None,
    global_exact_hit_callback: ExactHitCallback | None = None,
    rollout_score_batch: int | None = None,
) -> tuple[float, MCTSValueComponents, TerminalHit | None]:
    current_key = node.key
    current_path = list(node.path)
    current_rank = node.rank
    total = MCTSValueComponents()
    discount = 1.0

    def score_components(key: BlockKey, *, survival: float) -> MCTSValueComponents:
        if compatibility_bank is None or not discovered_exact_classes:
            return MCTSValueComponents(escape=0.0, survival=float(survival), novelty=0.0)
        active_classes = sorted(discovered_exact_classes)
        current_total = compatibility_bank.total_compat_count(key, active_classes)
        escape = 0.0 if current_total <= 0 else 1.0 / (1.0 + float(current_total))
        signature = compatibility_bank.active_signature(key, active_classes)
        novelty = 1.0 / (1.0 + float(seen_signatures[signature]))
        return MCTSValueComponents(escape=float(escape), survival=float(survival), novelty=float(novelty))

    for _depth in range(max(0, cfg.max_depth - len(current_path))):
        if current_rank >= 25:
            terminal = scorer.terminal_label(current_key)
            terminal_score = _resolve_terminal_score(
                scorer,
                current_key,
                terminal,
                terminal_score_fn=terminal_score_fn,
            )
            _trace_emit(
                "terminal_observation",
                source="rollout_rank25",
                path=list(current_path),
                rank=current_rank,
                label=terminal.label,
                score=terminal_score,
            )
            if terminal.is_exact:
                record_exact_discovery(
                    terminal,
                    score=terminal_score,
                    iteration_index=iteration_index,
                    depth=len(current_path),
                    key=current_key,
                    path=current_path,
                    rank=current_rank,
                )
            components = score_components(current_key, survival=terminal_score)
            best = _register_hit(
                node,
                terminal_score,
                terminal=terminal,
                terminal_bests=terminal_bests,
                encountered=encountered,
                scorer=scorer,
                best=best,
                global_discovered_label_counts=global_discovered_label_counts,
                global_exact_hit_callback=global_exact_hit_callback,
            )
            total = MCTSValueComponents(
                escape=total.escape + discount * components.escape,
                survival=total.survival + discount * components.survival,
                novelty=total.novelty + discount * components.novelty,
            )
            return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best

        actions = unselected_blocks(current_key)
        if not actions:
            estimate = _estimate_state_value(
                current_key,
                scorer,
                cfg,
                compatibility_bank=compatibility_bank,
                discovered_exact_classes=discovered_exact_classes,
                seen_signatures=seen_signatures,
                terminal_score_fn=terminal_score_fn,
            )
            total = MCTSValueComponents(
                escape=total.escape + discount * estimate.escape,
                survival=total.survival + discount * estimate.survival,
                novelty=total.novelty + discount * estimate.novelty,
            )
            return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best

        requested_score_batch = (
            int(cfg.rollout_score_batch)
            if rollout_score_batch is None
            else int(rollout_score_batch)
        )
        score_batch = max(int(cfg.rollout_candidate_pool), requested_score_batch, 1)
        candidate_actions = (
            list(actions)
            if len(actions) <= score_batch
            else rng.sample(actions, score_batch)
        )
        scored = [
            _score_action_with_terminal_reward(
                scorer,
                current_key,
                action,
                terminal_score_fn=terminal_score_fn,
            )
            for action in candidate_actions
        ]
        limit = max(1, min(cfg.rollout_candidate_pool, len(scored)))
        pool, pool_trace = _rollout_top_k_pool(
            scored,
            limit,
            tie_rng=tie_rng if tie_rng is not None else rng,
        )
        weights = _softmax([item.score for item in pool], temperature=cfg.rollout_temperature)
        chosen_index = _weighted_choice_index(weights, rng)
        step = pool[chosen_index]
        _trace_emit(
            "rollout_choice",
            rollout_depth=len(current_path),
            current_path=list(current_path),
            current_rank=current_rank,
            candidate_actions=list(candidate_actions),
            candidate_scores=[
                {"action": item.action, "score": float(item.score)}
                for item in scored
            ],
            scored_candidates=[
                {
                    "action": item.action,
                    "score": float(item.score),
                    "new_rank": item.new_rank,
                    "terminal": None if item.terminal is None else item.terminal.label,
                }
                for item in sorted(scored, key=lambda candidate: candidate.score, reverse=True)
            ],
            boundary_score=pool_trace["boundary_score"],
            strictly_above_boundary_actions=pool_trace["strictly_above_boundary_actions"],
            boundary_tied_actions=pool_trace["boundary_tied_actions"],
            boundary_selected_actions=pool_trace["boundary_selected_actions"],
            pool=[item.action for item in pool],
            weights=list(weights),
            chosen_index=chosen_index,
            chosen_action=step.action,
            chosen_score=float(step.score),
            chosen_new_rank=step.new_rank,
            chosen_terminal=None if step.terminal is None else step.terminal.label,
        )
        if step.terminal is not None:
            _trace_emit(
                "terminal_observation",
                source="rollout_step",
                path=[*current_path, int(step.action)],
                rank=step.new_rank,
                label=step.terminal.label,
                score=float(step.score),
            )
        if step.terminal is not None and step.terminal.is_exact:
            record_exact_discovery(
                step.terminal,
                score=float(step.score),
                iteration_index=iteration_index,
                depth=len(current_path) + 1,
                key=step.key,
                path=[*current_path, int(step.action)],
                rank=step.new_rank,
            )
        step_components = score_components(step.key, survival=float(step.score))
        total = MCTSValueComponents(
            escape=total.escape + discount * step_components.escape,
            survival=total.survival + discount * step_components.survival,
            novelty=total.novelty + discount * step_components.novelty,
        )
        discount *= cfg.discount

        current_key = step.key
        current_path = [*current_path, int(step.action)]
        current_rank = step.new_rank

        if step.terminal is not None:
            if step.terminal.is_exact:
                record_exact_discovery(
                    step.terminal,
                    score=float(step.score),
                    iteration_index=iteration_index,
                    depth=len(current_path),
                    key=current_key,
                    path=current_path,
                    rank=current_rank,
                )
                if compatibility_bank is not None:
                    seen_signatures[compatibility_bank.active_signature(current_key, discovered_exact_classes)] += 1
            best = _register_hit(
                node,
                float(step.score),
                terminal=step.terminal,
                terminal_bests=terminal_bests,
                encountered=encountered,
                scorer=scorer,
                best=best,
                key=current_key,
                path=current_path,
                global_discovered_label_counts=global_discovered_label_counts,
                global_exact_hit_callback=global_exact_hit_callback,
            )
            return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best

    estimate = _estimate_state_value(
        current_key,
        scorer,
        cfg,
        compatibility_bank=compatibility_bank,
        discovered_exact_classes=discovered_exact_classes,
        seen_signatures=seen_signatures,
        terminal_score_fn=terminal_score_fn,
    )
    total = MCTSValueComponents(
        escape=total.escape + discount * estimate.escape,
        survival=total.survival + discount * estimate.survival,
        novelty=total.novelty + discount * estimate.novelty,
    )
    return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best


def _rollout_top_k_pool(
    scored: Sequence[ExpansionScore],
    limit: int,
    *,
    tie_rng: random.Random,
) -> tuple[list[ExpansionScore], dict[str, object]]:
    """Select top-k scores with uniform sampling at a tied cutoff."""

    normalized_limit = max(1, min(int(limit), len(scored)))
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    boundary_score = float(ranked[normalized_limit - 1].score)
    strictly_above = [item for item in ranked if float(item.score) > boundary_score]
    boundary_tied = [item for item in scored if float(item.score) == boundary_score]
    remaining_slots = normalized_limit - len(strictly_above)
    if remaining_slots >= len(boundary_tied):
        boundary_selected = list(boundary_tied)
    else:
        boundary_selected = tie_rng.sample(boundary_tied, remaining_slots)
    pool = strictly_above + boundary_selected
    return pool, {
        "boundary_score": boundary_score,
        "strictly_above_boundary_actions": [item.action for item in strictly_above],
        "boundary_tied_actions": [item.action for item in boundary_tied],
        "boundary_selected_actions": [item.action for item in boundary_selected],
    }


def _estimate_state_value(
    key: BlockKey,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    *,
    compatibility_bank: ClassCompatibilityBank | None = None,
    discovered_exact_classes: set[int] | None = None,
    seen_signatures: Counter[tuple[int, ...]] | None = None,
    terminal_score_fn: TerminalScoreFn | None = None,
) -> MCTSValueComponents:
    rank = scorer.affine_rank(key)
    if rank >= 25:
        terminal = scorer.terminal_label(key)
        survival = _resolve_terminal_score(
            scorer,
            key,
            terminal,
            terminal_score_fn=terminal_score_fn,
        )
        if compatibility_bank is None or not discovered_exact_classes:
            return MCTSValueComponents(survival=survival)
        active_classes = sorted(discovered_exact_classes)
        escape = 0.0 if compatibility_bank.total_compat_count(key, active_classes) <= 0 else 1.0 / (1.0 + float(compatibility_bank.total_compat_count(key, active_classes)))
        signature = compatibility_bank.active_signature(key, active_classes)
        novelty = 1.0 if seen_signatures is None else 1.0 / (1.0 + float(seen_signatures[signature]))
        return MCTSValueComponents(escape=float(escape), survival=survival, novelty=float(novelty))

    actions = unselected_blocks(key)
    if not actions:
        return MCTSValueComponents(survival=-100.0)

    score_limit = max(1, min(int(cfg.widening_score_batch), len(actions)))
    scored = [
        _score_action_with_terminal_reward(
            scorer,
            key,
            action,
            terminal_score_fn=terminal_score_fn,
        )
        for action in actions[:score_limit]
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    limit = max(1, min(cfg.expansion_candidate_pool, len(scored)))
    survival = float(sum(item.score for item in scored[:limit]) / limit)
    if compatibility_bank is None or not discovered_exact_classes:
        return MCTSValueComponents(survival=survival)
    active_classes = sorted(discovered_exact_classes)
    escape = 0.0 if compatibility_bank.total_compat_count(key, active_classes) <= 0 else 1.0 / (1.0 + float(compatibility_bank.total_compat_count(key, active_classes)))
    signature = compatibility_bank.active_signature(key, active_classes)
    novelty = 1.0 if seen_signatures is None else 1.0 / (1.0 + float(seen_signatures[signature]))
    return MCTSValueComponents(escape=float(escape), survival=survival, novelty=float(novelty))


def _backpropagate_components(
    path_nodes: Sequence[MCTSNode],
    value: MCTSValueComponents,
    *,
    survival_weight: float,
    novelty_weight: float,
) -> None:
    total_value = value.total(survival_weight=survival_weight, novelty_weight=novelty_weight)
    before = [
        {
            "path": list(node.path),
            "visits": node.visits,
            "value_visits": node.value_visits,
            "escape_q": node.escape_q_value,
            "survival_q": node.survival_q_value,
            "novelty_q": node.novelty_q_value,
        }
        for node in path_nodes
    ]
    for node in path_nodes:
        node.visits += 1
        node.value_visits += 1.0
        node.value_sum += float(total_value)
        node.escape_sum += float(value.escape)
        node.survival_sum += float(value.survival)
        node.novelty_sum += float(value.novelty)
    edge_updates = []
    for parent, child in zip(path_nodes, path_nodes[1:]):
        action = parent.selected_edge_action
        if action is None or parent.children.get(int(action)) is not child:
            raise RuntimeError("missing or inconsistent traversed edge during backpropagation")
        normalized = int(action)
        parent.edge_value_visits[normalized] = parent.edge_value_visits.get(normalized, 0.0) + 1.0
        parent.edge_value_sums[normalized] = parent.edge_value_sums.get(normalized, 0.0) + float(total_value)
        parent.edge_escape_sums[normalized] = parent.edge_escape_sums.get(normalized, 0.0) + float(value.escape)
        parent.edge_survival_sums[normalized] = parent.edge_survival_sums.get(normalized, 0.0) + float(value.survival)
        parent.edge_novelty_sums[normalized] = parent.edge_novelty_sums.get(normalized, 0.0) + float(value.novelty)
        edge_updates.append({
            "parent_path": list(parent.path),
            "action": normalized,
            "child_path": list(child.path),
            "edge_value_visits": parent.edge_value_visits[normalized],
            "edge_survival_q": parent.edge_survival_sums[normalized] / parent.edge_value_visits[normalized],
        })
        parent.selected_edge_action = None
    _trace_emit(
        "backpropagation",
        value={"escape": value.escape, "survival": value.survival, "novelty": value.novelty, "total": total_value},
        edge_updates=edge_updates,
        before=before,
        after=[
            {
                "path": list(node.path),
                "visits": node.visits,
                "value_visits": node.value_visits,
                "escape_q": node.escape_q_value,
                "survival_q": node.survival_q_value,
                "novelty_q": node.novelty_q_value,
            }
            for node in path_nodes
        ],
    )


def _backpropagate(path_nodes: Sequence[MCTSNode], value: float) -> None:
    _backpropagate_components(
        path_nodes,
        MCTSValueComponents(survival=float(value)),
        survival_weight=1.0,
        novelty_weight=1.0,
    )


def _register_hit(
    node: MCTSNode,
    score: float,
    *,
    terminal: FacetLabel | None = None,
    terminal_bests: dict[str, TerminalHit],
    encountered: Counter[str],
    scorer: ExpansionScorer,
    best: TerminalHit | None,
    key: BlockKey | None = None,
    path: Sequence[int] | None = None,
    global_discovered_label_counts: Counter[str] | None = None,
    global_exact_hit_callback: ExactHitCallback | None = None,
) -> TerminalHit | None:
    terminal = node.terminal if terminal is None else terminal
    if terminal is None:
        return best

    hit_key = node.key if key is None else key
    hit_path = list(node.path) if path is None else list(path)
    hit = TerminalHit(
        label=terminal.label,
        key=hit_key,
        path=hit_path,
        score=float(score),
        rank=int(node.rank if key is None else scorer.affine_rank(hit_key)),
    )
    encountered[terminal.label] += 1
    if terminal.is_exact:
        scorer.discovered_label_counts[terminal.label] += 1
        if global_discovered_label_counts is not None:
            global_discovered_label_counts[terminal.label] += 1
            if global_exact_hit_callback is not None:
                global_exact_hit_callback(
                    terminal.label,
                    int(global_discovered_label_counts[terminal.label]),
                )
    previous = terminal_bests.get(terminal.label)
    if previous is None or hit.score > previous.score:
        terminal_bests[terminal.label] = hit
    if best is None or hit.score > best.score:
        best = hit
    return best


def _resolve_terminal_score(
    scorer: ExpansionScorer,
    key: BlockKey,
    terminal: FacetLabel,
    *,
    terminal_score_fn: TerminalScoreFn | None,
) -> float:
    if terminal_score_fn is None:
        return float(scorer.terminal_score(key))
    return float(terminal_score_fn(key, terminal))


def _score_action_with_terminal_reward(
    scorer: ExpansionScorer,
    key: BlockKey,
    action: int,
    *,
    terminal_score_fn: TerminalScoreFn | None,
) -> ExpansionScore:
    return scorer.score_action(
        key,
        action,
        terminal_score_fn=terminal_score_fn,
    )


def _softmax(scores: Sequence[float], *, temperature: float) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-6)
    max_score = max(float(score) for score in scores)
    exponents = [pow(2.718281828459045, (float(score) - max_score) / temp) for score in scores]
    total = sum(exponents)
    if total <= 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exponents]

def _weighted_choice_index(weights: Sequence[float], rng: random.Random) -> int:
    if not weights:
        raise ValueError("cannot sample from an empty weight list")
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0.0:
        return len(weights) - 1
    threshold = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += max(0.0, float(weight))
        if running >= threshold:
            return index
    return len(weights) - 1
