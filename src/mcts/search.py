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


TerminalScoreFn = Callable[[BlockKey, FacetLabel], float]
ExactHitCallback = Callable[[str, int], None]
Rank23TailFn = Callable[
    [BlockKey, Sequence[int], "TerminalHit | None"],
    tuple["MCTSValueComponents", "TerminalHit | None"] | None,
]


class Rank23TailDeferred(RuntimeError):
    """Suspend one MCTS iteration until its bounded tail scan is complete."""

    def __init__(self, prefix_key: BlockKey, prefix_path: Sequence[int]) -> None:
        super().__init__("rank-23 tail scan is incomplete")
        self.prefix_key = prefix_key
        self.prefix_path = tuple(int(value) for value in prefix_path)
        self.rollout_total: MCTSValueComponents | None = None
        self.rollout_discount: float = 1.0
        self.rollout_best: TerminalHit | None = None


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
    rank23_tail_enabled: bool = False
    rank23_tail_max_prefixes: int = 24
    rank23_tail_candidates_per_step: int = 0
    rank23_tail_active_service: bool = False


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


class MCTSSearchSession:
    """A resumable add-only MCTS search advanced one iteration at a time."""

    def __init__(
        self,
        scorer: ExpansionScorer,
        *,
        config: MCTSConfig | None = None,
        start_key: BlockKey | None = None,
    ) -> None:
        self.scorer = scorer
        self.config = MCTSConfig() if config is None else config
        if self.config.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.config.iterations}")

        block_count = len(scorer.blocks)
        root_key = empty_key(block_count) if start_key is None else start_key
        self.rng = random.Random(self.config.seed)
        self.compatibility_bank = ClassCompatibilityBank.from_scorer(
            scorer,
            self.config.compatibility_examples_path,
        )
        self.nodes: dict[BlockKey, MCTSNode] = {}
        self.root = _get_or_create_node(self.nodes, scorer, root_key, path=[])
        self.terminal_bests: dict[str, TerminalHit] = {}
        self.exact_discoveries: list[ExactClassDiscovery] = []
        self.discovered_exact_classes: set[int] = set()
        self.seen_signatures: Counter[tuple[int, ...]] = Counter()
        self.encountered: Counter[str] = Counter()
        self.best: TerminalHit | None = None
        self.iterations_completed = 0
        self._record_signature(self.root.key)

    @property
    def finished(self) -> bool:
        return self.iterations_completed >= self.config.iterations

    def _record_exact_discovery(
        self,
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
        if class_id is None or class_id in self.discovered_exact_classes:
            return
        self.discovered_exact_classes.add(class_id)
        self.scorer.rare_target_classes.discard(class_id)
        self.exact_discoveries.append(
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

    def _record_signature(self, key: BlockKey) -> None:
        if self.compatibility_bank is None:
            return
        signature = self.compatibility_bank.active_signature(
            key,
            self.discovered_exact_classes,
        )
        self.seen_signatures[signature] += 1

    def _node_components(
        self,
        key: BlockKey,
        *,
        survival: float = 0.0,
    ) -> MCTSValueComponents:
        if self.compatibility_bank is None or not self.discovered_exact_classes:
            return MCTSValueComponents(
                escape=0.0,
                survival=float(survival),
                novelty=0.0,
            )
        active_classes = sorted(self.discovered_exact_classes)
        current_total = self.compatibility_bank.total_compat_count(key, active_classes)
        escape = 0.0 if current_total <= 0 else 1.0 / (1.0 + float(current_total))
        signature = self.compatibility_bank.active_signature(key, active_classes)
        novelty = 1.0 / (1.0 + float(self.seen_signatures[signature]))
        return MCTSValueComponents(
            escape=float(escape),
            survival=float(survival),
            novelty=float(novelty),
        )

    def step(self) -> list[ExactClassDiscovery]:
        """Advance one complete MCTS iteration and return its new exact classes."""
        if self.finished:
            return []

        cfg = self.config
        scorer = self.scorer
        discovery_start = len(self.exact_discoveries)
        self.iterations_completed += 1
        iteration_index = self.iterations_completed
        path_nodes: list[MCTSNode] = [self.root]
        node = self.root

        while True:
            if node.is_terminal:
                terminal_score = float(scorer.terminal_score(node.key))
                if node.terminal is not None and node.terminal.is_exact:
                    self._record_exact_discovery(
                        node.terminal,
                        score=terminal_score,
                        iteration_index=iteration_index,
                        depth=len(node.path),
                        key=node.key,
                        path=node.path,
                        rank=node.rank,
                    )
                node_value = self._node_components(node.key, survival=terminal_score)
                self.best = _register_hit(
                    node,
                    terminal_score,
                    terminal_bests=self.terminal_bests,
                    encountered=self.encountered,
                    scorer=scorer,
                    best=self.best,
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
                    self._record_exact_discovery(
                        terminal,
                        score=terminal_score,
                        iteration_index=iteration_index,
                        depth=len(node.path),
                        key=node.key,
                        path=node.path,
                        rank=node.rank,
                    )
                node_value = self._node_components(node.key, survival=terminal_score)
                self.best = _register_hit(
                    node,
                    terminal_score,
                    terminal=terminal,
                    terminal_bests=self.terminal_bests,
                    encountered=self.encountered,
                    scorer=scorer,
                    best=self.best,
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
                    self.rng,
                    scorer=scorer,
                    cfg=cfg,
                    compatibility_bank=self.compatibility_bank,
                    discovered_exact_classes=self.discovered_exact_classes,
                    seen_signatures=self.seen_signatures,
                )
                child_key = add_block(node.key, action)
                child_path = [*node.path, int(action)]
                child = _get_or_create_node(
                    self.nodes,
                    scorer,
                    child_key,
                    path=child_path,
                    parent=node,
                    action=action,
                )
                child.prior = node.action_priors.get(action, child.prior)
                node.children[action] = child
                node.unexpanded_actions = [item for item in node.unexpanded_actions if item != action]
                path_nodes.append(child)

                action_score = node.action_scores[action]
                if child.is_terminal or child.rank >= 25:
                    terminal = (
                        child.terminal
                        if child.terminal is not None
                        else scorer.terminal_label(child.key)
                    )
                    terminal_score = float(action_score.score)
                    child_value = self._node_components(child.key, survival=terminal_score)
                    if terminal.is_exact:
                        self._record_exact_discovery(
                            terminal,
                            score=terminal_score,
                            iteration_index=iteration_index,
                            depth=len(child.path),
                            key=child.key,
                            path=child.path,
                            rank=child.rank,
                        )
                    self.best = _register_hit(
                        child,
                        terminal_score,
                        terminal=terminal,
                        terminal_bests=self.terminal_bests,
                        encountered=self.encountered,
                        scorer=scorer,
                        best=self.best,
                    )
                    self._record_signature(child.key)
                    _backpropagate_components(
                        path_nodes,
                        child_value,
                        survival_weight=cfg.selection_survival_weight,
                        novelty_weight=cfg.selection_novelty_weight,
                    )
                else:
                    _rollout_value, rollout_components, self.best = _rollout(
                        child,
                        scorer,
                        cfg,
                        self.rng,
                        self.terminal_bests,
                        self.encountered,
                        self.best,
                        self._record_exact_discovery,
                        self.compatibility_bank,
                        self.discovered_exact_classes,
                        self.seen_signatures,
                        iteration_index,
                    )
                    child_components = MCTSValueComponents(
                        escape=rollout_components.escape,
                        survival=float(action_score.score) + rollout_components.survival,
                        novelty=rollout_components.novelty,
                    )
                    self._record_signature(child.key)
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
                survival_weight=cfg.selection_survival_weight,
                novelty_weight=cfg.selection_novelty_weight,
            )
            if next_node is None:
                components = _estimate_state_value(
                    node.key,
                    scorer,
                    cfg,
                    compatibility_bank=self.compatibility_bank,
                    discovered_exact_classes=self.discovered_exact_classes,
                    seen_signatures=self.seen_signatures,
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

        return list(self.exact_discoveries[discovery_start:])

    def run(self) -> MCTSResult:
        while not self.finished:
            self.step()
        return self.result()

    def result(self) -> MCTSResult:
        """Return a snapshot; it is also valid while the session is paused."""
        return MCTSResult(
            best=self.best,
            terminal_bests=self.terminal_bests,
            exact_discoveries=self.exact_discoveries,
            encountered_label_counts=self.encountered,
            iterations_completed=self.iterations_completed,
            nodes_created=len(self.nodes),
            root_visits=self.root.visits,
        )

    def release_tree(self) -> None:
        for node in self.nodes.values():
            node.parent = None
            node.children.clear()
            node.unexpanded_actions.clear()
            node.action_priors.clear()
            node.action_scores.clear()
        self.nodes.clear()
        if self.compatibility_bank is not None:
            self.compatibility_bank.clear_cache()
        self.seen_signatures.clear()


def run_mcts_search(
    scorer: ExpansionScorer,
    *,
    config: MCTSConfig | None = None,
    start_key: BlockKey | None = None,
) -> MCTSResult:
    """Run add-only Monte Carlo tree search from the empty block state."""
    return MCTSSearchSession(
        scorer,
        config=config,
        start_key=start_key,
    ).run()


def _get_or_create_node(
    nodes: dict[BlockKey, MCTSNode],
    scorer: ExpansionScorer,
    key: BlockKey,
    *,
    path: list[int],
    parent: MCTSNode | None = None,
    action: int | None = None,
) -> MCTSNode:
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
    node.actions_initialized = True
    node.unexpanded_actions = list(actions)
    node.action_priors = {}
    node.action_scores = {}


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

    bucket = _next_progressive_bucket(node, cfg)
    active_classes = sorted(discovered_exact_classes)
    if bucket == 0:
        scored_candidates = _score_lazy_action_batch(
            node,
            candidates,
            scorer,
            cfg,
            rng,
            terminal_score_fn=terminal_score_fn,
            adaptive_score_batch=adaptive_score_batch,
        )
        action = _best_action_random_tie(
            scored_candidates,
            {item: float(node.action_scores[item].score) for item in scored_candidates},
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
        action = _best_action_random_tie(candidates, values, rng)
    elif bucket == 3 and compatibility_bank is not None and active_classes:
        values = {}
        for action in candidates:
            signature = compatibility_bank.active_signature(add_block(node.key, action), active_classes)
            values[action] = 1.0 / (1.0 + float(seen_signatures[signature]))
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
        action = _best_action_random_tie(scored_candidates, values, rng)
    else:
        action = rng.choice(candidates)

    _ensure_action_scored(
        node,
        action,
        scorer,
        cfg,
        terminal_score_fn=terminal_score_fn,
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
        # Geometry is immutable for a node/action pair. Only the scalar reward
        # depends on the discovery epoch; comparing the full dataclass can ask
        # NumPy payloads inside supportability/validation to produce a boolean.
        if float(current.score) == float(score_item.score):
            continue
        node.action_scores[action] = current
        changed = True
    if changed:
        _refresh_action_priors(node, cfg)


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
    survival_weight: float = 1.0,
    novelty_weight: float = 1.0,
) -> MCTSNode | None:
    if not node.children:
        return None

    parent_visits = max(node.value_visits, 1.0)
    best_child: MCTSNode | None = None
    best_value = float("-inf")
    for child in node.children.values():
        exploitation = child.escape_q_value + survival_weight * child.survival_q_value + novelty_weight * child.novelty_q_value
        exploration = (
            exploration_constant
            * child.prior
            * (parent_visits ** 0.5)
            / (1.0 + child.value_visits)
        )
        score = exploitation + exploration
        if score > best_value:
            best_value = score
            best_child = child
    return best_child


def _rollout(
    node: MCTSNode,
    scorer: ExpansionScorer,
    cfg: MCTSConfig,
    rng: random.Random,
    terminal_bests: dict[str, TerminalHit],
    encountered: Counter[str],
    best: TerminalHit | None,
    record_exact_discovery: Callable[..., float | None],
    compatibility_bank: ClassCompatibilityBank | None,
    discovered_exact_classes: set[int],
    seen_signatures: Counter[tuple[int, ...]],
    iteration_index: int,
    *,
    terminal_score_fn: TerminalScoreFn | None = None,
    global_discovered_label_counts: Counter[str] | None = None,
    global_exact_hit_callback: ExactHitCallback | None = None,
    rollout_score_batch: int | None = None,
    rank23_tail_fn: Rank23TailFn | None = None,
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
            if terminal.is_exact:
                resolved_score = record_exact_discovery(
                    terminal,
                    score=terminal_score,
                    iteration_index=iteration_index,
                    depth=len(current_path),
                    key=current_key,
                    path=current_path,
                    rank=current_rank,
                )
                if resolved_score is not None:
                    terminal_score = float(resolved_score)
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

        if current_rank == 23 and rank23_tail_fn is not None:
            try:
                tail = rank23_tail_fn(current_key, current_path, best)
            except Rank23TailDeferred as pending:
                pending.rollout_total = total
                pending.rollout_discount = float(discount)
                pending.rollout_best = best
                raise
            if tail is not None:
                tail_components, best = tail
                total = MCTSValueComponents(
                    escape=total.escape + discount * tail_components.escape,
                    survival=total.survival + discount * tail_components.survival,
                    novelty=total.novelty + discount * tail_components.novelty,
                )
                return (
                    total.total(
                        survival_weight=cfg.selection_survival_weight,
                        novelty_weight=cfg.selection_novelty_weight,
                    ),
                    total,
                    best,
                )

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
        scored.sort(key=lambda item: item.score, reverse=True)
        limit = max(1, min(cfg.rollout_candidate_pool, len(scored)))
        pool = scored[:limit]
        weights = _softmax([item.score for item in pool], temperature=cfg.rollout_temperature)
        step = pool[_weighted_choice_index(weights, rng)]
        step_score = float(step.score)
        if step.terminal is not None and step.terminal.is_exact:
            resolved_score = record_exact_discovery(
                step.terminal,
                score=step_score,
                iteration_index=iteration_index,
                depth=len(current_path) + 1,
                key=step.key,
                path=[*current_path, int(step.action)],
                rank=step.new_rank,
            )
            if resolved_score is not None:
                step_score = float(resolved_score)
        step_components = score_components(step.key, survival=step_score)
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
                if compatibility_bank is not None:
                    seen_signatures[compatibility_bank.active_signature(current_key, discovered_exact_classes)] += 1
            best = _register_hit(
                node,
                step_score,
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
    for node in path_nodes:
        node.visits += 1
        node.value_visits += 1.0
        node.value_sum += float(total_value)
        node.escape_sum += float(value.escape)
        node.survival_sum += float(value.survival)
        node.novelty_sum += float(value.novelty)


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
            increment = getattr(global_discovered_label_counts, "increment", None)
            if callable(increment):
                global_count = int(increment(terminal.label))
            else:
                global_discovered_label_counts[terminal.label] += 1
                global_count = int(global_discovered_label_counts[terminal.label])
            if global_exact_hit_callback is not None:
                global_exact_hit_callback(
                    terminal.label,
                    global_count,
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
