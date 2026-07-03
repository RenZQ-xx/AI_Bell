from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from baseline.facet_validator import FacetLabel
from baseline.orbit_blocks import BlockKey, add_block, empty_key, selected_blocks, unselected_blocks
from baseline.reference_classes import DEFAULT_EXAMPLES_PATH, parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScore, ExpansionScorer, exact_class_id


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
    seed: int = 0
    selection_survival_weight: float = 1.0
    selection_novelty_weight: float = 1.0
    compatibility_examples_path: Path | None = DEFAULT_EXAMPLES_PATH
    # Progressive widening parameters
    progressive_k0: int = 1
    progressive_alpha: float = 1.0
    progressive_beta: float = 0.5


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

    def __init__(self, *, blocks: Sequence[Sequence[int]], examples_path: Path) -> None:
        self.blocks = [tuple(int(vertex) for vertex in block) for block in blocks]
        example_rows = parse_example_rows(examples_path)
        self.masks_by_class: dict[int, list[frozenset[int]]] = {}
        for class_id, rows in example_rows.items():
            masks: list[frozenset[int]] = []
            for row in rows.values():
                support = support_mask_from_row(row)
                support_vertices = {index for index, value in enumerate(support) if int(value) == 1}
                block_mask = frozenset(
                    index
                    for index, block in enumerate(self.blocks)
                    if all(int(vertex) in support_vertices for vertex in block)
                )
                masks.append(block_mask)
            self.masks_by_class[int(class_id)] = masks
        self._count_cache: dict[tuple[BlockKey, int], int] = {}
        self._signature_cache: dict[tuple[BlockKey, tuple[int, ...]], tuple[int, ...]] = {}

    @classmethod
    def from_scorer(cls, scorer: ExpansionScorer, examples_path: Path | None) -> ClassCompatibilityBank | None:
        if examples_path is None:
            return None
        return cls(blocks=scorer.blocks, examples_path=examples_path)

    def compat_count(self, key: BlockKey, class_id: int) -> int:
        cache_key = (key, int(class_id))
        if cache_key not in self._count_cache:
            selected = frozenset(selected_blocks(key))
            masks = self.masks_by_class.get(int(class_id), [])
            self._count_cache[cache_key] = sum(1 for mask in masks if selected <= mask)
        return self._count_cache[cache_key]

    def total_compat_count(self, key: BlockKey, class_ids: Sequence[int]) -> int:
        return sum(self.compat_count(key, class_id) for class_id in class_ids)

    def active_signature(self, key: BlockKey, class_ids: Sequence[int]) -> tuple[int, ...]:
        normalized_ids = tuple(sorted(int(class_id) for class_id in class_ids))
        cache_key = (key, normalized_ids)
        if cache_key not in self._signature_cache:
            self._signature_cache[cache_key] = tuple(
                class_id for class_id in normalized_ids if self.compat_count(key, class_id) > 0
            )
        return self._signature_cache[cache_key]


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
    value_sum: float = 0.0
    escape_sum: float = 0.0
    survival_sum: float = 0.0
    novelty_sum: float = 0.0
    prior: float = 1.0
    terminal: FacetLabel | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    unexpanded_actions: list[int] = field(default_factory=list)
    action_priors: dict[int, float] = field(default_factory=dict)
    action_scores: dict[int, ExpansionScore] = field(default_factory=dict)
    # rotation pointer for bucket selection when progressively expanding
    next_bucket: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    @property
    def escape_q_value(self) -> float:
        return self.escape_sum / self.visits if self.visits > 0 else 0.0

    @property
    def survival_q_value(self) -> float:
        return self.survival_sum / self.visits if self.visits > 0 else 0.0

    @property
    def novelty_q_value(self) -> float:
        return self.novelty_sum / self.visits if self.visits > 0 else 0.0


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

            if not node.unexpanded_actions:
                _prepare_actions(node, scorer, cfg)

            # Progressive widening: compute allowed child count K(s)
            def _compute_expansion_k(n: MCTSNode, conf: MCTSConfig) -> int:
                visits = float(max(1, n.visits))
                return int(max(1, conf.progressive_k0 + conf.progressive_alpha * (visits ** conf.progressive_beta)))

            def _select_action_progressive(
                n: MCTSNode,
                conf: MCTSConfig,
                rng: random.Random,
                compatibility_bank: ClassCompatibilityBank | None,
                discovered_exact_classes: set[int] | None,
                seen_signatures: Counter[tuple[int, ...]] | None,
            ) -> int:
                # Round-robin buckets 1..6
                buckets = list(range(6))
                start_bucket = n.next_bucket % len(buckets)
                active_classes = sorted(discovered_exact_classes) if discovered_exact_classes else []
                current_total = None
                if compatibility_bank is not None and active_classes:
                    current_total = compatibility_bank.total_compat_count(n.key, active_classes)

                candidates = list(n.unexpanded_actions)
                if not candidates:
                    raise ValueError("cannot choose from an empty action list")

                # precompute metrics for candidates
                metrics: dict[int, dict[str, float]] = {}
                for action in candidates:
                    score_item = n.action_scores.get(action)
                    base_score = float(score_item.score) if score_item is not None else 0.0
                    child_key = add_block(n.key, action)
                    old_compat_removed = 0.0
                    old_active_removed = 0.0
                    signature_novelty = 0.0
                    if compatibility_bank is not None and active_classes:
                        new_total = compatibility_bank.total_compat_count(child_key, active_classes)
                        old_compat_removed = float((current_total or 0) - new_total)
                        # count classes that go to zero
                        old_active_removed = float(
                            sum(1 for cid in active_classes if compatibility_bank.compat_count(n.key, cid) > 0 and compatibility_bank.compat_count(child_key, cid) == 0)
                        )
                        signature = compatibility_bank.active_signature(child_key, active_classes)
                        signature_novelty = 1.0 / (1.0 + float(seen_signatures.get(signature, 0) if seen_signatures is not None else 0))

                    low_aggression = base_score / (1.0 + max(0.0, old_compat_removed))
                    metrics[action] = {
                        "base": base_score,
                        "old_compat_removed": old_compat_removed,
                        "old_active_removed": old_active_removed,
                        "signature_novelty": signature_novelty,
                        "low_aggression": low_aggression,
                    }

                # Try buckets in round-robin order starting from start_bucket
                for offset in range(len(buckets)):
                    b = (start_bucket + offset) % len(buckets)
                    chosen: int | None = None
                    if b == 0:
                        # bucket1: baseline scorer high
                        chosen = max(candidates, key=lambda a: metrics[a]["base"], default=None)
                    elif b == 1:
                        # bucket2: old_compat_removed high
                        chosen = max(candidates, key=lambda a: metrics[a]["old_compat_removed"], default=None)
                    elif b == 2:
                        # bucket3: old_active_classes_removed high
                        chosen = max(candidates, key=lambda a: metrics[a]["old_active_removed"], default=None)
                    elif b == 3:
                        # bucket4: signature novelty high
                        chosen = max(candidates, key=lambda a: metrics[a]["signature_novelty"], default=None)
                    elif b == 4:
                        # bucket5: tie/random representative -> random among remaining
                        chosen = rng.choice(candidates) if candidates else None
                    elif b == 5:
                        # bucket6: low-aggression survivor
                        chosen = max(candidates, key=lambda a: metrics[a]["low_aggression"], default=None)

                    if chosen is not None:
                        n.next_bucket = (b + 1) % len(buckets)
                        return chosen

                # fallback
                return _choose_unexpanded_action(n, rng)

            K = _compute_expansion_k(node, cfg)
            if len(node.children) < K and node.unexpanded_actions:
                action = _select_action_progressive(node, cfg, rng, compatibility_bank, discovered_exact_classes, seen_signatures)
            elif node.unexpanded_actions:
                action = _choose_unexpanded_action(node, rng)
                child_key = add_block(node.key, action)
                child_path = [*node.path, int(action)]
                child = _get_or_create_node(nodes, scorer, child_key, path=child_path, parent=node, action=action)
                child.prior = node.action_priors.get(action, child.prior)
                node.children[action] = child
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


def _prepare_actions(node: MCTSNode, scorer: ExpansionScorer, cfg: MCTSConfig) -> None:
    actions = unselected_blocks(node.key)
    if not actions:
        node.unexpanded_actions = []
        node.action_priors = {}
        node.action_scores = {}
        return

    # Score all available actions so progressive widening can consider any of them
    scored = [scorer.score_action(node.key, action) for action in actions]
    scored.sort(key=lambda item: item.score, reverse=True)
    selected = scored
    priors = _softmax([item.score for item in selected], temperature=cfg.prior_temperature)
    node.unexpanded_actions = [item.action for item in selected]
    node.action_priors = {item.action: prior for item, prior in zip(selected, priors)}
    node.action_scores = {item.action: item for item in selected}


def _choose_unexpanded_action(node: MCTSNode, rng: random.Random) -> int:
    actions = list(node.unexpanded_actions)
    if not actions:
        raise ValueError("cannot choose from an empty action list")
    weights = [node.action_priors.get(action, 1.0) for action in actions]
    index = _weighted_choice_index(weights, rng)
    return actions[index]


def _select_child(
    node: MCTSNode,
    exploration_constant: float,
    *,
    survival_weight: float = 1.0,
    novelty_weight: float = 1.0,
) -> MCTSNode | None:
    if not node.children:
        return None

    parent_visits = max(node.visits, 1)
    best_child: MCTSNode | None = None
    best_value = float("-inf")
    for child in node.children.values():
        exploitation = child.escape_q_value + survival_weight * child.survival_q_value + novelty_weight * child.novelty_q_value
        exploration = exploration_constant * child.prior * (parent_visits ** 0.5) / (1 + child.visits)
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
    record_exact_discovery: Callable[..., None],
    compatibility_bank: ClassCompatibilityBank | None,
    discovered_exact_classes: set[int],
    seen_signatures: Counter[tuple[int, ...]],
    iteration_index: int,
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
            terminal_score = float(scorer.terminal_score(current_key))
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
            )
            total = MCTSValueComponents(
                escape=total.escape + discount * estimate.escape,
                survival=total.survival + discount * estimate.survival,
                novelty=total.novelty + discount * estimate.novelty,
            )
            return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best

        scored = [scorer.score_action(current_key, action) for action in actions]
        scored.sort(key=lambda item: item.score, reverse=True)
        limit = max(1, min(cfg.rollout_candidate_pool, len(scored)))
        pool = scored[:limit]
        weights = _softmax([item.score for item in pool], temperature=cfg.rollout_temperature)
        step = pool[_weighted_choice_index(weights, rng)]
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
            )
            return total.total(survival_weight=cfg.selection_survival_weight, novelty_weight=cfg.selection_novelty_weight), total, best

    estimate = _estimate_state_value(
        current_key,
        scorer,
        cfg,
        compatibility_bank=compatibility_bank,
        discovered_exact_classes=discovered_exact_classes,
        seen_signatures=seen_signatures,
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
) -> MCTSValueComponents:
    rank = scorer.affine_rank(key)
    if rank >= 25:
        survival = float(scorer.terminal_score(key))
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

    scored = [scorer.score_action(key, action) for action in actions]
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
    previous = terminal_bests.get(terminal.label)
    if previous is None or hit.score > previous.score:
        terminal_bests[terminal.label] = hit
    if best is None or hit.score > best.score:
        best = hit
    return best


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