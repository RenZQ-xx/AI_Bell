from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from baseline.facet_validator import FacetLabel
from baseline.orbit_blocks import BlockKey, add_block, empty_key, selected_blocks, unselected_blocks
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
    prior: float = 1.0
    terminal: FacetLabel | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    unexpanded_actions: list[int] = field(default_factory=list)
    action_priors: dict[int, float] = field(default_factory=dict)
    action_scores: dict[int, ExpansionScore] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


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

    nodes: dict[BlockKey, MCTSNode] = {}
    root = _get_or_create_node(nodes, scorer, root_key, path=[])
    terminal_bests: dict[str, TerminalHit] = {}
    exact_discoveries: list[ExactClassDiscovery] = []
    discovered_exact_classes: set[int] = set()
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
                best = _register_hit(
                    node,
                    terminal_score,
                    terminal_bests=terminal_bests,
                    encountered=encountered,
                    scorer=scorer,
                    best=best,
                )
                _backpropagate(path_nodes, terminal_score)
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
                best = _register_hit(
                    node,
                    terminal_score,
                    terminal=terminal,
                    terminal_bests=terminal_bests,
                    encountered=encountered,
                    scorer=scorer,
                    best=best,
                )
                _backpropagate(path_nodes, terminal_score)
                break

            if not node.unexpanded_actions:
                _prepare_actions(node, scorer, cfg)

            if node.unexpanded_actions:
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
                    _backpropagate(path_nodes, terminal_score)
                else:
                    rollout_value, best = _rollout(
                        child,
                        scorer,
                        cfg,
                        rng,
                        terminal_bests,
                        encountered,
                        best,
                        record_exact_discovery,
                        iteration_index,
                    )
                    total_value = float(step.score) + rollout_value
                    _backpropagate(path_nodes, total_value)
                break

            next_node = _select_child(node, cfg.exploration_constant)
            if next_node is None:
                value = _estimate_state_value(node.key, scorer, cfg)
                _backpropagate(path_nodes, value)
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

    scored = [scorer.score_action(node.key, action) for action in actions]
    scored.sort(key=lambda item: item.score, reverse=True)
    limit = max(1, min(cfg.expansion_candidate_pool, len(scored)))
    selected = scored[:limit]
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


def _select_child(node: MCTSNode, exploration_constant: float) -> MCTSNode | None:
    if not node.children:
        return None

    parent_visits = max(node.visits, 1)
    best_child: MCTSNode | None = None
    best_value = float("-inf")
    for child in node.children.values():
        exploitation = child.q_value
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
    iteration_index: int,
) -> tuple[float, TerminalHit | None]:
    current_key = node.key
    current_path = list(node.path)
    current_rank = node.rank
    total = 0.0
    discount = 1.0

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
            best = _register_hit(
                node,
                terminal_score,
                terminal=terminal,
                terminal_bests=terminal_bests,
                encountered=encountered,
                scorer=scorer,
                best=best,
            )
            return total + discount * terminal_score, best

        actions = unselected_blocks(current_key)
        if not actions:
            return total + discount * _estimate_state_value(current_key, scorer, cfg), best

        scored = [scorer.score_action(current_key, action) for action in actions]
        scored.sort(key=lambda item: item.score, reverse=True)
        limit = max(1, min(cfg.rollout_candidate_pool, len(scored)))
        pool = scored[:limit]
        weights = _softmax([item.score for item in pool], temperature=cfg.rollout_temperature)
        step = pool[_weighted_choice_index(weights, rng)]
        total += discount * float(step.score)
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
            return total, best

    return total + discount * _estimate_state_value(current_key, scorer, cfg), best


def _estimate_state_value(key: BlockKey, scorer: ExpansionScorer, cfg: MCTSConfig) -> float:
    rank = scorer.affine_rank(key)
    if rank >= 25:
        return float(scorer.terminal_score(key))

    actions = unselected_blocks(key)
    if not actions:
        return -100.0

    scored = [scorer.score_action(key, action) for action in actions]
    scored.sort(key=lambda item: item.score, reverse=True)
    limit = max(1, min(cfg.expansion_candidate_pool, len(scored)))
    return float(sum(item.score for item in scored[:limit]) / limit)


def _backpropagate(path_nodes: Sequence[MCTSNode], value: float) -> None:
    for node in path_nodes:
        node.visits += 1
        node.value_sum += float(value)


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