from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

from .problem import validate_word
from mcts.symmetry_game.patterns import geometric_state_key


Role = Literal["filler", "corrector", "done"]
ActionKind = Literal["add", "remove", "rewrite", "stop"]


def size_bucket(size: int) -> str:
    if size <= 1:
        return "singleton"
    if size == 2:
        return "pair"
    if size <= 4:
        return "small"
    if size <= 8:
        return "medium"
    return "large"


def rank_band(rank: int) -> str:
    if rank < 32:
        return "r00_31"
    if rank < 56:
        return "r32_55"
    if rank < 72:
        return "r56_71"
    if rank < 79:
        return "r72_78"
    return "r79"


@dataclass(frozen=True)
class SearchState:
    support_word: int
    role: Role = "filler"
    corrections_used: int = 0
    context_pattern_id: str = "BFS422-IDENTITY"
    source_facet_word: int = 0

    def __post_init__(self) -> None:
        validate_word(self.support_word)
        validate_word(self.source_facet_word)
        if self.role not in ("filler", "corrector", "done"):
            raise ValueError(f"unknown role: {self.role}")
        if self.corrections_used < 0:
            raise ValueError("corrections_used must be nonnegative")

    @property
    def key(self) -> tuple[int, Role, int, str, int]:
        return (
            self.support_word,
            self.role,
            self.corrections_used,
            self.context_pattern_id,
            self.source_facet_word,
        )

    @property
    def geometry_key(self) -> tuple[int, str, int, int]:
        return geometric_state_key(
            self.support_word,
            self.role,
            self.corrections_used,
            self.source_facet_word,
        )


@dataclass(frozen=True)
class MacroAction:
    kind: ActionKind
    pattern_id: str
    block_indices: tuple[int, ...]
    effective_word: int
    source: str
    structure: str
    remove_word: int = 0
    add_word: int = 0

    @cached_property
    def action_id(self) -> str:
        blocks = ",".join(map(str, self.block_indices))
        return (
            f"{self.kind}:{self.pattern_id}:{blocks}:"
            f"{self.effective_word:064x}:{self.remove_word:064x}:{self.add_word:064x}"
        )

    @property
    def block_size(self) -> int:
        return self.effective_word.bit_count()

    @property
    def arity(self) -> int:
        return len(self.block_indices)

    @property
    def bucket(self) -> str:
        return size_bucket(self.block_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "pattern_id": self.pattern_id,
            "block_indices": list(self.block_indices),
            "effective_word_hex": f"0x{self.effective_word:064x}",
            "remove_word_hex": f"0x{self.remove_word:064x}",
            "add_word_hex": f"0x{self.add_word:064x}",
            "source": self.source,
            "structure": self.structure,
            "block_size": self.block_size,
            "size_bucket": self.bucket,
            "arity": self.arity,
        }


@dataclass
class EdgeStats:
    action: MacroAction
    child_key: tuple
    prior: float
    rank_before: int
    rank_after: int
    visits: int = 0
    value_sum: float = 0.0
    epoch: int = -1
    epoch_visits: int = 0
    epoch_value_sum: float = 0.0
    discovery_hits: int = 0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def epoch_mean(self, epoch: int) -> float | None:
        if self.epoch != epoch or not self.epoch_visits:
            return None
        return self.epoch_value_sum / self.epoch_visits

    def update(self, reward: float, epoch: int) -> None:
        self.visits += 1
        self.value_sum += reward
        if self.epoch != epoch:
            self.epoch = epoch
            self.epoch_visits = 0
            self.epoch_value_sum = 0.0
        self.epoch_visits += 1
        self.epoch_value_sum += reward


@dataclass
class GameNode:
    state: SearchState
    rank: int
    visits: int = 0
    value_sum: float = 0.0
    edges: dict[str, EdgeStats] = field(default_factory=dict)
    expanded_patterns: set[str] = field(default_factory=set)
    exhausted_patterns: set[str] = field(default_factory=set)
    rejected_action_ids: set[str] = field(default_factory=set)
    child_keys: set[tuple] = field(default_factory=set)
    context_pattern_ids: set[str] = field(default_factory=set)
    pattern_action_counts: dict[str, int] = field(default_factory=dict)
    pattern_cursor: int = 0
    bucket_cursor: int = 0
    relation_cursor: int = 0
    action_lane_cursor: int = 0
    knowledge_epoch_seen: int = -1
    retired_action_ids: set[str] = field(default_factory=set)
    last_recycle_visit: int = 0

    def __post_init__(self) -> None:
        self.context_pattern_ids.add(self.state.context_pattern_id)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0
