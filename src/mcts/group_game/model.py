from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Literal


Role = Literal["filler", "corrector", "done"]
ActionKind = Literal["add", "remove", "rewrite", "stop"]


@lru_cache(maxsize=65536)
def support_word_to_key(word: int) -> tuple[int, ...]:
    """Convert a 64-bit support into the singleton partition key."""
    normalized = int(word)
    if normalized < 0 or normalized.bit_length() > 64:
        raise ValueError("support word must fit in 64 bits")
    return tuple((normalized >> index) & 1 for index in range(64))


def block_word(block: tuple[int, ...]) -> int:
    word = 0
    for vertex in block:
        value = int(vertex)
        if value < 0 or value >= 64:
            raise ValueError(f"vertex index must be in [0, 63], got {value}")
        word |= 1 << value
    return word


def size_bucket(size: int) -> str:
    normalized = int(size)
    if normalized <= 1:
        return "singleton"
    if normalized == 2:
        return "pair"
    if normalized <= 4:
        return "small"
    if normalized <= 8:
        return "medium"
    return "large"


def rank_band(rank: int) -> str:
    value = int(rank)
    if value <= 7:
        return "r00_07"
    if value <= 15:
        return "r08_15"
    if value <= 21:
        return "r16_21"
    if value <= 24:
        return "r22_24"
    return "r25"


@dataclass(frozen=True)
class SymmetryState:
    """One state in the shared alternating Filler-Corrector tree."""

    support_word: int
    role: Role = "filler"
    corrections_used: int = 0
    context_pattern_id: str = "BFS322-C1-00"
    repair_source_word: int = 0

    def __post_init__(self) -> None:
        if self.support_word < 0 or self.support_word.bit_length() > 64:
            raise ValueError("support word must fit in 64 bits")
        if self.role not in ("filler", "corrector", "done"):
            raise ValueError(f"unknown role {self.role!r}")
        if self.corrections_used < 0:
            raise ValueError("corrections_used must be nonnegative")
        if not self.context_pattern_id:
            raise ValueError("context_pattern_id must not be empty")

    @property
    def support_size(self) -> int:
        return self.support_word.bit_count()

    @property
    def key(self) -> tuple[int, Role, int, str, int]:
        return (
            self.support_word,
            self.role,
            self.corrections_used,
            self.context_pattern_id,
            self.repair_source_word,
        )


@dataclass(frozen=True)
class MacroAction:
    """A symmetry-derived union of one or more subgroup orbits."""

    kind: ActionKind
    pattern_id: str
    orbit_indices: tuple[int, ...]
    block_word: int
    effective_word: int
    source: str
    structure: str
    block_size: int
    arity: int
    remove_word: int = 0
    add_word: int = 0
    remove_orbit_indices: tuple[int, ...] = ()
    add_orbit_indices: tuple[int, ...] = ()
    remove_pattern_id: str = ""
    add_pattern_id: str = ""
    source_facet_word: int = 0

    @cached_property
    def action_id(self) -> str:
        orbit_part = ",".join(str(value) for value in self.orbit_indices)
        return (
            f"{self.kind}:{self.pattern_id}:{orbit_part}:"
            f"{self.effective_word:016x}:"
            f"{self.remove_word:016x}:{self.add_word:016x}:"
            f"{','.join(str(value) for value in self.remove_orbit_indices)}:"
            f"{','.join(str(value) for value in self.add_orbit_indices)}:"
            f"{self.remove_pattern_id}:{self.add_pattern_id}:{self.source_facet_word:016x}"
        )

    @property
    def bucket(self) -> str:
        return size_bucket(self.block_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "pattern_id": self.pattern_id,
            "orbit_indices": list(self.orbit_indices),
            "block_word_hex": f"0x{self.block_word:016x}",
            "effective_word_hex": f"0x{self.effective_word:016x}",
            "remove_word_hex": f"0x{self.remove_word:016x}",
            "add_word_hex": f"0x{self.add_word:016x}",
            "remove_orbit_indices": list(self.remove_orbit_indices),
            "add_orbit_indices": list(self.add_orbit_indices),
            "remove_pattern_id": self.remove_pattern_id,
            "add_pattern_id": self.add_pattern_id,
            "source_facet_word_hex": f"0x{self.source_facet_word:016x}",
            "source": self.source,
            "structure": self.structure,
            "block_size": self.block_size,
            "size_bucket": self.bucket,
            "arity": self.arity,
        }


@dataclass
class EdgeStats:
    action: MacroAction
    child_key: tuple[int, Role, int, str, int]
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
        if self.epoch != int(epoch) or self.epoch_visits <= 0:
            return None
        return self.epoch_value_sum / self.epoch_visits

    def update(self, value: float, epoch: int) -> None:
        self.visits += 1
        self.value_sum += float(value)
        if self.epoch != int(epoch):
            self.epoch = int(epoch)
            self.epoch_visits = 0
            self.epoch_value_sum = 0.0
        self.epoch_visits += 1
        self.epoch_value_sum += float(value)

    def to_dict(self, *, epoch: int) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "child_key": list(self.child_key),
            "prior": self.prior,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "visits": self.visits,
            "mean_value": self.mean_value,
            "epoch_mean": self.epoch_mean(epoch),
            "discovery_hits": self.discovery_hits,
        }


@dataclass
class GameNode:
    state: SymmetryState
    rank: int
    facet_word: int = 0
    visits: int = 0
    value_sum: float = 0.0
    edges: dict[str, EdgeStats] = field(default_factory=dict)
    expanded_patterns: set[str] = field(default_factory=set)
    exhausted_patterns: set[str] = field(default_factory=set)
    pattern_action_counts: dict[str, int] = field(default_factory=dict)
    rejected_action_ids: set[str] = field(default_factory=set)
    child_support_words: set[int] = field(default_factory=set)
    child_state_keys: set[tuple[int, Role, int, str, int]] = field(default_factory=set)
    bucket_cursor: int = 0
    relation_cursor: int = 0
    action_lane_cursor: int = 0
    action_bucket_cursor: int = 0
    knowledge_epoch_seen: int = -1
    retired_action_ids: set[str] = field(default_factory=set)
    last_recycle_visit: int = 0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0
