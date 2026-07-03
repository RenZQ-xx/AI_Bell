from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from baseline2.primitives.orbit_blocks import add_block


class ExpansionFactOracle(Protocol):
    config: object

    def affine_rank(self, key: tuple[int, ...]) -> int:
        ...

    def flat_capacity(self, key: tuple[int, ...]) -> int:
        ...

    def supportability_metrics(self, key: tuple[int, ...]):
        ...


@dataclass(frozen=True)
class ExpansionFacts:
    key: tuple[int, ...]
    action: int
    parent_key: tuple[int, ...]
    old_rank: int
    new_rank: int
    rank_gain: int
    flat_capacity: int
    supportability: object | None


def build_expansion_facts(
    oracle: ExpansionFactOracle,
    key: tuple[int, ...],
    action: int,
    *,
    support_override=None,
) -> ExpansionFacts:
    candidate = add_block(key, int(action))
    old_rank = oracle.affine_rank(key)
    new_rank = oracle.affine_rank(candidate)
    flat_capacity = 0 if new_rank >= 25 else int(oracle.flat_capacity(candidate))

    support = support_override
    supportability_start_rank = getattr(oracle.config, "supportability_start_rank", None)
    if support is None and supportability_start_rank is not None and new_rank >= int(supportability_start_rank):
        support = oracle.supportability_metrics(candidate)

    return ExpansionFacts(
        key=tuple(candidate),
        action=int(action),
        parent_key=tuple(key),
        old_rank=int(old_rank),
        new_rank=int(new_rank),
        rank_gain=int(new_rank - old_rank),
        flat_capacity=int(flat_capacity),
        supportability=support,
    )

