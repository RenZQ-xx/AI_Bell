from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import torch

if __package__ in (None, ""):
    from geometry import generate_states_322
else:
    from .geometry import generate_states_322


State = Tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class SymmetryGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    states: List[State]
    generator_names: List[str]


def _generator_definitions() -> List[tuple[str, Callable[[State], State]]]:
    idx: Dict[str, int] = {"A0": 0, "A1": 1, "B0": 2, "B1": 3, "C0": 4, "C1": 5}

    def swap(state: State, first: str, second: str) -> State:
        buffer = list(state)
        pos_a = idx[first]
        pos_b = idx[second]
        buffer[pos_a], buffer[pos_b] = buffer[pos_b], buffer[pos_a]
        return tuple(buffer)  # type: ignore[return-value]

    def flip(state: State, name: str) -> State:
        buffer = list(state)
        buffer[idx[name]] = -buffer[idx[name]]
        return tuple(buffer)  # type: ignore[return-value]

    return [
        ("ABswap", lambda state: swap(swap(state, "A0", "B0"), "A1", "B1")),
        ("ACswap", lambda state: swap(swap(state, "A0", "C0"), "A1", "C1")),
        ("FlipIn_A", lambda state: swap(state, "A0", "A1")),
        ("FlipIn_B", lambda state: swap(state, "B0", "B1")),
        ("FlipIn_C", lambda state: swap(state, "C0", "C1")),
        ("FlipOut_A0", lambda state: flip(state, "A0")),
        ("FlipOut_A1", lambda state: flip(state, "A1")),
        ("FlipOut_B0", lambda state: flip(state, "B0")),
        ("FlipOut_B1", lambda state: flip(state, "B1")),
        ("FlipOut_C0", lambda state: flip(state, "C0")),
        ("FlipOut_C1", lambda state: flip(state, "C1")),
    ]


def build_symmetry_graph() -> SymmetryGraph:
    states = generate_states_322()
    state_to_id = {state: index for index, state in enumerate(states)}

    undirected_edges: set[tuple[int, int, int]] = set()
    names: List[str] = []
    for edge_type, (name, transform) in enumerate(_generator_definitions()):
        names.append(name)
        for source, state in enumerate(states):
            target = state_to_id[transform(state)]
            if source == target:
                continue
            low, high = sorted((source, target))
            undirected_edges.add((low, high, edge_type))

    senders: List[int] = []
    receivers: List[int] = []
    edge_types: List[int] = []
    for source, target, edge_type in sorted(undirected_edges):
        senders.extend([source, target])
        receivers.extend([target, source])
        edge_types.extend([edge_type, edge_type])

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
    return SymmetryGraph(edge_index=edge_index, edge_type=edge_type_tensor, states=states, generator_names=names)


def _compose_permutations(first: Tuple[int, ...], second: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(second[index] for index in first)


@lru_cache(maxsize=1)
def build_state_group_permutations() -> Tuple[Tuple[int, ...], ...]:
    states = generate_states_322()
    state_to_id = {state: index for index, state in enumerate(states)}
    generator_permutations = []
    for _, transform in _generator_definitions():
        generator_permutations.append(tuple(state_to_id[transform(state)] for state in states))

    identity = tuple(range(len(states)))
    seen = {identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for generator in generator_permutations:
            candidate = _compose_permutations(current, generator)
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return tuple(sorted(seen))


if __name__ == "__main__":
    graph = build_symmetry_graph()
    print({"nodes": len(graph.states), "directed_edges": int(graph.edge_index.shape[1]), "generators": graph.generator_names})
