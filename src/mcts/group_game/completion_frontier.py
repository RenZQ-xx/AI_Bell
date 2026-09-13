from __future__ import annotations

"""Bounded completion work shared in canonical facet coordinates."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib

import numpy as np


@dataclass
class CompletionJob:
    key: tuple[int, int, str]
    source: int
    retained: int
    pattern_id: str
    rng: np.random.Generator
    attempts: int = 0
    continuation_batches: int = 0
    exit_keys: set[tuple[int, int, int]] = field(default_factory=set)


class CompletionFrontier:
    def __init__(self, *, seed: int, max_attempts: int):
        if max_attempts < 0:
            raise ValueError("completion budget must be nonnegative")
        self.seed = seed
        self.max_attempts = max_attempts
        self.jobs: dict[tuple[int, int, str], CompletionJob] = {}
        self.queues: dict[int, deque] = defaultdict(deque)

    def register(self, key, *, source: int, retained: int, pattern_id: str) -> CompletionJob:
        if key not in self.jobs:
            digest = hashlib.sha256(repr((self.seed, key)).encode("ascii")).digest()
            self.jobs[key] = CompletionJob(
                key, source, retained, pattern_id,
                np.random.default_rng(int.from_bytes(digest[:16], "little")),
            )
            self.queues[key[0]].append(key)
        return self.jobs[key]

    def pending(self, canonical_source: int) -> bool:
        queue = self.queues.get(canonical_source)
        while queue and self.jobs[queue[0]].attempts >= self.max_attempts:
            queue.popleft()
        return bool(queue)

    def next_job(self, canonical_source: int) -> CompletionJob | None:
        if not self.pending(canonical_source):
            return None
        queue = self.queues[canonical_source]
        job = self.jobs[queue[0]]
        queue.rotate(-1)
        return job

    def remaining_attempts(self, canonical_source: int) -> int:
        return sum(max(0, self.max_attempts - self.jobs[key].attempts)
                   for key in self.queues.get(canonical_source, ()))

    def to_dict(self) -> dict[str, object]:
        return {
            "max_attempts_per_key": self.max_attempts,
            "class_labels_used": False,
            "jobs": [{
                "canonical_source_hex": f"0x{key[0]:016x}",
                "canonical_retained_hex": f"0x{key[1]:016x}",
                "canonical_pattern_id": key[2],
                "attempts": job.attempts,
                "continuation_batches": job.continuation_batches,
                "unique_exit_keys": len(job.exit_keys),
                "finished": job.attempts >= self.max_attempts,
            } for key, job in self.jobs.items()],
        }
