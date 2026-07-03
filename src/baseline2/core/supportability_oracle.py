from __future__ import annotations

from dataclasses import dataclass
import time
from collections import Counter
from typing import Any


@dataclass
class SupportabilityOracle:
    supportability: Any
    config: Any
    profiling_enabled: bool = False
    progressive_enabled: bool = False
    progressive_schedule: tuple[int, ...] = (32, 96, 160, 256, 384, 512)
    call_count_value: int = 0
    cache_miss_count_value: int = 0
    elapsed_seconds_value: float = 0.0
    progressive_stage_counts_value: Counter[int] | None = None

    @classmethod
    def from_parts(
        cls,
        *,
        supportability: Any,
        config: Any,
        profiling_enabled: bool = False,
        progressive_enabled: bool = False,
        progressive_schedule: tuple[int, ...] = (32, 96, 160, 256, 384, 512),
        call_count_value: int = 0,
        cache_miss_count_value: int = 0,
        elapsed_seconds_value: float = 0.0,
        progressive_stage_counts_value: Counter[int] | None = None,
    ) -> "SupportabilityOracle":
        return cls(
            supportability=supportability,
            config=config,
            profiling_enabled=bool(profiling_enabled),
            progressive_enabled=bool(progressive_enabled),
            progressive_schedule=tuple(int(value) for value in progressive_schedule),
            call_count_value=int(call_count_value),
            cache_miss_count_value=int(cache_miss_count_value),
            elapsed_seconds_value=float(elapsed_seconds_value),
            progressive_stage_counts_value=Counter(progressive_stage_counts_value or {}),
        )

    @classmethod
    def from_scorer(cls, scorer: Any) -> "SupportabilityOracle":
        return cls.from_parts(
            supportability=scorer.supportability,
            config=scorer.config,
            profiling_enabled=bool(getattr(scorer, "supportability_profiling_enabled", False)),
            progressive_enabled=bool(getattr(scorer, "progressive_supportability_enabled", False)),
            progressive_schedule=tuple(int(value) for value in getattr(
                scorer,
                "progressive_supportability_schedule",
                (32, 96, 160, 256, 384, 512),
            )),
            call_count_value=int(getattr(scorer, "supportability_call_count", 0)),
            cache_miss_count_value=int(getattr(scorer, "supportability_cache_miss_count", 0)),
            elapsed_seconds_value=float(getattr(scorer, "supportability_elapsed_seconds", 0.0)),
            progressive_stage_counts_value=Counter(getattr(scorer, "progressive_supportability_stage_counts", {})),
        )

    def metrics(self, key: tuple[int, ...]):
        rank = self.supportability_rank(key)
        indices = self.vertex_indices(key)
        deterministic_verifier = (
            bool(self.config.supportability_constraint_verifier_enabled)
            and rank >= int(self.config.supportability_constraint_verifier_min_rank)
        )
        if not self.profiling_enabled:
            if self.progressive_enabled:
                return self._progressive_metrics(
                    indices,
                    key=key,
                    deterministic_verifier=deterministic_verifier,
                )
            return self.supportability.metrics(
                indices,
                direction_samples=self.config.supportability_direction_samples,
                seed=self.config.supportability_seed,
                direction_bank=self.config.supportability_direction_bank,
                cache_key=key,
                deterministic_verifier=deterministic_verifier,
            )

        sample_count = max(1, int(self.config.supportability_direction_samples))
        bank_mode = str(self.config.supportability_direction_bank)
        cache_key = (
            tuple(sorted(set(int(index) for index in indices))),
            sample_count,
            int(self.config.supportability_seed),
            bank_mode,
            key,
            bool(deterministic_verifier),
        )
        self.call_count_value += 1
        if cache_key not in self.supportability._metrics_cache:
            self.cache_miss_count_value += 1
        started = time.perf_counter()
        if self.progressive_enabled:
            out = self._progressive_metrics(
                indices,
                key=key,
                deterministic_verifier=deterministic_verifier,
            )
        else:
            out = self.supportability.metrics(
                indices,
                direction_samples=self.config.supportability_direction_samples,
                seed=self.config.supportability_seed,
                direction_bank=self.config.supportability_direction_bank,
                cache_key=key,
                deterministic_verifier=deterministic_verifier,
            )
        self.elapsed_seconds_value += time.perf_counter() - started
        return out

    def supportability_rank(self, key: tuple[int, ...]) -> int:
        rank_fn = getattr(self, "_rank_fn", None)
        if rank_fn is None:
            raise RuntimeError("SupportabilityOracle requires a rank function before metrics() is used.")
        return int(rank_fn(key))

    def vertex_indices(self, key: tuple[int, ...]) -> list[int]:
        vertex_indices_fn = getattr(self, "_vertex_indices_fn", None)
        if vertex_indices_fn is None:
            raise RuntimeError("SupportabilityOracle requires a vertex-index function before metrics() is used.")
        return list(vertex_indices_fn(key))

    def bind_geometry(self, *, rank_fn, vertex_indices_fn) -> None:
        self._rank_fn = rank_fn
        self._vertex_indices_fn = vertex_indices_fn

    def _progressive_metrics(
        self,
        indices,
        *,
        key: tuple[int, ...],
        deterministic_verifier: bool,
    ):
        final_samples = max(1, int(self.config.supportability_direction_samples))
        schedule = [
            int(value)
            for value in self.progressive_schedule
            if 0 < int(value) < final_samples
        ]
        schedule.append(final_samples)

        out = None
        for sample_count in schedule:
            out = self.supportability.metrics(
                indices,
                direction_samples=int(sample_count),
                seed=self.config.supportability_seed,
                direction_bank=self.config.supportability_direction_bank,
                cache_key=key,
                deterministic_verifier=deterministic_verifier if int(sample_count) == final_samples else False,
            )
            if (
                float(out.closer_side) == 0.0
                and float(out.supporting_shift) == 0.0
            ):
                self.progressive_stage_counts[int(sample_count)] += 1
                return out
        assert out is not None
        self.progressive_stage_counts[int(final_samples)] += 1
        return out

    @property
    def call_count(self) -> int:
        return int(self.call_count_value)

    @property
    def cache_miss_count(self) -> int:
        return int(self.cache_miss_count_value)

    @property
    def elapsed_seconds(self) -> float:
        return float(self.elapsed_seconds_value)

    @property
    def progressive_stage_counts(self):
        if self.progressive_stage_counts_value is None:
            self.progressive_stage_counts_value = Counter()
        return self.progressive_stage_counts_value
