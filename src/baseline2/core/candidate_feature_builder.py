from __future__ import annotations


def is_supportable(support) -> bool:
    return (
        support is not None
        and float(support.closer_side) == 0.0
        and float(support.supporting_shift) <= 1e-9
    )

