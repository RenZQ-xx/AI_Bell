from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXAMPLES_PATH = PROJECT_ROOT / "data" / "facet_classes_322_examples.txt"


@dataclass
class SeedItem:
    class_id: int
    example_id: int
    row: List[int]
    mask: List[int]
    cardinality: int


def row_to_mask(row: List[int], points: torch.Tensor, tol: float = 1e-8) -> List[int]:
    row_tensor = torch.tensor(row, dtype=points.dtype, device=points.device)
    signed = row_tensor[0] + points @ row_tensor[1:]
    support = torch.isclose(signed, torch.zeros_like(signed), atol=tol, rtol=0.0)
    return support.to(dtype=torch.int64).cpu().tolist()


def load_seed_bank(points: torch.Tensor, examples_path: Path | None = None) -> List[SeedItem]:
    path = examples_path or DEFAULT_EXAMPLES_PATH
    class_pattern = re.compile(r"^\[class\s+(\d+)\]")
    rep_pattern = re.compile(r"^rep(\d+):\s+(.*)$")

    items: List[SeedItem] = []
    current_class_id: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_match = class_pattern.match(line)
        if class_match:
            current_class_id = int(class_match.group(1))
            continue
        rep_match = rep_pattern.match(line)
        if rep_match and current_class_id is not None:
            example_id = int(rep_match.group(1))
            row = [int(token) for token in rep_match.group(2).split()]
            mask = row_to_mask(row, points=points)
            items.append(
                SeedItem(
                    class_id=current_class_id,
                    example_id=example_id,
                    row=row,
                    mask=mask,
                    cardinality=sum(mask),
                )
            )
    if not items:
        raise ValueError(f"No seed items could be parsed from {path}")
    return items


def seed_tensor(seed_items: List[SeedItem], device: torch.device) -> torch.Tensor:
    return torch.tensor([item.mask for item in seed_items], dtype=torch.float32, device=device)


def summarize_seed_bank(seed_items: List[SeedItem]) -> Dict[str, int]:
    class_ids = {item.class_id for item in seed_items}
    return {
        "num_seed_items": len(seed_items),
        "num_classes": len(class_ids),
    }
