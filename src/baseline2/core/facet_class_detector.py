from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baseline2.primitives.reference_classes import (
    build_reference_database,
    classify_hyperplane,
    exact_label_from_match,
)

from .facet_terminal import FacetFamily, FacetFamilyRegistry


@dataclass
class KnownFacetClassDetector:
    """External 322 reference-class detector for discovered facet families."""

    reference: dict[str, Any]

    @classmethod
    def build(cls) -> "KnownFacetClassDetector":
        return cls(reference=build_reference_database())

    def classify_family(self, family: FacetFamily) -> dict[str, Any]:
        match = classify_hyperplane(family.normal, family.offset, self.reference)
        label = exact_label_from_match(match)
        classes = [int(value) for value in match.get("matched_classes", [])]
        return {
            "family_id": family.family_id,
            "representation": family.representation,
            "detected_label": label,
            "detected_class_ids": classes,
            "tier": match.get("tier"),
            "match_source": match.get("match_source"),
            "canonical_key": match.get("canonical_key"),
            "recovered_integer_row": match.get("recovered_integer_row"),
            "canonical_integer_row": match.get("canonical_integer_row"),
            "direction_error": match.get("direction_error"),
            "max_ratio_error": match.get("max_ratio_error"),
        }

    def audit_registry(self, registry: FacetFamilyRegistry) -> dict[str, Any]:
        families = []
        class_family_counts: dict[str, int] = {}
        class_hit_counts: dict[str, int] = {}
        unknown_family_count = 0

        for family in sorted(registry.families.values(), key=lambda item: item.family_id):
            item = self.classify_family(family)
            hit_count = registry.family_hits(family.family_id)
            item["hit_count"] = hit_count
            item["duplicate_count"] = max(0, hit_count - 1)
            families.append(item)

            class_ids = [int(value) for value in item.get("detected_class_ids", [])]
            if not class_ids:
                unknown_family_count += 1
                continue
            for class_id in class_ids:
                key = str(int(class_id))
                class_family_counts[key] = int(class_family_counts.get(key, 0)) + 1
                class_hit_counts[key] = int(class_hit_counts.get(key, 0)) + int(hit_count)

        return {
            "detector": "known_322_reference_classes",
            "family_count": len(families),
            "classified_family_count": len(families) - unknown_family_count,
            "unknown_family_count": unknown_family_count,
            "class_family_counts": dict(sorted(class_family_counts.items(), key=lambda item: int(item[0]))),
            "class_hit_counts": dict(sorted(class_hit_counts.items(), key=lambda item: int(item[0]))),
            "families": families,
        }
