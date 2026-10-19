from __future__ import annotations

import importlib

import pytest


module = importlib.import_module("mcts.422.subgroup_completion")
HANDOFF = module.DEFAULT_HANDOFF_ROOT


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_complete_node_inventory_covers_the_full_catalog() -> None:
    inventory = module.load_inventory(HANDOFF)

    assert len(inventory.nodes) == 721
    assert len(inventory.nonempty_nodes) == 560
    assert len(inventory.required_nodes) == 363
    assert inventory.facet_row_count == 43_974
    assert inventory.class_count == 34_836
    assert len(set().union(*(node.class_ids for node in inventory.required_nodes))) == 34_836


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_typical_scope_spans_all_subgroup_structures() -> None:
    inventory = module.load_inventory(HANDOFF)
    selected = module.select_nodes(inventory, scope="typical")

    assert {node.structure for node in selected} == {
        "C2 x C2",
        "C4",
        "C6",
        "S3",
    }
    assert max(node.intrinsic_dimension for node in selected) == 29
    assert max(node.class_count for node in selected) == 1_729
