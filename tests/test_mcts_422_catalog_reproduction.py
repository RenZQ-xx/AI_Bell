from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest


module = importlib.import_module("mcts.422.catalog_reproduction")
HANDOFF = module.DEFAULT_HANDOFF_ROOT


def test_json_write_retries_transient_atomic_replace_denial(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    module.save_json({"state": "OLD"}, path)
    original = Path.replace
    attempts = []
    delays = []

    def replace(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError("reader temporarily blocks replacement")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    module.save_json({"state": "NEW"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "NEW"
    assert len(attempts) == 3
    assert delays == [0.05, 0.1]


def test_json_write_permanent_denial_is_bounded_and_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    module.save_json({"state": "OLD"}, path)
    attempts = []
    delays = []

    def replace(source, target):
        attempts.append(source)
        raise PermissionError("permanent denial")

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="permanent"):
        module.save_json({"state": "NEW"}, path)
    assert len(attempts) == 8
    assert len(delays) == 7
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "OLD"


@pytest.fixture(scope="module")
def inventory():
    if not HANDOFF.exists():
        pytest.skip("4-2-2 handoff is external")
    return module.load_inventory(HANDOFF)


def runner(output: Path, nodes, *, resume=False, batch=2):
    targets = set().union(*(n.class_ids for n in nodes))
    return module.CatalogReproduction(
        handoff_root=HANDOFF, nodes=nodes, catalog_ids=targets,
        config=module.ReproductionConfig(seed=4222027, batch_iterations=batch),
        output_dir=output, resume=resume,
    )


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        module.ReproductionConfig(batch_iterations=0)
    with pytest.raises(ValueError):
        module.ReproductionConfig(stagnation_iterations=0)


def test_restart_seeds_are_stable_and_node_specific():
    assert module.node_seed(1, "A") == module.node_seed(1, "A", 0)
    assert module.node_seed(1, "A", 1) != module.node_seed(1, "A", 0)
    assert module.node_seed(1, "A", 1) != module.node_seed(1, "B", 1)


def test_checksum_resume_preserves_tree_rng_and_known_classes(tmp_path, inventory):
    node = inventory.by_id["BFS422-H-0000"]
    original = runner(tmp_path, [node])
    state = original.run(max_batches=1)
    assert state["state"] == "BATCH_LIMIT"
    assert state["missing_catalog_class_count"] > 0
    lane = original.manifest["nodes"][0]
    engine = original._load_engine(lane)
    restored = runner(tmp_path, [node], resume=True)
    loaded = restored._load_engine(restored.manifest["nodes"][0])
    assert loaded.current_iteration == engine.current_iteration == 2
    assert loaded.rng.getstate() == engine.rng.getstate()
    assert loaded.root.visits == engine.root.visits
    assert loaded.discovered_classes == engine.discovered_classes
    assert loaded.geometry.full_oracle is restored.full_oracle
    path = tmp_path / lane["checkpoint"]["path"]
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        restored._load_engine(restored.manifest["nodes"][0])


def test_restart_keeps_novelty_but_resets_tree(tmp_path, inventory):
    node = inventory.by_id["BFS422-H-0000"]
    experiment = runner(tmp_path, [node])
    experiment.run(max_batches=1)
    engine = experiment._load_engine(experiment.manifest["nodes"][0])
    fresh = module.restart_tree(engine, 17, HANDOFF)
    assert len(fresh.nodes) == 1
    assert fresh.root.visits == 0
    assert fresh.discovered_classes is engine.discovered_classes
    assert fresh.shared_discovery is engine.shared_discovery
    assert fresh.geometry is engine.geometry
    assert fresh.current_iteration == 2
    assert fresh.stats["completion_tree_restarts"] == 1


def test_completed_node_is_independently_audited(tmp_path, inventory):
    node = inventory.by_id["BFS422-H-0078"]
    experiment = runner(tmp_path, [node], batch=10)
    state = experiment.run(max_batches=1)
    assert state["state"] == "COMPLETED"
    assert state["missing_catalog_class_count"] == 0
    lane = experiment.manifest["nodes"][0]
    assert lane["state"] == "COMPLETED"
    assert lane["checkpoint"] is None
    audit = json.loads((tmp_path / lane["audit_output"]).read_text(encoding="utf-8"))
    assert audit["failure_count"] == 0
    result = json.loads((tmp_path / lane["output"]).read_text(encoding="utf-8"))
    assert result["runtime"]["planned_iterations"] is None
    assert result["coverage"]["missing_target_ids"] == []


def test_resume_input_change_is_rejected(tmp_path, inventory):
    nodes = [inventory.by_id["BFS422-H-0000"]]
    runner(tmp_path, nodes).run(max_batches=1)
    with pytest.raises(ValueError, match="configuration"):
        runner(tmp_path, nodes, resume=True, batch=3)


def test_empty_checklist_is_not_a_claim_of_no_geometric_facets(tmp_path, inventory):
    node = next(n for n in inventory.nodes if not n.class_ids)
    state = runner(tmp_path, [node]).run()
    assert state["state"] == "COMPLETED"
    assert state["iterations_completed"] == 0
    assert state["empty_checklist_node_count"] == 1


def test_round_robin_cursor_is_not_advanced_by_invocation_limit(tmp_path, inventory):
    nodes = [inventory.by_id["BFS422-H-0000"], inventory.by_id["BFS422-H-0078"]]
    experiment = runner(tmp_path, nodes)
    experiment.run(max_batches=1)
    assert experiment.manifest["runtime"]["cursor"] == 1
    resumed = runner(tmp_path, nodes, resume=True)
    resumed.run(max_batches=1)
    assert resumed.manifest["nodes"][1]["iterations"] > 0


def test_audit_rejects_changed_lift_witness(tmp_path, inventory):
    node = inventory.by_id["BFS422-H-0078"]
    experiment = runner(tmp_path, [node], batch=10)
    experiment.run(max_batches=1)
    lane = experiment.manifest["nodes"][0]
    result = json.loads((tmp_path / lane["output"]).read_text(encoding="utf-8"))
    events = result["coverage"]["discoveries"]
    events[0]["original_support_word_hex"] = "0x0"
    problem = module.ReducedSubgroupProblem.load(HANDOFF, node.node_id)
    audit = module.audit_node(problem, events, HANDOFF, experiment.audit_oracle)
    assert audit["failure_count"] == 1
    assert "lift support" in audit["failures"][0]["error"]


def test_resumed_search_matches_uninterrupted_actions(tmp_path, inventory):
    node = inventory.by_id["BFS422-H-0000"]
    continuous = runner(tmp_path / "continuous", [node])
    continuous.run(max_batches=2)
    staged = runner(tmp_path / "staged", [node])
    staged.run(max_batches=1)
    resumed = runner(tmp_path / "staged", [node], resume=True)
    resumed.run(max_batches=1)
    normalize = lambda events: [
        {k: v for k, v in e.items() if k != "suite_elapsed_seconds"} for e in events
    ]
    left = continuous._load_engine(continuous.manifest["nodes"][0])
    right = resumed._load_engine(resumed.manifest["nodes"][0])
    assert normalize(left.discovery_events) == normalize(right.discovery_events)
    assert left.rng.getstate() == right.rng.getstate()
    assert left.class_frequencies == right.class_frequencies
    assert left.root.visits == right.root.visits


def test_failure_reports_last_committed_progress(tmp_path, inventory, monkeypatch):
    experiment = runner(tmp_path, [inventory.by_id["BFS422-H-0000"]])

    def fail(lane):
        experiment.manifest["runtime"]["iterations_completed"] = 99
        raise RuntimeError("injected failure")

    monkeypatch.setattr(experiment, "_advance", fail)
    with pytest.raises(RuntimeError, match="injected"):
        experiment.run()
    status = json.loads(experiment.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "FAILED"
    assert status["iterations_completed"] == 0


def test_low_disk_space_preserves_committed_checkpoint(tmp_path, inventory, monkeypatch):
    nodes = [inventory.by_id["BFS422-H-0000"]]
    runner(tmp_path, nodes).run(max_batches=1)
    experiment = runner(tmp_path, nodes, resume=True)
    checkpoint = experiment.manifest["nodes"][0]["checkpoint"]
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="disk space"):
        experiment.run()
    status = json.loads(experiment.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "FAILED"
    assert status["iterations_completed"] == 2
    assert experiment.manifest["nodes"][0]["checkpoint"] == checkpoint
    assert (tmp_path / checkpoint["path"]).exists()
