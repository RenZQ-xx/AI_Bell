from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

runner = importlib.import_module('mcts.422.long_fullspace_as_v31')


def reference():
    return {'config': asdict(runner.FullspaceASConfig(iterations=1000, seed=19)),
            'runtime': {'iterations_completed': 1000, 'elapsed_seconds': 5},
            'coverage': {'discoveries': [{'canonical_h_row': [1, 2]}]}}


def test_long_config_preserves_all_search_parameters():
    short = reference()
    config = asdict(runner.long_config(short, 10000))
    assert config['iterations'] == 10000
    assert all(config[key] == value for key, value in short['config'].items() if key != 'iterations')


@pytest.mark.parametrize('mode', ['budget', 'disabled', 'wall_cap', 'missing_as'])
def test_invalid_long_configuration(mode):
    short = reference()
    iterations = 1000 if mode == 'budget' else 10000
    if mode == 'disabled':
        short['config']['enable_lazy_as'] = False
    if mode == 'wall_cap':
        short['config']['max_wall_seconds'] = 10
    if mode == 'missing_as':
        del short['config']['enable_lazy_as']
    with pytest.raises(ValueError):
        runner.long_config(short, iterations)


def test_milestone_and_rolling_checkpoints(tmp_path):
    paths = runner.checkpoint_paths(tmp_path, 10000)
    assert len(paths) == 9 and 10000 not in paths
    assert paths[1000].name == 'checkpoint_i1000.json'
    assert paths[5000].name == 'checkpoint_i5000.json'
    assert paths[2000] == paths[9000] == tmp_path/'rolling.json'


def test_extended_long_run_retains_large_budget_milestones(tmp_path):
    paths = runner.checkpoint_paths(tmp_path, 250000)
    assert len(paths) == 249 and 250000 not in paths
    for iteration in (1000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000):
        assert paths[iteration].name == f'checkpoint_i{iteration}.json'
    assert paths[249000].name == 'rolling.json'


def test_comparison_uses_exact_canonical_rows():
    old = reference()
    new = reference()
    new['coverage']['discoveries'] = [{'canonical_h_row': [2, 1]}]
    comparison = runner.compare_classes(old, new)
    assert comparison['intersection'] == 0 and comparison['union'] == 2
    assert not comparison['native_config_differences']


@pytest.fixture
def fake_long_run(monkeypatch, tmp_path):
    save = runner.Bell422LazyASSearch._write_result
    output_dir = tmp_path/'long'
    observed = []
    result = reference()
    result['coverage']['discoveries'] = [{
        'canonical_h_row': [1, 2], 'catalog_index': None, 'iteration': 1001,
        'elapsed_seconds': 1, 'local_path': [{'source': 'fullspace_as_completion'}],
        'path': [{'source': 'fullspace_as_completion'}], 'episode_kind': 'continuation'}]
    result['runtime'] = {'iterations_completed': 1001, 'elapsed_seconds': 2}
    result['fullspace_as'] = {'executed_proofs': []}

    class FakeSearch:
        _write_result = staticmethod(save)

        def __init__(self, config, **kwargs):
            self.config = config
            self.started_at = time.perf_counter()-1
            self.as_budget_started = self.started_at-3
            self.discovered = {'test': {}}
            self.nodes = [1]
            self.as_bank = {}
            self.as_executed_proofs = {}
            self.as_compute_seconds = 0
            self.adjacency = SimpleNamespace(frames={}, rotation_cache={})

        def run_iteration(self, iteration):
            assert self.as_budget_started == self.started_at

        def run(self, *, checkpoint_paths):
            assert 1000 in checkpoint_paths
            self.run_iteration(1000)
            self.started_at += 0.01
            self.run_iteration(1001)
            result['config'] = asdict(self.config)
            return result

    def inspect_stage(stage):
        status = json.loads((output_dir/'status.json').read_text(encoding='utf-8'))
        observed.append(status)
        assert status['state'] == 'RUNNING' and status['audit_stage'] == stage

    def paths(*args, **kwargs):
        inspect_stage('paths')
        return {'failure_count': 0}

    def canonical(*args, **kwargs):
        inspect_stage('canonical')
        return {'canonical_audit': {'revalidated_terminal_count': 1,
            'duplicate_canonical_h_row_count': 0, 'duplicate_support_count': 0,
            'duplicate_class_id_count': 0, 'generator_invariance_failure_count': 0,
            'nonprimitive_canonical_rows': 0}}

    def analyze(*args, **kwargs):
        inspect_stage('discovery_paths')
        return {'windows': []}

    monkeypatch.setattr(runner, 'Bell422LazyASSearch', FakeSearch)
    monkeypatch.setattr(runner, 'audit_paths', paths)
    monkeypatch.setattr(runner, 'analyze_run', canonical)
    monkeypatch.setattr(runner, 'analyze_paths', analyze)
    monkeypatch.setattr(runner, 'deterministic_vertices', lambda: [])
    ref = tmp_path/'reference.json'
    ref.write_text(json.dumps(reference()), encoding='utf-8')
    return ref, output_dir, observed


def test_completion_waits_for_all_audits_and_path_analysis(fake_long_run):
    ref, directory, observed = fake_long_run
    status = runner.run_long(reference_config=ref, output_dir=directory, iterations=1001)
    assert status['state'] == 'COMPLETED' and status['phase'] == 'done'
    assert len(observed) == 3 and status['iterations_completed'] == 1001
    assert status['discovery_summary']['as_direct_final_step'] == 1
    assert (directory/'audit.json').exists() and (directory/'comparison.json').exists()
    assert (directory/'path_analysis.json').exists() and (directory/'report.md').exists()


def test_audit_error_preserves_result_and_failure_status(fake_long_run, monkeypatch):
    ref, directory, _ = fake_long_run
    def fail(*args, **kwargs):
        raise ValueError('audit error')
    monkeypatch.setattr(runner, 'audit_paths', fail)
    with pytest.raises(ValueError, match='audit error'):
        runner.run_long(reference_config=ref, output_dir=directory, iterations=1001)
    status = json.loads((directory/'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'FAILED' and status['phase'] == 'audits'
    assert (directory/'v31_as.json').exists()


def test_existing_directory_is_never_replaced(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run_long(reference_config=tmp_path/'missing', output_dir=tmp_path, iterations=10000)
