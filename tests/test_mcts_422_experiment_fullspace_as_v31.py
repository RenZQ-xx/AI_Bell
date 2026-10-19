from __future__ import annotations

import importlib
import json
from dataclasses import asdict

import pytest

experiment = importlib.import_module('mcts.422.experiment_fullspace_as_v31')


@pytest.fixture
def fake_searches(monkeypatch):
    seen = []
    save = experiment.Bell422EarlyNoopSearch._write_result

    class Native:
        _write_result = staticmethod(save)
        is_as = False

        def __init__(self, config, **kwargs):
            seen.append((self.is_as, config, kwargs))
            self.config = config

        def run(self):
            rows = [[1, 2], [2, 3]] if self.is_as else [[1, 2]]
            result = {
                'coverage': {'discoveries': [
                    {'canonical_h_row': row, 'catalog_index': 1 if i == 0 else None}
                    for i, row in enumerate(rows)]},
                'runtime': {'elapsed_seconds': 2 if self.is_as else 1,
                            'iterations_completed': self.config.iterations},
            }
            if self.is_as:
                result['fullspace_as'] = {'executed_proofs': [{'test': True}],
                                         'banked_ridges': 2, 'compute_seconds': 0.2}
            return result

    class WithAS(Native):
        is_as = True

    def canonical(path, **kwargs):
        count = len(json.loads(path.read_text(encoding='utf-8'))['coverage']['discoveries'])
        return {'canonical_audit': {
            'revalidated_terminal_count': count, 'duplicate_canonical_h_row_count': 0,
            'duplicate_support_count': 0, 'duplicate_class_id_count': 0,
            'generator_invariance_failure_count': 0, 'nonprimitive_canonical_rows': 0}}

    monkeypatch.setattr(experiment, 'Bell422EarlyNoopSearch', Native)
    monkeypatch.setattr(experiment, 'Bell422LazyASSearch', WithAS)
    monkeypatch.setattr(experiment, 'analyze_run', canonical)
    monkeypatch.setattr(experiment, 'audit_paths', lambda *a, **k: {'failure_count': 0})
    monkeypatch.setattr(experiment, 'deterministic_vertices', lambda: 'original_points')
    monkeypatch.setattr(experiment, 'audit_adjacency_proof', lambda points, proof: True)
    return seen


def test_pair_uses_reference_config_without_reusing_saved_discoveries(tmp_path, fake_searches):
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps({'config': {'new_class_reward': 12, 'iterations': 999},
                                    'coverage': {'discoveries': 'must never be seeded'}}),
                         encoding='utf-8')
    summary = experiment.run_experiment(reference_config=reference, iterations=7, seed=8,
                                        output_dir=tmp_path/'pair')
    assert summary['state'] == 'COMPLETED' and summary['phase'] == 'done'
    assert [is_as for is_as, _, _ in fake_searches] == [False, True]
    native, new = [asdict(config) for _, config, _ in fake_searches]
    assert all(new[key] == value for key, value in native.items())
    assert native['iterations'] == 7 and native['seed'] == 8 and native['new_class_reward'] == 12
    assert summary['comparison']['intersection'] == 1
    assert summary['comparison']['v30_only'] == 0
    assert summary['comparison']['v31_only'] == 1
    assert summary['runs'][1]['unmatched'] == 1
    assert summary['runs'][1]['as_distinct_executed'] == 1
    assert not summary['comparison']['catalog_is_complete']


def test_pair_rejects_existing_directory(tmp_path, fake_searches):
    with pytest.raises(FileExistsError):
        experiment.run_experiment(reference_config=None, iterations=7, seed=8,
                                  output_dir=tmp_path)
    assert not fake_searches


def test_audit_failure_preserves_failed_status_and_partial_result(tmp_path, fake_searches, monkeypatch):
    monkeypatch.setattr(experiment, 'audit_paths', lambda *a, **k: {'failure_count': 1})
    directory = tmp_path/'failed'
    with pytest.raises(ValueError, match='audit failed'):
        experiment.run_experiment(reference_config=None, iterations=7, seed=8,
                                  output_dir=directory)
    status = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert status['state'] == 'FAILED' and status['phase'] == 'audits'
    assert not status['runs'][0]['audit_passed']
    assert (directory/'v30.json').exists() and (directory/'v30.audit.json').exists()
    assert len(fake_searches) == 1
