import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

runner = importlib.import_module('mcts.422.analyze_stopped_as_run')
canonical = importlib.import_module('mcts.422.analyze_quotient_run')
paths = importlib.import_module('mcts.422.audit_continuation_run')
detail = importlib.import_module('mcts.422.analyze_continuation_paths')


@pytest.mark.parametrize('total,count', [(0, 128), (1, 128), (5, 128), (1000, 128), (5, 1)])
def test_proof_strata_are_unique_bounded_and_include_endpoints(total, count):
    indices = runner.sample_indices(total, count)
    assert len(indices) == min(total, count) and indices == sorted(set(indices))
    assert all(0 <= i < total for i in indices)
    if len(indices) > 1:
        assert indices[0] == 0 and indices[-1] == total-1


def event(iteration, row, matched=None):
    return {'iteration': iteration, 'canonical_h_row': row, 'catalog_index': matched,
            'episode_kind': 'continuation', 'local_path': []}


def test_windows_use_actual_runtime_and_nonoverlapping_boundaries():
    payload = {'coverage': {'discoveries': [event(1, [1], 0), event(1000, [2]),
        event(1001, [3]), event(2000, [4], 1)]},
        'runtime': {'iterations_completed': 2000, 'elapsed_seconds': 50},
        'continuation_game': {'growth_windows': [
            {'iteration': 1000, 'elapsed_seconds': 20}, {'iteration': 2000, 'elapsed_seconds': 49}]}}
    rows = runner.window_summary(payload, 1000)
    assert [r['new_classes'] for r in rows] == [2, 2]
    assert [r['search_seconds'] for r in rows] == [20, 30]
    assert [r['catalog_matched'] for r in rows] == [1, 1]


def test_comparison_filters_reference_to_actual_cutoff():
    reference = {'coverage': {'discoveries': [event(1, [1]), event(99, [2]), event(101, [3])]}}
    result = runner.set_comparison([event(1, [2]), event(2, [4])], reference, 100)
    assert result['reference_classes'] == 2 and result['intersection'] == 1
    assert result['reference_only'] == result['candidate_only'] == 1


def test_path_analyzers_accept_preloaded_payload_without_reading_a_file():
    payload = {'coverage': {'discoveries': []}, 'config': {'extended_max_corrections': 8},
               'continuation_game': {'anchors': [], 'growth_windows': []}}
    missing = Path('this-file-does-not-exist.json')
    assert paths.audit_paths(missing, payload=payload)['paths_checked'] == 0
    assert detail.analyze_paths(missing, payload=payload)['class_count'] == 0


@pytest.mark.parametrize('mismatch', [False, True])
def test_terminal_audit_checks_saved_rows_catalog_and_closure(monkeypatch, mismatch):
    row = (1, *([0]*80))
    facet = SimpleNamespace(valid=True, canonical_h_row=row, support_word=3, catalog_index=7)
    monkeypatch.setattr(canonical, 'FacetOracle', lambda **kw: SimpleNamespace(validate=lambda word: facet))
    monkeypatch.setattr(canonical, '_catalog_rows', lambda root: [])
    monkeypatch.setattr(canonical, '_generator_invariance_failures', lambda rows: [])
    e = event(1, [2]*81 if mismatch else list(row), 8 if mismatch else 7)
    e.update(class_id='X', support_word_hex='0x4' if mismatch else '0x3', support_size=2,
             depth=1, path=[{'source': 'fullspace_supporting_face_lp', 'pattern_id': 'id', 'structure': '1'}])
    payload = {'coverage': {'discoveries': [e]}}
    audit = canonical.analyze_run(Path('missing.json'), handoff_root=Path('.'), payload=payload)['canonical_audit']
    assert audit['stored_canonical_h_row_mismatch_count'] == int(mismatch)
    assert audit['stored_catalog_index_mismatch_count'] == int(mismatch)
    assert audit['stored_support_closure_mismatch_count'] == int(mismatch)
