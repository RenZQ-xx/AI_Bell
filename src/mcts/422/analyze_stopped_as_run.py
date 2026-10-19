from __future__ import annotations

"""Offline analysis of one stopped full-space AS checkpoint; never starts search."""

import argparse
import gc
import json
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .analyze_continuation_paths import analyze_paths
from .analyze_quotient_run import analyze_run
from .atlas import DEFAULT_HANDOFF_ROOT
from .audit_continuation_run import audit_paths
from .continuation_search_v31 import Bell422LazyASSearch
from .long_fullspace_as_v31 import discovery_summary
from .problem import deterministic_vertices
from .recursive_adjacency import audit_adjacency_proof


def sample_indices(total, count):
    if count < 1:
        raise ValueError('positive sample count required')
    count = min(total, count)
    if count <= 1:
        return list(range(count))
    return [i*(total-1)//(count-1) for i in range(count)]


def window_summary(payload, width):
    events = payload['coverage']['discoveries']
    elapsed = {item['iteration']: item['elapsed_seconds']
               for item in payload['continuation_game']['growth_windows']}
    completed = payload['runtime']['iterations_completed']
    elapsed[0] = 0
    elapsed[completed] = payload['runtime']['elapsed_seconds']
    rows = []
    for start in range(0, completed, width):
        end = min(start+width, completed)
        selected = [e for e in events if start < e['iteration'] <= end]
        seconds = elapsed[end]-elapsed[start]
        matched = sum(e['catalog_index'] is not None for e in selected)
        rows.append({'start_iteration': start+1, 'end_iteration': end,
            'new_classes': len(selected), 'catalog_matched': matched,
            'catalog_unmatched': len(selected)-matched, 'search_seconds': seconds,
            'seconds_per_iteration': seconds/(end-start),
            'classes_per_hour': len(selected)*3600/seconds if seconds else None,
            'continuation_new_classes': sum(e['episode_kind'] == 'continuation' for e in selected),
            'as_direct_final': sum(bool(e.get('local_path')) and
                e['local_path'][-1]['source'] == 'fullspace_as_completion' for e in selected)})
    return rows


def set_comparison(events, reference, cutoff):
    chosen = [e for e in reference['coverage']['discoveries'] if e['iteration'] <= cutoff]
    old, new = ({tuple(e['canonical_h_row']) for e in es} for es in (chosen, events))
    return {'reference_events_through_iteration': cutoff, 'reference_classes': len(old),
        'reference_matched': sum(e['catalog_index'] is not None for e in chosen),
        'candidate_classes': len(new), 'intersection': len(old & new),
        'reference_only': len(old-new), 'candidate_only': len(new-old),
        'union': len(old | new), 'union_uses_multiple_runs': True}


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['windows_10000']
    x = [r['end_iteration']/1000 for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].bar(x, [r['catalog_unmatched'] for r in rows], width=7, color='#3978a8',
                label='Catalog unmatched')
    axes[0].bar(x, [r['catalog_matched'] for r in rows], width=7,
                bottom=[r['catalog_unmatched'] for r in rows], color='#3d935a', label='Catalog matched')
    axes[0].set_ylabel('New canonical classes')
    axes[0].legend(loc='upper left')
    axes[1].plot(x, [r['seconds_per_iteration'] for r in rows], 'o-', color='#c54b46')
    axes[1].set_ylabel('Seconds / iteration')
    axes[2].plot(x, [r['classes_per_hour'] for r in rows], 'o-', color='#3978a8')
    axes[2].set_ylabel('New classes / hour')
    axes[2].set_xlabel('Iteration window end (thousands)')
    axes[2].set_xticks(x)
    for ax in axes:
        ax.grid(axis='y', alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle('Bell 4-2-2 v31: stopped at 90,000 iterations', fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(*, checkpoint, output_dir, references, proof_sample=128,
                 handoff_root=DEFAULT_HANDOFF_ROOT):
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite an analysis directory')
    output_dir.mkdir(parents=True)
    save = Bell422LazyASSearch._write_result
    started = time.perf_counter()
    state = {'state': 'RUNNING', 'phase': 'load', 'checkpoint': str(checkpoint.resolve()),
             'search_started': False, 'full_as_proof_audit': False}
    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.perf_counter()-started)
        save(state, output_dir/'status.json')
    update()
    try:
        payload = json.loads(checkpoint.read_text(encoding='utf-8'))
        proofs = payload['fullspace_as'].pop('executed_proofs')
        indices = sample_indices(len(proofs), proof_sample)
        samples = [(i, proofs[i]) for i in indices]
        proof_total = len(proofs)
        del proofs
        gc.collect()
        events = payload['coverage']['discoveries']
        summary = discovery_summary(payload)
        summary.update(runtime=payload['runtime'], config=payload['config'],
            windows_1000=window_summary(payload, 1000), windows_10000=window_summary(payload, 10000),
            terminal_final_source_matched=dict(Counter(e['path'][-1]['source'] for e in events
                                                       if e['catalog_index'] is not None)),
            terminal_final_source_unmatched=dict(Counter(e['path'][-1]['source'] for e in events
                                                         if e['catalog_index'] is None)),
            root_pattern_yields=dict(Counter(e['path'][0]['pattern_id'] for e in events)),
            root_structure_yields=dict(Counter(e['path'][0]['structure'] for e in events)),
            fullspace_as=payload['fullspace_as'], tree=payload['tree'],
            rank_oracle=payload['rank_oracle'], facet_oracle=payload['facet_oracle'],
            early_noop_guard=payload['early_noop_guard'], search_stats=payload['search_stats'],
            atlas=payload['atlas'], quotient_game=payload['quotient_game'],
            knowledge=payload['knowledge'], recorded_as_proofs=proof_total)
        save(summary, output_dir/'summary.json')
        try:
            plot_summary(summary, output_dir/'growth_and_speed.png')
        except ModuleNotFoundError as error:
            if error.name != 'matplotlib':
                raise
            update(plot_unavailable='matplotlib is not installed in the project virtual environment')
        update(phase='full_path_audit', classes=len(events), as_proof_sample=len(samples))
        paths = audit_paths(checkpoint, handoff_root=handoff_root, payload=payload)
        save(paths, output_dir/'paths.json')
        update(phase='terminal_canonical_audit', path_failures=paths['failure_count'])
        canonical = analyze_run(checkpoint, handoff_root=handoff_root, payload=payload)
        save(canonical, output_dir/'canonical_analysis.json')
        update(phase='discovery_path_analysis')
        detailed = analyze_paths(checkpoint, handoff_root=handoff_root, payload=payload)
        save(detailed, output_dir/'path_analysis.json')
        update(phase='sampled_as_audit')
        failures = []
        points = deterministic_vertices()
        for done, (index, proof) in enumerate(samples, 1):
            try:
                audit_adjacency_proof(points, proof)
            except (ValueError, KeyError, TypeError) as error:
                failures.append({'proof_index': index, 'error': str(error)})
            if done % 16 == 0 or done == len(samples):
                update(as_proofs_checked=done, as_sample_failures=len(failures))
        sample_audit = {'total_recorded': proof_total, 'checked': len(samples),
            'indices': indices, 'selection': 'deterministic evenly spaced arrival-order strata',
            'failure_count': len(failures), 'failures': failures, 'full_audit': len(samples) == proof_total}
        save(sample_audit, output_dir/'as_sample_audit.json')
        update(phase='saved_result_comparisons')
        comparisons = {}
        for path in references:
            reference = json.loads(path.read_text(encoding='utf-8'))
            cutoff = min(payload['runtime']['iterations_completed'], reference['runtime']['iterations_completed'])
            comparisons[str(path)] = set_comparison(events, reference, cutoff)
            del reference
            gc.collect()
        save(comparisons, output_dir/'comparisons.json')
        checked = canonical['canonical_audit']
        valid = (paths['failure_count'] == 0 and not failures
            and checked['revalidated_terminal_count'] == len(events)
            and all(checked[k] == 0 for k in ('duplicate_canonical_h_row_count',
                'duplicate_support_count', 'duplicate_class_id_count', 'nonprimitive_canonical_rows',
                'generator_invariance_failure_count', 'stored_canonical_h_row_mismatch_count',
                'stored_catalog_index_mismatch_count', 'stored_support_closure_mismatch_count')))
        update(state='COMPLETED' if valid else 'AUDIT_FAILED', phase='done',
            terminals_and_paths_passed=valid, canonical_audit=checked,
            sampled_as_passed=not failures, full_as_proof_audit=sample_audit['full_audit'])
        return state
    except BaseException as error:
        update(state='FAILED', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--references', type=Path, nargs='*', default=[])
    parser.add_argument('--proof-sample', type=int, default=128)
    parser.add_argument('--handoff-root', type=Path, default=DEFAULT_HANDOFF_ROOT)
    result = run_analysis(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
