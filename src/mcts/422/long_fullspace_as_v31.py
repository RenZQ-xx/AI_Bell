from __future__ import annotations

"""One checkpointed v31 search, followed by independent full-space audits."""

import argparse
import gc
import json
import os
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .analyze_continuation_paths import analyze_paths
from .analyze_quotient_run import analyze_run
from .atlas import DEFAULT_HANDOFF_ROOT
from .audit_continuation_run import audit_paths
from .continuation_search_v31 import Bell422LazyASSearch, FullspaceASConfig
from .problem import deterministic_vertices
from .recursive_adjacency import audit_adjacency_proof


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def long_config(reference, iterations):
    if reference['config'].get('enable_lazy_as') is not True:
        raise ValueError('reference must contain an enabled v31 AS configuration')
    if iterations <= reference['config']['iterations']:
        raise ValueError('long budget must exceed the reference short budget')
    values = asdict(FullspaceASConfig())
    values.update({key: value for key, value in reference['config'].items() if key in values})
    values['iterations'] = iterations
    config = FullspaceASConfig(**values)
    if not config.enable_lazy_as or config.max_wall_seconds is not None:
        raise ValueError('expected enabled v31 AS with no wall-time cap')
    return config


def checkpoint_paths(directory, iterations):
    milestones = {1000, 5000, 10000, 25000, 50000, 75000} | set(range(100000, iterations, 50000))
    return {
        i: directory/(f'checkpoint_i{i}.json' if i in milestones else 'rolling.json')
        for i in range(1000, iterations, 1000)
    }


def compare_classes(reference, candidate):
    old, new = ({tuple(event['canonical_h_row']) for event in result['coverage']['discoveries']}
                for result in (reference, candidate))
    return {
        'reference_classes': len(old), 'candidate_classes': len(new),
        'intersection': len(old & new), 'reference_only': len(old-new),
        'candidate_only': len(new-old), 'union': len(old | new),
        'union_uses_two_budgets': True,
        'reference_runtime': reference['runtime'], 'candidate_runtime': candidate['runtime'],
        'native_config_differences': {
            key: {'reference': value, 'candidate': candidate['config'].get(key)}
            for key, value in reference['config'].items()
            if key not in {'iterations', 'progress_interval'} and candidate['config'].get(key) != value
        },
    }


def discovery_summary(result):
    events = result['coverage']['discoveries']
    has_as = lambda path: any(action.get('source', '').startswith('fullspace_as_') for action in path)
    last = max(events, key=lambda event: event['iteration']) if events else None
    return {
        'canonical_classes': len(events),
        'catalog_matched': sum(event['catalog_index'] is not None for event in events),
        'catalog_unmatched': sum(event['catalog_index'] is None for event in events),
        'as_direct_final_step': sum(bool(event.get('local_path')) and
            event['local_path'][-1]['source'] == 'fullspace_as_completion' for event in events),
        'as_in_local_path': sum(has_as(event.get('local_path', [])) for event in events),
        'as_in_historical_prefix_only': sum(has_as(event['path']) and
            not has_as(event.get('local_path', [])) for event in events),
        'no_as_in_saved_path': sum(not has_as(event['path']) for event in events),
        'episode_new_classes': dict(Counter(event.get('episode_kind') for event in events)),
        'local_final_action_sources': dict(Counter(
            event['local_path'][-1]['source'] if event.get('local_path') else 'empty' for event in events)),
        'last_discovery_iteration': last['iteration'] if last else None,
        'last_discovery_seconds': last['elapsed_seconds'] if last else None,
        'per_1000_new_classes': [sum(i < event['iteration'] <= i+1000 for event in events)
            for i in range(0, result['runtime']['iterations_completed'], 1000)],
    }


def run_long(*, reference_config, output_dir, iterations, baseline=None,
             handoff_root=DEFAULT_HANDOFF_ROOT):
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite a long-run directory')
    reference = load(reference_config)
    config = long_config(reference, iterations)
    output_dir.mkdir(parents=True)
    save = Bell422LazyASSearch._write_result
    output = output_dir/'v31_as.json'
    status_path = output_dir/'status.json'
    started = time.perf_counter()
    status = {
        'state': 'RUNNING', 'phase': 'initializing', 'pid': os.getpid(),
        'started_at': now(), 'updated_at': now(), 'config': asdict(config),
        'planned_iterations': iterations, 'iterations_completed': 0, 'class_count': 0,
        'reference_config': str(reference_config.resolve()), 'output': str(output.resolve()),
        'wall_time_limit_seconds': None, 'catalog_is_complete': False,
        'reference_classes_used_for_search': False, 'snapshots_are_restartable': False,
    }
    save(status, status_path)

    def update(**values):
        status.update(values, updated_at=now(), host_wall_seconds=time.perf_counter()-started)
        save(status, status_path)

    class TrackedSearch(Bell422LazyASSearch):
        def run_iteration(self, iteration):
            # The base runner excludes checkpoint writing from its search clock.
            if self.as_budget_started is not None:
                self.as_budget_started = self.started_at
            super().run_iteration(iteration)
            if iteration % 100 == 0 or iteration == self.config.iterations:
                update(phase='search', iterations_completed=iteration,
                    class_count=len(self.discovered), node_count=len(self.nodes),
                    elapsed_search_seconds=time.perf_counter()-self.started_at,
                    as_compute_seconds=self.as_compute_seconds,
                    as_banked_ridges=sum(map(len, self.as_bank.values())),
                    as_distinct_executed=len(self.as_executed_proofs),
                    as_geometry_frames=len(self.adjacency.frames),
                    as_rotation_cache_entries=len(self.adjacency.rotation_cache))

    try:
        engine = TrackedSearch(config, handoff_root=handoff_root)
        update(phase='search')
        result = engine.run(checkpoint_paths=checkpoint_paths(output_dir, iterations))
        save(result, output)
        update(phase='audits', audit_stage='paths', runtime=result['runtime'],
            discovery_summary=discovery_summary(result), search_finished_at=now(),
            iterations_completed=result['runtime']['iterations_completed'],
            class_count=len(result['coverage']['discoveries']))
        del engine
        gc.collect()
        audit_started = time.perf_counter()
        paths = audit_paths(output, handoff_root=handoff_root)
        save(paths, output_dir/'paths.json')
        update(audit_stage='canonical')
        analysis = analyze_run(output, handoff_root=handoff_root)
        save(analysis, output_dir/'analysis.json')
        proofs = result['fullspace_as']['executed_proofs']
        failures = []
        points = deterministic_vertices()
        update(audit_stage='as_rotation_chains', as_proofs_checked=0, as_proofs_total=len(proofs))
        for i, proof in enumerate(proofs, 1):
            try:
                audit_adjacency_proof(points, proof)
            except (ValueError, KeyError, TypeError) as error:
                failures.append({'proof': i-1, 'error': str(error)})
            if i % 25 == 0 or i == len(proofs):
                update(as_proofs_checked=i, as_proof_failures=len(failures))
        checked = analysis['canonical_audit']
        verified = (paths['failure_count'] == 0 and not failures
            and checked['revalidated_terminal_count'] == len(result['coverage']['discoveries'])
            and all(checked[key] == 0 for key in ('duplicate_canonical_h_row_count',
                'duplicate_support_count', 'duplicate_class_id_count',
                'generator_invariance_failure_count', 'nonprimitive_canonical_rows')))
        audit = {'passed': verified, 'paths': paths, 'canonical': checked,
                 'as_proofs_checked': len(proofs), 'as_failures': failures,
                 'audit_seconds': time.perf_counter()-audit_started}
        save(audit, output_dir/'audit.json')
        update(phase='analysis', audit_stage='discovery_paths', audit_passed=verified,
               audit_seconds=audit['audit_seconds'])
        detailed_paths = analyze_paths(output, handoff_root=handoff_root)
        save(detailed_paths, output_dir/'path_analysis.json')
        comparisons = {'short_v31': compare_classes(reference, result)}
        if baseline is not None:
            comparisons['saved_v30'] = compare_classes(load(baseline), result)
            comparisons['saved_v30']['reference_path'] = str(baseline.resolve())
        save(comparisons, output_dir/'comparison.json')
        report = [f'# Bell 422 v31 i{iterations} Long Run', '',
            f"- Completed iterations: {result['runtime']['iterations_completed']}.",
            f"- Canonical classes: {status['class_count']}.",
            f"- Search seconds: {result['runtime']['elapsed_seconds']:.3f}.",
            f"- Audit passed: {verified}; audit seconds: {audit['audit_seconds']:.3f}.",
            '- Catalog is partial; the complete 422 class count is unknown.', '',
            '## Discovery Summary', '', '```json',
            json.dumps(status['discovery_summary'], indent=2), '```', '',
            '## Thousand-Iteration Windows', '',
            '| End iteration | New classes | Search seconds | Classes/hour |',
            '| --- | ---: | ---: | ---: |']
        report.extend(f"| {w['end_iteration']} | {w['new_classes']} | {w['search_seconds']:.3f} | "
                      f"{w['classes_per_hour']:.2f} |" for w in detailed_paths['windows'])
        report.extend(['', '## Saved-Result Comparisons', '', '```json',
                       json.dumps(comparisons, indent=2), '```', ''])
        (output_dir/'report.md').write_text('\n'.join(report), encoding='utf-8')
        update(state='COMPLETED' if verified else 'AUDIT_FAILED', phase='done',
               finished_at=now(), host_wall_seconds_including_audits=time.perf_counter()-started)
        return status
    except BaseException as error:
        update(state='FAILED', error=f'{type(error).__name__}: {error}',
               failed_at=now(), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--handoff-root', type=Path, default=DEFAULT_HANDOFF_ROOT)
    args = parser.parse_args()
    status = run_long(**vars(args))
    print(json.dumps(status, indent=2), flush=True)
    if status['state'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
