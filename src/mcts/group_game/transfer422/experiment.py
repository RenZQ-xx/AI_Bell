from __future__ import annotations

"""Fresh-process 322 transfer benchmark with native baselines and full audits."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..search import _write_json
from .audit import audit_result


def compare(old, new):
    a, b = (set(p['summary']['discovered_class_ids']) for p in (old, new))
    return {'intersection': sorted(a & b), 'baseline_only': sorted(a-b),
        'candidate_only': sorted(b-a), 'baseline_count': len(a), 'candidate_count': len(b)}


def run_experiment(output_dir, iterations, seeds, profiles):
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite experiment directory')
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    status = {'state': 'RUNNING', 'iterations_limit': iterations, 'seeds': seeds, 'completed': []}
    _write_json(output_dir/'status.json', status)
    cases, results = [], {}
    try:
        for seed_index, seed in enumerate(seeds):
            order = profiles if seed_index % 2 == 0 else list(reversed(profiles))
            for profile in order:
                stem = output_dir/f'{profile}_i{iterations}_s{seed}'
                path = stem.with_suffix('.json')
                status.update(current_seed=seed, current_profile=profile, phase='search')
                _write_json(output_dir/'status.json', status)
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
                command = [sys.executable, '-u', '-m', 'mcts.group_game.transfer422.search',
                    '--iterations', str(iterations), '--seed', str(seed), '--profile', profile,
                    '--output', str(path.resolve())]
                before = time.perf_counter()
                with stem.with_suffix('.log').open('w', encoding='utf-8') as stdout, \
                        stem.with_suffix('.stderr.log').open('w', encoding='utf-8') as stderr:
                    subprocess.run(command, cwd=Path(__file__).resolve().parents[3], env=env,
                                   stdout=stdout, stderr=stderr, check=True)
                search_process_wall = time.perf_counter()-before
                payload = json.loads(path.read_text(encoding='utf-8'))
                status['phase'] = 'audit'
                _write_json(output_dir/'status.json', status)
                audit = audit_result(payload)
                _write_json(stem.with_suffix('.audit.json'), audit)
                if not audit['passed']:
                    raise ValueError(f'failed audit: {profile}/{seed}: {audit["failures"][:2]}')
                s = payload['summary']
                events = payload['discoveries']
                row = {'seed': seed, 'profile': profile, 'result': str(path.resolve()),
                    'iterations_completed': s['iterations_completed'], 'class_count': s['discovered_class_count'],
                    'missing': s['missing_target_class_ids'], 'search_seconds': s['elapsed_seconds'],
                    'search_process_wall_seconds': search_process_wall, 'audit_seconds': audit['elapsed_seconds'],
                    'last_discovery_iteration': events[-1]['iteration'] if events else None,
                    'class_discovery_iterations': {str(e['class_id']): e['iteration'] for e in events},
                    'as_final_discoveries': sum(bool(e['path']) and e['path'][-1]['source'] == 'fullspace_as_completion' for e in events),
                    'continuation_discoveries': sum(e.get('continuation_origin', {}).get('episode_kind') == 'continuation' for e in events),
                    'transfer': {k: v for k, v in payload.get('transfer422', {}).items()
                                 if k not in ('as_executed_proofs', 'as_geometry')},
                    'as_proofs_audited': audit['all_executed_as_proofs_checked']}
                cases.append(row)
                results[seed, profile] = {'summary': s}
                status['completed'].append(row)
                _write_json(output_dir/'status.json', status)
                print(json.dumps({k: row[k] for k in ('seed', 'profile', 'class_count',
                    'iterations_completed', 'search_seconds', 'missing')}, ensure_ascii=False), flush=True)
        comparisons = {str(seed): {profile: compare(results[seed, 'baseline'], results[seed, profile])
            for profile in profiles if profile != 'baseline'} for seed in seeds} if 'baseline' in profiles else {}
        summary = {'cases': cases, 'comparisons': comparisons,
            'wall_seconds': time.perf_counter()-started, 'iterations_limit': iterations,
            'fresh_process_per_case': True, 'native_v25_defaults_preserved': True,
            'as_is_time_gated_not_strict_seed_reproducible': True}
        _write_json(output_dir/'summary.json', summary)
        status.update(state='COMPLETED', phase='done', wall_seconds=summary['wall_seconds'])
        _write_json(output_dir/'status.json', status)
        return summary
    except BaseException as error:
        status.update(state='FAILED', error=f'{type(error).__name__}: {error}')
        _write_json(output_dir/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[3222026, 673963275, 1706386020])
    parser.add_argument('--profiles', nargs='+', choices=('baseline', 'continuation', 'as_only', 'full'),
                        default=['baseline', 'full'])
    args = parser.parse_args()
    run_experiment(args.output_dir.resolve(), args.iterations, args.seeds, args.profiles)


if __name__ == '__main__':
    main()
