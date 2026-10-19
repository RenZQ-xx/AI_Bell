from __future__ import annotations

"""Paired original-space v30/v31 searches with independent post-search audits."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .analyze_quotient_run import analyze_run
from .atlas import DEFAULT_HANDOFF_ROOT
from .audit_continuation_run import audit_paths
from .continuation_search_v30 import Bell422EarlyNoopSearch, EarlyNoopSearchConfig
from .continuation_search_v31 import Bell422LazyASSearch, FullspaceASConfig
from .problem import deterministic_vertices
from .recursive_adjacency import audit_adjacency_proof


def run_experiment(*, reference_config, iterations, seed, output_dir, handoff_root=DEFAULT_HANDOFF_ROOT):
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite experiment directory')
    output_dir.mkdir(parents=True)
    save = Bell422EarlyNoopSearch._write_result
    summary = {'state':'RUNNING', 'phase':'search', 'iterations':iterations, 'seed':seed,
               'scope':'original 256 vertices / dimension 80; no reduced-node search', 'runs':[]}
    summary_path = output_dir/'summary.json'
    save(summary, summary_path)
    started = time.perf_counter()
    try:
        values = asdict(EarlyNoopSearchConfig())
        if reference_config is not None:
            reference = json.loads(reference_config.read_text(encoding='utf-8'))
            values.update({k:v for k,v in reference['config'].items() if k in values})
        values.update(iterations=iterations, seed=seed, progress_interval=100)
        for name, engine_type, config in (
            ('v30', Bell422EarlyNoopSearch, EarlyNoopSearchConfig(**values)),
            ('v31_as', Bell422LazyASSearch, FullspaceASConfig(**values)),
        ):
            summary.update(phase='search', current=name)
            save(summary, summary_path)
            print(json.dumps({'start':name,'iterations':iterations,'seed':seed}), flush=True)
            engine = engine_type(config, handoff_root=handoff_root)
            result = engine.run()
            result_path = output_dir/f'{name}.json'
            save(result, result_path)
            summary.update(phase='audits')
            save(summary, summary_path)
            audit_started = time.perf_counter()
            paths = audit_paths(result_path, handoff_root=handoff_root)
            canonical = analyze_run(result_path, handoff_root=handoff_root)
            proofs = result.get('fullspace_as', {}).get('executed_proofs', [])
            failures = []
            for i,proof in enumerate(proofs):
                try:
                    audit_adjacency_proof(deterministic_vertices(), proof)
                except (ValueError, KeyError, TypeError) as error:
                    failures.append({'proof':i,'error':str(error)})
            checked = canonical['canonical_audit']
            valid = (paths['failure_count']==0 and not failures
                and checked['revalidated_terminal_count']==len(result['coverage']['discoveries'])
                and all(checked[key]==0 for key in ('duplicate_canonical_h_row_count',
                    'duplicate_support_count','duplicate_class_id_count','generator_invariance_failure_count',
                    'nonprimitive_canonical_rows')))
            audit = {'passed':valid, 'paths':paths, 'canonical':checked,
                     'as_proofs_checked':len(proofs), 'as_failures':failures,
                     'audit_seconds':time.perf_counter()-audit_started}
            save(audit, output_dir/f'{name}.audit.json')
            save(canonical, output_dir/f'{name}.analysis.json')
            record = {'mode':name,'result':str(result_path.resolve()), 'config':asdict(config),
                'runtime':result['runtime'], 'canonical_classes':len(result['coverage']['discoveries']),
                'matched':sum(e['catalog_index'] is not None for e in result['coverage']['discoveries']),
                'unmatched':sum(e['catalog_index'] is None for e in result['coverage']['discoveries']),
                'canonical_h_rows':[e['canonical_h_row'] for e in result['coverage']['discoveries']],
                'as_compute_seconds':result.get('fullspace_as',{}).get('compute_seconds',0),
                'banked_ridges':result.get('fullspace_as',{}).get('banked_ridges',0),
                'as_distinct_executed':len(proofs), 'audit_passed':valid,
                'audit_seconds':audit['audit_seconds']}
            summary['runs'].append(record)
            save(summary, summary_path)
            print(json.dumps({k:v for k,v in record.items() if k not in ('config','canonical_h_rows')}), flush=True)
            if not valid:
                raise ValueError(f'independent audit failed: {name}')
        old,new=summary['runs']
        a,b=({tuple(row) for row in r['canonical_h_rows']} for r in (old,new))
        summary.update(state='COMPLETED',phase='done',comparison={
            'intersection':len(a&b),'v30_only':len(a-b),'v31_only':len(b-a),
            'same_native_config':all(new['config'][k]==v for k,v in old['config'].items()),
            'search_time_ratio_v31_over_v30':new['runtime']['elapsed_seconds']/old['runtime']['elapsed_seconds'],
            'catalog_is_complete':False})
    except BaseException as error:
        summary.update(state='FAILED',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        summary['wall_seconds_including_audits']=time.perf_counter()-started
        save(summary, summary_path)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=4222025)
    parser.add_argument('--reference-config',type=Path)
    parser.add_argument('--handoff-root',type=Path,default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    summary=run_experiment(**vars(args))
    print(json.dumps({'state':summary['state'],'comparison':summary['comparison']}))


if __name__=='__main__':
    main()
