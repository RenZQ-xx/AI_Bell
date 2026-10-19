from __future__ import annotations

"""Independent terminal, local/history path and executed AS proof revalidation."""

import time

import numpy as np

from baseline.scorer import exact_class_id
from ..model import support_word_to_key
from ..search import GroupGameConfig, _make_singleton_scorer
from .search import adjacency_module


def audit_result(payload):
    started = time.perf_counter()
    config = GroupGameConfig(**{k: v for k, v in payload['config'].items()
                               if k in GroupGameConfig.__dataclass_fields__})
    scorer = _make_singleton_scorer(config)
    points = np.asarray(scorer.points)
    transitions, terminal_cache = {}, {}
    failures = []

    def terminal(word):
        if word not in terminal_cache:
            label = scorer.terminal_label(support_word_to_key(word))
            if not label.validation.valid:
                raise ValueError('invalid supporting facet')
            values = points @ label.validation.normal + label.validation.offset
            tight = sum(1 << int(i) for i in np.flatnonzero(np.abs(values) <= scorer.config.support_tol))
            terminal_cache[word] = tight, exact_class_id(str(label.label)) if label.is_exact else None
        return terminal_cache[word]

    def replay(actions, *, initial=0, resets=()):
        word, corrections = initial, 0
        for i, action in enumerate(actions):
            if i in resets:
                word, _ = terminal(word)
                corrections = 0
            kind = action['kind']
            effective = int(action['effective_word_hex'], 16)
            removed = int(action['remove_word_hex'], 16)
            added = int(action['add_word_hex'], 16)
            key = word, kind, effective, removed, added, action['source']
            if key not in transitions:
                if kind == 'add':
                    after = word | effective
                elif kind == 'remove':
                    after = word & ~effective
                elif kind == 'rewrite':
                    after = (word & ~removed) | added
                else:
                    raise ValueError('non-discovery action in witness')
                rank = scorer.affine_rank(support_word_to_key(after))
                if rank > 25 or (kind in ('remove', 'rewrite') and rank < config.min_corrector_rank):
                    raise ValueError('path rank outside allowed range')
                if action['source'] == 'fullspace_as_completion':
                    after, _ = terminal(after)
                transitions[key] = after
            word = transitions[key]
            corrections += int(kind in ('remove', 'rewrite'))
            if corrections > config.extended_max_corrections:
                raise ValueError('local correction budget exceeded')
        return word

    seen = set()
    for event in payload['discoveries']:
        try:
            origin = event.get('continuation_origin', {})
            expected = int(event['selected_support_word_hex'], 16)
            full = replay(event['path'], resets=origin.get('historical_resets', ()))
            initial = int(origin['support_word_hex'], 16) if origin.get('support_word_hex') else 0
            prefix = origin.get('historical_path_length', 0)
            if prefix:
                prefix_word = replay(event['path'][:prefix], resets=origin.get('historical_resets', ()))
                if terminal(prefix_word)[0] != initial:
                    raise ValueError('continuation prefix does not reach claimed source')
            local = replay(event.get('local_path', event['path']), initial=initial)
            if full != expected or local != expected:
                raise ValueError('full/local witness support mismatch')
            tight, class_id = terminal(expected)
            if tight != int(event['tight_support_word_hex'], 16) or class_id != event['class_id']:
                raise ValueError('terminal closure/class mismatch')
            if class_id in seen:
                raise ValueError('duplicate discovery class')
            seen.add(class_id)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({'class_id': event['class_id'], 'error': str(error)})
    proof_failures = []
    proofs = payload.get('transfer422', {}).get('as_executed_proofs', [])
    for index, proof in enumerate(proofs):
        try:
            adjacency_module.audit_adjacency_proof(points, proof)
        except (ValueError, KeyError, TypeError) as error:
            proof_failures.append({'index': index, 'error': str(error)})
    return {'passed': not failures and not proof_failures,
        'terminals_and_paths_checked': len(payload['discoveries']), 'unique_transitions': len(transitions),
        'failures': failures, 'all_executed_as_proofs_checked': len(proofs),
        'as_proof_failures': proof_failures, 'full_as_proof_audit': True,
        'elapsed_seconds': time.perf_counter()-started}
