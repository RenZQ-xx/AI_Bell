import importlib

import numpy as np
import pytest


m = importlib.import_module("mcts.422.analyze_unmatched_symmetry")
p = importlib.import_module("mcts.422.problem")


@pytest.fixture(scope="module")
def group():
    return m.RelabelingStabilizers()


def chsh():
    row = [2] + [0] * 80
    for term, value in [(((0, 0), (1, 0)), -1), (((0, 0), (1, 1)), -1),
                        (((0, 1), (1, 0)), -1), (((0, 1), (1, 1)), 1)]:
        row[1+p.CORRELATOR_BASIS.index(term)] = value
    return row


def conditioned_chsh():
    row = [0]*81
    for term, value in zip(((), *p.CORRELATOR_BASIS), chsh()):
        if not value:
            continue
        for extension in ((), ((2, 0),), ((3, 0),), ((2, 0), (3, 0))):
            combined = term + extension
            index = 0 if not combined else 1+p.CORRELATOR_BASIS.index(combined)
            row[index] += value
    return row


def test_group_action_matches_public_contract(group):
    for u, flip in [(0, 0), (29, 97), (105, 255), (383, 42)]:
        parties, swaps = group.unsigned[u]
        relabeling = p.Relabeling(party_permutation=parties,
                                 setting_permutations=tuple((1, 0) if b else (0, 1) for b in swaps),
                                 outcome_signs=tuple(-1 if flip & (1 << j) else 1 for j in range(8)))
        action = p.signed_coordinate_action(relabeling)
        assert [target for target, _ in action] == list(group.targets[u])
        assert [sign for _, sign in action] == list(group.signs[flip])
        for state in p.iter_deterministic_states():
            moved = p.apply_relabeling_to_state(relabeling, state)
            for j, image in enumerate(group.observable_permutation((u, flip))[:8]):
                assert moved[image % 8] == state[j] * (-1 if image >= 8 else 1)


def test_filtered_stabilizer_matches_full_group_brute_force(group):
    for row in [chsh(), [10]+[1]*8+[0]*72, [4]+list(range(1, 81))]:
        coefficients = np.asarray(row[1:], dtype=np.int64)
        brute = set()
        for unsigned, targets in enumerate(group.targets):
            match = np.flatnonzero(np.all(group.signs * coefficients == coefficients[targets], axis=1))
            brute.update((unsigned, int(flip)) for flip in match)
        assert set(group.elements(row)) == brute


def test_party_symmetric_linear_row(group):
    row = [10]+[1]*8+[0]*72
    details = group.describe(row, group.elements(row))
    assert details["stabilizer_order"] == 384
    assert details["pure_party_order"] == 24
    assert details["pure_input_order"] == 16
    assert details["pure_output_order"] == 1
    assert details["party_orbit_sizes"] == [4]


def test_asymmetric_row(group):
    row = [4]+list(range(1, 81))
    assert group.elements(row) == ((0, 0),)
    assert group.describe(row, ((0, 0),))["equivalence_orbit_size"] == 98304


def test_exact_facet_rank_and_invalid_cases():
    checked = m.validate_facet(conditioned_chsh())
    assert checked["affine_rank_mod_prime"] == 79
    assert checked["tight_vertex_count"] == 224
    with pytest.raises(ValueError):
        m.validate_facet(chsh())
    with pytest.raises(ValueError):
        m.validate_facet([-5]+chsh()[1:])
    with pytest.raises(ValueError):
        m.validate_facet([10]+[1]*8+[0]*72)
    with pytest.raises(ValueError):
        m.validate_facet([2*v for v in conditioned_chsh()])


def test_modular_rank():
    assert m.modular_rank([[1, 2, 3], [2, 4, 6], [0, 1, 1]]) == 2
    assert m.modular_rank(np.zeros((0, 80), dtype=int)) == 0


def test_matrix_preserves_all_256_evaluations():
    row = [7]+list(range(-40, 40))
    matrix = np.asarray(m.coefficient_matrix(row)["matrix"])
    for state, vertex in zip(p.iter_deterministic_states(), p.deterministic_vertices()):
        a, b, c, d = state[:2], state[2:4], state[4:6], state[6:8]
        left = np.asarray([1, *a, *b, *(x*y for x in a for y in b)])
        right = np.asarray([1, *c, *d, *(x*y for x in c for y in d)])
        assert left @ matrix @ right == row[0]+vertex.astype(np.int64) @ row[1:]


def test_integer_validation(group):
    with pytest.raises(ValueError):
        group.elements([0]*81)
    with pytest.raises(ValueError):
        group.elements([0.5]*81)
    with pytest.raises(ValueError):
        group.elements([0]*80)
    with pytest.raises(ValueError):
        m.integer_row([2**64]+[0]*80)


def test_representative_closure_and_inequality(group):
    row = conditioned_chsh()
    elements = group.elements(row)
    entry = {"class_id": "test", "canonical_h_row": row, **group.describe(row, elements)}
    result = m.representative(entry, group)
    assert result["generator_closure_verified"]
    assert result["stabilizer_order"] == 128
    assert result["facet_validation"]["affine_rank_mod_prime"] == 79
    assert len(result["bell_terms"]) == 19
    assert result["bell_upper_bound"] == 2
