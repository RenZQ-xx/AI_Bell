import itertools
import random

try:
    from pynauty import Graph, isomorphic
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pynauty。请使用项目虚拟环境运行：\n"
        "  .venv/bin/python src/experiment/422/test_graph_isomorphism_422.py"
    ) from exc


VAR_NAMES = ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]
N_VARS = len(VAR_NAMES)
STATES = list(itertools.product([1, -1], repeat=N_VARS))
N_VERTICES = len(STATES)
STATE_TO_ID = {state: i for i, state in enumerate(STATES)}
ID_TO_STATE = {i: state for i, state in enumerate(STATES)}
IDX = {name: i for i, name in enumerate(VAR_NAMES)}


def create_permutation(transform_func):
    perm = []
    for i in range(N_VERTICES):
        new_state = transform_func(ID_TO_STATE[i])
        perm.append(STATE_TO_ID[new_state])
    return perm


def op_abswap(s):
    l = list(s)
    l[IDX["A0"]], l[IDX["B0"]] = l[IDX["B0"]], l[IDX["A0"]]
    l[IDX["A1"]], l[IDX["B1"]] = l[IDX["B1"]], l[IDX["A1"]]
    return tuple(l)


def op_acswap(s):
    l = list(s)
    l[IDX["A0"]], l[IDX["C0"]] = l[IDX["C0"]], l[IDX["A0"]]
    l[IDX["A1"]], l[IDX["C1"]] = l[IDX["C1"]], l[IDX["A1"]]
    return tuple(l)


def op_adswap(s):
    l = list(s)
    l[IDX["A0"]], l[IDX["D0"]] = l[IDX["D0"]], l[IDX["A0"]]
    l[IDX["A1"]], l[IDX["D1"]] = l[IDX["D1"]], l[IDX["A1"]]
    return tuple(l)


def op_flipin_a(s):
    l = list(s)
    l[IDX["A0"]], l[IDX["A1"]] = l[IDX["A1"]], l[IDX["A0"]]
    return tuple(l)


def op_flipin_b(s):
    l = list(s)
    l[IDX["B0"]], l[IDX["B1"]] = l[IDX["B1"]], l[IDX["B0"]]
    return tuple(l)


def op_flipin_c(s):
    l = list(s)
    l[IDX["C0"]], l[IDX["C1"]] = l[IDX["C1"]], l[IDX["C0"]]
    return tuple(l)


def op_flipin_d(s):
    l = list(s)
    l[IDX["D0"]], l[IDX["D1"]] = l[IDX["D1"]], l[IDX["D0"]]
    return tuple(l)


def make_flipout(var_name):
    target = IDX[var_name]

    def op(s):
        l = list(s)
        l[target] = -l[target]
        return tuple(l)

    return op


def build_generators():
    gens = []
    gen_names = []

    gens.append(create_permutation(op_abswap))
    gen_names.append("ABswap")
    gens.append(create_permutation(op_acswap))
    gen_names.append("ACswap")
    gens.append(create_permutation(op_adswap))
    gen_names.append("ADswap")
    gens.append(create_permutation(op_flipin_a))
    gen_names.append("FlipIn_A")
    gens.append(create_permutation(op_flipin_b))
    gen_names.append("FlipIn_B")
    gens.append(create_permutation(op_flipin_c))
    gen_names.append("FlipIn_C")
    gens.append(create_permutation(op_flipin_d))
    gen_names.append("FlipIn_D")

    for var in VAR_NAMES:
        gens.append(create_permutation(make_flipout(var)))
        gen_names.append(f"FlipOut_{var}")

    return gen_names, gens


def edge_orbits_from_generators(generators):
    all_edges = {(i, j) for i in range(N_VERTICES) for j in range(i + 1, N_VERTICES)}
    processed = set()
    orbits = []

    for edge in sorted(all_edges):
        if edge in processed:
            continue

        orbit = set([edge])
        stack = [edge]
        processed.add(edge)
        cursor = 0
        while cursor < len(stack):
            u, v = stack[cursor]
            cursor += 1
            for gen in generators:
                u2 = gen[u]
                v2 = gen[v]
                e2 = (u2, v2) if u2 < v2 else (v2, u2)
                if e2 not in orbit:
                    orbit.add(e2)
                    processed.add(e2)
                    stack.append(e2)
        orbits.append(sorted(orbit))

    return orbits


def permute_bits(node_bits, perm):
    out = [0] * len(node_bits)
    for old, bit in enumerate(node_bits):
        out[perm[old]] = bit
    return out


def build_colored_graph(node_bits, edge_orbits):
    total_virtual = sum(len(orbit) for orbit in edge_orbits)
    total_nodes = N_VERTICES + total_virtual
    g = Graph(total_nodes)

    bit0 = {i for i, b in enumerate(node_bits) if b == 0}
    bit1 = {i for i, b in enumerate(node_bits) if b == 1}
    if not bit0 or not bit1:
        raise ValueError("节点 bit 颜色类不能为空，请使用同时包含 0/1 的标注。")

    partition = [bit0, bit1]
    next_virtual = N_VERTICES
    for orbit in edge_orbits:
        cls = set()
        for u, v in orbit:
            w = next_virtual
            next_virtual += 1
            g.connect_vertex(u, [w])
            g.connect_vertex(v, [w])
            cls.add(w)
        partition.append(cls)

    g.set_vertex_coloring(partition)
    return g


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def group_closure(generators):
    gens = [tuple(g) for g in generators]
    identity = tuple(range(N_VERTICES))
    seen = {identity}
    queue = [identity]
    head = 0

    while head < len(queue):
        cur = queue[head]
        head += 1
        for g in gens:
            nxt = compose(g, cur)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def op_illegal_example(s):
    l = list(s)
    l[IDX["A0"]], l[IDX["B1"]] = l[IDX["B1"]], l[IDX["A0"]]
    return tuple(l)


def find_outside_group_noniso_perm(base_bits, g_base, group_elems, edge_orbits):
    candidates = [tuple(create_permutation(op_illegal_example))]

    for i in range(0, 12):
        j = (i * 7 + 5) % N_VERTICES
        if i == j:
            continue
        p = list(range(N_VERTICES))
        p[i], p[j] = p[j], p[i]
        candidates.append(tuple(p))

    for perm in candidates:
        if perm in group_elems:
            continue
        bits2 = permute_bits(base_bits, perm)
        g2 = build_colored_graph(bits2, edge_orbits)
        if not isomorphic(g_base, g2):
            return perm

    rng = random.Random(422)
    for _ in range(200):
        p = list(range(N_VERTICES))
        rng.shuffle(p)
        perm = tuple(p)
        if perm in group_elems:
            continue
        bits2 = permute_bits(base_bits, perm)
        g2 = build_colored_graph(bits2, edge_orbits)
        if not isomorphic(g_base, g2):
            return perm

    raise RuntimeError("未找到“群外且不同构”的变换，请调整节点 bit 标注。")


def main():
    gen_names, generators = build_generators()
    edge_orbits = edge_orbits_from_generators(generators)
    group_elems = group_closure(generators)

    rng = random.Random(20260422)
    base_bits = [rng.randint(0, 1) for _ in range(N_VERTICES)]
    if len(set(base_bits)) == 1:
        base_bits[0] = 1 - base_bits[0]

    g_base = build_colored_graph(base_bits, edge_orbits)

    known_name = "ADswap"
    known_perm = generators[gen_names.index(known_name)]
    bits_known = permute_bits(base_bits, known_perm)
    g_known = build_colored_graph(bits_known, edge_orbits)
    iso_known = isomorphic(g_base, g_known)

    outside_perm = find_outside_group_noniso_perm(base_bits, g_base, group_elems, edge_orbits)
    bits_outside = permute_bits(base_bits, outside_perm)
    g_outside = build_colored_graph(bits_outside, edge_orbits)
    iso_outside = isomorphic(g_base, g_outside)

    print(f"顶点数: {N_VERTICES}, 生成元数: {len(generators)}, 边轨道数: {len(edge_orbits)}")
    print(f"已知操作 {known_name}: isomorphic = {iso_known}")
    print(f"群外操作(自动找到): isomorphic = {iso_outside}")

    assert iso_known, "失败：群内操作得到的两个图应同构。"
    assert not iso_outside, "失败：群外操作得到的两个图应不同构。"
    print("测试通过：群内同构，群外不同构。")


if __name__ == "__main__":
    main()
