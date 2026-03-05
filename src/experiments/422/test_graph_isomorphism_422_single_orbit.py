import itertools
import random

try:
    from pynauty import Graph, isomorphic
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pynauty。请使用项目虚拟环境运行：\n"
        "  .venv/bin/python src/experiments/422/test_graph_isomorphism_422_single_orbit.py"
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
    # 只保留第一条边 (0,1) 的轨道，忽略其余边轨道。
    seed = (0, 1)
    orbit = set([seed])
    stack = [seed]
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
                stack.append(e2)

    return [sorted(orbit)]


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


def permutation_to_cycles(perm):
    n = len(perm)
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i] or perm[i] == i:
            seen[i] = True
            continue
        cur = i
        cyc = []
        while not seen[cur]:
            seen[cur] = True
            cyc.append(cur)
            cur = perm[cur]
        if len(cyc) > 1:
            cycles.append(tuple(cyc))
    return cycles


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


def sample_outside_permutations(group_elems, sample_size, seed=422):
    rng = random.Random(seed)
    picked = set()

    # 先放入一个固定候选，增加可复现性。
    illegal = tuple(create_permutation(op_illegal_example))
    if illegal not in group_elems:
        picked.add(illegal)

    attempts = 0
    max_attempts = sample_size * 2000
    while len(picked) < sample_size and attempts < max_attempts:
        p = list(range(N_VERTICES))
        rng.shuffle(p)
        perm = tuple(p)
        attempts += 1
        if perm in group_elems:
            continue
        picked.add(perm)

    if len(picked) < sample_size:
        raise RuntimeError(
            f"仅采样到 {len(picked)} 个群外变换，少于目标 {sample_size}。"
        )
    return list(picked)


def main():
    gen_names, generators = build_generators()
    edge_orbits = edge_orbits_from_generators(generators)
    group_elems = group_closure(generators)

    rng = random.Random(20260422)
    base_bits = [rng.randint(0, 1) for _ in range(N_VERTICES)]
    if len(set(base_bits)) == 1:
        base_bits[0] = 1 - base_bits[0]

    g_base = build_colored_graph(base_bits, edge_orbits)

    iso_by_gen = {}
    for name, perm in zip(gen_names, generators):
        bits2 = permute_bits(base_bits, perm)
        g2 = build_colored_graph(bits2, edge_orbits)
        iso_by_gen[name] = isomorphic(g_base, g2)
    all_gens_iso = all(iso_by_gen.values())

    outside_perms = sample_outside_permutations(group_elems, sample_size=50, seed=422)
    outside_iso_flags = []
    outside_iso_perms = []
    for perm in outside_perms:
        bits2 = permute_bits(base_bits, perm)
        g2 = build_colored_graph(bits2, edge_orbits)
        iso = isomorphic(g_base, g2)
        outside_iso_flags.append(iso)
        if iso:
            outside_iso_perms.append(perm)
    all_outside_noniso = all(not x for x in outside_iso_flags)
    outside_iso_count = sum(1 for x in outside_iso_flags if x)

    print(f"顶点数: {N_VERTICES}, 生成元数: {len(generators)}")
    print(f"只使用 (0,1) 对应轨道着色，轨道大小: {len(edge_orbits[0])}")
    print(f"所有生成元是否都同构: {all_gens_iso}")
    if not all_gens_iso:
        bad = [k for k, v in iso_by_gen.items() if not v]
        print(f"不同构的生成元: {bad}")

    print(f"随机群外 50 个变换是否全都不同构: {all_outside_noniso}")
    print(f"群外 50 个中仍同构的数量: {outside_iso_count}")
    if outside_iso_perms:
        p = outside_iso_perms[0]
        print(f"示例群外同构置换(一行表示): {p}")
        print(f"示例群外同构置换(循环分解): {permutation_to_cycles(p)}")

    if all_gens_iso and all_outside_noniso:
        print("结论: 该单色图通过了“群内全同构 + 群外50个全不同构”检验。")
    elif all_gens_iso:
        print("结论: 群内全同构成立，但群外50个并非全部不同构。")
    else:
        print("结论: 该单色图未通过群内全同构检验。")


if __name__ == "__main__":
    main()
