from pynauty import Graph, isomorphic

from test_graph_isomorphism_222_single_orbit import (
    N_VERTICES,
    ID_TO_STATE,
    build_augmented_graph_structure,
    build_generators,
    edge_orbits_from_generators,
)


def bits_from_indices(index_ids):
    bits = [0] * N_VERTICES
    for idx in index_ids:
        if not 0 <= idx < N_VERTICES:
            raise ValueError(f"节点ID越界: {idx}, 合法范围 [0, {N_VERTICES - 1}]")
        bits[idx] = 1
    if len(set(bits)) == 1:
        raise ValueError("输入索引不能让全部节点都同色，请至少选择一个且非全部节点。")
    return bits


def graph_from_index_ids(index_ids, edge_orbits):
    node_bits = bits_from_indices(index_ids)
    structure = build_augmented_graph_structure(node_bits, edge_orbits)
    g = Graph(structure["total_nodes"])
    for u, v in structure["edges"]:
        g.connect_vertex(u, [v])
    g.set_vertex_coloring(structure["partition"])
    return g, structure


def are_graphs_isomorphic_from_index_ids(index_ids_a, index_ids_b):
    _, generators = build_generators()
    edge_orbits = edge_orbits_from_generators(generators)
    g_a, _ = graph_from_index_ids(index_ids_a, edge_orbits)
    g_b, _ = graph_from_index_ids(index_ids_b, edge_orbits)
    return isomorphic(g_a, g_b)


def print_node_id_to_state():
    _, generators = build_generators()
    edge_orbits = edge_orbits_from_generators(generators)
    # 构建结构仅用于拿 node_id_to_state；bit 需同时含 0/1。
    demo_bits = [0] * N_VERTICES
    demo_bits[0] = 1
    structure = build_augmented_graph_structure(demo_bits, edge_orbits)

    print("node_id_to_state:")
    for node_id in range(structure["total_nodes"]):
        state = structure["node_id_to_state"].get(node_id, None)
        if node_id < N_VERTICES:
            # 真实节点应等价于 ID_TO_STATE[node_id]
            assert state == ID_TO_STATE[node_id]
        print(f"  {node_id}: {state}")


def main():
    print_node_id_to_state()

    # 示例：两个输入均为“被索引ID置1，其他置0”
    a_ids = [0, 1, 6, 7, 9, 11, 12, 14]
    b_ids = [12,13,2,0,3,8,9,14]
    ans = are_graphs_isomorphic_from_index_ids(a_ids, b_ids)
    print(f"isomorphic({a_ids}, {b_ids}) = {ans}")


if __name__ == "__main__":
    main()
