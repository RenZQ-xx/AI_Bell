import hashlib
import random

import numpy as np
import pynauty


# -----------------------------
# Step 1: 构建带颜色的 Bell 不等式图
# -----------------------------

# 2-2-2 场景下的 16 个物理节点到整数索引的严格映射
NODE_MAP = {
    # Parties
    "PA": 0,
    "PB": 1,
    # Dummy + Real settings
    "SAid": 2,
    "SA0": 3,
    "SA1": 4,
    "SBid": 5,
    "SB0": 6,
    "SB1": 7,
    # Alice outcomes
    "A0+": 8,
    "A0-": 9,
    "A1+": 10,
    "A1-": 11,
    # Bob outcomes
    "B0+": 12,
    "B0-": 13,
    "B1+": 14,
    "B1-": 15,
}


# 顶点颜色分区：限制 GI 只允许物理上合法的对称变换
COLOR_PARTITION = [
    [0, 1],  # Party
    [2, 5],  # Dummy setting
    [3, 4, 6, 7],  # Real setting
    [8, 9, 10, 11, 12, 13, 14, 15],  # Outcomes
]


def as_pynauty_partition(color_partition):
    """pynauty 颜色分区要求每个颜色类是集合。"""
    return [set(cls) for cls in color_partition]


def add_undirected_edge(graph, u, v):
    """给 pynauty.Graph 添加无向边（双向写入邻接表）。"""
    graph.connect_vertex(u, [v])
    graph.connect_vertex(v, [u])


def build_bell_graph():
    """按 Bell 场景物理结构与不等式项编码建图。"""
    g = pynauty.Graph(16)

    # 1.1 设置顶点颜色
    g.set_vertex_coloring(as_pynauty_partition(COLOR_PARTITION))

    # 1.2 层级骨架连线：Party -> Settings -> Outcomes
    add_undirected_edge(g, NODE_MAP["PA"], NODE_MAP["SAid"])
    add_undirected_edge(g, NODE_MAP["PA"], NODE_MAP["SA0"])
    add_undirected_edge(g, NODE_MAP["PA"], NODE_MAP["SA1"])

    add_undirected_edge(g, NODE_MAP["PB"], NODE_MAP["SBid"])
    add_undirected_edge(g, NODE_MAP["PB"], NODE_MAP["SB0"])
    add_undirected_edge(g, NODE_MAP["PB"], NODE_MAP["SB1"])

    add_undirected_edge(g, NODE_MAP["SA0"], NODE_MAP["A0+"])
    add_undirected_edge(g, NODE_MAP["SA0"], NODE_MAP["A0-"])
    add_undirected_edge(g, NODE_MAP["SA1"], NODE_MAP["A1+"])
    add_undirected_edge(g, NODE_MAP["SA1"], NODE_MAP["A1-"])

    add_undirected_edge(g, NODE_MAP["SB0"], NODE_MAP["B0+"])
    add_undirected_edge(g, NODE_MAP["SB0"], NODE_MAP["B0-"])
    add_undirected_edge(g, NODE_MAP["SB1"], NODE_MAP["B1+"])
    add_undirected_edge(g, NODE_MAP["SB1"], NODE_MAP["B1-"])

    # 1.3 不等式项编码：
    # +A0B0 -> 同号相连（+,+）和（-,-），表示正系数相关
    add_undirected_edge(g, NODE_MAP["A0+"], NODE_MAP["B0+"])
    add_undirected_edge(g, NODE_MAP["A0-"], NODE_MAP["B0-"])

    # -A0B1 -> 异号相连（+,-）和（-,+），表示负系数相关
    add_undirected_edge(g, NODE_MAP["A0+"], NODE_MAP["B1-"])
    add_undirected_edge(g, NODE_MAP["A0-"], NODE_MAP["B1+"])

    return g


def graph_to_adjacency_matrix(graph, n):
    """从 pynauty 的邻接字典构造 0/1 邻接矩阵。"""
    mat = np.zeros((n, n), dtype=np.uint8)
    for u, neighbors in graph.adjacency_dict.items():
        for v in neighbors:
            mat[u, v] = 1
            mat[v, u] = 1
    np.fill_diagonal(mat, 0)
    return mat


# -----------------------------
# Step 2: 自同构群与规范型
# -----------------------------

def canonicalize_adjacency(adj_matrix, canon_perm):
    """
    用 canon_label 给出的排列对邻接矩阵做行列同步重排。
    数学上等价于 P * A * P^T，其中 P 是置换矩阵。
    """
    perm = np.array(canon_perm, dtype=np.int64)
    return adj_matrix[np.ix_(perm, perm)]


def canonical_color_partition(canon_perm, color_partition):
    """
    将颜色分区也映射到规范序。
    做法：先构造 old->new 的逆映射，再把每个颜色类中的旧编号转成新编号并排序。
    """
    n = len(canon_perm)
    old_to_new = np.empty(n, dtype=np.int64)
    for new_idx, old_idx in enumerate(canon_perm):
        old_to_new[old_idx] = new_idx

    canon_parts = []
    for color_class in color_partition:
        mapped = sorted(int(old_to_new[v]) for v in color_class)
        canon_parts.append(mapped)
    return canon_parts


# -----------------------------
# Step 3: 规范哈希指纹
# -----------------------------

def fingerprint_from_canonical(canon_adj, canon_color_part):
    """
    把规范邻接矩阵 + 规范颜色分区序列化后做 SHA-256。
    加入颜色信息可进一步压低极小概率的哈希碰撞风险。
    """
    adj_flat_str = "".join(str(int(x)) for x in canon_adj.flatten())
    color_str = "|".join(",".join(str(v) for v in cls) for cls in canon_color_part)
    payload = adj_flat_str + "#" + color_str
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def analyze_graph_and_fingerprint(graph, title):
    """执行 Step 2 + Step 3 并打印关键结果，返回指纹与规范矩阵。"""
    n = 16
    adj = graph_to_adjacency_matrix(graph, n)

    aut_data = pynauty.autgrp(graph)
    # pynauty.autgrp 返回元组：(generators, grpsize1, grpsize2, orbits, numorbits)
    # 群大小 = grpsize1 * 10^grpsize2
    group_size = aut_data[1] * (10 ** aut_data[2])

    canon_perm = pynauty.canon_label(graph)
    canon_adj = canonicalize_adjacency(adj, canon_perm)
    # 使用“当前图自身”的颜色分区，避免打乱后错误引用原图分区。
    graph_colors = [sorted(list(cls)) for cls in graph.vertex_coloring]
    canon_colors = canonical_color_partition(canon_perm, graph_colors)
    fp = fingerprint_from_canonical(canon_adj, canon_colors)

    print(f"[{title}] 自同构群大小: {group_size}")
    print(f"[{title}] 规范排列 canon_label: {canon_perm}")
    print(f"[{title}] 指纹 SHA-256: {fp}")

    return fp, canon_adj


# -----------------------------
# Step 4: Shuffle 抗干扰实验
# -----------------------------

def permute_graph(original_graph, original_colors, perm):
    """
    根据随机置换创建新图 G_shuffled。
    物理意义：模拟把同一个 Bell 不等式在标签层面做复杂重命名/对称变形。
    """
    n = 16
    shuffled = pynauty.Graph(n)

    # 颜色分区重映射：旧顶点 old -> 新顶点 perm[old]
    shuffled_colors = []
    for color_class in original_colors:
        shuffled_class = sorted(perm[v] for v in color_class)
        shuffled_colors.append(shuffled_class)
    shuffled.set_vertex_coloring(as_pynauty_partition(shuffled_colors))

    # 边重映射：若旧图有 (u,v)，新图加入 (perm[u], perm[v])
    old_adj = graph_to_adjacency_matrix(original_graph, n)
    for u in range(n):
        for v in range(u + 1, n):
            if old_adj[u, v] == 1:
                add_undirected_edge(shuffled, perm[u], perm[v])

    return shuffled


def main():
    # 固定随机种子仅用于可复现实验；不影响理论正确性。
    random.seed(20260227)

    g = build_bell_graph()
    fp_original, canon_adj_original = analyze_graph_and_fingerprint(g, "Original")
    print("[Original] 字典序最小规范邻接矩阵:")
    print(canon_adj_original)

    # 生成 0~15 的随机置换
    shuffled_perm_list = random.sample(range(16), 16)

    # 用该置换构建“表面完全不同”的同构图
    g_shuffled = permute_graph(g, COLOR_PARTITION, shuffled_perm_list)
    fp_shuffled, canon_adj_shuffled = analyze_graph_and_fingerprint(g_shuffled, "Shuffled")
    print("[Shuffled] 字典序最小规范邻接矩阵:")
    print(canon_adj_shuffled)

    # 同构图在规范化后应给出完全一致的哈希指纹
    assert fp_original == fp_shuffled, "错误：同构图的规范哈希不一致！"
    assert np.array_equal(
        canon_adj_original, canon_adj_shuffled
    ), "错误：同构图的规范邻接矩阵不一致！"
    print("[SUCCESS] 同构图必产出相同规范哈希，等价类查重验证通过。")
    return {
        "fingerprint": fp_original,
        "canonical_adjacency_matrix": canon_adj_original,
    }


if __name__ == "__main__":
    main()
