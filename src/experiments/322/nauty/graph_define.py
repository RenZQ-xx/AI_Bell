import itertools

try:
    from pynauty import Graph, certificate
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pynauty。请使用项目虚拟环境运行：\n"
        "  .venv/bin/python src/experiments/322/nauty/graph_define.py"
    ) from exc

# ==========================================
# 1. 定义变量与节点 (Vertices)
# ==========================================

# 如果你需要 256 个节点，请在这里补充 ['D0', 'D1']
VAR_NAMES = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1'] 
N_VARS = len(VAR_NAMES)

# 生成所有可能的状态 (+1, -1)，例如 (1, 1, -1, 1, -1, 1)
# 对应节点数 N = 2^6 = 64
states = list(itertools.product([1, -1], repeat=N_VARS))
n_vertices = len(states)

# 建立映射: 状态元组 -> 节点ID (0 到 63)
state_to_id = {state: i for i, state in enumerate(states)}
id_to_state = {i: state for i, state in enumerate(states)}

print(f"变量数: {N_VARS}, 节点(V)总数: {n_vertices}")

# ==========================================
# 2. 定义群生成元 (Generators)
# ==========================================

# 辅助函数：将一个状态变换函数转化为全排列列表
def create_permutation(transform_func):
    perm = []
    for i in range(n_vertices):
        current_state = id_to_state[i]
        #以此状态为输入，计算变换后的新状态
        new_state = transform_func(current_state)
        # 找到新状态对应的 ID
        perm.append(state_to_id[new_state])
    return perm

# --- 定义具体的变换逻辑 ---

# 索引辅助：找到 A0, A1 等在列表中的位置
idx = {name: i for i, name in enumerate(VAR_NAMES)}

generators = []
gen_names = []

# (1) Swap 元: 交换参与方 (Block Swap)
# ABswap: (A0,A1) <-> (B0,B1)
def op_ABswap(s):
    l = list(s)
    # 交换 A0<->B0, A1<->B1
    l[idx['A0']], l[idx['B0']] = l[idx['B0']], l[idx['A0']]
    l[idx['A1']], l[idx['B1']] = l[idx['B1']], l[idx['A1']]
    return tuple(l)

# ACswap: (A0,A1) <-> (C0,C1)
def op_ACswap(s):
    l = list(s)
    l[idx['A0']], l[idx['C0']] = l[idx['C0']], l[idx['A0']]
    l[idx['A1']], l[idx['C1']] = l[idx['C1']], l[idx['A1']]
    return tuple(l)

generators.append(create_permutation(op_ABswap)); gen_names.append("ABswap")
generators.append(create_permutation(op_ACswap)); gen_names.append("ACswap")

# (2) FlipInput 元: 交换同一方的输入 (Index Swap)
# FlipInput_A: A0 <-> A1
def op_flipIn_A(s):
    l = list(s)
    l[idx['A0']], l[idx['A1']] = l[idx['A1']], l[idx['A0']]
    return tuple(l)

def op_flipIn_B(s):
    l = list(s)
    l[idx['B0']], l[idx['B1']] = l[idx['B1']], l[idx['B0']]
    return tuple(l)

def op_flipIn_C(s):
    l = list(s)
    l[idx['C0']], l[idx['C1']] = l[idx['C1']], l[idx['C0']]
    return tuple(l)

generators.append(create_permutation(op_flipIn_A)); gen_names.append("FlipIn_A")
generators.append(create_permutation(op_flipIn_B)); gen_names.append("FlipIn_B")
generators.append(create_permutation(op_flipIn_C)); gen_names.append("FlipIn_C")

# (3) FlipOutput 元: 结果反转 (Value Flip)
# 针对每个变量 X，将值 v 变为 -v
for var in VAR_NAMES:
    def op_flipOut(s, target_idx=idx[var]):
        l = list(s)
        l[target_idx] = -l[target_idx] # +1 -> -1, -1 -> +1
        return tuple(l)
    
    # 注意：Python闭包问题，需绑定变量，但在循环中直接调用create helper比较安全
    # 这里使用默认参数技巧来绑定 target_idx
    generators.append(create_permutation(op_flipOut))
    gen_names.append(f"FlipOut_{var}")

print(f"生成元总数: {len(generators)}")

# ==========================================
# 3. 计算边轨道 (Edge Orbits)
# ==========================================

print("正在计算边轨道... (这可能需要几秒钟)")

# 初始化所有可能的边 (只考虑 i < j 的无向边)
# 数量 = 64*63/2 = 2016 条边
all_edges = set()
for i in range(n_vertices):
    for j in range(i + 1, n_vertices):
        all_edges.add((i, j))

edge_orbits = {} # {orbit_id: [edge_list]}
processed_edges = set()
orbit_count = 0

# 简单的 BFS 轨道搜索算法
sorted_edges = sorted(list(all_edges)) # 排序保证确定性

for edge in sorted_edges:
    if edge in processed_edges:
        continue
    
    # 发现新轨道
    current_orbit = set()
    stack = [edge]
    current_orbit.add(edge)
    processed_edges.add(edge)
    
    idx_stack = 0
    while idx_stack < len(stack):
        u, v = stack[idx_stack]
        idx_stack += 1
        
        # 将所有生成元作用于当前边 (u, v)
        for gen in generators:
            # gen 是一个置换列表，gen[u] 给出 u 变换后的节点 ID
            u_prime = gen[u]
            v_prime = gen[v]
            
            # 保证无向边顺序 (min, max)
            if u_prime < v_prime:
                new_edge = (u_prime, v_prime)
            else:
                new_edge = (v_prime, u_prime)
            
            if new_edge not in current_orbit:
                current_orbit.add(new_edge)
                processed_edges.add(new_edge)
                stack.append(new_edge)
    
    edge_orbits[orbit_count] = list(current_orbit)
    # print(f"  轨道 {orbit_count}: 大小 {len(current_orbit)}")
    orbit_count += 1

print(f"计算完成！共发现 {len(edge_orbits)} 个边轨道。")

# ==========================================
# 4. 构建用于 pynauty 的图 (虚拟节点法)
# ==========================================

# 我们将所有非空的轨道都加入图中，用虚拟节点区分轨道颜色
# 这种构造保证了图的自同构群包含了你的物理对称性 G

# 计算新图的总节点数
total_edges_in_orbits = sum(len(edges) for edges in edge_orbits.values())
virtual_nodes_start = n_vertices
total_nodes = n_vertices + total_edges_in_orbits

g = Graph(total_nodes)

# 分区列表：partition[0]是原始点，后面每个 partition 对应一个轨道
# pynauty 要求每个颜色类是 set
partition = [set(range(n_vertices))]

current_virtual = virtual_nodes_start

# 按轨道ID顺序处理，确保确定性
for orb_id in sorted(edge_orbits.keys()):
    edges = edge_orbits[orb_id]
    orbit_virtual_nodes = []
    
    for u, v in edges:
        # u -> virtual -> v
        # 注意：pynauty 的边是无向的 (connect_vertex 实际上建立邻接)
        g.connect_vertex(u, [current_virtual])
        g.connect_vertex(v, [current_virtual]) # 这样 virtual 节点度数为2
        
        orbit_virtual_nodes.append(current_virtual)
        current_virtual += 1
        
    # 将此轨道的所有虚拟节点放入同一个颜色分区
    partition.append(set(orbit_virtual_nodes))

# 设置颜色
g.set_vertex_coloring(partition)

# 计算这一结构的证书
cert = certificate(g)
print(f"图构建完成。pynauty Certificate 生成成功 (长度 {len(cert)})。")
print("你可以使用这个对象 g 进行同构判定。")

# 验证演示：
# 如果我们手动交换 A0 和 A1 的标签（这属于群G的操作），生成的图应该同构
# 但如果我们修改一个不在 G 内的关系，图应该不同构。
