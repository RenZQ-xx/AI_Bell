import numpy as np
import itertools
from typing import List, Tuple, Optional, Dict

# --- 库导入检查 ---
try:
    from aibell import get_true_C, true_Q_solver

    HAS_AIBELL = True
except ImportError:
    HAS_AIBELL = False
    print("Warning: 'aibell' library not found. Classification by C and Q will be disabled.")


def check_hyperplane_and_normal(points: np.ndarray) -> List[Tuple[List[int], np.ndarray]]:
    """
    检查点集中所有8点组合是否构成超平面，并计算法向量
    """
    results = []
    n_points = points.shape[0]

    # 遍历所有8个点的组合
    for indices in itertools.combinations(range(n_points), 8):
        selected_points = points[list(indices)]
        is_hyperplane, normal = check_points_form_hyperplane(selected_points)

        if is_hyperplane:
            results.append((list(indices), normal))

    return results


def check_points_form_hyperplane(points: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """检查8个8维点是否构成一个超平面，并计算法向量（L2归一化）"""
    homogeneous_matrix = np.column_stack((points, np.ones(8)))

    if np.linalg.matrix_rank(homogeneous_matrix) == 8:
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        normal = Vh[-1, :8]
        return True, normal / np.linalg.norm(normal)

    # 备用方法
    base_point = points[0]
    relative_vectors = points[1:] - base_point

    if np.linalg.matrix_rank(relative_vectors) == 7:
        A = relative_vectors.T
        U, S, Vh = np.linalg.svd(A)
        normal = Vh[-1, :]
        if np.linalg.norm(normal) > 1e-10:
            return True, normal / np.linalg.norm(normal)

    return False, None


def classify_results_detailed(results: List[Tuple[List[int], np.ndarray]], all_points: np.ndarray) -> Dict:
    """
    【优化版】支持 Batch 计算，大幅提高速度
    """
    categories = {}
    n_samples = len(results)

    if n_samples == 0:
        return categories

    print(f"正在准备 Batch 数据 (共 {n_samples} 个超平面)...")

    # 1. 提取所有法向量构建矩阵 (Batch Size, Dimension)
    # results[i][1] 是 normal 向量
    normals_batch = np.array([item[1] for item in results])

    # 2. 向量化计算非零分量个数
    # axis=1 表示对每一行计算
    non_zero_counts = np.sum(np.abs(normals_batch) > 1e-9, axis=1)

    # 3. Batch 计算 C 和 Q
    # 初始化默认值
    c_values = np.full(n_samples, -1.0)
    q_values = np.full(n_samples, -1.0)

    if HAS_AIBELL:
        print("正在进行 Batch 物理性质计算 (C & Q)...")
        try:
            # --- 计算 C 值 ---
            # 假设 get_true_C 支持 batch 输入 (N, Dim)，返回 (N,) 或 (N, 1)
            # 如果 get_true_C 不支持 batch，可以使用列表推导式（虽然慢点，但不影响 Q 的加速）
            # 这里先尝试 Batch 调用，你可以根据实际 aibell 库的情况调整
            try:
                raw_c = get_true_C(normals_batch, all_points)
            except:
                # Fallback: 如果 get_true_C 不支持 batch，回退到循环
                raw_c = np.array([get_true_C(n, all_points)[0] for n in normals_batch])

            # 确保维度展平为 (N,)
            c_values = np.array(raw_c).flatten()

            # --- 计算 Q 值 ---
            # 1. 在循环外初始化 Solver (这是提速的关键！)
            # 请确保 n_parties 和 level 与你的问题设置一致
            # 根据你之前的代码，貌似是 2-party, level 1 ?
            solver = true_Q_solver(n_parties=2, level=1)

            # 2. 调用 Batch 计算接口
            # 假设 compute_from_batch 返回 (N,)
            raw_q = solver.compute_from_batch(normals_batch, verbose=True)
            q_values = np.array(raw_q).flatten()

        except Exception as e:
            print(f"Batch 计算出错: {e}")
            c_values[:] = -999.0
            q_values[:] = -999.0

    # 4. 组装结果 (分类)
    print("正在归类...")
    for i in range(n_samples):
        indices = results[i][0]
        normal = normals_batch[i]  # 或者 results[i][1]

        nz_count = int(non_zero_counts[i])

        # 四舍五入
        c_rounded = round(float(c_values[i]), 4)
        q_rounded = round(float(q_values[i]), 4)

        # 构建 Key
        key = (nz_count, c_rounded, q_rounded)

        if key not in categories:
            categories[key] = []

        categories[key].append((indices, normal))

    return categories


def format_vector(vec: np.ndarray) -> str:
    """辅助函数：将向量格式化为易读的字符串"""
    return "[" + ", ".join([f"{x:.3f}" if abs(x) > 1e-9 else "0.000" for x in vec]) + "]"


def main():
    # --- 1. 数据准备 ---
    points = []
    # Alice +1
    points.extend([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, 1, -1, 1, -1],
                   [1, 1, -1, 1, -1, 1, -1, 1], [1, 1, -1, -1, -1, -1, -1, -1]])
    # Alice -1
    points.extend([[-1, -1, 1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, 1, -1, 1],
                   [-1, -1, -1, 1, 1, -1, 1, -1], [-1, -1, -1, -1, 1, 1, 1, 1]])
    # Alice Follow
    points.extend([[1, -1, 1, 1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, -1, 1],
                   [1, -1, -1, 1, -1, 1, 1, -1], [1, -1, -1, -1, -1, -1, 1, 1]])
    # Alice Flip
    points.extend([[-1, 1, 1, 1, -1, -1, 1, 1], [-1, 1, 1, -1, -1, 1, 1, -1],
                   [-1, 1, -1, 1, 1, -1, -1, 1], [-1, 1, -1, -1, 1, 1, -1, -1]])
    points = np.array(points)

    print(f"输入点集形状: {points.shape}")
    print("正在计算有效超平面（排除放射相关组合）...\n")

    # --- 2. 核心计算 ---
    results = check_hyperplane_and_normal(points)
    print(f"总计找到 {len(results)} 个有效的超平面组合。")

    # --- 3. 详细分类 (Batch 优化版) ---
    # 使用上一轮提供的优化版函数
    categories = classify_results_detailed(results, points)

    # --- 4. 展示结果 (修改部分) ---
    print("\n" + "=" * 80)
    print("细致分类统计结果 (Key: 非零分量数 -> 子类: C值, Q值)")
    print("=" * 80)

    # 排序：非零个数升序 -> Q值降序 -> C值降序
    sorted_keys = sorted(categories.keys(), key=lambda x: (x[0], -x[2], -x[1]))

    current_nz_count = -1

    for key in sorted_keys:
        nz_count, c_val, q_val = key
        items = categories[key]
        count = len(items)

        if nz_count != current_nz_count:
            print(f"\n" + "#" * 50)
            print(f"【 大类: {nz_count} 个分量不为0 】")
            print("#" * 50)
            current_nz_count = nz_count

        ratio_str = f"{(q_val / c_val):.4f}" if c_val > 1e-9 else "N/A"
        print(f"\n  >>> 子类 [C={c_val:.4f}, Q={q_val:.4f}, Q/C={ratio_str}]")
        print(f"      包含数量: {count}")

        # 取第一个作为示例
        ex_indices, ex_normal = items[0]

        # ---------------------------------------------------------
        # 【修改点】展示详细的点组合信息
        # ---------------------------------------------------------
        print(f"      [示例详情]")
        print(f"      1. 法向量 Normal: {format_vector(ex_normal)}")
        print(f"      2. 点集索引 Indices: {ex_indices}")

        # 提取这8个点的坐标，并转为int类型（因为原始点通常是+1/-1，int看起来更干净）
        ex_points_matrix = points[ex_indices].astype(int)

        print(f"      3. 点集坐标矩阵 (8x8):")
        # 使用 numpy 的 array2string 设置格式，使输出对齐更漂亮
        print(np.array2string(ex_points_matrix, prefix="         "))

        # 检查是否等权
        non_zeros = ex_normal[np.abs(ex_normal) > 1e-9]
        if len(non_zeros) > 0 and np.allclose(np.abs(non_zeros), np.abs(non_zeros[0])):
            print(f"      (备注: 等权向量)")

        print("-" * 40)  # 子类之间的分割线


if __name__ == "__main__":
    main()
