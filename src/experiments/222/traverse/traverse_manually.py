import numpy as np
import itertools
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from aibell import Points_222

TOLERANCE = 1e-8
MAX_EXAMPLES_PER_CATEGORY = 3

# --- 库导入检查 ---
try:
    from aibell import get_true_C, true_Q_solver

    HAS_AIBELL = True
except ImportError:
    HAS_AIBELL = False
    print("Warning: 'aibell' library not found. Classification by C and Q will be disabled.")


def canonicalize_hyperplane(normal: np.ndarray, offset: float) -> Tuple[np.ndarray, float]:
    """归一化并固定符号，避免同一超平面因整体乘以 -1 被重复计数。"""
    norm = np.linalg.norm(normal)
    if norm < TOLERANCE:
        raise ValueError("Zero normal vector is not a valid hyperplane.")

    normal = normal / norm
    offset = offset / norm

    for value in normal:
        if abs(value) > TOLERANCE:
            if value < 0:
                normal = -normal
                offset = -offset
            break

    return normal, offset


def check_hyperplane_and_normal(points: np.ndarray) -> List[Dict[str, object]]:
    """
    检查点集中所有8点组合是否构成超平面，并回收该超平面上的全部顶点。
    """
    unique_hyperplanes: Dict[Tuple[int, ...], Dict[str, object]] = {}
    n_points = points.shape[0]

    # 遍历所有8个点的组合
    for indices in itertools.combinations(range(n_points), 8):
        selected_points = points[list(indices)]
        is_hyperplane, normal, offset = check_points_form_hyperplane(selected_points)

        if is_hyperplane:
            all_indices = find_points_on_hyperplane(points, normal, offset)
            key = tuple(all_indices)

            if key not in unique_hyperplanes:
                unique_hyperplanes[key] = {
                    "all_indices": all_indices,
                    "seed_indices": list(indices),
                    "normal": normal,
                    "offset": offset,
                    "num_vertices": len(all_indices),
                }

    return list(unique_hyperplanes.values())


def check_points_form_hyperplane(points: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """检查8个8维点是否构成一个超平面，并计算标准化后的法向量与截距。"""
    homogeneous_matrix = np.column_stack((points, np.ones(8)))

    if np.linalg.matrix_rank(homogeneous_matrix) == 8:
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        normal = Vh[-1, :8]
        offset = Vh[-1, 8]
        normal, offset = canonicalize_hyperplane(normal, offset)
        return True, normal, offset

    # 备用方法
    base_point = points[0]
    relative_vectors = points[1:] - base_point

    if np.linalg.matrix_rank(relative_vectors) == 7:
        _, _, Vh = np.linalg.svd(relative_vectors)
        normal = Vh[-1, :]
        if np.linalg.norm(normal) > TOLERANCE:
            offset = -float(np.dot(normal, base_point))
            normal, offset = canonicalize_hyperplane(normal, offset)
            return True, normal, offset

    return False, None, None


def find_points_on_hyperplane(all_points: np.ndarray, normal: np.ndarray, offset: float) -> List[int]:
    """返回所有位于该超平面上的顶点索引。"""
    distances = all_points @ normal + offset
    on_plane_mask = np.abs(distances) <= TOLERANCE
    return np.flatnonzero(on_plane_mask).tolist()


def classify_results_detailed(results: List[Dict[str, object]], all_points: np.ndarray) -> Dict:
    """
    【优化版】支持 Batch 计算，大幅提高速度
    """
    categories = {}
    n_samples = len(results)

    if n_samples == 0:
        return categories

    print(f"正在准备 Batch 数据 (共 {n_samples} 个超平面)...")

    # 1. 提取所有法向量构建矩阵 (Batch Size, Dimension)
    normals_batch = np.array([item["normal"] for item in results], dtype=float)

    # 2. 向量化计算非零分量个数
    # axis=1 表示对每一行计算
    non_zero_counts = np.sum(np.abs(normals_batch) > 1e-9, axis=1)
    plane_sizes = np.array([item["num_vertices"] for item in results], dtype=int)

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
            solver = true_Q_solver(n_parties=2, level=1, verbose=False)


            # 2. 调用 Batch 计算接口
            # 假设 compute_from_batch 返回 (N,)
            raw_q = solver.compute_from_batch(normals_batch)
            q_values = np.array(raw_q).flatten()

        except Exception as e:
            print(f"Batch 计算出错: {e}")
            c_values[:] = -999.0
            q_values[:] = -999.0

    # 4. 组装结果 (分类)
    print("正在归类...")
    for i in range(n_samples):
        item = results[i]
        normal = normals_batch[i]

        nz_count = int(non_zero_counts[i])
        plane_size = int(plane_sizes[i])

        # 四舍五入
        c_rounded = round(float(c_values[i]), 4)
        q_rounded = round(float(q_values[i]), 4)

        # 构建 Key
        key = (plane_size, nz_count, c_rounded, q_rounded)

        if key not in categories:
            categories[key] = []

        categories[key].append(item)

    return categories


def format_vector(vec: np.ndarray) -> str:
    """辅助函数：将向量格式化为易读的字符串"""
    return "[" + ", ".join([f"{x:.3f}" if abs(x) > 1e-9 else "0.000" for x in vec]) + "]"


def main():
    # --- 1. 数据准备 ---
    
    points = Points_222()

    print(f"输入点集形状: {points.shape}")
    print("正在计算有效超平面，并检查每个超平面上包含的全部顶点...\n")

    # --- 2. 核心计算 ---
    results = check_hyperplane_and_normal(points)
    print(f"去重后总计找到 {len(results)} 个有效超平面。")

    # --- 3. 详细分类 (Batch 优化版) ---
    categories = classify_results_detailed(results, points)

    # --- 4. 展示结果 ---
    print("\n" + "=" * 80)
    print("细致分类统计结果 (Key: 超平面顶点数, 非零分量数, C值, Q值)")
    print("=" * 80)

    # 排序：超平面顶点数降序 -> 非零个数升序 -> Q值降序 -> C值降序
    sorted_keys = sorted(categories.keys(), key=lambda x: (-x[0], x[1], -x[3], -x[2]))

    current_plane_size = -1

    for key in sorted_keys:
        plane_size, nz_count, c_val, q_val = key
        items = categories[key]
        count = len(items)

        if plane_size != current_plane_size:
            print(f"\n" + "#" * 50)
            print(f"【 大类: 超平面上共有 {plane_size} 个顶点 】")
            print("#" * 50)
            current_plane_size = plane_size

        ratio_str = f"{(q_val / c_val):.4f}" if c_val > 1e-9 else "N/A"
        print(f"\n  >>> 子类 [非零分量={nz_count}, C={c_val:.4f}, Q={q_val:.4f}, Q/C={ratio_str}]")
        print(f"      包含数量: {count}")

        num_to_show = min(len(items), MAX_EXAMPLES_PER_CATEGORY)
        print(f"      [展示前 {num_to_show} 个示例详情]")

        for k in range(num_to_show):
            example = items[k]
            ex_normal = example["normal"]
            all_indices = example["all_indices"]
            seed_indices = example["seed_indices"]
            extra_indices = [idx for idx in all_indices if idx not in seed_indices]

            print(f"      > 示例 {k + 1}:")
            print(f"        Normal : {format_vector(ex_normal)}")
            print(f"        Seed 8 Indices : {seed_indices}")
            print(f"        All On-Plane   : {all_indices}")
            print(f"        Extra Indices  : {extra_indices if extra_indices else 'None'}")
            print(f"        Vertex Count   : {example['num_vertices']}")

            if k == 0:
                ex_points_matrix = points[all_indices].astype(int)
                mean_vector = np.mean(ex_points_matrix, axis=0)
                print(f"        Matrix (该超平面上的全部顶点，仅首例展示):")
                print(np.array2string(ex_points_matrix, prefix="         "))
                print(f"        Mean Vector : {format_vector(mean_vector)}")

                non_zeros = ex_normal[np.abs(ex_normal) > 1e-9]
                if len(non_zeros) > 0 and np.allclose(np.abs(non_zeros), np.abs(non_zeros[0])):
                    print(f"        (备注: 等权向量)")

        print("-" * 60)  # 类与类之间的分割线

if __name__ == "__main__":
    output_path = Path(__file__).with_name("traverse_manually_output.txt")
    with output_path.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(output_file), redirect_stderr(output_file):
            main()

    print(f"结果已写入: {output_path}", file=sys.stderr)
