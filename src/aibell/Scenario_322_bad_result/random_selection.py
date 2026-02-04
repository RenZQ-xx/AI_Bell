import numpy as np
import random
import time


# ==========================================
# 1. 你提供的核心功能函数
# ==========================================

def Points_322():
    """生成 64 个 26 维的特定几何结构点"""
    all_points = []
    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    for c0 in [1, -1]:
                        for c1 in [1, -1]:
                            point = [a0, a1, b0, b1, c0, c1,
                                     a0 * b0, a0 * b1, a1 * b0, a1 * b1,
                                     a0 * c0, a0 * c1, a1 * c0, a1 * c1,
                                     b0 * c0, b0 * c1, b1 * c0, b1 * c1,
                                     a0 * b0 * c0, a0 * b1 * c0, a1 * b0 * c0, a1 * b1 * c0,
                                     a0 * b0 * c1, a0 * b1 * c1, a1 * b0 * c1, a1 * b1 * c1]
                            all_points.append(point)
    all_points = np.array(all_points, dtype=np.float32)
    return all_points


def check_points_form_hyperplane(points: list | np.ndarray):
    """
    检查 D个 D维点 是否构成一个超平面，并计算法向量
    返回: (是否构成超平面, 法向量或None, 截距b或None)
    """
    points = np.array(points)
    D = points.shape[1]

    # 构造齐次坐标矩阵: [Points | 1]
    homogeneous_matrix = np.column_stack((points, np.ones(D)))

    # 如果矩阵的秩为D（满秩），则D个点可以唯一确定一个超平面
    # 注意：homogeneous_matrix 是 D x (D+1) 的矩阵
    if np.linalg.matrix_rank(homogeneous_matrix) == D:
        # 使用SVD求解零空间 A * n = 0
        U, S, Vh = np.linalg.svd(homogeneous_matrix)

        # 法向量是V的最后一行的前D个元素
        normal = Vh[-1, :D]
        # 截距 b
        b = Vh[-1, D]

        # 归一化处理
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-10:
            return True, normal / norm_val, b / norm_val
        else:
            return True, normal, b

    return False, None, None


# ==========================================
# 2. 随机采样与统计分析模块
# ==========================================

def run_hyperplane_sampling(num_trials=50000):
    print(f"正在初始化数据...")
    all_points = Points_322()
    total_points_num, dim = all_points.shape

    print(f"数据加载完成: {total_points_num} 个顶点，维度 {dim}")
    print(f"开始进行 {num_trials} 次随机采样，寻找构成超平面的组合...")

    start_time = time.time()

    success_count = 0
    successful_samples = []

    # 索引列表，用于随机抽取
    indices_pool = list(range(total_points_num))

    for i in range(num_trials):
        # 1. 随机无放回抽取 26 个索引
        selected_indices = random.sample(indices_pool, dim)

        # 2. 提取对应的点矩阵
        current_matrix = all_points[selected_indices]

        # 3. 检查是否构成超平面
        is_hyperplane, normal, b = check_points_form_hyperplane(current_matrix)

        if is_hyperplane:
            success_count += 1
            # 记录结果 (为了节省内存，只存前 100 个样本的完整矩阵，后续只存索引)
            sample_data = {
                "trial_id": i,
                "indices": sorted(selected_indices),  # 排序以便查看
                "normal": normal,
                "intercept": b,
                "matrix": current_matrix if len(successful_samples) < 100 else None
            }
            successful_samples.append(sample_data)

        # 简单的进度打印
        if (i + 1) % (num_trials // 10) == 0:
            print(f"已完成 {i + 1} 次尝试...")

    end_time = time.time()
    duration = end_time - start_time

    ratio = success_count / num_trials

    print("\n" + "=" * 40)
    print("采样统计结果")
    print("=" * 40)
    print(f"总采样次数: $$ {num_trials} $$")
    print(f"耗时: {duration:.2f} 秒")
    print(f"成功构成超平面的组合数: $$ {success_count} $$")
    print(f"构成超平面的比例: $$ {ratio:.6f} $$ ({ratio * 100:.4f}%)")

    return successful_samples


# ==========================================
# 3. 执行脚本
# ==========================================

if __name__ == "__main__":
    # 执行采样，你可以修改 num_trials 的大小
    # 注意：由于组合数 C(64, 26) 极其巨大，这里计算的是概率估算
    results = run_hyperplane_sampling(num_trials=50000)

    # 如果找到了结果，打印其中一个示例
    if results:
        print("\n" + "-" * 40)
        print("展示第一个成功的样本详情：")
        sample = results[0]

        print(f"1. 选取的顶点索引 (Indices):")
        print(f"{sample['indices']}")

        print(f"\n2. 超平面方程:")
        # 显示法向量的前几位
        n_vec = sample['normal']
        b_val = sample['intercept']
        print(f"法向量 (前5维): [ {n_vec[0]:.4f}, {n_vec[1]:.4f}, {n_vec[2]:.4f}, {n_vec[3]:.4f}, {n_vec[4]:.4f} ... ]")
        print(f"截距 b: {b_val:.4f}")
        print(f"方程形式: $$ n \\cdot x + ({b_val:.4f}) = 0 $$")

        print(f"\n3. 对应的矩阵 (Shape: 26x26):")
        # 打印矩阵的前几行
        print(sample['matrix'][:3, :])
        print("... (省略剩余行) ...")
    else:
        print("在当前的采样次数内，未找到能够构成唯一超平面的组合。")
        print("提示：如果该几何结构的对称性极高，可能导致大部分随机组合线性相关(秩<26)或无法唯一确定超平面。")
