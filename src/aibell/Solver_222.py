import numpy as np
import ncpol2sdpa as ncp
from tqdm import tqdm  # 进度条库，单核跑大数据的神器

def get_classical_bound_batch(s_data: list | np.ndarray, points):
    s_data = np.array(s_data)
    num_points, dim = points.shape
    if len(s_data.shape) == 1:
        s_data = np.reshape(s_data, (1, dim))
    result = [0] * s_data.shape[0]
    for i in range(s_data.shape[0]):
        result_tmp = 0
        for j in range(num_points):
            result_tmp = max(result_tmp, np.dot(points[j], s_data[i]))
        result[i] = result_tmp
    return result

def get_quantum_bound_batch(s_data, level=1, verbose=False):
    """
    单核批处理函数：
    1. 只初始化一次 SDP (速度最快)。
    2. 使用 try-except 捕获错误 (稳定性最高)。
    3. 带有进度条 (体验最好)。
    """
    if len(s_data.shape) == 1:
        s_data = np.reshape(s_data, (1, s_data.shape[0]))
    n_samples = s_data.shape[0]

    # 结果容器：初始化为 NaN，方便后续筛除失败的样本
    results = np.full((n_samples, 2), np.nan)

    # =========================================================
    # 1. 稳健的 SDP 初始化 (只做一次)
    # =========================================================
    try:
        M = ncp.Probability([2, 2], [2, 2])
        substitutions = M.substitutions
        A = M.get_extra_monomials("A")
        B = M.get_extra_monomials("B")
        ops = ncp.flatten([M.get_all_operators()])

        A0 = M([0], [0], "A") - M([1], [0], "A")
        A1 = M([0], [1], "A") - M([1], [1], "A")
        B0 = M([0], [0], "B") - M([1], [0], "B")
        B1 = M([0], [1], "B") - M([1], [1], "B")
        A0B0 = M([0, 0], [0, 0]) - M([0, 1], [0, 0]) - M([1, 0], [0, 0]) + M([1, 1], [0, 0])
        A0B1 = M([0, 0], [0, 1]) - M([0, 1], [0, 1]) - M([1, 0], [0, 1]) + M([1, 1], [0, 1])
        A1B0 = M([0, 0], [1, 0]) - M([0, 1], [1, 0]) - M([1, 0], [1, 0]) + M([1, 1], [1, 0])
        A1B1 = M([0, 0], [1, 1]) - M([0, 1], [1, 1]) - M([1, 0], [1, 1]) + M([1, 1], [1, 1])

        # 3. sdp
        sdp = ncp.SdpRelaxation(ops, verbose=False, normalized=True)
        sdp.get_relaxation(level=1,
                           objective=(A[0] * B[0] + A[0] * B[1] + A[1] * B[0] - A[1] * B[1]),
                           substitutions=substitutions)



    except Exception as e:
        print(f"致命错误：SDP 初始化失败。原因: {e}")
        return results

    # =========================================================
    # 2. 带有进度条的循环求解
    # =========================================================
    if verbose == True:
        # 使用 tqdm 包装 range，会显示进度条、预计剩余时间
        print(f"开始处理 {n_samples} 条数据 (Level={level})...")

    for i in tqdm(range(n_samples), desc="Solving SDP", unit="sample", disable=not verbose):
        s_vector = s_data[i]

        # 构建目标函数
        # 向量顺序: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
        objective = (
                s_vector[0] * A0 +
                s_vector[1] * A1 +
                s_vector[2] * B0 +
                s_vector[3] * B1 +
                s_vector[4] * A0B0 +
                s_vector[5] * A0B1 +
                s_vector[6] * A1B0 +
                s_vector[7] * A1B1
        )

        sdp.set_objective(objective)

        try:
            # 调用 Mosek
            sdp.solve(solver='mosek')

            # 只有当状态是 optimal 时才记录
            if sdp.status == 'optimal':
                results[i] = [sdp.primal, sdp.dual]
            # 如果 solver 返回 'inaccurate' 或其他，保持 NaN 即可

        except Exception:
            # 捕获任何可能的求解器崩溃，保证循环不中断
            continue

    return results


# =========================================================
# 使用示例
# =========================================================
if __name__ == "__main__":
    # 1. 制造一些假数据 (100条)
    # 前50条是随机的，后50条是标准 -CHSH (应该得到 -2.828)
    data = np.random.uniform(-1, 1, (100, 8))

    # 手动设置最后一条为标准 -CHSH 向量
    # -A0B0 - A0B1 - A1B0 + A1B1 -> [0,0,0,0, -1, -1, -1, 1]
    data[-1] = [0, 0, 0, 0, -1, -1, -1, 1]

    # 2. 运行
    bounds = get_quantum_bound_batch(data, level=1)

    # 3. 检查结果
    print("\n最后一条数据的计算结果 (期望 -2.828):")
    print(f"Primal: {bounds[-1][0]:.5f}")

    # 统计成功率
    success_count = np.sum(~np.isnan(bounds[:, 0]))
    print(f"成功求解: {success_count}/{len(data)}")
