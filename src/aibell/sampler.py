import numpy as np
import ncpol2sdpa as ncp

class BellSampler:
    """Bell 场景采样基类"""

    @staticmethod
    def normalize_vectors(vectors):
        """
        归一化向量，使得 L2 范数 = 1。
        这对神经网络训练至关重要，消除了数值尺度的影响。
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除以 0
        norms[norms == 0] = 1
        return vectors / norms

    @staticmethod
    def perturb_vector(base_vector, n_samples, noise_level=0.2):
        """
        在给定基向量附近添加高斯噪声进行采样。
        """
        num_facet, dim = base_vector.shape

        # 生成噪声
        noise = np.random.normal(0, 1, (n_samples, dim))
        # 归一化噪声方向，乘以强度
        noise = BellSampler.normalize_vectors(noise) * noise_level

        def add_noise_with_base_vectorized(noise, base_vector):
            """
            向量化版本，避免显式循环
            """
            m, n = noise.shape
            k, _ = base_vector.shape

            # 创建索引数组，指示每行应该使用base_vector的哪一行
            indices = np.arange(m) % k

            # 使用高级索引获取对应的base_vector行
            selected_base = base_vector[indices]

            # 相加
            return noise + selected_base

        # 叠加
        perturbed_data = add_noise_with_base_vectorized(noise, base_vector)
        return BellSampler.normalize_vectors(perturbed_data)


class Sampler222(BellSampler):
    """
    针对 2-party 2-input 2-output (8维) 的采样器
    向量定义: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    """

    def __init__(self):
        self.dim = 8
        # 定义 CHSH 向量: <A0B0> + <A0B1> + <A1B0> - <A1B1>
        # 前4个边缘项为0，后4个相关项为 1,1,1,-1
        self.chsh_vector = self._generate_chsh_symmetries()

    def _generate_chsh_symmetries(self):
        """
        生成 CHSH 不等式的 8 种对称形式。
        CHSH 的相关项部分通常是 3 个 +1 和 1 个 -1 (或者反过来)。
        """
        variants = []

        # 基础形式：三个系数为 +1，一个系数为 -1
        # 例如: <A0B0> + <A0B1> + <A1B0> - <A1B1>
        base_patterns = np.array([
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1]
        ])

        for p in base_patterns:
            # 1. 正向形式 (对应上界 2)
            # 构造完整 8 维向量: 前 4 个(边缘项)为 0, 后 4 个(相关项)为 p
            vec_pos = np.concatenate([np.zeros(4), np.array(p)])
            variants.append(vec_pos)

            # 2. 负向形式 (对应下界 -2，或者说是另一种翻转)
            # 物理上这对应系数互为相反数，但也是边界的一部分
            variants.append(-1 * vec_pos)

        return np.array(variants)

    def generate_data(self, n_samples, ratio_random=0.5, noise_level=0.3):
        """
        生成混合数据
        :param ratio_random: 纯随机数据的比例 (0.0 - 1.0)
        :param noise_level: 扰动幅度
        """
        n_random = int(n_samples * ratio_random)
        n_perturbed = n_samples - n_random

        # 1. 纯随机球面采样
        random_data = np.random.normal(0, 1, (n_random, self.dim))
        random_data = self.normalize_vectors(random_data)

        # 2. CHSH 附近的扰动采样
        if n_perturbed > 0:
            perturbed_data = self.perturb_vector(self.chsh_vector, n_perturbed, noise_level)
            # 合并
            dataset = np.vstack([random_data, perturbed_data])
        else:
            dataset = random_data

        # 打乱顺序
        np.random.shuffle(dataset)
        return dataset

'''
class Sampler322(BellSampler):
    """
    针对 3-party 2-input 2-output (26维) 的采样器
    向量定义顺序:
    [0-5]:   A0, A1, B0, B1, C0, C1 (Singles)
    [6-9]:   A0B0...A1B1 (AB Correlations)
    [10-13]: B0C0...B1C1 (BC Correlations)
    [14-17]: A0C0...A1C1 (AC Correlations)
    [18-25]: A0B0C0...A1B1C1 (ABC Correlations - Binary Order 000 to 111)
    """

    def __init__(self):
        self.dim = 26

        # 定义 Mermin 不等式向量 (检测多体纠缠)
        # M3 = A1B0C0 + A0B1C0 + A0B0C1 - A1B1C1
        self.mermin_vector = np.zeros(26)
        # 映射到对应的索引 (ABC correlations start at index 18)
        # 100(4) -> idx 22, 010(2) -> idx 20, 001(1) -> idx 19, 111(7) -> idx 25
        self.mermin_vector[[22, 20, 19]] = 1.0
        self.mermin_vector[25] = -1.0

        # 定义 Svetlichny 不等式向量 (检测真正的非定域性)
        # S3 = M3 + M3' (M3' 是输入全部翻转的形式)
        # 这里简化处理，只生成 Mermin 的变体作为第二个锚点
        self.svetlichny_vector = np.zeros(26)
        # A0B0C0 + A0B1C1 + A1B0C1 + A1B1C0 ... 这种形式
        indices_pos = [18, 21, 23, 24]  # 000, 011, 101, 110
        indices_neg = [25, 22, 20, 19]  # 111, 100, 010, 001
        self.svetlichny_vector[indices_pos] = 1.0
        self.svetlichny_vector[indices_neg] = -1.0

    def generate_data(self, n_samples, ratio_random=0.5, noise_level=0.3):
        """
        生成混合数据: 随机 + Mermin扰动 + Svetlichny扰动
        """
        n_random = int(n_samples * ratio_random)
        n_perturbed = n_samples - n_random

        # 1. 纯随机采样
        random_data = np.random.normal(0, 1, (n_random, self.dim))
        random_data = self.normalize_vectors(random_data)

        datasets = [random_data]

        # 2. 对称 CHSH 扰动采样
        if n_perturbed > 0:
            # 平均分配样本给 8 种变体
            n_per_variant = n_perturbed // len(self.chsh_variants)

            for variant in self.chsh_variants:
                # 对每个变体进行扰动
                p_data = self.perturb_vector(variant, n_per_variant, noise_level)
                datasets.append(p_data)

            # 如果整除有余数，补齐余数（用第一个变体）
            remainder = n_perturbed - (n_per_variant * len(self.chsh_variants))
            if remainder > 0:
                p_data = self.perturb_vector(self.chsh_variants[0], remainder, noise_level)
                datasets.append(p_data)

        final_data = np.vstack(datasets)
        np.random.shuffle(final_data)
        return final_data
    def generate_data(self, n_samples, ratio_random=0.4, noise_level=0.3):
        """
        生成混合数据: 随机 + Mermin扰动 + Svetlichny扰动
        """
        n_random = int(n_samples * ratio_random)
        # 剩余样本平分给两种已知不等式
        n_remaining = n_samples - n_random
        n_mermin = n_remaining // 2
        n_svet = n_remaining - n_mermin

        # 1. 随机采样
        data_random = np.random.normal(0, 1, (n_random, self.dim))
        data_random = self.normalize_vectors(data_random)

        datasets = [data_random]

        # 2. Mermin 扰动
        if n_mermin > 0:
            data_mermin = self.perturb_vector(self.mermin_vector, n_mermin, noise_level)
            datasets.append(data_mermin)

        # 3. Svetlichny 扰动
        if n_svet > 0:
            data_svet = self.perturb_vector(self.svetlichny_vector, n_svet, noise_level)
            datasets.append(data_svet)

        final_data = np.vstack(datasets)
        np.random.shuffle(final_data)
        return final_data


def embed_222_to_322(data_222):
    """
    将 8维 (2-2-2) 数据嵌入到 26维 (3-2-2) 空间
    遵循 '全域补零' 策略，假设 C 的系数为 0
    """
    n_samples = data_222.shape[0]
    data_322 = np.zeros((n_samples, 26))

    # Mapping
    # 1. Marginals A, B (222 idx 0-3) -> (322 idx 0-3)
    data_322[:, 0:4] = data_222[:, 0:4]

    # 2. Correlations AB (222 idx 4-7) -> (322 idx 6-9)
    # 注意: 322中 idx 4,5 是 C0, C1，被跳过
    data_322[:, 6:10] = data_222[:, 4:8]

    # 其余位置 (C相关项) 保持为 0

    return data_322
'''
def get_quantum_bound_222(s_data, level=1, verbose=False):
    """
    计算给定 Bell 表达式向量的 Quantum Bound。

    参数:
    s_vector (list or np.array): 8维向量，顺序为:
        [<A0>, <A1>, <B0>, <B1>, <A0B0>, <A0B1>, <A1B0>, <A1B1>]
    level (int): NPA 层级，默认为 1 (对于 2-2-2 CHSH 情形通常足够)

    返回:
    float: 计算出的量子界限
    """
    n_samples, dim = s_data.shape
    quantum_bound = [0] * n_samples


    # 2. 定义算符及其约束（归一化，对易，投影约束）
    M = ncp.Probability([2, 2], [2, 2])
    substitutions = M.substitutions
    A = M.get_extra_monomials("A")
    B = M.get_extra_monomials("B")
    ops = ncp.flatten([M.get_all_operators()])

    A0 = M([0], [0], "A") - M([1], [0], "A")
    A1 = M([0], [1], "A") - M([1], [1], "A")
    B0 = M([0], [0], "B") - M([1], [0], "B")
    B1 = M([0], [1], "B") - M([1], [1], "B")
    A0B0 = M([0, 0], [0, 0]) - M([0, 1], [0, 0]) - M([1, 0], [0, 0])+ M([1, 1], [0, 0])
    A0B1 = M([0, 0], [0, 1]) - M([0, 1], [0, 1]) - M([1, 0], [0, 1]) + M([1, 1], [0, 1])
    A1B0 = M([0, 0], [1, 0]) - M([0, 1], [1, 0]) - M([1, 0], [1, 0]) + M([1, 1], [1, 0])
    A1B1 = M([0, 0], [1, 1]) - M([0, 1], [1, 1]) - M([1, 0], [1, 1]) + M([1, 1], [1, 1])

    # 3. sdp
    sdp = ncp.SdpRelaxation(ops, verbose=verbose, normalized=True)
    sdp.get_relaxation(level=1,
                       objective=(A[0] * B[0] + A[0] * B[1] + A[1] * B[0] - A[1] * B[1]),
                       substitutions=substitutions)
    # 4. 构建目标函数 (Bell Operator)
    for i in range(n_samples):
        s_vector = s_data[i]

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
        sdp.process_constraints(momentequalities=[],
                                momentinequalities=[])
        sdp.solve(solver='mosek')
        if sdp.status == 'optimal':
            quantum_bound[i] = [sdp.primal, sdp.dual ]
        else:

            print('Bad solve: ', i, sdp.status)
            break


    # 7. 返回结果 (SDP 的最优值即为 Quantum Bound)
    return quantum_bound
# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # 1. 生成 2-2-2 训练数据
    sampler2 = Sampler222()
    # 生成 50 个样本，30% 随机，70% CHSH 扰动
    data_2 = sampler2.generate_data(n_samples=50, ratio_random=0.3)

    print("=== 2-2-2 Data Sample (8 dim) ===")
    print("Shape:", data_2.shape)
    print(data_2[0])  # 打印第一个样本
    print(get_quantum_bound_222(np.array([[ 0. ,-0.,  0. ,  0. ,-0.5 ,-0.5, -0.5 , 0.5]]), level=2))
    bound = get_quantum_bound_222(data_2, level=1)
    print(bound)
'''
    # 2. 生成 3-2-2 训练数据
    sampler3 = Sampler322()
    data_3 = sampler3.generate_data(n_samples=5, ratio_random=0.4)

    print("\n=== 3-2-2 Data Sample (26 dim) ===")
    print("Shape:", data_3.shape)
    print("Example (Mermin-like?):")
    print(data_3[0])

    # 3. 演示嵌入 (Scalability)
    # 将上面的 2-2-2 数据转换成 3-2-2 格式
    embedded_data = embed_222_to_322(data_2)

    print("\n=== Embedded 2-2-2 into 3-2-2 (26 dim) ===")
    print("Shape:", embedded_data.shape)
    print("Original 2-2-2 vector:", data_2[0])
    print("Embedded 3-2-2 vector:", embedded_data[0])

    # 验证嵌入正确性: 检查 C (index 4,5) 和 C相关项 (index > 9) 是否为 0
    is_sparse_correct = np.all(embedded_data[:, 4:6] == 0) and np.all(embedded_data[:, 10:] == 0)
    print(f"嵌入稀疏性检查 (C相关项为0): {is_sparse_correct}")
'''