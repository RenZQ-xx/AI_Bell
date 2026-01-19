import numpy as np
import ncpol2sdpa as ncp
from tqdm import tqdm
import time
import itertools


class BellInequalitySolver:
    def __init__(self, n_parties, level=1):
        self.n_parties = n_parties
        self.level = level
        self.sdp = None
        self.term_list = []  # 存储算符对象
        self.term_strings = []  # 存储算符的字符串表示，用于快速查找
        self.vars_list = []  # 存储原始变量 [[A0, A1], [B0, B1]...]

        self._initialize_sdp()

    def _initialize_sdp(self):
        print(f"正在初始化 {self.n_parties}-Party (Level {self.level}) SDP 结构...")

        # 1. 生成变量
        party_names = [chr(65 + i) for i in range(self.n_parties)]
        self.vars_list = []
        for name in party_names:
            self.vars_list.append(ncp.generate_variables(name, 2, hermitian=True, commutative=False))

        all_vars = sum(self.vars_list, [])

        # 2. 构建替换规则 (对易关系 + op^2=1)
        substitutions = {op ** 2: 1 for op in all_vars}
        for i in range(self.n_parties):
            for j in range(i + 1, self.n_parties):
                ops_party_i = self.vars_list[i]
                ops_party_j = self.vars_list[j]
                for op_i in ops_party_i:
                    for op_j in ops_party_j:
                        substitutions.update({op_i * op_j: op_j * op_i})

        # 3. 生成所有项 (1-body 到 N-body)
        self.term_list = []
        self.term_strings = []  # 用于映射

        for k in range(1, self.n_parties + 1):
            party_combinations = itertools.combinations(self.vars_list, k)
            for party_group in party_combinations:
                op_combinations = itertools.product(*party_group)
                for ops in op_combinations:
                    # 乘积
                    term = ops[0]
                    for i in range(1, len(ops)):
                        term = term * ops[i]
                    self.term_list.append(term)
                    # 记录字符串，例如 "A0*B0" 或 "A0 B0" (取决于库版本，这里我们用标准化的格式)
                    # ncpol2sdpa 的 str() 通常返回类似 "A0 B0" 或 "A[0] B[0]"
                    # 我们这里简单存一下 str(term)
                    self.term_strings.append(str(term))

        print(f"  - 项数统计: {len(self.term_list)} (2-Party应为8, 3-Party应为26)")

        # 4. 初始化 SDP
        dummy_obj = sum(self.term_list)
        self.sdp = ncp.SdpRelaxation(variables=all_vars, verbose=False, normalized=True)
        self.sdp.get_relaxation(level=self.level, objective=dummy_obj, substitutions=substitutions)
        print("  - 初始化完成。")

    def _find_index_by_names(self, target_names):
        """
        辅助函数：根据变量名列表查找 term_list 中的索引
        例如 target_names = ['A0', 'B1'] -> 对应 A0*B1 的索引
        """
        # 这种查找方式比较鲁棒，不依赖字符串格式
        # 我们遍历 term_list，看哪个 term 是由这些变量组成的

        # 构建一个目标项用于对比
        target_term = 1
        # 我们需要从 self.vars_list 里找到对应的变量对象
        # 假设输入格式是 'A0', 'B1'
        for name in target_names:
            party_idx = ord(name[0]) - 65  # 'A'->0, 'B'->1
            var_idx = int(name[1])  # '0'->0, '1'->1
            target_term = target_term * self.vars_list[party_idx][var_idx]

        # 在 term_list 中查找 (依赖 ncpol2sdpa 的相等性判断)
        # 注意：ncpol2sdpa 的对象比较可能需要 str 辅助
        target_str = str(target_term)
        try:
            return self.term_strings.index(target_str)
        except ValueError:
            print(f"警告: 未找到项 {target_names} ({target_str})")
            return None

    def verify_correctness(self):
        """
        验证特定的物理不等式 (CHSH / Mermin)
        """
        print(f"\n--- 开始验证 {self.n_parties}-Party 正确性 ---")

        coeffs = np.zeros(len(self.term_list))
        target_bound = 0
        inequality_name = ""

        if self.n_parties == 2:
            inequality_name = "CHSH"
            target_bound = 2 * np.sqrt(2)  # ~2.8284
            # CHSH = A0B0 + A0B1 + A1B0 - A1B1
            # 这里的 A0, B0 对应索引需要查找
            map_dict = {
                ('A0', 'B0'): 1,
                ('A0', 'B1'): 1,
                ('A1', 'B0'): 1,
                ('A1', 'B1'): -1
            }

        elif self.n_parties == 3:
            inequality_name = "Mermin"
            target_bound = 4.0
            # Mermin = A1B0C0 + A0B1C0 + A0B0C1 - A1B1C1 (这是一种常见形式)
            # 或者 A0B0C0 - A0B1C1 - A1B0C1 - A1B1C0 (宇称形式)
            # 我们使用 Mermin-Klyshko 的标准形式 A0B0C0 - A0B1C1 - A1B0C1 - A1B1C0 (Quantum=4)
            map_dict = {
                ('A0', 'B0', 'C0'): 1,
                ('A0', 'B1', 'C1'): -1,
                ('A1', 'B0', 'C1'): -1,
                ('A1', 'B1', 'C0'): -1
            }
        else:
            print("目前仅支持 2-Party 和 3-Party 的自动验证。")
            return

        # 填充系数
        print(f"正在构建 {inequality_name} 向量...")
        for vars_tuple, val in map_dict.items():
            idx = self._find_index_by_names(vars_tuple)
            if idx is not None:
                coeffs[idx] = val

        # 求解
        objective = 0
        for idx, term in enumerate(self.term_list):
            objective += coeffs[idx] * term

        self.sdp.set_objective(objective)
        self.sdp.solve(solver='mosek')

        if self.sdp.status == 'optimal':
            result = -self.sdp.primal
            error = abs(result - target_bound)
            print(f"理论值: {target_bound:.5f}")
            print(f"计算值: {result:.5f}")
            if error < 1e-4:
                print(f"✅ 验证成功! 误差: {error:.2e}")
            else:
                print(f"❌ 验证失败! 误差较大。")
        else:
            print("求解器未收敛。")

    def solve_batch(self, batch_size=10, s_vectors=None):
        # ... (保持之前的代码不变) ...
        n_terms = len(self.term_list)
        if s_vectors is None:
            raw_coeffs = np.random.randn(batch_size, n_terms)
        else:
            raw_coeffs = s_vectors
            if len(raw_coeffs.shape) == 1:
                raw_coeffs = np.reshape(raw_coeffs, (1, n_terms))

        norms = np.linalg.norm(raw_coeffs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        random_coeffs = raw_coeffs / norms

        times = []
        results = []
        pbar = tqdm(range(batch_size), desc=f"{self.n_parties}-Party SDP")

        for i in pbar:
            coeffs = random_coeffs[i]
            objective = 0
            for idx, term in enumerate(self.term_list):
                objective += coeffs[idx] * term

            start_t = time.time()
            try:
                self.sdp.set_objective(objective)
                self.sdp.solve(solver='mosek')
                if self.sdp.status == 'optimal':
                    val = -self.sdp.primal
                else:
                    val = np.nan
            except Exception:
                val = np.nan
            end_t = time.time()
            times.append(end_t - start_t)
            results.append(val)
            pbar.set_postfix({"AvgTime": f"{np.mean(times):.4f}s"})

        return np.mean(times), np.array(results)

    def compute_from_batch(self, coeffs_batch, verbose=True):
        """
        输入指定的系数 batch 计算最大违背值。

        Args:
            coeffs_batch (np.ndarray): 形状为 (batch_size, n_terms) 的数组。
            verbose (bool): 是否显示进度条。

        Returns:
            np.ndarray: 形状为 (batch_size,) 的最大违背值数组。
        """
        # 1. 输入格式校验
        coeffs_batch = np.array(coeffs_batch)
        if coeffs_batch.ndim == 1:
            # 如果输入是单个向量，扩展为 batch 形式
            coeffs_batch = coeffs_batch.reshape(1, -1)

        batch_size, n_cols = coeffs_batch.shape
        n_terms = len(self.term_list)

        if n_cols != n_terms:
            raise ValueError(
                f"维度不匹配: 输入向量长度为 {n_cols}, 但当前 {self.n_parties}-Party 设置需要 {n_terms} 项。")

        results = []

        # 2. 根据 verbose 决定是否显示进度条
        iterator = tqdm(range(batch_size), desc="Calculating Rewards") if verbose else range(batch_size)

        for i in iterator:
            coeffs = coeffs_batch[i]

            # ======================================================
            # 逻辑: Maximize Obj = - Min ( -Obj )
            # ======================================================
            objective = 0
            for idx, term in enumerate(self.term_list):
                # 输入取负 (-1 * coeffs)
                objective += (-1 * coeffs[idx]) * term

            try:
                self.sdp.set_objective(objective)
                self.sdp.solve(solver='mosek')

                if self.sdp.status == 'optimal':
                    # 结果取负
                    results.append(-self.sdp.primal)
                else:
                    # 如果求解器未收敛（极少情况），返回 NaN 或 0
                    results.append(np.nan)
            except Exception as e:
                # 捕获潜在的 Mosek 错误
                results.append(np.nan)

        return np.array(results)


if __name__ == "__main__":
    # 1. 初始化并验证 2-Party (CHSH)
    solver_2 = BellInequalitySolver(n_parties=2, level=1)
    solver_2.verify_correctness()  # <--- 验证步骤


    print("\n" + "=" * 40 + "\n")

    # 2. 初始化并验证 3-Party (Mermin)
    solver_3 = BellInequalitySolver(n_parties=3, level=2)
    solver_3.verify_correctness()  # <--- 验证步骤


    benchmark_results = {}

    # 1. Test 2-Party (CHSH Scenario) - Level 1
    # 通常 Level 1 (NPA 1) 对于 2-party CHSH 已经足够 tight (Tsirelson bound)
    solver_2 = BellInequalitySolver(n_parties=2, level=1)
    t2, _ = solver_2.solve_batch(batch_size=200)
    benchmark_results['2-Party (L1)'] = t2

    # 2. Test 3-Party (GHZ/Mermin Scenario) - Level 2
    # 3-party 通常需要 Level 2 才能看到非经典关联的紧界
    # 注意：Level 2 的矩阵会显著变大
    solver_3 = BellInequalitySolver(n_parties=3, level=2)
    t3, _ = solver_3.solve_batch(batch_size=200)
    benchmark_results['3-Party (L2)'] = t3

    # 3. Test 4-Party (GHZ/Mermin Scenario) - Level 2
    # 4-party 通常需要 Level 2 才能看到非经典关联的紧界
    # 注意：Level 2 的矩阵会显著变大
    solver_4 = BellInequalitySolver(n_parties=4, level=2)
    t4, _ = solver_4.solve_batch(batch_size=200)
    benchmark_results['4-Party (L2)'] = t4

    print("\n" + "=" * 40)
    print("Benchmark 结果汇总 (平均每步耗时)")
    print("=" * 40)
    for k, v in benchmark_results.items():
        print(f"{k}: {v:.5f} 秒/步 (FPS: {1 / v:.2f})")
