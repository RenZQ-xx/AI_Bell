import numpy as np
import itertools
from aibell.tool import check_points_form_hyperplane, Points_322
import torch

class SetEnv_322:
    def __init__(self, device=None):
        # 自动检测设备，或者使用传入的设备
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_points = Points_322()
        self.D = 26
        self.m = 64

        # 预计算动作表
        self.action_map_4_35 = {
            i + 4: params for i, params in enumerate(itertools.product([1, -1], repeat=5))
        }
        self.action_map_36_67 = {
            i + 36: params for i, params in enumerate(itertools.product([1, -1], repeat=5))
        }
        self.action_map_68_99 = {
            i + 68: params for i, params in enumerate(itertools.product([1, -1], repeat=5))
        }

        self.mult_action_1 = np.array([-1, -1, -1, -1, -1, -1] + [1] * 20)
        self.mult_action_2 = np.array([1, -1, 1, -1, 1, -1] + [1] * 20)
        self.mult_action_3 = np.array([-1, 1, -1, 1, -1, 1] + [1] * 20)
        self.swap_indices = [4, 5, 0, 1, 2, 3] + list(range(6, 26))

        self.reset()

    def reset(self):
        self.current_step = 0
        while True:
            indices = np.random.choice(self.m, self.D, replace=False)
            candidate_state = self.all_points[indices].copy()
            is_valid, normal, b = check_points_form_hyperplane(candidate_state)
            if is_valid:
                # 必须排序，保持输入的一致性
                self.state = candidate_state[np.lexsort(candidate_state.T[::-1])]
                # 计算初始距离
                self.prev_dist = self._calculate_dist(normal, b)
                # 保存当前的几何信息，供 step 使用（可选，或者每次 step 重新算）
                self.current_normal = normal
                self.current_b = b
                break

        return self._get_observation(self.state, self.current_normal, self.current_b, self.prev_dist)

    @staticmethod
    def FA(first, second, mode):
        # 向量化 FA
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        out1 = np.empty_like(first)
        out2 = np.empty_like(first)

        cond_1_1 = (first == 1.) & (second == 1.)
        cond_n1_n1 = (first == -1.) & (second == -1.)
        cond_n1_1 = (first == -1.) & (second == 1.)
        cond_1_n1 = (first == 1.) & (second == -1.)

        if mode == 1:
            out1[cond_1_1], out2[cond_1_1] = -1., -1.
            out1[cond_n1_n1], out2[cond_n1_n1] = -1., 1.
            out1[cond_n1_1], out2[cond_n1_1] = 1., -1.
            out1[cond_1_n1], out2[cond_1_n1] = 1., 1.
        else:
            out1[cond_1_1], out2[cond_1_1] = 1., -1.
            out1[cond_n1_n1], out2[cond_n1_n1] = 1., 1.
            out1[cond_n1_1], out2[cond_n1_1] = -1., -1.
            out1[cond_1_n1], out2[cond_1_n1] = -1., 1.
        return out1, out2

    def _get_observation(self, state_matrix, normal, b, dist):
        """
        组装增强型状态：
        1. Flattened State (676)
        2. Normal Vector (26)
        3. Bias (1)
        4. Distance (1)
        """
        # 1. 原始状态展平
        flat_state = state_matrix.flatten().astype(np.float32)

        # 2. 处理几何信息 (处理 None 的情况，防止初始化崩溃)
        if normal is None:
            normal = np.zeros(self.D, dtype=np.float32)
        if b is None:
            b = 0.0

        # 3. 归一化/预处理
        # normal 已经是归一化的。
        # b 和 dist 的数值范围可能较大，建议简单缩放一下，或者直接传入
        # 这里为了稳健，将其转为 numpy array

        geom_features = np.concatenate([
            normal,
            np.array([b], dtype=np.float32),
            np.array([dist], dtype=np.float32)
        ])

        # 拼接所有特征
        return np.concatenate([flat_state, geom_features])

    def _apply_action_to_state(self, state_in, action):
        """辅助函数：仅计算动作结果，不改变内部 self.state"""
        new_state = state_in.copy()

        if action == 1:
            new_state *= self.mult_action_1
        elif action == 2:
            new_state *= self.mult_action_2
        elif action == 3:
            new_state *= self.mult_action_3
        elif action in self.action_map_4_35:
            b0, b1, c0, c1, sign = self.action_map_4_35[action]
            mask = (new_state[:, 2] == b0) & (new_state[:, 3] == b1) & \
                   (new_state[:, 4] == c0) & (new_state[:, 5] == c1)
            if np.any(mask):
                new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], sign)
        elif action in self.action_map_36_67:
            a0, a1, c0, c1, sign = self.action_map_36_67[action]
            mask = (new_state[:, 0] == a0) & (new_state[:, 1] == a1) & \
                   (new_state[:, 4] == c0) & (new_state[:, 5] == c1)
            if np.any(mask):
                new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], sign)
        elif action in self.action_map_68_99:
            a0, a1, b0, b1, sign = self.action_map_68_99[action]
            mask = (new_state[:, 0] == a0) & (new_state[:, 1] == a1) & \
                   (new_state[:, 2] == b0) & (new_state[:, 3] == b1)
            if np.any(mask):
                new_state[mask, 4], new_state[mask, 5] = self.FA(new_state[mask, 4], new_state[mask, 5], sign)
        elif action == 100:
            new_state = new_state[:, self.swap_indices]

        # 必须重新计算派生列 (6-25)
        A0, A1 = new_state[:, 0], new_state[:, 1]
        B0, B1 = new_state[:, 2], new_state[:, 3]
        C0, C1 = new_state[:, 4], new_state[:, 5]

        new_state[:, 6] = A0 * B0
        new_state[:, 7] = A0 * B1
        new_state[:, 8] = A1 * B0
        new_state[:, 9] = A1 * B1
        new_state[:, 10] = A0 * C0
        new_state[:, 11] = A0 * C1
        new_state[:, 12] = A1 * C0
        new_state[:, 13] = A1 * C1
        new_state[:, 14] = B0 * C0
        new_state[:, 15] = B0 * C1
        new_state[:, 16] = B1 * C0
        new_state[:, 17] = B1 * C1
        new_state[:, 18] = A0 * B0 * C0
        new_state[:, 19] = A0 * B1 * C0
        new_state[:, 20] = A1 * B0 * C0
        new_state[:, 21] = A1 * B1 * C0
        new_state[:, 22] = A0 * B0 * C1
        new_state[:, 23] = A0 * B1 * C1
        new_state[:, 24] = A1 * B0 * C1
        new_state[:, 25] = A1 * B1 * C1

        new_state = new_state[np.lexsort(new_state.T[::-1])]
        return new_state

    def get_valid_actions_mask(self):
        """
        【GPU 加速版】
        并行计算所有动作后的状态是否构成超平面。
        """
        valid_mask = np.zeros(101, dtype=bool)
        valid_mask[100] = True  # Swap 总是合法的

        # 1. 快速生成所有候选状态 (这一步计算量小，CPU Numpy 循环生成即可)
        candidate_states = []
        action_indices = []

        current_state_matrix = self.state  # (26, 26)

        for action in range(1, 100):
            # 生成临时状态
            temp_state = self._apply_action_to_state(current_state_matrix, action)
            candidate_states.append(temp_state)
            action_indices.append(action)

        if not candidate_states:
            return valid_mask

        # 2. 将数据转为 Tensor 并移动到 GPU
        # Shape: (99, 26, 26)
        batch_points = torch.tensor(np.array(candidate_states), dtype=torch.float32, device=self.device)

        # 3. 构造齐次坐标矩阵 (Batch Homogeneous Matrix)
        # 我们需要在每个 (26, 26) 矩阵右边拼一列 1，变成 (99, 26, 27)
        batch_size = batch_points.shape[0]
        ones = torch.ones((batch_size, 26, 1), device=self.device)
        homogeneous_matrix = torch.cat((batch_points, ones), dim=2)  # (99, 26, 27)

        # 4. GPU 上并行计算 SVD
        # 只需要计算奇异值 S，不需要 U 和 V
        # torch.linalg.svd 对于 batch 输入是并行的
        try:
            # compute_uv=False 只返回奇异值，速度更快
            S = torch.linalg.svdvals(homogeneous_matrix)
        except:
            # 极少数情况 SVD 可能不收敛，回退到 CPU
            S = torch.linalg.svdvals(homogeneous_matrix.cpu()).to(self.device)

        # 5. 检查秩 (Rank)
        # 满秩条件：在 26x27 矩阵中，应该有 26 个非零奇异值。
        # 我们看最小的那个奇异值 (index 25, 也就是第26个) 是否 > tolerance
        # S 的 shape 是 (99, 26) (因为 svdvals 返回 min(M, N) 个值)
        tol = 1e-4
        # S[:, -1] 是每个矩阵最小的奇异值
        is_full_rank = S[:, -1] > tol

        # 6. 更新 Mask
        valid_results = is_full_rank.cpu().numpy()
        for i, is_valid in enumerate(valid_results):
            if is_valid:
                valid_mask[action_indices[i]] = True

        return valid_mask



    # 辅助函数：计算距离
    def _calculate_dist(self, normal, b):
        values = np.dot(self.all_points, normal) + b
        tol = 1e-5
        pos_mask = values > tol
        neg_mask = values < -tol
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        # 如果已经找到边界（一边为0），距离设为0
        if min(n_pos, n_neg) == 0:
            return 0.0

        if n_pos < n_neg:
            return np.mean(values[pos_mask])
        else:
            return np.mean(-values[neg_mask])

    def step(self, action):
        old_state = self.state.copy()
        old_normal = self.current_normal.copy()
        old_b = self.current_b
        # 1. 执行动作
        self.state = self._apply_action_to_state(self.state, action)
        self.current_step += 1
        # 2. 计算新状态的几何属性
        is_hyperplane, normal, b = check_points_form_hyperplane(self.state)

        # 3. 处理崩塌（非超平面）的情况
        if not is_hyperplane:
            # 回滚状态
            self.state = old_state
            # 给予惩罚
            # 返回旧的状态视作当前观测，但在训练循环中通常这步不仅是observation，也是next_state
            # 此时 next_state = old_state 是合理的
            obs = self._get_observation(old_state, old_normal, old_b, self.prev_dist)
            return obs, -1.0, False

        # 4. 更新当前的几何缓存
        self.current_normal = normal
        self.current_b = b

        # 5.计算距离和奖励
        curr_dist = self._calculate_dist(normal, b)

        reward = -0.05  # 步数惩罚
        # 3. 计算距离和奖励
        # values 是所有点到超平面的距离
        values = np.dot(self.all_points, normal) + b
        tol = 1e-5
        pos_mask = values > tol
        neg_mask = values < -tol
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        # 核心指标：异常点数量（取两边较少的那一边作为"错误"的一边）
        n_outliers = min(n_pos, n_neg)
        self.prev_dist = curr_dist

        done = False
        if curr_dist <= 1e-6:  # 浮点数判定
            reward += 100.0
            done = True

        # 返回中增加 info
        info = {
                'n_outliers': n_outliers,
                'n_pos': n_pos,
                'n_neg': n_neg
            }

        obs = self._get_observation(self.state, normal, b, curr_dist)
        return obs, reward, done, info  # 注意这里返回了4个值

    def _compute_reward(self):
        # 逻辑保持不变
        is_hyperplane, normal, b = check_points_form_hyperplane(self.state)

        if not is_hyperplane:
            return -10.0, False, False

        values = np.dot(self.all_points, normal) + b
        tol = 1e-5
        pos_mask = values > tol
        neg_mask = values < -tol
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        total_outliers = min(n_pos, n_neg)

        if total_outliers == 0:
            return 100.0, True, True
        else:
            if n_pos < n_neg:
                avg_dist = np.mean(values[pos_mask])
            else:
                avg_dist = np.mean(-values[neg_mask])
            dist_score = np.exp(-avg_dist)
            # 因为我们现在屏蔽了非法动作，Agent 很容易活下来
            # 我们稍微提高一点奖励的区分度
            reward = 0.5 + dist_score * 2.0
            return reward, False, False
