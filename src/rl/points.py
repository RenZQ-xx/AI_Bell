import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from aibell import get_true_C, get_true_Q

def check_points_form_hyperplane(points: np.ndarray):
    """
    检查8个8维点是否构成一个超平面，并计算法向量
    
    参数:
        points: 形状为(8, 8)的numpy数组，每行是一个点
        
    返回:
        (是否构成超平面, 法向量或None, 截距b或None)
    """
    # 方法: 使用齐次坐标法
    # 对于8维空间中的超平面，方程为 a1*x1 + a2*x2 + ... + a8*x8 + b = 0
    # 构造齐次坐标矩阵
    homogeneous_matrix = np.column_stack((points, np.ones(8)))
    
    # 如果矩阵的秩为8（满秩），则8个点可以唯一确定一个超平面
    if np.linalg.matrix_rank(homogeneous_matrix) == 8:
        # 求解齐次线性方程组得到法向量
        # 我们需要求解 A * n = 0，其中A是8×9的齐次坐标矩阵
        # 使用SVD求解零空间
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        
        # 法向量是V的最后一行的前8个元素
        normal = Vh[-1, :8]
        # 截距 b
        b = Vh[-1, 8]
        return True, normal / np.linalg.norm(normal), b  # 归一化
    
    return False, None, None

class PointSetEnv:
    def __init__(self):
        # 生成所有16个可能的点
        self.all_points = []
        for a0 in [-1, 1]:
            for a1 in [-1, 1]:
                for b0 in [-1, 1]:
                    for b1 in [-1, 1]:
                        point = [a0, a1, b0, b1, 
                                a0*b0, a1*b0, a0*b1, a1*b1]
                        self.all_points.append(point)
        self.all_points = np.array(self.all_points, dtype=np.float32)
        
        # 初始状态：随机选择8个点
        self.reset()

    def reset(self):
        # 1. 重置步数计数器（配合防止死循环）
        self.current_step = 0

        # 2. 【关键】完全随机初始化
        # 无论上一轮结束时状态如何，这一轮我们重新从16个点里抽8个
        # 这就是你说的“多条路径搜索”的起点
        indices = np.random.choice(16, 8, replace=False)

        self.state = self.all_points[indices].copy()
        self.state = self.state[np.lexsort(self.state.T[::-1])]  # 排序保持状态一致性

        # 3. 返回初始状态
        return self.state.flatten()

    @staticmethod
    def FA(first, second, mode):
        # 1. 确保输入是 numpy 数组
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)

        # 2. 初始化输出数组（形状与输入相同）
        out1 = np.empty_like(first)
        out2 = np.empty_like(first)

        # 3. 创建条件掩码（找出哪些位置满足特定条件）
        # 注意：比较浮点数时，最好使用 np.isclose，但如果只有 1.0 和 -1.0，直接比较也可以
        cond_1_1 = (first == 1.) & (second == 1.)
        cond_n1_n1 = (first == -1.) & (second == -1.)
        cond_n1_1 = (first == -1.) & (second == 1.)
        cond_1_n1 = (first == 1.) & (second == -1.)

        # 4. 根据 mode 填充输出数组
        if mode == 1:
            # 对应 (1., 1.) -> (-1., -1.)
            out1[cond_1_1] = -1.
            out2[cond_1_1] = -1.

            # 对应 (-1., -1.) -> (-1., 1.)
            out1[cond_n1_n1] = -1.
            out2[cond_n1_n1] = 1.

            # 对应 (-1., 1.) -> (1., -1.)
            out1[cond_n1_1] = 1.
            out2[cond_n1_1] = -1.

            # 对应 (1., -1.) -> (1., 1.)
            out1[cond_1_n1] = 1.
            out2[cond_1_n1] = 1.

        else:  # mode != 1 的情况
            # 对应 (1., 1.) -> (1., -1.)
            out1[cond_1_1] = 1.
            out2[cond_1_1] = -1.

            # 对应 (-1., -1.) -> (1., 1.)
            out1[cond_n1_n1] = 1.
            out2[cond_n1_n1] = 1.

            # 对应 (-1., 1.) -> (-1., -1.)
            out1[cond_n1_1] = -1.
            out2[cond_n1_1] = -1.

            # 对应 (1., -1.) -> (-1., 1.)
            out1[cond_1_n1] = -1.
            out2[cond_1_n1] = 1.

        return out1, out2

    def step(self, action):
        """执行动作并返回新状态、奖励、是否结束"""
        new_state = self.state.copy()
        
        # 根据动作类型执行不同的操作
        if action == 0:  # AB-I
            new_state = new_state
        elif action == 1:  # AB-NOT
            new_state *= np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        elif action == 2:  # AB-XOR
            new_state *= np.array([1, -1, 1, -1, 1, -1, -1, 1])
        elif action == 3:  # AB-NOR
            new_state *= np.array([-1, 1, -1, 1, 1, -1, -1, 1])

        elif action == 4:  # B-0-A-FA+
            # 找出所有满足 b0=b1=1 的点
            mask = (new_state[:, 2] == 1) & (new_state[:, 3] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], 1)

        elif action == 5:  # B-0-A-FA-
            # 找出所有满足 b0=b1=1 的点
            mask = (new_state[:, 2] == 1) & (new_state[:, 3] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], -1)

        elif action == 6:  # B-y-A-FA+
            # 找出所有满足 b0=1,b1=-1 的点
            mask = (new_state[:, 2] == 1) & (new_state[:, 3] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], 1)

        elif action == 7:  # B-y-A-FA-
            # 找出所有满足 b0=1,b1=-1 的点
            mask = (new_state[:, 2] == 1) & (new_state[:, 3] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], -1)

        elif action == 8:  # B-ny-A-FA+
            # 找出所有满足 b0=-1,b1=1 的点
            mask = (new_state[:, 2] == -1) & (new_state[:, 3] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], 1)

        elif action == 9:  # B-ny-A-FA-
            # 找出所有满足 b0=-1,b1=1 的点
            mask = (new_state[:, 2] == -1) & (new_state[:, 3] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], -1)

        elif action == 10:  # B-1-A-FA+
            # 找出所有满足 b0=-1,b1=-1 的点
            mask = (new_state[:, 2] == -1) & (new_state[:, 3] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], 1)

        elif action == 11:  # B-1-A-FA-
            # 找出所有满足 b0=-1,b1=-1 的点
            mask = (new_state[:, 2] == -1) & (new_state[:, 3] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 0], new_state[mask, 1] = self.FA(new_state[mask, 0], new_state[mask, 1], -1)

        elif action == 12:  # A-0-B-FA+
            # 找出所有满足 b0=b1=1 的点
            mask = (new_state[:, 0] == 1) & (new_state[:, 1] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], 1)

        elif action == 13:  # A-0-B-FA-
            # 找出所有满足 b0=b1=1 的点
            mask = (new_state[:, 0] == 1) & (new_state[:, 1] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], -1)

        elif action == 14:  # A-x-B-FA+
            # 找出所有满足 b0=1,b1=-1 的点
            mask = (new_state[:, 0] == 1) & (new_state[:, 1] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], 1)

        elif action == 15:  # A-x-B-FA-
            # 找出所有满足 b0=1,b1=-1 的点
            mask = (new_state[:, 0] == 1) & (new_state[:, 1] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], -1)

        elif action == 16:  # A-nx-B-FA+
            # 找出所有满足 b0=-1,b1=1 的点
            mask = (new_state[:, 0] == -1) & (new_state[:, 1] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], 1)

        elif action == 17:  # A-nx-B-FA-
            # 找出所有满足 b0=-1,b1=1 的点
            mask = (new_state[:, 0] == -1) & (new_state[:, 1] == 1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], -1)

        elif action == 18:  # A-1-B-FA+
            # 找出所有满足 b0=-1,b1=-1 的点
            mask = (new_state[:, 0] == -1) & (new_state[:, 1] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], 1)

        elif action == 19:  # A-1-B-FA-
            # 找出所有满足 b0=-1,b1=-1 的点
            mask = (new_state[:, 0] == -1) & (new_state[:, 1] == -1)
            # 对mask的点：将a0, a1做加法变换
            new_state[mask, 2], new_state[mask, 3] = self.FA(new_state[mask, 2], new_state[mask, 3], -1)

        elif action == 20:  # AB-swap
            # a0, a1与b0, b1交换
            new_state[:] = new_state[:, [2, 3, 0, 1, 4, 5, 6, 7]]


        
        # 重新计算后4位（因为a0,a1,b0,b1可能改变了）
        for i in range(len(new_state)):
            a0, a1, b0, b1 = new_state[i, :4]
            new_state[i, 4:] = [a0*b0, a1*b0, a0*b1, a1*b1]
        
        # 重新排序
        new_state = new_state[np.lexsort(new_state.T[::-1])]
        
        # 更新状态
        self.state = new_state
        self.current_step += 1
        # 计算奖励
        reward, done = self._compute_reward()
        
        return self.state.flatten(), reward, done

    # 修改 points.py 中的 helper 函数
    def get_matrix_rank(self):
        homogeneous_matrix = np.column_stack((self.state, np.ones(8)))
        return np.linalg.matrix_rank(homogeneous_matrix)



    def _compute_reward(self):
        """
        检查当前8个点是否构成边界超平面
        r1: 仿射无关奖励
        r2: 边界奖励
        r3: Q(s) - C(s)
        输出：奖励和是否结束
        """
        points = self.state
        rank = self.get_matrix_rank()
        is_hyperplane, normal, b = check_points_form_hyperplane(points)
        is_boundary = False
        positive = []
        negative = []
        # 1. 基础奖励：基于矩阵的秩 (Rank 4 -> -4, Rank 7 -> -1, Rank 8 -> +5)
        # 这能引导 Agent 优先把点散开，避免共面
        r1 = (rank - 8) * 1.0
        if is_hyperplane:
            # 判断是否为边界（即所有其他点均在同一侧）
            r1 += 5.0  # 仿射无关奖励

            for i, point in enumerate(self.all_points):
                if any(np.allclose(point, p) for p in points):
                    continue  # 跳过当前8个点
                value = np.dot(normal, point) + b
                if value > 1e-6:
                    positive.append(point)
                elif value < -1e-6:
                    negative.append(point)
            # compute r2
            if len(positive) == 0 or len(negative) == 0:
                is_boundary = True
                r2 = 100.0  # 边界奖励
            else:
                r2 = -min(len(positive), len(negative))/len(self.all_points) * 3 # 非边界惩罚
            # compute r3
            # C_value = get_true_C(normal, self.all_points)
            # Q_value = get_true_Q(normal)
            # r3 = (- Q_value[0][0] - C_value[0]) * 10
            r3 = 0.0  # 暂时不计算Q-C奖励
            r4 = -1.0
        else:
            r1 = -5.0  # 非仿射无关惩罚
            r2 = 0.0  # 非边界惩罚
            r3 = 0.0
            r4 = -1.0
        
        reward = r1 + r2 + r3 + r4
        return reward, is_boundary
    
    def _get_current_indices(self):
        """获取当前8个点在all_points中的索引"""
        indices = []
        for i, point in enumerate(self.all_points):
            for p in self.state:
                if np.allclose(point, p):
                    indices.append(i)
                    break
        return indices[:8]  # 确保返回8个索引