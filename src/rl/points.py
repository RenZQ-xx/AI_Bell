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
        indices = np.random.choice(16, 8, replace=False)
        self.state = self.all_points[indices].copy()
        self.state = self.state[np.lexsort(self.state.T[::-1])]  # 排序
        return self.state.flatten()
    
    def step(self, action):
        """执行动作并返回新状态、奖励、是否结束"""
        new_state = self.state.copy()
        
        # 根据动作类型执行不同的操作
        if action == 0:  # A-0-flip
            # 找出所有满足 a0=a1=1 的点
            mask1 = (new_state[:, 0] == 1) & (new_state[:, 1] == 1)
            # 找出所有满足 a0=a1=-1 的点
            mask2 = (new_state[:, 0] == -1) & (new_state[:, 1] == -1)
            
            # 对mask1的点：将a0, a1从1变为-1
            new_state[mask1, 0] = -1
            new_state[mask1, 1] = -1
            
            # 对mask2的点：将a0, a1从-1变为1
            new_state[mask2, 0] = 1
            new_state[mask2, 1] = 1
            
        elif action == 1:  # A-0-shift
            # 对所有满足 a0=1 的点，对a1位取反
            mask = new_state[:, 0] == 1
            new_state[mask, 1] *= -1
            
        elif action == 2:  # A-0-flip-shift
            # 对所有满足 a1=1 的点，对a0位取反
            mask = new_state[:, 1] == 1
            new_state[mask, 0] *= -1
            
        elif action == 3:  # A-0-flip-only
            # 随机选择一个点（或者可以选择第一个点，但为了探索，最好随机）
            idx = np.random.randint(0, len(new_state))
            # 对该点的a0和a1取反
            new_state[idx, 0] *= -1
            new_state[idx, 1] *= -1
            
        elif action == 4:  # B-0-flip
            # 找出所有满足 b0=b1=1 的点
            mask1 = (new_state[:, 2] == 1) & (new_state[:, 3] == 1)
            # 找出所有满足 b0=b1=-1 的点
            mask2 = (new_state[:, 2] == -1) & (new_state[:, 3] == -1)
            
            # 对mask1的点：将b0, b1从1变为-1
            new_state[mask1, 2] = -1
            new_state[mask1, 3] = -1
            
            # 对mask2的点：将b0, b1从-1变为1
            new_state[mask2, 2] = 1
            new_state[mask2, 3] = 1
            
        elif action == 5:  # B-0-shift
            # 对所有满足 b0=1 的点，对b1位取反
            mask = new_state[:, 2] == 1
            new_state[mask, 3] *= -1
            
        elif action == 6:  # B-0-flip-shift
            # 对所有满足 b1=1 的点，对b0位取反
            mask = new_state[:, 3] == 1
            new_state[mask, 2] *= -1
            
        elif action == 7:  # B-0-flip-only
            # 随机选择一个点
            idx = np.random.randint(0, len(new_state))
            # 对该点的b0和b1取反
            new_state[idx, 2] *= -1
            new_state[idx, 3] *= -1
        
        # 重新计算后4位（因为a0,a1,b0,b1可能改变了）
        for i in range(len(new_state)):
            a0, a1, b0, b1 = new_state[i, :4]
            new_state[i, 4:] = [a0*b0, a1*b0, a0*b1, a1*b1]
        
        # 重新排序
        new_state = new_state[np.lexsort(new_state.T[::-1])]
        
        # 更新状态
        self.state = new_state
        
        # 计算奖励
        reward, done = self._compute_reward()
        
        return self.state.flatten(), reward, done
    
    def _compute_reward(self):
        """
        检查当前8个点是否构成边界超平面
        r1: 仿射无关奖励
        r2: 边界奖励
        r3: Q(s) - C(s)
        输出：奖励和是否结束
        """
        points = self.state
        is_hyperplane, normal, b = check_points_form_hyperplane(points)
        is_boundary = False
        positive = []
        negative = []
        if is_hyperplane:
            # 判断是否为边界（即所有其他点均在同一侧）
            r1 = 5.0  # 仿射无关奖励
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
                r2 = 10.0  # 边界奖励
            else:
                r2 = -min(len(positive), len(negative))/len(self.all_points) * 3 # 非边界惩罚
            # compute r3
            # C_value = get_true_C(normal, self.all_points)
            # Q_value = get_true_Q(normal)
            # r3 = (- Q_value[0][0] - C_value[0]) * 10
            r3 = 0.0  # 暂时不计算Q-C奖励
        else:
            r1 = -5.0  # 非仿射无关惩罚
            r2 = 0.0  # 非边界惩罚
            r3 = 0.0
        
        reward = r1 + r2 + r3
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