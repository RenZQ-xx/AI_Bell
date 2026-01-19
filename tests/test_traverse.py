import numpy as np
import itertools
from typing import List, Tuple, Optional
from aibell import get_true_C, get_true_Q

def check_hyperplane_and_normal(points: np.ndarray) -> List[Tuple[List[int], np.ndarray]]:
    """
    检查点集中所有8点组合是否构成超平面，并计算法向量
    
    参数:
        points: 形状为(16, 8)的numpy数组，每行是一个8维点
        
    返回:
        列表，每个元素是(点的索引列表, 法向量)的元组
    """
    results = []
    n_points = points.shape[0]
    
    # 遍历所有8个点的组合
    for indices in itertools.combinations(range(n_points), 8):
        # 获取这8个点
        selected_points = points[list(indices)]
        
        # 检查8个点是否线性独立（构成超平面）
        # 在8维空间中，8个点构成超平面的充要条件是：
        # 将其中一个点作为原点，其他7个点相对于它的向量线性无关
        is_hyperplane, normal = check_points_form_hyperplane(selected_points)
        
        if is_hyperplane:
            results.append((list(indices), normal))
    
    return results

def check_points_form_hyperplane(points: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    检查8个8维点是否构成一个超平面，并计算法向量
    
    参数:
        points: 形状为(8, 8)的numpy数组，每行是一个点
        
    返回:
        (是否构成超平面, 法向量或None)
    """
    # 方法1: 使用齐次坐标法
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
        return True, normal / np.linalg.norm(normal)  # 归一化
    
    # 方法2: 使用点差法（如果齐次坐标法失败）
    # 以第一个点为基准，计算其他7个点的相对向量
    base_point = points[0]
    relative_vectors = points[1:] - base_point
    
    # 检查这7个向量是否线性独立（秩为7）
    if np.linalg.matrix_rank(relative_vectors) == 7:
        # 法向量与这7个向量都正交
        # 构造矩阵 M = [relative_vectors; 随机向量]，然后求正交补
        # 更简单的方法：求解线性方程组 relative_vectors·n = 0
        A = relative_vectors.T  # 8×7矩阵
        # 求解 A^T n = 0，即 n 在 A 的零空间中
        U, S, Vh = np.linalg.svd(A)
        
        # 法向量是V的最后一列（对应最小奇异值）
        normal = Vh[-1, :]
        
        # 还需要确保法向量不为零
        if np.linalg.norm(normal) > 1e-10:
            return True, normal / np.linalg.norm(normal)
    
    return False, None

def verify_hyperplane_equation(points: np.ndarray, normal: np.ndarray) -> bool:
    """
    验证法向量是否正确：检查所有点是否满足超平面方程
    
    参数:
        points: 8个点
        normal: 法向量
        
    返回:
        是否所有点都在同一个超平面上
    """
    # 超平面方程: normal·(x - p0) = 0
    # 先找到超平面上的一个点（取第一个点）
    p0 = points[0]
    
    # 计算常数项 b = -normal·p0
    b = -np.dot(normal, p0)
    
    # 检查所有点是否满足方程 normal·x + b = 0
    for point in points:
        value = np.dot(normal, point) + b
        if abs(value) > 1e-10:  # 允许很小的数值误差
            return False
    
    return True

# check if inputs are boundary points

def main():
    points = []
    
    # 第一组：Alice 总是出 +1 (即 $A_0=1, A_1=1$)
    points.extend([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, -1, 1, -1, 1, -1],
                   [1, 1, -1, 1, -1, 1, -1, 1],
                   [1, 1, -1, -1, -1, -1, -1, -1],])
    # 第二组：Alice 总是出 -1 (即 $A_0=-1, A_1=-1$)
    points.extend([[-1, -1, 1, 1, -1, -1, -1, -1],
                   [-1, -1, 1, -1, -1, 1, -1, 1],
                   [-1, -1, -1, 1, 1, -1, 1, -1],
                   [-1, -1, -1, -1, 1, 1, 1, 1],])
    # 第三组：Alice 跟随输入 (即 $A_0=1, A_1=-1$)
    points.extend([[1, -1, 1, 1, 1, 1, -1, -1],
                   [1, -1, 1, -1, 1, -1, -1, 1],
                   [1, -1, -1, 1, -1, 1, 1, -1],
                   [1, -1, -1, -1, -1, -1, 1, 1],])
    # 第四组：Alice 反转输入 (即 $A_0=-1, A_1=1$)
    points.extend([[-1, 1, 1, 1, -1, -1, 1, 1],
                   [-1, 1, 1, -1, -1, 1, 1, -1],
                   [-1, 1, -1, 1, 1, -1, -1, 1],
                   [-1, 1, -1, -1, 1, 1, -1, -1],])
    points = np.array(points)
    
    print("点集形状:", points.shape)
    print("\n开始检查所有8点组合是否构成超平面...")
    
    # 检查所有组合
    results = check_hyperplane_and_normal(points)

    print(f"\n找到 {len(results)} 个构成超平面的8点组合")
    
    # 显示结果
    for i, (indices, normal) in enumerate(results[:5]):
        print(f"\n组合 {i+1}:")
        print(f"  点的索引: {indices}")
        print(f"  法向量: {normal}")
        
        # 验证法向量是否正确
        selected_points = points[indices]
        if verify_hyperplane_equation(selected_points, normal):
            print("  ✓ 验证通过：所有点都在该超平面上")
            C_value = get_true_C(normal, points)
            Q_value = get_true_Q(normal)
            print(f"    计算得到的C值: {C_value[0]}")
            print(f"    计算得到的Q值: {Q_value[0][0]}")
        else:
            print("  ✗ 验证失败：点不在同一个超平面上")

    start_time = np.datetime64('now')
    best_value = -float('inf')
    best_indices = []

    for indices, normal in results:
        C_value = get_true_C(normal, points)
        Q_value = get_true_Q(normal)
        value = - Q_value[0][0] - C_value[0]
        if value > best_value:
            best_value = value
            best_indices.clear()
            best_indices.append((indices, normal))
        elif value == best_value:
            best_indices.append((indices, normal))
        
    end_time = np.datetime64('now')
    time = (end_time - start_time).astype('timedelta64[s]').item()

    print(f"\n最佳组合 (|Q + C| 最大值 = {best_value}), 用时{time}秒")
    for indices, normal in best_indices:
        print(f"  点的索引: {indices}")
        print(f"  法向量: {normal}")
    

if __name__ == "__main__":
    main()