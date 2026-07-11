import numpy as np

def check_points_form_hyperplane(points: np.ndarray):
    """
    检查26个26维点是否构成一个超平面，并计算法向量
    
    参数:
        points: 形状为(26, 26)的numpy数组，每行是一个点
        
    返回:
        (是否构成超平面, 法向量或None, 截距b或None)
    """
    # 方法: 使用齐次坐标法
    # 对于26维空间中的超平面，方程为 a1*x1 + a2*x2 + ... + a26*x26 + b = 0
    # 构造齐次坐标矩阵
    homogeneous_matrix = np.column_stack((points, np.ones(26)))
    rank = np.linalg.matrix_rank(homogeneous_matrix)
    # 如果矩阵的秩为26（满秩），则26个点可以唯一确定一个超平面
    if rank == 26:
        # 求解齐次线性方程组得到法向量
        # 我们需要求解 A * n = 0，其中A是26×27的齐次坐标矩阵
        # 使用SVD求解零空间
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        
        # 法向量是V的最后一行的前26个元素
        normal = Vh[-1, :26]
        # 截距 b
        b = Vh[-1, 26]
        return True, normal / np.linalg.norm(normal), b, rank  # 归一化
    
    return False, None, None, rank

