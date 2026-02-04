import numpy as np
def check_points_form_hyperplane(points: list | np.ndarray):
    """
    检查 D个 D维点 是否构成一个超平面，并计算法向量

    参数:
        points: 形状为(D, D)的numpy数组，每行是一个点

    返回:
        (是否构成超平面, 法向量或None, 截距b或None)
    """
    points = np.array(points)
    D = points.shape[1]
    # 方法: 使用齐次坐标法
    # 对于D维空间中的超平面，方程为 a1*x1 + a2*x2 + ... + aD*xD + b = 0
    # 构造齐次坐标矩阵
    homogeneous_matrix = np.column_stack((points, np.ones(D)))

    # 如果矩阵的秩为D（满秩），则D个点可以唯一确定一个超平面
    if np.linalg.matrix_rank(homogeneous_matrix) == D:
        # 求解齐次线性方程组得到法向量
        # 我们需要求解 A * n = 0，其中A是D×(D+1)的齐次坐标矩阵
        # 使用SVD求解零空间
        U, S, Vh = np.linalg.svd(homogeneous_matrix)

        # 法向量是V的最后一行的前D个元素
        normal = Vh[-1, :D]
        # 截距 b
        b = Vh[-1, D]
        return True, normal / np.linalg.norm(normal), b  # 归一化

    return False, None, None
def Points_322():
    D = 26
    m = 64
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
def Points_222():
    D = 8
    m = 16
    all_points = []

    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    point = [a0, a1, b0, b1,
                             a0 * b0, a0 * b1, a1 * b0, a1 * b1]
                    all_points.append(point)
    all_points = np.array(all_points, dtype=np.float32)

    return all_points

def Points_232():
    D = 15
    m = 64
    all_points = []

    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for a2 in [1, -1]:
                for b0 in [1, -1]:
                    for b1 in [1, -1]:
                        for b2 in [1, -1]:
                            point = [a0, a1, a2, b0, b1, b2,
                                    a0 * b0, a0 * b1, a0 * b2, a1 * b0, a1 * b1, a1 * b2, a2 * b0, a2 * b1, a2 * b2]
                            all_points.append(point)
    all_points = np.array(all_points, dtype=np.float32)

    return all_points



def Points_223():
    D = 24
    m = 81
    all_points = []
    def encode(x):
        if x == 0: y = [1, 1]
        if x == 1: y = [-1, 1]
        if x == 2: y = [1, -1]
        return y

    for a0 in [0, 1, 2]:
        for a1 in [0, 1, 2]:
            for b0 in [0, 1, 2]:
                for b1 in [0, 1, 2]:
                    a01, a02 = encode(a0)
                    a11, a12 = encode(a1)
                    b01, b02 = encode(b0)
                    b11, b12 = encode(b1)
                    point = [a01, a02, a11, a12, b01, b02, b11, b12,
                             a01 * b01, a01 * b02, a02 * b01, a02 * b02,
                             a01 * b11, a01 * b12, a02 * b11, a02 * b12,
                             a11 * b01, a11 * b02, a12 * b01, a12 * b02,
                             a11 * b11, a11 * b12, a12 * b11, a12 * b12, ]
                    all_points.append(point)
    all_points = np.array(all_points, dtype=np.float32)

    return all_points

if __name__ == '__main__':
    print(Points_223())