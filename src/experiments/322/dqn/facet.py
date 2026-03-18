import numpy as np
from points import generate_points

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
    
    # 如果矩阵的秩为26（满秩），则26个点可以唯一确定一个超平面
    if np.linalg.matrix_rank(homogeneous_matrix) == 26:
        # 求解齐次线性方程组得到法向量
        # 我们需要求解 A * n = 0，其中A是26×27的齐次坐标矩阵
        # 使用SVD求解零空间
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        
        # 法向量是V的最后一行的前26个元素
        normal = Vh[-1, :26]
        # 截距 b
        b = Vh[-1, 26]
        return True, normal / np.linalg.norm(normal), b  # 归一化
    
    return False, None, None

def find_facet(vertices):
    """
    找到26个顶点构成的facet的顶点集
    输入:
        vertices: 形状为(64, 26)的numpy数组，每行是一个顶点
    输出:
        facet_vertices: 形状为(26, 26)的numpy数组，每行是一个facet的顶点
    """
    v = vertices[-1]  # 选择最后一个顶点作为参考
    facet_vertices = [v]
    mark = np.ones(len(vertices), dtype=bool)  # 标记哪些顶点已经被选中
    mark[-1] = False  # 已经选中最后一个顶点
    # 计算与facet_vertices 顶点集中所有顶点的欧式距离的最小值，并选择距离为最近的顶点，加入顶点集，重复直到找到26个顶点
    while len(facet_vertices) < 26:
        average_distances = None
        t = 0
        for facet_vertex in facet_vertices:
            distances = np.linalg.norm(vertices[mark] - facet_vertex, axis=1)
            t += 1
            if average_distances is None:
                average_distances = distances
            else:
                average_distances += distances
        average_distances /= t  # 计算平均距离
        nearest_index = np.argmin(average_distances)
        nearest_vertex = vertices[mark][nearest_index]
        facet_vertices.append(nearest_vertex)
        print(f"Selected vertex {len(facet_vertices)}: {nearest_vertex}, distance: {average_distances[nearest_index]}")
        mark[np.where(mark)[0][nearest_index]] = False  # 标记该顶点已被选中
        if len(facet_vertices) >= 26:
            break

    return facet_vertices


if __name__ == "__main__":
    vertices = generate_points()

    # 找到facet的顶点集
    facet_vertices = find_facet(vertices)
    # 检查这些顶点是否构成一个超平面，并计算法向量
    is_hyperplane, normal_vector, b = check_points_form_hyperplane(facet_vertices)
    if is_hyperplane:
        print("The 26 vertices form a hyperplane.")
        print("Normal vector:", normal_vector)
        print("Intercept b:", b)
    else:
        print("The 26 vertices do not form a hyperplane.")
