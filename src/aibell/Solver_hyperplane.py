import numpy as np
import itertools
from typing import List, Tuple, Optional, Dict

def check_hyperplane_and_normal(points: np.ndarray) -> List[Tuple[List[int], np.ndarray]]:
    """
    检查点集中所有8点组合是否构成超平面，并计算法向量
    """
    results = []
    n_points = points.shape[0]

    # 遍历所有8个点的组合
    for indices in itertools.combinations(range(n_points), 8):
        selected_points = points[list(indices)]
        is_hyperplane, normal = check_points_form_hyperplane(selected_points)

        if is_hyperplane:
            results.append((list(indices), normal))

    return results


def check_points_form_hyperplane(points: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """检查8个8维点是否构成一个超平面，并计算法向量（L2归一化）"""
    homogeneous_matrix = np.column_stack((points, np.ones(8)))

    if np.linalg.matrix_rank(homogeneous_matrix) == 8:
        U, S, Vh = np.linalg.svd(homogeneous_matrix)
        normal = Vh[-1, :8]
        return True, normal / np.linalg.norm(normal)

    # 备用方法
    base_point = points[0]
    relative_vectors = points[1:] - base_point

    if np.linalg.matrix_rank(relative_vectors) == 7:
        A = relative_vectors.T
        U, S, Vh = np.linalg.svd(A)
        normal = Vh[-1, :]
        if np.linalg.norm(normal) > 1e-10:
            return True, normal / np.linalg.norm(normal)

    return False, None


def format_vector(vec: np.ndarray) -> str:
    """辅助函数：将向量格式化为易读的字符串"""
    return "[" + ", ".join([f"{x:.3f}" if abs(x) > 1e-9 else "0.000" for x in vec]) + "]"


def Points_222():
    # --- 1. 数据准备 ---
    points = []
    # Alice +1
    points.extend([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, 1, -1, 1, -1],
                   [1, 1, -1, 1, -1, 1, -1, 1], [1, 1, -1, -1, -1, -1, -1, -1]])
    # Alice -1
    points.extend([[-1, -1, 1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, 1, -1, 1],
                   [-1, -1, -1, 1, 1, -1, 1, -1], [-1, -1, -1, -1, 1, 1, 1, 1]])
    # Alice Follow
    points.extend([[1, -1, 1, 1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, -1, 1],
                   [1, -1, -1, 1, -1, 1, 1, -1], [1, -1, -1, -1, -1, -1, 1, 1]])
    # Alice Flip
    points.extend([[-1, 1, 1, 1, -1, -1, 1, 1], [-1, 1, 1, -1, -1, 1, 1, -1],
                   [-1, 1, -1, 1, 1, -1, -1, 1], [-1, 1, -1, -1, 1, 1, -1, -1]])
    points = np.array(points)
    return points

def solve_222_hyperplane_from_index(index: list, points: np.ndarray) -> Tuple[bool, np.ndarray]:
    return check_points_form_hyperplane(points[index])

if __name__ == "__main__":
    points = Points_222()
    index = [15, 14, 13, 11, 9, 7, 1, 0]
    print(solve_222_hyperplane_from_index(index, points))

