import numpy as np
import itertools
from typing import List, Tuple, Optional, Dict
from tool import check_points_form_hyperplane, Points_222, Points_322




def format_vector(vec: np.ndarray) -> str:
    """辅助函数：将向量格式化为易读的字符串"""
    return "[" + ", ".join([f"{x:.3f}" if abs(x) > 1e-9 else "0.000" for x in vec]) + "]"




def solve_hyperplane_from_index(index: list, points: np.ndarray) -> Tuple[bool, np.ndarray]:
    return check_points_form_hyperplane(points[index])

if __name__ == "__main__":
    '''
    points = Points_222()
    index = [15, 14, 13, 11, 9, 7, 1, 0]
    print(solve_hyperplane_from_index(index, points))
    '''
    points = Points_322()
    index = [i for i in range(4,30)]
    print(solve_hyperplane_from_index(index, points))

