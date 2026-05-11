import numpy as np


def check_points_form_hyperplane(points: list | np.ndarray):
    """
    Check whether D points in D-dimensional space define a hyperplane.

    Args:
        points: Array with shape (D, D), where each row is a point.

    Returns:
        tuple[bool, np.ndarray | None, float | None]:
        Whether a hyperplane is formed, the normalized normal vector, and the
        offset term.
    """
    points = np.array(points)
    dim = points.shape[1]
    homogeneous_matrix = np.column_stack((points, np.ones(dim)))

    if np.linalg.matrix_rank(homogeneous_matrix) == dim:
        _, _, vh = np.linalg.svd(homogeneous_matrix)
        normal = vh[-1, :dim]
        offset = vh[-1, dim]
        return True, normal / np.linalg.norm(normal), offset

    return False, None, None
