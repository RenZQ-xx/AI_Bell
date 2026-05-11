import numpy as np


def Points_322():
    all_points = []

    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    for c0 in [1, -1]:
                        for c1 in [1, -1]:
                            point = [
                                a0,
                                a1,
                                b0,
                                b1,
                                c0,
                                c1,
                                a0 * b0,
                                a0 * b1,
                                a1 * b0,
                                a1 * b1,
                                a0 * c0,
                                a0 * c1,
                                a1 * c0,
                                a1 * c1,
                                b0 * c0,
                                b0 * c1,
                                b1 * c0,
                                b1 * c1,
                                a0 * b0 * c0,
                                a0 * b1 * c0,
                                a1 * b0 * c0,
                                a1 * b1 * c0,
                                a0 * b0 * c1,
                                a0 * b1 * c1,
                                a1 * b0 * c1,
                                a1 * b1 * c1,
                            ]
                            all_points.append(point)

    return np.array(all_points, dtype=np.float32)


def Points_222():
    all_points = []

    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    point = [
                        a0,
                        a1,
                        b0,
                        b1,
                        a0 * b0,
                        a0 * b1,
                        a1 * b0,
                        a1 * b1,
                    ]
                    all_points.append(point)

    return np.array(all_points, dtype=np.float32)


def Points_232():
    all_points = []

    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for a2 in [1, -1]:
                for b0 in [1, -1]:
                    for b1 in [1, -1]:
                        for b2 in [1, -1]:
                            point = [
                                a0,
                                a1,
                                a2,
                                b0,
                                b1,
                                b2,
                                a0 * b0,
                                a0 * b1,
                                a0 * b2,
                                a1 * b0,
                                a1 * b1,
                                a1 * b2,
                                a2 * b0,
                                a2 * b1,
                                a2 * b2,
                            ]
                            all_points.append(point)

    return np.array(all_points, dtype=np.float32)


def Points_223():
    all_points = []

    def encode(x):
        if x == 0:
            return [1, 1]
        if x == 1:
            return [-1, 1]
        if x == 2:
            return [1, -1]
        raise ValueError(f"Unsupported ternary outcome: {x}")

    for a0 in [0, 1, 2]:
        for a1 in [0, 1, 2]:
            for b0 in [0, 1, 2]:
                for b1 in [0, 1, 2]:
                    a01, a02 = encode(a0)
                    a11, a12 = encode(a1)
                    b01, b02 = encode(b0)
                    b11, b12 = encode(b1)
                    point = [
                        a01,
                        a02,
                        a11,
                        a12,
                        b01,
                        b02,
                        b11,
                        b12,
                        a01 * b01,
                        a01 * b02,
                        a02 * b01,
                        a02 * b02,
                        a01 * b11,
                        a01 * b12,
                        a02 * b11,
                        a02 * b12,
                        a11 * b01,
                        a11 * b02,
                        a12 * b01,
                        a12 * b02,
                        a11 * b11,
                        a11 * b12,
                        a12 * b11,
                        a12 * b12,
                    ]
                    all_points.append(point)

    return np.array(all_points, dtype=np.float32)


points_222 = Points_222
points_223 = Points_223
points_232 = Points_232
points_322 = Points_322
