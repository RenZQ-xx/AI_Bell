import numpy as np
import itertools

N = 3
O = 2
def generate_points(n = N):
    """生成 4^N 个 3^N - 1 维的点"""
    points = []
    for i in range(4**n):
        # 将 i 转换为 2 进制表示，得到一个长度为 2N 的数组，每个元素在 [0, 1] 之间, 并将所有的0替换为-1
        point = [2 * ((i >> j) & 1) - 1 for j in range(2*n)]
        for elenum in range(2, N + 1):
            for pair in itertools.combinations(range(N), elenum):
                for offset in range(O ** elenum):
                    res = 1
                    for idx in pair:
                        res *= point[2 * idx + ((offset >> idx) & 1)]  # 计算 A_j * B_k 项
                    point.append(res)  # 添加 A_j * B_k 项
             
        point = np.array(point, dtype=np.float32)
        points.append(point)

    points = np.array(points, dtype=np.float32)
    return points

if __name__ == "__main__":
    points = generate_points()
    print(points[1])
    
    
    