from aibell import true_Q_solver
from math import sqrt
import numpy as np
# 2-2-2 scenario, 8-dim
# p = [<A_0>, <A_1>, <B_0>, <B_1>, <A_0*B_0>, <A_0*B_1>, <A_1*B_0>, <A_1*B_1>,]

# 初始化 2-Party Solver
solver = true_Q_solver(n_parties=2, level=1)

# ==============================
# 场景一：直接输入向量
# ==============================
# 可能出现的s向量：
s_1 = [0.5, -0.5, 0, 0, 0, 0.5, 0, -0.5]
s_2 = [0.5773502691896258, 0, 0, 0.5773502691896258, 0, 0.5773502691896258, 0, 0]
s_3 = [-0.5773502691896258, 0, 0, -0.5773502691896258, 0, -0.5773502691896258, 0, 0]
s_4 = [0, 0, 0, 0, -0.5, 0.5, -0.5, -0.5]
s_5 = [0.000, -0.447, -0.447, 0.000, 0.447, -0.447, 0.000, 0.447]
s_6 = [0.000, 0.447, 0.447, 0.000, -0.447, 0.447, 0.000, -0.447]
# Batch 计算：
Q_values_batch = solver.compute_from_batch([s_1,s_2,s_3,s_4,s_5,s_6])
print(f"s: {s_1}, Q_values: {Q_values_batch[0]}")
print(f"s: {s_2}, Q_values: {Q_values_batch[1]}")
print(f"s: {s_3}, Q_values: {Q_values_batch[2]}")
print(f"s: {s_4}, Q_values: {Q_values_batch[3]}")
print(f"s: {s_5}, Q_values: {Q_values_batch[4]}")
print(f"s: {s_6}, Q_values: {Q_values_batch[5]}")
# ==============================
# 场景二：从变量名间接生成s向量（不常用）
# ==============================
# 我们先创建一个全 0 向量
chsh_vector = np.zeros(len(solver.term_list))
idx1 = solver._find_index_by_names(('A0', 'B0'))
idx2 = solver._find_index_by_names(('A0', 'B1'))
idx3 = solver._find_index_by_names(('A1', 'B0'))
idx4 = solver._find_index_by_names(('A1', 'B1'))

if None not in [idx1, idx2, idx3, idx4]:
    chsh_vector[idx1] = 1
    chsh_vector[idx2] = 1
    chsh_vector[idx3] = 1
    chsh_vector[idx4] = -1

    chsh_normalized = chsh_vector / np.linalg.norm(chsh_vector)


    print("\n正在计算传入的 CHSH 向量...")
    # 传入 batch (这里 batch_size=1)
    rewards = solver.compute_from_batch([chsh_normalized])

    print(f"CHSH 计算结果: {rewards[0]:.5f}")
    print(f"理论值: {2 * np.sqrt(2):.5f}")



