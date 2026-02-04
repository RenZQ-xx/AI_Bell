# ===============================
# How to calculate C(s) and Q(s)?
# ===============================

from aibell import true_Q_solver, get_true_C, Points_222
# We discard the function "get_ture_Q"
import numpy as np

# ============================================================================
# 2-2-2 scenario, 8-dim
# p = [<A_0>, <A_1>, <B_0>, <B_1>, <A_0*B_0>, <A_0*B_1>, <A_1*B_0>, <A_1*B_1>,]
# ============================================================================

# Step 1: Initialize 2-Party Solver and settings (only once)
solver = true_Q_solver(n_parties=2, level=1, verbose=False)
points = Points_222()

# Step 2: Generate s_vector batch
s_batch = [[0.5, -0.5, 0, 0, 0, 0.5, 0, -0.5],
           [0.5773502691896258, 0, 0, 0.5773502691896258, 0, 0.5773502691896258, 0, 0],
           [-0.5773502691896258, 0, 0, -0.5773502691896258, 0, -0.5773502691896258, 0, 0],
           [0, 0, 0, 0, -0.5, 0.5, -0.5, -0.5],
           [0.000, -0.447, -0.447, 0.000, 0.447, -0.447, 0.000, 0.447],
           [0.000, 0.447, 0.447, 0.000, -0.447, 0.447, 0.000, -0.447]]

# Step 3: Calculate Q and C
# Batch 计算：
Q_values_batch = solver.compute_from_batch(s_batch)
C_values_batch = get_true_C(s_batch, points)
for i in range(len(s_batch)):
    print(f"s: {s_batch[i]}, \n Q_values: {Q_values_batch[i]}, C_values: {C_values_batch[i]}")


'''
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

'''

