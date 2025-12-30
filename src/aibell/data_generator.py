from src.aibell.sampler import Sampler222
from src.aibell.Solver_222 import get_quantum_bound_batch

import numpy as np
sampler2 = Sampler222()
# 生成 50 个样本，30% 随机，70% CHSH 扰动
data_2 = sampler2.generate_data(n_samples=1000000, ratio_random=0.3)
bounds = get_quantum_bound_batch(data_2, level=1)

# 统计成功率
success_count = np.sum(~np.isnan(bounds[:, 0]))
print(f"成功求解: {success_count}/{len(data_2)}")



def save_dataset(s_data, bounds, filename="data_222.npz"):
    """
    清洗并将输入向量和计算结果保存为同一个 .npz 文件

    参数:
    s_data: 输入向量 (N, 8)
    bounds: 计算结果 (N, 2)
    filename: 保存路径
    """
    print(f"原始数据量: {len(s_data)}")

    # 1. 找出成功的索引 (即 bounds 第一列不是 NaN 的行)
    # np.isnan(bounds[:, 0]) 返回布尔数组，~ 取反
    valid_mask = ~np.isnan(bounds[:, 0])

    # 2. 筛选数据
    clean_inputs = s_data[valid_mask]
    clean_labels = bounds[valid_mask]

    print(f"清洗后数据量: {len(clean_inputs)} (剔除了 {len(s_data) - len(clean_inputs)} 条失败样本)")

    # 3. 保存为压缩格式 (.npz)
    # 这种格式读取速度极快，且节省空间
    np.savez_compressed(
        filename,
        inputs=clean_inputs,  # 键名可以自定义，这里叫 inputs
        labels=clean_labels  # 这里叫 labels
    )
    print(f"数据已保存至: {filename}")


# --- 使用示例 ---
# 假设 s_data 是你的输入，results 是刚刚算出来的结果
save_dataset(data_2, bounds, )
