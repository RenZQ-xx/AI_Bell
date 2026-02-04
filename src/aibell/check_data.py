import numpy as np

# 加载 npz 文件
data = np.load('../../data/data_222.npz')
# 查看文件中有哪些数组
print("Keys in npz file:", data.files)

# 查看每个数组的内容
for key in data.files:
    print(f"\nKey: {key}")
    print(f"Shape: {data[key].shape}")
    print(f"Data type: {data[key].dtype}")
    print(f"First few values:\n{data[key][:5] if data[key].size > 5 else data[key]}")  # 显示前5个值
