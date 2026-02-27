import numpy as np
from pathlib import Path
'''
def generate_lrs_input():
    # 1. 生成你的 64 个点 (复用之前的逻辑)
    points = []
    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    for c0 in [1, -1]:
                        for c1 in [1, -1]:
                            # 原始 26 维坐标
                            p = [a0, a1, b0, b1, c0, c1,
                                 a0*b0, a0*b1, a1*b0, a1*b1,
                                 a0*c0, a0*c1, a1*c0, a1*c1,
                                 b0*c0, b0*c1, b1*c0, b1*c1,
                                 a0*b0*c0, a0*b1*c0, a1*b0*c0, a1*b1*c0,
                                 a0*b0*c1, a0*b1*c1, a1*b0*c1, a1*b1*c1]
                            points.append(p)
    
    # 转为 numpy 数组去重 (虽然后面的逻辑也能处理，但去重更好)
    unique_points = np.unique(points, axis=0)
    n_rows, n_cols = unique_points.shape # 应该是 64, 26
    
    
    # 2. 写入 .ext 文件
    with open("polytope.ext", "w") as f:
        f.write("my_polytope_26d\n")
        f.write("V-representation\n") # 告诉 lrs 这是顶点
        f.write("begin\n")
        # 格式: 行数 列数 rational
        # 列数 = 维度 + 1 (因为齐次坐标需要加一列 '1')
        f.write(f"{n_rows} {n_cols + 1} rational\n")
        
        for p in unique_points:
            # V-rep 格式要求每一行以 1 开头，后面跟着坐标
            # 1 coord1 coord2 ... coord26
            line = "1 " + " ".join(map(str, p))
            f.write(line + "\n")
            
        f.write("end\n")

    print(f"文件已生成: polytope.ext")
    print(f"包含 {n_rows} 个点，维度 {n_cols}")
'''
def generate_lrs_input():

    # 1. 生成你的 64 个点 (复用之前的逻辑)
    points = []
    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    for c0 in [1, -1]:
                        for c1 in [1, -1]:
                            # 原始 26 维坐标
                            p = [a0, a1, b0, b1, c0, c1,
                                 a0*b0, a0*b1, a1*b0, a1*b1,
                                 a0*c0, a0*c1, a1*c0, a1*c1,
                                 b0*c0, b0*c1, b1*c0, b1*c1,
                                 a0*b0*c0, a0*b1*c0, a1*b0*c0, a1*b1*c0,
                                 a0*b0*c1, a0*b1*c1, a1*b0*c1, a1*b1*c1]
                            points.append(p)
    
    # 转为 numpy 数组去重 (虽然后面的逻辑也能处理，但去重更好)
    unique_points = np.unique(points, axis=0)
    n_rows, n_cols = unique_points.shape # 应该是 64, 26


    # 1. 构建目标路径：固定写到项目根目录 data/polytope_322.ext
    project_root = Path(__file__).resolve().parents[4]
    file_path = project_root / "data" / "polytope_322.ext"

    # 2. 关键步骤：确保 'data' 文件夹存在
    # 如果 data 文件夹不存在，这行代码会自动创建它，防止报错
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. 写入文件 (使用新的 file_path)
    with open(file_path, "w") as f:
        f.write("my_polytope_26d\n")
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{n_rows} {n_cols + 1} rational\n")
    
        for p in unique_points:
            line = "1 " + " ".join(map(str, p))
            f.write(line + "\n")
        
        f.write("end\n")

    print(f"文件已成功保存至: {file_path}")



if __name__ == "__main__":
    generate_lrs_input()
