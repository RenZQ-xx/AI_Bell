from pathlib import Path


def generate_lrs_input():
    # 222 场景：遍历 a0,a1,b0,b1，构造 8 维点
    points = []
    for a0 in [1, -1]:
        for a1 in [1, -1]:
            for b0 in [1, -1]:
                for b1 in [1, -1]:
                    p = [
                        a0, a1, b0, b1,
                        a0 * b0, a0 * b1, a1 * b0, a1 * b1,
                    ]
                    points.append(p)

    unique_points = sorted(set(tuple(p) for p in points))
    n_rows = len(unique_points)
    n_cols = len(unique_points[0]) if unique_points else 0

    project_root = Path(__file__).resolve().parents[4]
    file_path = project_root / "data" / "polytope_222.ext"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        f.write("my_polytope_8d_222\n")
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{n_rows} {n_cols + 1} rational\n")

        for p in unique_points:
            line = "1 " + " ".join(map(str, p))
            f.write(line + "\n")

        f.write("end\n")

    print(f"文件已成功保存至: {file_path}")
    print(f"共 {n_rows} 个顶点, 维度 {n_cols}")


if __name__ == "__main__":
    generate_lrs_input()
