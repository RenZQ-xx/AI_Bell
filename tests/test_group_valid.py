from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

# 1. 定义原有的 6 个生成元
# 注意：必须将循环组放在一个列表中: [[a,b], [c,d], ...]
g1 = Permutation([[0, 12], [1, 13], [2, 14], [3, 15]])
g2 = Permutation([[0, 4], [1, 5], [2, 6], [3, 7]])
g3 = Permutation([[0, 8], [1, 9], [2, 10], [3, 11]])
g4 = Permutation([[0, 1], [4, 5], [8, 9], [12, 13]])
g5 = Permutation([[0, 2], [4, 6], [8, 10], [12, 14]])
g6 = Permutation([[0, 3], [4, 7], [8, 11], [12, 15]])

# 2. 定义你想添加的额外生成元
# 对于单个对换，可以直接传两个整数 Permutation(a, b) 或列表 Permutation([[a, b]])
extra1 = Permutation(0, 3)   # 对应 (0 3)
extra2 = Permutation(0, 12)  # 对应 (0 12)

# 3. 创建群并验证
# 将所有生成元放入列表中
generators = [g1, g2, g3, g4, g5, g6, extra1, extra2]
G = PermutationGroup(generators)

# 4. 输出结果
print("正在计算群的阶 (Order)...")
order = G.order()
print(f"群的阶: {order}")

# 16! = 20,922,789,888,000
fact_16 = 20922789888000
if order == fact_16:
    print("成功！该生成组可以覆盖所有 16! 种排列。")
    print(f"这是一个全对称群 S_16: {G.is_symmetric}")
else:
    print(f"失败。生成的群大小仅为: {order}")
    print(f"覆盖率: {order / fact_16:.20f}")

# 测试是否能生成任意相邻对换 (例如 (1 2))
# 如果是全对称群，这应该返回 True
test_swap = Permutation(1, 2)
print(f"是否包含 (1 2): {test_swap in G}")
