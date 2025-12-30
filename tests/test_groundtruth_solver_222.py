from aibell import get_true_C, get_true_Q
from math import sqrt
import numpy as np
# 2-2-2 scenario, 8-dim
# p = [<A_0>, <A_1>, <B_0>, <B_1>, <A_0*B_0>, <A_0*B_1>, <A_1*B_0>, <A_1*B_1>,]
points = []
# 第一组：Alice 总是出 +1 (即 $A_0=1, A_1=1$)
points.extend([[1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, -1, 1, -1, 1, -1],
               [1, 1, -1, 1, -1, 1, -1, 1],
               [1, 1, -1, -1, -1, -1, -1, -1],])
# 第二组：Alice 总是出 -1 (即 $A_0=-1, A_1=-1$)
points.extend([[-1, -1, 1, 1, -1, -1, -1, -1],
               [-1, -1, 1, -1, -1, 1, -1, 1],
               [-1, -1, -1, 1, 1, -1, 1, -1],
               [-1, -1, -1, -1, 1, 1, 1, 1],])
# 第三组：Alice 跟随输入 (即 $A_0=1, A_1=-1$)
points.extend([[1, -1, 1, 1, 1, 1, -1, -1],
               [1, -1, 1, -1, 1, -1, -1, 1],
               [1, -1, -1, 1, -1, 1, 1, -1],
               [1, -1, -1, -1, -1, -1, 1, 1],])
# 第四组：Alice 反转输入 (即 $A_0=-1, A_1=1$)
points.extend([[-1, 1, 1, 1, -1, -1, 1, 1],
               [-1, 1, 1, -1, -1, 1, 1, -1],
               [-1, 1, -1, 1, 1, -1, -1, 1],
               [-1, 1, -1, -1, 1, 1, -1, -1],])
# CHSH inequality
s_0 = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, -0.5])

C_value = get_true_C(s_0, np.array(points))
Q_value = get_true_Q(s_0)


assert C_value[0] == 1.
assert abs(Q_value[0][0] + sqrt(2)) < 1e-7

