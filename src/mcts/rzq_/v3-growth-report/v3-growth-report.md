# v3：300 轮完整树生长记录

300 次扩展动作，277 个新节点 + 根节点 = 278 个节点；23 次对称复用。节点编号按首次创建顺序分配。

Q 和候选数值来自原始 trace。compat 为当前参考数据库重算，并已与 300 次实际选中 expansion 的历史 compat 全部核对一致；compat 表包含 46 类，已发现集合另列。兼容表示存在包含当前所选 blocks 的参考 facet mask，不等于实际命中。

K=floor(1+sqrt(max(1,visits)))，另有两个 bucket 的最低配额。selection 仅在不可 expansion 时进入；minimum_visits 阶段只要求最少访问候选，按 prior 抽样，并不要求 UCB 最大。

## iteration 1

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[27, 3, 23, 34, 11, 37, 29, 17, 6, 38, 20, 35, 22, 30, 10, 21, 4]

### N0：expansion → action 24

path=[]；visits=0；children=0；K=2。已有 0 条动作边 < K=2，且尚余 3 个代表动作。trace 行 3。

bucket=0，compatibility_richness_prior；到达 N1（新建）；closure=[]。

## iteration 2

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 11, 16, 27, 37, 6, 19, 3, 31, 36, 13, 33, 38, 26, 29, 22, 12]

### N0：expansion → action 0

path=[]；visits=1；children=1；K=2。已有 1 条动作边 < K=2，且尚余 2 个代表动作。trace 行 26。

bucket=1，uniform_random；到达 N2（新建）；closure=[]。

## iteration 3

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 2, 8, 26, 5, 17, 19, 3, 15, 29, 39, 35, 33, 20, 37, 11]

### N0：selection → action 0

path=[]；visits=2；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [24, 0]，并列按 prior 抽样。trace 行 49。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 1 | 0.3427686 | 58.072408 | 0.0 | 0.3393236 | 0.3393236 |  |
| 0 | 1 | 0.3308308 | 68.4391879 | 1.0 | 0.3275058 | 1.3275058 | ✓ |

### N2：expansion → action 28

path=[0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 51。

bucket=0，compatibility_richness_prior；到达 N3（新建）；closure=[]。

## iteration 4

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[3, 21, 39, 5, 27, 37, 26, 9, 25, 29, 35, 4, 10, 30, 19, 36, 13]

### N0：selection → action 24

path=[]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [24]，并列按 prior 抽样。trace 行 73。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 1 | 0.3427686 | 58.072408 | 0.0 | 0.4155848 | 0.4155848 | ✓ |
| 0 | 2 | 0.3308308 | 65.564188 | 1.0 | 0.2674074 | 1.2674074 |  |

### N1：expansion → action 22

path=[24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 20 个代表动作。trace 行 75。

bucket=0，compatibility_richness_prior；到达 N4（新建）；closure=[]。

## iteration 5

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[1, 25, 21, 8, 4, 10, 32, 15, 7, 9, 36, 19, 30, 5, 22, 12]

### N0：expansion → action 2

path=[]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 1 个代表动作。trace 行 98。

bucket=0，compatibility_richness_prior；到达 N5（新建）；closure=[]。

## iteration 6

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[26, 4, 33, 21, 34, 36, 7, 1, 10, 31, 12, 29, 23, 20, 28]

### N0：selection → action 2

path=[]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [2]，并列按 prior 抽样。trace 行 120。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 2 | 0.3427686 | 63.8579015 | 0.0 | 0.3576785 | 0.3576785 |  |
| 0 | 2 | 0.3308308 | 65.564188 | 0.0900436 | 0.3452214 | 0.435265 |  |
| 2 | 1 | 0.3264006 | 82.8074626 | 1.0 | 0.5108977 | 1.5108977 | ✓ |

### N5：expansion → action 9

path=[2]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 14 个代表动作。trace 行 122。

bucket=0，compatibility_richness_prior；到达 N6（新建）；closure=[]。

## iteration 7

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 8, 35, 36, 32, 22, 5, 15, 17, 21, 34, 29, 23, 10, 30, 0]

### N0：selection → action 2

path=[]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [24, 0, 2]，并列按 prior 抽样。trace 行 143。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 2 | 0.3427686 | 63.8579015 | 0.0 | 0.3918171 | 0.3918171 |  |
| 0 | 2 | 0.3308308 | 65.564188 | 0.2065449 | 0.3781711 | 0.5847161 |  |
| 2 | 2 | 0.3264006 | 72.118992 | 1.0 | 0.3731069 | 1.3731069 | ✓ |

### N5：expansion → action 37

path=[2]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 13 个代表动作。trace 行 145。

bucket=1，uniform_random；到达 N7（新建）；closure=[]。

## iteration 8

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class28；rollout：[25, 12, 11, 18, 2, 1, 39, 21, 16, 26, 15, 30, 28, 32, 13]

### N0：selection → action 0

path=[]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [24, 0]，并列按 prior 抽样。trace 行 167。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 2 | 0.3427686 | 63.8579015 | 0.0 | 0.4232109 | 0.4232109 |  |
| 0 | 2 | 0.3308308 | 65.564188 | 0.1755229 | 0.4084715 | 0.5839945 | ✓ |
| 2 | 3 | 0.3264006 | 73.5790608 | 1.0 | 0.3022512 | 1.3022512 |  |

### N2：expansion → action 23

path=[0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 16 个代表动作。trace 行 169。

bucket=1，uniform_random；到达 N8（新建）；closure=[]。

## iteration 9

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[28, 15, 14, 20, 38, 0, 6, 22, 16, 17, 19, 27, 32, 25, 9, 7]

### N0：selection → action 24

path=[]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [24]，并列按 prior 抽样。trace 行 190。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 2 | 0.3427686 | 63.8579015 | 0.0 | 0.4524315 | 0.4524315 | ✓ |
| 0 | 3 | 0.3308308 | 70.5309468 | 0.6864454 | 0.3275058 | 1.0139512 |  |
| 2 | 3 | 0.3264006 | 73.5790608 | 1.0 | 0.3231201 | 1.3231201 |  |

### N1：expansion → action 3

path=[24]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 19 个代表动作。trace 行 192。

bucket=1，uniform_random；到达 N9（新建）；closure=[]。

## iteration 10

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[26, 34, 5, 33, 11, 21, 25, 8, 14, 16, 6, 23, 22, 12]

### N0：selection → action 2

path=[]；visits=9；children=3；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [24, 0, 2]，并列按 prior 抽样。trace 行 214。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 3 | 0.3427686 | 67.6355973 | 0.0 | 0.359907 | 0.359907 |  |
| 0 | 3 | 0.3308308 | 70.5309468 | 0.4871485 | 0.3473724 | 0.8345209 |  |
| 2 | 3 | 0.3264006 | 73.5790608 | 1.0 | 0.3427206 | 1.3427206 | ✓ |

### N5：selection → action 9

path=[2]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [9, 37]，并列按 prior 抽样。trace 行 216。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 1 | 0.073067 | 61.4305213 | 0.0 | 0.0885891 | 0.0885891 | ✓ |
| 37 | 1 | 0.072067 | 76.4991986 | 1.0 | 0.0873767 | 1.0873767 |  |

### N6：expansion → action 0

path=[2, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 9 个代表动作。trace 行 218。

bucket=0，compatibility_richness_prior；到达 N10（新建）；closure=[]。

## iteration 11

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 27, 20, 11, 16, 37, 22, 33, 35, 21, 31, 25, 28, 36, 13]

### N0：selection → action 24

path=[]；visits=10；children=3；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [24, 0]，并列按 prior 抽样。trace 行 238。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 3 | 0.3427686 | 67.6355973 | 0.0 | 0.3793753 | 0.3793753 | ✓ |
| 0 | 3 | 0.3308308 | 70.5309468 | 0.4394344 | 0.3661626 | 0.8055971 |  |
| 2 | 4 | 0.3264006 | 74.224406 | 1.0 | 0.2890074 | 1.2890074 |  |

### N1：selection → action 3

path=[24]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [22, 3]，并列按 prior 抽样。trace 行 240。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 1 | 0.0579127 | 69.643395 | 0.0 | 0.0702154 | 0.0702154 |  |
| 3 | 1 | 0.0586854 | 75.1909889 | 1.0 | 0.0711523 | 1.0711523 | ✓ |

### N9：expansion → action 9

path=[24, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 242。

bucket=0，compatibility_richness_prior；到达 N11（新建）；closure=[]。

## iteration 12

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[33, 14, 22, 9, 6, 16, 34, 21, 26, 24, 36, 11, 2, 12]

### N0：selection → action 0

path=[]；visits=11；children=3；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [0]，并列按 prior 抽样。trace 行 263。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 4 | 0.3427686 | 67.9982816 | 0.0 | 0.3183137 | 0.3183137 |  |
| 0 | 3 | 0.3308308 | 70.5309468 | 0.4067804 | 0.3840346 | 0.790815 | ✓ |
| 2 | 4 | 0.3264006 | 74.224406 | 1.0 | 0.3031135 | 1.3031135 |  |

### N2：selection → action 28

path=[0]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [28, 23]，并列按 prior 抽样。trace 行 265。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 1 | 0.0670232 | 62.6891881 | 0.0 | 0.0812613 | 0.0812613 | ✓ |
| 23 | 1 | 0.0316243 | 80.4644643 | 1.0 | 0.0383425 | 1.0383425 |  |

### N3：expansion → action 25

path=[0, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 38 个代表动作。trace 行 267。

bucket=0，compatibility_richness_prior；到达 N12（新建）；closure=[]。

## iteration 13

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 22, 33, 9, 26, 8, 21, 35, 12, 10, 32, 37, 7, 24, 31, 3, 19]

### N0：selection → action 2

path=[]；visits=12；children=3；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [24, 0, 2]，并列按 prior 抽样。trace 行 287。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 4 | 0.3427686 | 67.9982816 | 0.0 | 0.3324679 | 0.3324679 |  |
| 0 | 4 | 0.3308308 | 68.9185879 | 0.1478137 | 0.3208889 | 0.4687025 |  |
| 2 | 4 | 0.3264006 | 74.224406 | 1.0 | 0.3165917 | 1.3165917 | ✓ |

### N5：expansion → action 0

path=[2]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 12 个代表动作。trace 行 289。

bucket=0，compatibility_richness_prior；到达 N13（新建）；closure=[]。

## iteration 14

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[0, 6, 11, 20, 35, 29, 25, 22, 26, 37, 12, 30, 13, 38, 10, 1]

### N0：selection → action 24

path=[]；visits=13；children=3；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [24, 0]，并列按 prior 抽样。trace 行 312。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 4 | 0.3427686 | 67.9982816 | 0.0 | 0.3460435 | 0.3460435 | ✓ |
| 0 | 4 | 0.3308308 | 68.9185879 | 0.1450783 | 0.3339917 | 0.47907 |  |
| 2 | 5 | 0.3264006 | 74.3417953 | 1.0 | 0.2745993 | 1.2745993 |  |

### N1：expansion → action 8

path=[24]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 18 个代表动作。trace 行 314。

bucket=0，compatibility_richness_prior；到达 N14（新建）；closure=[]。

## iteration 15

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 8, 11, 16, 20, 22, 15, 37, 18, 25, 30, 32, 28, 27, 24, 1]

### N0：selection → action 0

path=[]；visits=14；children=3；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [0]，并列按 prior 抽样。trace 行 336。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.2992553 | 0.2992553 |  |
| 0 | 4 | 0.3308308 | 68.9185879 | 0.1184343 | 0.3465996 | 0.4650339 | ✓ |
| 2 | 5 | 0.3264006 | 74.3417953 | 1.0 | 0.2849651 | 1.2849651 |  |

### N2：expansion → action 6

path=[0]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 15 个代表动作。trace 行 338。

bucket=0，compatibility_richness_prior；到达 N15（新建）；closure=[]。

## iteration 16

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[25, 5, 14, 11, 36, 13, 16, 31, 33, 19, 34, 1, 4, 12]

### N0：selection → action 2

path=[]；visits=15；children=3；K=4。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 360。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3097586 | 0.3097586 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.1944084 | 0.2989705 | 0.493379 |  |
| 2 | 5 | 0.3264006 | 74.3417953 | 1.0 | 0.2949669 | 1.2949669 | ✓ |

### N5：selection → action 0

path=[2]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [37, 0]，并列按 prior 抽样。trace 行 362。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.073067 | 68.7954814 | 0.0 | 0.0762453 | 0.0762453 |  |
| 37 | 1 | 0.072067 | 76.4991986 | 1.0 | 0.1128028 | 1.1128028 |  |
| 0 | 1 | 0.085828 | 74.8113523 | 0.780905 | 0.134342 | 0.915247 | ✓ |

### N13：expansion → action 6

path=[2, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 28 个代表动作。trace 行 364。

bucket=0，compatibility_richness_prior；到达 N16（新建）；closure=[]。

## iteration 17

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 38, 12, 39, 0, 9, 36, 32, 19, 7, 28, 31, 11, 5, 22, 10]

### N0：selection → action 2

path=[]；visits=16；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 384。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3199174 | 0.3199174 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.2761987 | 0.3087754 | 0.5849741 |  |
| 2 | 6 | 0.3264006 | 72.520076 | 1.0 | 0.2611205 | 1.2611205 | ✓ |

### N5：selection → action 37

path=[2]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [37]，并列按 prior 抽样。trace 行 386。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.073067 | 68.7954814 | 0.0 | 0.0835226 | 0.0835226 |  |
| 37 | 1 | 0.072067 | 76.4991986 | 1.0 | 0.1235692 | 1.1235692 | ✓ |
| 0 | 2 | 0.085828 | 69.111416 | 0.0410107 | 0.0981096 | 0.1391202 |  |

### N7：expansion → action 3

path=[2, 37]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 388。

bucket=0，compatibility_richness_prior；到达 N17（新建）；closure=[]。

## iteration 18

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 38, 30, 7, 39, 8, 32, 29, 35, 6, 24, 4, 12, 15, 9]

### N0：selection → action 2

path=[]；visits=17；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 410。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3297633 | 0.3297633 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.2606177 | 0.3182784 | 0.5788961 |  |
| 2 | 7 | 0.3264006 | 72.7789489 | 1.0 | 0.2355122 | 1.2355122 | ✓ |

### N5：selection → action 0

path=[2]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 37, 0]，并列按 prior 抽样。trace 行 412。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.073067 | 68.7954814 | 0.0 | 0.0902147 | 0.0902147 |  |
| 37 | 2 | 0.072067 | 75.4156924 | 1.0 | 0.08898 | 1.08898 |  |
| 0 | 2 | 0.085828 | 69.111416 | 0.0477227 | 0.1059704 | 0.1536932 | ✓ |

### N13：expansion → action 23

path=[2, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 27 个代表动作。trace 行 414。

bucket=1，uniform_random；到达 N18（新建）；closure=[]。

## iteration 19

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[23, 16, 8, 33, 14, 35, 11, 32, 30, 10, 19, 3, 39, 25]

### N0：selection → action 2

path=[]；visits=18；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 435。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3393236 | 0.3393236 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.3246008 | 0.3275058 | 0.6521067 |  |
| 2 | 8 | 0.3264006 | 71.8744067 | 1.0 | 0.2154134 | 1.2154134 | ✓ |

### N5：selection → action 9

path=[2]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 37]，并列按 prior 抽样。trace 行 437。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.073067 | 68.7954814 | 0.1165841 | 0.0964436 | 0.2130277 | ✓ |
| 37 | 2 | 0.072067 | 75.4156924 | 1.0 | 0.0951237 | 1.0951237 |  |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0 | 0.0849654 | 0.0849654 |  |

### N6：expansion → action 15

path=[2, 9]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 8 个代表动作。trace 行 439。

bucket=1，uniform_random；到达 N19（新建）；closure=[]。

## iteration 20

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 13, 26, 8, 16, 4, 27, 37, 3, 35, 32, 36, 24, 30, 10, 12]

### N0：selection → action 2

path=[]；visits=19；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 459。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3486219 | 0.3486219 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.4146456 | 0.3364802 | 0.7511258 |  |
| 2 | 9 | 0.3264006 | 71.0742995 | 1.0 | 0.1991846 | 1.1991846 | ✓ |

### N5：expansion → action 6

path=[2]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 11 个代表动作。trace 行 461。

bucket=1，uniform_random；到达 N20（新建）；closure=[]。

## iteration 21

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[13, 25, 19, 9, 34, 26, 10, 28, 36, 7, 39, 32, 27, 30, 3, 18]

### N0：selection → action 2

path=[]；visits=20；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 483。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3576785 | 0.3576785 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.3755834 | 0.3452214 | 0.7208048 |  |
| 2 | 10 | 0.3264006 | 71.3742775 | 1.0 | 0.185781 | 1.185781 | ✓ |

### N5：selection → action 6

path=[2]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [6]，并列按 prior 抽样。trace 行 485。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0808704 | 0.0808704 |  |
| 37 | 2 | 0.072067 | 75.4156924 | 1.0 | 0.1063515 | 1.1063515 |  |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0625884 | 0.0949942 | 0.1575826 |  |
| 6 | 1 | 0.0814506 | 74.0740801 | 0.8321773 | 0.1802986 | 1.0124759 | ✓ |

### N20：expansion → action 21

path=[2, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 38 个代表动作。trace 行 487。

bucket=0，compatibility_richness_prior；到达 N21（新建）；closure=[]。

## iteration 22

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 28, 33, 11, 34, 39, 23, 25, 31, 16, 5, 37, 10, 1, 12]

### N0：selection → action 2

path=[]；visits=21；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 509。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3665114 | 0.3665114 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.3907082 | 0.3537467 | 0.7444549 |  |
| 2 | 11 | 0.3264006 | 71.2510105 | 1.0 | 0.1745048 | 1.1745048 | ✓ |

### N5：selection → action 6

path=[2]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [37, 6]，并列按 prior 抽样。trace 行 511。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0848176 | 0.0848176 |  |
| 37 | 2 | 0.072067 | 75.4156924 | 1.0 | 0.1115424 | 1.1115424 |  |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0625884 | 0.0996307 | 0.1622192 |  |
| 6 | 2 | 0.0814506 | 72.0462102 | 0.5785104 | 0.1260658 | 0.7045763 | ✓ |

### N20：expansion → action 38

path=[2, 6]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 37 个代表动作。trace 行 513。

bucket=1，uniform_random；到达 N22（新建）；closure=[]。

## iteration 23

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 34, 33, 18, 9, 14, 5, 28, 1, 13, 27, 29, 31, 15, 7]

### N0：selection → action 2

path=[]；visits=22；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 534。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3751364 | 0.3751364 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.4293703 | 0.3620713 | 0.7914416 |  |
| 2 | 12 | 0.3264006 | 70.975386 | 1.0 | 0.164872 | 1.164872 | ✓ |

### N5：selection → action 37

path=[2]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [37]，并列按 prior 抽样。trace 行 536。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0885891 | 0.0885891 |  |
| 37 | 2 | 0.072067 | 75.4156924 | 1.0 | 0.1165022 | 1.1165022 | ✓ |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0625884 | 0.1040609 | 0.1666493 |  |
| 6 | 3 | 0.0814506 | 70.6786455 | 0.4074413 | 0.0987536 | 0.5061949 |  |

### N7：expansion → action 36

path=[2, 37]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 18 个代表动作。trace 行 538。

bucket=1，uniform_random；到达 N23（新建）；closure=[]。

## iteration 24

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 13, 34, 37, 3, 9, 11, 36, 31, 4, 24, 25, 7, 22, 17]

### N0：selection → action 2

path=[]；visits=23；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 559。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3835674 | 0.3835674 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.4219923 | 0.3702088 | 0.792201 |  |
| 2 | 13 | 0.3264006 | 71.0240851 | 1.0 | 0.1565362 | 1.1565362 | ✓ |

### N5：selection → action 6

path=[2]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [9, 37, 0, 6]，并列按 prior 抽样。trace 行 561。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0922064 | 0.0922064 |  |
| 37 | 3 | 0.072067 | 74.1466197 | 1.0 | 0.0909445 | 1.0909445 |  |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0743992 | 0.10831 | 0.1827092 |  |
| 6 | 3 | 0.0814506 | 70.6786455 | 0.4843277 | 0.102786 | 0.5871137 | ✓ |

### N20：selection → action 38

path=[2, 6]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [21, 38]，并列按 prior 抽样。trace 行 563。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 1 | 0.0231677 | 70.0183403 | 1.0 | 0.0280894 | 1.0280894 |  |
| 38 | 1 | 0.0257828 | 67.943516 | 0.0 | 0.03126 | 0.03126 | ✓ |

### N22：expansion → action 32

path=[2, 6, 38]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 565。

bucket=0，compatibility_richness_prior；到达 N24（新建）；closure=[]。

## iteration 25

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[3, 1, 28, 5, 7, 14, 16, 4, 13, 33, 6, 10, 17]

### N0：selection → action 2

path=[]；visits=24；children=3；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 586。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3918171 | 0.3918171 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.5254821 | 0.3781711 | 0.9036532 |  |
| 2 | 14 | 0.3264006 | 70.465934 | 1.0 | 0.1492428 | 1.1492428 | ✓ |

### N5：selection → action 37

path=[2]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [9, 37, 0]，并列按 prior 抽样。trace 行 588。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0956871 | 0.0956871 |  |
| 37 | 3 | 0.072067 | 74.1466197 | 1.0 | 0.0943776 | 1.0943776 | ✓ |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0743992 | 0.1123986 | 0.1867978 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.206688 | 0.0853329 | 0.2920209 |  |

### N7：selection → action 36

path=[2, 37]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [3, 36]，并列按 prior 抽样。trace 行 590。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | 0.0622865 | 74.3321862 | 1.0 | 0.0755184 | 1.0755184 |  |
| 36 | 1 | 0.0777 | 71.6084743 | 0.0 | 0.0942062 | 0.0942062 | ✓ |

### N23：expansion → action 9

path=[2, 37, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 592。

bucket=0，compatibility_richness_prior；到达 N25（新建）；closure=[]。

## iteration 26

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[25, 34, 9, 17, 12, 21, 36, 39, 29, 5, 27, 24, 11]

### N0：selection → action 2

path=[]；visits=25；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 611。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.3998967 | 0.3998967 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.5042809 | 0.3859693 | 0.8902503 |  |
| 2 | 15 | 0.3264006 | 70.5616193 | 1.0 | 0.1428002 | 1.1428002 | ✓ |

### N5：selection → action 0

path=[2]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [9, 0]，并列按 prior 抽样。trace 行 613。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.0 | 0.0990456 | 0.0990456 |  |
| 37 | 4 | 0.072067 | 73.5852681 | 1.0 | 0.0781521 | 1.0781521 |  |
| 0 | 3 | 0.085828 | 67.9218144 | 0.0811749 | 0.1163436 | 0.1975186 | ✓ |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.2255116 | 0.0883279 | 0.3138395 |  |

### N13：selection → action 6

path=[2, 0]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 23]，并列按 prior 抽样。trace 行 615。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.0446317 | 63.4114796 | 0.0 | 0.0541131 | 0.0541131 | ✓ |
| 23 | 1 | 0.0215647 | 65.5426112 | 1.0 | 0.0261458 | 1.0261458 |  |

### N16：expansion → action 8

path=[2, 0, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 617。

bucket=0，compatibility_richness_prior；到达 N26（新建）；closure=[]。

## iteration 27

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 30, 7, 32, 22, 10, 29, 26, 39, 5, 37, 9, 21, 0, 1, 14]

### N0：selection → action 2

path=[]；visits=26；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 636。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4078162 | 0.4078162 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.657843 | 0.393613 | 1.051456 |  |
| 2 | 16 | 0.3264006 | 70.008007 | 1.0 | 0.1370619 | 1.1370619 | ✓ |

### N5：expansion → action 28

path=[2]；visits=16；children=4；K=5。已有 4 条动作边 < K=5，且尚余 10 个代表动作。trace 行 638。

bucket=0，compatibility_richness_prior；到达 N14（复用）；closure=[]。

## iteration 28

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 35, 23, 20, 27, 4, 10, 3, 32, 38, 11, 29, 1, 15, 6]

### N0：selection → action 2

path=[]；visits=27；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 661。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4155848 | 0.4155848 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.5785878 | 0.4011111 | 0.9796988 |  |
| 2 | 17 | 0.3264006 | 70.2570379 | 1.0 | 0.1319132 | 1.1319132 | ✓ |

### N5：selection → action 28

path=[2]；visits=17；children=5；K=5。最低访问优先：child.visits < 5；最少 2 次；候选 [28]，并列按 prior 抽样。trace 行 663。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.1338739 | 0.1054421 | 0.239316 |  |
| 37 | 4 | 0.072067 | 73.5852681 | 0.9166566 | 0.0831992 | 0.9998558 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.0 | 0.0990858 | 0.0990858 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.3104005 | 0.0940322 | 0.4044327 |  |
| 28 | 2 | 0.0777326 | 74.2415322 | 1.0 | 0.1495665 | 1.1495665 | ✓ |

### N14：expansion → action 12

path=[24, 8]；visits=2；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 665。

bucket=0，compatibility_richness_prior；到达 N27（新建）；closure=[]。

## iteration 29

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[10, 39, 12, 7, 17, 8, 3, 36, 19, 5, 35, 24, 15]

### N0：selection → action 2

path=[]；visits=28；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 686。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4232109 | 0.4232109 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.6395725 | 0.4084715 | 1.048044 |  |
| 2 | 18 | 0.3264006 | 70.0599414 | 1.0 | 0.1272636 | 1.1272636 | ✓ |

### N5：selection → action 9

path=[2]；visits=18；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [9, 28]，并列按 prior 抽样。trace 行 688。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.073067 | 67.4214683 | 0.1460459 | 0.108499 | 0.2545449 | ✓ |
| 37 | 4 | 0.072067 | 73.5852681 | 1.0 | 0.0856113 | 1.0856113 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.0 | 0.1019584 | 0.1019584 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.3386224 | 0.0967584 | 0.4353808 |  |
| 28 | 3 | 0.0777326 | 70.475417 | 0.5691505 | 0.115427 | 0.6845775 |  |

### N6：selection → action 0

path=[2, 9]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [0, 15]，并列按 prior 抽样。trace 行 690。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0.144154 | 76.1604416 | 1.0 | 0.1747775 | 1.1747775 | ✓ |
| 15 | 1 | 0.0145787 | 64.673442 | 0.0 | 0.0176757 | 0.0176757 |  |

### N10：expansion → action 22

path=[2, 9, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 692。

bucket=0，compatibility_richness_prior；到达 N28（新建）；closure=[]。

## iteration 30

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[2, 39, 14, 25, 29, 31, 11, 13, 26, 21, 23, 28, 32, 20, 4]

### N0：selection → action 0

path=[]；visits=29；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 711。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4307019 | 0.4307019 |  |
| 0 | 5 | 0.3308308 | 69.3859649 | 0.9083137 | 0.4157017 | 1.3240153 | ✓ |
| 2 | 19 | 0.3264006 | 69.5066866 | 1.0 | 0.1230405 | 1.1230405 |  |

### N2：selection → action 6

path=[0]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [23, 6]，并列按 prior 抽样。trace 行 713。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.0699386 | 0.0699386 |  |
| 23 | 1 | 0.0316243 | 80.4644643 | 1.0 | 0.0494999 | 1.0494999 |  |
| 6 | 1 | 0.0693847 | 71.2554729 | 0.4608039 | 0.1086042 | 0.5694081 | ✓ |

### N15：expansion → action 34

path=[0, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 11 个代表动作。trace 行 715。

bucket=0，compatibility_richness_prior；到达 N29（新建）；closure=[]。

## iteration 31

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[24, 8, 12, 2, 39, 7, 21, 36, 25, 33, 30, 26, 38, 10, 5]

### N0：selection → action 0

path=[]；visits=30；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 736。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4380649 | 0.4380649 |  |
| 0 | 6 | 0.3308308 | 69.2810558 | 0.8286368 | 0.362407 | 1.1910438 | ✓ |
| 2 | 19 | 0.3264006 | 69.5066866 | 1.0 | 0.1251439 | 1.1251439 |  |

### N2：selection → action 23

path=[0]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [23]，并列按 prior 抽样。trace 行 738。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.0766139 | 0.0766139 |  |
| 23 | 1 | 0.0316243 | 80.4644643 | 1.0 | 0.0542245 | 1.0542245 | ✓ |
| 6 | 2 | 0.0693847 | 70.0059916 | 0.3876455 | 0.0793133 | 0.4669588 |  |

### N8：expansion → action 6

path=[0, 23]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 9 个代表动作。trace 行 740。

bucket=0，compatibility_richness_prior；到达 N30（新建）；closure=[]。

## iteration 32

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[27, 20, 38, 32, 19, 0, 13, 11, 22, 37, 12, 5, 10, 6]

### N0：selection → action 2

path=[]；visits=31；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 761。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4453061 | 0.4453061 |  |
| 0 | 7 | 0.3308308 | 68.9483703 | 0.5759671 | 0.3223479 | 0.898315 |  |
| 2 | 19 | 0.3264006 | 69.5066866 | 1.0 | 0.1272125 | 1.1272125 | ✓ |

### N5：selection → action 28

path=[2]；visits=19；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [28]，并列按 prior 抽样。trace 行 763。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 4 | 0.073067 | 65.4531264 | 0.0 | 0.0891777 | 0.0891777 |  |
| 37 | 4 | 0.072067 | 73.5852681 | 1.0 | 0.0879572 | 1.0879572 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1124169 | 0.1047523 | 0.2171692 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.4129724 | 0.0994098 | 0.5123822 |  |
| 28 | 3 | 0.0777326 | 70.475417 | 0.6175852 | 0.11859 | 0.7361752 | ✓ |

### N14：expansion → action 3

path=[24, 8]；visits=3；children=1；K=2。已有 1 条动作边 < K=2，且尚余 25 个代表动作。trace 行 765。

bucket=1，uniform_random；到达 N31（新建）；closure=[]。

## iteration 33

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[11, 23, 32, 13, 31, 2, 39, 34, 21, 9, 33, 26, 36, 10, 15]

### N0：selection → action 0

path=[]；visits=32；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 785。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4524315 | 0.4524315 |  |
| 0 | 7 | 0.3308308 | 68.9483703 | 0.8373955 | 0.3275058 | 1.1649013 | ✓ |
| 2 | 20 | 0.3264006 | 69.0956287 | 1.0 | 0.1230934 | 1.1230934 |  |

### N2：selection → action 6

path=[0]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [28, 23, 6]，并列按 prior 抽样。trace 行 787。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.0827524 | 0.0827524 |  |
| 23 | 2 | 0.0316243 | 73.7083606 | 1.0 | 0.0390461 | 1.0390461 |  |
| 6 | 2 | 0.0693847 | 70.0059916 | 0.641348 | 0.0856681 | 0.7270161 | ✓ |

### N15：expansion → action 12

path=[0, 6]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 10 个代表动作。trace 行 789。

bucket=1，uniform_random；到达 N32（新建）；closure=[18]。

## iteration 34

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[28, 3, 39, 7, 10, 31, 0, 33, 4, 22, 13, 37, 5, 34, 17]

### N0：selection → action 2

path=[]；visits=33；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 810。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4594463 | 0.4594463 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5877769 | 0.29563 | 0.8834069 |  |
| 2 | 20 | 0.3264006 | 69.0956287 | 1.0 | 0.1250019 | 1.1250019 | ✓ |

### N5：selection → action 9

path=[2]；visits=20；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [9, 37, 0, 6, 28]，并列按 prior 抽样。trace 行 812。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 4 | 0.073067 | 65.4531264 | 0.0 | 0.0914944 | 0.0914944 | ✓ |
| 37 | 4 | 0.072067 | 73.5852681 | 1.0 | 0.0902422 | 1.0902422 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1124169 | 0.1074736 | 0.2198905 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.4129724 | 0.1019923 | 0.5149647 |  |
| 28 | 4 | 0.0777326 | 67.412121 | 0.2408953 | 0.0973366 | 0.3382319 |  |

### N6：expansion → action 24

path=[2, 9]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 7 个代表动作。trace 行 814。

bucket=0，compatibility_richness_prior；到达 N33（新建）；closure=[]。

## iteration 35

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[36, 5, 8, 35, 34, 17, 33, 39, 10, 12, 3, 15, 4, 7]

### N0：selection → action 2

path=[]；visits=34；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 835。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4663557 | 0.4663557 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.7303088 | 0.3000758 | 1.0303846 |  |
| 2 | 21 | 0.3264006 | 68.9188812 | 1.0 | 0.1211144 | 1.1211144 | ✓ |

### N5：selection → action 37

path=[2]；visits=21；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [37, 0, 6, 28]，并列按 prior 抽样。trace 行 837。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0781282 | 0.0781282 |  |
| 37 | 4 | 0.072067 | 73.5852681 | 1.0 | 0.0924708 | 1.0924708 | ✓ |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1139248 | 0.1101277 | 0.2240525 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.4139697 | 0.104511 | 0.5184807 |  |
| 28 | 4 | 0.0777326 | 67.412121 | 0.2421849 | 0.0997403 | 0.3419253 |  |

### N7：expansion → action 38

path=[2, 37]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 17 个代表动作。trace 行 839。

bucket=0，compatibility_richness_prior；到达 N34（新建）；closure=[]。

## iteration 36

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[3, 35, 7, 10, 33, 32, 1, 4, 16, 27, 19, 26, 37, 11, 17]

### N0：selection → action 2

path=[]；visits=35；children=3；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 859。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4731641 | 0.4731641 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5202711 | 0.3044567 | 0.8247278 |  |
| 2 | 22 | 0.3264006 | 69.2131346 | 1.0 | 0.1175398 | 1.1175398 | ✓ |

### N5：selection → action 28

path=[2]；visits=22；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [0, 6, 28]，并列按 prior 抽样。trace 行 861。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0799668 | 0.0799668 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.0788724 | 1.0788724 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1090847 | 0.1127193 | 0.221804 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.3963822 | 0.1069704 | 0.5033526 |  |
| 28 | 4 | 0.0777326 | 67.412121 | 0.2318957 | 0.1020875 | 0.3339832 | ✓ |

### N14：expansion → action 9

path=[24, 8]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 24 个代表动作。trace 行 863。

bucket=0，compatibility_richness_prior；到达 N35（新建）；closure=[]。

## iteration 37

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[31, 21, 26, 1, 9, 18, 3, 0, 35, 33, 10, 34, 19, 28, 8]

### N0：selection → action 2

path=[]；visits=36；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 884。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.479876 | 0.479876 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5114209 | 0.3087754 | 0.8201963 |  |
| 2 | 23 | 0.3264006 | 69.2308399 | 1.0 | 0.1142402 | 1.1142402 | ✓ |

### N5：selection → action 6

path=[2]；visits=23；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [0, 6]，并列按 prior 抽样。trace 行 886。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.081764 | 0.081764 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.080645 | 1.080645 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1090847 | 0.1152526 | 0.2243373 |  |
| 6 | 4 | 0.0814506 | 68.8114767 | 0.3963822 | 0.1093745 | 0.5057568 | ✓ |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2967872 | 0.0869849 | 0.3837721 |  |

### N20：expansion → action 4

path=[2, 6]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 36 个代表动作。trace 行 888。

bucket=0，compatibility_richness_prior；到达 N36（新建）；closure=[]。

## iteration 38

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 22, 36, 32, 25, 13, 1, 21, 9, 33, 34, 38, 8, 20, 3]

### N0：selection → action 2

path=[]；visits=37；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 909。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4864953 | 0.4864953 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.6335421 | 0.3130346 | 0.9465767 |  |
| 2 | 24 | 0.3264006 | 69.0302092 | 1.0 | 0.1111834 | 1.1111834 | ✓ |

### N5：selection → action 0

path=[2]；visits=24；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [0]，并列按 prior 抽样。trace 行 911。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0835226 | 0.0835226 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.0823795 | 1.0823795 |  |
| 0 | 4 | 0.085828 | 66.3673162 | 0.1090847 | 0.1177315 | 0.2268161 | ✓ |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2930424 | 0.0931058 | 0.3861482 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2967872 | 0.0888558 | 0.385643 |  |

### N13：expansion → action 26

path=[2, 0]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 26 个代表动作。trace 行 913。

bucket=0，compatibility_richness_prior；到达 N37（新建）；closure=[]。

## iteration 39

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[38, 4, 37, 33, 1, 35, 7, 20, 10, 8, 11, 22, 39, 30, 0, 15]

### N0：selection → action 2

path=[]；visits=38；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 934。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4930258 | 0.4930258 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.7037016 | 0.3172366 | 1.0209382 |  |
| 2 | 25 | 0.3264006 | 68.9464403 | 1.0 | 0.1083421 | 1.1083421 | ✓ |

### N5：expansion → action 25

path=[2]；visits=25；children=5；K=6。已有 5 条动作边 < K=6，且尚余 9 个代表动作。trace 行 936。

bucket=1，uniform_random；到达 N9（复用）；closure=[]。

## iteration 40

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[38, 15, 37, 28, 20, 17, 27, 35, 7, 12, 30, 25, 0, 2, 10]

### N0：selection → action 2

path=[]；visits=39；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 959。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.4994708 | 0.4994708 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.7054916 | 0.3213837 | 1.0268753 |  |
| 2 | 26 | 0.3264006 | 68.944521 | 1.0 | 0.1056933 | 1.1056933 | ✓ |

### N5：selection → action 25

path=[2]；visits=26；children=6；K=6。最低访问优先：child.visits < 5；最少 3 次；候选 [25]，并列按 prior 抽样。trace 行 961。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.086933 | 0.086933 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.0857433 | 1.0857433 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1224535 | 0.1021157 | 0.2245692 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2930424 | 0.0969076 | 0.38995 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2967872 | 0.092484 | 0.3892712 |  |
| 25 | 3 | 0.0940229 | 68.8965383 | 0.4063808 | 0.1677985 | 0.5741793 | ✓ |

### N9：expansion → action 29

path=[24, 3]；visits=3；children=1；K=2。已有 1 条动作边 < K=2，且尚余 18 个代表动作。trace 行 963。

bucket=1，uniform_random；到达 N38（新建）；closure=[]。

## iteration 41

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 8, 30, 21, 35, 39, 36, 32, 12, 11, 6, 28, 33, 29, 4, 13]

### N0：selection → action 2

path=[]；visits=40；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 984。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5058338 | 0.5058338 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.577986 | 0.3254779 | 0.9034639 |  |
| 2 | 27 | 0.3264006 | 69.1109698 | 1.0 | 0.1032169 | 1.1032169 | ✓ |

### N5：selection → action 25

path=[2]；visits=27；children=6；K=6。最低访问优先：child.visits < 5；最少 4 次；候选 [25]，并列按 prior 抽样。trace 行 986。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0885891 | 0.0885891 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.0873767 | 1.0873767 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1224535 | 0.1040609 | 0.2265144 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2930424 | 0.0987536 | 0.391796 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2967872 | 0.0942458 | 0.391033 |  |
| 25 | 4 | 0.0940229 | 71.1675881 | 0.6733301 | 0.136796 | 0.8101261 | ✓ |

### N9：expansion → action 4

path=[24, 3]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 17 个代表动作。trace 行 988。

bucket=0，compatibility_richness_prior；到达 N37（复用）；closure=[]。

## iteration 42

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[36, 18, 12, 17, 14, 10, 23, 15, 29, 13, 31, 19, 16]

### N0：selection → action 2

path=[]；visits=41；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1011。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5121176 | 0.5121176 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5404915 | 0.3295213 | 0.8700128 |  |
| 2 | 28 | 0.3264006 | 69.174858 | 1.0 | 0.1008957 | 1.1008957 | ✓ |

### N5：selection → action 37

path=[2]；visits=28；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1013。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0902147 | 0.0902147 |  |
| 37 | 5 | 0.072067 | 73.9467056 | 1.0 | 0.08898 | 1.08898 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1224535 | 0.1059704 | 0.228424 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2930424 | 0.1005658 | 0.3936082 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2967872 | 0.0959752 | 0.3927624 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.6628393 | 0.1160885 | 0.7789279 |  |

### N7：selection → action 3

path=[2, 37]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [3, 38]，并列按 prior 抽样。trace 行 1015。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | 0.0622865 | 74.3321862 | 0.708526 | 0.0974939 | 0.8060198 | ✓ |
| 36 | 2 | 0.0777 | 71.7548438 | 0.0 | 0.0810798 | 0.0810798 |  |
| 38 | 1 | 0.0500079 | 75.3924556 | 1.0 | 0.0782747 | 1.0782747 |  |

### N17：expansion → action 38

path=[2, 37, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 18 个代表动作。trace 行 1017。

bucket=0，compatibility_richness_prior；到达 N39（新建）；closure=[]。

## iteration 43

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[14, 33, 9, 0, 36, 31, 39, 6, 34, 17, 5, 10, 28, 19, 1]

### N0：selection → action 2

path=[]；visits=42；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1036。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5183254 | 0.5183254 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.4889761 | 0.3335156 | 0.8224917 |  |
| 2 | 29 | 0.3264006 | 69.278616 | 1.0 | 0.0987148 | 1.0987148 | ✓ |

### N5：selection → action 37

path=[2]；visits=29；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1038。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0918115 | 0.0918115 |  |
| 37 | 6 | 0.072067 | 73.6528946 | 1.0 | 0.0776186 | 1.0776186 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1268339 | 0.1078462 | 0.23468 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3035249 | 0.1023458 | 0.4058707 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3074037 | 0.097674 | 0.4050776 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.6865499 | 0.1181433 | 0.8046933 |  |

### N7：selection → action 38

path=[2, 37]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [38]，并列按 prior 抽样。trace 行 1040。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 0.0622865 | 73.2580128 | 0.4132296 | 0.0711995 | 0.4844291 |  |
| 36 | 2 | 0.0777 | 71.7548438 | 0.0 | 0.0888185 | 0.0888185 |  |
| 38 | 1 | 0.0500079 | 75.3924556 | 1.0 | 0.0857457 | 1.0857457 | ✓ |

### N34：expansion → action 3

path=[2, 37, 38]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 13 个代表动作。trace 行 1042。

bucket=0，compatibility_richness_prior；到达 N39（复用）；closure=[]。

## iteration 44

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[6, 39, 15, 3, 13, 19, 24, 8, 35, 16, 22, 29, 10]

### N0：selection → action 2

path=[]；visits=43；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1064。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5244596 | 0.5244596 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.4700495 | 0.3374627 | 0.8075122 |  |
| 2 | 30 | 0.3264006 | 69.322449 | 1.0 | 0.096661 | 1.096661 | ✓ |

### N5：selection → action 37

path=[2]；visits=30；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1066。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0933811 | 0.0933811 |  |
| 37 | 7 | 0.072067 | 73.2158533 | 1.0 | 0.0690773 | 1.0690773 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1339619 | 0.1096898 | 0.2436517 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.320583 | 0.1040954 | 0.4246784 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3246797 | 0.0993438 | 0.4240234 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.7251339 | 0.120163 | 0.8452969 |  |

### N7：selection → action 36

path=[2, 37]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [3, 36, 38]，并列按 prior 抽样。trace 行 1068。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 0.0622865 | 73.2580128 | 1.0 | 0.0769042 | 1.0769042 |  |
| 36 | 2 | 0.0777 | 71.7548438 | 0.0 | 0.0959349 | 0.0959349 | ✓ |
| 38 | 2 | 0.0500079 | 72.9930306 | 0.8237176 | 0.0617439 | 0.8854615 |  |

### N23：expansion → action 0

path=[2, 37, 36]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 1070。

bucket=1，uniform_random；到达 N40（新建）；closure=[]。

## iteration 45

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[22, 36, 5, 20, 27, 25, 1, 28, 10, 7, 8, 9, 30]

### N0：selection → action 2

path=[]；visits=44；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1089。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5305229 | 0.5305229 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5159955 | 0.3413641 | 0.8573596 |  |
| 2 | 31 | 0.3264006 | 69.2216123 | 1.0 | 0.094723 | 1.094723 | ✓ |

### N5：selection → action 37

path=[2]；visits=31；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1091。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0949247 | 0.0949247 |  |
| 37 | 8 | 0.072067 | 72.3384355 | 1.0 | 0.062417 | 1.062417 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1509988 | 0.111503 | 0.2625018 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.361354 | 0.1058161 | 0.4671701 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3659717 | 0.1009859 | 0.4669576 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.8173547 | 0.1221493 | 0.9395041 |  |

### N7：selection → action 3

path=[2, 37]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [3, 38]，并列按 prior 抽样。trace 行 1093。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 0.0622865 | 73.2580128 | 1.0 | 0.0822141 | 1.0822141 | ✓ |
| 36 | 3 | 0.0777 | 69.9020661 | 0.0 | 0.0769191 | 0.0769191 |  |
| 38 | 2 | 0.0500079 | 72.9930306 | 0.921041 | 0.066007 | 0.987048 |  |

### N17：expansion → action 21

path=[2, 37, 3]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 17 个代表动作。trace 行 1095。

bucket=1，uniform_random；到达 N41（新建）；closure=[]。

## iteration 46

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 26, 22, 14, 33, 5, 19, 0, 29, 12, 16, 35, 10, 39, 11]

### N0：selection → action 2

path=[]；visits=45；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1114。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5365177 | 0.5365177 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5601329 | 0.3452214 | 0.9053544 |  |
| 2 | 32 | 0.3264006 | 69.1403236 | 1.0 | 0.0928905 | 1.0928905 | ✓ |

### N5：selection → action 37

path=[2]；visits=32；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1116。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0964436 | 0.0964436 |  |
| 37 | 9 | 0.072067 | 71.7030953 | 1.0 | 0.0570742 | 1.0570742 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1663147 | 0.1132872 | 0.2796019 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3980062 | 0.1075093 | 0.5055155 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.4030923 | 0.1026018 | 0.5056941 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.9002593 | 0.1241038 | 1.0243632 |  |

### N7：expansion → action 28

path=[2, 37]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 16 个代表动作。trace 行 1118。

bucket=1，uniform_random；到达 N42（新建）；closure=[]。

## iteration 47

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[4, 9, 14, 27, 11, 7, 23, 33, 3, 38, 1, 29, 8, 18]

### N0：selection → action 2

path=[]；visits=46；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1139。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5424463 | 0.5424463 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.5457592 | 0.3490362 | 0.8947954 |  |
| 2 | 33 | 0.3264006 | 69.1653522 | 1.0 | 0.0911547 | 1.0911547 | ✓ |

### N5：selection → action 37

path=[2]；visits=33；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1141。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0979389 | 0.0979389 |  |
| 37 | 10 | 0.072067 | 71.5294125 | 1.0 | 0.0526901 | 1.0526901 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1710578 | 0.1150436 | 0.2861014 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.4093569 | 0.1091762 | 0.5185331 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.414588 | 0.1041926 | 0.5187806 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 0.9259336 | 0.1260281 | 1.0519616 |  |

### N7：selection → action 28

path=[2, 37]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [28]，并列按 prior 抽样。trace 行 1143。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 0.3699171 | 0.0689386 | 0.4388556 |  |
| 36 | 3 | 0.0777 | 69.9020661 | 0.0 | 0.0859981 | 0.0859981 |  |
| 38 | 2 | 0.0500079 | 72.9930306 | 1.0 | 0.0737981 | 1.0737981 |  |
| 28 | 1 | 0.0589352 | 69.9662677 | 0.0207707 | 0.1304586 | 0.1512293 | ✓ |

### N42：expansion → action 16

path=[2, 37, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1145。

bucket=0，compatibility_richness_prior；到达 N43（新建）；closure=[]。

## iteration 48

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[21, 6, 8, 15, 3, 18, 24, 4, 39, 10, 7, 30, 5, 22]

### N0：selection → action 0

path=[]；visits=47；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1165。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5483107 | 0.5483107 |  |
| 0 | 8 | 0.3308308 | 68.7223097 | 0.798726 | 0.3528096 | 1.1515356 | ✓ |
| 2 | 34 | 0.3264006 | 68.8564472 | 1.0 | 0.0895076 | 1.0895076 |  |

### N2：selection → action 23

path=[0]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [28, 23]，并列按 prior 抽样。trace 行 1167。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.0884661 | 0.0884661 |  |
| 23 | 2 | 0.0316243 | 73.7083606 | 1.0 | 0.041742 | 1.041742 | ✓ |
| 6 | 3 | 0.0693847 | 69.0506231 | 0.5488005 | 0.0686873 | 0.6174878 |  |

### N8：expansion → action 1

path=[0, 23]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 8 个代表动作。trace 行 1169。

bucket=1，uniform_random；到达 N44（新建）；closure=[]。

## iteration 49

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[4, 32, 38, 37, 15, 27, 23, 30, 20, 13, 31, 16, 1]

### N0：selection → action 2

path=[]；visits=48；children=3；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1189。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5541131 | 0.5541131 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.6471569 | 0.3208889 | 0.9680458 |  |
| 2 | 34 | 0.3264006 | 68.8564472 | 1.0 | 0.0904548 | 1.0904548 | ✓ |

### N5：selection → action 25

path=[2]；visits=34；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1191。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.0994117 | 0.0994117 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.8725605 | 0.0490256 | 0.9215861 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1847409 | 0.1167737 | 0.3015146 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.4421018 | 0.1108181 | 0.5529198 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.4477513 | 0.1057595 | 0.5535108 |  |
| 25 | 5 | 0.0940229 | 71.0783388 | 1.0 | 0.1279233 | 1.1279233 | ✓ |

### N9：selection → action 29

path=[24, 3]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [9, 29]，并列按 prior 抽样。trace 行 1193。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 1 | 0.0632242 | 69.0863346 | 0.0 | 0.0989615 | 0.0989615 |  |
| 29 | 1 | 0.0579545 | 73.4386379 | 1.0 | 0.0907132 | 1.0907132 | ✓ |
| 4 | 2 | 0.058363 | 70.89984 | 0.4166772 | 0.0609017 | 0.4775788 |  |

### N38：expansion → action 18

path=[24, 3, 29]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1195。

bucket=0，compatibility_richness_prior；到达 N45（新建）；closure=[]。

## iteration 50

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[11, 4, 8, 27, 13, 6, 34, 29, 0, 37, 12, 36, 28, 15]

### N0：selection → action 2

path=[]；visits=49；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1214。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5598554 | 0.5598554 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.6469977 | 0.3242142 | 0.9712119 |  |
| 2 | 35 | 0.3264006 | 68.8566112 | 1.0 | 0.0888535 | 1.0888535 | ✓ |

### N5：selection → action 25

path=[2]；visits=35；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1216。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1008631 | 0.1008631 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.9676304 | 0.0497413 | 1.0173718 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2048694 | 0.1184785 | 0.3233479 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.490271 | 0.1124359 | 0.6027069 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.4965361 | 0.1073035 | 0.6038397 |  |
| 25 | 6 | 0.0940229 | 70.5243006 | 1.0 | 0.1112493 | 1.1112493 | ✓ |

### N9：selection → action 9

path=[24, 3]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [9]，并列按 prior 抽样。trace 行 1218。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 1 | 0.0632242 | 69.0863346 | 0.0 | 0.1084069 | 0.1084069 | ✓ |
| 29 | 2 | 0.0579545 | 71.1504119 | 1.0 | 0.0662475 | 1.0662475 |  |
| 4 | 2 | 0.058363 | 70.89984 | 0.8786034 | 0.0667144 | 0.9453179 |  |

### N11：expansion → action 20

path=[24, 3, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 1220。

bucket=0，compatibility_richness_prior；到达 N46（新建）；closure=[]。

## iteration 51

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class40；rollout：[21, 25, 23, 18, 14, 11, 6, 28, 26, 39, 31, 29, 15, 0, 17]

### N0：selection → action 2

path=[]；visits=50；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1240。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5655393 | 0.5655393 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.7506264 | 0.3275058 | 1.0781322 |  |
| 2 | 36 | 0.3264006 | 68.7645821 | 1.0 | 0.0873298 | 1.0873298 | ✓ |

### N5：expansion → action 36

path=[2]；visits=36；children=6；K=7。已有 6 条动作边 < K=7，且尚余 8 个代表动作。trace 行 1242。

bucket=0，compatibility_richness_prior；到达 N47（新建）；closure=[]。

## iteration 52

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 12, 21, 25, 17, 4, 18, 15, 6, 28, 10, 35, 16, 24, 7]

### N0：selection → action 2

path=[]；visits=51；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1263。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5711667 | 0.5711667 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.4979408 | 0.3307647 | 0.8287054 |  |
| 2 | 37 | 0.3264006 | 69.0561576 | 1.0 | 0.0858777 | 1.0858777 | ✓ |

### N5：selection → action 36

path=[2]；visits=37；children=7；K=7。最低访问优先：child.visits < 5；最少 1 次；候选 [36]，并列按 prior 抽样。trace 行 1265。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1037049 | 0.1037049 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.3486295 | 0.0511428 | 0.3997723 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.0738128 | 0.1218166 | 0.1956294 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.1766407 | 0.1156038 | 0.2922445 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.178898 | 0.1103267 | 0.2892247 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.2897113 | 0.1000858 | 0.389797 |  |
| 36 | 1 | 0.1156766 | 79.5528764 | 1.0 | 0.4925431 | 1.4925431 | ✓ |

### N47：expansion → action 31

path=[2, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 1267。

bucket=0，compatibility_richness_prior；到达 N48（新建）；closure=[]。

## iteration 53

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 19, 12, 0, 33, 22, 39, 37, 35, 5, 25, 24, 14, 27, 10]

### N0：selection → action 2

path=[]；visits=52；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1288。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5767392 | 0.5767392 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.4516941 | 0.3339917 | 0.7856858 |  |
| 2 | 38 | 0.3264006 | 69.1448386 | 1.0 | 0.0844921 | 1.0844921 | ✓ |

### N5：selection → action 36

path=[2]；visits=38；children=7；K=7。最低访问优先：child.visits < 5；最少 2 次；候选 [36]，并列按 prior 抽样。trace 行 1290。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1050969 | 0.1050969 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.4663824 | 0.0518293 | 0.5182117 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.0987438 | 0.1234518 | 0.2221956 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2363028 | 0.1171556 | 0.3534584 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2393225 | 0.1118077 | 0.3511302 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.387564 | 0.1014293 | 0.4889933 |  |
| 36 | 2 | 0.1156766 | 75.9894564 | 1.0 | 0.3327698 | 1.3327698 | ✓ |

### N47：expansion → action 8

path=[2, 36]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 18 个代表动作。trace 行 1292。

bucket=1，uniform_random；到达 N49（新建）；closure=[]。

## iteration 54

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[34, 23, 18, 32, 9, 12, 17, 5, 0, 26, 33, 19, 22]

### N0：selection → action 2

path=[]；visits=53；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1313。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5822584 | 0.5822584 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.5187445 | 0.3371879 | 0.8559323 |  |
| 2 | 39 | 0.3264006 | 69.0214215 | 1.0 | 0.0831681 | 1.0831681 | ✓ |

### N5：selection → action 36

path=[2]；visits=39；children=7；K=7。最低访问优先：child.visits < 5；最少 3 次；候选 [36]，并列按 prior 抽样。trace 行 1315。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1064708 | 0.1064708 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.7383344 | 0.0525068 | 0.7908413 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1563222 | 0.1250656 | 0.2813878 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3740932 | 0.1186871 | 0.4927803 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3788737 | 0.1132693 | 0.492143 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.6135562 | 0.1027552 | 0.7163114 |  |
| 36 | 3 | 0.1156766 | 72.1034943 | 1.0 | 0.25284 | 1.25284 | ✓ |

### N47：selection → action 8

path=[2, 36]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [31, 8]，并列按 prior 抽样。trace 行 1317。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 1 | 0.0602182 | 72.4260365 | 1.0 | 0.0730107 | 1.0730107 |  |
| 8 | 1 | 0.059354 | 64.3315699 | 0.0 | 0.0719629 | 0.0719629 | ✓ |

### N49：expansion → action 21

path=[2, 36, 8]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 1319。

bucket=0，compatibility_richness_prior；到达 N50（新建）；closure=[]。

## iteration 55

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 14, 34, 22, 31, 10, 16, 29, 30, 5, 35, 18, 26, 7, 9, 3]

### N0：selection → action 2

path=[]；visits=54；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1338。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5877257 | 0.5877257 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.4802456 | 0.340354 | 0.8205996 |  |
| 2 | 40 | 0.3264006 | 69.0880719 | 1.0 | 0.0819015 | 1.0819015 | ✓ |

### N5：selection → action 36

path=[2]；visits=40；children=7；K=7。最低访问优先：child.visits < 5；最少 4 次；候选 [36]，并列按 prior 抽样。trace 行 1340。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1078272 | 0.1078272 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.750041 | 0.0531757 | 0.8032167 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1588007 | 0.1266589 | 0.2854596 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3800246 | 0.1201991 | 0.5002236 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3848809 | 0.1147123 | 0.4995932 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.6232844 | 0.1040642 | 0.7273486 |  |
| 36 | 4 | 0.1156766 | 71.9994803 | 1.0 | 0.2048488 | 1.2048488 | ✓ |

### N47：expansion → action 37

path=[2, 36]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 17 个代表动作。trace 行 1342。

bucket=0，compatibility_richness_prior；到达 N23（复用）；closure=[]。

## iteration 56

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 29, 5, 10, 19, 11, 39, 27, 38, 28, 0, 15, 1, 12]

### N0：selection → action 2

path=[]；visits=55；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1365。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5931427 | 0.5931427 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.4623382 | 0.343491 | 0.8058292 |  |
| 2 | 41 | 0.3264006 | 69.122856 | 1.0 | 0.0806884 | 1.0806884 | ✓ |

### N5：selection → action 36

path=[2]；visits=41；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1367。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1091667 | 0.1091667 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 0.7856144 | 0.0538363 | 0.8394507 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1663324 | 0.1282324 | 0.2945648 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3980486 | 0.1216923 | 0.5197409 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.4031353 | 0.1161373 | 0.5192726 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.6528459 | 0.105357 | 0.7582029 |  |
| 36 | 5 | 0.1156766 | 71.702428 | 1.0 | 0.172828 | 1.172828 | ✓ |

### N47：selection → action 31

path=[2, 36]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [31]，并列按 prior 抽样。trace 行 1369。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 1 | 0.0602182 | 72.4260365 | 1.0 | 0.0942564 | 1.0942564 | ✓ |
| 8 | 2 | 0.059354 | 68.0095041 | 0.0 | 0.0619358 | 0.0619358 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 0.5671224 | 0.0306333 | 0.5977557 |  |

### N48：expansion → action 6

path=[2, 36, 31]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 1371。

bucket=0，compatibility_richness_prior；到达 N51（新建）；closure=[]。

## iteration 57

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[26, 19, 30, 39, 5, 0, 11, 25, 35, 32, 29, 17, 15]

### N0：selection → action 2

path=[]；visits=56；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1391。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.5985106 | 0.5985106 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.5506739 | 0.3465996 | 0.8972735 |  |
| 2 | 42 | 0.3264006 | 68.9732138 | 1.0 | 0.0795252 | 1.0795252 | ✓ |

### N5：selection → action 36

path=[2]；visits=42；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1393。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.11049 | 0.11049 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 1.0 | 0.0544889 | 1.0544889 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2117227 | 0.1297867 | 0.3415095 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.5066718 | 0.1231674 | 0.6298391 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.5131465 | 0.1175451 | 0.6306916 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.8310004 | 0.1066341 | 0.9376345 |  |
| 36 | 6 | 0.1156766 | 70.2250043 | 0.972625 | 0.149934 | 1.1225589 | ✓ |

### N47：selection → action 8

path=[2, 36]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [31, 8]，并列按 prior 抽样。trace 行 1395。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 2 | 0.0602182 | 67.6319613 | 0.0 | 0.0688352 | 0.0688352 |  |
| 8 | 2 | 0.059354 | 68.0095041 | 0.1309886 | 0.0678473 | 0.1988359 | ✓ |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0335571 | 1.0335571 |  |

### N49：expansion → action 22

path=[2, 36, 8]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 1397。

bucket=1，uniform_random；到达 N52（新建）；closure=[]。

## iteration 58

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 33, 13, 39, 11, 1, 38, 3, 23, 21, 0, 27, 16, 10]

### N0：selection → action 2

path=[]；visits=57；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1416。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6038308 | 0.6038308 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.5200671 | 0.3496805 | 0.8697477 |  |
| 2 | 43 | 0.3264006 | 69.0193069 | 1.0 | 0.0784086 | 1.0784086 | ✓ |

### N5：selection → action 36

path=[2]；visits=43；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1418。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1117976 | 0.1117976 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 1.0 | 0.0551338 | 1.0551338 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2117227 | 0.1313227 | 0.3430455 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.5066718 | 0.124625 | 0.6312968 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.5131465 | 0.1189362 | 0.6320827 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.8310004 | 0.1078961 | 0.9388965 |  |
| 36 | 7 | 0.1156766 | 70.3293205 | 0.9938257 | 0.1327448 | 1.1265705 | ✓ |

### N47：selection → action 31

path=[2, 36]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [31]，并列按 prior 抽样。trace 行 1420。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 2 | 0.0602182 | 67.6319613 | 0.0 | 0.0743505 | 0.0743505 | ✓ |
| 8 | 3 | 0.059354 | 68.9914085 | 0.4716606 | 0.0549626 | 0.5266232 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0362458 | 1.0362458 |  |

### N48：expansion → action 9

path=[2, 36, 31]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 1422。

bucket=1，uniform_random；到达 N53（新建）；closure=[]。

## iteration 59

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[18, 7, 25, 30, 15, 4, 22, 10, 36, 0, 1, 26, 39, 17]

### N0：selection → action 2

path=[]；visits=58；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1442。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6091045 | 0.6091045 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.60758 | 0.3527346 | 0.9603146 |  |
| 2 | 44 | 0.3264006 | 68.8998583 | 1.0 | 0.0773358 | 1.0773358 | ✓ |

### N5：selection → action 37

path=[2]；visits=44；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1444。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1130901 | 0.1130901 |  |
| 37 | 11 | 0.072067 | 70.3597008 | 1.0 | 0.0557712 | 1.0557712 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2117227 | 0.132841 | 0.3445637 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.5066718 | 0.1260658 | 0.6327376 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.5131465 | 0.1203113 | 0.6334577 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 0.8310004 | 0.1091435 | 0.9401439 |  |
| 36 | 8 | 0.1156766 | 69.5086012 | 0.8270268 | 0.1193596 | 0.9463864 |  |

### N7：selection → action 28

path=[2, 37]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [38, 28]，并列按 prior 抽样。trace 行 1446。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 0.7755902 | 0.0723034 | 0.8478936 |  |
| 36 | 4 | 0.0777 | 69.9020661 | 0.6438409 | 0.0721565 | 0.7159973 |  |
| 38 | 2 | 0.0500079 | 72.9930306 | 1.0 | 0.0774001 | 1.0774001 |  |
| 28 | 2 | 0.0589352 | 64.3144254 | 0.0 | 0.0912174 | 0.0912174 | ✓ |

### N42：expansion → action 29

path=[2, 37, 28]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 1448。

bucket=1，uniform_random；到达 N54（新建）；closure=[]。

## iteration 60

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 32, 11, 8, 30, 12, 29, 20, 38, 27, 25, 36, 31, 33, 3, 16]

### N0：selection → action 0

path=[]；visits=59；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1468。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.614333 | 0.614333 |  |
| 0 | 9 | 0.3308308 | 68.6212977 | 0.9043855 | 0.3557624 | 1.2601479 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 1.0 | 0.076304 | 1.076304 |  |

### N2：expansion → action 19

path=[0]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 14 个代表动作。trace 行 1470。

bucket=1，uniform_random；到达 N55（新建）；closure=[]。

## iteration 61

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 36, 5, 8, 9, 21, 22, 34, 25, 39, 6, 35, 28, 20, 1]

### N0：selection → action 0

path=[]；visits=60；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1492。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6195173 | 0.6195173 |  |
| 0 | 10 | 0.3308308 | 69.211814 | 1.0 | 0.3261497 | 1.3261497 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 0.4667118 | 0.0769479 | 0.5436597 |  |

### N2：selection → action 19

path=[0]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [19]，并列按 prior 抽样。trace 行 1494。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.0989081 | 0.0989081 |  |
| 23 | 3 | 0.0316243 | 71.7433075 | 0.7501907 | 0.0350017 | 0.7851925 |  |
| 6 | 3 | 0.0693847 | 69.0506231 | 0.5085017 | 0.0767948 | 0.5852964 |  |
| 19 | 1 | 0.0798098 | 74.5264604 | 1.0 | 0.1766666 | 1.1766666 | ✓ |

### N55：expansion → action 10

path=[0, 19]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 16 个代表动作。trace 行 1496。

bucket=0，compatibility_richness_prior；到达 N56（新建）；closure=[]。

## iteration 62

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class29；rollout：[8, 32, 10, 5, 3, 12, 22, 36, 33, 2, 11, 15, 7, 39]

### N0：selection → action 0

path=[]；visits=61；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1517。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6246586 | 0.6246586 |  |
| 0 | 11 | 0.3308308 | 68.926993 | 1.0 | 0.3014517 | 1.3014517 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 0.6470802 | 0.0775865 | 0.7246666 |  |

### N2：selection → action 19

path=[0]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [28, 19]，并列按 prior 抽样。trace 行 1519。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.1037357 | 0.1037357 |  |
| 23 | 3 | 0.0316243 | 71.7433075 | 1.0 | 0.0367101 | 1.0367101 |  |
| 6 | 3 | 0.0693847 | 69.0506231 | 0.6778299 | 0.080543 | 0.7583729 |  |
| 19 | 2 | 0.0798098 | 70.3026219 | 0.8276271 | 0.1235263 | 0.9511534 | ✓ |

### N55：expansion → action 9

path=[0, 19]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 15 个代表动作。trace 行 1521。

bucket=1，uniform_random；到达 N57（新建）；closure=[]。

## iteration 63

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 5, 14, 25, 31, 6, 22, 34, 19, 16, 39, 29, 35, 36, 7]

### N0：selection → action 0

path=[]；visits=62；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1541。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6297579 | 0.6297579 |  |
| 0 | 12 | 0.3308308 | 69.472921 | 1.0 | 0.2805347 | 1.2805347 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 0.3717237 | 0.0782198 | 0.4499436 |  |

### N2：selection → action 28

path=[0]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [28]，并列按 prior 抽样。trace 行 1543。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.0670232 | 63.3853498 | 0.0 | 0.1083484 | 0.1083484 | ✓ |
| 23 | 3 | 0.0316243 | 71.7433075 | 0.967083 | 0.0383425 | 1.0054255 |  |
| 6 | 3 | 0.0693847 | 69.0506231 | 0.6555177 | 0.0841245 | 0.7396422 |  |
| 19 | 3 | 0.0798098 | 72.0277909 | 1.0 | 0.0967643 | 1.0967643 |  |

### N3：expansion → action 9

path=[0, 28]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1545。

bucket=1，uniform_random；到达 N58（新建）；closure=[]。

## iteration 64

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 24, 20, 16, 11, 23, 38, 9, 14, 22, 31, 26, 1, 13]

### N0：selection → action 0

path=[]；visits=63；children=3；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1566。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6348163 | 0.6348163 |  |
| 0 | 13 | 0.3308308 | 69.3637697 | 1.0 | 0.2625888 | 1.2625888 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 0.4062912 | 0.0788481 | 0.4851393 |  |

### N2：selection → action 6

path=[0]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 23, 6, 19]，并列按 prior 抽样。trace 行 1568。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0670232 | 64.9415514 | 0.0 | 0.0845794 | 0.0845794 |  |
| 23 | 3 | 0.0316243 | 71.7433075 | 0.9598541 | 0.0399081 | 0.9997622 |  |
| 6 | 3 | 0.0693847 | 69.0506231 | 0.5798663 | 0.0875595 | 0.6674258 | ✓ |
| 19 | 3 | 0.0798098 | 72.0277909 | 1.0 | 0.1007155 | 1.1007155 |  |

### N15：selection → action 34

path=[0, 6]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [34, 12]，并列按 prior 抽样。trace 行 1570。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 1 | 0.1116194 | 68.7565103 | 1.0 | 0.1353313 | 1.1353313 | ✓ |
| 12 | 1 | 0.032629 | 67.1398861 | 0.0 | 0.0395606 | 0.0395606 |  |

### N29：expansion → action 25

path=[0, 6, 34]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1572。

bucket=0，compatibility_richness_prior；到达 N59（新建）；closure=[]。

## iteration 65

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 2, 31, 38, 25, 39, 22, 7, 27, 8, 34, 29, 36, 14]

### N0：selection → action 0

path=[]；visits=64；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1592。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6398347 | 0.6398347 |  |
| 0 | 14 | 0.3308308 | 69.0902159 | 1.0 | 0.2470204 | 1.2470204 | ✓ |
| 2 | 45 | 0.3264006 | 68.6668953 | 0.529754 | 0.0794714 | 0.6092254 |  |

### N2：selection → action 19

path=[0]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 23, 19]，并列按 prior 抽样。trace 行 1594。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0670232 | 64.9415514 | 0.0 | 0.0877722 | 0.0877722 |  |
| 23 | 3 | 0.0316243 | 71.7433075 | 0.9598541 | 0.0414146 | 1.0012687 |  |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.4558017 | 0.0726918 | 0.5284935 |  |
| 19 | 3 | 0.0798098 | 72.0277909 | 1.0 | 0.1045174 | 1.1045174 | ✓ |

### N55：selection → action 10

path=[0, 19]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [10, 9]，并列按 prior 抽样。trace 行 1596。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 10 | 1 | 0.0656648 | 66.0787833 | 0.0 | 0.0796143 | 0.0796143 | ✓ |
| 9 | 1 | 0.0681279 | 75.4781289 | 1.0 | 0.0826007 | 1.0826007 |  |

### N56：expansion → action 4

path=[0, 19, 10]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1598。

bucket=0，compatibility_richness_prior；到达 N60（新建）；closure=[]。

## iteration 66

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[22, 14, 37, 19, 5, 11, 6, 27, 25, 13, 0, 17]

### N0：selection → action 2

path=[]；visits=65；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1618。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.644814 | 0.644814 |  |
| 0 | 15 | 0.3308308 | 68.2681672 | 0.1638997 | 0.2333838 | 0.3972835 |  |
| 2 | 45 | 0.3264006 | 68.6668953 | 1.0 | 0.0800899 | 1.0800899 | ✓ |

### N5：selection → action 36

path=[2]；visits=45；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1620。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.114368 | 0.114368 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.9599603 | 0.0520628 | 1.0120232 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2547805 | 0.134342 | 0.3891226 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.609713 | 0.1274904 | 0.7372033 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6175045 | 0.1216708 | 0.7391752 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 1.0 | 0.1103768 | 1.1103768 |  |
| 36 | 8 | 0.1156766 | 69.5086012 | 0.9952183 | 0.1207083 | 1.1159266 | ✓ |

### N47：selection → action 31

path=[2, 36]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 3 次；候选 [31, 8]，并列按 prior 抽样。trace 行 1622。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 3 | 0.0602182 | 66.3424964 | 0.0 | 0.059613 | 0.059613 | ✓ |
| 8 | 3 | 0.059354 | 68.9914085 | 0.6349685 | 0.0587575 | 0.693726 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0387484 | 1.0387484 |  |

### N48：selection → action 9

path=[2, 36, 31]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 9]，并列按 prior 抽样。trace 行 1624。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.0430011 | 62.8378862 | 0.0 | 0.0521361 | 0.0521361 |  |
| 9 | 1 | 0.049907 | 63.7635665 | 1.0 | 0.060509 | 1.060509 | ✓ |

### N53：expansion → action 16

path=[2, 36, 31, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 1626。

bucket=0，compatibility_richness_prior；到达 N61（新建）；closure=[]。

## iteration 67

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[16, 38, 0, 4, 13, 1, 34, 10, 9, 31, 18, 19, 14]

### N0：selection → action 2

path=[]；visits=66；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1644。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6497552 | 0.6497552 |  |
| 0 | 15 | 0.3308308 | 68.2681672 | 0.1668983 | 0.2351722 | 0.4020706 |  |
| 2 | 46 | 0.3264006 | 68.6583272 | 1.0 | 0.0789865 | 1.0789865 | ✓ |

### N5：selection → action 25

path=[2]；visits=46；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1646。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1156318 | 0.1156318 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.9599603 | 0.0526381 | 1.0125985 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2547805 | 0.1358265 | 0.3906071 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.609713 | 0.1288991 | 0.7386121 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6175045 | 0.1230152 | 0.7405197 |  |
| 25 | 7 | 0.0940229 | 69.528153 | 1.0 | 0.1115965 | 1.1115965 | ✓ |
| 36 | 9 | 0.1156766 | 69.3712858 | 0.9616355 | 0.1098379 | 1.0714734 |  |

### N9：selection → action 29

path=[24, 3]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 29, 4]，并列按 prior 抽样。trace 行 1648。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0632242 | 67.3149488 | 0.0 | 0.0780619 | 0.0780619 |  |
| 29 | 2 | 0.0579545 | 71.1504119 | 1.0 | 0.0715555 | 1.0715555 | ✓ |
| 4 | 2 | 0.058363 | 70.89984 | 0.9346697 | 0.0720598 | 1.0067295 |  |

### N38：expansion → action 33

path=[24, 3, 29]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 1650。

bucket=1，uniform_random；到达 N62（新建）；closure=[]。

## iteration 68

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[31, 11, 26, 16, 28, 5, 30, 14, 15, 29, 35, 34, 0, 18, 7]

### N0：selection → action 2

path=[]；visits=67；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1669。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6546591 | 0.6546591 |  |
| 0 | 15 | 0.3308308 | 68.2681672 | 0.7064911 | 0.2369471 | 0.9434383 |  |
| 2 | 47 | 0.3264006 | 68.3006394 | 1.0 | 0.0779247 | 1.0779247 | ✓ |

### N5：selection → action 36

path=[2]；visits=47；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1671。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1168819 | 0.1168819 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.998258 | 0.0532072 | 1.0514652 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.264945 | 0.137295 | 0.40224 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.6340375 | 0.1302927 | 0.7643302 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6421398 | 0.1243452 | 0.766485 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2904393 | 0.1002693 | 0.3907086 |  |
| 36 | 9 | 0.1156766 | 69.3712858 | 1.0 | 0.1110254 | 1.1110254 | ✓ |

### N47：expansion → action 39

path=[2, 36]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 16 个代表动作。trace 行 1673。

bucket=1，uniform_random；到达 N63（新建）；closure=[]。

## iteration 69

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 13, 11, 21, 23, 25, 31, 14, 18, 10, 30, 32, 24, 4]

### N0：selection → action 2

path=[]；visits=68；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1694。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6595265 | 0.6595265 |  |
| 0 | 15 | 0.3308308 | 68.2681672 | 0.7766352 | 0.2387088 | 1.015344 |  |
| 2 | 48 | 0.3264006 | 68.2906471 | 1.0 | 0.0769019 | 1.0769019 | ✓ |

### N5：selection → action 36

path=[2]；visits=48；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1696。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1181188 | 0.1181188 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 1.0 | 0.0537702 | 1.0537702 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2654074 | 0.1387479 | 0.4041552 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.6351439 | 0.1316715 | 0.7668154 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6432604 | 0.125661 | 0.7689214 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2909462 | 0.1013304 | 0.3922765 |  |
| 36 | 10 | 0.1156766 | 69.2162583 | 0.9622491 | 0.1020003 | 1.0642494 | ✓ |

### N47：selection → action 39

path=[2, 36]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [39]，并列按 prior 抽样。trace 行 1698。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.0 | 0.0533195 | 0.0533195 |  |
| 8 | 3 | 0.059354 | 68.9914085 | 0.5872199 | 0.0656929 | 0.6529128 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.043322 | 1.043322 |  |
| 39 | 1 | 0.0880029 | 67.8210109 | 0.2699664 | 0.1948028 | 0.4647691 | ✓ |

### N63：expansion → action 9

path=[2, 36, 39]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 13 个代表动作。trace 行 1700。

bucket=0，compatibility_richness_prior；到达 N64（新建）；closure=[]。

## iteration 70

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[2, 30, 36, 16, 35, 26, 13, 18, 33, 37, 23, 17, 3, 7]

### N0：selection → action 0

path=[]；visits=69；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1720。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.0 | 0.6643583 | 0.6643583 |  |
| 0 | 15 | 0.3308308 | 68.2681672 | 1.0 | 0.2404576 | 1.2404576 | ✓ |
| 2 | 49 | 0.3264006 | 68.2198229 | 0.3814872 | 0.075916 | 0.4574032 |  |

### N2：selection → action 28

path=[0]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 23]，并列按 prior 抽样。trace 行 1722。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0670232 | 64.9415514 | 0.0 | 0.0908529 | 0.0908529 | ✓ |
| 23 | 3 | 0.0316243 | 71.7433075 | 1.0 | 0.0428682 | 1.0428682 |  |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.4748656 | 0.0752432 | 0.5501088 |  |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.4806352 | 0.0865486 | 0.5671838 |  |

### N3：selection → action 25

path=[0, 28]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [25, 9]，并列按 prior 抽样。trace 行 1724。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 1 | 0.0324844 | 64.0815114 | 0.0 | 0.0393852 | 0.0393852 | ✓ |
| 9 | 1 | 0.0315619 | 68.0539547 | 1.0 | 0.0382668 | 1.0382668 |  |

### N12：expansion → action 39

path=[0, 28, 25]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 1726。

bucket=0，compatibility_richness_prior；到达 N65（新建）；closure=[]。

## iteration 71

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 16, 6, 21, 9, 23, 33, 1, 25, 18, 34, 26, 8, 13, 3]

### N0：selection → action 24

path=[]；visits=70；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1746。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 5 | 0.3427686 | 68.190005 | 0.9575724 | 0.6691552 | 1.6267276 | ✓ |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.0 | 0.2279471 | 0.2279471 |  |
| 2 | 49 | 0.3264006 | 68.2198229 | 1.0 | 0.0764642 | 1.0764642 |  |

### N1：selection → action 22

path=[24]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [22]，并列按 prior 抽样。trace 行 1748。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 1 | 0.0579127 | 69.643395 | 0.2157597 | 0.0906477 | 0.3064074 | ✓ |
| 3 | 8 | 0.0586854 | 72.1386618 | 1.0 | 0.0204127 | 1.0204127 |  |
| 8 | 5 | 0.0485177 | 68.9568987 | 0.0 | 0.0253141 | 0.0253141 |  |

### N4：expansion → action 28

path=[24, 22]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 28 个代表动作。trace 行 1750。

bucket=0，compatibility_richness_prior；到达 N66（新建）；closure=[]。

## iteration 72

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[19, 25, 38, 31, 29, 27, 17, 37, 39, 23, 10, 13, 12, 6, 1]

### N0：selection → action 24

path=[]；visits=71；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1771。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 6 | 0.3427686 | 68.0963743 | 0.8243462 | 0.5776439 | 1.4019901 | ✓ |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.0 | 0.2295695 | 0.2295695 |  |
| 2 | 49 | 0.3264006 | 68.2198229 | 1.0 | 0.0770084 | 1.0770084 |  |

### N1：selection → action 22

path=[24]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [22]，并列按 prior 抽样。trace 行 1773。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 2 | 0.0579127 | 68.6358078 | 0.0 | 0.0661997 | 0.0661997 | ✓ |
| 3 | 8 | 0.0586854 | 72.1386618 | 1.0 | 0.022361 | 1.022361 |  |
| 8 | 5 | 0.0485177 | 68.9568987 | 0.0916655 | 0.0277302 | 0.1193957 |  |

### N4：expansion → action 4

path=[24, 22]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 27 个代表动作。trace 行 1775。

bucket=1，uniform_random；到达 N67（新建）；closure=[]。

## iteration 73

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[11, 26, 31, 14, 6, 0, 9, 29, 23, 34, 7, 12, 37, 13, 3]

### N0：selection → action 2

path=[]；visits=72；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1796。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5089854 | 0.5089854 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.7139197 | 0.2311806 | 0.9451003 |  |
| 2 | 49 | 0.3264006 | 68.2198229 | 1.0 | 0.0775488 | 1.0775488 | ✓ |

### N5：expansion → action 24

path=[2]；visits=49；children=7；K=8。已有 7 条动作边 < K=8，且尚余 7 个代表动作。trace 行 1798。

bucket=1，uniform_random；到达 N68（新建）；closure=[]。

## iteration 74

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[39, 8, 3, 15, 13, 22, 35, 17, 32, 33, 28, 0, 37, 10]

### N0：selection → action 2

path=[]；visits=73；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1819。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5125078 | 0.5125078 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.6622625 | 0.2327805 | 0.8950429 |  |
| 2 | 50 | 0.3264006 | 68.4114432 | 1.0 | 0.0765544 | 1.0765544 | ✓ |

### N5：selection → action 24

path=[2]；visits=50；children=8；K=8。最低访问优先：child.visits < 5；最少 1 次；候选 [24]，并列按 prior 抽样。trace 行 1821。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1205544 | 0.1205544 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.3175289 | 0.054879 | 0.3724079 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.0842745 | 0.1416089 | 0.2258834 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.2016765 | 0.1343866 | 0.3360632 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.2042537 | 0.1282522 | 0.332506 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.0923838 | 0.1034199 | 0.1958037 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.2732129 | 0.0954283 | 0.3686412 |  |
| 24 | 1 | 0.0252137 | 77.8008374 | 1.0 | 0.1248015 | 1.1248015 | ✓ |

### N68：expansion → action 9

path=[2, 24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 1823。

bucket=0，compatibility_richness_prior；到达 N33（复用）；closure=[]。

## iteration 75

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[28, 3, 38, 0, 39, 16, 33, 36, 7, 30, 35, 12, 17, 10, 11]

### N0：selection → action 2

path=[]；visits=74；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1844。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5160062 | 0.5160062 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.6592225 | 0.2343694 | 0.893592 |  |
| 2 | 51 | 0.3264006 | 68.4236553 | 1.0 | 0.0755947 | 1.0755947 | ✓ |

### N5：selection → action 24

path=[2]；visits=51；children=8；K=8。最低访问优先：child.visits < 5；最少 2 次；候选 [24]，并列按 prior 抽样。trace 行 1846。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.121754 | 0.121754 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.4919805 | 0.0554251 | 0.5474056 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.1305753 | 0.143018 | 0.2735933 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.3124784 | 0.1357239 | 0.4482023 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.3164716 | 0.1295284 | 0.446 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.1431398 | 0.1044489 | 0.2475888 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.4233172 | 0.0963779 | 0.519695 |  |
| 24 | 2 | 0.0252137 | 73.4175482 | 1.0 | 0.0840289 | 1.0840289 | ✓ |

### N68：expansion → action 25

path=[2, 24]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 18 个代表动作。trace 行 1848。

bucket=1，uniform_random；到达 N69（新建）；closure=[]。

## iteration 76

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 33, 22, 13, 16, 21, 7, 32, 26, 18, 17, 20, 12, 3]

### N0：selection → action 2

path=[]；visits=75；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1869。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.519481 | 0.519481 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.6787129 | 0.2359477 | 0.9146606 |  |
| 2 | 52 | 0.3264006 | 68.3472557 | 1.0 | 0.0746679 | 1.0746679 | ✓ |

### N5：selection → action 24

path=[2]；visits=52；children=8；K=8。最低访问优先：child.visits < 5；最少 3 次；候选 [24]，并列按 prior 抽样。trace 行 1871。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1229419 | 0.1229419 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.7867022 | 0.0559658 | 0.8426681 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2087966 | 0.1444134 | 0.3532099 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.4996692 | 0.137048 | 0.6367172 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.5060544 | 0.1307921 | 0.6368465 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.228888 | 0.105468 | 0.334356 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.676906 | 0.0973181 | 0.7742241 |  |
| 24 | 3 | 0.0252137 | 70.4286578 | 1.0 | 0.0636366 | 1.0636366 | ✓ |

### N68：selection → action 25

path=[2, 24]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [25]，并列按 prior 抽样。trace 行 1873。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0573819 | 69.034259 | 1.0 | 0.0463812 | 1.0463812 |  |
| 25 | 1 | 0.0655241 | 64.450877 | 0.0 | 0.0794437 | 0.0794437 | ✓ |

### N69：expansion → action 0

path=[2, 24, 25]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 1875。

bucket=0，compatibility_richness_prior；到达 N70（新建）；closure=[1]。

## iteration 77

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[3, 39, 35, 1, 15, 7, 32, 36, 9, 18, 17, 10, 25, 14]

### N0：selection → action 2

path=[]；visits=76；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1895。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5229328 | 0.5229328 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.7146168 | 0.2375155 | 0.9521323 |  |
| 2 | 53 | 0.3264006 | 68.2174264 | 1.0 | 0.0737721 | 1.0737721 | ✓ |

### N5：selection → action 24

path=[2]；visits=53；children=8；K=8。最低访问优先：child.visits < 5；最少 4 次；候选 [24]，并列按 prior 抽样。trace 行 1897。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1241184 | 0.1241184 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 1.0 | 0.0565014 | 1.0565014 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2654074 | 0.1457953 | 0.4112027 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.6351439 | 0.1383595 | 0.7735035 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6432604 | 0.1320438 | 0.7753041 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2909462 | 0.1064773 | 0.3974234 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.8604348 | 0.0982494 | 0.9586843 |  |
| 24 | 4 | 0.0252137 | 68.1880691 | 0.7003 | 0.0513964 | 0.7516964 | ✓ |

### N68：expansion → action 8

path=[2, 24]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 17 个代表动作。trace 行 1899。

bucket=0，compatibility_richness_prior；到达 N71（新建）；closure=[]。

## iteration 78

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 32, 3, 22, 37, 29, 0, 12, 38, 4, 34, 23, 11, 18]

### N0：selection → action 2

path=[]；visits=77；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1919。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5263619 | 0.5263619 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.6696713 | 0.239073 | 0.9087442 |  |
| 2 | 54 | 0.3264006 | 68.3821447 | 1.0 | 0.0729057 | 1.0729057 | ✓ |

### N5：selection → action 24

path=[2]；visits=54；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1921。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1252839 | 0.1252839 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 0.8657888 | 0.057032 | 0.9228207 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2297867 | 0.1471643 | 0.376951 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.5499005 | 0.1396587 | 0.6895592 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.5569276 | 0.1332836 | 0.6902112 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2518979 | 0.1074771 | 0.359375 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.7449548 | 0.099172 | 0.8441268 |  |
| 24 | 5 | 0.0252137 | 69.9728977 | 1.0 | 0.0432325 | 1.0432325 | ✓ |

### N68：selection → action 8

path=[2, 24]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [8]，并列按 prior 抽样。trace 行 1923。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0573819 | 69.034259 | 0.429266 | 0.0598779 | 0.4891439 |  |
| 25 | 2 | 0.0655241 | 62.95859 | 0.0 | 0.0683743 | 0.0683743 |  |
| 8 | 1 | 0.060259 | 77.1122122 | 1.0 | 0.0943202 | 1.0943202 | ✓ |

### N71：expansion → action 15

path=[2, 24, 8]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 1925。

bucket=0，compatibility_richness_prior；到达 N72（新建）；closure=[]。

## iteration 79

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 3, 1, 16, 14, 33, 11, 13, 36, 39, 34, 23, 25, 18, 9]

### N0：selection → action 2

path=[]；visits=78；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1945。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5297688 | 0.5297688 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.6936449 | 0.2406204 | 0.9342653 |  |
| 2 | 55 | 0.3264006 | 68.2916286 | 1.0 | 0.0720673 | 1.0720673 | ✓ |

### N5：selection → action 37

path=[2]；visits=55；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1947。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1264386 | 0.1264386 |  |
| 37 | 12 | 0.072067 | 69.3644362 | 1.0 | 0.0575576 | 1.0575576 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2654074 | 0.1485207 | 0.4139281 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.6351439 | 0.1409459 | 0.7760898 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6432604 | 0.1345121 | 0.7777725 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2909462 | 0.1084677 | 0.3994138 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.8604348 | 0.100086 | 0.9605209 |  |
| 24 | 6 | 0.0252137 | 68.8780417 | 0.8760825 | 0.037398 | 0.9134805 |  |

### N7：selection → action 38

path=[2, 37]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [38]，并列按 prior 抽样。trace 行 1949。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 0.8170369 | 0.0755184 | 0.8925553 |  |
| 36 | 4 | 0.0777 | 69.9020661 | 0.7096206 | 0.075365 | 0.7849856 |  |
| 38 | 2 | 0.0500079 | 72.9930306 | 1.0 | 0.0808418 | 1.0808418 | ✓ |
| 28 | 3 | 0.0589352 | 62.3484589 | 0.0 | 0.0714551 | 0.0714551 |  |

### N34：expansion → action 28

path=[2, 37, 38]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 12 个代表动作。trace 行 1951。

bucket=1，uniform_random；到达 N73（新建）；closure=[]。

## iteration 80

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[36, 14, 3, 5, 39, 6, 9, 38, 33, 13, 18, 17, 10]

### N0：selection → action 2

path=[]；visits=79；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1972。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5331539 | 0.5331539 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.693161 | 0.2421579 | 0.9353189 |  |
| 2 | 56 | 0.3264006 | 68.2933937 | 1.0 | 0.0712554 | 1.0712554 | ✓ |

### N5：selection → action 37

path=[2]；visits=56；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1974。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1275828 | 0.1275828 |  |
| 37 | 13 | 0.072067 | 69.2895162 | 1.0 | 0.05393 | 1.05393 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.2705718 | 0.1498648 | 0.4204366 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.6475029 | 0.1422215 | 0.7897244 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.6557773 | 0.1357294 | 0.7915067 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.2966076 | 0.1094493 | 0.4060569 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.8771777 | 0.1009918 | 0.9781695 |  |
| 24 | 6 | 0.0252137 | 68.8780417 | 0.8931299 | 0.0377364 | 0.9308663 |  |

### N7：selection → action 28

path=[2, 37]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [3, 38, 28]，并列按 prior 抽样。trace 行 1976。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 0.9546255 | 0.0786021 | 1.0332276 |  |
| 36 | 4 | 0.0777 | 69.9020661 | 0.8291203 | 0.0784424 | 0.9075627 |  |
| 38 | 3 | 0.0500079 | 71.4588458 | 1.0 | 0.0631071 | 1.0631071 |  |
| 28 | 3 | 0.0589352 | 62.3484589 | 0.0 | 0.0743729 | 0.0743729 | ✓ |

### N42：selection → action 16

path=[2, 37, 28]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [16, 29]，并列按 prior 抽样。trace 行 1978。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 16 | 1 | 0.0318651 | 58.6625832 | 1.0 | 0.0386344 | 1.0386344 | ✓ |
| 29 | 1 | 0.0227677 | 58.4165258 | 0.0 | 0.0276044 | 0.0276044 |  |

### N43：expansion → action 31

path=[2, 37, 28, 16]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 1980。

bucket=0，compatibility_richness_prior；到达 N74（新建）；closure=[]。

## iteration 81

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 8, 33, 21, 18, 7, 16, 37, 1, 32, 9, 31, 26, 13]

### N0：selection → action 2

path=[]；visits=80；children=3；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 1999。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5365177 | 0.5365177 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.7513117 | 0.2436857 | 0.9949975 |  |
| 2 | 57 | 0.3264006 | 68.0975587 | 1.0 | 0.0704686 | 1.0704686 | ✓ |

### N5：selection → action 36

path=[2]；visits=57；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2001。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1287169 | 0.1287169 |  |
| 37 | 14 | 0.072067 | 68.4210361 | 0.8671014 | 0.0507821 | 0.9178836 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.3029479 | 0.151197 | 0.4541449 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.7249818 | 0.1434857 | 0.8684675 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.7342463 | 0.1369359 | 0.8711822 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.332099 | 0.1104222 | 0.4425212 |  |
| 36 | 11 | 0.1156766 | 68.8166221 | 0.982139 | 0.1018895 | 1.0840285 | ✓ |
| 24 | 6 | 0.0252137 | 68.8780417 | 1.0 | 0.0380719 | 1.0380719 |  |

### N47：selection → action 39

path=[2, 36]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [39]，并列按 prior 抽样。trace 行 2003。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.1202855 | 0.0559219 | 0.1762075 |  |
| 8 | 3 | 0.059354 | 68.9914085 | 0.6368714 | 0.0688993 | 0.7057707 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0454365 | 1.0454365 |  |
| 39 | 2 | 0.0880029 | 66.3206355 | 0.0 | 0.1362072 | 0.1362072 | ✓ |

### N63：expansion → action 25

path=[2, 36, 39]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 12 个代表动作。trace 行 2005。

bucket=1，uniform_random；到达 N75（新建）；closure=[]。

## iteration 82

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 14, 12, 31, 34, 36, 21, 27, 17, 26, 23, 19, 1, 10]

### N0：selection → action 2

path=[]；visits=81；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2025。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5398605 | 0.5398605 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.7761946 | 0.245204 | 1.0213987 |  |
| 2 | 58 | 0.3264006 | 68.0227244 | 1.0 | 0.0697059 | 1.0697059 | ✓ |

### N5：selection → action 24

path=[2]；visits=58；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2027。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1298411 | 0.1298411 |  |
| 37 | 14 | 0.072067 | 68.4210361 | 0.8671014 | 0.0512257 | 0.9183271 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.3029479 | 0.1525175 | 0.4554654 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.7249818 | 0.1447388 | 0.8697207 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.7342463 | 0.1381319 | 0.8723782 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.332099 | 0.1113866 | 0.4434856 |  |
| 36 | 12 | 0.1156766 | 68.3950014 | 0.8595304 | 0.0948733 | 0.9544038 |  |
| 24 | 6 | 0.0252137 | 68.8780417 | 1.0 | 0.0384044 | 1.0384044 | ✓ |

### N68：selection → action 9

path=[2, 24]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 25, 8]，并列按 prior 抽样。trace 行 2029。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0573819 | 69.034259 | 0.8323522 | 0.065593 | 0.8979452 | ✓ |
| 25 | 2 | 0.0655241 | 62.95859 | 0.0 | 0.0749003 | 0.0749003 |  |
| 8 | 2 | 0.060259 | 70.2579868 | 1.0 | 0.0688817 | 1.0688817 |  |

### N33：expansion → action 39

path=[2, 9, 24]；visits=2；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 2031。

bucket=0，compatibility_richness_prior；到达 N76（新建）；closure=[]。

## iteration 83

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[1, 33, 28, 18, 21, 7, 15, 37, 30, 5, 10, 35, 14]

### N0：selection → action 2

path=[]；visits=82；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2051。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5431828 | 0.5431828 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.8135493 | 0.246713 | 1.0602623 |  |
| 2 | 59 | 0.3264006 | 67.9189762 | 1.0 | 0.0689659 | 1.0689659 | ✓ |

### N5：selection → action 36

path=[2]；visits=59；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2053。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1309556 | 0.1309556 |  |
| 37 | 14 | 0.072067 | 68.4210361 | 1.0 | 0.0516654 | 1.0516654 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.34938 | 0.1538267 | 0.5032067 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.8360981 | 0.1459813 | 0.9820793 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.8467825 | 0.1393176 | 0.9861001 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.3829991 | 0.1123427 | 0.4953418 |  |
| 36 | 12 | 0.1156766 | 68.3950014 | 0.9912686 | 0.0956877 | 1.0869563 | ✓ |
| 24 | 7 | 0.0252137 | 67.8814041 | 0.8190216 | 0.0338923 | 0.8529139 |  |

### N47：selection → action 39

path=[2, 36]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [8, 39]，并列按 prior 抽样。trace 行 2055。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.2691949 | 0.0584086 | 0.3276035 |  |
| 8 | 3 | 0.059354 | 68.9914085 | 0.6983382 | 0.0719629 | 0.7703011 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0474569 | 1.0474569 |  |
| 39 | 3 | 0.0880029 | 65.4661481 | 0.0 | 0.1066979 | 0.1066979 | ✓ |

### N63：selection → action 9

path=[2, 36, 39]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [9, 25]，并列按 prior 抽样。trace 行 2057。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 1 | 0.0954236 | 64.8202601 | 1.0 | 0.115695 | 1.115695 | ✓ |
| 25 | 1 | 0.1068308 | 63.7571732 | 0.0 | 0.1295255 | 0.1295255 |  |

### N64：expansion → action 12

path=[2, 36, 39, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 2059。

bucket=0，compatibility_richness_prior；到达 N77（新建）；closure=[]。

## iteration 84

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[37, 5, 0, 38, 6, 11, 19, 20, 15, 27, 3, 32, 31, 17, 7]

### N0：selection → action 0

path=[]；visits=83；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2078。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5464848 | 0.5464848 |  |
| 0 | 16 | 0.3308308 | 67.5170282 | 0.8653151 | 0.2482128 | 1.1135278 | ✓ |
| 2 | 60 | 0.3264006 | 67.7900106 | 1.0 | 0.0682477 | 1.0682477 |  |

### N2：expansion → action 38

path=[0]；visits=16；children=4；K=5。已有 4 条动作边 < K=5，且尚余 13 个代表动作。trace 行 2080。

bucket=0，compatibility_richness_prior；到达 N4（复用）；closure=[]。

## iteration 85

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 37, 3, 30, 35, 17, 29, 16, 15, 2, 26, 12, 8, 6]

### N0：selection → action 0

path=[]；visits=84；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2102。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5497671 | 0.5497671 |  |
| 0 | 17 | 0.3308308 | 68.1712082 | 1.0 | 0.2358311 | 1.2358311 | ✓ |
| 2 | 60 | 0.3264006 | 67.7900106 | 0.8416966 | 0.0686576 | 0.9103543 |  |

### N2：selection → action 23

path=[0]；visits=17；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [23]，并列按 prior 抽样。trace 行 2104。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0773762 | 0.0773762 |  |
| 23 | 3 | 0.0316243 | 71.7433075 | 0.5655309 | 0.0456367 | 0.6111676 | ✓ |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.3404545 | 0.0801025 | 0.420557 |  |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.3429274 | 0.092138 | 0.4350654 |  |
| 38 | 4 | 0.0714477 | 78.6380884 | 1.0 | 0.0824842 | 1.0824842 |  |

### N8：selection → action 1

path=[0, 23]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 1]，并列按 prior 抽样。trace 行 2106。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.1473096 | 66.9522569 | 0.0 | 0.1786034 | 0.1786034 |  |
| 1 | 1 | 0.0832401 | 67.8132014 | 1.0 | 0.1009233 | 1.1009233 | ✓ |

### N44：expansion → action 38

path=[0, 23, 1]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2108。

bucket=0，compatibility_richness_prior；到达 N78（新建）；closure=[]。

## iteration 86

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[18, 36, 1, 7, 8, 39, 11, 30, 34, 25, 35, 17, 14]

### N0：selection → action 2

path=[]；visits=85；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2128。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5530298 | 0.5530298 |  |
| 0 | 18 | 0.3308308 | 67.3955485 | 0.805379 | 0.2247449 | 1.0301239 |  |
| 2 | 60 | 0.3264006 | 67.7900106 | 1.0 | 0.0690651 | 1.0690651 | ✓ |

### N5：selection → action 37

path=[2]；visits=60；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2130。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1320608 | 0.1320608 |  |
| 37 | 14 | 0.072067 | 68.4210361 | 1.0 | 0.0521014 | 1.0521014 | ✓ |
| 0 | 5 | 0.085828 | 66.4810506 | 0.34938 | 0.1551248 | 0.5045048 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.8360981 | 0.1472132 | 0.9833113 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 0.8467825 | 0.1404933 | 0.9872758 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.3829991 | 0.1132908 | 0.4962899 |  |
| 36 | 13 | 0.1156766 | 67.7631582 | 0.7793651 | 0.0896027 | 0.8689677 |  |
| 24 | 7 | 0.0252137 | 67.8814041 | 0.8190216 | 0.0341783 | 0.8532 |  |

### N7：selection → action 38

path=[2, 37]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [3, 38]，并列按 prior 抽样。trace 行 2132。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 0.9603085 | 0.0815692 | 1.0418777 |  |
| 36 | 4 | 0.0777 | 69.9020661 | 0.8505224 | 0.0814035 | 0.9319259 |  |
| 38 | 3 | 0.0500079 | 71.4588458 | 1.0 | 0.0654893 | 1.0654893 | ✓ |
| 28 | 4 | 0.0589352 | 61.0440429 | 0.0 | 0.0617443 | 0.0617443 |  |

### N34：selection → action 28

path=[2, 37, 38]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [28]，并列按 prior 抽样。trace 行 2134。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 0.1235034 | 70.5936056 | 1.0 | 0.0998266 | 1.0998266 |  |
| 28 | 1 | 0.0857718 | 68.3904763 | 0.0 | 0.1039928 | 0.1039928 | ✓ |

### N73：expansion → action 4

path=[2, 37, 38, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 2136。

bucket=0，compatibility_richness_prior；到达 N79（新建）；closure=[]。

## iteration 87

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[5, 28, 0, 14, 26, 13, 21, 12, 15, 1, 33, 35, 4, 18]

### N0：selection → action 0

path=[]；visits=86；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2155。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5562734 | 0.5562734 |  |
| 0 | 18 | 0.3308308 | 67.3955485 | 0.889606 | 0.2260631 | 1.1156691 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 1.0 | 0.0683497 | 1.0683497 |  |

### N2：selection → action 38

path=[0]；visits=18；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [28, 23, 6, 19, 38]，并列按 prior 抽样。trace 行 2157。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0796195 | 0.0796195 |  |
| 23 | 4 | 0.0316243 | 67.3598141 | 0.2893086 | 0.0375678 | 0.3268764 |  |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.3404545 | 0.0824248 | 0.4228793 |  |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.3429274 | 0.0948092 | 0.4377367 |  |
| 38 | 4 | 0.0714477 | 78.6380884 | 1.0 | 0.0848756 | 1.0848756 | ✓ |

### N4：expansion → action 19

path=[24, 22]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 26 个代表动作。trace 行 2159。

bucket=0，compatibility_richness_prior；到达 N80（新建）；closure=[]。

## iteration 88

已发现集合：[1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class19；rollout：[6, 25, 20, 33, 14, 11, 31, 9, 34, 38, 13, 16, 4, 27]

### N0：selection → action 0

path=[]；visits=87；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2179。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5594982 | 0.5594982 |  |
| 0 | 19 | 0.3308308 | 67.8600087 | 1.0 | 0.2160049 | 1.2160049 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.8750987 | 0.0687459 | 0.9438446 |  |

### N2：selection → action 23

path=[0]；visits=19；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [28, 23, 6, 19]，并列按 prior 抽样。trace 行 2181。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0818012 | 0.0818012 |  |
| 23 | 4 | 0.0316243 | 67.3598141 | 0.3131648 | 0.0385973 | 0.351762 | ✓ |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.3685281 | 0.0846834 | 0.4532116 |  |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.371205 | 0.0974072 | 0.4686122 |  |
| 38 | 5 | 0.0714477 | 77.4291907 | 1.0 | 0.0726678 | 1.0726678 |  |

### N8：expansion → action 24

path=[0, 23]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 7 个代表动作。trace 行 2183。

bucket=0，compatibility_richness_prior；到达 N81（新建）；closure=[]。

## iteration 89

已发现集合：[1, 7, 8, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 24, 32, 34, 11, 8, 5, 35, 22, 13, 27, 33, 38, 1, 14]

### N0：selection → action 0

path=[]；visits=88；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2203。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5627045 | 0.5627045 |  |
| 0 | 20 | 0.3308308 | 71.4545948 | 1.0 | 0.2068979 | 1.2068979 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.3224027 | 0.0691399 | 0.3915426 |  |

### N2：selection → action 6

path=[0]；visits=20；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [28, 6, 19]，并列按 prior 抽样。trace 行 2205。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0839263 | 0.0839263 |  |
| 23 | 5 | 0.0316243 | 81.8381973 | 1.0 | 0.033 | 1.033 |  |
| 6 | 4 | 0.0693847 | 68.1714712 | 0.283322 | 0.0868834 | 0.3702053 | ✓ |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.2853799 | 0.0999377 | 0.3853176 |  |
| 38 | 5 | 0.0714477 | 77.4291907 | 0.7687933 | 0.0745556 | 0.8433489 |  |

### N15：expansion → action 26

path=[0, 6]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 9 个代表动作。trace 行 2207。

bucket=0，compatibility_richness_prior；到达 N82（新建）；closure=[]。

## iteration 90

已发现集合：[1, 7, 8, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 22, 7, 8, 30, 5, 17, 37, 20, 35, 31, 21, 25, 29, 2]

### N0：selection → action 0

path=[]；visits=89；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2228。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5658927 | 0.5658927 |  |
| 0 | 21 | 0.3308308 | 70.3272916 | 1.0 | 0.1986124 | 1.1986124 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.402034 | 0.0695316 | 0.4715656 |  |

### N2：selection → action 19

path=[0]；visits=21；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [28, 19]，并列按 prior 抽样。trace 行 2230。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0859989 | 0.0859989 |  |
| 23 | 5 | 0.0316243 | 81.8381973 | 1.0 | 0.0338149 | 1.0338149 |  |
| 6 | 5 | 0.0693847 | 66.4311235 | 0.1920588 | 0.0741908 | 0.2662496 |  |
| 19 | 4 | 0.0798098 | 68.2107148 | 0.2853799 | 0.1024057 | 0.3877856 | ✓ |
| 38 | 5 | 0.0714477 | 77.4291907 | 0.7687933 | 0.0763967 | 0.8451901 |  |

### N55：expansion → action 26

path=[0, 19]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 14 个代表动作。trace 行 2232。

bucket=0，compatibility_richness_prior；到达 N83（新建）；closure=[]。

## iteration 91

已发现集合：[1, 7, 8, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 6, 17, 14, 20, 38, 34, 29, 19, 31, 35, 21, 15, 22, 2]

### N0：selection → action 0

path=[]；visits=90；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2253。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.569063 | 0.569063 |  |
| 0 | 22 | 0.3308308 | 69.3266118 | 1.0 | 0.1910414 | 1.1910414 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.5149331 | 0.0699211 | 0.5848543 |  |

### N2：selection → action 28

path=[0]；visits=22；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [28]，并列按 prior 抽样。trace 行 2255。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 4 | 0.0670232 | 62.7686491 | 0.0 | 0.0880226 | 0.0880226 | ✓ |
| 23 | 5 | 0.0316243 | 81.8381973 | 1.0 | 0.0346106 | 1.0346106 |  |
| 6 | 5 | 0.0693847 | 66.4311235 | 0.1920588 | 0.0759367 | 0.2679955 |  |
| 19 | 5 | 0.0798098 | 65.7666238 | 0.1572127 | 0.0873463 | 0.244559 |  |
| 38 | 5 | 0.0714477 | 77.4291907 | 0.7687933 | 0.0781946 | 0.8469879 |  |

### N3：expansion → action 37

path=[0, 28]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 36 个代表动作。trace 行 2257。

bucket=0，compatibility_richness_prior；到达 N84（新建）；closure=[]。

## iteration 92

已发现集合：[1, 7, 8, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[8, 30, 35, 38, 7, 16, 6, 21, 22, 26, 36, 4, 11, 3]

### N0：selection → action 0

path=[]；visits=91；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2278。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5722157 | 0.5722157 |  |
| 0 | 23 | 0.3308308 | 68.5484068 | 1.0 | 0.1840956 | 1.1840956 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.6588082 | 0.0703085 | 0.7291167 |  |

### N2：selection → action 23

path=[0]；visits=23；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2280。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0670232 | 62.9348103 | 0.0 | 0.0750008 | 0.0750008 |  |
| 23 | 5 | 0.0316243 | 81.8381973 | 1.0 | 0.0353885 | 1.0353885 | ✓ |
| 6 | 5 | 0.0693847 | 66.4311235 | 0.184957 | 0.0776434 | 0.2626003 |  |
| 19 | 5 | 0.0798098 | 65.7666238 | 0.1498046 | 0.0893094 | 0.2391139 |  |
| 38 | 5 | 0.0714477 | 77.4291907 | 0.766761 | 0.079952 | 0.846713 |  |

### N8：selection → action 24

path=[0, 23]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 24]，并列按 prior 抽样。trace 行 2282。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.1473096 | 66.9522569 | 0.0754503 | 0.230576 | 0.3060263 |  |
| 1 | 2 | 0.0832401 | 61.0112676 | 0.0 | 0.0868609 | 0.0868609 |  |
| 24 | 1 | 0.1667107 | 139.7517298 | 1.0 | 0.2609435 | 1.2609435 | ✓ |

### N81：expansion → action 20

path=[0, 23, 24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2284。

bucket=0，compatibility_richness_prior；到达 N85（新建）；closure=[]。

## iteration 93

已发现集合：[1, 7, 8, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class11；rollout：[20, 33, 9, 5, 37, 27, 7, 3, 12, 38, 28, 11, 35]

### N0：selection → action 0

path=[]；visits=92；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2304。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5753512 | 0.5753512 |  |
| 0 | 24 | 0.3308308 | 67.6705087 | 1.0 | 0.1777002 | 1.1777002 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.9620433 | 0.0706938 | 1.0327371 |  |

### N2：selection → action 38

path=[0]；visits=24；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2306。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0670232 | 62.9348103 | 0.0 | 0.0766139 | 0.0766139 |  |
| 23 | 6 | 0.0316243 | 72.4200315 | 0.6544068 | 0.0309854 | 0.6853922 |  |
| 6 | 5 | 0.0693847 | 66.4311235 | 0.2412185 | 0.0793133 | 0.3205318 |  |
| 19 | 5 | 0.0798098 | 65.7666238 | 0.1953732 | 0.0912302 | 0.2866034 |  |
| 38 | 5 | 0.0714477 | 77.4291907 | 1.0 | 0.0816716 | 1.0816716 | ✓ |

### N4：selection → action 19

path=[24, 22]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [28, 4, 19]，并列按 prior 抽样。trace 行 2308。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 1 | 0.04314 | 67.6282207 | 0.6486754 | 0.0675248 | 0.7162002 |  |
| 4 | 1 | 0.0068055 | 51.7640733 | 0.0 | 0.0106522 | 0.0106522 |  |
| 19 | 1 | 0.0483064 | 76.2202929 | 1.0 | 0.0756115 | 1.0756115 | ✓ |

### N80：expansion → action 0

path=[24, 22, 19]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2310。

bucket=0，compatibility_richness_prior；到达 N86（新建）；closure=[]。

## iteration 94

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 14, 21, 9, 25, 11, 24, 5, 17, 35, 28, 33, 19, 31, 36, 4, 10]

### N0：selection → action 0

path=[]；visits=93；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2329。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5784696 | 0.5784696 |  |
| 0 | 25 | 0.3308308 | 74.8591733 | 1.0 | 0.1717917 | 1.1717917 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.2017291 | 0.0710769 | 0.272806 |  |

### N2：expansion → action 26

path=[0]；visits=25；children=5；K=6。已有 5 条动作边 < K=6，且尚余 12 个代表动作。trace 行 2331。

bucket=1，uniform_random；到达 N87（新建）；closure=[]。

## iteration 95

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[8, 25, 6, 2, 35, 39, 28, 17, 12, 34, 24, 23, 36, 5, 13]

### N0：selection → action 0

path=[]；visits=94；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2354。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5815713 | 0.5815713 |  |
| 0 | 26 | 0.3308308 | 75.8015387 | 1.0 | 0.1663161 | 1.1663161 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.1827915 | 0.0714581 | 0.2542495 |  |

### N2：selection → action 26

path=[0]；visits=26；children=6；K=6。最低访问优先：child.visits < 5；最少 1 次；候选 [26]，并列按 prior 抽样。trace 行 2356。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.079892 | 0.079892 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.1696424 | 0.0331744 | 0.2028168 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.0625313 | 0.0817256 | 0.1442569 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.0506468 | 0.0968351 | 0.1474818 |  |
| 38 | 6 | 0.0717799 | 118.8478335 | 1.0 | 0.0732014 | 1.0732014 |  |
| 26 | 1 | 0.0570798 | 78.1574521 | 0.2722557 | 0.2037357 | 0.4759915 | ✓ |

### N87：expansion → action 29

path=[0, 26]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 28 个代表动作。trace 行 2358。

bucket=0，compatibility_richness_prior；到达 N88（新建）；closure=[]。

## iteration 96

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[19, 22, 2, 20, 25, 5, 23, 3, 8, 7, 11, 9, 35, 29]

### N0：selection → action 0

path=[]；visits=95；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2379。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5846566 | 0.5846566 |  |
| 0 | 27 | 0.3308308 | 73.4913104 | 1.0 | 0.161227 | 1.161227 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.2374348 | 0.0718371 | 0.3092719 |  |

### N2：selection → action 26

path=[0]；visits=27；children=6；K=6。最低访问优先：child.visits < 5；最少 2 次；候选 [26]，并列按 prior 抽样。trace 行 2381。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0814139 | 0.0814139 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.1696424 | 0.0338063 | 0.2034487 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.0625313 | 0.0832824 | 0.1458137 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.0506468 | 0.0986797 | 0.1493265 |  |
| 38 | 6 | 0.0717799 | 118.8478335 | 1.0 | 0.0745959 | 1.0745959 |  |
| 26 | 2 | 0.0570798 | 71.7814817 | 0.158222 | 0.1384112 | 0.2966332 | ✓ |

### N87：expansion → action 38

path=[0, 26]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2383。

bucket=1，uniform_random；到达 N89（新建）；closure=[]。

## iteration 97

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[37, 8, 32, 20, 25, 21, 2, 18, 15, 30, 23, 27, 36, 3]

### N0：selection → action 0

path=[]；visits=96；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2403。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5877257 | 0.5877257 |  |
| 0 | 28 | 0.3308308 | 72.9043706 | 1.0 | 0.1564846 | 1.1564846 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.2569497 | 0.0722142 | 0.329164 |  |

### N2：selection → action 26

path=[0]；visits=28；children=6；K=6。最低访问优先：child.visits < 5；最少 3 次；候选 [26]，并列按 prior 抽样。trace 行 2405。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0829078 | 0.0829078 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.1696424 | 0.0344267 | 0.2040691 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.0625313 | 0.0848107 | 0.1473419 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.0506468 | 0.1004905 | 0.1511373 |  |
| 38 | 6 | 0.0717799 | 118.8478335 | 1.0 | 0.0759647 | 1.0759647 |  |
| 26 | 3 | 0.0570798 | 71.2753684 | 0.1491702 | 0.1057133 | 0.2548835 | ✓ |

### N87：selection → action 29

path=[0, 26]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [29, 38]，并列按 prior 抽样。trace 行 2407。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 29 | 1 | 0.0445819 | 65.4055113 | 0.0 | 0.0540527 | 0.0540527 | ✓ |
| 38 | 1 | 0.0084801 | 70.2631418 | 1.0 | 0.0102816 | 1.0102816 |  |

### N88：expansion → action 16

path=[0, 26, 29]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 2409。

bucket=0，compatibility_richness_prior；到达 N90（新建）；closure=[]。

## iteration 98

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 21, 34, 39, 17, 9, 36, 32, 18, 33, 23, 7, 37, 24, 1]

### N0：selection → action 0

path=[]；visits=97；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2429。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5907789 | 0.5907789 |  |
| 0 | 29 | 0.3308308 | 68.8850883 | 1.0 | 0.1520543 | 1.1520543 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.587759 | 0.0725894 | 0.6603484 |  |

### N2：selection → action 26

path=[0]；visits=29；children=6；K=6。最低访问优先：child.visits < 5；最少 4 次；候选 [26]，并列按 prior 抽样。trace 行 2431。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0843754 | 0.0843754 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.1696424 | 0.035036 | 0.2046785 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.0625313 | 0.0863119 | 0.1488431 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.0506468 | 0.1022692 | 0.152916 |  |
| 38 | 6 | 0.0717799 | 118.8478335 | 1.0 | 0.0773093 | 1.0773093 |  |
| 26 | 4 | 0.0570798 | 65.1512851 | 0.0396415 | 0.0860676 | 0.125709 | ✓ |

### N87：expansion → action 6

path=[0, 26]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 26 个代表动作。trace 行 2433。

bucket=0，compatibility_richness_prior；到达 N82（复用）；closure=[]。

## iteration 99

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 9, 21, 3, 5, 32, 19, 27, 7, 12, 26, 11, 37, 2]

### N0：selection → action 0

path=[]；visits=98；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2455。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5938163 | 0.5938163 |  |
| 0 | 30 | 0.3308308 | 68.8175973 | 1.0 | 0.1479059 | 1.1479059 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.6007463 | 0.0729626 | 0.6737089 |  |

### N2：selection → action 38

path=[0]；visits=30；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2457。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0858178 | 0.0858178 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.1696424 | 0.035635 | 0.2052774 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.0625313 | 0.0877874 | 0.1503187 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.0506468 | 0.1040175 | 0.1546643 |  |
| 38 | 6 | 0.0717799 | 118.8478335 | 1.0 | 0.078631 | 1.078631 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.0511866 | 0.0729491 | 0.1241357 |  |

### N4：selection → action 28

path=[24, 22]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [28, 4]，并列按 prior 抽样。trace 行 2459。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 1 | 0.04314 | 67.6282207 | 0.211168 | 0.0739697 | 0.2851377 | ✓ |
| 4 | 1 | 0.0068055 | 51.7640733 | 0.0 | 0.0116689 | 0.0116689 |  |
| 19 | 2 | 0.0483064 | 126.8897825 | 1.0 | 0.0552188 | 1.0552188 |  |

### N66：expansion → action 16

path=[24, 22, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 2461。

bucket=0，compatibility_richness_prior；到达 N91（新建）；closure=[]。

## iteration 100

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[20, 23, 25, 3, 37, 38, 36, 28, 18, 6, 15, 7, 29, 30]

### N0：selection → action 0

path=[]；visits=99；children=3；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2481。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.5968383 | 0.5968383 |  |
| 0 | 31 | 0.3308308 | 68.0811662 | 1.0 | 0.144013 | 1.144013 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.7916058 | 0.0733339 | 0.8649397 |  |

### N2：selection → action 38

path=[0]；visits=31；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2483。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0872363 | 0.0872363 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.6334074 | 0.036224 | 0.6696314 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.233478 | 0.0892385 | 0.3227165 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.1891038 | 0.105737 | 0.2948408 |  |
| 38 | 7 | 0.0717799 | 77.909724 | 1.0 | 0.0699394 | 1.0699394 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.1911196 | 0.0741549 | 0.2652745 |  |

### N4：selection → action 4

path=[24, 22]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [4]，并列按 prior 抽样。trace 行 2485。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.04314 | 62.8561852 | 0.1476473 | 0.0532643 | 0.2009116 |  |
| 4 | 1 | 0.0068055 | 51.7640733 | 0.0 | 0.0126039 | 0.0126039 | ✓ |
| 19 | 2 | 0.0483064 | 126.8897825 | 1.0 | 0.0596432 | 1.0596432 |  |

### N67：expansion → action 1

path=[24, 22, 4]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2487。

bucket=0，compatibility_richness_prior；到达 N92（新建）；closure=[]。

## iteration 101

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 37, 1, 16, 30, 7, 39, 20, 0, 38, 33, 27, 4, 13]

### N0：selection → action 0

path=[]；visits=100；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2507。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.599845 | 0.599845 |  |
| 0 | 32 | 0.3308308 | 68.3556142 | 1.0 | 0.1403525 | 1.1403525 | ✓ |
| 2 | 61 | 0.3264006 | 67.5981132 | 0.7078022 | 0.0737034 | 0.7815056 |  |

### N2：selection → action 38

path=[0]；visits=32；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2509。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0886322 | 0.0886322 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 0.7947822 | 0.0368036 | 0.8315858 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.2929618 | 0.0906664 | 0.3836282 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.2372823 | 0.1074289 | 0.3447111 |  |
| 38 | 8 | 0.0717799 | 74.869176 | 1.0 | 0.0631631 | 1.0631631 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.2398116 | 0.0753415 | 0.3151531 |  |

### N4：selection → action 28

path=[24, 22]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [28, 4, 19]，并列按 prior 抽样。trace 行 2511。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 2 | 0.04314 | 62.8561852 | 0.0 | 0.0569419 | 0.0569419 | ✓ |
| 4 | 2 | 0.0068055 | 69.5752255 | 0.1049299 | 0.0089828 | 0.1139127 |  |
| 19 | 2 | 0.0483064 | 126.8897825 | 1.0 | 0.0637612 | 1.0637612 |  |

### N66：expansion → action 8

path=[24, 22, 28]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 2513。

bucket=1，uniform_random；到达 N93（新建）；closure=[]。

## iteration 102

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 21, 32, 22, 13, 6, 0, 24, 26, 12, 33, 10, 3, 7]

### N0：selection → action 2

path=[]；visits=101；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2533。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6028368 | 0.6028368 |  |
| 0 | 33 | 0.3308308 | 67.3256705 | 0.8515238 | 0.1369039 | 0.9884277 |  |
| 2 | 61 | 0.3264006 | 67.5981132 | 1.0 | 0.074071 | 1.074071 | ✓ |

### N5：selection → action 28

path=[2]；visits=61；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2535。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1331567 | 0.1331567 |  |
| 37 | 15 | 0.072067 | 67.5985849 | 0.8552037 | 0.0492504 | 0.9044541 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.4125971 | 0.1564122 | 0.5690093 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 0.9873823 | 0.1484349 | 1.1358172 |  |
| 28 | 5 | 0.0777326 | 67.9641801 | 1.0 | 0.1416592 | 1.1416592 | ✓ |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.4522992 | 0.114231 | 0.5665302 |  |
| 36 | 13 | 0.1156766 | 67.7631582 | 0.920384 | 0.0903463 | 1.0107303 |  |
| 24 | 7 | 0.0252137 | 67.8814041 | 0.967216 | 0.034462 | 1.001678 |  |

### N14：selection → action 3

path=[24, 8]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [12, 3, 9]，并列按 prior 抽样。trace 行 2537。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 1 | 0.0414515 | 66.7093017 | 0.650736 | 0.0648819 | 0.7156179 |  |
| 3 | 1 | 0.0453323 | 61.285529 | 0.0 | 0.0709563 | 0.0709563 | ✓ |
| 9 | 1 | 0.0318902 | 69.6203573 | 1.0 | 0.049916 | 1.049916 |  |

### N31：expansion → action 35

path=[24, 8, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 2539。

bucket=0，compatibility_richness_prior；到达 N53（复用）；closure=[]。

## iteration 103

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 0, 9, 28, 26, 17, 39, 36, 8, 15, 5, 10, 31, 16]

### N0：selection → action 2

path=[]；visits=102；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2560。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6058138 | 0.6058138 |  |
| 0 | 33 | 0.3308308 | 67.3256705 | 0.8318943 | 0.13758 | 0.9694743 |  |
| 2 | 62 | 0.3264006 | 67.6414103 | 1.0 | 0.0732552 | 1.0732552 | ✓ |

### N5：selection → action 6

path=[2]；visits=62；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2562。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.0 | 0.1342438 | 0.1342438 |  |
| 37 | 15 | 0.072067 | 67.5985849 | 0.8661323 | 0.0496524 | 0.9157847 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.4178696 | 0.157689 | 0.5755587 |  |
| 6 | 5 | 0.0814506 | 67.9323218 | 1.0 | 0.1496466 | 1.1496466 | ✓ |
| 28 | 6 | 0.0777326 | 67.8380203 | 0.962174 | 0.1224134 | 1.0845874 |  |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.4580791 | 0.1151635 | 0.5732426 |  |
| 36 | 13 | 0.1156766 | 67.7631582 | 0.9321455 | 0.0910838 | 1.0232293 |  |
| 24 | 7 | 0.0252137 | 67.8814041 | 0.979576 | 0.0347433 | 1.0143193 |  |

### N20：selection → action 21

path=[2, 6]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [21, 4]，并列按 prior 抽样。trace 行 2564。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 1 | 0.0231677 | 70.0183403 | 1.0 | 0.0362632 | 1.0362632 | ✓ |
| 38 | 2 | 0.0257828 | 65.5767431 | 0.2072311 | 0.0269043 | 0.2341354 |  |
| 4 | 1 | 0.0280851 | 64.4157022 | 0.0 | 0.0439601 | 0.0439601 |  |

### N21：expansion → action 25

path=[2, 6, 21]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 20 个代表动作。trace 行 2566。

bucket=0，compatibility_richness_prior；到达 N94（新建）；closure=[]。

## iteration 104

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[24, 20, 11, 25, 31, 27, 14, 38, 33, 22, 9, 5, 1, 13]

### N0：selection → action 0

path=[]；visits=103；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2586。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6087762 | 0.6087762 |  |
| 0 | 33 | 0.3308308 | 67.3256705 | 1.0 | 0.1382527 | 1.1382527 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.4989948 | 0.0724632 | 0.571458 |  |

### N2：selection → action 23

path=[0]；visits=33；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2588。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0900064 | 0.0900064 |  |
| 23 | 6 | 0.0325301 | 72.4200315 | 1.0 | 0.0373743 | 1.0373743 | ✓ |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.3686064 | 0.0920722 | 0.4606786 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.2985501 | 0.1090945 | 0.4076446 |  |
| 38 | 9 | 0.0717799 | 69.7349624 | 0.7169208 | 0.0577282 | 0.774649 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.3017324 | 0.0765097 | 0.3782421 |  |

### N8：selection → action 6

path=[0, 23]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [6]，并列按 prior 抽样。trace 行 2590。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.1473096 | 66.9522569 | 0.3843575 | 0.2525833 | 0.6369408 | ✓ |
| 1 | 2 | 0.0832401 | 61.0112676 | 0.0 | 0.0951514 | 0.0951514 |  |
| 24 | 2 | 0.1667107 | 76.4682054 | 1.0 | 0.1905662 | 1.1905662 |  |

### N30：expansion → action 34

path=[0, 23, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 21 个代表动作。trace 行 2592。

bucket=0，compatibility_richness_prior；到达 N95（新建）；closure=[]。

## iteration 105

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 26, 34, 1, 8, 2, 37, 11, 16, 3, 9, 7, 27, 35, 13]

### N0：selection → action 0

path=[]；visits=104；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2612。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6117243 | 0.6117243 |  |
| 0 | 34 | 0.3308308 | 66.8188285 | 1.0 | 0.134953 | 1.134953 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.7385759 | 0.0728141 | 0.8113901 |  |

### N2：selection → action 38

path=[0]；visits=34；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2614。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.09136 | 0.09136 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.3668261 | 0.0331943 | 0.4000204 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.5141522 | 0.0934568 | 0.607609 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.4164338 | 0.1107351 | 0.527169 |  |
| 38 | 9 | 0.0717799 | 69.7349624 | 1.0 | 0.0585963 | 1.0585963 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.4208728 | 0.0776602 | 0.498533 |  |

### N4：expansion → action 6

path=[24, 22]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 25 个代表动作。trace 行 2616。

bucket=1，uniform_random；到达 N96（新建）；closure=[]。

## iteration 106

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[8, 37, 32, 9, 27, 5, 26, 34, 18, 19, 11, 7, 3, 2]

### N0：selection → action 0

path=[]；visits=105；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2637。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6146583 | 0.6146583 |  |
| 0 | 35 | 0.3308308 | 67.0931238 | 1.0 | 0.1318336 | 1.1318336 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.5862468 | 0.0731634 | 0.6594102 |  |

### N2：selection → action 38

path=[0]；visits=35；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2639。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0926938 | 0.0926938 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.3606133 | 0.0336789 | 0.3942922 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.5054442 | 0.0948212 | 0.6002654 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.4093809 | 0.1123518 | 0.5217326 |  |
| 38 | 10 | 0.0717799 | 69.8521183 | 1.0 | 0.0540471 | 1.0540471 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.4137446 | 0.078794 | 0.4925386 |  |

### N4：selection → action 6

path=[24, 22]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [6]，并列按 prior 抽样。trace 行 2641。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0476703 | 0.0476703 |  |
| 4 | 2 | 0.0056816 | 69.5752255 | 0.1395575 | 0.0083845 | 0.147942 |  |
| 19 | 2 | 0.0477591 | 126.8897825 | 1.0 | 0.0704796 | 1.0704796 |  |
| 6 | 1 | 0.0365193 | 70.2475196 | 0.1496504 | 0.0808389 | 0.2304892 | ✓ |

### N96：expansion → action 35

path=[24, 22, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 2643。

bucket=0，compatibility_richness_prior；到达 N97（新建）；closure=[]。

## iteration 107

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[27, 11, 38, 6, 5, 19, 16, 3, 33, 13, 30, 28, 36, 22, 37]

### N0：selection → action 0

path=[]；visits=106；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2663。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6175783 | 0.6175783 |  |
| 0 | 36 | 0.3308308 | 66.5828169 | 1.0 | 0.1288799 | 1.1288799 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.9512486 | 0.0735109 | 1.0247595 |  |

### N2：expansion → action 20

path=[0]；visits=36；children=6；K=7。已有 6 条动作边 < K=7，且尚余 11 个代表动作。trace 行 2665。

bucket=0，compatibility_richness_prior；到达 N98（新建）；closure=[]。

## iteration 108

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 24, 32, 25, 22, 27, 8, 29, 33, 11, 30, 10, 34, 7, 2]

### N0：selection → action 0

path=[]；visits=107；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2686。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6204845 | 0.6204845 |  |
| 0 | 37 | 0.3308308 | 66.8487395 | 1.0 | 0.1260789 | 1.1260789 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.7182254 | 0.0738569 | 0.7920823 |  |

### N2：selection → action 20

path=[0]；visits=37；children=7；K=7。最低访问优先：child.visits < 5；最少 1 次；候选 [20]，并列按 prior 抽样。trace 行 2688。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0953054 | 0.0953054 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.3324242 | 0.0346278 | 0.367052 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.4659337 | 0.0974927 | 0.5634265 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.3773796 | 0.1155173 | 0.4928969 |  |
| 38 | 11 | 0.0717799 | 68.0571159 | 0.6826205 | 0.050939 | 0.7335595 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.3814023 | 0.081014 | 0.4624163 |  |
| 20 | 1 | 0.0589459 | 70.4386954 | 1.0 | 0.2509878 | 1.2509878 | ✓ |

### N98：expansion → action 19

path=[0, 20]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 28 个代表动作。trace 行 2690。

bucket=0，compatibility_richness_prior；到达 N99（新建）；closure=[]。

## iteration 109

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 8, 13, 27, 29, 32, 6, 36, 25, 37, 30, 34, 1, 11, 17]

### N0：selection → action 0

path=[]；visits=108；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2711。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6233773 | 0.6233773 |  |
| 0 | 38 | 0.3308308 | 67.0609895 | 1.0 | 0.1234188 | 1.1234188 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.6007626 | 0.0742012 | 0.6749638 |  |

### N2：selection → action 20

path=[0]；visits=38；children=7；K=7。最低访问优先：child.visits < 5；最少 2 次；候选 [20]，并列按 prior 抽样。trace 行 2713。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0965847 | 0.0965847 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.3392067 | 0.0350926 | 0.3742993 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.4754402 | 0.0988014 | 0.5742416 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.3850793 | 0.1170679 | 0.5021472 |  |
| 38 | 11 | 0.0717799 | 68.0571159 | 0.696548 | 0.0516228 | 0.7481708 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.3891841 | 0.0821015 | 0.4712855 |  |
| 20 | 2 | 0.0589459 | 70.2886547 | 1.0 | 0.1695713 | 1.1695713 | ✓ |

### N98：expansion → action 4

path=[0, 20]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2715。

bucket=1，uniform_random；到达 N100（新建）；closure=[]。

## iteration 110

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[27, 25, 29, 15, 8, 35, 38, 12, 37, 36, 26, 2, 4, 21]

### N0：selection → action 0

path=[]；visits=109；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2736。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6262566 | 0.6262566 |  |
| 0 | 39 | 0.3308308 | 66.6465208 | 1.0 | 0.1208891 | 1.1208891 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.8826467 | 0.0745439 | 0.9571906 |  |

### N2：selection → action 20

path=[0]；visits=39；children=7；K=7。最低访问优先：child.visits < 5；最少 3 次；候选 [20]，并列按 prior 抽样。trace 行 2738。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0978473 | 0.0978473 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.4869825 | 0.0355514 | 0.5225339 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.6825663 | 0.100093 | 0.7826593 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.5528396 | 0.1185983 | 0.6714379 |  |
| 38 | 11 | 0.0717799 | 68.0571159 | 1.0 | 0.0522976 | 1.0522976 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.5587326 | 0.0831748 | 0.6419073 |  |
| 20 | 3 | 0.0589459 | 66.9331884 | 0.7805817 | 0.128841 | 0.9094227 | ✓ |

### N98：selection → action 19

path=[0, 20]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [19, 4]，并列按 prior 抽样。trace 行 2740。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 1 | 0.0544664 | 70.1386141 | 1.0 | 0.0660371 | 1.0660371 | ✓ |
| 4 | 1 | 0.0073295 | 60.2222557 | 0.0 | 0.0088865 | 0.0088865 |  |

### N99：expansion → action 30

path=[0, 20, 19]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2742。

bucket=0，compatibility_richness_prior；到达 N101（新建）；closure=[]。

## iteration 111

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[38, 11, 24, 19, 6, 16, 5, 22, 37, 3, 35, 25, 32, 14]

### N0：selection → action 0

path=[]；visits=110；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2762。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.6291228 | 0.6291228 |  |
| 0 | 40 | 0.3308308 | 66.6691223 | 1.0 | 0.1184804 | 1.1184804 | ✓ |
| 2 | 63 | 0.3264006 | 66.5428588 | 0.8606262 | 0.0748851 | 0.9355112 |  |

### N2：selection → action 20

path=[0]；visits=40；children=7；K=7。最低访问优先：child.visits < 5；最少 4 次；候选 [20]，并列按 prior 抽样。trace 行 2764。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.0990938 | 0.0990938 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.4869825 | 0.0360043 | 0.5229868 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.6825663 | 0.1013681 | 0.7839344 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.5528396 | 0.1201091 | 0.6729487 |  |
| 38 | 11 | 0.0717799 | 68.0571159 | 1.0 | 0.0529639 | 1.0529639 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.5587326 | 0.0842344 | 0.6429669 |  |
| 20 | 4 | 0.0589459 | 66.9604031 | 0.7858947 | 0.1043859 | 0.8902806 | ✓ |

### N98：expansion → action 27

path=[0, 20]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 26 个代表动作。trace 行 2766。

bucket=0，compatibility_richness_prior；到达 N102（新建）；closure=[]。

## iteration 112

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 20, 11, 13, 5, 26, 1, 27, 35, 18, 28, 39, 22, 4]

### N0：selection → action 2

path=[]；visits=111；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2786。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.0 | 0.631976 | 0.631976 |  |
| 0 | 41 | 0.3308308 | 66.4829678 | 0.9231842 | 0.116184 | 1.0393681 |  |
| 2 | 63 | 0.3264006 | 66.5428588 | 1.0 | 0.0752247 | 1.0752247 | ✓ |

### N5：selection → action 28

path=[2]；visits=63；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2788。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.073067 | 65.4392872 | 0.5150433 | 0.135322 | 0.6503654 |  |
| 37 | 15 | 0.072067 | 67.5985849 | 0.9438376 | 0.0500513 | 0.9938889 |  |
| 0 | 5 | 0.085828 | 66.4810506 | 0.7219172 | 0.1589556 | 0.8808728 |  |
| 6 | 6 | 0.0814506 | 62.8456614 | 0.0 | 0.1292988 | 0.1292988 |  |
| 28 | 6 | 0.0777326 | 67.8380203 | 0.9913848 | 0.1233967 | 1.1147815 | ✓ |
| 25 | 8 | 0.0940229 | 66.5812942 | 0.7418236 | 0.1160885 | 0.8579121 |  |
| 36 | 13 | 0.1156766 | 67.7631582 | 0.9765187 | 0.0918154 | 1.0683341 |  |
| 24 | 7 | 0.0252137 | 67.8814041 | 1.0 | 0.0350223 | 1.0350223 |  |

### N14：selection → action 9

path=[24, 8]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [12, 9]，并列按 prior 抽样。trace 行 2790。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 1 | 0.0414515 | 66.7093017 | 0.0 | 0.0710745 | 0.0710745 |  |
| 3 | 2 | 0.0453323 | 67.422895 | 0.2451321 | 0.0518192 | 0.2969513 |  |
| 9 | 1 | 0.0318902 | 69.6203573 | 1.0 | 0.0546803 | 1.0546803 | ✓ |

### N35：expansion → action 15

path=[24, 8, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 2792。

bucket=0，compatibility_richness_prior；到达 N103（新建）；closure=[]。

## iteration 113

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[19, 39, 7, 8, 35, 28, 16, 33, 29, 13, 37, 32, 5, 15]

### N0：selection → action 24

path=[]；visits=112；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2812。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 7 | 0.3427686 | 65.7631885 | 0.5629608 | 0.6348163 | 1.1977771 | ✓ |
| 0 | 41 | 0.3308308 | 66.4829678 | 1.0 | 0.1167061 | 1.1167061 |  |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.0 | 0.0744003 | 0.0744003 |  |

### N1：selection → action 3

path=[24]；visits=7；children=3；K=3。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2814。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 11 | 0.0579127 | 63.0118963 | 0.0 | 0.017876 | 0.017876 |  |
| 3 | 8 | 0.0586854 | 72.1386618 | 1.0 | 0.0241527 | 1.0241527 | ✓ |
| 8 | 7 | 0.0485177 | 68.9568987 | 0.6513811 | 0.022464 | 0.6738451 |  |

### N9：selection → action 4

path=[24, 3]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 4]，并列按 prior 抽样。trace 行 2816。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0632242 | 67.3149488 | 0.4202862 | 0.0834517 | 0.5037379 |  |
| 29 | 3 | 0.0579545 | 64.7159413 | 0.0 | 0.0573721 | 0.0573721 |  |
| 4 | 2 | 0.058363 | 70.89984 | 1.0 | 0.0770352 | 1.0770352 | ✓ |

### N37：expansion → action 22

path=[2, 0, 26]；visits=2；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2818。

bucket=0，compatibility_richness_prior；到达 N104（新建）；closure=[]。

## iteration 114

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[11, 21, 0, 5, 32, 13, 31, 34, 39, 35, 7, 3, 8]

### N0：selection → action 0

path=[]；visits=113；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2838。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5667947 | 0.5667947 |  |
| 0 | 41 | 0.3308308 | 66.4829678 | 1.0 | 0.117226 | 1.117226 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.6294118 | 0.0747317 | 0.7041435 |  |

### N2：selection → action 38

path=[0]；visits=41；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2840。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1003248 | 0.1003248 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.4869825 | 0.0364515 | 0.523434 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 0.6825663 | 0.1026274 | 0.7851937 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.5528396 | 0.1216012 | 0.6744408 |  |
| 38 | 11 | 0.0717799 | 68.0571159 | 1.0 | 0.0536218 | 1.0536218 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.5587326 | 0.0852808 | 0.6440134 |  |
| 20 | 5 | 0.0589459 | 66.2133752 | 0.6400565 | 0.0880689 | 0.7281254 |  |

### N4：selection → action 6

path=[24, 22]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [4, 19, 6]，并列按 prior 抽样。trace 行 2842。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.049997 | 0.049997 |  |
| 4 | 2 | 0.0056816 | 69.5752255 | 0.1395575 | 0.0087938 | 0.1483512 |  |
| 19 | 2 | 0.0477591 | 126.8897825 | 1.0 | 0.0739196 | 1.0739196 |  |
| 6 | 2 | 0.0365193 | 65.22575 | 0.0742604 | 0.056523 | 0.1307834 | ✓ |

### N96：expansion → action 26

path=[24, 22, 6]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 2844。

bucket=1，uniform_random；到达 N105（新建）；closure=[]。

## iteration 115

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[11, 33, 20, 19, 37, 30, 32, 29, 24, 17, 26, 38, 9, 1]

### N0：selection → action 0

path=[]；visits=114；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2863。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5692971 | 0.5692971 |  |
| 0 | 42 | 0.3308308 | 65.9648901 | 1.0 | 0.1150053 | 1.1150053 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7124682 | 0.0750616 | 0.7875298 |  |

### N2：selection → action 6

path=[0]；visits=42；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2865。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.101541 | 0.101541 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7134582 | 0.0368934 | 0.7503516 |  |
| 6 | 5 | 0.0686902 | 66.4311235 | 1.0 | 0.1038714 | 1.1038714 | ✓ |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.8099427 | 0.1230752 | 0.933018 |  |
| 38 | 12 | 0.0717799 | 66.2254826 | 0.9411835 | 0.0500971 | 0.9912806 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.8185763 | 0.0863145 | 0.9048908 |  |
| 20 | 5 | 0.0589459 | 66.2133752 | 0.9377206 | 0.0891364 | 1.026857 |  |

### N15：selection → action 12

path=[0, 6]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [12]，并列按 prior 抽样。trace 行 2867。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.1118522 | 67.1452629 | 1.0 | 0.1167175 | 1.1167175 |  |
| 12 | 1 | 0.0299881 | 67.1398861 | 0.9978094 | 0.0469388 | 1.0447483 | ✓ |
| 26 | 2 | 0.0936646 | 64.6907758 | 0.0 | 0.0977389 | 0.0977389 |  |

### N32：expansion → action 5

path=[0, 6, 12]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 5 个代表动作。trace 行 2869。

bucket=0，compatibility_richness_prior；到达 N106（新建）；closure=[]。

## iteration 116

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[6, 34, 8, 19, 37, 29, 9, 22, 3, 38, 35, 39, 14]

### N0：selection → action 0

path=[]；visits=115；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2889。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5717886 | 0.5717886 |  |
| 0 | 43 | 0.3308308 | 65.8093416 | 1.0 | 0.1128834 | 1.1128834 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7418603 | 0.0753901 | 0.8172504 |  |

### N2：selection → action 20

path=[0]；visits=43；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2891。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1027427 | 0.1027427 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7580437 | 0.03733 | 0.7953737 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3220197 | 0.0900863 | 0.412106 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.8605577 | 0.1245318 | 0.9850895 |  |
| 38 | 12 | 0.0717799 | 66.2254826 | 1.0 | 0.05069 | 1.05069 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.8697308 | 0.087336 | 0.9570668 |  |
| 20 | 5 | 0.0589459 | 66.2133752 | 0.9963207 | 0.0901913 | 1.086512 | ✓ |

### N98：selection → action 27

path=[0, 20]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [4, 27]，并列按 prior 抽样。trace 行 2893。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 2 | 0.0544664 | 68.5903306 | 1.0 | 0.0568357 | 1.0568357 |  |
| 4 | 1 | 0.0073295 | 60.2222557 | 0.0 | 0.0114724 | 0.0114724 |  |
| 27 | 1 | 0.0554026 | 63.2252638 | 0.3588649 | 0.0867188 | 0.4455837 | ✓ |

### N102：expansion → action 5

path=[0, 20, 27]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 2895。

bucket=0，compatibility_richness_prior；到达 N107（新建）；closure=[]。

## iteration 117

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[27, 8, 24, 32, 23, 6, 34, 3, 10, 13, 38, 22, 35, 7]

### N0：selection → action 0

path=[]；visits=116；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2914。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5742692 | 0.5742692 |  |
| 0 | 44 | 0.3308308 | 66.2288461 | 1.0 | 0.1108538 | 1.1108538 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.6675852 | 0.0757172 | 0.7433024 |  |

### N2：selection → action 20

path=[0]；visits=44；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2916。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1039305 | 0.1039305 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.5291122 | 0.0377616 | 0.5668738 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2247688 | 0.0911278 | 0.3158966 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.6006667 | 0.1259715 | 0.7266382 |  |
| 38 | 12 | 0.0717799 | 66.2254826 | 0.697997 | 0.051276 | 0.749273 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.6070695 | 0.0883457 | 0.6954152 |  |
| 20 | 6 | 0.0589459 | 67.6492608 | 1.0 | 0.0782006 | 1.0782006 | ✓ |

### N98：selection → action 4

path=[0, 20]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [4]，并列按 prior 抽样。trace 行 2918。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 2 | 0.0544664 | 68.5903306 | 0.9504078 | 0.0622603 | 1.0126681 |  |
| 4 | 1 | 0.0073295 | 60.2222557 | 0.0 | 0.0125674 | 0.0125674 | ✓ |
| 27 | 2 | 0.0554026 | 69.0269762 | 1.0 | 0.0633305 | 1.0633305 |  |

### N100：expansion → action 9

path=[0, 20, 4]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 2920。

bucket=0，compatibility_richness_prior；到达 N108（新建）；closure=[]。

## iteration 118

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[29, 5, 12, 10, 3, 38, 17, 34, 9, 22, 28, 19, 2]

### N0：selection → action 0

path=[]；visits=117；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2940。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5767392 | 0.5767392 |  |
| 0 | 45 | 0.3308308 | 65.8638235 | 1.0 | 0.1089103 | 1.1089103 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7312935 | 0.0760429 | 0.8073364 |  |

### N2：selection → action 20

path=[0]；visits=45；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2942。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1051049 | 0.1051049 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7472405 | 0.0381883 | 0.7854288 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3174304 | 0.0921575 | 0.409588 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.8482936 | 0.127395 | 0.9756885 |  |
| 38 | 12 | 0.0717799 | 66.2254826 | 0.9857486 | 0.0518554 | 1.037604 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.8573359 | 0.089344 | 0.9466799 |  |
| 20 | 7 | 0.0589459 | 66.2730572 | 1.0 | 0.0691987 | 1.0691987 | ✓ |

### N98：selection → action 27

path=[0, 20]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [19, 4, 27]，并列按 prior 抽样。trace 行 2944。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 2 | 0.0544664 | 68.5903306 | 0.9559297 | 0.0672488 | 1.0231785 |  |
| 4 | 2 | 0.0073295 | 59.1190459 | 0.0 | 0.0090496 | 0.0090496 |  |
| 27 | 2 | 0.0554026 | 69.0269762 | 1.0 | 0.0684047 | 1.0684047 | ✓ |

### N102：expansion → action 37

path=[0, 20, 27]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 2946。

bucket=1，uniform_random；到达 N109（新建）；closure=[]。

## iteration 119

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 20, 3, 9, 5, 33, 35, 38, 27, 34, 13, 7, 15, 14, 4]

### N0：selection → action 0

path=[]；visits=118；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2965。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5791987 | 0.5791987 |  |
| 0 | 46 | 0.3308308 | 65.5703803 | 1.0 | 0.1070477 | 1.1070477 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7920581 | 0.0763672 | 0.8684253 |  |

### N2：selection → action 38

path=[0]；visits=46；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2967。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1062663 | 0.1062663 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7580437 | 0.0386103 | 0.7966539 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3220197 | 0.0931759 | 0.4151956 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.8605577 | 0.1288027 | 0.9893604 |  |
| 38 | 12 | 0.0717799 | 66.2254826 | 1.0 | 0.0524284 | 1.0524284 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.8697308 | 0.0903313 | 0.9600621 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.7369629 | 0.0621897 | 0.7991526 |  |

### N4：selection → action 19

path=[24, 22]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [4, 19]，并列按 prior 抽样。trace 行 2969。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0522202 | 0.0522202 |  |
| 4 | 2 | 0.0056816 | 69.5752255 | 0.1395575 | 0.0091848 | 0.1487423 |  |
| 19 | 2 | 0.0477591 | 126.8897825 | 1.0 | 0.0772065 | 1.0772065 | ✓ |
| 6 | 3 | 0.0365193 | 62.2773178 | 0.0299966 | 0.0442773 | 0.0742739 |  |

### N80：expansion → action 37

path=[24, 22, 19]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 2971。

bucket=1，uniform_random；到达 N110（新建）；closure=[]。

## iteration 120

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[23, 38, 25, 1, 21, 0, 36, 6, 10, 13, 8, 9, 30]

### N0：selection → action 0

path=[]；visits=119；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2992。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5816477 | 0.5816477 |  |
| 0 | 47 | 0.3308308 | 65.8115334 | 1.0 | 0.1052607 | 1.1052607 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7414293 | 0.0766901 | 0.8181194 |  |

### N2：selection → action 38

path=[0]；visits=47；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 2994。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1074151 | 0.1074151 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.6231566 | 0.0390277 | 0.6621843 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2647192 | 0.0941832 | 0.3589024 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.7074292 | 0.1301952 | 0.8376244 |  |
| 38 | 13 | 0.0717799 | 66.9377741 | 1.0 | 0.0492098 | 1.0492098 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.71497 | 0.0913079 | 0.8062779 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.605827 | 0.062862 | 0.668689 |  |

### N4：selection → action 4

path=[24, 22]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [4]，并列按 prior 抽样。trace 行 2996。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0543525 | 0.0543525 |  |
| 4 | 2 | 0.0056816 | 69.5752255 | 0.3811002 | 0.0095598 | 0.39066 | ✓ |
| 19 | 3 | 0.0477591 | 84.6717634 | 1.0 | 0.0602693 | 1.0602693 |  |
| 6 | 3 | 0.0365193 | 62.2773178 | 0.0819141 | 0.0460852 | 0.1279993 |  |

### N67：expansion → action 39

path=[24, 22, 4]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 2998。

bucket=1，uniform_random；到达 N111（新建）；closure=[]。

## iteration 121

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 39, 1, 21, 8, 7, 33, 23, 16, 31, 19, 34, 9]

### N0：selection → action 0

path=[]；visits=120；children=3；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3017。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5840865 | 0.5840865 |  |
| 0 | 48 | 0.3308308 | 65.8567965 | 1.0 | 0.1035449 | 1.1035449 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7326395 | 0.0770116 | 0.8096511 |  |

### N2：selection → action 38

path=[0]；visits=48；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3019。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1085518 | 0.1085518 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.6226372 | 0.0394407 | 0.6620778 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2644985 | 0.0951799 | 0.3596784 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.7068395 | 0.1315729 | 0.8384125 |  |
| 38 | 14 | 0.0717799 | 66.9411135 | 1.0 | 0.0464152 | 1.0464152 | ✓ |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.714374 | 0.0922741 | 0.8066482 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.605322 | 0.0635272 | 0.6688492 |  |

### N4：selection → action 6

path=[24, 22]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 4, 19, 6]，并列按 prior 抽样。trace 行 3021。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0564042 | 0.0564042 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.3292317 | 0.0074405 | 0.3366722 |  |
| 19 | 3 | 0.0477591 | 84.6717634 | 1.0 | 0.0625444 | 1.0625444 |  |
| 6 | 3 | 0.0365193 | 62.2773178 | 0.0819141 | 0.0478249 | 0.129739 | ✓ |

### N96：selection → action 26

path=[24, 22, 6]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [35, 26]，并列按 prior 抽样。trace 行 3023。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 35 | 1 | 0.0387423 | 60.2039803 | 1.0 | 0.0469726 | 1.0469726 |  |
| 26 | 1 | 0.0317325 | 56.3804535 | 0.0 | 0.0384736 | 0.0384736 | ✓ |

### N105：expansion → action 2

path=[24, 22, 6, 26]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3025。

bucket=0，compatibility_richness_prior；到达 N112（新建）；closure=[]。

## iteration 122

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 29, 30, 5, 35, 22, 11, 38, 25, 10, 18, 23, 6, 28, 33, 4]

### N0：selection → action 0

path=[]；visits=121；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3044。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5865152 | 0.5865152 |  |
| 0 | 49 | 0.3308308 | 65.4457662 | 1.0 | 0.1018959 | 1.1018959 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.8210289 | 0.0773318 | 0.8983607 |  |

### N2：expansion → action 24

path=[0]；visits=49；children=7；K=8。已有 7 条动作边 < K=8，且尚余 10 个代表动作。trace 行 3046。

bucket=1，uniform_random；到达 N113（新建）；closure=[]。

## iteration 123

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 17, 25, 34, 19, 20, 38, 23, 16, 22, 28, 33, 15, 9, 1]

### N0：selection → action 0

path=[]；visits=122；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3068。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5889338 | 0.5889338 |  |
| 0 | 50 | 0.3308308 | 65.5104578 | 1.0 | 0.1003099 | 1.1003099 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.8057295 | 0.0776507 | 0.8833803 |  |

### N2：selection → action 24

path=[0]；visits=50；children=8；K=8。最低访问优先：child.visits < 5；最少 1 次；候选 [24]，并列按 prior 抽样。trace 行 3070。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1107903 | 0.1107903 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.5814656 | 0.040254 | 0.6217196 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2470087 | 0.0971426 | 0.3441512 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.6601001 | 0.1342861 | 0.7943862 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.6360868 | 0.0444116 | 0.6804984 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.6671364 | 0.0941769 | 0.7613133 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.5652953 | 0.0648372 | 0.6301326 |  |
| 24 | 1 | 0.0739182 | 67.2247861 | 1.0 | 0.3658765 | 1.3658765 | ✓ |

### N113：expansion → action 27

path=[0, 24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 28 个代表动作。trace 行 3072。

bucket=0，compatibility_richness_prior；到达 N114（新建）；closure=[]。

## iteration 124

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[31, 17, 14, 9, 12, 3, 10, 37, 1, 33, 18, 13, 34, 5, 8]

### N0：selection → action 0

path=[]；visits=123；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3093。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5913425 | 0.5913425 |  |
| 0 | 51 | 0.3308308 | 65.5381626 | 1.0 | 0.0987832 | 1.0987832 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7993504 | 0.0779683 | 0.8773188 |  |

### N2：selection → action 24

path=[0]；visits=51；children=8；K=8。最低访问优先：child.visits < 5；最少 2 次；候选 [24]，并列按 prior 抽样。trace 行 3095。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1118927 | 0.1118927 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.6517061 | 0.0406545 | 0.6923606 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2768471 | 0.0981092 | 0.3749563 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.7398396 | 0.1356223 | 0.8754619 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.7129255 | 0.0448535 | 0.757779 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.7477259 | 0.095114 | 0.8428398 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.6335825 | 0.0654824 | 0.6990649 |  |
| 24 | 2 | 0.0739182 | 66.7624149 | 1.0 | 0.2463448 | 1.2463448 | ✓ |

### N113：expansion → action 28

path=[0, 24]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 27 个代表动作。trace 行 3097。

bucket=1，uniform_random；到达 N115（新建）；closure=[]。

## iteration 125

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[37, 33, 12, 29, 20, 10, 30, 22, 31, 19, 16, 27, 5]

### N0：selection → action 0

path=[]；visits=124；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3118。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5937415 | 0.5937415 |  |
| 0 | 52 | 0.3308308 | 65.6476279 | 1.0 | 0.0973126 | 1.0973126 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7751038 | 0.0782846 | 0.8533884 |  |

### N2：selection → action 24

path=[0]；visits=52；children=8；K=8。最低访问优先：child.visits < 5；最少 3 次；候选 [24]，并列按 prior 抽样。trace 行 3120。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1129843 | 0.1129843 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.5548292 | 0.0410512 | 0.5958804 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2356934 | 0.0990664 | 0.3347598 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.6298615 | 0.1369455 | 0.766807 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.6069483 | 0.0452911 | 0.6522394 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.6365755 | 0.0960419 | 0.7326175 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.5393997 | 0.0661212 | 0.6055209 |  |
| 24 | 3 | 0.0739182 | 67.4307404 | 1.0 | 0.1865612 | 1.1865612 | ✓ |

### N113：selection → action 28

path=[0, 24]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [27, 28]，并列按 prior 抽样。trace 行 3122。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 1 | 0.0443344 | 66.3000438 | 0.0 | 0.0537527 | 0.0537527 |  |
| 28 | 1 | 0.0441969 | 68.7673912 | 1.0 | 0.0535859 | 1.0535859 | ✓ |

### N115：expansion → action 14

path=[0, 24, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 3124。

bucket=0，compatibility_richness_prior；到达 N116（新建）；closure=[]。

## iteration 126

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[19, 27, 22, 5, 37, 12, 20, 25, 39, 33, 11, 35, 32, 10, 3]

### N0：selection → action 0

path=[]；visits=125；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3143。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5961308 | 0.5961308 |  |
| 0 | 53 | 0.3308308 | 65.7617154 | 1.0 | 0.0958948 | 1.0958948 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7513508 | 0.0785996 | 0.8299504 |  |

### N2：selection → action 24

path=[0]；visits=53；children=8；K=8。最低访问优先：child.visits < 5；最少 4 次；候选 [24]，并列按 prior 抽样。trace 行 3145。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1140656 | 0.1140656 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.5069997 | 0.041444 | 0.5484437 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2153753 | 0.1000144 | 0.3153897 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.5755638 | 0.138256 | 0.7138198 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.5546258 | 0.0457245 | 0.6003503 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.581699 | 0.096961 | 0.67866 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.4929003 | 0.066754 | 0.5596543 |  |
| 24 | 4 | 0.0739182 | 67.8548792 | 1.0 | 0.1506772 | 1.1506772 | ✓ |

### N113：expansion → action 36

path=[0, 24]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 26 个代表动作。trace 行 3147。

bucket=0，compatibility_richness_prior；到达 N117（新建）；closure=[]。

## iteration 127

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[11, 38, 23, 25, 21, 15, 4, 1, 32, 33, 36, 31, 35, 12, 10]

### N0：selection → action 0

path=[]；visits=126；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3168。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.5985106 | 0.5985106 |  |
| 0 | 54 | 0.3308308 | 65.5839101 | 1.0 | 0.0945272 | 1.0945272 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7890352 | 0.0789134 | 0.8679486 |  |

### N2：selection → action 24

path=[0]；visits=54；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3170。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1151366 | 0.1151366 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7377343 | 0.0418332 | 0.7795674 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3133922 | 0.1009535 | 0.4143457 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.8375018 | 0.1395542 | 0.977056 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.8070349 | 0.0461538 | 0.8531888 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.8464291 | 0.0978715 | 0.9443006 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.7172183 | 0.0673808 | 0.7845991 |  |
| 24 | 5 | 0.0739182 | 66.3160729 | 1.0 | 0.1267433 | 1.1267433 | ✓ |

### N113：selection → action 27

path=[0, 24]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [27, 36]，并列按 prior 抽样。trace 行 3172。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 1 | 0.0443344 | 66.3000438 | 0.6987081 | 0.0693944 | 0.7681025 | ✓ |
| 28 | 2 | 0.0441969 | 68.9473435 | 1.0 | 0.0461194 | 1.0461194 |  |
| 36 | 1 | 0.0083742 | 60.1608476 | 0.0 | 0.0131078 | 0.0131078 |  |

### N114：expansion → action 30

path=[0, 24, 27]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 3174。

bucket=0，compatibility_richness_prior；到达 N118（新建）；closure=[]。

## iteration 128

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 37, 19, 38, 23, 12, 17, 30, 26, 25, 32, 35, 9, 4]

### N0：selection → action 0

path=[]；visits=127；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3195。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6008809 | 0.6008809 |  |
| 0 | 55 | 0.3308308 | 65.5554146 | 1.0 | 0.0932069 | 1.0932069 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.7954289 | 0.0792259 | 0.8746548 |  |

### N2：selection → action 24

path=[0]；visits=55；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3197。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1161978 | 0.1161978 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8034027 | 0.0422187 | 0.8456215 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3412884 | 0.101884 | 0.4431724 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.912051 | 0.1408404 | 1.0528914 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.8788721 | 0.0465792 | 0.9254514 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 0.9217729 | 0.0987735 | 1.0205465 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.7810606 | 0.0680018 | 0.8490624 |  |
| 24 | 6 | 0.0739182 | 66.0396954 | 1.0 | 0.1096384 | 1.1096384 | ✓ |

### N113：selection → action 36

path=[0, 24]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [36]，并列按 prior 抽样。trace 行 3199。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 2 | 0.0443344 | 65.4789259 | 0.6052559 | 0.0506785 | 0.6559344 |  |
| 28 | 2 | 0.0441969 | 68.9473435 | 1.0 | 0.0505213 | 1.0505213 |  |
| 36 | 1 | 0.0083742 | 60.1608476 | 0.0 | 0.0143588 | 0.0143588 | ✓ |

### N117：expansion → action 22

path=[0, 24, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 3201。

bucket=0，compatibility_richness_prior；到达 N119（新建）；closure=[]。

## iteration 129

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[38, 7, 28, 3, 17, 14, 12, 31, 34, 29, 27, 23, 24, 6]

### N0：selection → action 0

path=[]；visits=128；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3221。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6032419 | 0.6032419 |  |
| 0 | 56 | 0.3308308 | 65.3067415 | 1.0 | 0.0919315 | 1.0919315 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.8559574 | 0.0795372 | 0.9354947 |  |

### N2：selection → action 19

path=[0]；visits=56；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3223。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0 | 0.1172494 | 0.1172494 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8715842 | 0.0426008 | 0.914185 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.3702521 | 0.102806 | 0.4730582 |  |
| 19 | 5 | 0.0813897 | 65.7666238 | 0.989453 | 0.142115 | 1.131568 | ✓ |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.9534584 | 0.0470008 | 1.0004592 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 1.0 | 0.0996674 | 1.0996674 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.847346 | 0.0686172 | 0.9159632 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.6448722 | 0.0968018 | 0.741674 |  |

### N55：selection → action 9

path=[0, 19]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [9, 26]，并列按 prior 抽样。trace 行 3225。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 10 | 2 | 0.0654586 | 61.4191349 | 0.0 | 0.0683059 | 0.0683059 |  |
| 9 | 1 | 0.0684216 | 75.4781289 | 1.0 | 0.1070968 | 1.1070968 | ✓ |
| 26 | 1 | 0.0529814 | 63.3225327 | 0.1353865 | 0.082929 | 0.2183155 |  |

### N57：expansion → action 33

path=[0, 19, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 22 个代表动作。trace 行 3227。

bucket=0，compatibility_richness_prior；到达 N120（新建）；closure=[]。

## iteration 130

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 6, 29, 15, 37, 24, 2, 28, 12, 36, 25, 13, 3, 11]

### N0：selection → action 0

path=[]；visits=129；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3247。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6055938 | 0.6055938 |  |
| 0 | 57 | 0.3308308 | 65.1403641 | 1.0 | 0.0906987 | 1.0906987 | ✓ |
| 2 | 64 | 0.3264006 | 64.8360234 | 0.9018741 | 0.0798473 | 0.9817214 |  |

### N2：selection → action 26

path=[0]；visits=57；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3249。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3125903 | 0.1182916 | 0.430882 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.9117258 | 0.0429795 | 0.9547053 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5671052 | 0.1037199 | 0.6708251 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1228957 | 0.1228957 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 0.9680069 | 0.0474186 | 1.0154254 |  |
| 26 | 5 | 0.0570798 | 65.7968093 | 1.0 | 0.1005534 | 1.1005534 | ✓ |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.8950642 | 0.0692272 | 0.9642913 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7558817 | 0.0976623 | 0.853544 |  |

### N87：selection → action 38

path=[0, 26]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [38]，并列按 prior 抽样。trace 行 3251。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 29 | 2 | 0.0445819 | 56.0922733 | 0.0 | 0.0465211 | 0.0465211 |  |
| 38 | 1 | 0.0084801 | 70.2631418 | 1.0 | 0.0132734 | 1.0132734 | ✓ |
| 6 | 2 | 0.0456667 | 68.3789061 | 0.8670346 | 0.0476531 | 0.9146877 |  |

### N89：expansion → action 8

path=[0, 26, 38]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 3253。

bucket=0，compatibility_richness_prior；到达 N121（新建）；closure=[]。

## iteration 131

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 35, 4, 16, 37, 1, 13, 5, 15, 30, 10, 31, 33, 24, 7, 6]

### N0：selection → action 2

path=[]；visits=130；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3273。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6079365 | 0.6079365 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.971799 | 0.0895063 | 1.0613053 |  |
| 2 | 64 | 0.3264006 | 64.8360234 | 1.0 | 0.0801562 | 1.0801562 | ✓ |

### N5：expansion → action 3

path=[2]；visits=64；children=8；K=9。已有 8 条动作边 < K=9，且尚余 6 个代表动作。trace 行 3275。

bucket=0，compatibility_richness_prior；到达 N122（新建）；closure=[]。

## iteration 132

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 15, 0, 37, 24, 32, 29, 26, 30, 34, 25, 18, 14, 23, 10]

### N0：selection → action 2

path=[]；visits=131；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3297。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6102702 | 0.6102702 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.7335444 | 0.0898499 | 0.8233943 |  |
| 2 | 65 | 0.3264006 | 65.7445486 | 1.0 | 0.0792448 | 1.0792448 | ✓ |

### N5：selection → action 3

path=[2]；visits=65；children=9；K=9。最低访问优先：child.visits < 5；最少 1 次；候选 [3]，并列按 prior 抽样。trace 行 3299。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.3399406 | 0.1386723 | 0.478613 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.5593948 | 0.0474625 | 0.6068573 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.4458174 | 0.1622379 | 0.6080552 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.0763448 | 0.1316262 | 0.207971 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1111151 | 0.1111151 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.4560053 | 0.1105125 | 0.5665178 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.5761208 | 0.0952604 | 0.6713812 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 0.5881384 | 0.0362269 | 0.6243653 |  |
| 3 | 1 | 0.0462906 | 71.9338763 | 1.0 | 0.2612448 | 1.2612448 | ✓ |

### N122：expansion → action 6

path=[2, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 8 个代表动作。trace 行 3301。

bucket=0，compatibility_richness_prior；到达 N123（新建）；closure=[]。

## iteration 133

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[37, 8, 4, 35, 36, 6, 1, 13, 27, 34, 9, 33, 30, 28, 10]

### N0：selection → action 2

path=[]；visits=132；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3322。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6125951 | 0.6125951 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.7389187 | 0.0901922 | 0.8291109 |  |
| 2 | 66 | 0.3264006 | 65.7175964 | 1.0 | 0.0783594 | 1.0783594 | ✓ |

### N5：selection → action 3

path=[2]；visits=66；children=9；K=9。最低访问优先：child.visits < 5；最少 2 次；候选 [3]，并列按 prior 抽样。trace 行 3324。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5048027 | 0.139735 | 0.6445377 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.8306862 | 0.0478262 | 0.8785124 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.6620268 | 0.1634811 | 0.8255078 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1133699 | 0.1326349 | 0.2460048 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1119666 | 0.1119666 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.6771556 | 0.1113594 | 0.788515 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.8555238 | 0.0959904 | 0.9515142 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 0.8733696 | 0.0365045 | 0.9098741 |  |
| 3 | 2 | 0.0462906 | 68.7204546 | 1.0 | 0.1754978 | 1.1754978 | ✓ |

### N122：expansion → action 18

path=[2, 3]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 7 个代表动作。trace 行 3326。

bucket=1，uniform_random；到达 N124（新建）；closure=[]。

## iteration 134

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[36, 6, 11, 37, 1, 4, 16, 8, 33, 9, 35, 27, 28, 0]

### N0：selection → action 2

path=[]；visits=133；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3347。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6149112 | 0.6149112 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.7555061 | 0.0905332 | 0.8460393 |  |
| 2 | 67 | 0.3264006 | 65.6368278 | 1.0 | 0.077499 | 1.077499 | ✓ |

### N5：selection → action 3

path=[2]；visits=67；children=9；K=9。最低访问优先：child.visits < 5；最少 3 次；候选 [3]，并列按 prior 抽样。trace 行 3349。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5779944 | 0.1407896 | 0.7187839 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.9511279 | 0.0481871 | 0.999315 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.7580144 | 0.1647149 | 0.9227293 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1298075 | 0.1336359 | 0.2634434 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1128116 | 0.1128116 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.7753368 | 0.1121998 | 0.8875367 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.9795667 | 0.0967148 | 1.0762816 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 1.0 | 0.03678 | 1.03678 |  |
| 3 | 3 | 0.0462906 | 67.4553212 | 0.9263715 | 0.1326168 | 1.0589883 | ✓ |

### N122：selection → action 18

path=[2, 3]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 18]，并列按 prior 抽样。trace 行 3351。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.143538 | 65.5070328 | 1.0 | 0.1740306 | 1.1740306 |  |
| 18 | 1 | 0.1242323 | 64.9250546 | 0.0 | 0.1506237 | 0.1506237 | ✓ |

### N124：expansion → action 38

path=[2, 3, 18]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 16 个代表动作。trace 行 3353。

bucket=0，compatibility_richness_prior；到达 N125（新建）；closure=[]。

## iteration 135

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 30, 11, 4, 15, 29, 38, 35, 25, 39, 31, 34, 33, 14, 12, 18]

### N0：selection → action 2

path=[]；visits=134；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3373。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6172185 | 0.6172185 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.7154207 | 0.0908729 | 0.8062936 |  |
| 2 | 68 | 0.3264006 | 65.8384251 | 1.0 | 0.0766624 | 1.0766624 | ✓ |

### N5：selection → action 3

path=[2]；visits=68；children=9；K=9。最低访问优先：child.visits < 5；最少 4 次；候选 [3]，并列按 prior 抽样。trace 行 3375。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5779944 | 0.1418363 | 0.7198307 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.9511279 | 0.0485454 | 0.9996733 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.7580144 | 0.1659396 | 0.923954 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1298075 | 0.1346295 | 0.264437 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1136504 | 0.1136504 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.7753368 | 0.113034 | 0.8883709 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.9795667 | 0.0974339 | 1.0770006 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 1.0 | 0.0370535 | 1.0370535 |  |
| 3 | 4 | 0.0462906 | 67.5456406 | 0.941979 | 0.1068822 | 1.0488612 | ✓ |

### N122：expansion → action 8

path=[2, 3]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 6 个代表动作。trace 行 3377。

bucket=0，compatibility_richness_prior；到达 N126（新建）；closure=[]。

## iteration 136

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 13, 36, 11, 37, 18, 38, 16, 1, 15, 35, 4, 0, 7]

### N0：selection → action 2

path=[]；visits=135；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3399。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6195173 | 0.6195173 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.5957015 | 0.0912114 | 0.6869129 |  |
| 2 | 69 | 0.3264006 | 66.6020363 | 1.0 | 0.0758486 | 1.0758486 | ✓ |

### N5：selection → action 3

path=[2]；visits=69；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3401。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.4837911 | 0.1428755 | 0.6266665 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.7961102 | 0.048901 | 0.8450112 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.6344709 | 0.1671553 | 0.8016262 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1086511 | 0.1356158 | 0.2442669 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.114483 | 0.114483 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.6489701 | 0.1138621 | 0.7628322 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.819914 | 0.0981477 | 0.9180617 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 0.837017 | 0.037325 | 0.8743419 |  |
| 3 | 5 | 0.0462906 | 69.0082288 | 1.0 | 0.089721 | 1.089721 | ✓ |

### N122：selection → action 6

path=[2, 3]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 8]，并列按 prior 抽样。trace 行 3403。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.143538 | 65.5070328 | 0.0 | 0.2246725 | 0.2246725 | ✓ |
| 18 | 2 | 0.1242323 | 66.3708267 | 0.0923691 | 0.1296362 | 0.2220053 |  |
| 8 | 1 | 0.1439711 | 74.8585816 | 1.0 | 0.2253504 | 1.2253504 |  |

### N123：expansion → action 32

path=[2, 3, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 22 个代表动作。trace 行 3405。

bucket=0，compatibility_richness_prior；到达 N127（新建）；closure=[]。

## iteration 137

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[6, 15, 5, 35, 11, 18, 27, 17, 21, 10, 32, 12, 7]

### N0：selection → action 2

path=[]；visits=136；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3425。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6218076 | 0.6218076 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.6101184 | 0.0915485 | 0.701667 |  |
| 2 | 70 | 0.3264006 | 66.4942092 | 1.0 | 0.0750568 | 1.0750568 | ✓ |

### N5：selection → action 3

path=[2]；visits=70；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3427。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5324044 | 0.1439071 | 0.6763114 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.8761066 | 0.0492541 | 0.9253607 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.6982251 | 0.1683622 | 0.8665873 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1195688 | 0.136595 | 0.2561638 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1153096 | 0.1153096 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.7141812 | 0.1146843 | 0.8288655 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.9023022 | 0.0988564 | 1.0011586 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 0.9211238 | 0.0375945 | 0.9587183 |  |
| 3 | 6 | 0.0462906 | 68.3769411 | 1.0 | 0.077459 | 1.077459 | ✓ |

### N122：selection → action 8

path=[2, 3]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [8]，并列按 prior 抽样。trace 行 3429。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.143538 | 65.3637676 | 0.0 | 0.1640776 | 0.1640776 |  |
| 18 | 2 | 0.1242323 | 66.3708267 | 0.1060641 | 0.1420094 | 0.2480735 |  |
| 8 | 1 | 0.1439711 | 74.8585816 | 1.0 | 0.246859 | 1.246859 | ✓ |

### N126：expansion → action 36

path=[2, 3, 8]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 3431。

bucket=0，compatibility_richness_prior；到达 N128（新建）；closure=[]。

## iteration 138

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[17, 30, 32, 11, 4, 27, 13, 12, 6, 34, 16, 22, 9]

### N0：selection → action 2

path=[]；visits=137；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3450。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6240894 | 0.6240894 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.5460497 | 0.0918845 | 0.6379342 |  |
| 2 | 71 | 0.3264006 | 67.0169648 | 1.0 | 0.074286 | 1.074286 | ✓ |

### N5：selection → action 3

path=[2]；visits=71；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3452。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.4747773 | 0.1449313 | 0.6197086 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.7812775 | 0.0496047 | 0.8308821 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.6226498 | 0.1695605 | 0.7922103 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1066267 | 0.1375672 | 0.2441939 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1161303 | 0.1161303 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.6368788 | 0.1155005 | 0.7523793 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.8046377 | 0.09956 | 0.9041977 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 0.8214221 | 0.037862 | 0.8592841 |  |
| 3 | 7 | 0.0462906 | 69.1394881 | 1.0 | 0.068259 | 1.068259 | ✓ |

### N122：selection → action 18

path=[2, 3]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [6, 18, 8]，并列按 prior 抽样。trace 行 3454。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.143538 | 65.3637676 | 0.0 | 0.1772241 | 0.1772241 |  |
| 18 | 2 | 0.1242323 | 66.3708267 | 0.1128622 | 0.1533877 | 0.2662499 | ✓ |
| 8 | 2 | 0.1439711 | 74.2866761 | 1.0 | 0.1777588 | 1.1777588 |  |

### N124：expansion → action 26

path=[2, 3, 18]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 15 个代表动作。trace 行 3456。

bucket=1，uniform_random；到达 N129（新建）；closure=[]。

## iteration 139

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[25, 9, 4, 26, 7, 33, 19, 11, 23, 10, 5, 12, 6]

### N0：selection → action 2

path=[]；visits=138；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3475。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.626363 | 0.626363 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.7689567 | 0.0922192 | 0.8611759 |  |
| 2 | 72 | 0.3264006 | 65.5738915 | 1.0 | 0.0735353 | 1.0735353 | ✓ |

### N5：selection → action 36

path=[2]；visits=72；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3477。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5779944 | 0.1459484 | 0.7239428 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.9511279 | 0.0499528 | 1.0010807 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.7580144 | 0.1707504 | 0.9287648 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1298075 | 0.1385326 | 0.2683401 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1169453 | 0.1169453 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.7753368 | 0.1163111 | 0.8916479 |  |
| 36 | 13 | 0.118156 | 67.7631582 | 0.9795667 | 0.1002587 | 1.0798254 | ✓ |
| 24 | 7 | 0.0256766 | 67.8814041 | 1.0 | 0.0381277 | 1.0381277 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.7098337 | 0.0611005 | 0.7709342 |  |

### N47：selection → action 8

path=[2, 36]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [8]，并列按 prior 抽样。trace 行 3479。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.4207954 | 0.0607936 | 0.4815889 |  |
| 8 | 3 | 0.059354 | 68.9914085 | 0.7609159 | 0.0749014 | 0.8358173 | ✓ |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0493947 | 1.0493947 |  |
| 39 | 4 | 0.0880029 | 64.144871 | 0.0 | 0.0888437 | 0.0888437 |  |

### N49：selection → action 21

path=[2, 36, 8]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [21, 22]，并列按 prior 抽样。trace 行 3481。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 1 | 0.042367 | 71.6874384 | 1.0 | 0.0513673 | 1.0513673 | ✓ |
| 22 | 1 | 0.0432386 | 70.9552173 | 0.0 | 0.052424 | 0.052424 |  |

### N50：expansion → action 32

path=[2, 36, 8, 21]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 3483。

bucket=0，compatibility_richness_prior；到达 N130（新建）；closure=[]。

## iteration 140

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[31, 0, 20, 25, 28, 3, 39, 10, 9, 21, 23, 29, 18]

### N0：selection → action 2

path=[]；visits=139；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3502。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6286283 | 0.6286283 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.8114004 | 0.0925528 | 0.9039531 |  |
| 2 | 73 | 0.3264006 | 65.3889756 | 1.0 | 0.0728039 | 1.0728039 | ✓ |

### N5：selection → action 24

path=[2]；visits=73；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3504。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.5779944 | 0.1469584 | 0.7249528 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 0.9511279 | 0.0502985 | 1.0014264 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.7580144 | 0.1719321 | 0.9299465 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1298075 | 0.1394913 | 0.2692988 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1177546 | 0.1177546 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.7753368 | 0.117116 | 0.8924528 |  |
| 36 | 14 | 0.118156 | 64.9420543 | 0.4920709 | 0.0942223 | 0.5862933 |  |
| 24 | 7 | 0.0256766 | 67.8814041 | 1.0 | 0.0383916 | 1.0383916 | ✓ |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.7098337 | 0.0615233 | 0.7713571 |  |

### N68：selection → action 8

path=[2, 24]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [25, 8]，并列按 prior 抽样。trace 行 3506。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.0573819 | 65.4679187 | 0.3437721 | 0.0531364 | 0.3969085 |  |
| 25 | 2 | 0.0655241 | 62.95859 | 0.0 | 0.0809015 | 0.0809015 |  |
| 8 | 2 | 0.060259 | 70.2579868 | 1.0 | 0.0744008 | 1.0744008 | ✓ |

### N71：expansion → action 27

path=[2, 24, 8]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 3508。

bucket=1，uniform_random；到达 N131（新建）；closure=[]。

## iteration 141

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[5, 14, 13, 16, 1, 20, 29, 32, 4, 34, 22, 23]

### N0：selection → action 2

path=[]；visits=140；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3527。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6308855 | 0.6308855 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 0.8249411 | 0.0928851 | 0.9178262 |  |
| 2 | 74 | 0.3264006 | 65.3339859 | 1.0 | 0.0720911 | 1.0720911 | ✓ |

### N5：selection → action 37

path=[2]；visits=74；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3529。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.6076936 | 0.1479616 | 0.7556552 |  |
| 37 | 15 | 0.0672799 | 67.5985849 | 1.0 | 0.0506418 | 1.0506418 | ✓ |
| 0 | 5 | 0.086242 | 66.4810506 | 0.7969637 | 0.1731057 | 0.9700694 |  |
| 6 | 6 | 0.0816311 | 62.8456614 | 0.1364774 | 0.1404435 | 0.2769209 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.0 | 0.1185584 | 0.1185584 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 0.8151762 | 0.1179154 | 0.9330916 |  |
| 36 | 14 | 0.118156 | 64.9420543 | 0.5173551 | 0.0948655 | 0.6122206 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.6195236 | 0.0343588 | 0.6538824 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.7463073 | 0.0619433 | 0.8082506 |  |

### N7：selection → action 3

path=[2, 37]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [3]，并列按 prior 抽样。trace 行 3531。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 3 | 3 | 0.0622865 | 71.0454666 | 1.0 | 0.0844322 | 1.0844322 | ✓ |
| 36 | 4 | 0.0777 | 69.9020661 | 0.8856762 | 0.0842606 | 0.9699368 |  |
| 38 | 4 | 0.0500079 | 67.6152012 | 0.6570223 | 0.0542303 | 0.7112526 |  |
| 28 | 4 | 0.0589352 | 61.0440429 | 0.0 | 0.0639114 | 0.0639114 |  |

### N17：selection → action 21

path=[2, 37, 3]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [21]，并列按 prior 抽样。trace 行 3533。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 38 | 2 | 0.0713241 | 72.1838395 | 1.0 | 0.0576506 | 1.0576506 |  |
| 21 | 1 | 0.0040611 | 66.6203741 | 0.0 | 0.0049238 | 0.0049238 | ✓ |

### N41：expansion → action 0

path=[2, 37, 3, 21]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 3535。

bucket=0，compatibility_richness_prior；到达 N132（新建）；closure=[]。

## iteration 142

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[5, 3, 31, 27, 32, 20, 11, 38, 13, 39, 14, 2]

### N0：selection → action 0

path=[]；visits=141；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3553。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6331347 | 0.6331347 |  |
| 0 | 58 | 0.3308308 | 64.7571397 | 1.0 | 0.0932162 | 1.0932162 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9199005 | 0.0713962 | 0.9912967 |  |

### N2：selection → action 38

path=[0]；visits=58；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3555。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3229216 | 0.1193248 | 0.4422464 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.9418588 | 0.0433549 | 0.9852137 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5858484 | 0.1046258 | 0.6904741 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.123969 | 0.123969 |  |
| 38 | 15 | 0.0717799 | 65.6636073 | 1.0 | 0.0478327 | 1.0478327 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4433063 | 0.0869414 | 0.5302476 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.9246465 | 0.0698318 | 0.9944783 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.780864 | 0.0985153 | 0.8793793 |  |

### N4：selection → action 19

path=[24, 22]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 4, 19]，并列按 prior 抽样。trace 行 3557。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0583839 | 0.0583839 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.3292317 | 0.0077017 | 0.3369334 |  |
| 19 | 3 | 0.0477591 | 84.6717634 | 1.0 | 0.0647396 | 1.0647396 | ✓ |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0069647 | 0.0396028 | 0.0465675 |  |

### N80：selection → action 0

path=[24, 22, 19]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [0, 37]，并列按 prior 抽样。trace 行 3559。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0.0681384 | 139.557155 | 1.0 | 0.0826134 | 1.0826134 | ✓ |
| 37 | 1 | 0.0434014 | 71.4786324 | 0.0 | 0.0526214 | 0.0526214 |  |

### N86：expansion → action 9

path=[24, 22, 19, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 3561。

bucket=0，compatibility_richness_prior；到达 N133（新建）；closure=[]。

## iteration 143

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[31, 28, 3, 39, 38, 33, 37, 32, 0, 16, 25, 35, 23, 13, 15, 7]

### N0：selection → action 0

path=[]；visits=142；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3579。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6353759 | 0.6353759 |  |
| 0 | 59 | 0.3308308 | 64.9377336 | 1.0 | 0.0919871 | 1.0919871 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.8625931 | 0.0716489 | 0.934242 |  |

### N2：selection → action 38

path=[0]；visits=59；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3581。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2842708 | 0.120349 | 0.4046199 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8291268 | 0.043727 | 0.8728538 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5157276 | 0.1055239 | 0.6212515 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1250332 | 0.1250332 |  |
| 38 | 16 | 0.0717799 | 66.2115795 | 1.0 | 0.0454055 | 1.0454055 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3902465 | 0.0876877 | 0.4779342 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.8139747 | 0.0704312 | 0.8844059 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.6874016 | 0.0993609 | 0.7867626 |  |

### N4：expansion → action 9

path=[24, 22]；visits=16；children=4；K=5。已有 4 条动作边 < K=5，且尚余 24 个代表动作。trace 行 3583。

bucket=0，compatibility_richness_prior；到达 N134（新建）；closure=[]。

## iteration 144

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[21, 34, 33, 28, 14, 18, 16, 31, 17, 29, 3, 27, 19, 5]

### N0：selection → action 0

path=[]；visits=143；children=3；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3605。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6376092 | 0.6376092 |  |
| 0 | 60 | 0.3308308 | 65.2169583 | 1.0 | 0.0907971 | 1.0907971 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.7868072 | 0.0719007 | 0.8587079 |  |

### N2：selection → action 38

path=[0]；visits=60；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3607。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2416027 | 0.1213647 | 0.3629674 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7046777 | 0.044096 | 0.7487737 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.4383186 | 0.1064144 | 0.544733 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1260883 | 0.1260883 |  |
| 38 | 17 | 0.0717799 | 67.020114 | 1.0 | 0.0432448 | 1.0432448 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3316718 | 0.0884276 | 0.4200995 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.6917998 | 0.0710256 | 0.7628254 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.584225 | 0.1001994 | 0.6844244 |  |

### N4：selection → action 9

path=[24, 22]；visits=17；children=5；K=5。最低访问优先：child.visits < 5；最少 1 次；候选 [9]，并列按 prior 抽样。trace 行 3609。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0621544 | 0.0621544 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.4310387 | 0.0081991 | 0.4392378 |  |
| 19 | 4 | 0.0477591 | 78.91049 | 1.0 | 0.0551365 | 1.0551365 |  |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0091184 | 0.0421604 | 0.0512788 |  |
| 9 | 1 | 0.0409347 | 75.4086596 | 0.8120455 | 0.1181448 | 0.9301903 | ✓ |

### N134：expansion → action 6

path=[24, 22, 9]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 3611。

bucket=0，compatibility_richness_prior；到达 N135（新建）；closure=[]。

## iteration 145

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 26, 11, 19, 31, 38, 1, 32, 29, 36, 10, 16, 39, 2, 15]

### N0：selection → action 0

path=[]；visits=144；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3631。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6398347 | 0.6398347 |  |
| 0 | 61 | 0.3308308 | 65.2195495 | 1.0 | 0.0896445 | 1.0896445 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.7861662 | 0.0721517 | 0.8583179 |  |

### N2：selection → action 38

path=[0]；visits=61；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3633。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2479383 | 0.1223718 | 0.3703101 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7231565 | 0.044462 | 0.7676185 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.4498127 | 0.1072975 | 0.5571102 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1271347 | 0.1271347 |  |
| 38 | 18 | 0.0717799 | 66.882466 | 1.0 | 0.0413088 | 1.0413088 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3403693 | 0.0891615 | 0.4295308 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.7099409 | 0.071615 | 0.781556 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.5995452 | 0.101031 | 0.7005761 |  |

### N4：selection → action 9

path=[24, 22]；visits=18；children=5；K=5。最低访问优先：child.visits < 5；最少 2 次；候选 [9]，并列按 prior 抽样。trace 行 3635。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0639564 | 0.0639564 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.4310387 | 0.0084368 | 0.4394755 |  |
| 19 | 4 | 0.0477591 | 78.91049 | 1.0 | 0.056735 | 1.056735 |  |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0091184 | 0.0433827 | 0.052501 |  |
| 9 | 2 | 0.0409347 | 70.3626898 | 0.541212 | 0.0810466 | 0.6222587 | ✓ |

### N134：expansion → action 34

path=[24, 22, 9]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3637。

bucket=1，uniform_random；到达 N136（新建）；closure=[]。

## iteration 146

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[10, 28, 21, 0, 26, 8, 12, 5, 36, 31, 39, 2, 13]

### N0：selection → action 0

path=[]；visits=145；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3658。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6420525 | 0.6420525 |  |
| 0 | 62 | 0.3308308 | 65.1925789 | 1.0 | 0.0885274 | 1.0885274 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.7928894 | 0.0724018 | 0.8652912 |  |

### N2：selection → action 38

path=[0]；visits=62；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3660。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2579628 | 0.1233708 | 0.3813336 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.7523947 | 0.0448249 | 0.7972197 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.4679993 | 0.1081734 | 0.5761727 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1281726 | 0.1281726 |  |
| 38 | 19 | 0.0717799 | 66.6784841 | 1.0 | 0.0395637 | 1.0395637 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3541309 | 0.0898894 | 0.4440203 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.7386449 | 0.0721997 | 0.8108445 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.6237856 | 0.1018557 | 0.7256413 |  |

### N4：selection → action 9

path=[24, 22]；visits=19；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 4, 9]，并列按 prior 抽样。trace 行 3662。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0657089 | 0.0657089 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.4310387 | 0.008668 | 0.4397067 |  |
| 19 | 4 | 0.0477591 | 78.91049 | 1.0 | 0.0582896 | 1.0582896 |  |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0091184 | 0.0445715 | 0.0536898 |  |
| 9 | 3 | 0.0409347 | 68.293196 | 0.4301356 | 0.0624506 | 0.4925863 | ✓ |

### N134：selection → action 6

path=[24, 22, 9]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [6, 34]，并列按 prior 抽样。trace 行 3664。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 1 | 0.0279405 | 65.3167199 | 1.0 | 0.033876 | 1.033876 | ✓ |
| 34 | 1 | 0.0430273 | 64.1542084 | 0.0 | 0.0521679 | 0.0521679 |  |

### N135：expansion → action 33

path=[24, 22, 9, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3666。

bucket=0，compatibility_richness_prior；到达 N137（新建）；closure=[]。

## iteration 147

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[20, 16, 14, 13, 37, 3, 7, 33, 38, 9, 6, 12]

### N0：selection → action 0

path=[]；visits=146；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3685。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6442627 | 0.6442627 |  |
| 0 | 63 | 0.3308308 | 64.8837961 | 1.0 | 0.0874441 | 1.0874441 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.878947 | 0.072651 | 0.951598 |  |

### N2：selection → action 38

path=[0]；visits=63；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3687。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3196266 | 0.1243618 | 0.4439884 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.9322483 | 0.045185 | 0.9774333 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5798705 | 0.1090423 | 0.6889128 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1292021 | 0.1292021 |  |
| 38 | 20 | 0.0717799 | 65.7051549 | 1.0 | 0.0379824 | 1.0379824 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4387829 | 0.0906114 | 0.5293943 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.9152116 | 0.0727796 | 0.9879912 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7728963 | 0.1026738 | 0.8755701 |  |

### N4：selection → action 4

path=[24, 22]；visits=20；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [28, 4]，并列按 prior 抽样。trace 行 3689。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0674159 | 0.0674159 |  |
| 4 | 3 | 0.0056816 | 68.3100213 | 0.4310387 | 0.0088931 | 0.4399318 | ✓ |
| 19 | 4 | 0.0477591 | 78.91049 | 1.0 | 0.0598039 | 1.0598039 |  |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0091184 | 0.0457294 | 0.0548477 |  |
| 9 | 4 | 0.0409347 | 64.3916163 | 0.2207253 | 0.0512584 | 0.2719837 |  |

### N67：selection → action 1

path=[24, 22, 4]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [1, 39]，并列按 prior 抽样。trace 行 3691。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.0545898 | 70.6884225 | 1.0 | 0.0661866 | 1.0661866 | ✓ |
| 39 | 1 | 0.0373461 | 66.9657419 | 0.0 | 0.0452797 | 0.0452797 |  |

### N92：expansion → action 31

path=[24, 22, 4, 1]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 3693。

bucket=0，compatibility_richness_prior；到达 N138（新建）；closure=[]。

## iteration 148

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[8, 18, 25, 21, 29, 35, 39, 36, 26, 23, 17, 32, 34, 30, 7, 1, 4]

### N0：selection → action 0

path=[]；visits=147；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3711。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6464653 | 0.6464653 |  |
| 0 | 64 | 0.3308308 | 64.8683862 | 1.0 | 0.0863932 | 1.0863932 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.8837337 | 0.0728994 | 0.9566332 |  |

### N2：expansion → action 8

path=[0]；visits=64；children=8；K=9。已有 8 条动作边 < K=9，且尚余 9 个代表动作。trace 行 3713。

bucket=0，compatibility_richness_prior；到达 N20（复用）；closure=[]。

## iteration 149

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[1, 26, 8, 35, 29, 10, 7, 22, 36, 25, 33, 30, 32, 19]

### N0：selection → action 0

path=[]；visits=148；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3737。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6486604 | 0.6486604 |  |
| 0 | 65 | 0.3308308 | 65.1757709 | 1.0 | 0.0853731 | 1.0853731 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.7971378 | 0.0731469 | 0.8702848 |  |

### N2：selection → action 8

path=[0]；visits=65；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3739。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.0798494 | 0.1263203 | 0.2061697 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.2328951 | 0.0458966 | 0.2787917 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.1448638 | 0.1107596 | 0.2556233 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1312369 | 0.1312369 |  |
| 38 | 21 | 0.0717799 | 65.6101391 | 0.2439913 | 0.0368269 | 0.2808182 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.1096171 | 0.0920384 | 0.2016555 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.228639 | 0.0739258 | 0.3025648 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.1930856 | 0.1042909 | 0.2973765 |  |
| 8 | 7 | 0.0625545 | 77.9322339 | 1.0 | 0.0882578 | 1.0882578 | ✓ |

### N20：selection → action 4

path=[2, 6]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [4]，并列按 prior 抽样。trace 行 3741。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 2 | 0.0231677 | 61.7715071 | 0.0 | 0.0286048 | 0.0286048 |  |
| 38 | 2 | 0.0257828 | 65.5767431 | 1.0 | 0.0318336 | 1.0318336 |  |
| 4 | 1 | 0.0280851 | 64.4157022 | 0.6948833 | 0.0520143 | 0.7468976 | ✓ |

### N36：expansion → action 21

path=[2, 6, 4]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 3743。

bucket=0，compatibility_richness_prior；到达 N139（新建）；closure=[]。

## iteration 150

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[23, 8, 32, 16, 33, 11, 0, 13, 24, 37, 31, 29, 28, 4]

### N0：selection → action 0

path=[]；visits=149；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3763。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6508482 | 0.6508482 |  |
| 0 | 66 | 0.3308308 | 65.1919654 | 1.0 | 0.0843825 | 1.0843825 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.7930437 | 0.0733936 | 0.8664373 |  |

### N2：selection → action 8

path=[0]；visits=66；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3765。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.1266885 | 0.1272883 | 0.2539768 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.3695098 | 0.0462483 | 0.4157581 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.2298399 | 0.1116083 | 0.3414482 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1322425 | 0.1322425 |  |
| 38 | 21 | 0.0717799 | 65.6101391 | 0.3871149 | 0.0371091 | 0.424224 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.1739178 | 0.0927437 | 0.2666615 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.362757 | 0.0744923 | 0.4372493 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.3063483 | 0.10509 | 0.4114384 |  |
| 8 | 8 | 0.0625545 | 71.9062335 | 1.0 | 0.0790525 | 1.0790525 | ✓ |

### N20：selection → action 38

path=[2, 6]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [21, 38, 4]，并列按 prior 抽样。trace 行 3767。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 2 | 0.0231677 | 61.7715071 | 0.0 | 0.0305798 | 0.0305798 |  |
| 38 | 2 | 0.0257828 | 65.5767431 | 0.9459697 | 0.0340315 | 0.9800013 | ✓ |
| 4 | 2 | 0.0280851 | 65.7940841 | 1.0 | 0.0370704 | 1.0370704 |  |

### N22：expansion → action 35

path=[2, 6, 38]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3769。

bucket=1，uniform_random；到达 N140（新建）；closure=[]。

## iteration 151

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[39, 33, 13, 14, 28, 38, 19, 0, 22, 31, 32, 35, 16, 24]

### N0：selection → action 0

path=[]；visits=150；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3789。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6530286 | 0.6530286 |  |
| 0 | 67 | 0.3308308 | 65.0285485 | 1.0 | 0.0834201 | 1.0834201 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.8363912 | 0.0736395 | 0.9100307 |  |

### N2：selection → action 8

path=[0]；visits=67；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3791。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2319566 | 0.128249 | 0.3602056 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.6765431 | 0.0465974 | 0.7231405 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.4208186 | 0.1124507 | 0.5332692 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1332406 | 0.1332406 |  |
| 38 | 21 | 0.0717799 | 65.6101391 | 0.7087768 | 0.0373892 | 0.7461659 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3184297 | 0.0934437 | 0.4118734 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.6641794 | 0.0750545 | 0.7392339 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.5608996 | 0.1058832 | 0.6667827 |  |
| 8 | 9 | 0.0625545 | 67.2441262 | 1.0 | 0.0716842 | 1.0716842 | ✓ |

### N20：expansion → action 11

path=[2, 6]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 35 个代表动作。trace 行 3793。

bucket=1，uniform_random；到达 N141（新建）；closure=[]。

## iteration 152

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[8, 39, 23, 1, 18, 36, 13, 35, 34, 16, 27, 0, 3]

### N0：selection → action 0

path=[]；visits=151；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3813。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6552017 | 0.6552017 |  |
| 0 | 68 | 0.3308308 | 64.9856604 | 1.0 | 0.0824847 | 1.0824847 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.848564 | 0.0738846 | 0.9224486 |  |

### N2：selection → action 8

path=[0]；visits=68；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3815。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2848432 | 0.1292025 | 0.4140457 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8307962 | 0.0469438 | 0.87774 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.516766 | 0.1132867 | 0.6300527 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1342313 | 0.1342313 |  |
| 38 | 21 | 0.0717799 | 65.6101391 | 0.8703792 | 0.0376672 | 0.9080463 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.3910323 | 0.0941384 | 0.4851707 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.8156136 | 0.0756125 | 0.8912261 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.6887857 | 0.1066704 | 0.7954561 |  |
| 8 | 10 | 0.0625545 | 66.20238 | 1.0 | 0.065652 | 1.065652 | ✓ |

### N20：selection → action 11

path=[2, 6]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [11]，并列按 prior 抽样。trace 行 3817。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 2 | 0.0248972 | 61.7715071 | 0.4272618 | 0.0367415 | 0.4640033 |  |
| 38 | 3 | 0.0244732 | 58.7706708 | 0.0 | 0.0270868 | 0.0270868 |  |
| 4 | 2 | 0.0282939 | 65.7940841 | 1.0 | 0.0417542 | 1.0417542 |  |
| 11 | 1 | 0.0375157 | 63.0771413 | 0.6131592 | 0.0830446 | 0.6962038 | ✓ |

### N141：expansion → action 32

path=[2, 6, 11]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 22 个代表动作。trace 行 3819。

bucket=0，compatibility_richness_prior；到达 N142（新建）；closure=[]。

## iteration 153

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[37, 7, 16, 20, 1, 9, 27, 38, 2, 4, 26, 31, 12]

### N0：selection → action 0

path=[]；visits=152；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3838。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6573677 | 0.6573677 |  |
| 0 | 69 | 0.3308308 | 64.8452347 | 1.0 | 0.0815751 | 1.0815751 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.8910241 | 0.0741288 | 0.965153 |  |

### N2：selection → action 38

path=[0]；visits=69；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3840。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3272633 | 0.1301491 | 0.4574124 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.9545222 | 0.0472877 | 1.0018099 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5937251 | 0.1141167 | 0.7078418 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1352146 | 0.1352146 |  |
| 38 | 21 | 0.0717799 | 65.6101391 | 1.0 | 0.0379431 | 1.0379431 | ✓ |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4492666 | 0.0948281 | 0.5440947 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.9370785 | 0.0761665 | 1.0132449 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7913628 | 0.1074519 | 0.8988147 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.7593377 | 0.0606219 | 0.8199596 |  |

### N4：selection → action 28

path=[24, 22]；visits=21；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [28]，并列按 prior 抽样。trace 行 3842。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 3 | 0.0430705 | 60.2792251 | 0.0 | 0.0690808 | 0.0690808 | ✓ |
| 4 | 4 | 0.0056816 | 66.9824343 | 0.3597828 | 0.0072902 | 0.367073 |  |
| 19 | 4 | 0.0477591 | 78.91049 | 1.0 | 0.0612808 | 1.0612808 |  |
| 6 | 4 | 0.0365193 | 60.4491115 | 0.0091184 | 0.0468587 | 0.055977 |  |
| 9 | 4 | 0.0409347 | 64.3916163 | 0.2207253 | 0.0525242 | 0.2732495 |  |

### N66：selection → action 8

path=[24, 22, 28]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [16, 8]，并列按 prior 抽样。trace 行 3844。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 16 | 1 | 0.0334571 | 62.5579329 | 1.0 | 0.0405646 | 1.0405646 |  |
| 8 | 1 | 0.010246 | 57.541205 | 0.0 | 0.0124226 | 0.0124226 | ✓ |

### N93：expansion → action 35

path=[24, 22, 28, 8]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3846。

bucket=0，compatibility_richness_prior；到达 N143（新建）；closure=[]。

## iteration 154

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[12, 37, 38, 29, 14, 31, 13, 25, 6, 24, 3, 16, 35, 21, 1]

### N0：selection → action 0

path=[]；visits=153；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3865。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6595265 | 0.6595265 |  |
| 0 | 70 | 0.3308308 | 64.6269696 | 1.0 | 0.0806903 | 1.0806903 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9661668 | 0.0743723 | 1.040539 |  |

### N2：selection → action 20

path=[0]；visits=70；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3867。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3428557 | 0.1310888 | 0.4739445 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 1.0 | 0.0476292 | 1.0476292 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.6220129 | 0.1149407 | 0.7369536 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1361909 | 0.1361909 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.8685456 | 0.0365555 | 0.9051011 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4706717 | 0.0955128 | 0.5661845 |  |
| 20 | 8 | 0.0589459 | 65.3599137 | 0.9817252 | 0.0767164 | 1.0584416 | ✓ |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.829067 | 0.1082277 | 0.9372947 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.7955161 | 0.0610596 | 0.8565757 |  |

### N98：selection → action 19

path=[0, 20]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [19, 4]，并列按 prior 抽样。trace 行 3869。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 2 | 0.0544664 | 68.5903306 | 1.0 | 0.071892 | 1.071892 | ✓ |
| 4 | 2 | 0.0073295 | 59.1190459 | 0.0 | 0.0096744 | 0.0096744 |  |
| 27 | 3 | 0.0554026 | 65.6739537 | 0.6920822 | 0.0548458 | 0.746928 |  |

### N99：expansion → action 27

path=[0, 20, 19]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 3871。

bucket=1，uniform_random；到达 N144（新建）；closure=[]。

## iteration 155

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 13, 24, 34, 17, 27, 19, 12, 38, 6, 39, 35, 21, 5, 1]

### N0：selection → action 0

path=[]；visits=154；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3892。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6616783 | 0.6616783 |  |
| 0 | 71 | 0.3308308 | 64.7615829 | 1.0 | 0.0798292 | 1.0798292 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9183993 | 0.0746149 | 0.9930142 |  |

### N2：selection → action 20

path=[0]；visits=71；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3894。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2977794 | 0.1320218 | 0.4298012 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8685269 | 0.0479682 | 0.916495 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.540235 | 0.1157587 | 0.6559937 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1371603 | 0.1371603 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.7543552 | 0.0368156 | 0.7911708 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.408791 | 0.0961926 | 0.5049836 |  |
| 20 | 9 | 0.0589459 | 66.0038917 | 1.0 | 0.0695362 | 1.0695362 | ✓ |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7200669 | 0.1089981 | 0.829065 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.6909271 | 0.0614942 | 0.7524213 |  |

### N98：expansion → action 29

path=[0, 20]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 25 个代表动作。trace 行 3896。

bucket=1，uniform_random；到达 N145（新建）；closure=[]。

## iteration 156

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[38, 24, 35, 3, 37, 27, 15, 28, 25, 36, 14, 2, 18, 9]

### N0：selection → action 0

path=[]；visits=155；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3917。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6638232 | 0.6638232 |  |
| 0 | 72 | 0.3308308 | 64.7990303 | 1.0 | 0.0789909 | 1.0789909 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9059395 | 0.0748568 | 0.9807962 |  |

### N2：selection → action 20

path=[0]；visits=72；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3919。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.2936716 | 0.1329483 | 0.4266199 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8565458 | 0.0483048 | 0.9048506 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5327826 | 0.1165711 | 0.6493537 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1381228 | 0.1381228 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.7439491 | 0.037074 | 0.7810231 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4031519 | 0.0968676 | 0.5000195 |  |
| 20 | 10 | 0.0589459 | 66.0650254 | 1.0 | 0.0636583 | 1.0636583 | ✓ |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7101338 | 0.109763 | 0.8198968 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.6813959 | 0.0619258 | 0.7433217 |  |

### N98：selection → action 29

path=[0, 20]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [29]，并列按 prior 抽样。trace 行 3921。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 3 | 0.0544664 | 69.445459 | 1.0 | 0.0602833 | 1.0602833 |  |
| 4 | 2 | 0.0073295 | 59.1190459 | 0.0 | 0.0108163 | 0.0108163 |  |
| 27 | 3 | 0.0554026 | 65.6739537 | 0.634771 | 0.0613195 | 0.6960905 |  |
| 29 | 1 | 0.0470672 | 66.6152286 | 0.7259232 | 0.1041878 | 0.830111 | ✓ |

### N145：expansion → action 16

path=[0, 20, 29]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 3923。

bucket=0，compatibility_richness_prior；到达 N146（新建）；closure=[]。

## iteration 157

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[16, 11, 38, 13, 6, 8, 19, 2, 29, 9, 37, 24, 12]

### N0：selection → action 0

path=[]；visits=156；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3943。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6659611 | 0.6659611 |  |
| 0 | 73 | 0.3308308 | 64.7583113 | 1.0 | 0.0781744 | 1.0781744 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9195042 | 0.0750979 | 0.994602 |  |

### N2：selection → action 20

path=[0]；visits=73；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3945。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3151497 | 0.1338684 | 0.4490181 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.9191906 | 0.0486391 | 0.9678296 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5717484 | 0.1173778 | 0.6891263 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1390787 | 0.1390787 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.7983589 | 0.0373306 | 0.8356895 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.432637 | 0.097538 | 0.530175 |  |
| 20 | 11 | 0.0589459 | 65.7629977 | 1.0 | 0.0587573 | 1.0587573 | ✓ |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7620705 | 0.1105226 | 0.8725931 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.7312308 | 0.0623543 | 0.7935852 |  |

### N98：selection → action 4

path=[0, 20]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [4, 29]，并列按 prior 抽样。trace 行 3947。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 3 | 0.0544664 | 69.445459 | 1.0 | 0.0632257 | 1.0632257 |  |
| 4 | 2 | 0.0073295 | 59.1190459 | 0.0 | 0.0113442 | 0.0113442 | ✓ |
| 27 | 3 | 0.0554026 | 65.6739537 | 0.634771 | 0.0643124 | 0.6990834 |  |
| 29 | 2 | 0.0470672 | 64.6789747 | 0.5384182 | 0.0728487 | 0.6112669 |  |

### N100：expansion → action 25

path=[0, 20, 4]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 18 个代表动作。trace 行 3949。

bucket=1，uniform_random；到达 N147（新建）；closure=[3]。

## iteration 158

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[3, 22, 5, 17, 24, 27, 14, 30, 16, 37, 19, 9, 36]

### N0：selection → action 0

path=[]；visits=157；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3968。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6680922 | 0.6680922 |  |
| 0 | 74 | 0.3308308 | 64.8229185 | 1.0 | 0.0773789 | 1.0773789 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.8981663 | 0.0753382 | 0.9735045 |  |

### N2：selection → action 20

path=[0]；visits=74；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3970。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3010405 | 0.1347822 | 0.4358227 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 0.8780385 | 0.0489711 | 0.9270096 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.5461513 | 0.1181791 | 0.6643304 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.1400281 | 0.1400281 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.7626165 | 0.0375854 | 0.8002019 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4132679 | 0.0982038 | 0.5114717 |  |
| 20 | 12 | 0.0589459 | 65.9565464 | 1.0 | 0.0546078 | 1.0546078 | ✓ |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.7279527 | 0.111277 | 0.8392297 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.6984937 | 0.0627799 | 0.7612737 |  |

### N98：selection → action 29

path=[0, 20]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [29]，并列按 prior 抽样。trace 行 3972。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 3 | 0.0544664 | 69.445459 | 1.0 | 0.0660371 | 1.0660371 |  |
| 4 | 3 | 0.0073295 | 62.1078912 | 0.0 | 0.0088865 | 0.0088865 |  |
| 27 | 3 | 0.0554026 | 65.6739537 | 0.4860006 | 0.0671721 | 0.5531727 |  |
| 29 | 2 | 0.0470672 | 64.6789747 | 0.3504 | 0.076088 | 0.426488 | ✓ |

### N145：expansion → action 11

path=[0, 20, 29]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 3974。

bucket=1，uniform_random；到达 N148（新建）；closure=[]。

## iteration 159

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[17, 18, 3, 25, 34, 6, 39, 5, 7, 2, 35, 9, 29, 14]

### N0：selection → action 0

path=[]；visits=158；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3993。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6702165 | 0.6702165 |  |
| 0 | 75 | 0.3308308 | 64.5624625 | 1.0 | 0.0766036 | 1.0766036 | ✓ |
| 2 | 75 | 0.3264006 | 64.5394045 | 0.9908632 | 0.0755777 | 1.0664409 |  |

### N2：selection → action 23

path=[0]；visits=75；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 3995。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.3428557 | 0.1356898 | 0.4785455 |  |
| 23 | 7 | 0.0325301 | 65.4292835 | 1.0 | 0.0493009 | 1.0493009 | ✓ |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.6220129 | 0.1189749 | 0.7409878 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.0 | 0.140971 | 0.140971 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.8685456 | 0.0378385 | 0.9063841 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.4706717 | 0.0988651 | 0.5695368 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.8388324 | 0.0510487 | 0.889881 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.829067 | 0.1120264 | 0.9410933 |  |
| 8 | 11 | 0.0625545 | 64.653077 | 0.7955161 | 0.0632027 | 0.8587188 |  |

### N8：selection → action 1

path=[0, 23]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [6, 1, 24]，并列按 prior 抽样。trace 行 3997。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.1473096 | 61.817886 | 0.0521849 | 0.1818808 | 0.2340657 |  |
| 1 | 2 | 0.0832401 | 61.0112676 | 0.0 | 0.1027752 | 0.1027752 | ✓ |
| 24 | 2 | 0.1667107 | 76.4682054 | 1.0 | 0.205835 | 1.205835 |  |

### N44：expansion → action 36

path=[0, 23, 1]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 3999。

bucket=1，uniform_random；到达 N149（新建）；closure=[]。

## iteration 160

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[39, 25, 1, 21, 34, 8, 33, 14, 6, 16, 5, 17, 3]

### N0：selection → action 2

path=[]；visits=159；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4019。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6723341 | 0.6723341 |  |
| 0 | 76 | 0.3308308 | 64.1910469 | 0.8606889 | 0.0758476 | 0.9365365 |  |
| 2 | 75 | 0.3264006 | 64.5394045 | 1.0 | 0.0758165 | 1.0758165 | ✓ |

### N5：selection → action 0

path=[2]；visits=75；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4021。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.8468453 | 0.148958 | 0.9958032 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0 | 0.0479839 | 0.0479839 |  |
| 0 | 5 | 0.086242 | 66.4810506 | 0.9865563 | 0.1742714 | 1.1608277 | ✓ |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.4990137 | 0.0824771 | 0.5814907 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.3982719 | 0.1193568 | 0.5176287 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 1.0 | 0.1187095 | 1.1187095 |  |
| 36 | 14 | 0.118156 | 64.9420543 | 0.7801613 | 0.0955043 | 0.8756656 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.8555776 | 0.0345902 | 0.8901678 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.949164 | 0.0623604 | 1.0115244 |  |

### N13：selection → action 23

path=[2, 0]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [23]，并列按 prior 抽样。trace 行 4023。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.0446317 | 62.5576507 | 0.0 | 0.0465731 | 0.0465731 |  |
| 23 | 1 | 0.0215647 | 65.5426112 | 0.6817566 | 0.0337541 | 0.7155107 | ✓ |
| 26 | 3 | 0.0430029 | 66.9359882 | 1.0 | 0.0336551 | 1.0336551 |  |

### N18：expansion → action 7

path=[2, 0, 23]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 4025。

bucket=0，compatibility_richness_prior；到达 N150（新建）；closure=[]。

## iteration 161

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 20, 33, 6, 5, 11, 16, 38, 8, 39, 35, 17, 26, 18]

### N0：selection → action 0

path=[]；visits=160；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4044。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.674445 | 0.674445 |  |
| 0 | 76 | 0.3308308 | 64.1910469 | 1.0 | 0.0760857 | 1.0760857 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.9534962 | 0.0750669 | 1.0285631 |  |

### N2：selection → action 24

path=[0]；visits=76；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4046。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.7372193 | 0.1365914 | 0.8738107 |  |
| 23 | 8 | 0.0325301 | 57.3365786 | 0.0 | 0.0441142 | 0.0441142 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.8767638 | 0.1197654 | 0.9965292 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.5658333 | 0.1419077 | 0.707741 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 1.0 | 0.0380899 | 1.0380899 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.8011117 | 0.099522 | 0.9006337 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.985147 | 0.0513879 | 1.0365349 |  |
| 24 | 7 | 0.0739182 | 64.7804339 | 0.9802655 | 0.1127707 | 1.0930362 | ✓ |
| 8 | 11 | 0.0625545 | 64.653077 | 0.9634942 | 0.0636227 | 1.0271168 |  |

### N113：selection → action 28

path=[0, 24]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [27, 28, 36]，并列按 prior 抽样。trace 行 4048。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 2 | 0.0443344 | 65.4789259 | 0.6617659 | 0.054739 | 0.7165049 |  |
| 28 | 2 | 0.0441969 | 68.9473435 | 1.0 | 0.0545692 | 1.0545692 | ✓ |
| 36 | 2 | 0.0083742 | 58.6928561 | 0.0 | 0.0103395 | 0.0103395 |  |

### N115：expansion → action 27

path=[0, 24, 28]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 4050。

bucket=1，uniform_random；到达 N151（新建）；closure=[]。

## iteration 162

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 15, 19, 20, 6, 13, 25, 4, 35, 17, 14, 38, 2, 11]

### N0：selection → action 0

path=[]；visits=161；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4070。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6765494 | 0.6765494 |  |
| 0 | 77 | 0.3308308 | 64.1556569 | 1.0 | 0.0753446 | 1.0753446 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.9694372 | 0.0753011 | 1.0447383 |  |

### N2：selection → action 24

path=[0]；visits=77；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4072。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.7372193 | 0.1374871 | 0.8747064 |  |
| 23 | 8 | 0.0325301 | 57.3365786 | 0.0 | 0.0444034 | 0.0444034 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.8767638 | 0.1205508 | 0.9973146 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.5658333 | 0.1428383 | 0.7086716 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 1.0 | 0.0383397 | 1.0383397 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.8011117 | 0.1001746 | 0.9012863 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.985147 | 0.0517248 | 1.0368719 |  |
| 24 | 8 | 0.0739182 | 64.465666 | 0.9388144 | 0.100898 | 1.0397124 | ✓ |
| 8 | 11 | 0.0625545 | 64.653077 | 0.9634942 | 0.0640399 | 1.027534 |  |

### N113：selection → action 27

path=[0, 24]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [27, 36]，并列按 prior 抽样。trace 行 4074。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 2 | 0.0443344 | 65.4789259 | 0.8454964 | 0.0585185 | 0.9040149 | ✓ |
| 28 | 3 | 0.0441969 | 66.7189927 | 1.0 | 0.0437527 | 1.0437527 |  |
| 36 | 2 | 0.0083742 | 58.6928561 | 0.0 | 0.0110534 | 0.0110534 |  |

### N114：expansion → action 29

path=[0, 24, 27]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 4076。

bucket=1，uniform_random；到达 N152（新建）；closure=[]。

## iteration 163

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 11, 6, 19, 8, 20, 30, 22, 37, 38, 35, 21, 26, 10, 3]

### N0：selection → action 0

path=[]；visits=162；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4096。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6786472 | 0.6786472 |  |
| 0 | 78 | 0.3308308 | 64.2257148 | 1.0 | 0.0746216 | 1.0746216 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.9383808 | 0.0755346 | 1.0139153 |  |

### N2：selection → action 24

path=[0]；visits=78；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4098。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.7372193 | 0.138377 | 0.8755962 |  |
| 23 | 8 | 0.0325301 | 57.3365786 | 0.0 | 0.0446909 | 0.0446909 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.8767638 | 0.121331 | 0.9980948 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.5658333 | 0.1437628 | 0.7095961 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 1.0 | 0.0385878 | 1.0385878 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.8011117 | 0.100823 | 0.9019347 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.985147 | 0.0520596 | 1.0372066 |  |
| 24 | 9 | 0.0739182 | 64.8632446 | 0.9911707 | 0.0913959 | 1.0825666 | ✓ |
| 8 | 11 | 0.0625545 | 64.653077 | 0.9634942 | 0.0644544 | 1.0279485 |  |

### N113：expansion → action 5

path=[0, 24]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 25 个代表动作。trace 行 4100。

bucket=1，uniform_random；到达 N153（新建）；closure=[]。

## iteration 164

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 6, 11, 19, 13, 38, 16, 24, 30, 25, 21, 39, 32, 4]

### N0：selection → action 0

path=[]；visits=163；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4121。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6807386 | 0.6807386 |  |
| 0 | 79 | 0.3308308 | 64.3858138 | 1.0 | 0.0739159 | 1.0739159 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.8743693 | 0.0757673 | 0.9501366 |  |

### N2：selection → action 24

path=[0]；visits=79；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4123。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.6690469 | 0.1392612 | 0.8083081 |  |
| 23 | 8 | 0.0325301 | 57.3365786 | 0.0 | 0.0449764 | 0.0449764 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.7956875 | 0.1221063 | 0.9177938 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.5135094 | 0.1446814 | 0.6581908 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.9075277 | 0.0388344 | 0.9463622 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.7270311 | 0.1014673 | 0.8284983 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.8940483 | 0.0523923 | 0.9464405 |  |
| 24 | 10 | 0.0739182 | 65.704051 | 1.0 | 0.0836181 | 1.0836181 | ✓ |
| 8 | 11 | 0.0625545 | 64.653077 | 0.8743977 | 0.0648662 | 0.9392639 |  |

### N113：selection → action 5

path=[0, 24]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [5]，并列按 prior 抽样。trace 行 4125。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 3 | 0.0443344 | 66.3339083 | 0.5241333 | 0.0490692 | 0.5732025 |  |
| 28 | 3 | 0.0441969 | 66.7189927 | 0.5505479 | 0.048917 | 0.5994649 |  |
| 36 | 2 | 0.0083742 | 58.6928561 | 0.0 | 0.0123581 | 0.0123581 |  |
| 5 | 1 | 0.0451795 | 73.2713088 | 1.0 | 0.100009 | 1.100009 | ✓ |

### N153：expansion → action 20

path=[0, 24, 5]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 4127。

bucket=0，compatibility_richness_prior；到达 N107（复用）；closure=[]。

## iteration 165

已发现集合：[1, 7, 8, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class10；rollout：[19, 28, 22, 27, 20, 38, 3, 37, 9, 33, 12, 30, 31]

### N0：selection → action 0

path=[]；visits=164；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4148。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6828235 | 0.6828235 |  |
| 0 | 80 | 0.3308308 | 64.4234276 | 1.0 | 0.0732269 | 1.0732269 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.8605773 | 0.0759994 | 0.9365767 |  |

### N2：selection → action 24

path=[0]；visits=80；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4150。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.067149 | 62.9348103 | 0.6629638 | 0.1401398 | 0.8031036 |  |
| 23 | 8 | 0.0325301 | 57.3365786 | 0.0 | 0.0452602 | 0.0452602 |  |
| 6 | 6 | 0.0686902 | 63.9944715 | 0.7884528 | 0.1228767 | 0.9113296 |  |
| 19 | 6 | 0.0813897 | 61.6333545 | 0.5088404 | 0.1455942 | 0.6544347 |  |
| 38 | 22 | 0.0717799 | 64.9302919 | 0.8992762 | 0.0390794 | 0.9383557 |  |
| 26 | 6 | 0.0570798 | 63.4199909 | 0.7204207 | 0.1021075 | 0.8225281 |  |
| 20 | 13 | 0.0589459 | 64.8175027 | 0.8859193 | 0.0527228 | 0.9386421 |  |
| 24 | 11 | 0.0739182 | 65.7808285 | 1.0 | 0.0771335 | 1.0771335 | ✓ |
| 8 | 11 | 0.0625545 | 64.653077 | 0.8664474 | 0.0652755 | 0.9317229 |  |

### N113：selection → action 5

path=[0, 24]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [36, 5]，并列按 prior 抽样。trace 行 4152。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 3 | 0.0443344 | 66.3339083 | 0.6811967 | 0.0514643 | 0.732661 |  |
| 28 | 3 | 0.0441969 | 66.7189927 | 0.7155269 | 0.0513046 | 0.7668314 |  |
| 36 | 2 | 0.0083742 | 58.6928561 | 0.0 | 0.0129613 | 0.0129613 |  |
| 5 | 2 | 0.0451795 | 69.9099564 | 1.0 | 0.0699269 | 1.0699269 | ✓ |

### N153：expansion → action 7

path=[0, 24, 5]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 4154。

bucket=1，uniform_random；到达 N154（新建）；closure=[]。

## iteration 166

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[5, 29, 15, 30, 32, 26, 17, 13, 19, 14, 36, 24, 37, 27, 10, 11]

### N0：selection → action 0

path=[]；visits=165；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4173。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6849021 | 0.6849021 |  |
| 0 | 81 | 0.3308308 | 65.6862386 | 1.0 | 0.0725541 | 1.0725541 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.5626269 | 0.0762307 | 0.6388576 |  |

### N2：expansion → action 1

path=[0]；visits=81；children=9；K=10。已有 9 条动作边 < K=10，且尚余 8 个代表动作。trace 行 4175。

bucket=1，uniform_random；到达 N155（新建）；closure=[]。

## iteration 167

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 24, 26, 14, 34, 33, 21, 28, 35, 36, 30, 5, 13, 9, 23, 6]

### N0：selection → action 0

path=[]；visits=166；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4197。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6869745 | 0.6869745 |  |
| 0 | 82 | 0.3308308 | 65.4414419 | 1.0 | 0.0718969 | 1.0718969 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.6031044 | 0.0764614 | 0.6795658 |  |

### N2：selection → action 1

path=[0]；visits=82；children=10；K=10。最低访问优先：child.visits < 5；最少 1 次；候选 [1]，并列按 prior 抽样。trace 行 4199。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.3864224 | 0.1421693 | 0.5285917 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0463784 | 0.0463784 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.4595664 | 0.1238322 | 0.5833986 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.2965884 | 0.1486417 | 0.44523 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.5241621 | 0.0397382 | 0.5639003 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.4199124 | 0.1036664 | 0.5235788 |  |
| 20 | 13 | 0.0586424 | 64.8175027 | 0.5163768 | 0.0531029 | 0.5694797 |  |
| 24 | 12 | 0.0744855 | 71.8239153 | 1.0 | 0.0726379 | 1.0726379 |  |
| 8 | 11 | 0.0620846 | 64.653077 | 0.5050271 | 0.06559 | 0.5706171 |  |
| 1 | 1 | 0.0197967 | 61.86129 | 0.3123218 | 0.1254865 | 0.4378083 | ✓ |

### N155：expansion → action 39

path=[0, 1]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 20 个代表动作。trace 行 4201。

bucket=0，compatibility_richness_prior；到达 N156（新建）；closure=[]。

## iteration 168

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 31, 12, 34, 24, 28, 10, 33, 22, 29, 23, 7, 6, 19, 4]

### N0：selection → action 0

path=[]；visits=167；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4223。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6890406 | 0.6890406 |  |
| 0 | 83 | 0.3308308 | 65.4052182 | 1.0 | 0.0712546 | 1.0712546 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.609594 | 0.0766914 | 0.6862854 |  |

### N2：selection → action 1

path=[0]；visits=83；children=10；K=10。最低访问优先：child.visits < 5；最少 2 次；候选 [1]，并列按 prior 抽样。trace 行 4225。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.3864224 | 0.1430336 | 0.529456 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0466603 | 0.0466603 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.4595664 | 0.124585 | 0.5841514 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.2965884 | 0.1495453 | 0.4461337 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.5241621 | 0.0399798 | 0.5641419 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.4199124 | 0.1042966 | 0.524209 |  |
| 20 | 13 | 0.0586424 | 64.8175027 | 0.5163768 | 0.0534257 | 0.5698025 |  |
| 24 | 12 | 0.0744855 | 71.8239153 | 1.0 | 0.0730795 | 1.0730795 |  |
| 8 | 11 | 0.0620846 | 64.653077 | 0.5050271 | 0.0659887 | 0.5710159 |  |
| 1 | 2 | 0.0197967 | 63.3502569 | 0.415099 | 0.0841662 | 0.4992652 | ✓ |

### N155：expansion → action 37

path=[0, 1]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 19 个代表动作。trace 行 4227。

bucket=1，uniform_random；到达 N157（新建）；closure=[]。

## iteration 169

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[24, 14, 7, 9, 10, 22, 31, 12, 21, 36, 34, 29, 33, 5]

### N0：selection → action 0

path=[]；visits=168；children=3；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4248。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6911005 | 0.6911005 |  |
| 0 | 84 | 0.3308308 | 65.200595 | 1.0 | 0.0706268 | 1.0706268 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.6490458 | 0.0769206 | 0.7259664 |  |

### N2：selection → action 1

path=[0]；visits=84；children=10；K=10。最低访问优先：child.visits < 5；最少 3 次；候选 [1]，并列按 prior 抽样。trace 行 4250。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.3864224 | 0.1438926 | 0.530315 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0469406 | 0.0469406 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.4595664 | 0.1253333 | 0.5848997 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.2965884 | 0.1504434 | 0.4470318 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.5241621 | 0.0402199 | 0.564382 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.4199124 | 0.104923 | 0.5248354 |  |
| 20 | 13 | 0.0586424 | 64.8175027 | 0.5163768 | 0.0537466 | 0.5701234 |  |
| 24 | 12 | 0.0744855 | 71.8239153 | 1.0 | 0.0735184 | 1.0735184 |  |
| 8 | 11 | 0.0620846 | 64.653077 | 0.5050271 | 0.0663851 | 0.5714122 |  |
| 1 | 3 | 0.0197967 | 62.8330826 | 0.3794006 | 0.0635038 | 0.4429044 | ✓ |

### N155：selection → action 39

path=[0, 1]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [39, 37]，并列按 prior 抽样。trace 行 4252。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 39 | 1 | 0.0887994 | 64.8392238 | 1.0 | 0.1076635 | 1.1076635 | ✓ |
| 37 | 1 | 0.0411433 | 61.798734 | 0.0 | 0.0498836 | 0.0498836 |  |

### N156：expansion → action 28

path=[0, 1, 39]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 19 个代表动作。trace 行 4254。

bucket=0，compatibility_richness_prior；到达 N158（新建）；closure=[]。

## iteration 170

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 28, 32, 17, 18, 35, 37, 7, 38, 33, 24, 8, 25, 10, 13]

### N0：selection → action 0

path=[]；visits=169；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4274。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6931543 | 0.6931543 |  |
| 0 | 85 | 0.3308308 | 65.2259928 | 1.0 | 0.070013 | 1.070013 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.6438736 | 0.0771492 | 0.7210229 |  |

### N2：selection → action 1

path=[0]；visits=85；children=10；K=10。最低访问优先：child.visits < 5；最少 4 次；候选 [1]，并列按 prior 抽样。trace 行 4276。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.3864224 | 0.1447466 | 0.531169 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0472192 | 0.0472192 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.4595664 | 0.1260771 | 0.5856435 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.2965884 | 0.1513363 | 0.4479247 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.5241621 | 0.0404586 | 0.5646207 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.4199124 | 0.1055457 | 0.5254581 |  |
| 20 | 13 | 0.0586424 | 64.8175027 | 0.5163768 | 0.0540656 | 0.5704424 |  |
| 24 | 12 | 0.0744855 | 71.8239153 | 1.0 | 0.0739547 | 1.0739547 |  |
| 8 | 11 | 0.0620846 | 64.653077 | 0.5050271 | 0.066779 | 0.5718062 |  |
| 1 | 4 | 0.0197967 | 63.5432192 | 0.4284183 | 0.0511045 | 0.4795229 | ✓ |

### N155：expansion → action 4

path=[0, 1]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 18 个代表动作。trace 行 4278。

bucket=0，compatibility_richness_prior；到达 N159（新建）；closure=[]。

## iteration 171

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 35, 2, 8, 23, 28, 18, 30, 25, 20, 17, 29, 33, 5]

### N0：selection → action 0

path=[]；visits=170；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4299。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.695202 | 0.695202 |  |
| 0 | 86 | 0.3308308 | 64.2257697 | 1.0 | 0.0694127 | 1.0694127 | ✓ |
| 2 | 76 | 0.3264006 | 64.0909608 | 0.9383572 | 0.0773771 | 1.0157344 |  |

### N2：selection → action 24

path=[0]；visits=86；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4301。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.3864224 | 0.1455956 | 0.532018 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0474961 | 0.0474961 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.4595664 | 0.1268166 | 0.586383 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.2965884 | 0.1522239 | 0.4488123 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.5241621 | 0.0406959 | 0.564858 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.4199124 | 0.1061648 | 0.5260771 |  |
| 20 | 13 | 0.0586424 | 64.8175027 | 0.5163768 | 0.0543827 | 0.5707595 |  |
| 24 | 12 | 0.0744855 | 71.8239153 | 1.0 | 0.0743885 | 1.0743885 | ✓ |
| 8 | 11 | 0.0620846 | 64.653077 | 0.5050271 | 0.0671707 | 0.5721979 |  |
| 1 | 5 | 0.0197967 | 59.9538984 | 0.1806626 | 0.0428369 | 0.2234995 |  |

### N113：selection → action 36

path=[0, 24]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [36]，并列按 prior 抽样。trace 行 4303。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 3 | 0.0443344 | 66.3339083 | 0.2246505 | 0.0537527 | 0.2784031 |  |
| 28 | 3 | 0.0441969 | 66.7189927 | 0.2359721 | 0.0535859 | 0.289558 |  |
| 36 | 2 | 0.0083742 | 58.6928561 | 0.0 | 0.0135376 | 0.0135376 | ✓ |
| 5 | 3 | 0.0451795 | 92.7059276 | 1.0 | 0.0547772 | 1.0547772 |  |

### N117：expansion → action 7

path=[0, 24, 36]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 4305。

bucket=1，uniform_random；到达 N160（新建）；closure=[]。

## iteration 172

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[20, 38, 0, 17, 27, 19, 5, 12, 35, 15, 22, 37, 30, 21]

### N0：selection → action 2

path=[]；visits=171；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4325。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6972437 | 0.6972437 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9393525 | 0.0688255 | 1.0081781 |  |
| 2 | 76 | 0.3264006 | 64.0909608 | 1.0 | 0.0776044 | 1.0776044 | ✓ |

### N5：selection → action 25

path=[2]；visits=76；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4327。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.857379 | 0.1499477 | 1.0073268 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0687787 | 0.0483027 | 0.1170814 |  |
| 0 | 6 | 0.086242 | 58.5740071 | 0.0 | 0.150368 | 0.150368 |  |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.5334709 | 0.0830251 | 0.6164959 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.439658 | 0.1201499 | 0.5598078 |  |
| 25 | 9 | 0.0979099 | 66.5812942 | 1.0 | 0.1194983 | 1.1194983 | ✓ |
| 36 | 14 | 0.118156 | 64.9420543 | 0.7952815 | 0.0961389 | 0.8914204 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.8655108 | 0.03482 | 0.9003308 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.9526604 | 0.0627748 | 1.0154352 |  |

### N9：expansion → action 28

path=[24, 3]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 16 个代表动作。trace 行 4329。

bucket=1，uniform_random；到达 N161（新建）；closure=[]。

## iteration 173

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[0, 38, 23, 9, 12, 17, 20, 11, 30, 27, 25, 37, 29, 8]

### N0：selection → action 2

path=[]；visits=172；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4349。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.6992795 | 0.6992795 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4539466 | 0.0690265 | 0.522973 |  |
| 2 | 77 | 0.3264006 | 66.2853062 | 1.0 | 0.0768331 | 1.0768331 | ✓ |

### N5：selection → action 25

path=[2]；visits=77；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4351。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.3997868 | 0.150931 | 0.5507178 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0320708 | 0.0486194 | 0.0806902 |  |
| 0 | 6 | 0.086242 | 58.5740071 | 0.0 | 0.1513541 | 0.1513541 |  |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.2487518 | 0.0835695 | 0.3323213 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.2050078 | 0.1209378 | 0.3259456 |  |
| 25 | 10 | 0.0979099 | 75.7463622 | 1.0 | 0.1093471 | 1.1093471 | ✓ |
| 36 | 14 | 0.118156 | 64.9420543 | 0.3708313 | 0.0967694 | 0.4676007 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.4035785 | 0.0350484 | 0.4386269 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.4442154 | 0.0631864 | 0.5074019 |  |

### N9：selection → action 28

path=[24, 3]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [28]，并列按 prior 抽样。trace 行 4353。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0620248 | 67.3149488 | 0.4036465 | 0.0915318 | 0.4951783 |  |
| 29 | 3 | 0.0592759 | 64.7159413 | 0.2368197 | 0.0656064 | 0.3024261 |  |
| 4 | 3 | 0.0594255 | 61.0265089 | 0.0 | 0.065772 | 0.065772 |  |
| 28 | 1 | 0.0677063 | 76.6055873 | 1.0 | 0.1498744 | 1.1498744 | ✓ |

### N161：expansion → action 39

path=[24, 3, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 4355。

bucket=0，compatibility_richness_prior；到达 N162（新建）；closure=[]。

## iteration 174

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[37, 27, 1, 20, 38, 6, 4, 13, 7, 30, 25, 36, 15, 10]

### N0：selection → action 2

path=[]；visits=173；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4375。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7013093 | 0.7013093 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6015311 | 0.0692268 | 0.6707579 |  |
| 2 | 78 | 0.3264006 | 65.2434419 | 1.0 | 0.0760808 | 1.0760808 | ✓ |

### N5：selection → action 25

path=[2]；visits=78；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4377。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.7367666 | 0.1519079 | 0.8886745 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0591032 | 0.0489341 | 0.1080373 |  |
| 0 | 6 | 0.086242 | 58.5740071 | 0.0 | 0.1523337 | 0.1523337 |  |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.4584244 | 0.0841104 | 0.5425349 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.3778087 | 0.1217205 | 0.4995293 |  |
| 25 | 11 | 0.0979099 | 67.8921279 | 1.0 | 0.1008837 | 1.1008837 | ✓ |
| 36 | 14 | 0.118156 | 64.9420543 | 0.6834047 | 0.0973957 | 0.7808004 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.7437544 | 0.0352752 | 0.7790296 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.8186442 | 0.0635954 | 0.8822396 |  |

### N9：selection → action 28

path=[24, 3]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [9, 28]，并列按 prior 抽样。trace 行 4379。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0620248 | 67.3149488 | 0.9078073 | 0.0959994 | 1.0038067 |  |
| 29 | 3 | 0.0592759 | 64.7159413 | 0.5326113 | 0.0688086 | 0.6014199 |  |
| 4 | 3 | 0.0594255 | 61.0265089 | 0.0 | 0.0689822 | 0.0689822 |  |
| 28 | 2 | 0.0677063 | 67.9535733 | 1.0 | 0.104793 | 1.104793 | ✓ |

### N161：expansion → action 29

path=[24, 3, 28]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 4381。

bucket=1，uniform_random；到达 N163（新建）；closure=[]。

## iteration 175

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[27, 33, 38, 32, 16, 23, 20, 22, 15, 25, 0, 35, 13, 2]

### N0：selection → action 2

path=[]；visits=174；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4401。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7033333 | 0.7033333 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6321458 | 0.0694266 | 0.7015724 |  |
| 2 | 79 | 0.3264006 | 65.0882432 | 1.0 | 0.0753466 | 1.0753466 | ✓ |

### N5：selection → action 25

path=[2]；visits=79；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4403。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.8501315 | 0.1528786 | 1.0030101 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0681973 | 0.0492468 | 0.1174441 |  |
| 0 | 6 | 0.086242 | 58.5740071 | 0.0 | 0.1533071 | 0.1533071 |  |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.5289613 | 0.0846479 | 0.6136092 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.4359415 | 0.1224983 | 0.5584398 |  |
| 25 | 12 | 0.0979099 | 66.649558 | 1.0 | 0.0937184 | 1.0937184 | ✓ |
| 36 | 14 | 0.118156 | 64.9420543 | 0.7885589 | 0.098018 | 0.8865769 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.8581945 | 0.0355006 | 0.8936951 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 0.9446074 | 0.0640018 | 1.0086092 |  |

### N9：selection → action 9

path=[24, 3]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [9]，并列按 prior 抽样。trace 行 4405。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0.0620248 | 67.3149488 | 1.0 | 0.1002681 | 1.1002681 | ✓ |
| 29 | 3 | 0.0592759 | 64.7159413 | 0.5867008 | 0.0718683 | 0.658569 |  |
| 4 | 3 | 0.0594255 | 61.0265089 | 0.0 | 0.0720496 | 0.0720496 |  |
| 28 | 3 | 0.0677063 | 66.6516913 | 0.8945275 | 0.0820896 | 0.9766171 |  |

### N11：expansion → action 37

path=[24, 3, 9]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 4407。

bucket=1，uniform_random；到达 N164（新建）；closure=[]。

## iteration 176

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[37, 1, 39, 14, 18, 13, 26, 36, 32, 9, 31, 33, 15, 19]

### N0：selection → action 2

path=[]；visits=175；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4427。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7053515 | 0.7053515 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8749286 | 0.0696258 | 0.9445544 |  |
| 2 | 80 | 0.3264006 | 64.2420659 | 1.0 | 0.0746299 | 1.0746299 | ✓ |

### N5：selection → action 3

path=[2]；visits=80；children=9；K=9。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4429。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0737151 | 65.4392872 | 0.8999839 | 0.1538431 | 1.053827 |  |
| 37 | 16 | 0.0672799 | 59.1247377 | 0.0721964 | 0.0495575 | 0.1217539 |  |
| 0 | 6 | 0.086242 | 58.5740071 | 0.0 | 0.1542744 | 0.1542744 |  |
| 6 | 11 | 0.0816311 | 62.8456614 | 0.5599801 | 0.0851819 | 0.645162 |  |
| 28 | 7 | 0.0787551 | 62.0944746 | 0.4615054 | 0.1232712 | 0.5847766 |  |
| 25 | 13 | 0.0979099 | 64.4692339 | 0.7728176 | 0.0875733 | 0.8603909 |  |
| 36 | 14 | 0.118156 | 64.9420543 | 0.8348006 | 0.0986365 | 0.9334371 |  |
| 24 | 8 | 0.0256766 | 65.5044007 | 0.9085198 | 0.0357246 | 0.9442444 |  |
| 3 | 8 | 0.0462906 | 66.2022324 | 1.0 | 0.0644056 | 1.0644056 | ✓ |

### N122：selection → action 6

path=[2, 3]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [6, 8]，并列按 prior 抽样。trace 行 4431。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.143538 | 65.3637676 | 0.3981436 | 0.1894605 | 0.5876041 | ✓ |
| 18 | 3 | 0.1242323 | 59.4610318 | 0.0 | 0.1229837 | 0.1229837 |  |
| 8 | 2 | 0.1439711 | 74.2866761 | 1.0 | 0.1900321 | 1.1900321 |  |

### N123：expansion → action 34

path=[2, 3, 6]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 21 个代表动作。trace 行 4433。

bucket=1，uniform_random；到达 N165（新建）；closure=[]。

## iteration 177

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 39, 32, 26, 22, 10, 29, 34, 17, 5, 13, 36, 27, 9, 14, 1]

### N0：selection → action 2

path=[]；visits=176；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4453。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7073639 | 0.7073639 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9437272 | 0.0698245 | 1.0135517 |  |
| 2 | 81 | 0.3264006 | 64.0814481 | 1.0 | 0.0739301 | 1.0739301 | ✓ |

### N5：expansion → action 8

path=[2]；visits=81；children=9；K=10。已有 9 条动作边 < K=10，且尚余 5 个代表动作。trace 行 4455。

bucket=1，uniform_random；到达 N166（新建）；closure=[]。

## iteration 178

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[0, 7, 35, 30, 17, 39, 5, 26, 36, 38, 31, 1, 23, 28, 16]

### N0：selection → action 2

path=[]；visits=177；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4477。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7093706 | 0.7093706 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7407489 | 0.0700226 | 0.8107715 |  |
| 2 | 82 | 0.3264006 | 64.641161 | 1.0 | 0.0732466 | 1.0732466 | ✓ |

### N5：selection → action 8

path=[2]；visits=82；children=10；K=10。最低访问优先：child.visits < 5；最少 1 次；候选 [8]，并列按 prior 抽样。trace 行 4479。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5970786 | 0.1555294 | 0.752608 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0478975 | 0.0463681 | 0.0942656 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1573223 | 0.1573223 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.371509 | 0.0864762 | 0.4579852 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3061777 | 0.1261641 | 0.4323418 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5127123 | 0.0912972 | 0.6040095 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5538339 | 0.1012502 | 0.655084 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6027415 | 0.0367658 | 0.6395073 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5614256 | 0.0542346 | 0.6156602 |  |
| 8 | 1 | 0.0824973 | 70.0721254 | 1.0 | 0.5229312 | 1.5229312 | ✓ |

### N166：expansion → action 12

path=[2, 8]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 9 个代表动作。trace 行 4481。

bucket=0，compatibility_richness_prior；到达 N167（新建）；closure=[]。

## iteration 179

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 30, 22, 15, 13, 32, 25, 34, 5, 11, 37, 27, 19, 31, 3]

### N0：selection → action 2

path=[]；visits=178；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4502。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7113716 | 0.7113716 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6810703 | 0.0702201 | 0.7512904 |  |
| 2 | 83 | 0.3264006 | 64.8691897 | 1.0 | 0.0725788 | 1.0725788 | ✓ |

### N5：selection → action 8

path=[2]；visits=83；children=10；K=10。最低访问优先：child.visits < 5；最少 2 次；候选 [8]，并列按 prior 抽样。trace 行 4504。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6785912 | 0.1564749 | 0.8350661 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0544364 | 0.04665 | 0.1010864 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1582787 | 0.1582787 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4222271 | 0.0870019 | 0.509229 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3479768 | 0.1269311 | 0.4749079 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5827073 | 0.0918522 | 0.6745595 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6294428 | 0.1018657 | 0.7313084 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6850273 | 0.0369893 | 0.7220165 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6380709 | 0.0545643 | 0.6926352 |  |
| 8 | 2 | 0.0824973 | 68.6909672 | 1.0 | 0.3507401 | 1.3507401 | ✓ |

### N166：expansion → action 14

path=[2, 8]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 8 个代表动作。trace 行 4506。

bucket=1，uniform_random；到达 N168（新建）；closure=[20]。

## iteration 180

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[7, 26, 18, 21, 36, 25, 33, 39, 24, 37, 38, 34, 35, 15, 13, 3]

### N0：selection → action 2

path=[]；visits=179；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4527。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7133671 | 0.7133671 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5922886 | 0.0704171 | 0.6627056 |  |
| 2 | 84 | 0.3264006 | 65.2934489 | 1.0 | 0.0719261 | 1.0719261 | ✓ |

### N5：selection → action 8

path=[2]；visits=84；children=10；K=10。最低访问优先：child.visits < 5；最少 3 次；候选 [8]，并列按 prior 抽样。trace 行 4529。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.645263 | 0.1574147 | 0.8026776 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0517628 | 0.0469302 | 0.098693 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1592294 | 0.1592294 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4014899 | 0.0875245 | 0.4890143 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3308863 | 0.1276934 | 0.4585798 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5540883 | 0.0924038 | 0.6464922 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5985284 | 0.1024775 | 0.7010059 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.651383 | 0.0372114 | 0.6885944 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6067328 | 0.054892 | 0.6616248 |  |
| 8 | 3 | 0.0824973 | 69.2135142 | 1.0 | 0.264635 | 1.264635 | ✓ |

### N166：selection → action 12

path=[2, 8]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [12, 14]，并列按 prior 抽样。trace 行 4531。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 1 | 0.1330445 | 67.3098089 | 0.0 | 0.1613079 | 0.1613079 | ✓ |
| 14 | 1 | 0.0216367 | 70.2586083 | 1.0 | 0.0262332 | 1.0262332 |  |

### N167：expansion → action 1

path=[2, 8, 12]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 4533。

bucket=0，compatibility_richness_prior；到达 N169（新建）；closure=[]。

## iteration 181

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[13, 39, 3, 9, 23, 1, 6, 4, 32, 35, 36, 18, 34, 28]

### N0：selection → action 2

path=[]；visits=180；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4555。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.715357 | 0.715357 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.496092 | 0.0706135 | 0.5667055 |  |
| 2 | 85 | 0.3264006 | 65.9245474 | 1.0 | 0.0712881 | 1.0712881 | ✓ |

### N5：selection → action 8

path=[2]；visits=85；children=10；K=10。最低访问优先：child.visits < 5；最少 4 次；候选 [8]，并列按 prior 抽样。trace 行 4557。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5807454 | 0.1583489 | 0.7390943 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0465872 | 0.0472087 | 0.0937959 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1601743 | 0.1601743 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3613463 | 0.0880439 | 0.4493902 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2978022 | 0.1284513 | 0.4262535 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.498687 | 0.0929522 | 0.5916393 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5386837 | 0.1030857 | 0.6417694 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.5862535 | 0.0374323 | 0.6236858 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5460678 | 0.0552178 | 0.6012855 |  |
| 8 | 4 | 0.0824973 | 70.3955033 | 1.0 | 0.2129644 | 1.2129644 | ✓ |

### N166：expansion → action 15

path=[2, 8]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 7 个代表动作。trace 行 4559。

bucket=0，compatibility_richness_prior；到达 N170（新建）；closure=[]。

## iteration 182

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[12, 32, 19, 30, 5, 37, 29, 10, 38, 31, 35, 33, 18, 3]

### N0：selection → action 2

path=[]；visits=181；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4579。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7173413 | 0.7173413 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4222517 | 0.0708094 | 0.493061 |  |
| 2 | 86 | 0.3264006 | 66.6040538 | 1.0 | 0.0706641 | 1.0706641 | ✓ |

### N5：selection → action 8

path=[2]；visits=86；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4581。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5311432 | 0.1592776 | 0.6904208 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0426081 | 0.0474856 | 0.0900937 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1611138 | 0.1611138 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3304833 | 0.0885603 | 0.4190436 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2723665 | 0.1292046 | 0.4015712 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4560935 | 0.0934974 | 0.5495909 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.492674 | 0.1036903 | 0.5963643 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.5361808 | 0.0376518 | 0.5738326 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.4994274 | 0.0555416 | 0.554969 |  |
| 8 | 5 | 0.0824973 | 71.4994855 | 1.0 | 0.1785113 | 1.1785113 | ✓ |

### N166：selection → action 15

path=[2, 8]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [14, 15]，并列按 prior 抽样。trace 行 4583。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 2 | 0.1330445 | 70.6256397 | 0.0648832 | 0.1388317 | 0.2037149 |  |
| 14 | 1 | 0.0216367 | 70.2586083 | 0.0 | 0.0338669 | 0.0338669 |  |
| 15 | 1 | 0.1111044 | 75.9154142 | 1.0 | 0.1739058 | 1.1739058 | ✓ |

### N170：expansion → action 36

path=[2, 8, 15]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 4585。

bucket=0，compatibility_richness_prior；到达 N171（新建）；closure=[]。

## iteration 183

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[12, 26, 22, 5, 34, 21, 33, 17, 13, 23, 18, 29, 4]

### N0：selection → action 2

path=[]；visits=182；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4605。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7193202 | 0.7193202 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4355766 | 0.0710047 | 0.5065813 |  |
| 2 | 87 | 0.3264006 | 66.4643964 | 1.0 | 0.0700538 | 1.0700538 | ✓ |

### N5：selection → action 8

path=[2]；visits=87；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4607。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5845745 | 0.160201 | 0.7447755 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0468944 | 0.0477609 | 0.0946553 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1620478 | 0.1620478 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3637288 | 0.0890737 | 0.4528025 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2997657 | 0.1299537 | 0.4297194 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.501975 | 0.0940394 | 0.5960145 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5422354 | 0.1042914 | 0.6465268 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.5901189 | 0.0378701 | 0.627989 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5496682 | 0.0558636 | 0.6055318 |  |
| 8 | 6 | 0.0824973 | 70.3180707 | 1.0 | 0.1538967 | 1.1538967 | ✓ |

### N166：selection → action 14

path=[2, 8]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [14]，并列按 prior 抽样。trace 行 4609。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 2 | 0.1330445 | 70.6256397 | 1.0 | 0.1520825 | 1.1520825 |  |
| 14 | 1 | 0.0216367 | 70.2586083 | 0.206306 | 0.0370993 | 0.2434052 | ✓ |
| 15 | 2 | 0.1111044 | 70.1632053 | 0.0 | 0.1270029 | 0.1270029 |  |

### N168：expansion → action 0

path=[2, 8, 14]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 4 个代表动作。trace 行 4611。

bucket=0，compatibility_richness_prior；到达 N172（新建）；closure=[]。

## iteration 184

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 25, 33, 35, 21, 39, 7, 26, 0, 30, 14, 1, 6, 4]

### N0：selection → action 2

path=[]；visits=183；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4630。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7212936 | 0.7212936 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4010335 | 0.0711995 | 0.472233 |  |
| 2 | 88 | 0.3264006 | 66.8455932 | 1.0 | 0.0694568 | 1.0694568 | ✓ |

### N5：selection → action 8

path=[2]；visits=88；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4632。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5672315 | 0.161119 | 0.7283506 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0455031 | 0.0480346 | 0.0935377 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1629764 | 0.1629764 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3529378 | 0.0895841 | 0.442522 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2908724 | 0.1306984 | 0.4215707 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4870826 | 0.0945783 | 0.5816609 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5261486 | 0.104889 | 0.6310376 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.5726114 | 0.0380871 | 0.6106985 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5333608 | 0.0561837 | 0.5895445 |  |
| 8 | 7 | 0.0824973 | 70.6771426 | 1.0 | 0.1354313 | 1.1354313 | ✓ |

### N166：selection → action 12

path=[2, 8]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [12, 14, 15]，并列按 prior 抽样。trace 行 4634。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 2 | 0.1330445 | 70.6256397 | 0.3346401 | 0.1642679 | 0.4989079 | ✓ |
| 14 | 2 | 0.0216367 | 71.5450914 | 1.0 | 0.0267145 | 1.0267145 |  |
| 15 | 2 | 0.1111044 | 70.1632053 | 0.0 | 0.1371788 | 0.1371788 |  |

### N167：expansion → action 17

path=[2, 8, 12]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 4636。

bucket=1，uniform_random；到达 N173（新建）；closure=[]。

## iteration 185

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[39, 35, 6, 24, 21, 20, 31, 17, 26, 23, 10, 9, 18]

### N0：selection → action 2

path=[]；visits=184；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4656。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7232617 | 0.7232617 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.410249 | 0.0713938 | 0.4816428 |  |
| 2 | 89 | 0.3264006 | 66.7376185 | 1.0 | 0.0688724 | 1.0688724 | ✓ |

### N5：selection → action 8

path=[2]；visits=89；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4658。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6029974 | 0.1620319 | 0.7650293 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0483723 | 0.0483067 | 0.096679 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1638998 | 0.1638998 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3751918 | 0.0900917 | 0.4652835 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3092129 | 0.1314389 | 0.4406518 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5177949 | 0.0951142 | 0.612909 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.559324 | 0.1054833 | 0.6648074 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6087166 | 0.0383029 | 0.6470194 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5669911 | 0.0565021 | 0.6234931 |  |
| 8 | 8 | 0.0824973 | 69.9592628 | 1.0 | 0.1210654 | 1.1210654 | ✓ |

### N166：selection → action 15

path=[2, 8]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [14, 15]，并列按 prior 抽样。trace 行 4660。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 3 | 0.1330445 | 68.7284611 | 0.0 | 0.1317073 | 0.1317073 |  |
| 14 | 2 | 0.0216367 | 71.5450914 | 1.0 | 0.028559 | 1.028559 |  |
| 15 | 2 | 0.1111044 | 70.1632053 | 0.5093832 | 0.1466503 | 0.6560335 | ✓ |

### N170：expansion → action 28

path=[2, 8, 15]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 16 个代表动作。trace 行 4662。

bucket=1，uniform_random；到达 N174（新建）；closure=[]。

## iteration 186

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 38, 4, 18, 34, 27, 32, 35, 6, 23, 0, 31, 13, 15, 28, 1]

### N0：selection → action 2

path=[]；visits=185；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4681。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7252244 | 0.7252244 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4135448 | 0.0715875 | 0.4851323 |  |
| 2 | 90 | 0.3264006 | 66.7001712 | 1.0 | 0.0683004 | 1.0683004 | ✓ |

### N5：selection → action 8

path=[2]；visits=90；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4683。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6269963 | 0.1629397 | 0.7899359 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0502974 | 0.0485774 | 0.0988748 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.164818 | 0.164818 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3901241 | 0.0905964 | 0.4807206 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3215193 | 0.1321753 | 0.4536946 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5384027 | 0.095647 | 0.6340497 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5815847 | 0.1060743 | 0.687659 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.632943 | 0.0385175 | 0.6714605 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5895569 | 0.0568186 | 0.6463755 |  |
| 8 | 9 | 0.0824973 | 69.523482 | 1.0 | 0.1095693 | 1.1095693 | ✓ |

### N166：expansion → action 25

path=[2, 8]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 6 个代表动作。trace 行 4685。

bucket=1，uniform_random；到达 N11（复用）；closure=[]。

## iteration 187

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class29；rollout：[30, 29, 4, 15, 7, 28, 21, 19, 16, 17, 38, 26, 18]

### N0：selection → action 2

path=[]；visits=186；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4708。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7271818 | 0.7271818 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3898527 | 0.0717807 | 0.4616334 |  |
| 2 | 91 | 0.3264006 | 66.9834493 | 1.0 | 0.0677404 | 1.0677404 | ✓ |

### N5：selection → action 8

path=[2]；visits=91；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4710。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6115904 | 0.1638424 | 0.7754328 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0490616 | 0.0488465 | 0.0979081 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1657312 | 0.1657312 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3805384 | 0.0910983 | 0.4716368 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3136193 | 0.1329075 | 0.4465268 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5251737 | 0.096177 | 0.6213506 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5672947 | 0.1066619 | 0.6739566 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.617391 | 0.0387309 | 0.6561219 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5750709 | 0.0571334 | 0.6322043 |  |
| 8 | 10 | 0.0824973 | 69.7992973 | 1.0 | 0.1001603 | 1.1001603 | ✓ |

### N166：selection → action 14

path=[2, 8]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [14]，并列按 prior 抽样。trace 行 4712。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 3 | 0.1330445 | 68.7284611 | 0.0 | 0.1472533 | 0.1472533 |  |
| 14 | 2 | 0.0216367 | 71.5450914 | 0.7927081 | 0.03193 | 0.8246381 | ✓ |
| 15 | 3 | 0.1111044 | 68.7878821 | 0.0167233 | 0.12297 | 0.1396933 |  |
| 25 | 4 | 0.1675189 | 72.2816357 | 1.0 | 0.1483275 | 1.1483275 |  |

### N168：expansion → action 25

path=[2, 8, 14]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 3 个代表动作。trace 行 4714。

bucket=1，uniform_random；到达 N175（新建）；closure=[]。

## iteration 188

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[26, 21, 9, 35, 33, 25, 5, 6, 22, 11, 28, 14]

### N0：selection → action 2

path=[]；visits=187；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4733。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.729134 | 0.729134 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3698562 | 0.0719734 | 0.4418296 |  |
| 2 | 92 | 0.3264006 | 67.2507837 | 1.0 | 0.0671919 | 1.0671919 | ✓ |

### N5：selection → action 8

path=[2]；visits=92；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4735。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5984142 | 0.1647401 | 0.7631544 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0480046 | 0.0491142 | 0.0971188 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1666393 | 0.1666393 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.37234 | 0.0915975 | 0.4639376 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3068626 | 0.1336358 | 0.4404984 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5138592 | 0.096704 | 0.6105632 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5550728 | 0.1072464 | 0.6623192 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6040899 | 0.0389431 | 0.643033 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5626815 | 0.0574464 | 0.620128 |  |
| 8 | 11 | 0.0824973 | 70.0464618 | 1.0 | 0.0923167 | 1.0923167 | ✓ |

### N166：selection → action 12

path=[2, 8]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [12, 14, 15]，并列按 prior 抽样。trace 行 4737。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 3 | 0.1330445 | 68.7284611 | 0.0 | 0.1544405 | 0.1544405 | ✓ |
| 14 | 3 | 0.0216367 | 71.8694298 | 0.8839894 | 0.0251163 | 0.9091057 |  |
| 15 | 3 | 0.1111044 | 68.7878821 | 0.0167233 | 0.128972 | 0.1456954 |  |
| 25 | 4 | 0.1675189 | 72.2816357 | 1.0 | 0.1555672 | 1.1555672 |  |

### N167：selection → action 1

path=[2, 8, 12]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [1, 17]，并列按 prior 抽样。trace 行 4739。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.0206411 | 73.9414706 | 1.0 | 0.025026 | 1.025026 | ✓ |
| 17 | 1 | 0.0423435 | 64.9341039 | 0.0 | 0.0513388 | 0.0513388 |  |

### N169：expansion → action 0

path=[2, 8, 12, 1]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 4741。

bucket=0，compatibility_richness_prior；到达 N176（新建）；closure=[24]。

## iteration 189

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[10, 34, 39, 12, 26, 18, 5, 11, 33, 6, 25, 37, 17]

### N0：selection → action 2

path=[]；visits=188；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4759。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.731081 | 0.731081 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3686388 | 0.0721656 | 0.4408044 |  |
| 2 | 93 | 0.3264006 | 67.2679948 | 1.0 | 0.0666546 | 1.0666546 | ✓ |

### N5：selection → action 8

path=[2]；visits=93；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4761。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6091312 | 0.1656331 | 0.7747643 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0488643 | 0.0493804 | 0.0982447 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1675425 | 0.1675425 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3790083 | 0.092094 | 0.4711023 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3123582 | 0.1343601 | 0.4467183 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5230619 | 0.0972281 | 0.62029 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5650136 | 0.1078277 | 0.6728412 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6149085 | 0.0391542 | 0.6540627 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5727586 | 0.0577578 | 0.6305164 |  |
| 8 | 12 | 0.0824973 | 69.8446166 | 1.0 | 0.0856773 | 1.0856773 | ✓ |

### N166：selection → action 15

path=[2, 8]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [14, 15]，并列按 prior 抽样。trace 行 4763。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 4 | 0.1330445 | 68.4524257 | 0.0 | 0.1290463 | 0.1290463 |  |
| 14 | 3 | 0.0216367 | 71.8694298 | 0.8923522 | 0.0262332 | 0.9185854 |  |
| 15 | 3 | 0.1111044 | 68.7878821 | 0.0876046 | 0.1347069 | 0.2223115 | ✓ |
| 25 | 4 | 0.1675189 | 72.2816357 | 1.0 | 0.1624847 | 1.1624847 |  |

### N170：selection → action 36

path=[2, 8, 15]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [36, 28]，并列按 prior 抽样。trace 行 4765。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0950075 | 64.4109965 | 0.0 | 0.1151905 | 0.1151905 | ✓ |
| 28 | 1 | 0.0146656 | 66.0372356 | 1.0 | 0.0177811 | 1.0177811 |  |

### N171：expansion → action 32

path=[2, 8, 15, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 4767。

bucket=0，compatibility_richness_prior；到达 N130（复用）；closure=[]。

## iteration 190

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[22, 15, 37, 13, 12, 10, 30, 31, 4, 1, 26, 23, 9]

### N0：selection → action 2

path=[]；visits=189；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4787。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7330227 | 0.7330227 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4028952 | 0.0723573 | 0.4752524 |  |
| 2 | 94 | 0.3264006 | 66.8233831 | 1.0 | 0.0661281 | 1.0661281 | ✓ |

### N5：selection → action 8

path=[2]；visits=94；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4789。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6667948 | 0.1665212 | 0.833316 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0534901 | 0.0496451 | 0.1031352 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1684408 | 0.1684408 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4148872 | 0.0925878 | 0.507475 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3419277 | 0.1350806 | 0.4770083 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5725778 | 0.0977494 | 0.6703272 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6185008 | 0.1084059 | 0.7269066 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.673119 | 0.0393641 | 0.7124832 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.626979 | 0.0580675 | 0.6850465 |  |
| 8 | 13 | 0.0824973 | 68.8699476 | 1.0 | 0.0799841 | 1.0799841 | ✓ |

### N166：selection → action 14

path=[2, 8]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [14]，并列按 prior 抽样。trace 行 4791。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 4 | 0.1330445 | 68.4524257 | 0.4014282 | 0.1343156 | 0.5357438 |  |
| 14 | 3 | 0.0216367 | 71.8694298 | 0.9355651 | 0.0273043 | 0.9628694 | ✓ |
| 15 | 4 | 0.1111044 | 65.8843913 | 0.0 | 0.1121659 | 0.1121659 |  |
| 25 | 4 | 0.1675189 | 72.2816357 | 1.0 | 0.1691194 | 1.1691194 |  |

### N168：selection → action 0

path=[2, 8, 14]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [0, 25]，并列按 prior 抽样。trace 行 4793。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0.2855866 | 72.8315744 | 1.0 | 0.3462553 | 1.3462553 | ✓ |
| 25 | 1 | 0.3301689 | 72.5181066 | 0.0 | 0.4003085 | 0.4003085 |  |

### N172：expansion → action 29

path=[2, 8, 14, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 4795。

bucket=0，compatibility_richness_prior；到达 N177（新建）；closure=[]。

## iteration 191

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 11, 23, 18, 32, 38, 34, 8, 37, 5, 15, 29, 1, 4]

### N0：selection → action 2

path=[]；visits=190；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4814。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7349594 | 0.7349594 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4218504 | 0.0725485 | 0.4943988 |  |
| 2 | 95 | 0.3264006 | 66.6083967 | 1.0 | 0.0656122 | 1.0656122 | ✓ |

### N5：selection → action 8

path=[2]；visits=95；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4816。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7015576 | 0.1674046 | 0.8689622 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0562787 | 0.0499085 | 0.1061872 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1693344 | 0.1693344 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.436517 | 0.093079 | 0.529596 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3597538 | 0.1357972 | 0.495551 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6024286 | 0.098268 | 0.7006966 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6507458 | 0.108981 | 0.7597267 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7082115 | 0.0395729 | 0.7477845 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.659666 | 0.0583756 | 0.7180415 |  |
| 8 | 14 | 0.0824973 | 68.3597751 | 1.0 | 0.0750478 | 1.0750478 | ✓ |

### N166：selection → action 25

path=[2, 8]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [12, 14, 15, 25]，并列按 prior 抽样。trace 行 4818。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 4 | 0.1330445 | 68.4524257 | 0.4014282 | 0.1393859 | 0.5408141 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 0.5392266 | 0.022668 | 0.5618946 |  |
| 15 | 4 | 0.1111044 | 65.8843913 | 0.0 | 0.1164 | 0.1164 |  |
| 25 | 4 | 0.1675189 | 72.2816357 | 1.0 | 0.1755035 | 1.1755035 | ✓ |

### N11：expansion → action 27

path=[24, 3, 9]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 23 个代表动作。trace 行 4820。

bucket=0，compatibility_richness_prior；到达 N178（新建）；closure=[]。

## iteration 192

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 37, 35, 15, 5, 18, 30, 39, 7, 10, 1, 28, 9, 17]

### N0：selection → action 2

path=[]；visits=191；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4840。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.736891 | 0.736891 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4277435 | 0.0727391 | 0.5004826 |  |
| 2 | 96 | 0.3264006 | 66.5454408 | 1.0 | 0.0651064 | 1.0651064 | ✓ |

### N5：selection → action 8

path=[2]；visits=96；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4842。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7177255 | 0.1682833 | 0.8860088 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0575757 | 0.0501705 | 0.1077462 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1702233 | 0.1702233 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4465768 | 0.0935676 | 0.5401444 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3680446 | 0.13651 | 0.5045546 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.616312 | 0.0987838 | 0.7150958 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6657426 | 0.109553 | 0.7752957 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7245327 | 0.0397807 | 0.7643134 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6748684 | 0.058682 | 0.7335504 |  |
| 8 | 15 | 0.0824973 | 68.139336 | 1.0 | 0.0707267 | 1.0707267 | ✓ |

### N166：selection → action 12

path=[2, 8]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [12, 14, 15]，并列按 prior 抽样。trace 行 4844。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 4 | 0.1330445 | 68.4524257 | 0.7444518 | 0.1442781 | 0.8887299 | ✓ |
| 14 | 4 | 0.0216367 | 69.3339556 | 1.0 | 0.0234636 | 1.0234636 |  |
| 15 | 4 | 0.1111044 | 65.8843913 | 0.0 | 0.1204855 | 0.1204855 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.8067746 | 0.1513861 | 0.9581608 |  |

### N167：expansion → action 3

path=[2, 8, 12]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 4846。

bucket=0，compatibility_richness_prior；到达 N179（新建）；closure=[]。

## iteration 193

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 25, 11, 4, 36, 21, 32, 9, 1, 16, 31, 30, 26, 18, 13]

### N0：selection → action 2

path=[]；visits=192；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4866。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7388175 | 0.7388175 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4315738 | 0.0729293 | 0.5045031 |  |
| 2 | 97 | 0.3264006 | 66.5054437 | 1.0 | 0.0646106 | 1.0646106 | ✓ |

### N5：selection → action 8

path=[2]；visits=97；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4868。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7302357 | 0.1691576 | 0.8993933 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0585793 | 0.0504311 | 0.1090104 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1711076 | 0.1711076 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4543609 | 0.0940537 | 0.5484145 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3744598 | 0.1372192 | 0.5116789 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6270546 | 0.099297 | 0.7263516 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6773468 | 0.1101221 | 0.787469 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7371616 | 0.0399873 | 0.777149 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6866316 | 0.0589868 | 0.7456185 |  |
| 8 | 16 | 0.0824973 | 67.9754645 | 1.0 | 0.0669121 | 1.0669121 | ✓ |

### N166：expansion → action 33

path=[2, 8]；visits=16；children=4；K=5。已有 4 条动作边 < K=5，且尚余 5 个代表动作。trace 行 4870。

bucket=0，compatibility_richness_prior；到达 N180（新建）；closure=[]。

## iteration 194

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[35, 6, 26, 32, 18, 11, 13, 25, 5, 21, 39, 10, 1]

### N0：selection → action 2

path=[]；visits=193；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4891。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.740739 | 0.740739 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4058973 | 0.073119 | 0.4790163 |  |
| 2 | 98 | 0.3264006 | 66.7879952 | 1.0 | 0.0641243 | 1.0641243 | ✓ |

### N5：selection → action 8

path=[2]；visits=98；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4893。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7034956 | 0.1700273 | 0.8735229 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0564342 | 0.0506904 | 0.1071246 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1719874 | 0.1719874 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4377229 | 0.0945372 | 0.5322601 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3607476 | 0.1379247 | 0.4986723 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6040928 | 0.0998075 | 0.7039004 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6525434 | 0.1106883 | 0.7632318 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7101679 | 0.0401929 | 0.7503609 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6614883 | 0.0592901 | 0.7207784 |  |
| 8 | 17 | 0.0824973 | 68.3328167 | 1.0 | 0.0635197 | 1.0635197 | ✓ |

### N166：selection → action 33

path=[2, 8]；visits=17；children=5；K=5。最低访问优先：child.visits < 5；最少 1 次；候选 [33]，并列按 prior 抽样。trace 行 4895。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.2425928 | 0.1279965 | 0.3705893 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 0.422427 | 0.024979 | 0.4474059 |  |
| 15 | 4 | 0.1111044 | 65.8843913 | 0.0 | 0.1282666 | 0.1282666 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.3408034 | 0.1611629 | 0.5019662 |  |
| 33 | 1 | 0.1164611 | 74.0504524 | 1.0 | 0.336127 | 1.336127 | ✓ |

### N180：expansion → action 34

path=[2, 8, 33]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 25 个代表动作。trace 行 4897。

bucket=0，compatibility_richness_prior；到达 N181（新建）；closure=[]。

## iteration 195

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[22, 25, 0, 5, 35, 17, 7, 12, 29, 36, 26, 28, 9]

### N0：selection → action 2

path=[]；visits=194；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4916。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7426555 | 0.7426555 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3851332 | 0.0733081 | 0.4584414 |  |
| 2 | 99 | 0.3264006 | 67.0440417 | 1.0 | 0.0636473 | 1.0636473 | ✓ |

### N5：selection → action 8

path=[2]；visits=99；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4918。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.681955 | 0.1708925 | 0.8528475 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0547062 | 0.0509484 | 0.1056546 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1728626 | 0.1728626 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.42432 | 0.0950183 | 0.5193384 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3497017 | 0.1386266 | 0.4883283 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5855958 | 0.1003155 | 0.6859113 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6325629 | 0.1112516 | 0.7438145 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.688423 | 0.0403975 | 0.7288204 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6412338 | 0.0595918 | 0.7008257 |  |
| 8 | 18 | 0.0824973 | 68.6410648 | 1.0 | 0.0604828 | 1.0604828 | ✓ |

### N166：selection → action 33

path=[2, 8]；visits=18；children=5；K=5。最低访问优先：child.visits < 5；最少 2 次；候选 [33]，并列按 prior 抽样。trace 行 4920。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.2451319 | 0.1317073 | 0.3768392 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 0.4268483 | 0.0257031 | 0.4525514 |  |
| 15 | 4 | 0.1111044 | 65.8843913 | 0.0 | 0.1319852 | 0.1319852 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.3443704 | 0.1658352 | 0.5102056 |  |
| 33 | 2 | 0.1164611 | 73.965868 | 1.0 | 0.2305812 | 1.2305812 | ✓ |

### N180：expansion → action 19

path=[2, 8, 33]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 24 个代表动作。trace 行 4922。

bucket=1，uniform_random；到达 N182（新建）；closure=[]。

## iteration 196

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[8, 26, 33, 16, 36, 4, 11, 1, 30, 18, 39, 10, 25, 17, 5, 9]

### N0：selection → action 2

path=[]；visits=195；children=3；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4941。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7445671 | 0.7445671 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3676741 | 0.0734968 | 0.441171 |  |
| 2 | 100 | 0.3264006 | 67.2817152 | 1.0 | 0.0631793 | 1.0631793 | ✓ |

### N5：expansion → action 21

path=[2]；visits=100；children=10；K=11。已有 10 条动作边 < K=11，且尚余 4 个代表动作。trace 行 4943。

bucket=0，compatibility_richness_prior；到达 N183（新建）；closure=[]。

## iteration 197

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 6, 36, 23, 8, 35, 34, 1, 26, 17, 16, 15, 30, 12, 5]

### N0：selection → action 2

path=[]；visits=196；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4965。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7464738 | 0.7464738 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3571771 | 0.073685 | 0.4308621 |  |
| 2 | 101 | 0.3264006 | 67.4357981 | 1.0 | 0.0627201 | 1.0627201 | ✓ |

### N5：selection → action 21

path=[2]；visits=101；children=11；K=11。最低访问优先：child.visits < 5；最少 1 次；候选 [21]，并列按 prior 抽样。trace 行 4967。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5167905 | 0.1726101 | 0.6894006 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0414568 | 0.0514604 | 0.0929172 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1746 | 0.1746 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3215529 | 0.0959733 | 0.4175262 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2650066 | 0.1400198 | 0.4050264 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4437688 | 0.1013237 | 0.5450925 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.4793609 | 0.1123698 | 0.5917306 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.521692 | 0.0408035 | 0.5624955 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.4859318 | 0.0601908 | 0.5461225 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 0.7785081 | 0.0580361 | 0.8365442 |  |
| 21 | 1 | 0.0718016 | 71.8584611 | 1.0 | 0.5051182 | 1.5051182 | ✓ |

### N183：expansion → action 25

path=[2, 21]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 7 个代表动作。trace 行 4969。

bucket=0，compatibility_richness_prior；到达 N184（新建）；closure=[]。

## iteration 198

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[18, 21, 6, 36, 23, 9, 35, 1, 11, 5, 30, 4, 0, 16]

### N0：selection → action 2

path=[]；visits=197；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 4990。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7483757 | 0.7483757 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3589883 | 0.0738728 | 0.4328611 |  |
| 2 | 102 | 0.3264006 | 67.4085679 | 1.0 | 0.0622694 | 1.0622694 | ✓ |

### N5：selection → action 21

path=[2]；visits=102；children=11；K=11。最低访问优先：child.visits < 5；最少 2 次；候选 [21]，并列按 prior 抽样。trace 行 4992。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6443189 | 0.1734625 | 0.8177814 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0516871 | 0.0517146 | 0.1034016 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1754622 | 0.1754622 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4009025 | 0.0964473 | 0.4973497 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3304022 | 0.1407113 | 0.4711135 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5532777 | 0.1018241 | 0.6551017 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5976527 | 0.1129247 | 0.7105774 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6504299 | 0.041005 | 0.6914349 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6058451 | 0.060488 | 0.6663331 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 0.9706205 | 0.0583227 | 1.0289432 |  |
| 21 | 2 | 0.0718016 | 69.2291034 | 1.0 | 0.3384084 | 1.3384084 | ✓ |

### N183：expansion → action 8

path=[2, 21]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 6 个代表动作。trace 行 4994。

bucket=1，uniform_random；到达 N170（复用）；closure=[]。

## iteration 199

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 19, 34, 39, 13, 22, 9, 31, 24, 20, 26, 8, 6, 3]

### N0：selection → action 2

path=[]；visits=198；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5015。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7502727 | 0.7502727 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3447524 | 0.07406 | 0.4188125 |  |
| 2 | 103 | 0.3264006 | 67.6303007 | 1.0 | 0.061827 | 1.061827 | ✓ |

### N5：selection → action 21

path=[2]；visits=103；children=11；K=11。最低访问优先：child.visits < 5；最少 3 次；候选 [21]，并列按 prior 抽样。trace 行 5017。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.5540333 | 0.1743107 | 0.728344 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0444444 | 0.0519674 | 0.0964118 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1763202 | 0.1763202 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3447257 | 0.0969189 | 0.4416446 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2841044 | 0.1413994 | 0.4255038 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4757492 | 0.102322 | 0.5780712 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5139062 | 0.1134769 | 0.6273831 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.559288 | 0.0412055 | 0.6004935 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5209506 | 0.0607838 | 0.5817344 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 0.8346116 | 0.0586079 | 0.8932195 |  |
| 21 | 3 | 0.0718016 | 70.9654656 | 1.0 | 0.2550474 | 1.2550474 | ✓ |

### N183：selection → action 25

path=[2, 21]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [25]，并列按 prior 抽样。trace 行 5019。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 1 | 0.2828124 | 66.5997458 | 0.0 | 0.3428919 | 0.3428919 | ✓ |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.065092 | 1.065092 |  |

### N184：expansion → action 0

path=[2, 21, 25]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 18 个代表动作。trace 行 5021。

bucket=0，compatibility_richness_prior；到达 N185（新建）；closure=[]。

## iteration 200

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[26, 12, 5, 39, 22, 25, 15, 31, 19, 14, 32, 34, 28, 3]

### N0：selection → action 2

path=[]；visits=199；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5041。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7521649 | 0.7521649 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3555441 | 0.0742468 | 0.4297909 |  |
| 2 | 104 | 0.3264006 | 67.4605853 | 1.0 | 0.0613926 | 1.0613926 | ✓ |

### N5：selection → action 21

path=[2]；visits=104；children=11；K=11。最低访问优先：child.visits < 5；最少 4 次；候选 [21]，并列按 prior 抽样。trace 行 5043。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6638217 | 0.1751549 | 0.8389765 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0532516 | 0.0522191 | 0.1054707 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1771741 | 0.1771741 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4130373 | 0.0973882 | 0.5104255 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3404031 | 0.1420841 | 0.4824872 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5700247 | 0.1028175 | 0.6728422 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6157429 | 0.1140264 | 0.7297693 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6701177 | 0.041405 | 0.7115227 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6241833 | 0.0610782 | 0.6852615 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 1.0 | 0.0588917 | 1.0588917 |  |
| 21 | 4 | 0.0718016 | 68.7441185 | 0.9833743 | 0.205026 | 1.1884004 | ✓ |

### N183：expansion → action 0

path=[2, 21]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 5 个代表动作。trace 行 5045。

bucket=0，compatibility_richness_prior；到达 N186（新建）；closure=[]。

## iteration 201

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[16, 19, 14, 25, 26, 33, 13, 36, 5, 12, 31, 23, 8]

### N0：selection → action 2

path=[]；visits=200；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5065。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7540524 | 0.7540524 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3351179 | 0.0744331 | 0.4095511 |  |
| 2 | 105 | 0.3264006 | 67.7910535 | 1.0 | 0.0609661 | 1.0609661 | ✓ |

### N5：selection → action 21

path=[2]；visits=105；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5067。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.565464 | 0.1759949 | 0.741459 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0453613 | 0.0524696 | 0.0978309 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1780238 | 0.1780238 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3518381 | 0.0978553 | 0.4496934 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.289966 | 0.1427656 | 0.4327316 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4855649 | 0.1033106 | 0.5888755 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.5245091 | 0.1145733 | 0.6390824 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.5708272 | 0.0416036 | 0.6124308 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.5316989 | 0.0613711 | 0.59307 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 0.8518312 | 0.0591742 | 0.9110054 |  |
| 21 | 5 | 0.0718016 | 70.714974 | 1.0 | 0.1716745 | 1.1716745 | ✓ |

### N183：selection → action 0

path=[2, 21]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [0]，并列按 prior 抽样。trace 行 5069。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 2 | 0.2828124 | 64.3399116 | 0.0 | 0.2951143 | 0.2951143 |  |
| 8 | 5 | 0.1610609 | 74.4381899 | 0.7082294 | 0.0840334 | 0.7922629 |  |
| 0 | 1 | 0.1722104 | 78.5983958 | 1.0 | 0.2695519 | 1.2695519 | ✓ |

### N186：expansion → action 22

path=[2, 21, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 5071。

bucket=0，compatibility_richness_prior；到达 N187（新建）；closure=[]。

## iteration 202

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 23, 9, 25, 17, 5, 35, 39, 31, 28, 13, 12, 26, 8]

### N0：selection → action 2

path=[]；visits=201；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5090。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7559352 | 0.7559352 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3539221 | 0.074619 | 0.4285411 |  |
| 2 | 106 | 0.3264006 | 67.4854338 | 1.0 | 0.0605471 | 1.0605471 | ✓ |

### N5：selection → action 21

path=[2]；visits=106；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5092。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6638217 | 0.176831 | 0.8406527 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0532516 | 0.0527188 | 0.1059704 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1788696 | 0.1788696 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4130373 | 0.0983202 | 0.5113575 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3404031 | 0.1434438 | 0.4838469 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5700247 | 0.1038014 | 0.6738261 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6157429 | 0.1151176 | 0.7308605 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6701177 | 0.0418013 | 0.7119189 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6241833 | 0.0616627 | 0.685846 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 1.0 | 0.0594553 | 1.0594553 |  |
| 21 | 6 | 0.0718016 | 68.459994 | 0.9559016 | 0.1478486 | 1.1037502 | ✓ |

### N183：selection → action 0

path=[2, 21]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [25, 0]，并列按 prior 抽样。trace 行 5094。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 2 | 0.2828124 | 64.3399116 | 0.0 | 0.3232815 | 0.3232815 |  |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.092054 | 1.092054 |  |
| 0 | 2 | 0.1722104 | 67.891745 | 0.3517266 | 0.1968529 | 0.5485795 | ✓ |

### N186：expansion → action 38

path=[2, 21, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 5096。

bucket=1，uniform_random；到达 N188（新建）；closure=[]。

## iteration 203

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 39, 1, 13, 23, 38, 37, 11, 25, 0, 32, 28, 14, 4]

### N0：selection → action 2

path=[]；visits=202；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5116。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7578133 | 0.7578133 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3711782 | 0.0748044 | 0.4459826 |  |
| 2 | 107 | 0.3264006 | 67.2322204 | 1.0 | 0.0601355 | 1.0601355 | ✓ |

### N5：selection → action 8

path=[2]；visits=107；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5118。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6638217 | 0.1776632 | 0.8414848 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0532516 | 0.0529669 | 0.1062185 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1797113 | 0.1797113 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4130373 | 0.0987829 | 0.5118202 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3404031 | 0.1441188 | 0.4845219 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5700247 | 0.1042899 | 0.6743146 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6157429 | 0.1156593 | 0.7314023 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6701177 | 0.041998 | 0.7121156 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6241833 | 0.0619528 | 0.6861361 |  |
| 8 | 19 | 0.0824973 | 68.9160621 | 1.0 | 0.0597351 | 1.0597351 | ✓ |
| 21 | 7 | 0.0718016 | 67.02927 | 0.8175612 | 0.1299763 | 0.9475375 |  |

### N166：selection → action 33

path=[2, 8]；visits=19；children=5；K=5。最低访问优先：child.visits < 5；最少 3 次；候选 [33]，并列按 prior 抽样。trace 行 5120。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.2461457 | 0.1353164 | 0.3814621 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 0.4286136 | 0.0264075 | 0.4550211 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1130016 | 0.1130016 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.3457946 | 0.1703795 | 0.5161741 |  |
| 33 | 3 | 0.1164611 | 73.9325827 | 1.0 | 0.1776747 | 1.1776747 | ✓ |

### N180：selection → action 34

path=[2, 8, 33]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [34, 19]，并列按 prior 抽样。trace 行 5122。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 1 | 0.0376976 | 73.8812836 | 1.0 | 0.045706 | 1.045706 | ✓ |
| 19 | 1 | 0.0504112 | 73.8660122 | 0.0 | 0.0611203 | 0.0611203 |  |

### N181：expansion → action 6

path=[2, 8, 33, 34]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 5124。

bucket=0，compatibility_richness_prior；到达 N189（新建）；closure=[]。

## iteration 204

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 35, 38, 12, 36, 18, 17, 4, 11, 1, 31, 7, 25, 13]

### N0：selection → action 2

path=[]；visits=203；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5144。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7596868 | 0.7596868 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3703597 | 0.0749893 | 0.445349 |  |
| 2 | 108 | 0.3264006 | 67.243698 | 1.0 | 0.0597311 | 1.0597311 | ✓ |

### N5：selection → action 8

path=[2]；visits=108；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5146。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.6678986 | 0.1784914 | 0.84639 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0535786 | 0.0532138 | 0.1067925 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1805491 | 0.1805491 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.415574 | 0.0992434 | 0.5148174 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3424937 | 0.1447907 | 0.4872845 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5735256 | 0.1047761 | 0.6783017 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6195246 | 0.1161985 | 0.7357231 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6742332 | 0.0421938 | 0.716427 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6280168 | 0.0622417 | 0.6902585 |  |
| 8 | 20 | 0.0824973 | 68.852933 | 1.0 | 0.0571558 | 1.0571558 | ✓ |
| 21 | 7 | 0.0718016 | 67.02927 | 0.8225823 | 0.1305823 | 0.9531646 |  |

### N166：selection → action 33

path=[2, 8]；visits=20；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [14, 33]，并列按 prior 抽样。trace 行 5148。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.3057888 | 0.1388317 | 0.4446205 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 0.5324703 | 0.0270935 | 0.5595638 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1159372 | 0.1159372 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.4295836 | 0.1748057 | 0.6043892 |  |
| 33 | 4 | 0.1164611 | 72.3628073 | 1.0 | 0.1458324 | 1.1458324 | ✓ |

### N180：expansion → action 29

path=[2, 8, 33]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 23 个代表动作。trace 行 5150。

bucket=0，compatibility_richness_prior；到达 N190（新建）；closure=[]。

## iteration 205

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[31, 11, 9, 3, 12, 1, 22, 23, 30, 26, 33, 34, 16]

### N0：selection → action 2

path=[]；visits=204；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5170。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7615556 | 0.7615556 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3981893 | 0.0751738 | 0.4733631 |  |
| 2 | 109 | 0.3264006 | 66.8799271 | 1.0 | 0.0593337 | 1.0593337 | ✓ |

### N5：selection → action 8

path=[2]；visits=109；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5172。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7189388 | 0.1793159 | 0.8982546 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.057673 | 0.0534596 | 0.1111327 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1813831 | 0.1813831 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4473318 | 0.0997018 | 0.5470336 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3686668 | 0.1454595 | 0.5141263 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6173538 | 0.10526 | 0.7226139 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6668681 | 0.1167353 | 0.7836033 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7257575 | 0.0423887 | 0.7681462 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6760092 | 0.0625291 | 0.7385384 |  |
| 8 | 21 | 0.0824973 | 68.1231934 | 1.0 | 0.0548098 | 1.0548098 | ✓ |
| 21 | 7 | 0.0718016 | 67.02927 | 0.8854433 | 0.1311854 | 1.0166287 |  |

### N166：selection → action 14

path=[2, 8]；visits=21；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [14]，并列按 prior 抽样。trace 行 5174。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.5742834 | 0.1422602 | 0.7165435 |  |
| 14 | 4 | 0.0216367 | 69.3339556 | 1.0 | 0.0277626 | 1.0277626 | ✓ |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1188003 | 0.1188003 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.8067746 | 0.1791225 | 0.9858971 |  |
| 33 | 5 | 0.1164611 | 68.5959258 | 0.7860513 | 0.1245281 | 0.9105794 |  |

### N168：expansion → action 24

path=[2, 8, 14]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 2 个代表动作。trace 行 5176。

bucket=0，compatibility_richness_prior；到达 N191（新建）；closure=[]。

## iteration 206

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[7, 9, 30, 0, 25, 12, 10, 23, 33, 21, 13, 26, 4]

### N0：selection → action 2

path=[]；visits=205；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5195。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7634199 | 0.7634199 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.3943804 | 0.0753578 | 0.4697382 |  |
| 2 | 110 | 0.3264006 | 66.9266823 | 1.0 | 0.0589431 | 1.0589431 | ✓ |

### N5：selection → action 8

path=[2]；visits=110；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5197。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.717006 | 0.1801366 | 0.8971425 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.057518 | 0.0537043 | 0.1112223 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1822132 | 0.1822132 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4461292 | 0.1001581 | 0.5462873 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3676756 | 0.1461252 | 0.5138009 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6156942 | 0.1057418 | 0.7214359 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6650753 | 0.1172695 | 0.7823448 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7238064 | 0.0425827 | 0.7663891 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6741919 | 0.0628153 | 0.7370072 |  |
| 8 | 22 | 0.0824973 | 68.1489345 | 1.0 | 0.0526667 | 1.0526667 | ✓ |
| 21 | 7 | 0.0718016 | 67.02927 | 0.8830629 | 0.1317858 | 1.0148487 |  |

### N166：selection → action 14

path=[2, 8]；visits=22；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5199。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.5965741 | 0.1456079 | 0.742182 |  |
| 14 | 5 | 0.0216367 | 69.2050641 | 1.0 | 0.0236799 | 1.0236799 | ✓ |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.121596 | 0.121596 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 0.8380895 | 0.1833377 | 1.0214272 |  |
| 33 | 5 | 0.1164611 | 68.5959258 | 0.8165618 | 0.1274586 | 0.9440203 |  |

### N168：selection → action 24

path=[2, 8, 14]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [25, 24]，并列按 prior 抽样。trace 行 5201。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.2855866 | 67.2795538 | 0.0 | 0.2980092 | 0.2980092 |  |
| 25 | 1 | 0.3301689 | 72.5181066 | 1.0 | 0.516796 | 1.516796 |  |
| 24 | 1 | 0.1720735 | 68.6894978 | 0.2691476 | 0.2693377 | 0.5384853 | ✓ |

### N191：expansion → action 6

path=[2, 8, 14, 24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 5203。

bucket=0，compatibility_richness_prior；到达 N192（新建）；closure=[]。

## iteration 207

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[23, 39, 16, 8, 36, 33, 32, 12, 4, 11, 17, 26, 31, 6]

### N0：selection → action 2

path=[]；visits=206；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5222。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7652796 | 0.7652796 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4113155 | 0.0755414 | 0.4868568 |  |
| 2 | 111 | 0.3264006 | 66.7254356 | 1.0 | 0.0585591 | 1.0585591 | ✓ |

### N5：selection → action 21

path=[2]；visits=111；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5224。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7483092 | 0.1809535 | 0.9292627 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0600291 | 0.0539479 | 0.113977 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1830396 | 0.1830396 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4656064 | 0.1006124 | 0.5662187 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3837277 | 0.1467879 | 0.5305157 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6425743 | 0.1062213 | 0.7487957 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6941113 | 0.1178014 | 0.8119127 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7554065 | 0.0427758 | 0.7981823 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7036259 | 0.0631002 | 0.7667261 |  |
| 8 | 23 | 0.0824973 | 67.7483964 | 1.0 | 0.0507012 | 1.0507012 |  |
| 21 | 7 | 0.0718016 | 67.02927 | 0.9216159 | 0.1323835 | 1.0539994 | ✓ |

### N183：selection → action 25

path=[2, 21]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [25]，并列按 prior 抽样。trace 行 5226。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 2 | 0.2828124 | 64.3399116 | 0.0 | 0.349184 | 0.349184 | ✓ |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.0994297 | 1.0994297 |  |
| 0 | 3 | 0.1722104 | 64.7428053 | 0.0398973 | 0.159469 | 0.1993663 |  |

### N184：expansion → action 30

path=[2, 21, 25]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 17 个代表动作。trace 行 5228。

bucket=1，uniform_random；到达 N193（新建）；closure=[]。

## iteration 208

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 8, 27, 6, 5, 35, 33, 17, 1, 28, 23, 2, 15]

### N0：selection → action 2

path=[]；visits=207；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5248。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7671349 | 0.7671349 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4186896 | 0.0757245 | 0.4944141 |  |
| 2 | 112 | 0.3264006 | 66.6428928 | 1.0 | 0.0581816 | 1.0581816 | ✓ |

### N5：selection → action 8

path=[2]；visits=112；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5250。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7483092 | 0.1817668 | 0.930076 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0600291 | 0.0541903 | 0.1142195 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1838622 | 0.1838622 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4656064 | 0.1010645 | 0.5666709 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3837277 | 0.1474477 | 0.5311754 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6425743 | 0.1066987 | 0.7492731 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6941113 | 0.1183308 | 0.8124421 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7554065 | 0.042968 | 0.7983746 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7036259 | 0.0633838 | 0.7670097 |  |
| 8 | 23 | 0.0824973 | 67.7483964 | 1.0 | 0.050929 | 1.050929 | ✓ |
| 21 | 8 | 0.0718016 | 66.5713223 | 0.8717 | 0.1182031 | 0.9899031 |  |

### N166：selection → action 25

path=[2, 8]；visits=23；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5252。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.7118262 | 0.1488804 | 0.8607066 |  |
| 14 | 6 | 0.0216367 | 67.4936466 | 0.5782404 | 0.0207532 | 0.5989937 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1243288 | 0.1243288 |  |
| 25 | 5 | 0.1675189 | 68.6674123 | 1.0 | 0.1874582 | 1.1874582 | ✓ |
| 33 | 5 | 0.1164611 | 68.5959258 | 0.9743133 | 0.1303231 | 1.1046365 |  |

### N11：selection → action 20

path=[24, 3, 9]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [20, 37, 27]，并列按 prior 抽样。trace 行 5254。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 20 | 1 | 0.0527497 | 65.5435629 | 1.0 | 0.0825664 | 1.0825664 | ✓ |
| 37 | 1 | 0.0649189 | 57.7238561 | 0.0 | 0.1016142 | 0.1016142 |  |
| 27 | 1 | 0.060169 | 65.053189 | 0.93729 | 0.0941794 | 1.0314693 |  |

### N46：expansion → action 37

path=[24, 3, 9, 20]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 5256。

bucket=0，compatibility_richness_prior；到达 N194（新建）；closure=[]。

## iteration 209

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[32, 36, 35, 14, 4, 31, 16, 21, 37, 12, 29, 3]

### N0：selection → action 2

path=[]；visits=208；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5275。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7689856 | 0.7689856 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4321559 | 0.0759072 | 0.5080631 |  |
| 2 | 113 | 0.3264006 | 66.4994276 | 1.0 | 0.0578104 | 1.0578104 | ✓ |

### N5：selection → action 8

path=[2]；visits=113；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5277。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.773201 | 0.1825764 | 0.9557774 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0620259 | 0.0544317 | 0.1164576 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1846812 | 0.1846812 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4810943 | 0.1015147 | 0.582609 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.396492 | 0.1481045 | 0.5445965 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6639489 | 0.107174 | 0.7711229 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.7172002 | 0.1188579 | 0.8360581 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.7805344 | 0.0431594 | 0.8236938 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7270313 | 0.0636661 | 0.7906975 |  |
| 8 | 24 | 0.0824973 | 67.4530442 | 1.0 | 0.0491097 | 1.0491097 | ✓ |
| 21 | 8 | 0.0718016 | 66.5713223 | 0.9006962 | 0.1187296 | 1.0194258 |  |

### N166：selection → action 33

path=[2, 8]；visits=24；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5279。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 0.7305927 | 0.1520825 | 0.8826752 |  |
| 14 | 6 | 0.0216367 | 67.4936466 | 0.5934851 | 0.0211996 | 0.6146847 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1270029 | 0.1270029 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.0419926 | 0.1641343 | 0.2061269 |  |
| 33 | 5 | 0.1164611 | 68.5959258 | 1.0 | 0.1331261 | 1.1331261 | ✓ |

### N180：selection → action 19

path=[2, 8, 33]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [19, 29]，并列按 prior 抽样。trace 行 5281。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.0376976 | 70.7673823 | 0.8476404 | 0.0393374 | 0.8869779 |  |
| 19 | 1 | 0.0504112 | 73.8660122 | 1.0 | 0.078906 | 1.078906 | ✓ |
| 29 | 1 | 0.0172087 | 53.5283999 | 0.0 | 0.0269359 | 0.0269359 |  |

### N182：expansion → action 27

path=[2, 8, 33, 19]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 5283。

bucket=0，compatibility_richness_prior；到达 N195（新建）；closure=[]。

## iteration 210

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 26, 7, 13, 4, 10, 38, 32, 39, 6, 9, 28, 18]

### N0：selection → action 2

path=[]；visits=209；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5301。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7708319 | 0.7708319 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4509269 | 0.0760895 | 0.5270164 |  |
| 2 | 114 | 0.3264006 | 66.313743 | 1.0 | 0.0574453 | 1.0574453 | ✓ |

### N5：selection → action 21

path=[2]；visits=114；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5303。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.8054324 | 0.1833825 | 0.9888149 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0646115 | 0.054672 | 0.1192836 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1854966 | 0.1854966 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5011491 | 0.1019629 | 0.603112 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4130201 | 0.1487583 | 0.5617784 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6916261 | 0.1076472 | 0.7992733 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.7470972 | 0.1193827 | 0.8664798 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8130715 | 0.04335 | 0.8564215 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7573381 | 0.0639472 | 0.8212854 |  |
| 8 | 25 | 0.0824973 | 67.0977271 | 1.0 | 0.0474293 | 1.0474293 |  |
| 21 | 8 | 0.0718016 | 66.5713223 | 0.9382424 | 0.1192538 | 1.0574962 | ✓ |

### N183：selection → action 25

path=[2, 21]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 3 次；候选 [25, 0]，并列按 prior 抽样。trace 行 5305。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 3 | 0.2828124 | 64.0151706 | 0.0 | 0.27997 | 0.27997 | ✓ |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.1062948 | 1.1062948 |  |
| 0 | 3 | 0.1722104 | 64.7428053 | 0.0698104 | 0.1704796 | 0.2402899 |  |

### N184：selection → action 30

path=[2, 21, 25]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [0, 30]，并列按 prior 抽样。trace 行 5307。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0.0704608 | 62.0800774 | 0.0 | 0.0854292 | 0.0854292 |  |
| 30 | 1 | 0.0685247 | 63.3656885 | 1.0 | 0.0830818 | 1.0830818 | ✓ |

### N193：expansion → action 1

path=[2, 21, 25, 30]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 5309。

bucket=0，compatibility_richness_prior；到达 N196（新建）；closure=[]。

## iteration 211

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 4, 33, 39, 1, 26, 32, 10, 31, 28, 9, 29, 18, 21, 19]

### N0：selection → action 2

path=[]；visits=210；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5328。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7726738 | 0.7726738 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4688938 | 0.0762713 | 0.5451651 |  |
| 2 | 115 | 0.3264006 | 66.1499385 | 1.0 | 0.0570861 | 1.0570861 | ✓ |

### N5：selection → action 8

path=[2]；visits=115；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5330。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.8054324 | 0.1841851 | 0.9896175 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0646115 | 0.0549113 | 0.1195228 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1863084 | 0.1863084 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5011491 | 0.1024091 | 0.6035582 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4130201 | 0.1494094 | 0.5624294 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6916261 | 0.1081183 | 0.7997444 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.7470972 | 0.1199051 | 0.8670023 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8130715 | 0.0435397 | 0.8566112 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7573381 | 0.0642271 | 0.8215652 |  |
| 8 | 25 | 0.0824973 | 67.0977271 | 1.0 | 0.0476369 | 1.0476369 | ✓ |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.8415665 | 0.1077981 | 0.9493646 |  |

### N166：expansion → action 32

path=[2, 8]；visits=25；children=5；K=6。已有 5 条动作边 < K=6，且尚余 4 个代表动作。trace 行 5332。

bucket=1，uniform_random；到达 N49（复用）；closure=[]。

## iteration 212

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[7, 33, 10, 37, 0, 29, 28, 20, 5, 4, 38, 21]

### N0：selection → action 2

path=[]；visits=211；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5354。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7745113 | 0.7745113 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.466383 | 0.0764527 | 0.5428357 |  |
| 2 | 116 | 0.3264006 | 66.1720708 | 1.0 | 0.0567328 | 1.0567328 | ✓ |

### N5：selection → action 8

path=[2]；visits=116；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5356。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.8052813 | 0.1849842 | 0.9902654 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0645994 | 0.0551495 | 0.1197489 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1871167 | 0.1871167 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.501055 | 0.1028534 | 0.6039085 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4129426 | 0.1500576 | 0.5630001 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6914963 | 0.1085874 | 0.8000837 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.746957 | 0.1204253 | 0.8673823 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8129189 | 0.0437286 | 0.8566475 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.757196 | 0.0645057 | 0.8217018 |  |
| 8 | 26 | 0.0824973 | 67.0993267 | 1.0 | 0.0460716 | 1.0460716 | ✓ |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.8414085 | 0.1082658 | 0.9496744 |  |

### N166：selection → action 12

path=[2, 8]；visits=26；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5358。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 5 | 0.1330445 | 67.8654187 | 1.0 | 0.1582925 | 1.1582925 | ✓ |
| 14 | 6 | 0.0216367 | 67.4936466 | 0.8123337 | 0.0220652 | 0.8343989 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1321888 | 0.1321888 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.0574775 | 0.1708364 | 0.2283139 |  |
| 33 | 6 | 0.1164611 | 66.9249576 | 0.525266 | 0.1187675 | 0.6440334 |  |
| 32 | 5 | 0.1859329 | 67.1393184 | 0.6334729 | 0.2212177 | 0.8546905 |  |

### N167：selection → action 17

path=[2, 8, 12]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [17, 3]，并列按 prior 抽样。trace 行 5360。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0.0206411 | 70.782895 | 1.0 | 0.0215389 | 1.0215389 |  |
| 17 | 1 | 0.0423435 | 64.9341039 | 0.0 | 0.0662781 | 0.0662781 | ✓ |
| 3 | 1 | 0.0185985 | 65.5173906 | 0.0997277 | 0.0291112 | 0.128839 |  |

### N173：expansion → action 36

path=[2, 8, 12, 17]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 21 个代表动作。trace 行 5362。

bucket=0，compatibility_richness_prior；到达 N197（新建）；closure=[]。

## iteration 213

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 10, 7, 37, 39, 0, 25, 24, 30, 32, 11, 15, 13]

### N0：selection → action 2

path=[]；visits=212；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5380。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7763445 | 0.7763445 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.4592854 | 0.0766336 | 0.535919 |  |
| 2 | 117 | 0.3264006 | 66.2359443 | 1.0 | 0.0563852 | 1.0563852 | ✓ |

### N5：selection → action 8

path=[2]；visits=117；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5382。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.7983723 | 0.1857798 | 0.9841521 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0640452 | 0.0553867 | 0.1194319 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1879215 | 0.1879215 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4967562 | 0.1032958 | 0.600052 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4093997 | 0.150703 | 0.5601027 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6855635 | 0.1090544 | 0.794618 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.7405484 | 0.1209433 | 0.8614917 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8059444 | 0.0439167 | 0.8498611 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.7506996 | 0.0647832 | 0.8154828 |  |
| 8 | 27 | 0.0824973 | 67.1731032 | 1.0 | 0.0446172 | 1.0446172 | ✓ |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.8341896 | 0.1087315 | 0.9429211 |  |

### N166：selection → action 12

path=[2, 8]；visits=27；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5384。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 6 | 0.1330445 | 68.0697309 | 1.0 | 0.1382639 | 1.1382639 | ✓ |
| 14 | 6 | 0.0216367 | 67.4936466 | 0.7363869 | 0.0224856 | 0.7588724 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0 | 0.1347069 | 0.1347069 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.0521038 | 0.1740907 | 0.2261945 |  |
| 33 | 6 | 0.1164611 | 66.9249576 | 0.4761577 | 0.1210299 | 0.5971876 |  |
| 32 | 5 | 0.1859329 | 67.1393184 | 0.5742481 | 0.2254317 | 0.7996798 |  |

### N167：selection → action 3

path=[2, 8, 12]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [3]，并列按 prior 抽样。trace 行 5386。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0.0206411 | 70.782895 | 1.0 | 0.0235947 | 1.0235947 |  |
| 17 | 2 | 0.0423435 | 67.0126978 | 0.2839818 | 0.0484027 | 0.3323844 |  |
| 3 | 1 | 0.0185985 | 65.5173906 | 0.0 | 0.0318897 | 0.0318897 | ✓ |

### N179：expansion → action 29

path=[2, 8, 12, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 5388。

bucket=0，compatibility_richness_prior；到达 N198（新建）；closure=[]。

## iteration 214

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[18, 39, 35, 26, 11, 32, 34, 28, 29, 17, 30, 23, 12, 6, 16]

### N0：selection → action 2

path=[]；visits=213；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5407。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7781733 | 0.7781733 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.496053 | 0.0768141 | 0.5728671 |  |
| 2 | 118 | 0.3264006 | 65.9248533 | 1.0 | 0.0560431 | 1.0560431 | ✓ |

### N5：selection → action 8

path=[2]；visits=118；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5409。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.8531758 | 0.186572 | 1.0397478 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0684415 | 0.0556229 | 0.1240644 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1887229 | 0.1887229 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5308555 | 0.1037363 | 0.6345918 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4375025 | 0.1513456 | 0.5888482 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7326234 | 0.1095195 | 0.8421428 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.7913826 | 0.121459 | 0.9128417 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8612677 | 0.044104 | 0.9053716 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8022306 | 0.0650594 | 0.8672901 |  |
| 8 | 28 | 0.0824973 | 66.6207431 | 1.0 | 0.0432624 | 1.0432624 | ✓ |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.8914517 | 0.1091951 | 1.0006469 |  |

### N166：selection → action 32

path=[2, 8]；visits=28；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5411。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.0 | 0.1232009 | 0.1232009 |  |
| 14 | 6 | 0.0216367 | 67.4936466 | 1.0 | 0.0228982 | 1.0228982 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.0864009 | 0.1371788 | 0.2235796 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.1510435 | 0.1772853 | 0.3283288 |  |
| 33 | 6 | 0.1164611 | 66.9249576 | 0.6771464 | 0.1232508 | 0.8003973 |  |
| 32 | 5 | 0.1859329 | 67.1393184 | 0.7988424 | 0.2295684 | 1.0284108 | ✓ |

### N49：expansion → action 4

path=[2, 36, 8]；visits=5；children=2；K=3。已有 2 条动作边 < K=3，且尚余 23 个代表动作。trace 行 5413。

bucket=0，compatibility_richness_prior；到达 N199（新建）；closure=[]。

## iteration 215

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[11, 21, 39, 16, 22, 14, 36, 13, 6, 33, 1, 5, 19]

### N0：selection → action 2

path=[]；visits=214；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5434。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7799979 | 0.7799979 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5166237 | 0.0769942 | 0.593618 |  |
| 2 | 119 | 0.3264006 | 65.7701212 | 1.0 | 0.0557063 | 1.0557063 | ✓ |

### N5：selection → action 9

path=[2]；visits=119；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5436。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 5 | 0.0736086 | 65.4392872 | 0.8837654 | 0.1873609 | 1.0711263 | ✓ |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0708954 | 0.0558581 | 0.1267535 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1895208 | 0.1895208 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5498887 | 0.1041749 | 0.6540637 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4531887 | 0.1519856 | 0.6051743 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7588907 | 0.1099826 | 0.8688733 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8197568 | 0.1219726 | 0.9417294 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8921474 | 0.0442904 | 0.9364379 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8309937 | 0.0653345 | 0.8963282 |  |
| 8 | 29 | 0.0824973 | 66.3422226 | 1.0 | 0.0419972 | 1.0419972 |  |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.9234137 | 0.1096569 | 1.0330706 |  |

### N6：selection → action 15

path=[2, 9]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [15]，并列按 prior 抽样。trace 行 5438。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.144154 | 67.8542711 | 1.0 | 0.1504245 | 1.1504245 |  |
| 15 | 1 | 0.0145787 | 64.673442 | 0.0 | 0.0228193 | 0.0228193 | ✓ |
| 24 | 3 | 0.0455751 | 65.3839303 | 0.2233658 | 0.0356681 | 0.2590339 |  |

### N19：expansion → action 18

path=[2, 9, 15]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 13 个代表动作。trace 行 5440。

bucket=0，compatibility_richness_prior；到达 N200（新建）；closure=[]。

## iteration 216

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 28, 38, 25, 31, 36, 6, 23, 29, 18, 7, 24, 17, 13]

### N0：selection → action 2

path=[]；visits=215；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5459。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7818182 | 0.7818182 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5073667 | 0.0771739 | 0.5845406 |  |
| 2 | 120 | 0.3264006 | 65.8381997 | 1.0 | 0.0553749 | 1.0553749 | ✓ |

### N5：selection → action 9

path=[2]；visits=120；children=11；K=11。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5461。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 6 | 0.0736086 | 68.8215212 | 1.0 | 0.1612684 | 1.1612684 | ✓ |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0537428 | 0.0560923 | 0.1098352 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1903155 | 0.1903155 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4168479 | 0.1046117 | 0.5214596 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3435436 | 0.1526228 | 0.4961664 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5752836 | 0.1104437 | 0.6857273 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6214236 | 0.122484 | 0.7439076 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6763 | 0.0444762 | 0.7207761 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6299419 | 0.0656085 | 0.6955503 |  |
| 8 | 29 | 0.0824973 | 66.3422226 | 0.7580585 | 0.0421732 | 0.8002318 |  |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.7000017 | 0.1101166 | 0.8101183 |  |

### N6：selection → action 0

path=[2, 9]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [0, 15]，并列按 prior 抽样。trace 行 5463。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.144154 | 67.8542711 | 0.6797957 | 0.1647818 | 0.8445774 | ✓ |
| 15 | 2 | 0.0145787 | 69.0178764 | 1.0 | 0.0166648 | 1.0166648 |  |
| 24 | 3 | 0.0455751 | 65.3839303 | 0.0 | 0.0390725 | 0.0390725 |  |

### N10：expansion → action 4

path=[2, 9, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 5465。

bucket=1，uniform_random；到达 N201（新建）；closure=[]。

## iteration 217

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[12, 38, 3, 37, 5, 24, 31, 33, 11, 17, 8, 15, 27, 13, 25, 7]

### N0：selection → action 2

path=[]；visits=216；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5485。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7836343 | 0.7836343 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5276282 | 0.0773532 | 0.6049814 |  |
| 2 | 121 | 0.3264006 | 65.6922993 | 1.0 | 0.0550486 | 1.0550486 | ✓ |

### N5：expansion → action 18

path=[2]；visits=121；children=11；K=12。已有 11 条动作边 < K=12，且尚余 3 个代表动作。trace 行 5487。

bucket=1，uniform_random；到达 N98（复用）；closure=[]。

## iteration 218

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[24, 7, 8, 32, 29, 33, 38, 18, 9, 3, 31, 2, 17]

### N0：selection → action 2

path=[]；visits=217；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5510。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7854462 | 0.7854462 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5190076 | 0.077532 | 0.5965397 |  |
| 2 | 122 | 0.3264006 | 65.7529826 | 1.0 | 0.0547273 | 1.0547273 | ✓ |

### N5：selection → action 18

path=[2]；visits=122；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5512。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.522052 | 0.1422809 | 0.6643329 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.054019 | 0.0565578 | 0.1105768 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1918949 | 0.1918949 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4189898 | 0.1054799 | 0.5244697 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3453088 | 0.1538894 | 0.4991983 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5782396 | 0.1113603 | 0.6895999 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.6246167 | 0.1235005 | 0.7481172 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.6797751 | 0.0448453 | 0.7246203 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.6331787 | 0.0661529 | 0.6993317 |  |
| 8 | 29 | 0.0824973 | 66.3422226 | 0.7619537 | 0.0425232 | 0.804477 |  |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.7035985 | 0.1110305 | 0.814629 |  |
| 18 | 14 | 0.0773162 | 68.7691349 | 1.0 | 0.0797053 | 1.0797053 | ✓ |

### N98：selection → action 27

path=[0, 20]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [19, 4, 27, 29]，并列按 prior 抽样。trace 行 5514。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 | 3 | 0.0544664 | 69.445459 | 1.0 | 0.0713282 | 1.0713282 |  |
| 4 | 3 | 0.0073295 | 62.1078912 | 0.2090141 | 0.0095985 | 0.2186126 |  |
| 27 | 3 | 0.0554026 | 65.6739537 | 0.5934337 | 0.0725542 | 0.6659879 | ✓ |
| 29 | 3 | 0.0470672 | 60.1689759 | 0.0 | 0.0616383 | 0.0616383 |  |

### N102：selection → action 37

path=[0, 20, 27]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [37]，并列按 prior 抽样。trace 行 5516。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 5 | 2 | 0.064174 | 74.8286885 | 1.0 | 0.0518713 | 1.0518713 |  |
| 37 | 1 | 0.0440677 | 58.9679087 | 0.0 | 0.0534292 | 0.0534292 | ✓ |

### N109：expansion → action 19

path=[0, 20, 27, 37]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 5518。

bucket=0，compatibility_richness_prior；到达 N202（新建）；closure=[]。

## iteration 219

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[26, 4, 7, 34, 18, 10, 17, 1, 12, 9, 11, 29]

### N0：selection → action 2

path=[]；visits=218；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5537。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7872539 | 0.7872539 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5417713 | 0.0777105 | 0.6194818 |  |
| 2 | 123 | 0.3264006 | 65.5969248 | 1.0 | 0.0544108 | 1.0544108 | ✓ |

### N5：selection → action 8

path=[2]；visits=123；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5539。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6851492 | 0.1428629 | 0.828012 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0708954 | 0.0567891 | 0.1276845 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1926797 | 0.1926797 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5498887 | 0.1059113 | 0.6558001 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4531887 | 0.1545188 | 0.6077075 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7588907 | 0.1118157 | 0.8707065 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8197568 | 0.1240056 | 0.9437624 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.8921474 | 0.0450287 | 0.9371761 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8309937 | 0.0664235 | 0.8974172 |  |
| 8 | 29 | 0.0824973 | 66.3422226 | 1.0 | 0.0426972 | 1.0426972 | ✓ |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.9234137 | 0.1114846 | 1.0348983 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5989423 | 0.0750294 | 0.6739716 |  |

### N166：selection → action 14

path=[2, 8]；visits=29；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5541。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.6213706 | 0.1253816 | 0.7467522 |  |
| 14 | 6 | 0.0216367 | 67.4936466 | 1.0 | 0.0233035 | 1.0233035 | ✓ |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.6540845 | 0.1396069 | 0.7936914 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.6785601 | 0.1804233 | 0.8589834 |  |
| 33 | 6 | 0.1164611 | 66.9249576 | 0.8777581 | 0.1254324 | 1.0031906 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2002559 | 0.2002559 |  |

### N168：selection → action 25

path=[2, 8, 14]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [25]，并列按 prior 抽样。trace 行 5543。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.2855866 | 67.2795538 | 0.3982187 | 0.3264527 | 0.7246714 |  |
| 25 | 1 | 0.3301689 | 72.5181066 | 1.0 | 0.5661217 | 1.5661217 | ✓ |
| 24 | 2 | 0.1720735 | 63.8130286 | 0.0 | 0.1966964 | 0.1966964 |  |

### N175：expansion → action 37

path=[2, 8, 14, 25]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 5545。

bucket=0，compatibility_richness_prior；到达 N203（新建）；closure=[]。

## iteration 220

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[4, 23, 31, 25, 28, 15, 14, 11, 1, 27, 35, 12, 19, 13, 6]

### N0：selection → action 2

path=[]；visits=219；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5563。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7890574 | 0.7890574 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5555161 | 0.0778885 | 0.6334046 |  |
| 2 | 124 | 0.3264006 | 65.5088887 | 1.0 | 0.0540992 | 1.0540992 | ✓ |

### N5：selection → action 21

path=[2]；visits=124；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5565。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7013556 | 0.1434424 | 0.844798 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0725723 | 0.0570195 | 0.1295919 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1934614 | 0.1934614 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5628957 | 0.106341 | 0.6692367 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4639084 | 0.1551457 | 0.6190541 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7768414 | 0.1122693 | 0.8891108 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8391472 | 0.1245087 | 0.9636559 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9132502 | 0.0452113 | 0.9584615 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8506499 | 0.066693 | 0.9173429 |  |
| 8 | 30 | 0.0824973 | 66.1627203 | 1.0 | 0.0414875 | 1.0414875 |  |
| 21 | 9 | 0.0718016 | 65.7472839 | 0.945256 | 0.1119369 | 1.0571929 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6131096 | 0.0753337 | 0.6884433 |  |

### N183：expansion → action 3

path=[2, 21]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 4 个代表动作。trace 行 5567。

bucket=1，uniform_random；到达 N204（新建）；closure=[]。

## iteration 221

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 35, 36, 12, 30, 10, 18, 11, 24, 7, 5, 16, 19]

### N0：selection → action 2

path=[]；visits=220；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5588。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7908569 | 0.7908569 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5583317 | 0.0780661 | 0.6363978 |  |
| 2 | 125 | 0.3264006 | 65.49139 | 1.0 | 0.0537923 | 1.0537923 | ✓ |

### N5：selection → action 8

path=[2]；visits=125；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5590。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7013556 | 0.1440197 | 0.8453753 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0725723 | 0.057249 | 0.1298213 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1942399 | 0.1942399 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5628957 | 0.1067689 | 0.6696647 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4639084 | 0.15577 | 0.6196784 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7768414 | 0.1127211 | 0.8895626 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8391472 | 0.1250097 | 0.9641569 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9132502 | 0.0453933 | 0.9586435 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8506499 | 0.0669614 | 0.9176113 |  |
| 8 | 30 | 0.0824973 | 66.1627203 | 1.0 | 0.0416544 | 1.0416544 | ✓ |
| 21 | 10 | 0.0718016 | 65.6294705 | 0.9297312 | 0.1021703 | 1.0319015 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6131096 | 0.0756369 | 0.6887465 |  |

### N166：selection → action 33

path=[2, 8]；visits=30；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5592。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.7079064 | 0.1275251 | 0.8354314 |  |
| 14 | 7 | 0.0216367 | 66.5598616 | 0.9105918 | 0.0207391 | 0.9313309 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.7451762 | 0.1419935 | 0.8871697 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.7730604 | 0.1835077 | 0.9565682 |  |
| 33 | 6 | 0.1164611 | 66.9249576 | 1.0 | 0.1275767 | 1.1275767 | ✓ |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2036793 | 0.2036793 |  |

### N180：selection → action 29

path=[2, 8, 33]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [29]，并列按 prior 抽样。trace 行 5594。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.0376976 | 70.7673823 | 1.0 | 0.043092 | 1.043092 |  |
| 19 | 2 | 0.0504112 | 66.2180642 | 0.7361029 | 0.0576248 | 0.7937277 |  |
| 29 | 1 | 0.0172087 | 53.5283999 | 0.0 | 0.0295069 | 0.0295069 | ✓ |

### N190：expansion → action 25

path=[2, 8, 33, 29]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 5596。

bucket=0，compatibility_richness_prior；到达 N205（新建）；closure=[37]。

## iteration 222

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[39, 8, 38, 37, 24, 12, 22, 32, 6, 25, 15, 10, 31]

### N0：selection → action 2

path=[]；visits=221；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5615。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7926522 | 0.7926522 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.579049 | 0.0782434 | 0.6572923 |  |
| 2 | 126 | 0.3264006 | 65.3678639 | 1.0 | 0.0534898 | 1.0534898 | ✓ |

### N5：selection → action 21

path=[2]；visits=126；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5617。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7242248 | 0.1445946 | 0.8688194 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0749387 | 0.0574775 | 0.1324162 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1950153 | 0.1950153 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5812501 | 0.1071951 | 0.6884453 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4790351 | 0.1563919 | 0.6354269 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.802172 | 0.1131711 | 0.9153431 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8665093 | 0.1255088 | 0.9920181 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9430286 | 0.0455745 | 0.9886031 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8783871 | 0.0672287 | 0.9456158 |  |
| 8 | 31 | 0.0824973 | 65.9230882 | 1.0 | 0.0405138 | 1.0405138 |  |
| 21 | 10 | 0.0718016 | 65.6294705 | 0.960047 | 0.1025782 | 1.0626252 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6331013 | 0.0759388 | 0.7090401 |  |

### N183：selection → action 3

path=[2, 21]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [3]，并列按 prior 抽样。trace 行 5619。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 4 | 0.2828124 | 62.8001221 | 0.0 | 0.2504128 | 0.2504128 |  |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.1188412 | 1.1188412 |  |
| 0 | 3 | 0.1722104 | 64.7428053 | 0.1669249 | 0.190602 | 0.3575268 |  |
| 3 | 1 | 0.0151285 | 64.5691497 | 0.1520035 | 0.0334884 | 0.185492 | ✓ |

### N204：expansion → action 20

path=[2, 21, 3]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 12 个代表动作。trace 行 5621。

bucket=0，compatibility_richness_prior；到达 N206（新建）；closure=[]。

## iteration 223

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[1, 29, 13, 31, 27, 34, 36, 28, 4, 39, 0, 16, 12, 11]

### N0：selection → action 2

path=[]；visits=222；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5640。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7944435 | 0.7944435 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5831341 | 0.0784202 | 0.6615543 |  |
| 2 | 127 | 0.3264006 | 65.3445422 | 1.0 | 0.0531919 | 1.0531919 | ✓ |

### N5：selection → action 8

path=[2]；visits=127；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5642。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7242248 | 0.1451673 | 0.869392 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0749387 | 0.0577052 | 0.1326439 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1957877 | 0.1957877 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5812501 | 0.1076197 | 0.6888698 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4790351 | 0.1570112 | 0.6360463 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.802172 | 0.1136193 | 0.9157913 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8665093 | 0.1260058 | 0.9925152 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9430286 | 0.045755 | 0.9887836 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8783871 | 0.0674949 | 0.945882 |  |
| 8 | 31 | 0.0824973 | 65.9230882 | 1.0 | 0.0406742 | 1.0406742 | ✓ |
| 21 | 11 | 0.0718016 | 65.4875891 | 0.940741 | 0.0944024 | 1.0351434 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6331013 | 0.0762396 | 0.7093409 |  |

### N166：selection → action 25

path=[2, 8]；visits=31；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5644。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.7774135 | 0.1296331 | 0.9070465 |  |
| 14 | 7 | 0.0216367 | 66.5598616 | 1.0 | 0.0210819 | 1.0210819 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.8183428 | 0.1443407 | 0.9626834 |  |
| 25 | 6 | 0.1675189 | 65.9982558 | 0.8489648 | 0.1865411 | 1.035506 | ✓ |
| 33 | 7 | 0.1164611 | 65.7548388 | 0.7835016 | 0.1134749 | 0.8969765 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2070462 | 0.2070462 |  |

### N11：selection → action 37

path=[24, 3, 9]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [37, 27]，并列按 prior 抽样。trace 行 5646。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 20 | 2 | 0.0527497 | 60.7350753 | 0.4108449 | 0.060298 | 0.4711429 |  |
| 37 | 1 | 0.0649189 | 57.7238561 | 0.0 | 0.1113128 | 0.1113128 | ✓ |
| 27 | 1 | 0.060169 | 65.053189 | 1.0 | 0.1031683 | 1.1031683 |  |

### N164：expansion → action 6

path=[24, 3, 9, 37]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 5648。

bucket=0，compatibility_richness_prior；到达 N207（新建）；closure=[]。

## iteration 224

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[5, 22, 24, 18, 25, 36, 28, 10, 9, 19, 8, 13, 17]

### N0：selection → action 2

path=[]；visits=223；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5668。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7962308 | 0.7962308 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5955965 | 0.0785966 | 0.6741931 |  |
| 2 | 128 | 0.3264006 | 65.2753732 | 1.0 | 0.0528983 | 1.0528983 | ✓ |

### N5：selection → action 21

path=[2]；visits=128；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5670。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7383552 | 0.1457377 | 0.8840929 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0764008 | 0.0579319 | 0.1343327 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.196557 | 0.196557 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.592591 | 0.1080425 | 0.7006335 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4883816 | 0.1576282 | 0.6460098 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8178233 | 0.1140658 | 0.9318891 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8834159 | 0.126501 | 1.0099169 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9614282 | 0.0459348 | 1.007363 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8955255 | 0.0677601 | 0.9632856 |  |
| 8 | 32 | 0.0824973 | 65.782443 | 1.0 | 0.0395967 | 1.0395967 |  |
| 21 | 11 | 0.0718016 | 65.4875891 | 0.959096 | 0.0947733 | 1.0538693 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6454538 | 0.0765392 | 0.721993 |  |

### N183：selection → action 3

path=[2, 21]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [3]，并列按 prior 抽样。trace 行 5672。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 4 | 0.2828124 | 62.8001221 | 0.0 | 0.2626352 | 0.2626352 |  |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.1246417 | 1.1246417 |  |
| 0 | 3 | 0.1722104 | 64.7428053 | 0.1669249 | 0.199905 | 0.3668299 |  |
| 3 | 2 | 0.0151285 | 64.3189625 | 0.1305062 | 0.0234153 | 0.1539215 | ✓ |

### N204：expansion → action 0

path=[2, 21, 3]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 11 个代表动作。trace 行 5674。

bucket=1，uniform_random；到达 N208（新建）；closure=[]。

## iteration 225

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 39, 1, 23, 36, 12, 32, 26, 8, 33, 30, 9, 5]

### N0：selection → action 2

path=[]；visits=224；children=3；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5693。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7980141 | 0.7980141 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.5880715 | 0.0787726 | 0.6668442 |  |
| 2 | 129 | 0.3264006 | 65.3167877 | 1.0 | 0.0526089 | 1.0526089 | ✓ |

### N5：selection → action 21

path=[2]；visits=129；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5695。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7383552 | 0.1463058 | 0.8846611 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0764008 | 0.0581578 | 0.1345586 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1973233 | 0.1973233 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.592591 | 0.1084638 | 0.7010548 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4883816 | 0.1582427 | 0.6466243 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8178233 | 0.1145105 | 0.9323338 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8834159 | 0.1269941 | 1.0104101 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9614282 | 0.0461139 | 1.0075421 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8955255 | 0.0680243 | 0.9635498 |  |
| 8 | 32 | 0.0824973 | 65.782443 | 1.0 | 0.039751 | 1.039751 |  |
| 21 | 12 | 0.0718016 | 65.66905 | 0.9842694 | 0.0878241 | 1.0720935 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6454538 | 0.0768376 | 0.7222914 |  |

### N183：selection → action 0

path=[2, 21]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [0, 3]，并列按 prior 抽样。trace 行 5697。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 4 | 0.2828124 | 62.8001221 | 0.0 | 0.2743135 | 0.2743135 |  |
| 8 | 5 | 0.1610609 | 74.4381899 | 1.0 | 0.130184 | 1.130184 |  |
| 0 | 3 | 0.1722104 | 64.7428053 | 0.1669249 | 0.208794 | 0.3757189 | ✓ |
| 3 | 3 | 0.0151285 | 65.4343483 | 0.2263457 | 0.0183424 | 0.244688 |  |

### N186：selection → action 22

path=[2, 21, 0]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [22, 38]，并列按 prior 抽样。trace 行 5699。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 1 | 0.054685 | 57.1850941 | 0.0 | 0.066302 | 0.066302 | ✓ |
| 38 | 1 | 0.0093692 | 58.444926 | 1.0 | 0.0113596 | 1.0113596 |  |

### N187：expansion → action 18

path=[2, 21, 0, 22]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 5701。

bucket=0，compatibility_richness_prior；到达 N209（新建）；closure=[37]。

## iteration 226

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[11, 28, 32, 19, 39, 37, 29, 7, 23, 1, 25, 3]

### N0：selection → action 2

path=[]；visits=225；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5720。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.7997934 | 0.7997934 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.639965 | 0.0789483 | 0.7189132 |  |
| 2 | 130 | 0.3264006 | 65.0509851 | 1.0 | 0.0523238 | 1.0523238 | ✓ |

### N5：selection → action 8

path=[2]；visits=130；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5722。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7383552 | 0.1468718 | 0.8852271 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0764008 | 0.0583827 | 0.1347836 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1980866 | 0.1980866 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.592591 | 0.1088834 | 0.7014744 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4883816 | 0.1588549 | 0.6472365 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8178233 | 0.1149534 | 0.9327767 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8834159 | 0.1274854 | 1.0109014 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9614282 | 0.0462922 | 1.0077204 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.8955255 | 0.0682875 | 0.9638129 |  |
| 8 | 32 | 0.0824973 | 65.782443 | 1.0 | 0.0399048 | 1.0399048 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8140019 | 0.0818664 | 0.8958684 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6454538 | 0.0771348 | 0.7225887 |  |

### N166：selection → action 14

path=[2, 8]；visits=32；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5724。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.7774135 | 0.1317073 | 0.9091208 |  |
| 14 | 7 | 0.0216367 | 66.5598616 | 1.0 | 0.0214193 | 1.0214193 | ✓ |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.8183428 | 0.1466503 | 0.964993 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.5413164 | 0.1658352 | 0.7071516 |  |
| 33 | 7 | 0.1164611 | 65.7548388 | 0.7835016 | 0.1152906 | 0.8987922 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2103591 | 0.2103591 |  |

### N168：selection → action 24

path=[2, 8, 14]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [0, 25, 24]，并列按 prior 抽样。trace 行 5726。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.2855866 | 67.2795538 | 1.0 | 0.3526092 | 1.3526092 |  |
| 25 | 2 | 0.3301689 | 66.7376289 | 0.8436691 | 0.4076542 | 1.2513233 |  |
| 24 | 2 | 0.1720735 | 63.8130286 | 0.0 | 0.2124564 | 0.2124564 | ✓ |

### N191：expansion → action 10

path=[2, 8, 14, 24]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 16 个代表动作。trace 行 5728。

bucket=1，uniform_random；到达 N210（新建）；closure=[]。

## iteration 227

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 1, 13, 29, 26, 34, 36, 30, 24, 19, 37, 16, 5, 9]

### N0：selection → action 2

path=[]；visits=226；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5746。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8015687 | 0.8015687 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6520964 | 0.0791235 | 0.7312199 |  |
| 2 | 131 | 0.3264006 | 64.9949478 | 1.0 | 0.0520426 | 1.0520426 | ✓ |

### N5：selection → action 8

path=[2]；visits=131；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5748。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7512308 | 0.1474356 | 0.8986664 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0777331 | 0.0586069 | 0.13634 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.198847 | 0.198847 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6029247 | 0.1093013 | 0.712226 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4968981 | 0.1594647 | 0.6563628 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8320846 | 0.1153947 | 0.9474793 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.8988211 | 0.1279748 | 1.0267959 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9781937 | 0.04647 | 1.0246637 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9111418 | 0.0685496 | 0.9796914 |  |
| 8 | 33 | 0.0824973 | 65.6588958 | 1.0 | 0.0388798 | 1.0388798 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8281966 | 0.0821807 | 0.9103773 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6567094 | 0.0774309 | 0.7341403 |  |

### N166：selection → action 15

path=[2, 8]；visits=33；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5750。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.9290226 | 0.1337494 | 1.062772 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 1.0 | 0.0193346 | 1.0193346 |  |
| 15 | 5 | 0.1111044 | 65.8843913 | 0.9779338 | 0.1489241 | 1.1268578 | ✓ |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6468824 | 0.1684065 | 0.8152889 |  |
| 33 | 7 | 0.1164611 | 65.7548388 | 0.936298 | 0.1170782 | 1.0533762 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2136207 | 0.2136207 |  |

### N170：expansion → action 6

path=[2, 8, 15]；visits=5；children=2；K=3。已有 2 条动作边 < K=3，且尚余 15 个代表动作。trace 行 5752。

bucket=0，compatibility_richness_prior；到达 N211（新建）；closure=[]。

## iteration 228

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[10, 15, 4, 25, 7, 31, 21, 1, 12, 9, 33, 26, 8, 13]

### N0：selection → action 2

path=[]；visits=227；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5772。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8033402 | 0.8033402 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6610335 | 0.0792984 | 0.7403318 |  |
| 2 | 132 | 0.3264006 | 64.9549815 | 1.0 | 0.0517655 | 1.0517655 | ✓ |

### N5：selection → action 36

path=[2]；visits=132；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5774。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7609926 | 0.1479973 | 0.9089899 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0787432 | 0.0588301 | 0.1375733 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.1996046 | 0.1996046 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6107593 | 0.1097177 | 0.7204771 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.503355 | 0.1600722 | 0.6634272 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8428971 | 0.1158343 | 0.9587314 |  |
| 36 | 14 | 0.1197987 | 64.9420543 | 0.9105007 | 0.1284623 | 1.0389631 | ✓ |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9909048 | 0.046647 | 1.0375518 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9229816 | 0.0688107 | 0.9917923 |  |
| 8 | 34 | 0.0824973 | 65.5680127 | 1.0 | 0.0379129 | 1.0379129 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8389585 | 0.0824938 | 0.9214523 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6652429 | 0.0777259 | 0.7429688 |  |

### N47：selection → action 39

path=[2, 36]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [31, 37, 39]，并列按 prior 抽样。trace 行 5776。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.462459 | 0.0630885 | 0.5255474 |  |
| 8 | 6 | 0.059354 | 63.6511971 | 0.0 | 0.0444165 | 0.0444165 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0512593 | 1.0512593 |  |
| 39 | 4 | 0.0880029 | 64.144871 | 0.0719324 | 0.0921975 | 0.1641299 | ✓ |

### N63：expansion → action 29

path=[2, 36, 39]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 11 个代表动作。trace 行 5778。

bucket=0，compatibility_richness_prior；到达 N212（新建）；closure=[]。

## iteration 229

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 10, 24, 39, 22, 26, 9, 35, 33, 37, 36, 11, 13]

### N0：selection → action 2

path=[]；visits=228；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5798。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8051077 | 0.8051077 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6646992 | 0.0794728 | 0.7441721 |  |
| 2 | 133 | 0.3264006 | 64.9388992 | 1.0 | 0.0514922 | 1.0514922 | ✓ |

### N5：selection → action 8

path=[2]；visits=133；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5800。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7609926 | 0.1485568 | 0.9095494 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0787432 | 0.0590525 | 0.1377958 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2003592 | 0.2003592 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6107593 | 0.1101325 | 0.7208919 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.503355 | 0.1606774 | 0.6640323 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8428971 | 0.1162723 | 0.9591693 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8141328 | 0.1208888 | 0.9350215 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 0.9909048 | 0.0468233 | 1.0377281 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9229816 | 0.0690709 | 0.9920524 |  |
| 8 | 34 | 0.0824973 | 65.5680127 | 1.0 | 0.0380562 | 1.0380562 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8389585 | 0.0828057 | 0.9217642 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6652429 | 0.0780197 | 0.7432627 |  |

### N166：selection → action 12

path=[2, 8]；visits=34；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5802。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 7 | 0.1330445 | 65.7322009 | 0.9290226 | 0.1357608 | 1.0647834 | ✓ |
| 14 | 8 | 0.0216367 | 65.9530519 | 1.0 | 0.0196253 | 1.0196253 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.7648244 | 0.1295688 | 0.8943933 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6468824 | 0.170939 | 0.8178215 |  |
| 33 | 7 | 0.1164611 | 65.7548388 | 0.936298 | 0.1188388 | 1.0551368 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2168332 | 0.2168332 |  |

### N167：selection → action 1

path=[2, 8, 12]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [1, 17, 3]，并列按 prior 抽样。trace 行 5804。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0.0206411 | 70.782895 | 1.0 | 0.0254852 | 1.0254852 | ✓ |
| 17 | 2 | 0.0423435 | 67.0126978 | 0.6902232 | 0.0522809 | 0.742504 |  |
| 3 | 2 | 0.0185985 | 58.6122057 | 0.0 | 0.0229632 | 0.0229632 |  |

### N169：expansion → action 3

path=[2, 8, 12, 1]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 35 个代表动作。trace 行 5806。

bucket=1，uniform_random；到达 N213（新建）；closure=[]。

## iteration 230

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 14, 28, 35, 15, 36, 0, 34, 12, 7, 38, 29, 8, 23, 3]

### N0：selection → action 2

path=[]；visits=229；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5825。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8068713 | 0.8068713 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6829409 | 0.0796469 | 0.7625879 |  |
| 2 | 134 | 0.3264006 | 64.8614369 | 1.0 | 0.0512228 | 1.0512228 | ✓ |

### N5：selection → action 24

path=[2]；visits=134；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5827。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7679775 | 0.1491143 | 0.9170918 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.079466 | 0.0592741 | 0.1387401 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.201111 | 0.201111 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6163653 | 0.1105458 | 0.7269111 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5079751 | 0.1612803 | 0.6692554 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8506338 | 0.1167086 | 0.9673423 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8216054 | 0.1213424 | 0.9429478 |  |
| 24 | 8 | 0.0261006 | 65.5044007 | 1.0 | 0.046999 | 1.046999 | ✓ |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9314533 | 0.0693301 | 1.0007834 |  |
| 8 | 35 | 0.0824973 | 65.411263 | 0.986561 | 0.0371379 | 1.0236989 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8466591 | 0.0831164 | 0.9297755 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.671349 | 0.0783125 | 0.7496615 |  |

### N68：selection → action 25

path=[2, 24]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [25]，并列按 prior 抽样。trace 行 5829。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 3 | 0.0573819 | 65.4679187 | 1.0 | 0.0568052 | 1.0568052 |  |
| 25 | 2 | 0.0655241 | 62.95859 | 0.0 | 0.0864874 | 0.0864874 | ✓ |
| 8 | 3 | 0.060259 | 65.1081867 | 0.8566421 | 0.0596533 | 0.9162955 |  |

### N69：expansion → action 39

path=[2, 24, 25]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 26 个代表动作。trace 行 5831。

bucket=1，uniform_random；到达 N214（新建）；closure=[]。

## iteration 231

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 30, 12, 10, 36, 11, 3, 37, 4, 25, 38, 26, 13, 15]

### N0：selection → action 2

path=[]；visits=230；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5852。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8086312 | 0.8086312 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6851578 | 0.0798207 | 0.7649785 |  |
| 2 | 135 | 0.3264006 | 64.8523042 | 1.0 | 0.050957 | 1.050957 | ✓ |

### N5：selection → action 8

path=[2]；visits=135；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5854。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.778439 | 0.1496696 | 0.9281086 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0805485 | 0.0594949 | 0.1400434 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2018601 | 0.2018601 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6247615 | 0.1109575 | 0.735719 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5148948 | 0.161881 | 0.6767758 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8622212 | 0.1171432 | 0.9793644 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8327974 | 0.1217943 | 0.9545917 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8818494 | 0.0424567 | 0.9243061 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9441417 | 0.0695883 | 1.0137299 |  |
| 8 | 35 | 0.0824973 | 65.411263 | 1.0 | 0.0372762 | 1.0372762 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8581924 | 0.083426 | 0.9416183 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6804942 | 0.0786042 | 0.7590983 |  |

### N166：selection → action 33

path=[2, 8]；visits=35；children=6；K=6。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5856。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.7020299 | 0.122438 | 0.8244679 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 1.0 | 0.0199118 | 1.0199118 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.7648244 | 0.1314604 | 0.8962849 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6468824 | 0.1734346 | 0.8203171 |  |
| 33 | 7 | 0.1164611 | 65.7548388 | 0.936298 | 0.1205738 | 1.0568718 | ✓ |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2199988 | 0.2199988 |  |

### N180：selection → action 19

path=[2, 8, 33]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [34, 19, 29]，并列按 prior 抽样。trace 行 5858。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.0376976 | 70.7673823 | 1.0 | 0.0465447 | 1.0465447 |  |
| 19 | 2 | 0.0504112 | 66.2180642 | 0.6891718 | 0.0622419 | 0.7514137 | ✓ |
| 29 | 2 | 0.0172087 | 56.1312631 | 0.0 | 0.0212474 | 0.0212474 |  |

### N182：expansion → action 5

path=[2, 8, 33, 19]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 35 个代表动作。trace 行 5860。

bucket=1，uniform_random；到达 N215（新建）；closure=[]。

## iteration 232

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 35, 37, 1, 39, 32, 11, 26, 22, 36, 6, 4, 15, 9, 14]

### N0：selection → action 2

path=[]；visits=231；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5880。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8103871 | 0.8103871 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6959352 | 0.079994 | 0.7759292 |  |
| 2 | 136 | 0.3264006 | 64.8087341 | 1.0 | 0.0506949 | 1.0506949 | ✓ |

### N5：selection → action 8

path=[2]；visits=136；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5882。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.789271 | 0.1502229 | 0.939494 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0816693 | 0.0597148 | 0.1413842 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2026063 | 0.2026063 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6334551 | 0.1113677 | 0.7448228 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5220596 | 0.1624794 | 0.684539 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8742191 | 0.1175763 | 0.9917954 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8443859 | 0.1222446 | 0.9666304 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8941204 | 0.0426136 | 0.9367341 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9572795 | 0.0698455 | 1.027125 |  |
| 8 | 36 | 0.0824973 | 65.3174275 | 1.0 | 0.0364028 | 1.0364028 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8701342 | 0.0837344 | 0.9538686 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6899633 | 0.0788948 | 0.7688581 |  |

### N166：expansion → action 3

path=[2, 8]；visits=36；children=6；K=7。已有 6 条动作边 < K=7，且尚余 3 个代表动作。trace 行 5884。

bucket=0，compatibility_richness_prior；到达 N126（复用）；closure=[]。

## iteration 233

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[38, 35, 15, 0, 25, 16, 39, 11, 13, 30, 6, 22, 33]

### N0：selection → action 2

path=[]；visits=232；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5906。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8121393 | 0.8121393 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.682559 | 0.0801669 | 0.7627259 |  |
| 2 | 137 | 0.3264006 | 64.8630165 | 1.0 | 0.0504364 | 1.0504364 | ✓ |

### N5：selection → action 8

path=[2]；visits=137；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5908。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7797152 | 0.1507742 | 0.9304895 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0806805 | 0.059934 | 0.1406145 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2033498 | 0.2033498 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6257858 | 0.1117764 | 0.7375622 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.515739 | 0.1630757 | 0.6788147 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8636348 | 0.1180078 | 0.9816426 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8341628 | 0.1226932 | 0.956856 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8832952 | 0.04277 | 0.9260652 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9456896 | 0.0701019 | 1.0157915 |  |
| 8 | 37 | 0.0824973 | 65.4000715 | 1.0 | 0.035575 | 1.035575 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8595994 | 0.0840416 | 0.943641 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6816098 | 0.0791843 | 0.7607941 |  |

### N166：selection → action 3

path=[2, 8]；visits=37；children=7；K=7。最低访问优先：child.visits < 5；最少 3 次；候选 [3]，并列按 prior 抽样。trace 行 5910。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.3947423 | 0.1258877 | 0.52063 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 0.562287 | 0.0204728 | 0.5827599 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.4300508 | 0.1351643 | 0.5652151 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.3637336 | 0.1783211 | 0.5420546 |  |
| 33 | 8 | 0.1164611 | 65.289632 | 0.4424013 | 0.1101964 | 0.5525977 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2261972 | 0.2261972 |  |
| 3 | 3 | 0.0736012 | 68.3752558 | 1.0 | 0.1566946 | 1.1566946 | ✓ |

### N126：expansion → action 32

path=[2, 3, 8]；visits=3；children=1；K=2。已有 1 条动作边 < K=2，且尚余 16 个代表动作。trace 行 5912。

bucket=1，uniform_random；到达 N216（新建）；closure=[]。

## iteration 234

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[32, 35, 15, 28, 21, 14, 18, 27, 26, 33, 0, 13, 22]

### N0：selection → action 2

path=[]；visits=233；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5931。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8138877 | 0.8138877 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6901301 | 0.0803395 | 0.7704696 |  |
| 2 | 138 | 0.3264006 | 64.8320336 | 1.0 | 0.0501813 | 1.0501813 | ✓ |

### N5：selection → action 8

path=[2]；visits=138；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5933。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7876212 | 0.1513235 | 0.9389447 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0814986 | 0.0601523 | 0.1416509 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2040906 | 0.2040906 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.632131 | 0.1121836 | 0.7443146 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5209684 | 0.1636698 | 0.6846381 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8723917 | 0.1184377 | 0.9908294 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8426209 | 0.1231401 | 0.965761 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8922515 | 0.0429258 | 0.9351773 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9552785 | 0.0703572 | 1.0256357 |  |
| 8 | 38 | 0.0824973 | 65.3315528 | 1.0 | 0.0347891 | 1.0347891 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8683154 | 0.0843478 | 0.9526632 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6885211 | 0.0794728 | 0.7679938 |  |

### N166：selection → action 3

path=[2, 8]；visits=38；children=7；K=7。最低访问优先：child.visits < 5；最少 4 次；候选 [3]，并列按 prior 抽样。trace 行 5935。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.7020299 | 0.1275775 | 0.8296074 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 1.0 | 0.0207477 | 1.0207477 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.7648244 | 0.1369786 | 0.9018031 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6468824 | 0.1807147 | 0.8275972 |  |
| 33 | 8 | 0.1164611 | 65.289632 | 0.7867892 | 0.1116756 | 0.8984648 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2292335 | 0.2292335 |  |
| 3 | 4 | 0.0736012 | 65.5858086 | 0.8819748 | 0.1270384 | 1.0090132 | ✓ |

### N126：expansion → action 24

path=[2, 3, 8]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 15 个代表动作。trace 行 5937。

bucket=0，compatibility_richness_prior；到达 N217（新建）；closure=[]。

## iteration 235

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[36, 9, 27, 15, 39, 20, 4, 22, 12, 33, 19, 1]

### N0：selection → action 2

path=[]；visits=234；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5956。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8156324 | 0.8156324 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.678977 | 0.0805118 | 0.7594888 |  |
| 2 | 139 | 0.3264006 | 64.8779155 | 1.0 | 0.0499297 | 1.0499297 | ✓ |

### N5：selection → action 8

path=[2]；visits=139；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5958。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7799068 | 0.1518708 | 0.9317776 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0807004 | 0.0603699 | 0.1410702 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2048287 | 0.2048287 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6259396 | 0.1125893 | 0.7385289 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5158657 | 0.1642617 | 0.6801274 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.863847 | 0.118866 | 0.982713 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8343678 | 0.1235855 | 0.9579533 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8835123 | 0.0430811 | 0.9265933 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.945922 | 0.0706117 | 1.0165337 |  |
| 8 | 39 | 0.0824973 | 65.3983945 | 1.0 | 0.034042 | 1.034042 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8598106 | 0.0846529 | 0.9444635 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6817773 | 0.0797602 | 0.7615375 |  |

### N166：selection → action 3

path=[2, 8]；visits=39；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5960。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.6190745 | 0.1292453 | 0.7483198 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 0.881835 | 0.0210189 | 0.9028539 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.674449 | 0.1387693 | 0.8132183 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.5704436 | 0.1830771 | 0.7535207 |  |
| 33 | 8 | 0.1164611 | 65.289632 | 0.6938183 | 0.1131354 | 0.8069537 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2322302 | 0.2322302 |  |
| 3 | 5 | 0.0736012 | 66.3699989 | 1.0 | 0.1072492 | 1.1072492 | ✓ |

### N126：selection → action 32

path=[2, 3, 8]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [36, 32, 24]，并列按 prior 抽样。trace 行 5962。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0899073 | 73.7147706 | 1.0 | 0.1407273 | 1.1407273 |  |
| 32 | 1 | 0.1091457 | 62.7963614 | 0.0 | 0.17084 | 0.17084 | ✓ |
| 24 | 1 | 0.041313 | 67.9383794 | 0.4709494 | 0.064665 | 0.5356144 |  |

### N216：expansion → action 31

path=[2, 3, 8, 32]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 5964。

bucket=0，compatibility_richness_prior；到达 N218（新建）；closure=[]。

## iteration 236

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[11, 34, 13, 1, 22, 16, 39, 26, 29, 30, 36, 33]

### N0：selection → action 2

path=[]；visits=235；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5982。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8173734 | 0.8173734 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.683963 | 0.0806836 | 0.7646466 |  |
| 2 | 140 | 0.3264006 | 64.8572192 | 1.0 | 0.0496814 | 1.0496814 | ✓ |

### N5：selection → action 8

path=[2]；visits=140；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5984。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7854961 | 0.1524161 | 0.9379122 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0812787 | 0.0605866 | 0.1418653 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2055642 | 0.2055642 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6304254 | 0.1129936 | 0.743419 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5195627 | 0.1648515 | 0.6844142 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8700379 | 0.1192928 | 0.9893307 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8403473 | 0.1240292 | 0.9643766 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.889844 | 0.0432358 | 0.9330798 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.952701 | 0.0708652 | 1.0235663 |  |
| 8 | 40 | 0.0824973 | 65.3498351 | 1.0 | 0.033331 | 1.033331 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8659725 | 0.0849568 | 0.9509293 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6866633 | 0.0800466 | 0.7667099 |  |

### N166：selection → action 14

path=[2, 8]；visits=40；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 5986。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.7020299 | 0.1308918 | 0.8329216 |  |
| 14 | 8 | 0.0216367 | 65.9530519 | 1.0 | 0.0212867 | 1.0212867 | ✓ |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.7648244 | 0.1405371 | 0.9053616 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6468824 | 0.1854094 | 0.8322918 |  |
| 33 | 8 | 0.1164611 | 65.289632 | 0.7867892 | 0.1145767 | 0.9013659 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2351886 | 0.2351886 |  |
| 3 | 6 | 0.0736012 | 65.6415028 | 0.8998739 | 0.093099 | 0.992973 |  |

### N168：selection → action 0

path=[2, 8, 14]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [0, 25]，并列按 prior 抽样。trace 行 5988。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 0.2855866 | 67.2795538 | 1.0 | 0.3769551 | 1.3769551 | ✓ |
| 25 | 2 | 0.3301689 | 66.7376289 | 0.8700131 | 0.4358007 | 1.3058138 |  |
| 24 | 3 | 0.1720735 | 63.1104806 | 0.0 | 0.1703441 | 0.1703441 |  |

### N172：expansion → action 6

path=[2, 8, 14, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 25 个代表动作。trace 行 5990。

bucket=1，uniform_random；到达 N219（新建）；closure=[]。

## iteration 237

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 39, 38, 23, 1, 28, 17, 7, 30, 26, 18, 8, 10, 12, 33, 15]

### N0：selection → action 2

path=[]；visits=236；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6008。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8191106 | 0.8191106 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7157554 | 0.0808551 | 0.7966105 |  |
| 2 | 141 | 0.3264006 | 64.7320321 | 1.0 | 0.0494364 | 1.0494364 | ✓ |

### N5：selection → action 3

path=[2]；visits=141；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6010。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8124559 | 0.1529595 | 0.9654153 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0840684 | 0.0608026 | 0.144871 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2062971 | 0.2062971 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6520629 | 0.1133964 | 0.7654593 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5373951 | 0.1654392 | 0.7028343 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8998993 | 0.1197181 | 1.0196174 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8691897 | 0.1244714 | 0.9936611 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9203852 | 0.0433899 | 0.9637751 |  |
| 3 | 9 | 0.04278 | 65.0293452 | 0.9853996 | 0.0711179 | 1.0565175 | ✓ |
| 8 | 41 | 0.0824973 | 65.1249924 | 1.0 | 0.0326534 | 1.0326534 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8956944 | 0.0852597 | 0.9809541 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.710231 | 0.0803319 | 0.7905629 |  |

### N122：expansion → action 36

path=[2, 3]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 5 个代表动作。trace 行 6012。

bucket=1，uniform_random；到达 N17（复用）；closure=[]。

## iteration 238

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[23, 4, 28, 25, 36, 5, 21, 39, 11, 35, 14, 15]

### N0：selection → action 2

path=[]；visits=237；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6035。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8208442 | 0.8208442 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6889989 | 0.0810262 | 0.7700251 |  |
| 2 | 142 | 0.3264006 | 64.8366194 | 1.0 | 0.0491946 | 1.0491946 | ✓ |

### N5：selection → action 3

path=[2]；visits=142；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6037。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6467313 | 0.1535009 | 0.8002322 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0669201 | 0.0610179 | 0.127938 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2070273 | 0.2070273 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5190552 | 0.1137978 | 0.6328531 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4277774 | 0.1660248 | 0.5938022 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.716338 | 0.1201419 | 0.8364799 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6918926 | 0.124912 | 0.8168046 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7326453 | 0.0435435 | 0.7761888 |  |
| 3 | 10 | 0.04278 | 66.8036788 | 1.0 | 0.0648815 | 1.0648815 | ✓ |
| 8 | 41 | 0.0824973 | 65.1249924 | 0.7960202 | 0.0327689 | 0.8287892 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7129909 | 0.0855615 | 0.7985524 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5653582 | 0.0806163 | 0.6459745 |  |

### N122：selection → action 18

path=[2, 3]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [6, 18]，并列按 prior 抽样。trace 行 6039。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 3 | 0.143538 | 63.5769699 | 0.2776229 | 0.1588675 | 0.4364903 |  |
| 18 | 3 | 0.1242323 | 59.4610318 | 0.0 | 0.1375 | 0.1375 | ✓ |
| 8 | 6 | 0.1439711 | 74.2866761 | 1.0 | 0.0910553 | 1.0910553 |  |
| 36 | 5 | 0.1853786 | 72.1266796 | 0.8543067 | 0.1367843 | 0.9910911 |  |

### N124：selection → action 26

path=[2, 3, 18]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [38, 26]，并列按 prior 抽样。trace 行 6041。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 38 | 1 | 0.0810675 | 67.8165987 | 1.0 | 0.0982891 | 1.0982891 |  |
| 26 | 1 | 0.0462194 | 45.6414421 | 0.0 | 0.056038 | 0.056038 | ✓ |

### N129：expansion → action 6

path=[2, 3, 18, 26]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 6043。

bucket=0，compatibility_richness_prior；到达 N220（新建）；closure=[]。

## iteration 239

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class8；rollout：[35, 4, 11, 13, 18, 6, 1, 37, 38, 30, 5, 16]

### N0：selection → action 2

path=[]；visits=238；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6061。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8225741 | 0.8225741 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.725057 | 0.081197 | 0.8062539 |  |
| 2 | 143 | 0.3264006 | 64.6974817 | 1.0 | 0.0489559 | 1.0489559 | ✓ |

### N5：selection → action 8

path=[2]；visits=143；children=12；K=12。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6063。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8124559 | 0.1540405 | 0.9664963 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0840684 | 0.0612323 | 0.1453007 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.207755 | 0.207755 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6520629 | 0.1141978 | 0.7662607 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5373951 | 0.1666084 | 0.7040035 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8998993 | 0.1205642 | 1.0204634 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8691897 | 0.1253511 | 0.9945408 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9203852 | 0.0436965 | 0.9640818 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8916122 | 0.0596837 | 0.9512959 |  |
| 8 | 41 | 0.0824973 | 65.1249924 | 1.0 | 0.0328841 | 1.0328841 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8956944 | 0.0858623 | 0.9815566 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.710231 | 0.0808997 | 0.7911306 |  |

### N166：selection → action 3

path=[2, 8]；visits=41；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6065。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.7801425 | 0.1325178 | 0.9126603 |  |
| 14 | 9 | 0.0216367 | 64.8617446 | 0.7215168 | 0.019396 | 0.7409128 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.849924 | 0.142283 | 0.992207 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.718859 | 0.1877127 | 0.9065717 |  |
| 33 | 8 | 0.1164611 | 65.289632 | 0.8743327 | 0.1160001 | 0.9903328 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2381103 | 0.2381103 |  |
| 3 | 6 | 0.0736012 | 65.6415028 | 1.0 | 0.0942556 | 1.0942556 | ✓ |

### N126：selection → action 36

path=[2, 3, 8]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [36, 24]，并列按 prior 抽样。trace 行 6067。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0899073 | 73.7147706 | 1.0 | 0.154159 | 1.154159 | ✓ |
| 32 | 2 | 0.1091457 | 63.1261881 | 0.0 | 0.1247639 | 0.1247639 |  |
| 24 | 1 | 0.041313 | 67.9383794 | 0.4544698 | 0.070837 | 0.5253068 |  |

### N128：expansion → action 32

path=[2, 3, 8, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 6069。

bucket=0，compatibility_richness_prior；到达 N221（新建）；closure=[]。

## iteration 240

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[39, 25, 26, 10, 22, 12, 8, 0, 21, 17, 32, 15, 5, 4, 20]

### N0：selection → action 2

path=[]；visits=239；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6087。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8243004 | 0.8243004 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6953915 | 0.0813674 | 0.7767589 |  |
| 2 | 144 | 0.3264006 | 64.8109 | 1.0 | 0.0487203 | 1.0487203 | ✓ |

### N5：expansion → action 29

path=[2]；visits=144；children=12；K=13。已有 12 条动作边 < K=13，且尚余 2 个代表动作。trace 行 6089。

bucket=0，compatibility_richness_prior；到达 N222（新建）；closure=[]。

## iteration 241

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 13, 39, 1, 16, 15, 25, 26, 20, 30, 34, 7, 22, 8, 5]

### N0：selection → action 2

path=[]；visits=240；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6110。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8260231 | 0.8260231 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6423533 | 0.0815374 | 0.7238907 |  |
| 2 | 145 | 0.3264006 | 65.0397856 | 1.0 | 0.0484877 | 1.0484877 | ✓ |

### N5：selection → action 29

path=[2]；visits=145；children=13；K=13。最低访问优先：child.visits < 5；最少 1 次；候选 [29]，并列按 prior 抽样。trace 行 6112。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.2303417 | 0.1551139 | 0.3854556 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0238345 | 0.061659 | 0.0854935 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2092028 | 0.2092028 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.1848682 | 0.1149936 | 0.2998619 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.1523584 | 0.1677695 | 0.3201279 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.255133 | 0.1214044 | 0.3765374 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.2464265 | 0.1262246 | 0.3726511 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.2609411 | 0.0440011 | 0.3049421 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.2527835 | 0.0600997 | 0.3128832 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.2915691 | 0.0323432 | 0.3239123 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.2539409 | 0.0864606 | 0.3404015 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.2013596 | 0.0814634 | 0.2828231 |  |
| 29 | 1 | 0.0875326 | 81.6804898 | 1.0 | 0.7378222 | 1.7378222 | ✓ |

### N222：expansion → action 36

path=[2, 29]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 26 个代表动作。trace 行 6114。

bucket=0，compatibility_richness_prior；到达 N223（新建）；closure=[]。

## iteration 242

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[39, 1, 17, 14, 12, 37, 15, 28, 30, 10, 23, 6, 4, 0]

### N0：selection → action 2

path=[]；visits=241；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6135。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8277422 | 0.8277422 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6623295 | 0.0817071 | 0.7440366 |  |
| 2 | 146 | 0.3264006 | 64.9492754 | 1.0 | 0.0482581 | 1.0482581 | ✓ |

### N5：selection → action 29

path=[2]；visits=146；children=13；K=13。最低访问优先：child.visits < 5；最少 2 次；候选 [29]，并列按 prior 抽样。trace 行 6137。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.4666537 | 0.1556479 | 0.6223016 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0482867 | 0.0618713 | 0.110158 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.209923 | 0.209923 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3745281 | 0.1153895 | 0.4899176 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3086659 | 0.168347 | 0.4770129 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.516879 | 0.1218223 | 0.6387012 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.4992402 | 0.1266591 | 0.6258993 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.5286456 | 0.0441525 | 0.5727981 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.5121191 | 0.0603065 | 0.5724256 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.5906955 | 0.0324546 | 0.6231501 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.5144638 | 0.0867582 | 0.601222 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.4079384 | 0.0817439 | 0.4896822 |  |
| 29 | 2 | 0.0875326 | 69.9794372 | 1.0 | 0.4935747 | 1.4935747 | ✓ |

### N222：expansion → action 7

path=[2, 29]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 25 个代表动作。trace 行 6139。

bucket=1，uniform_random；到达 N224（新建）；closure=[]。

## iteration 243

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[18, 30, 15, 25, 4, 12, 28, 3, 26, 14, 22, 27, 17]

### N0：selection → action 2

path=[]；visits=242；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6159。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8294577 | 0.8294577 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6304414 | 0.0818765 | 0.7123178 |  |
| 2 | 147 | 0.3264006 | 65.0964875 | 1.0 | 0.0480314 | 1.0480314 | ✓ |

### N5：selection → action 29

path=[2]；visits=147；children=13；K=13。最低访问优先：child.visits < 5；最少 3 次；候选 [29]，并列按 prior 抽样。trace 行 6161。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.3959073 | 0.15618 | 0.5520874 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0409663 | 0.0620828 | 0.1030491 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2106406 | 0.2106406 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3177483 | 0.115784 | 0.4335323 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.261871 | 0.1689225 | 0.4307936 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4385182 | 0.1222388 | 0.560757 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.4235536 | 0.1270922 | 0.5506457 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.448501 | 0.0443035 | 0.4928045 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.43448 | 0.0605127 | 0.4949927 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.501144 | 0.0325655 | 0.5337095 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.4364692 | 0.0870548 | 0.5235241 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.3460935 | 0.0820233 | 0.4281168 |  |
| 29 | 3 | 0.0875326 | 72.0175226 | 1.0 | 0.3714466 | 1.3714466 | ✓ |

### N222：selection → action 7

path=[2, 29]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [36, 7]，并列按 prior 抽样。trace 行 6163。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0613801 | 58.2783846 | 0.0 | 0.0744194 | 0.0744194 |  |
| 7 | 1 | 0.038653 | 76.0936935 | 1.0 | 0.0468643 | 1.0468643 | ✓ |

### N224：expansion → action 21

path=[2, 29, 7]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 6165。

bucket=0，compatibility_richness_prior；到达 N225（新建）；closure=[]。

## iteration 244

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[0, 5, 21, 19, 13, 6, 31, 32, 27, 34, 35, 25, 30, 10, 28]

### N0：selection → action 2

path=[]；visits=243；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6184。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8311697 | 0.8311697 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6210548 | 0.0820454 | 0.7031003 |  |
| 2 | 148 | 0.3264006 | 65.1427005 | 1.0 | 0.0478075 | 1.0478075 | ✓ |

### N5：selection → action 29

path=[2]；visits=148；children=13；K=13。最低访问优先：child.visits < 5；最少 4 次；候选 [29]，并列按 prior 抽样。trace 行 6186。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.4224308 | 0.1567103 | 0.5791412 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0437108 | 0.0622936 | 0.1060044 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2113559 | 0.2113559 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3390356 | 0.1161771 | 0.4552127 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2794149 | 0.1694961 | 0.448911 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4678964 | 0.1226538 | 0.5905503 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.4519292 | 0.1275237 | 0.5794529 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.478548 | 0.0444539 | 0.5230019 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.4635876 | 0.0607182 | 0.5243058 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.5347177 | 0.0326761 | 0.5673938 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.4657101 | 0.0873504 | 0.5530606 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.3692797 | 0.0823018 | 0.4515816 |  |
| 29 | 4 | 0.0875326 | 71.1734343 | 1.0 | 0.2981663 | 1.2981663 | ✓ |

### N222：expansion → action 13

path=[2, 29]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 24 个代表动作。trace 行 6188。

bucket=0，compatibility_richness_prior；到达 N134（复用）；closure=[]。

## iteration 245

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 4, 16, 21, 27, 15, 13, 8, 18, 19, 35, 12, 25, 1]

### N0：selection → action 2

path=[]；visits=244；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6210。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8328781 | 0.8328781 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6099957 | 0.0822141 | 0.6922098 |  |
| 2 | 149 | 0.3264006 | 65.198973 | 1.0 | 0.0475864 | 1.0475864 | ✓ |

### N5：selection → action 29

path=[2]；visits=149；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6212。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.4338504 | 0.1572389 | 0.5910893 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0448924 | 0.0625037 | 0.1073961 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2120687 | 0.2120687 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3482008 | 0.116569 | 0.4647697 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2869683 | 0.1700678 | 0.4570361 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4805451 | 0.1230675 | 0.6036126 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.4641462 | 0.1279538 | 0.5921 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.4914846 | 0.0446038 | 0.5360884 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.4761198 | 0.060923 | 0.5370428 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.5491728 | 0.0327863 | 0.5819591 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.4782997 | 0.0876451 | 0.5659448 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.3792625 | 0.0825794 | 0.4618419 |  |
| 29 | 5 | 0.0875326 | 70.8417981 | 1.0 | 0.24931 | 1.24931 | ✓ |

### N222：selection → action 36

path=[2, 29]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [36]，并列按 prior 抽样。trace 行 6214。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0613801 | 58.2783846 | 0.0 | 0.096075 | 0.096075 | ✓ |
| 7 | 2 | 0.038653 | 72.3674313 | 1.0 | 0.0403343 | 1.0403343 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.7975606 | 0.0249356 | 0.8224962 |  |

### N223：expansion → action 11

path=[2, 29, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 6216。

bucket=0，compatibility_richness_prior；到达 N226（新建）；closure=[]。

## iteration 246

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 6, 13, 39, 21, 14, 16, 22, 26, 31, 17, 10, 18, 19]

### N0：selection → action 2

path=[]；visits=245；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6236。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8345831 | 0.8345831 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6123579 | 0.0823824 | 0.6947403 |  |
| 2 | 150 | 0.3264006 | 65.1867826 | 1.0 | 0.047368 | 1.047368 | ✓ |

### N5：selection → action 29

path=[2]；visits=150；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6238。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.4765997 | 0.1577656 | 0.6343653 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0493158 | 0.0627131 | 0.112029 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2127792 | 0.2127792 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3825105 | 0.1169595 | 0.49947 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3152446 | 0.1706375 | 0.4858821 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5278953 | 0.1234798 | 0.6513752 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.5098806 | 0.1283825 | 0.6382631 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.5399127 | 0.0447533 | 0.584666 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.523034 | 0.0611271 | 0.5841611 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.6032852 | 0.0328961 | 0.6361813 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.5254287 | 0.0879387 | 0.6133674 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.4166329 | 0.0828561 | 0.499489 |  |
| 29 | 6 | 0.0875326 | 69.7414229 | 1.0 | 0.2144101 | 1.2144101 | ✓ |

### N222：selection → action 36

path=[2, 29]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [36, 7]，并列按 prior 抽样。trace 行 6240。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0613801 | 61.2589658 | 0.0 | 0.0701633 | 0.0701633 | ✓ |
| 7 | 2 | 0.038653 | 72.3674313 | 1.0 | 0.0441841 | 1.0441841 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.7432428 | 0.0273156 | 0.7705584 |  |

### N223：expansion → action 34

path=[2, 29, 36]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6242。

bucket=1，uniform_random；到达 N227（新建）；closure=[]。

## iteration 247

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[30, 36, 5, 35, 8, 10, 17, 19, 15, 22, 39, 26, 6]

### N0：selection → action 2

path=[]；visits=246；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6262。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8362846 | 0.8362846 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6148581 | 0.0825503 | 0.6974085 |  |
| 2 | 151 | 0.3264006 | 65.1739819 | 1.0 | 0.0471523 | 1.0471523 | ✓ |

### N5：selection → action 29

path=[2]；visits=151；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6264。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.5131988 | 0.1582907 | 0.6714895 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0531029 | 0.0629218 | 0.1160247 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2134873 | 0.2134873 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4118844 | 0.1173487 | 0.5292331 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.339453 | 0.1712054 | 0.5106583 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5684336 | 0.1238907 | 0.6923244 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.5490355 | 0.1288097 | 0.6778452 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.5813739 | 0.0449022 | 0.6262761 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.563199 | 0.0613305 | 0.6245295 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.6496129 | 0.0330056 | 0.6826185 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.5657776 | 0.0882313 | 0.6540089 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.4486271 | 0.0831318 | 0.5317589 |  |
| 29 | 7 | 0.0875326 | 68.9450097 | 1.0 | 0.1882332 | 1.1882332 | ✓ |

### N222：selection → action 7

path=[2, 29]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [7]，并列按 prior 抽样。trace 行 6266。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 3 | 0.0613801 | 62.228154 | 0.0 | 0.0568388 | 0.0568388 |  |
| 7 | 2 | 0.038653 | 72.3674313 | 1.0 | 0.0477242 | 1.0477242 | ✓ |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.7187001 | 0.0295042 | 0.7482043 |  |

### N224：expansion → action 37

path=[2, 29, 7]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6268。

bucket=1，uniform_random；到达 N228（新建）；closure=[]。

## iteration 248

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[13, 8, 18, 25, 11, 26, 32, 39, 22, 33, 27, 12, 3]

### N0：selection → action 2

path=[]；visits=247；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6287。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8379827 | 0.8379827 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6083948 | 0.082718 | 0.6911127 |  |
| 2 | 152 | 0.3264006 | 65.2072887 | 1.0 | 0.0469392 | 1.0469392 | ✓ |

### N5：selection → action 29

path=[2]；visits=152；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6289。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.5199868 | 0.1588139 | 0.6788007 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0538053 | 0.0631298 | 0.1169351 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.214193 | 0.214193 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4173323 | 0.1177366 | 0.5350689 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3439428 | 0.1717713 | 0.5157141 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.5759522 | 0.1243003 | 0.7002524 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.5562975 | 0.1292355 | 0.685533 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.5890636 | 0.0450506 | 0.6341142 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.5706483 | 0.0615332 | 0.6321815 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.6582051 | 0.0331147 | 0.6913198 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.573261 | 0.088523 | 0.661784 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.454561 | 0.0834066 | 0.5379676 |  |
| 29 | 8 | 0.0875326 | 68.8096259 | 1.0 | 0.1678715 | 1.1678715 | ✓ |

### N222：selection → action 36

path=[2, 29]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 3 次；候选 [36, 7]，并列按 prior 抽样。trace 行 6291。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 3 | 0.0613801 | 62.228154 | 0.0 | 0.0607632 | 0.0607632 | ✓ |
| 7 | 3 | 0.038653 | 70.8656007 | 1.0 | 0.0382645 | 1.0382645 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8436636 | 0.0315413 | 0.8752049 |  |

### N223：selection → action 34

path=[2, 29, 36]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [11, 34]，并列按 prior 抽样。trace 行 6293。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 11 | 1 | 0.0269176 | 64.2395471 | 1.0 | 0.0326359 | 1.0326359 |  |
| 34 | 1 | 0.0251158 | 64.1665302 | 0.0 | 0.0304513 | 0.0304513 | ✓ |

### N227：expansion → action 4

path=[2, 29, 36, 34]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 21 个代表动作。trace 行 6295。

bucket=0，compatibility_richness_prior；到达 N229（新建）；closure=[]。

## iteration 249

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[8, 39, 5, 6, 36, 37, 18, 25, 9, 26, 32, 3, 11, 17, 14]

### N0：selection → action 2

path=[]；visits=248；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6314。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8396773 | 0.8396773 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6223644 | 0.0828852 | 0.7052497 |  |
| 2 | 153 | 0.3264006 | 65.1361691 | 1.0 | 0.0467287 | 1.0467287 | ✓ |

### N5：selection → action 29

path=[2]；visits=153；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6316。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.579167 | 0.1593355 | 0.7385025 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0599289 | 0.0633371 | 0.1232661 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2148964 | 0.2148964 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4648293 | 0.1181233 | 0.5829526 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3830873 | 0.1723354 | 0.5554227 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6415019 | 0.1247085 | 0.7662104 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6196103 | 0.1296599 | 0.7492702 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.6561055 | 0.0451986 | 0.7013041 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.6355944 | 0.0617353 | 0.6973297 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.7331161 | 0.0332235 | 0.7663396 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.6385044 | 0.0888137 | 0.7273181 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.506295 | 0.0836805 | 0.5899755 |  |
| 29 | 9 | 0.0875326 | 67.7637337 | 1.0 | 0.1515805 | 1.1515805 | ✓ |

### N222：expansion → action 0

path=[2, 29]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 23 个代表动作。trace 行 6318。

bucket=1，uniform_random；到达 N230（新建）；closure=[]。

## iteration 250

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 25, 36, 21, 14, 34, 17, 28, 37, 26, 16, 12, 3, 8]

### N0：selection → action 2

path=[]；visits=249；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6339。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8413685 | 0.8413685 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6116832 | 0.0830522 | 0.6947354 |  |
| 2 | 154 | 0.3264006 | 65.1902549 | 1.0 | 0.0465208 | 1.0465208 | ✓ |

### N5：selection → action 29

path=[2]；visits=154；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6341。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.567765 | 0.1598553 | 0.7276204 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0587491 | 0.0635438 | 0.1222929 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2155976 | 0.2155976 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4556783 | 0.1185087 | 0.574187 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3755455 | 0.1728977 | 0.5484432 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6288727 | 0.1251154 | 0.7539881 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6074121 | 0.130083 | 0.737495 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.6431888 | 0.0453461 | 0.6885349 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.6230815 | 0.0619367 | 0.6850182 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.7186833 | 0.0333319 | 0.7520152 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.6259342 | 0.0891035 | 0.7150377 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.4963276 | 0.0839536 | 0.5802812 |  |
| 29 | 10 | 0.0875326 | 67.948284 | 1.0 | 0.1382501 | 1.1382501 | ✓ |

### N222：selection → action 0

path=[2, 29]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [0]，并列按 prior 抽样。trace 行 6343。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 4 | 0.0613801 | 61.5202644 | 0.0 | 0.0543482 | 0.0543482 |  |
| 7 | 3 | 0.038653 | 70.8656007 | 1.0 | 0.042781 | 1.042781 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8555057 | 0.0352643 | 0.89077 |  |
| 0 | 1 | 0.0464689 | 69.6092371 | 0.8655625 | 0.1028633 | 0.9684258 | ✓ |

### N230：expansion → action 22

path=[2, 29, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 6345。

bucket=0，compatibility_richness_prior；到达 N231（新建）；closure=[]。

## iteration 251

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 32, 22, 36, 15, 16, 13, 11, 25, 37, 5, 3, 20, 9]

### N0：selection → action 2

path=[]；visits=250；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6365。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8430563 | 0.8430563 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6110356 | 0.0832188 | 0.6942544 |  |
| 2 | 155 | 0.3264006 | 65.1935948 | 1.0 | 0.0463153 | 1.0463153 | ✓ |

### N5：selection → action 29

path=[2]；visits=155；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6367。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.5817476 | 0.1603735 | 0.7421211 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.060196 | 0.0637498 | 0.1239457 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2162964 | 0.2162964 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4669005 | 0.1188928 | 0.5857933 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.3847942 | 0.1734582 | 0.5582524 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6443602 | 0.1255209 | 0.7698811 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6223711 | 0.1305046 | 0.7528757 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.6590289 | 0.045493 | 0.7045219 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.6384264 | 0.0621375 | 0.7005639 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.7363827 | 0.0334399 | 0.7698226 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.6413494 | 0.0893923 | 0.7307417 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5085509 | 0.0842257 | 0.5927766 |  |
| 29 | 11 | 0.0875326 | 67.7229687 | 1.0 | 0.12714 | 1.12714 | ✓ |

### N222：selection → action 0

path=[2, 29]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [0]，并列按 prior 抽样。trace 行 6369。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 4 | 0.0613801 | 61.5202644 | 0.0 | 0.0570009 | 0.0570009 |  |
| 7 | 3 | 0.038653 | 70.8656007 | 1.0 | 0.0448691 | 1.0448691 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8555057 | 0.0369855 | 0.8924912 |  |
| 0 | 2 | 0.0464689 | 67.5395262 | 0.6440926 | 0.0719226 | 0.7160152 | ✓ |

### N230：expansion → action 30

path=[2, 29, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6371。

bucket=1，uniform_random；到达 N232（新建）；closure=[]。

## iteration 252

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[35, 25, 15, 10, 12, 22, 28, 36, 9, 23, 18, 1]

### N0：selection → action 2

path=[]；visits=251；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6391。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8447407 | 0.8447407 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.61582 | 0.0833851 | 0.699205 |  |
| 2 | 156 | 0.3264006 | 65.1690852 | 1.0 | 0.0461122 | 1.0461122 | ✓ |

### N5：selection → action 29

path=[2]；visits=156；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6393。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6072195 | 0.16089 | 0.7681095 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0628317 | 0.0639551 | 0.1267867 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.216993 | 0.216993 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.4873438 | 0.1192757 | 0.6066195 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4016425 | 0.1740168 | 0.5756593 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.6725736 | 0.1259252 | 0.7984988 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6496217 | 0.1309249 | 0.7805466 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.6878846 | 0.0456396 | 0.7335242 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.66638 | 0.0623376 | 0.7287176 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.7686253 | 0.0335476 | 0.8021729 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.669431 | 0.0896802 | 0.7591112 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5308179 | 0.0844969 | 0.6153149 |  |
| 29 | 12 | 0.0875326 | 67.3391838 | 1.0 | 0.117738 | 1.117738 | ✓ |

### N222：selection → action 7

path=[2, 29]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [7, 0]，并列按 prior 抽样。trace 行 6395。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 4 | 0.0613801 | 61.5202644 | 0.0 | 0.0595355 | 0.0595355 |  |
| 7 | 3 | 0.038653 | 70.8656007 | 1.0 | 0.0468643 | 1.0468643 | ✓ |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8555057 | 0.0386301 | 0.8941358 |  |
| 0 | 3 | 0.0464689 | 66.0655342 | 0.4863677 | 0.0563406 | 0.5427083 |  |

### N224：selection → action 21

path=[2, 29, 7]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [21, 37]，并列按 prior 抽样。trace 行 6397。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 1 | 0.0250483 | 68.6411692 | 1.0 | 0.0303694 | 1.0303694 | ✓ |
| 37 | 1 | 0.0312581 | 67.8619395 | 0.0 | 0.0378984 | 0.0378984 |  |

### N225：expansion → action 26

path=[2, 29, 7, 21]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6399。

bucket=0，compatibility_richness_prior；到达 N233（新建）；closure=[]。

## iteration 253

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[26, 10, 8, 5, 6, 17, 25, 28, 35, 37, 4, 13]

### N0：selection → action 2

path=[]；visits=252；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6417。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8464218 | 0.8464218 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.643764 | 0.083551 | 0.727315 |  |
| 2 | 157 | 0.3264006 | 65.0332097 | 1.0 | 0.0459116 | 1.0459116 | ✓ |

### N5：selection → action 29

path=[2]；visits=157；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6419。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6909964 | 0.1614049 | 0.8524013 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0715004 | 0.0641597 | 0.1356602 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2176874 | 0.2176874 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5545817 | 0.1196574 | 0.6742391 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4570563 | 0.1745737 | 0.63163 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7653673 | 0.1263282 | 0.8916955 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7392487 | 0.1313439 | 0.8705926 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7827907 | 0.0457856 | 0.8285763 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7583191 | 0.0625371 | 0.8208563 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.874671 | 0.033655 | 0.908326 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7617911 | 0.0899672 | 0.8517582 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6040538 | 0.0847673 | 0.6888212 |  |
| 29 | 13 | 0.0875326 | 66.2764875 | 1.0 | 0.109678 | 1.109678 | ✓ |

### N222：selection → action 0

path=[2, 29]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [0]，并列按 prior 抽样。trace 行 6421。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 4 | 0.0613801 | 61.5202644 | 0.0 | 0.0619665 | 0.0619665 |  |
| 7 | 4 | 0.038653 | 66.5302336 | 0.6266387 | 0.0390223 | 0.665661 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 1.0 | 0.0402074 | 1.0402074 |  |
| 0 | 3 | 0.0464689 | 66.0655342 | 0.5685148 | 0.0586411 | 0.627156 | ✓ |

### N230：selection → action 22

path=[2, 29, 0]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [22, 30]，并列按 prior 抽样。trace 行 6423。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 1 | 0.0411156 | 65.4698153 | 1.0 | 0.0498501 | 1.0498501 | ✓ |
| 30 | 1 | 0.0399502 | 63.1175502 | 0.0 | 0.0484371 | 0.0484371 |  |

### N231：expansion → action 21

path=[2, 29, 0, 22]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6425。

bucket=0，compatibility_richness_prior；到达 N234（新建）；closure=[]。

## iteration 254

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[31, 30, 35, 14, 17, 15, 5, 25, 26, 10, 33, 13, 18, 20, 6]

### N0：selection → action 2

path=[]；visits=253；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6443。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8480995 | 0.8480995 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6415131 | 0.0837166 | 0.7252297 |  |
| 2 | 158 | 0.3264006 | 65.0437161 | 1.0 | 0.0457132 | 1.0457132 | ✓ |

### N5：selection → action 29

path=[2]；visits=158；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6445。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6931326 | 0.1619181 | 0.8550507 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0717215 | 0.0643637 | 0.1360852 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2183796 | 0.2183796 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5562961 | 0.1200379 | 0.676334 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4584693 | 0.1751287 | 0.633598 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7677334 | 0.1267298 | 0.8944632 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7415341 | 0.1317615 | 0.8732956 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7852106 | 0.0459312 | 0.8311418 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7606634 | 0.062736 | 0.8233994 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.877375 | 0.033762 | 0.911137 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7641461 | 0.0902532 | 0.8543993 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6059212 | 0.0850369 | 0.6909581 |  |
| 29 | 14 | 0.0875326 | 66.2527493 | 1.0 | 0.1026916 | 1.1026916 | ✓ |

### N222：selection → action 36

path=[2, 29]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [36, 7, 0]，并列按 prior 抽样。trace 行 6447。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 4 | 0.0613801 | 61.5202644 | 0.0 | 0.0643057 | 0.0643057 | ✓ |
| 7 | 4 | 0.038653 | 66.5302336 | 0.6266387 | 0.0404954 | 0.667134 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 1.0 | 0.0417252 | 1.0417252 |  |
| 0 | 4 | 0.0464689 | 66.0351887 | 0.5647193 | 0.0486838 | 0.6134031 |  |

### N223：expansion → action 27

path=[2, 29, 36]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 6449。

bucket=0，compatibility_richness_prior；到达 N235（新建）；closure=[]。

## iteration 255

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[36, 37, 39, 11, 25, 31, 6, 12, 28, 4, 1, 9, 21, 17]

### N0：selection → action 2

path=[]；visits=254；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6470。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8497739 | 0.8497739 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6512067 | 0.0838819 | 0.7350885 |  |
| 2 | 159 | 0.3264006 | 64.9989867 | 1.0 | 0.0455172 | 1.0455172 | ✓ |

### N5：selection → action 29

path=[2]；visits=159；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6472。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7254556 | 0.1624297 | 0.8878852 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0750661 | 0.0645671 | 0.1396332 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2190696 | 0.2190696 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.582238 | 0.1204172 | 0.7026551 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4798492 | 0.1756821 | 0.6555313 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8035352 | 0.1271302 | 0.9306655 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7761142 | 0.1321778 | 0.908292 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8218275 | 0.0460763 | 0.8679038 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7961356 | 0.0629342 | 0.8590698 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9182898 | 0.0338686 | 0.9521584 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7997806 | 0.0905384 | 0.890319 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6341772 | 0.0853056 | 0.7194828 |  |
| 29 | 15 | 0.0875326 | 65.9106197 | 1.0 | 0.0965776 | 1.0965776 | ✓ |

### N222：selection → action 0

path=[2, 29]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [7, 0]，并列按 prior 抽样。trace 行 6474。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0554689 | 0.0554689 |  |
| 7 | 4 | 0.038653 | 66.5302336 | 0.6303327 | 0.0419167 | 0.6722493 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 1.0 | 0.0431897 | 1.0431897 |  |
| 0 | 4 | 0.0464689 | 66.0351887 | 0.5690259 | 0.0503925 | 0.6194184 | ✓ |

### N230：expansion → action 5

path=[2, 29, 0]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 6476。

bucket=0，compatibility_richness_prior；到达 N236（新建）；closure=[]。

## iteration 256

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 11, 26, 36, 21, 18, 9, 34, 13, 32, 1, 0, 22, 6, 15]

### N0：selection → action 2

path=[]；visits=255；children=3；K=16。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6496。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8514451 | 0.8514451 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6486629 | 0.0840468 | 0.7327098 |  |
| 2 | 160 | 0.3264006 | 65.010595 | 1.0 | 0.0453235 | 1.0453235 | ✓ |

### N5：selection → action 29

path=[2]；visits=160；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6498。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7247267 | 0.1629397 | 0.8876664 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0749906 | 0.0647698 | 0.1397605 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2197574 | 0.2197574 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.581653 | 0.1207952 | 0.7024482 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4793671 | 0.1762337 | 0.6556008 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8027279 | 0.1275294 | 0.9302573 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7753344 | 0.1325928 | 0.9079273 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8210018 | 0.046221 | 0.8672228 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7953357 | 0.0631318 | 0.8584675 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9173672 | 0.033975 | 0.9513422 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7989771 | 0.0908227 | 0.8897998 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6335401 | 0.0855734 | 0.7191135 |  |
| 29 | 16 | 0.0875326 | 65.9179981 | 1.0 | 0.0911819 | 1.0911819 | ✓ |

### N222：expansion → action 14

path=[2, 29]；visits=16；children=4；K=5。已有 4 条动作边 < K=5，且尚余 22 个代表动作。trace 行 6500。

bucket=0，compatibility_richness_prior；到达 N180（复用）；closure=[]。

## iteration 257

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[15, 30, 28, 25, 10, 14, 23, 21, 12, 17, 39, 33, 24]

### N0：selection → action 2

path=[]；visits=256；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6522。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8531129 | 0.8531129 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6341477 | 0.0842115 | 0.7183592 |  |
| 2 | 161 | 0.3264006 | 65.0786165 | 1.0 | 0.0451319 | 1.0451319 | ✓ |

### N5：selection → action 29

path=[2]；visits=161；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6524。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6957786 | 0.163448 | 0.8592266 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0719952 | 0.0649719 | 0.1369672 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2204431 | 0.2204431 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5584197 | 0.1211721 | 0.6795918 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4602195 | 0.1767835 | 0.637003 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7706642 | 0.1279273 | 0.8985915 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7443648 | 0.1330065 | 0.8773714 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7882081 | 0.0463652 | 0.8345733 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7635672 | 0.0633288 | 0.826896 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.8807243 | 0.034081 | 0.9148053 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7670632 | 0.091106 | 0.8581692 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6082343 | 0.0858404 | 0.6940746 |  |
| 29 | 17 | 0.0875326 | 66.2235477 | 1.0 | 0.086385 | 1.086385 | ✓ |

### N222：selection → action 7

path=[2, 29]；visits=17；children=5；K=5。最低访问优先：child.visits < 5；最少 4 次；候选 [7]，并列按 prior 抽样。trace 行 6526。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0590512 | 0.0590512 |  |
| 7 | 4 | 0.038653 | 66.5302336 | 0.5262487 | 0.0446237 | 0.5708724 | ✓ |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8348745 | 0.045979 | 0.8808534 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.4749305 | 0.0447058 | 0.5196363 |  |
| 14 | 9 | 0.0350093 | 71.1123421 | 1.0 | 0.0202086 | 1.0202086 |  |

### N224：expansion → action 4

path=[2, 29, 7]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 6528。

bucket=0，compatibility_richness_prior；到达 N237（新建）；closure=[]。

## iteration 258

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 25, 16, 39, 22, 21, 6, 1, 7, 36, 12, 0, 3, 19]

### N0：selection → action 2

path=[]；visits=257；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6547。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8547776 | 0.8547776 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6210933 | 0.0843758 | 0.7054691 |  |
| 2 | 162 | 0.3264006 | 65.142508 | 1.0 | 0.0449426 | 1.0449426 | ✓ |

### N5：selection → action 29

path=[2]；visits=162；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6549。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6730479 | 0.1639549 | 0.8370028 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0696432 | 0.0651734 | 0.1348166 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2211266 | 0.2211266 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5401765 | 0.1215479 | 0.6617244 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4451844 | 0.1773317 | 0.6225161 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7454871 | 0.128324 | 0.873811 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7200469 | 0.133419 | 0.8534659 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7624579 | 0.046509 | 0.8089668 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.738622 | 0.0635251 | 0.8021471 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.8519516 | 0.0341867 | 0.8861383 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7420037 | 0.0913885 | 0.8333922 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5883636 | 0.0861066 | 0.6744702 |  |
| 29 | 18 | 0.0875326 | 66.4818934 | 1.0 | 0.0820922 | 1.0820922 | ✓ |

### N222：selection → action 14

path=[2, 29]；visits=18；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6551。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0607632 | 0.0607632 |  |
| 7 | 5 | 0.038653 | 67.3989407 | 0.6160656 | 0.0382645 | 0.6543301 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 0.8348745 | 0.047312 | 0.8821864 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.4749305 | 0.0460019 | 0.5209323 |  |
| 14 | 9 | 0.0350093 | 71.1123421 | 1.0 | 0.0207945 | 1.0207945 | ✓ |

### N180：expansion → action 30

path=[2, 8, 33]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 22 个代表动作。trace 行 6553。

bucket=1，uniform_random；到达 N238（新建）；closure=[]。

## iteration 259

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[33, 1, 7, 20, 38, 32, 35, 10, 11, 31, 39, 4, 21, 0]

### N0：selection → action 2

path=[]；visits=258；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6573。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8564389 | 0.8564389 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6208301 | 0.0845398 | 0.7053699 |  |
| 2 | 163 | 0.3264006 | 65.143824 | 1.0 | 0.0447553 | 1.0447553 | ✓ |

### N5：selection → action 29

path=[2]；visits=163；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6575。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6785517 | 0.1644601 | 0.8430119 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0702127 | 0.0653742 | 0.1355869 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.221808 | 0.221808 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5445938 | 0.1219224 | 0.6665162 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4488249 | 0.1778782 | 0.6267031 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7515832 | 0.1287194 | 0.8803027 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7259351 | 0.1338301 | 0.8597652 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7686928 | 0.0466523 | 0.8153451 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.744662 | 0.0637209 | 0.8083829 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.8589184 | 0.034292 | 0.8932104 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7480714 | 0.0916702 | 0.8397416 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.593175 | 0.0863719 | 0.6795469 |  |
| 29 | 19 | 0.0875326 | 66.4177515 | 1.0 | 0.0782279 | 1.0782279 | ✓ |

### N222：selection → action 13

path=[2, 29]；visits=19；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6577。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0624282 | 0.0624282 |  |
| 7 | 5 | 0.038653 | 67.3989407 | 0.7379141 | 0.039313 | 0.7772271 |  |
| 13 | 5 | 0.0477923 | 69.5152532 | 1.0 | 0.0486084 | 1.0486084 | ✓ |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.5688645 | 0.0472624 | 0.6161269 |  |
| 14 | 10 | 0.0350093 | 68.1877701 | 0.8356034 | 0.0194221 | 0.8550254 |  |

### N134：expansion → action 37

path=[24, 22, 9]；visits=5；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 6579。

bucket=0，compatibility_richness_prior；到达 N239（新建）；closure=[]。

## iteration 260

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[39, 7, 12, 11, 5, 0, 19, 38, 26, 25, 29, 21]

### N0：selection → action 2

path=[]；visits=259；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6599。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8580971 | 0.8580971 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6366134 | 0.0847035 | 0.7213169 |  |
| 2 | 164 | 0.3264006 | 65.0668432 | 1.0 | 0.0445702 | 1.0445702 | ✓ |

### N5：selection → action 29

path=[2]；visits=164；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6601。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7169907 | 0.1649638 | 0.8819545 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0741902 | 0.0655745 | 0.1397646 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2224874 | 0.2224874 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5754442 | 0.1222959 | 0.69774 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4742501 | 0.178423 | 0.6526731 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7941593 | 0.1291137 | 0.9232729 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7670582 | 0.13424 | 0.9012982 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8122381 | 0.0467952 | 0.8590333 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.786846 | 0.0639161 | 0.850762 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9075748 | 0.034397 | 0.9419719 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7904485 | 0.0919509 | 0.8823994 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6267774 | 0.0866365 | 0.7134138 |  |
| 29 | 20 | 0.0875326 | 65.997237 | 1.0 | 0.0747309 | 1.0747309 | ✓ |

### N222：selection → action 14

path=[2, 29]；visits=20；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6603。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.06405 | 0.06405 |  |
| 7 | 5 | 0.038653 | 67.3989407 | 0.8830913 | 0.0403343 | 0.9234256 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.3439822 | 0.0427467 | 0.3867289 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.6807829 | 0.0484902 | 0.7292732 |  |
| 14 | 10 | 0.0350093 | 68.1877701 | 1.0 | 0.0199266 | 1.0199266 | ✓ |

### N180：selection → action 30

path=[2, 8, 33]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [30]，并列按 prior 抽样。trace 行 6605。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.0376976 | 70.7673823 | 1.0 | 0.0556315 | 1.0556315 |  |
| 19 | 3 | 0.0504112 | 64.8231042 | 0.5938624 | 0.055795 | 0.6496574 |  |
| 29 | 2 | 0.0172087 | 56.1312631 | 0.0 | 0.0253955 | 0.0253955 |  |
| 30 | 1 | 0.0497912 | 65.2631981 | 0.6239314 | 0.1102175 | 0.7341489 | ✓ |

### N238：expansion → action 36

path=[2, 8, 33, 30]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 6607。

bucket=0，compatibility_richness_prior；到达 N240（新建）；closure=[]。

## iteration 261

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[16, 9, 14, 35, 6, 37, 28, 22, 29, 5, 31, 26, 13]

### N0：selection → action 2

path=[]；visits=260；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6625。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8597521 | 0.8597521 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6200589 | 0.0848668 | 0.7049257 |  |
| 2 | 165 | 0.3264006 | 65.147686 | 1.0 | 0.0443872 | 1.0443872 | ✓ |

### N5：selection → action 29

path=[2]；visits=165；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6627。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6876779 | 0.165466 | 0.8531439 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.071157 | 0.0657741 | 0.1369311 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2231647 | 0.2231647 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5519183 | 0.1226681 | 0.6745864 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4548613 | 0.1789661 | 0.6338275 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7616917 | 0.1295067 | 0.8911984 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7356985 | 0.1346487 | 0.8703472 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7790314 | 0.0469376 | 0.825969 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.7546773 | 0.0641106 | 0.8187879 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.8704704 | 0.0345018 | 0.9049722 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.7581326 | 0.0922309 | 0.8503634 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6011529 | 0.0869002 | 0.688053 |  |
| 29 | 21 | 0.0875326 | 66.3136571 | 1.0 | 0.0715512 | 1.0715512 | ✓ |

### N222：selection → action 14

path=[2, 29]；visits=21；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6629。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0656317 | 0.0656317 |  |
| 7 | 5 | 0.038653 | 67.3989407 | 0.7238158 | 0.0413304 | 0.7651462 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.2819411 | 0.0438024 | 0.3257435 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.557996 | 0.0496877 | 0.6076837 |  |
| 14 | 11 | 0.0350093 | 69.6725334 | 1.0 | 0.0187172 | 1.0187172 | ✓ |

### N180：selection → action 34

path=[2, 8, 33]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [34, 29, 30]，并列按 prior 抽样。trace 行 6631。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 2 | 0.0376976 | 70.7673823 | 1.0 | 0.0583468 | 1.0583468 | ✓ |
| 19 | 3 | 0.0504112 | 64.8231042 | 0.5938624 | 0.0585183 | 0.6523807 |  |
| 29 | 2 | 0.0172087 | 56.1312631 | 0.0 | 0.026635 | 0.026635 |  |
| 30 | 2 | 0.0497912 | 68.9526291 | 0.8760086 | 0.0770647 | 0.9530733 |  |

### N181：expansion → action 3

path=[2, 8, 33, 34]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 16 个代表动作。trace 行 6633。

bucket=1，uniform_random；到达 N241（新建）；closure=[]。

## iteration 262

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[1, 30, 33, 39, 28, 22, 12, 0, 32, 34, 20, 10, 15]

### N0：selection → action 2

path=[]；visits=261；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6652。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8614038 | 0.8614038 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6402856 | 0.0850299 | 0.7253155 |  |
| 2 | 166 | 0.3264006 | 65.0494768 | 1.0 | 0.0442062 | 1.0442062 | ✓ |

### N5：selection → action 29

path=[2]；visits=166；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6654。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7327183 | 0.1659667 | 0.898685 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0758176 | 0.0659731 | 0.1417907 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2238399 | 0.2238399 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5880669 | 0.1230393 | 0.7111062 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4846531 | 0.1795076 | 0.6641607 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8115797 | 0.1298986 | 0.9414782 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.7838841 | 0.1350561 | 0.9189402 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8300551 | 0.0470796 | 0.8771347 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8041059 | 0.0643046 | 0.8684105 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9274831 | 0.0346061 | 0.9620892 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8077875 | 0.0925099 | 0.9002974 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6405262 | 0.0871631 | 0.7276893 |  |
| 29 | 22 | 0.0875326 | 65.8378987 | 1.0 | 0.0686474 | 1.0686474 | ✓ |

### N222：selection → action 7

path=[2, 29]；visits=22；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6656。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0671762 | 0.0671762 |  |
| 7 | 5 | 0.038653 | 67.3989407 | 1.0 | 0.042303 | 1.042303 | ✓ |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.3895205 | 0.0448331 | 0.4343537 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.7709089 | 0.050857 | 0.8217659 |  |
| 14 | 12 | 0.0350093 | 66.2161429 | 0.8014963 | 0.017684 | 0.8191803 |  |

### N224：selection → action 37

path=[2, 29, 7]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [37, 4]，并列按 prior 抽样。trace 行 6658。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 21 | 2 | 0.0250483 | 61.0826508 | 0.0 | 0.0261379 | 0.0261379 |  |
| 37 | 1 | 0.0312581 | 67.8619395 | 0.6923917 | 0.0489267 | 0.7413184 | ✓ |
| 4 | 1 | 0.0307868 | 70.8737688 | 1.0 | 0.048189 | 1.048189 |  |

### N228：expansion → action 18

path=[2, 29, 7, 37]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6660。

bucket=0，compatibility_richness_prior；到达 N242（新建）；closure=[]。

## iteration 263

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[15, 38, 25, 14, 37, 21, 4, 22, 11, 28, 36, 17]

### N0：selection → action 2

path=[]；visits=262；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6679。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8630525 | 0.8630525 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6587285 | 0.0851926 | 0.7439211 |  |
| 2 | 167 | 0.3264006 | 64.9651857 | 1.0 | 0.0440271 | 1.0440271 | ✓ |

### N5：selection → action 29

path=[2]；visits=167；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6681。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.773729 | 0.1664658 | 0.9401948 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0800611 | 0.0661715 | 0.1462326 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2245131 | 0.2245131 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6209814 | 0.1234093 | 0.7443907 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5117794 | 0.1800475 | 0.6918269 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8570043 | 0.1302892 | 0.9872935 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8277586 | 0.1354623 | 0.9632208 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8765138 | 0.0472212 | 0.923735 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8491122 | 0.064498 | 0.9136102 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9793949 | 0.0347102 | 1.0141051 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8529999 | 0.0927881 | 0.945788 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6763768 | 0.0874253 | 0.7638021 |  |
| 29 | 23 | 0.0875326 | 65.4528837 | 1.0 | 0.0659849 | 1.0659849 | ✓ |

### N222：selection → action 14

path=[2, 29]；visits=23；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6683。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.068686 | 0.068686 |  |
| 7 | 6 | 0.038653 | 65.6628762 | 0.8841513 | 0.0370746 | 0.921226 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.4859917 | 0.0458408 | 0.5318324 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 0.9618371 | 0.052 | 1.0138371 |  |
| 14 | 12 | 0.0350093 | 66.2161429 | 1.0 | 0.0180814 | 1.0180814 | ✓ |

### N180：selection → action 30

path=[2, 8, 33]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [29, 30]，并列按 prior 抽样。trace 行 6685。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 3 | 0.0376976 | 65.7939119 | 0.7536365 | 0.045706 | 0.7993424 |  |
| 19 | 3 | 0.0504112 | 64.8231042 | 0.6779185 | 0.0611203 | 0.7390388 |  |
| 29 | 2 | 0.0172087 | 56.1312631 | 0.0 | 0.0278193 | 0.0278193 |  |
| 30 | 2 | 0.0497912 | 68.9526291 | 1.0 | 0.0804914 | 1.0804914 | ✓ |

### N238：expansion → action 3

path=[2, 8, 33, 30]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 23 个代表动作。trace 行 6687。

bucket=1，uniform_random；到达 N243（新建）；closure=[]。

## iteration 264

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[26, 15, 30, 14, 6, 37, 11, 16, 13, 28, 34, 3]

### N0：selection → action 2

path=[]；visits=263；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6705。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8646979 | 0.8646979 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6655786 | 0.085355 | 0.7509336 |  |
| 2 | 168 | 0.3264006 | 64.9350677 | 1.0 | 0.0438501 | 1.0438501 | ✓ |

### N5：selection → action 29

path=[2]；visits=168；children=13；K=13。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6707。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7899996 | 0.1669635 | 0.9569631 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0817447 | 0.0663693 | 0.148114 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2251843 | 0.2251843 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6340399 | 0.1237783 | 0.7578182 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5225415 | 0.1805858 | 0.7031273 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8750261 | 0.1306787 | 1.0057048 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8451653 | 0.1358672 | 0.9810326 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8949458 | 0.0473624 | 0.9423082 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8669681 | 0.0646908 | 0.9316589 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.9999904 | 0.034814 | 1.0348044 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8709374 | 0.0930655 | 0.964003 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6906002 | 0.0876866 | 0.7782868 |  |
| 29 | 24 | 0.0875326 | 65.3112085 | 1.0 | 0.0635349 | 1.0635349 | ✓ |

### N222：selection → action 0

path=[2, 29]；visits=24；children=5；K=5。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6709。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0701633 | 0.0701633 |  |
| 7 | 6 | 0.038653 | 65.6628762 | 0.9192319 | 0.037872 | 0.9571039 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.5052744 | 0.0468267 | 0.5521011 |  |
| 0 | 5 | 0.0464689 | 66.0338857 | 1.0 | 0.0531184 | 1.0531184 | ✓ |
| 14 | 13 | 0.0350093 | 65.3834501 | 0.8584013 | 0.017151 | 0.8755523 |  |

### N230：selection → action 5

path=[2, 29, 0]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [30, 5]，并列按 prior 抽样。trace 行 6711。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 22 | 2 | 0.0411156 | 65.7069838 | 0.8894964 | 0.0429041 | 0.9324005 |  |
| 30 | 1 | 0.0399502 | 63.1175502 | 0.0 | 0.062532 | 0.062532 |  |
| 5 | 1 | 0.0402108 | 66.0286734 | 1.0 | 0.0629399 | 1.0629399 | ✓ |

### N236：expansion → action 22

path=[2, 29, 0, 5]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6713。

bucket=0，compatibility_richness_prior；到达 N244（新建）；closure=[]。

## iteration 265

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[25, 0, 23, 22, 21, 26, 8, 11, 15, 16, 17, 29, 1, 7, 19]

### N0：selection → action 2

path=[]；visits=264；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6731。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8663403 | 0.8663403 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6826391 | 0.0855172 | 0.7681563 |  |
| 2 | 169 | 0.3264006 | 64.8626848 | 1.0 | 0.0436749 | 1.0436749 | ✓ |

### N5：expansion → action 20

path=[2]；visits=169；children=13；K=14。已有 13 条动作边 < K=14，且尚余 1 个代表动作。trace 行 6733。

bucket=1，uniform_random；到达 N245（新建）；closure=[]。

## iteration 266

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[34, 9, 38, 7, 10, 19, 31, 33, 37, 21, 28, 39, 5, 4, 12]

### N0：selection → action 2

path=[]；visits=265；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6754。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8679795 | 0.8679795 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6496181 | 0.085679 | 0.7352971 |  |
| 2 | 170 | 0.3264006 | 65.0062254 | 1.0 | 0.0435017 | 1.0435017 | ✓ |

### N5：selection → action 20

path=[2]；visits=170；children=14；K=14。最低访问优先：child.visits < 5；最少 1 次；候选 [20]，并列按 prior 抽样。trace 行 6756。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.2601796 | 0.1679543 | 0.4281339 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0269219 | 0.0667632 | 0.0936851 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2265207 | 0.2265207 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.2088156 | 0.1245129 | 0.3333285 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.1720946 | 0.1816575 | 0.3537521 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.2881823 | 0.1314543 | 0.4196366 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.278348 | 0.1366736 | 0.4150215 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.2947427 | 0.0476435 | 0.3423862 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.2855285 | 0.0650747 | 0.3506032 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.3293383 | 0.0350206 | 0.3643589 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.2868358 | 0.0936179 | 0.3804536 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.2274433 | 0.088207 | 0.3156503 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.3147776 | 0.0614538 | 0.3762314 |  |
| 20 | 1 | 0.0072303 | 79.0305937 | 1.0 | 0.0659904 | 1.0659904 | ✓ |

### N245：expansion → action 8

path=[2, 20]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 9 个代表动作。trace 行 6758。

bucket=0，compatibility_richness_prior；到达 N168（复用）；closure=[14]。

## iteration 267

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 22, 33, 21, 9, 26, 8, 12, 34, 11, 6, 32, 30, 19, 3]

### N0：selection → action 2

path=[]；visits=266；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6780。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8696157 | 0.8696157 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6427002 | 0.0858405 | 0.7285407 |  |
| 2 | 171 | 0.3264006 | 65.0381657 | 1.0 | 0.0433303 | 1.0433303 | ✓ |

### N5：selection → action 20

path=[2]；visits=171；children=14；K=14。最低访问优先：child.visits < 5；最少 2 次；候选 [20]，并列按 prior 抽样。trace 行 6782。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.3539603 | 0.1684476 | 0.5224079 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0366258 | 0.0669593 | 0.1035851 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.227186 | 0.227186 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.2840823 | 0.1248786 | 0.4089609 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2341254 | 0.182191 | 0.4163164 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.3920565 | 0.1318404 | 0.5238969 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.3786774 | 0.137075 | 0.5157523 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.4009816 | 0.0477834 | 0.448765 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.3884461 | 0.0652659 | 0.453712 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.4480469 | 0.0351235 | 0.4831704 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.3902246 | 0.0938928 | 0.4841174 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.3094243 | 0.0884661 | 0.3978904 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.4282379 | 0.0616343 | 0.4898722 |  |
| 20 | 2 | 0.0072303 | 73.6106835 | 1.0 | 0.0441228 | 1.0441228 | ✓ |

### N245：expansion → action 0

path=[2, 20]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 8 个代表动作。trace 行 6784。

bucket=1，uniform_random；到达 N246（新建）；closure=[]。

## iteration 268

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[1, 22, 21, 25, 38, 18, 29, 4, 28, 12, 17, 31, 27, 8]

### N0：selection → action 2

path=[]；visits=267；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6805。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8712488 | 0.8712488 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6429345 | 0.0860017 | 0.7289362 |  |
| 2 | 172 | 0.3264006 | 65.0370727 | 1.0 | 0.0431607 | 1.0431607 | ✓ |

### N5：selection → action 20

path=[2]；visits=172；children=14；K=14。最低访问优先：child.visits < 5；最少 3 次；候选 [20]，并列按 prior 抽样。trace 行 6807。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.4383277 | 0.1689394 | 0.6072672 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0453557 | 0.0671548 | 0.1125105 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2278493 | 0.2278493 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.3517942 | 0.1252432 | 0.4770373 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.2899298 | 0.182723 | 0.4726528 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.4855043 | 0.1322253 | 0.6177296 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.4689362 | 0.1374752 | 0.6064114 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.4965567 | 0.0479229 | 0.5444796 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.4810333 | 0.0654564 | 0.5464898 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.5548402 | 0.035226 | 0.5900662 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.4832357 | 0.0941669 | 0.5774027 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.3831764 | 0.0887244 | 0.4719008 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.5303096 | 0.0618143 | 0.5921238 |  |
| 20 | 3 | 0.0072303 | 70.7164886 | 1.0 | 0.0331887 | 1.0331887 | ✓ |

### N245：selection → action 0

path=[2, 20]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [0]，并列按 prior 抽样。trace 行 6809。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 8 | 10 | 0.1802896 | 68.1907732 | 1.0 | 0.0397435 | 1.0397435 |  |
| 0 | 1 | 0.1170492 | 64.9280988 | 0.0 | 0.1419146 | 0.1419146 | ✓ |

### N246：expansion → action 39

path=[2, 20, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 27 个代表动作。trace 行 6811。

bucket=0，compatibility_richness_prior；到达 N247（新建）；closure=[]。

## iteration 269

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[38, 19, 34, 37, 18, 28, 22, 17, 12, 31, 27, 14, 5, 13]

### N0：selection → action 2

path=[]；visits=268；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6831。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8728788 | 0.8728788 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6629819 | 0.0861626 | 0.7491445 |  |
| 2 | 173 | 0.3264006 | 64.9464114 | 1.0 | 0.0429929 | 1.0429929 | ✓ |

### N5：selection → action 20

path=[2]；visits=173；children=14；K=14。最低访问优先：child.visits < 5；最少 4 次；候选 [20]，并列按 prior 抽样。trace 行 6833。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.6323007 | 0.1694298 | 0.8017305 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0654269 | 0.0673497 | 0.1327766 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2285107 | 0.2285107 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.5074735 | 0.1256067 | 0.6330802 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.4182323 | 0.1832534 | 0.6014857 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.7003543 | 0.1326091 | 0.8329634 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.6764543 | 0.1378742 | 0.8143286 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.7162977 | 0.048062 | 0.7643597 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.6939048 | 0.0656464 | 0.7595512 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 0.8003734 | 0.0353283 | 0.8357016 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.6970818 | 0.0944403 | 0.7915221 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.5527433 | 0.0889819 | 0.6417252 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.7649873 | 0.0619937 | 0.8269809 |  |
| 20 | 4 | 0.0072303 | 66.9914995 | 1.0 | 0.026628 | 1.026628 | ✓ |

### N245：expansion → action 6

path=[2, 20]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 7 个代表动作。trace 行 6835。

bucket=0，compatibility_richness_prior；到达 N248（新建）；closure=[]。

## iteration 270

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[20, 1, 26, 12, 10, 25, 34, 33, 7, 30, 6, 11, 16]

### N0：selection → action 2

path=[]；visits=269；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6855。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8745058 | 0.8745058 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7004666 | 0.0863232 | 0.7867898 |  |
| 2 | 174 | 0.3264006 | 64.7908152 | 1.0 | 0.0428269 | 1.0428269 | ✓ |

### N5：selection → action 8

path=[2]；visits=174；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6857。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7900072 | 0.1699188 | 0.959926 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0817455 | 0.0675441 | 0.1492896 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2291702 | 0.2291702 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.634046 | 0.1259692 | 0.7600152 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5225466 | 0.1837822 | 0.7063288 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8750345 | 0.1329918 | 1.0080263 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8451735 | 0.1382721 | 0.9834456 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.8949544 | 0.0482007 | 0.9431552 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8669764 | 0.0658359 | 0.9328123 |  |
| 8 | 42 | 0.0824973 | 65.3111437 | 1.0 | 0.0354302 | 1.0354302 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8709458 | 0.0947128 | 0.9656586 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6906069 | 0.0892387 | 0.7798456 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.955788 | 0.0621726 | 1.0179606 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7143142 | 0.0222541 | 0.7365683 |  |

### N166：selection → action 3

path=[2, 8]；visits=42；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6859。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.5127265 | 0.1341242 | 0.6468507 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.4741965 | 0.0178465 | 0.4920429 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.5585884 | 0.1440077 | 0.7025961 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.4724497 | 0.1899881 | 0.6624378 |  |
| 33 | 13 | 0.1164611 | 65.289632 | 0.5746304 | 0.0754754 | 0.6501058 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2409966 | 0.2409966 |  |
| 3 | 7 | 0.0736012 | 67.1018714 | 1.0 | 0.0834733 | 1.0834733 | ✓ |

### N126：selection → action 24

path=[2, 3, 8]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [24]，并列按 prior 抽样。trace 行 6861。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0899073 | 73.0976308 | 1.0 | 0.1110072 | 1.1110072 |  |
| 32 | 2 | 0.1091457 | 63.1261881 | 0.0 | 0.1347604 | 0.1347604 |  |
| 24 | 1 | 0.041313 | 67.9383794 | 0.4825973 | 0.0765127 | 0.55911 | ✓ |

### N217：expansion → action 32

path=[2, 3, 8, 24]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 6863。

bucket=0，compatibility_richness_prior；到达 N249（新建）；closure=[]。

## iteration 271

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[36, 25, 13, 38, 4, 39, 23, 7, 16, 28, 30, 35, 34, 17, 9]

### N0：selection → action 2

path=[]；visits=270；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6882。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8761298 | 0.8761298 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7180603 | 0.0864835 | 0.8045438 |  |
| 2 | 175 | 0.3264006 | 64.7233872 | 1.0 | 0.0426627 | 1.0426627 | ✓ |

### N5：selection → action 29

path=[2]；visits=175；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6884。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8110402 | 0.1704064 | 0.9814466 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0839219 | 0.0677379 | 0.1516598 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2298278 | 0.2298278 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6509267 | 0.1263307 | 0.7772574 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5364588 | 0.1843096 | 0.7207683 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8983313 | 0.1333734 | 1.0317047 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8676752 | 0.1386689 | 1.0063441 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9187815 | 0.0483391 | 0.9671206 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8900586 | 0.0660248 | 0.9560834 |  |
| 8 | 43 | 0.0824973 | 65.1364268 | 1.0 | 0.0347243 | 1.0347243 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8941337 | 0.0949846 | 0.9891183 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7089935 | 0.0894948 | 0.7984883 |  |
| 29 | 25 | 0.0875326 | 65.0132815 | 0.9812348 | 0.062351 | 1.0435858 | ✓ |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.733332 | 0.0223179 | 0.7556499 |  |

### N222：expansion → action 19

path=[2, 29]；visits=25；children=5；K=6。已有 5 条动作边 < K=6，且尚余 21 个代表动作。trace 行 6886。

bucket=1，uniform_random；到达 N250（新建）；closure=[]。

## iteration 272

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[25, 39, 26, 14, 34, 0, 12, 13, 31, 37, 36, 6, 21, 11, 15]

### N0：selection → action 2

path=[]；visits=271；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6907。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8777507 | 0.8777507 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7035645 | 0.0866435 | 0.790208 |  |
| 2 | 176 | 0.3264006 | 64.7786978 | 1.0 | 0.0425001 | 1.0425001 | ✓ |

### N5：selection → action 29

path=[2]；visits=176；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6909。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.800255 | 0.1708925 | 0.9711476 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0828059 | 0.0679312 | 0.1507371 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2304835 | 0.2304835 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6422707 | 0.1266911 | 0.7689618 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5293249 | 0.1848354 | 0.7141604 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8863853 | 0.133754 | 1.0201392 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8561369 | 0.1390645 | 0.9952014 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9065636 | 0.048477 | 0.9550406 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8782227 | 0.0662132 | 0.9444358 |  |
| 8 | 43 | 0.0824973 | 65.1364268 | 0.986702 | 0.0348234 | 1.0215254 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8822436 | 0.0952556 | 0.9774992 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.6995653 | 0.0897501 | 0.7893154 |  |
| 29 | 26 | 0.0875326 | 65.2248699 | 1.0 | 0.060213 | 1.060213 | ✓ |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7235802 | 0.0223816 | 0.7459618 |  |

### N222：selection → action 19

path=[2, 29]；visits=26；children=6；K=6。最低访问优先：child.visits < 5；最少 1 次；候选 [19]，并列按 prior 抽样。trace 行 6911。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0730283 | 0.0730283 |  |
| 7 | 6 | 0.038653 | 65.6628762 | 0.4653304 | 0.0394185 | 0.5047489 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.2557783 | 0.0487388 | 0.304517 |  |
| 0 | 6 | 0.0464689 | 64.672077 | 0.3561418 | 0.0473892 | 0.403531 |  |
| 14 | 13 | 0.0350093 | 65.3834501 | 0.4345369 | 0.0178513 | 0.4523882 |  |
| 19 | 1 | 0.0408954 | 70.5145782 | 1.0 | 0.1459685 | 1.1459685 | ✓ |

### N250：expansion → action 28

path=[2, 29, 19]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 6913。

bucket=0，compatibility_richness_prior；到达 N251（新建）；closure=[]。

## iteration 273

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 28, 36, 39, 26, 34, 4, 13, 10, 22, 17, 25, 3, 6]

### N0：selection → action 2

path=[]；visits=272；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6934。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8793687 | 0.8793687 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.6966894 | 0.0868032 | 0.7834926 |  |
| 2 | 177 | 0.3264006 | 64.8057358 | 1.0 | 0.0423393 | 1.0423393 | ✓ |

### N5：selection → action 29

path=[2]；visits=177；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6936。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.7896494 | 0.1713774 | 0.9610267 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0817085 | 0.0681239 | 0.1498324 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2311373 | 0.2311373 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6337588 | 0.1270505 | 0.7608093 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5223099 | 0.1853598 | 0.7076696 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8746381 | 0.1341334 | 1.0087715 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8447906 | 0.1394591 | 0.9842497 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.894549 | 0.0486145 | 0.9431635 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8665837 | 0.066401 | 0.9329847 |  |
| 8 | 43 | 0.0824973 | 65.1364268 | 0.9736254 | 0.0349222 | 1.0085476 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8705513 | 0.0955259 | 0.9660771 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.690294 | 0.0900047 | 0.7802988 |  |
| 29 | 27 | 0.0875326 | 65.3141968 | 1.0 | 0.0582273 | 1.0582273 | ✓ |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7139906 | 0.0224451 | 0.7364357 |  |

### N222：selection → action 19

path=[2, 29]；visits=27；children=6；K=6。最低访问优先：child.visits < 5；最少 2 次；候选 [19]，并列按 prior 抽样。trace 行 6938。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.0 | 0.0744194 | 0.0744194 |  |
| 7 | 6 | 0.038653 | 65.6628762 | 0.5530265 | 0.0401694 | 0.5931958 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.3039822 | 0.0496672 | 0.3536494 |  |
| 0 | 6 | 0.0464689 | 64.672077 | 0.4232603 | 0.0482919 | 0.4715522 |  |
| 14 | 13 | 0.0350093 | 65.3834501 | 0.5164297 | 0.0181914 | 0.5346211 |  |
| 19 | 2 | 0.0408954 | 69.0756374 | 1.0 | 0.0991661 | 1.0991661 | ✓ |

### N250：expansion → action 24

path=[2, 29, 19]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6940。

bucket=1，uniform_random；到达 N252（新建）；closure=[]。

## iteration 274

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[15, 39, 5, 9, 1, 7, 19, 11, 32, 28, 29, 23, 14]

### N0：selection → action 2

path=[]；visits=273；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6960。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8809837 | 0.8809837 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7488393 | 0.0869626 | 0.8358019 |  |
| 2 | 178 | 0.3264006 | 64.6130458 | 1.0 | 0.0421801 | 1.0421801 | ✓ |

### N5：selection → action 8

path=[2]；visits=178；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6962。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8110402 | 0.1718608 | 0.982901 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0839219 | 0.0683161 | 0.1522379 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2317894 | 0.2317894 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6509267 | 0.1274089 | 0.7783356 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5364588 | 0.1858827 | 0.7223414 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.8983313 | 0.1345118 | 1.032843 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8676752 | 0.1398525 | 1.0075277 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9187815 | 0.0487516 | 0.9675332 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8900586 | 0.0665883 | 0.956647 |  |
| 8 | 43 | 0.0824973 | 65.1364268 | 1.0 | 0.0350207 | 1.0350207 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.8941337 | 0.0957953 | 0.989929 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7089935 | 0.0902586 | 0.7992521 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.912426 | 0.056378 | 0.968804 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.733332 | 0.0225084 | 0.7558404 |  |

### N166：selection → action 3

path=[2, 8]；visits=43；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6964。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.8061175 | 0.1357115 | 0.941829 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.7455399 | 0.0180577 | 0.7635976 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.8782224 | 0.145712 | 1.0239344 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.7427936 | 0.1922366 | 0.9350301 |  |
| 33 | 13 | 0.1164611 | 65.289632 | 0.9034438 | 0.0763686 | 0.9798125 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2438488 | 0.2438488 |  |
| 3 | 8 | 0.0736012 | 65.5512795 | 1.0 | 0.0750766 | 1.0750766 | ✓ |

### N126：selection → action 36

path=[2, 3, 8]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [36, 32, 24]，并列按 prior 抽样。trace 行 6966。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0899073 | 73.0976308 | 1.0 | 0.1186716 | 1.1186716 | ✓ |
| 32 | 2 | 0.1091457 | 63.1261881 | 0.0252059 | 0.144065 | 0.1692709 |  |
| 24 | 2 | 0.041313 | 62.8683495 | 0.0 | 0.0545304 | 0.0545304 |  |

### N128：expansion → action 35

path=[2, 3, 8, 36]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 23 个代表动作。trace 行 6968。

bucket=1，uniform_random；到达 N253（新建）；closure=[]。

## iteration 275

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[10, 28, 20, 12, 4, 31, 37, 6, 35, 36, 16, 11, 2]

### N0：selection → action 2

path=[]；visits=274；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6987。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8825957 | 0.8825957 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7527108 | 0.0871218 | 0.8398325 |  |
| 2 | 179 | 0.3264006 | 64.5998056 | 1.0 | 0.0420225 | 1.0420225 | ✓ |

### N5：selection → action 25

path=[2]；visits=179；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 6989。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.816553 | 0.1723429 | 0.9888958 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0844923 | 0.0685077 | 0.153 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2324395 | 0.2324395 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6553511 | 0.1277663 | 0.7831174 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5401051 | 0.1864041 | 0.7265092 |  |
| 25 | 13 | 0.1008208 | 64.4692339 | 0.9044373 | 0.1348891 | 1.0393264 | ✓ |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8735729 | 0.1402447 | 1.0138176 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9250266 | 0.0488884 | 0.973915 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8961085 | 0.0667751 | 0.9628836 |  |
| 8 | 44 | 0.0824973 | 65.0921225 | 1.0 | 0.0343385 | 1.0343385 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9002112 | 0.096064 | 0.9962753 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7138126 | 0.0905118 | 0.8043244 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9186278 | 0.0565362 | 0.975164 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7383165 | 0.0225715 | 0.7608881 |  |

### N9：selection → action 29

path=[24, 3]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [29, 4, 28]，并列按 prior 抽样。trace 行 6991。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0620248 | 58.0144953 | 0.0 | 0.0391359 | 0.0391359 |  |
| 29 | 3 | 0.0592759 | 64.7159413 | 0.7758821 | 0.0748029 | 0.850685 | ✓ |
| 4 | 3 | 0.0594255 | 61.0265089 | 0.3487259 | 0.0749916 | 0.4237174 |  |
| 28 | 3 | 0.0677063 | 66.6516913 | 1.0 | 0.0854415 | 1.0854415 |  |

### N38：selection → action 18

path=[24, 3, 29]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [18, 33]，并列按 prior 抽样。trace 行 6993。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 18 | 1 | 0.0253072 | 68.8621859 | 1.0 | 0.0306833 | 1.0306833 | ✓ |
| 33 | 1 | 0.0053354 | 51.8470002 | 0.0 | 0.0064689 | 0.0064689 |  |

### N45：expansion → action 38

path=[24, 3, 29, 18]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 6995。

bucket=0，compatibility_richness_prior；到达 N254（新建）；closure=[]。

## iteration 276

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[18, 30, 16, 32, 29, 35, 5, 4, 25, 36, 3, 9, 17]

### N0：selection → action 2

path=[]；visits=275；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7014。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8842049 | 0.8842049 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7673261 | 0.0872806 | 0.8546067 |  |
| 2 | 180 | 0.3264006 | 64.5510264 | 1.0 | 0.0418665 | 1.0418665 | ✓ |

### N5：selection → action 8

path=[2]；visits=180；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7016。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.816553 | 0.1728236 | 0.9893766 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0844923 | 0.0686988 | 0.1531911 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2330879 | 0.2330879 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6553511 | 0.1281227 | 0.7834738 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5401051 | 0.186924 | 0.7270292 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7486657 | 0.1262477 | 0.8749133 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8735729 | 0.1406359 | 1.0142088 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9250266 | 0.0490247 | 0.9740513 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.8961085 | 0.0669614 | 0.9630698 |  |
| 8 | 44 | 0.0824973 | 65.0921225 | 1.0 | 0.0344343 | 1.0344343 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9002112 | 0.096332 | 0.9965432 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7138126 | 0.0907643 | 0.8045768 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9186278 | 0.0566939 | 0.9753217 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7383165 | 0.0226345 | 0.760951 |  |

### N166：selection → action 15

path=[2, 8]；visits=44；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7018。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.8922719 | 0.1372805 | 1.0295523 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.82522 | 0.0182664 | 0.8434864 |  |
| 15 | 6 | 0.1111044 | 65.2212871 | 0.972083 | 0.1473966 | 1.1194796 | ✓ |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.8221801 | 0.194459 | 1.0166391 |  |
| 33 | 13 | 0.1164611 | 65.289632 | 1.0 | 0.0772515 | 1.0772515 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2466679 | 0.2466679 |  |
| 3 | 9 | 0.0736012 | 65.2135307 | 0.9689147 | 0.0683502 | 1.0372649 |  |

### N170：selection → action 6

path=[2, 8, 15]；visits=6；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [28, 6]，并列按 prior 抽样。trace 行 7020。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0950075 | 60.7924577 | 0.0 | 0.1086026 | 0.1086026 |  |
| 28 | 1 | 0.0146656 | 66.0372356 | 1.0 | 0.0251463 | 1.0251463 |  |
| 6 | 1 | 0.0790746 | 62.5688704 | 0.3387012 | 0.1355847 | 0.4742859 | ✓ |

### N211：expansion → action 11

path=[2, 8, 15, 6]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7022。

bucket=0，compatibility_richness_prior；到达 N255（新建）；closure=[]。

## iteration 277

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[6, 38, 37, 11, 4, 27, 16, 0, 35, 1, 31, 17, 12]

### N0：selection → action 2

path=[]；visits=276；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7041。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.885811 | 0.885811 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.775088 | 0.0874391 | 0.8625272 |  |
| 2 | 181 | 0.3264006 | 64.5258688 | 1.0 | 0.0417121 | 1.0417121 | ✓ |

### N5：selection → action 8

path=[2]；visits=181；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7043。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.825847 | 0.173303 | 0.99915 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.085454 | 0.0688893 | 0.1543433 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2337345 | 0.2337345 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6628104 | 0.1284781 | 0.7912885 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5462526 | 0.1874425 | 0.7336952 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.757187 | 0.1265979 | 0.8837849 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.883516 | 0.1410261 | 1.024542 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9355553 | 0.0491607 | 0.9847161 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.906308 | 0.0671471 | 0.9734551 |  |
| 8 | 45 | 0.0824973 | 65.0187677 | 1.0 | 0.0337792 | 1.0337792 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9104575 | 0.0965992 | 1.0070567 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7219372 | 0.0910161 | 0.8129533 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9290837 | 0.0568511 | 0.9859348 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7467201 | 0.0226973 | 0.7694174 |  |

### N166：selection → action 33

path=[2, 8]；visits=45；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7045。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.8922719 | 0.1388317 | 1.0311036 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.82522 | 0.0184728 | 0.8436928 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.7385642 | 0.1304294 | 0.8689936 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.8221801 | 0.1966564 | 1.0188365 |  |
| 33 | 13 | 0.1164611 | 65.289632 | 1.0 | 0.0781245 | 1.0781245 | ✓ |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2494552 | 0.2494552 |  |
| 3 | 9 | 0.0736012 | 65.2135307 | 0.9689147 | 0.0691225 | 1.0380372 |  |

### N180：selection → action 29

path=[2, 8, 33]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 2 次；候选 [29]，并列按 prior 抽样。trace 行 7047。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 34 | 3 | 0.0376976 | 65.7939119 | 0.918382 | 0.0475723 | 0.9659543 |  |
| 19 | 3 | 0.0504112 | 64.8231042 | 0.8261121 | 0.0636161 | 0.8897281 |  |
| 29 | 2 | 0.0172087 | 56.1312631 | 0.0 | 0.0289553 | 0.0289553 | ✓ |
| 30 | 3 | 0.0497912 | 66.6526458 | 1.0 | 0.0628336 | 1.0628336 |  |

### N190：expansion → action 34

path=[2, 8, 33, 29]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 23 个代表动作。trace 行 7049。

bucket=1，uniform_random；到达 N256（新建）；closure=[]。

## iteration 278

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 18, 23, 38, 34, 33, 12, 32, 1, 10, 29, 39, 17, 13]

### N0：selection → action 2

path=[]；visits=277；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7068。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8874143 | 0.8874143 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7879855 | 0.0875974 | 0.8755829 |  |
| 2 | 182 | 0.3264006 | 64.4851617 | 1.0 | 0.0415593 | 1.0415593 | ✓ |

### N5：selection → action 36

path=[2]；visits=182；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7070。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8400125 | 0.1737811 | 1.0137936 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0869198 | 0.0690794 | 0.1559991 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2343793 | 0.2343793 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6741793 | 0.1288325 | 0.8030119 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5556223 | 0.1879596 | 0.7435819 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7701748 | 0.1269471 | 0.8971219 |  |
| 36 | 15 | 0.1197987 | 64.2680561 | 0.8986706 | 0.1414151 | 1.0400857 | ✓ |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9516026 | 0.0492964 | 1.0008989 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9218536 | 0.0673323 | 0.9891859 |  |
| 8 | 46 | 0.0824973 | 64.9100873 | 1.0 | 0.0331517 | 1.0331517 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9260743 | 0.0968657 | 1.02294 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7343204 | 0.0912671 | 0.8255875 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.94502 | 0.057008 | 1.0020279 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7595283 | 0.0227599 | 0.7822882 |  |

### N47：selection → action 31

path=[2, 36]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [31, 37]，并列按 prior 抽样。trace 行 7072。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 31 | 4 | 0.0602182 | 66.8250629 | 0.462459 | 0.0653028 | 0.5277617 | ✓ |
| 8 | 6 | 0.059354 | 63.6511971 | 0.0 | 0.0459754 | 0.0459754 |  |
| 37 | 4 | 0.0489272 | 70.5142186 | 1.0 | 0.0530584 | 1.0530584 |  |
| 39 | 5 | 0.0880029 | 63.9733695 | 0.0469432 | 0.0795279 | 0.1264711 |  |

### N48：expansion → action 4

path=[2, 36, 31]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 23 个代表动作。trace 行 7074。

bucket=0，compatibility_richness_prior；到达 N257（新建）；closure=[]。

## iteration 279

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 11, 39, 36, 29, 35, 5, 22, 12, 32, 0, 37, 6, 14]

### N0：selection → action 2

path=[]；visits=278；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7094。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8890147 | 0.8890147 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.7927054 | 0.0877554 | 0.8804608 |  |
| 2 | 183 | 0.3264006 | 64.470596 | 1.0 | 0.0414079 | 1.0414079 | ✓ |

### N5：selection → action 8

path=[2]；visits=183；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7096。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8400125 | 0.1742578 | 1.0142703 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0869198 | 0.0692689 | 0.1561887 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2350223 | 0.2350223 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6741793 | 0.129186 | 0.8033653 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5556223 | 0.1884753 | 0.7440976 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7701748 | 0.1272954 | 0.8974702 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8079605 | 0.1334617 | 0.9414222 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 0.9516026 | 0.0494316 | 1.0010342 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9218536 | 0.0675171 | 0.9893707 |  |
| 8 | 46 | 0.0824973 | 64.9100873 | 1.0 | 0.0332426 | 1.0332426 | ✓ |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9260743 | 0.0971314 | 1.0232057 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7343204 | 0.0915175 | 0.8258379 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.94502 | 0.0571644 | 1.0021843 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7595283 | 0.0228223 | 0.7823507 |  |

### N166：selection → action 3

path=[2, 8]；visits=46；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7098。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 0.9208982 | 0.1403658 | 1.061264 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.8516952 | 0.018677 | 0.8703721 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.7622593 | 0.1318706 | 0.8941299 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.8485578 | 0.1988294 | 1.0473872 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.7852182 | 0.0737219 | 0.8589401 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2522117 | 0.2522117 |  |
| 3 | 9 | 0.0736012 | 65.2135307 | 1.0 | 0.0698863 | 1.0698863 | ✓ |

### N126：expansion → action 25

path=[2, 3, 8]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 14 个代表动作。trace 行 7100。

bucket=1，uniform_random；到达 N258（新建）；closure=[]。

## iteration 280

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[1, 26, 25, 19, 36, 23, 6, 11, 7, 15, 28, 10]

### N0：selection → action 2

path=[]；visits=279；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7120。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8906122 | 0.8906122 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8364759 | 0.0879131 | 0.924389 |  |
| 2 | 184 | 0.3264006 | 64.3433483 | 1.0 | 0.0412581 | 1.0412581 | ✓ |

### N5：selection → action 21

path=[2]；visits=184；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7122。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8827346 | 0.1747333 | 1.0574679 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0913404 | 0.0694579 | 0.1607983 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2356635 | 0.2356635 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7084673 | 0.1295385 | 0.8380058 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5838806 | 0.1889896 | 0.7728702 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.809345 | 0.1276427 | 0.9369877 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8490524 | 0.1338259 | 0.9828783 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 1.0 | 0.0495665 | 1.0495665 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9687381 | 0.0677013 | 1.0364393 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 0.9987009 | 0.0326389 | 1.0313398 |  |
| 21 | 13 | 0.0718016 | 64.4416879 | 0.9731734 | 0.0973965 | 1.0705698 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7716671 | 0.0917672 | 0.8634343 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9930826 | 0.0573203 | 1.0504029 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7981571 | 0.0228846 | 0.8210417 |  |

### N183：selection → action 3

path=[2, 21]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [3]，并列按 prior 抽样。trace 行 7124。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 4 | 0.2828124 | 62.8001221 | 0.1348931 | 0.2855145 | 0.4204076 |  |
| 8 | 7 | 0.1610609 | 74.4381899 | 1.0 | 0.1016249 | 1.1016249 |  |
| 0 | 4 | 0.1722104 | 60.9854395 | 0.0 | 0.1738557 | 0.1738557 |  |
| 3 | 3 | 0.0151285 | 65.4343483 | 0.3307063 | 0.0190913 | 0.3497976 | ✓ |

### N204：selection → action 20

path=[2, 21, 3]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [20, 0]，并列按 prior 抽样。trace 行 7126。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 20 | 1 | 0.0312054 | 64.0687753 | 0.0 | 0.0378345 | 0.0378345 | ✓ |
| 0 | 1 | 0.1015482 | 67.66512 | 1.0 | 0.1231206 | 1.1231206 |  |

### N206：expansion → action 0

path=[2, 21, 3, 20]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 5 个代表动作。trace 行 7128。

bucket=0，compatibility_richness_prior；到达 N259（新建）；closure=[]。

## iteration 281

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[9, 35, 25, 4, 10, 26, 7, 32, 19, 23, 1, 17, 34, 16]

### N0：selection → action 2

path=[]；visits=280；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7146。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8922069 | 0.8922069 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8372623 | 0.0880705 | 0.9253327 |  |
| 2 | 185 | 0.3264006 | 64.3411839 | 1.0 | 0.0411098 | 1.0411098 | ✓ |

### N5：selection → action 21

path=[2]；visits=185；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7148。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8827346 | 0.1752075 | 1.0579421 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0913404 | 0.0696464 | 0.1609868 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2363031 | 0.2363031 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7084673 | 0.12989 | 0.8383573 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5838806 | 0.1895024 | 0.7733831 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.809345 | 0.1279891 | 0.9373341 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8490524 | 0.134189 | 0.9832415 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 1.0 | 0.049701 | 1.049701 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9687381 | 0.067885 | 1.0366231 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 0.9987009 | 0.0327275 | 1.0314283 |  |
| 21 | 14 | 0.0718016 | 64.4170851 | 0.9690929 | 0.0911501 | 1.060243 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7716671 | 0.0920163 | 0.8636833 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9930826 | 0.0574759 | 1.0505585 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7981571 | 0.0229467 | 0.8211038 |  |

### N183：selection → action 25

path=[2, 21]；visits=14；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [25, 0, 3]，并列按 prior 抽样。trace 行 7150。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 4 | 0.2828124 | 62.8001221 | 0.1348931 | 0.2962924 | 0.4311855 | ✓ |
| 8 | 7 | 0.1610609 | 74.4381899 | 1.0 | 0.1054611 | 1.1054611 |  |
| 0 | 4 | 0.1722104 | 60.9854395 | 0.0 | 0.1804186 | 0.1804186 |  |
| 3 | 4 | 0.0151285 | 65.1000737 | 0.3058582 | 0.0158496 | 0.3217078 |  |

### N184：expansion → action 9

path=[2, 21, 25]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 16 个代表动作。trace 行 7152。

bucket=0，compatibility_richness_prior；到达 N50（复用）；closure=[]。

## iteration 282

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[14, 39, 22, 10, 3, 29, 31, 4, 0, 13, 17, 33, 24, 19]

### N0：selection → action 2

path=[]；visits=281；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7173。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8937987 | 0.8937987 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8466835 | 0.0882276 | 0.9349111 |  |
| 2 | 186 | 0.3264006 | 64.3155651 | 1.0 | 0.0409629 | 1.0409629 | ✓ |

### N5：selection → action 9

path=[2]；visits=186；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7175。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 7 | 0.0736086 | 63.8963936 | 0.8827346 | 0.1756804 | 1.058415 | ✓ |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0913404 | 0.0698344 | 0.1611748 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2369409 | 0.2369409 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7084673 | 0.1302406 | 0.8387079 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5838806 | 0.1900139 | 0.7738945 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.809345 | 0.1283345 | 0.9376796 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8490524 | 0.1345512 | 0.9836036 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 1.0 | 0.0498351 | 1.0498351 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9687381 | 0.0680682 | 1.0368063 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 0.9987009 | 0.0328158 | 1.0315167 |  |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9357624 | 0.0856838 | 1.0214462 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7716671 | 0.0922646 | 0.8639317 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9930826 | 0.057631 | 1.0507136 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7981571 | 0.0230087 | 0.8211657 |  |

### N6：selection → action 15

path=[2, 9]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [15]，并列按 prior 抽样。trace 行 7177。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 0.144154 | 58.8673322 | 0.0 | 0.1334885 | 0.1334885 |  |
| 15 | 2 | 0.0145787 | 69.0178764 | 1.0 | 0.0180001 | 1.0180001 | ✓ |
| 24 | 3 | 0.0455751 | 65.3839303 | 0.6419949 | 0.0422031 | 0.6841981 |  |

### N19：expansion → action 28

path=[2, 9, 15]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 12 个代表动作。trace 行 7179。

bucket=1，uniform_random；到达 N260（新建）；closure=[]。

## iteration 283

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[0, 26, 13, 22, 31, 14, 36, 24, 35, 3, 16, 33, 8]

### N0：selection → action 2

path=[]；visits=282；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7199。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8953877 | 0.8953877 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8654372 | 0.0883845 | 0.9538217 |  |
| 2 | 187 | 0.3264006 | 64.2662292 | 1.0 | 0.0408174 | 1.0408174 | ✓ |

### N5：selection → action 29

path=[2]；visits=187；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7201。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5977486 | 0.1565796 | 0.7543282 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0913404 | 0.0700218 | 0.1613623 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2375769 | 0.2375769 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7084673 | 0.1305902 | 0.8390576 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5838806 | 0.190524 | 0.7744046 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.809345 | 0.1286791 | 0.9380241 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8490524 | 0.1349124 | 0.9839649 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 1.0 | 0.0499689 | 1.0499689 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9687381 | 0.068251 | 1.036989 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 0.9987009 | 0.0329039 | 1.0316048 |  |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9357624 | 0.0859138 | 1.0216763 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7716671 | 0.0925123 | 0.8641794 |  |
| 29 | 28 | 0.0875326 | 64.5617293 | 0.9930826 | 0.0577857 | 1.0508683 | ✓ |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7981571 | 0.0230704 | 0.8212275 |  |

### N222：selection → action 19

path=[2, 29]；visits=28；children=6；K=6。最低访问优先：child.visits < 5；最少 3 次；候选 [19]，并列按 prior 抽样。trace 行 7203。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 5 | 0.0613801 | 61.4403727 | 0.1319012 | 0.075785 | 0.2076863 |  |
| 7 | 6 | 0.038653 | 65.6628762 | 1.0 | 0.0409065 | 1.0409065 |  |
| 13 | 6 | 0.0477923 | 63.7613573 | 0.6090693 | 0.0505786 | 0.6596479 |  |
| 0 | 6 | 0.0464689 | 64.672077 | 0.7963029 | 0.0491781 | 0.845481 |  |
| 14 | 14 | 0.0350093 | 65.3834501 | 0.9425532 | 0.0172902 | 0.9598433 |  |
| 19 | 3 | 0.0408954 | 60.7987942 | 0.0 | 0.0757394 | 0.0757394 | ✓ |

### N250：selection → action 28

path=[2, 29, 19]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [28, 24]，并列按 prior 抽样。trace 行 7205。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 1 | 0.0283603 | 67.6366966 | 1.0 | 0.0343851 | 1.0343851 | ✓ |
| 24 | 1 | 0.0032783 | 44.2451078 | 0.0 | 0.0039747 | 0.0039747 |  |

### N251：expansion → action 25

path=[2, 29, 19, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7207。

bucket=0，compatibility_richness_prior；到达 N261（新建）；closure=[]。

## iteration 284

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[12, 14, 38, 13, 10, 19, 31, 0, 30, 33, 28, 26, 5, 9, 15]

### N0：selection → action 2

path=[]；visits=283；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7226。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8969738 | 0.8969738 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9073273 | 0.088541 | 0.9958684 |  |
| 2 | 188 | 0.3264006 | 64.1633931 | 1.0 | 0.0406734 | 1.0406734 | ✓ |

### N5：selection → action 24

path=[2]；visits=188；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7228。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5977486 | 0.1569977 | 0.7547463 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0913404 | 0.0702088 | 0.1615492 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2382113 | 0.2382113 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7084673 | 0.1309389 | 0.8394063 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5838806 | 0.1910327 | 0.7749134 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.809345 | 0.1290227 | 0.9383677 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8490524 | 0.1352727 | 0.9843251 |  |
| 24 | 9 | 0.0261006 | 64.6034372 | 1.0 | 0.0501023 | 1.0501023 | ✓ |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9687381 | 0.0684332 | 1.0371713 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 0.9987009 | 0.0329917 | 1.0316926 |  |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9357624 | 0.0861433 | 1.0219057 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7716671 | 0.0927593 | 0.8644264 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9227564 | 0.0560087 | 0.9787651 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7981571 | 0.023132 | 0.8212891 |  |

### N68：expansion → action 37

path=[2, 24]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 16 个代表动作。trace 行 7230。

bucket=1，uniform_random；到达 N262（新建）；closure=[]。

## iteration 285

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[0, 36, 11, 13, 8, 25, 35, 30, 28, 34, 10, 21, 14]

### N0：selection → action 2

path=[]；visits=284；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7251。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.8985572 | 0.8985572 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9247698 | 0.0886973 | 1.0134671 |  |
| 2 | 189 | 0.3264006 | 64.1233209 | 1.0 | 0.0405307 | 1.0405307 | ✓ |

### N5：selection → action 3

path=[2]；visits=189；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7253。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5985262 | 0.1574147 | 0.7559408 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0914592 | 0.0703953 | 0.1618545 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.238844 | 0.238844 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7093889 | 0.1312867 | 0.8406756 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5846402 | 0.1915401 | 0.7761803 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8103978 | 0.1293654 | 0.9397632 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8501569 | 0.135632 | 0.9857889 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6383394 | 0.0456686 | 0.6840079 |  |
| 3 | 11 | 0.04278 | 64.4149455 | 0.9699982 | 0.068615 | 1.0386132 | ✓ |
| 8 | 47 | 0.0824973 | 64.5956042 | 1.0 | 0.0330794 | 1.0330794 |  |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9369797 | 0.0863721 | 1.0233517 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7726709 | 0.0930057 | 0.8656766 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9239567 | 0.0561574 | 0.9801141 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7991953 | 0.0231935 | 0.8223888 |  |

### N122：selection → action 6

path=[2, 3]；visits=11；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [6]，并列按 prior 抽样。trace 行 7255。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 3 | 0.143538 | 63.5769699 | 0.386432 | 0.1666216 | 0.5530536 | ✓ |
| 18 | 4 | 0.1242323 | 56.8318776 | 0.0 | 0.115369 | 0.115369 |  |
| 8 | 10 | 0.1439711 | 74.2866761 | 1.0 | 0.0607725 | 1.0607725 |  |
| 36 | 5 | 0.1853786 | 72.1266796 | 0.876252 | 0.1434606 | 1.0197126 |  |

### N123：selection → action 32

path=[2, 3, 6]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [32, 34]，并列按 prior 抽样。trace 行 7257。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 32 | 1 | 0.0499085 | 65.2205025 | 1.0 | 0.0605109 | 1.0605109 | ✓ |
| 34 | 1 | 0.0550298 | 62.683571 | 0.0 | 0.0667201 | 0.0667201 |  |

### N127：expansion → action 38

path=[2, 3, 6, 32]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7259。

bucket=0，compatibility_richness_prior；到达 N263（新建）；closure=[]。

## iteration 286

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[30, 15, 38, 32, 39, 22, 17, 19, 21, 33, 11, 6]

### N0：selection → action 2

path=[]；visits=285；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7278。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9001378 | 0.9001378 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9297991 | 0.0888533 | 1.0186525 |  |
| 2 | 190 | 0.3264006 | 64.1120458 | 1.0 | 0.0403895 | 1.0403895 | ✓ |

### N5：selection → action 8

path=[2]；visits=190；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7280。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5985262 | 0.1578305 | 0.7563567 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0914592 | 0.0705813 | 0.1620405 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2394751 | 0.2394751 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7093889 | 0.1316336 | 0.8410225 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5846402 | 0.1920462 | 0.7766864 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8103978 | 0.1297071 | 0.940105 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8501569 | 0.1359903 | 0.9861472 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6383394 | 0.0457892 | 0.6841286 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9248825 | 0.0635042 | 0.9883867 |  |
| 8 | 47 | 0.0824973 | 64.5956042 | 1.0 | 0.0331668 | 1.0331668 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9369797 | 0.0866002 | 1.0235799 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7726709 | 0.0932514 | 0.8659223 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9239567 | 0.0563058 | 0.9802625 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7991953 | 0.0232547 | 0.8224501 |  |

### N166：selection → action 12

path=[2, 8]；visits=47；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7282。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 8 | 0.1330445 | 65.0258976 | 1.0 | 0.1418833 | 1.1418833 | ✓ |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.9248526 | 0.0188789 | 0.9437315 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.8277345 | 0.1332963 | 0.9610308 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.9214458 | 0.200979 | 1.1224248 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.8526656 | 0.0745189 | 0.9271845 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2549384 | 0.2549384 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.2227272 | 0.0642199 | 0.2869471 |  |

### N167：selection → action 3

path=[2, 8, 12]；visits=8；children=3；K=3。最低访问优先：child.visits < 5；最少 2 次；候选 [17, 3]，并列按 prior 抽样。trace 行 7284。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 3 | 0.0206411 | 67.2158548 | 1.0 | 0.0204336 | 1.0204336 |  |
| 17 | 2 | 0.0423435 | 67.0126978 | 0.9763871 | 0.0558906 | 1.0322777 |  |
| 3 | 2 | 0.0185985 | 58.6122057 | 0.0 | 0.0245487 | 0.0245487 | ✓ |

### N179：expansion → action 37

path=[2, 8, 12, 3]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 35 个代表动作。trace 行 7286。

bucket=1，uniform_random；到达 N264（新建）；closure=[]。

## iteration 287

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[30, 7, 32, 38, 36, 23, 19, 34, 10, 11, 18, 0, 33, 3]

### N0：selection → action 2

path=[]；visits=286；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7304。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9017156 | 0.9017156 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.915572 | 0.0890091 | 1.004581 |  |
| 2 | 191 | 0.3264006 | 64.1442617 | 1.0 | 0.0402495 | 1.0402495 | ✓ |

### N5：selection → action 8

path=[2]；visits=191；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7306。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5916225 | 0.1582453 | 0.7498679 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0904043 | 0.0707668 | 0.1611711 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2401044 | 0.2401044 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7012065 | 0.1319795 | 0.833186 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5778967 | 0.1925509 | 0.7704476 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8010503 | 0.130048 | 0.9310984 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8403508 | 0.1363477 | 0.9766985 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6309765 | 0.0459096 | 0.676886 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9142145 | 0.0636711 | 0.9778856 |  |
| 8 | 48 | 0.0824973 | 64.6658705 | 1.0 | 0.0325753 | 1.0325753 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9261721 | 0.0868278 | 1.013 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7637585 | 0.0934965 | 0.8572551 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9132994 | 0.0564538 | 0.9697532 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7899771 | 0.0233159 | 0.8132929 |  |

### N166：selection → action 12

path=[2, 8]；visits=48；children=7；K=7。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7308。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 9 | 0.1330445 | 65.3528406 | 1.0 | 0.1290463 | 1.1290463 | ✓ |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.8044499 | 0.0190787 | 0.8235286 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.7199752 | 0.1347069 | 0.8546821 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.8014866 | 0.2031058 | 1.0045924 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.7416606 | 0.0753075 | 0.8169681 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2576362 | 0.2576362 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.1937313 | 0.0648995 | 0.2586307 |  |

### N167：expansion → action 39

path=[2, 8, 12]；visits=9；children=3；K=4。已有 3 条动作边 < K=4，且尚余 34 个代表动作。trace 行 7310。

bucket=1，uniform_random；到达 N265（新建）；closure=[]。

## iteration 288

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class43；rollout：[5, 19, 36, 32, 22, 39, 9, 21, 35, 7, 10, 26, 16, 13]

### N0：selection → action 2

path=[]；visits=287；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7330。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9032906 | 0.9032906 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9093271 | 0.0891646 | 0.9984916 |  |
| 2 | 192 | 0.3264006 | 64.1587209 | 1.0 | 0.0401109 | 1.0401109 | ✓ |

### N5：selection → action 8

path=[2]；visits=192；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7332。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5892072 | 0.1586591 | 0.7478662 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0900352 | 0.0709518 | 0.160987 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2407322 | 0.2407322 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6983438 | 0.1323245 | 0.8306683 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5755373 | 0.1930543 | 0.7685916 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.79778 | 0.130388 | 0.928168 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.83692 | 0.1367042 | 0.9736241 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6284005 | 0.0460296 | 0.67443 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9104821 | 0.0638376 | 0.9743197 |  |
| 8 | 49 | 0.0824973 | 64.6908432 | 1.0 | 0.0320072 | 1.0320072 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9223909 | 0.0870548 | 1.0094458 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7606404 | 0.0937409 | 0.8543814 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9095707 | 0.0566014 | 0.9661721 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7867519 | 0.0233768 | 0.8101287 |  |

### N166：expansion → action 0

path=[2, 8]；visits=49；children=7；K=8。已有 7 条动作边 < K=8，且尚余 2 个代表动作。trace 行 7334。

bucket=1，uniform_random；到达 N266（新建）；closure=[]。

## iteration 289

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 39, 7, 30, 22, 26, 25, 23, 12, 24, 21, 33, 10, 15]

### N0：selection → action 2

path=[]；visits=288；children=3；K=17。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7354。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9048629 | 0.9048629 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8615987 | 0.0893198 | 0.9509185 |  |
| 2 | 193 | 0.3264006 | 64.2761524 | 1.0 | 0.0399736 | 1.0399736 | ✓ |

### N5：selection → action 8

path=[2]；visits=193；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7356。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5638403 | 0.1590717 | 0.722912 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.086159 | 0.0711363 | 0.1572953 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2413583 | 0.2413583 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6682783 | 0.1326687 | 0.800947 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.550759 | 0.1935564 | 0.7443154 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7634335 | 0.1307271 | 0.8941606 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8008884 | 0.1370597 | 0.9379482 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6013462 | 0.0461493 | 0.6474955 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.8712835 | 0.0640036 | 0.9352872 |  |
| 8 | 50 | 0.0824973 | 64.9660362 | 1.0 | 0.0314613 | 1.0314613 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.8826797 | 0.0872813 | 0.9699609 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7278929 | 0.0939847 | 0.8218777 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.8704114 | 0.0567486 | 0.92716 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7528802 | 0.0234376 | 0.7763178 |  |

### N166：selection → action 0

path=[2, 8]；visits=50；children=8；K=8。最低访问优先：child.visits < 5；最少 1 次；候选 [0]，并列按 prior 抽样。trace 行 7358。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 0.1643298 | 0.1197339 | 0.2840637 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.1294291 | 0.0194721 | 0.1489012 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.1158379 | 0.1374846 | 0.2533225 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.1289523 | 0.207294 | 0.3362464 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.1193269 | 0.0768604 | 0.1961873 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2629489 | 0.2629489 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.0311697 | 0.0662377 | 0.0974074 |  |
| 0 | 1 | 0.1446565 | 78.4504965 | 1.0 | 0.7160133 | 1.7160133 | ✓ |

### N266：expansion → action 36

path=[2, 8, 0]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 37 个代表动作。trace 行 7360。

bucket=0，compatibility_richness_prior；到达 N267（新建）；closure=[]。

## iteration 290

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 36, 5, 19, 35, 22, 7, 25, 30, 28, 34, 26, 33, 21]

### N0：selection → action 2

path=[]；visits=289；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7380。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9064325 | 0.9064325 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8773154 | 0.0894747 | 0.9667901 |  |
| 2 | 194 | 0.3264006 | 64.2360718 | 1.0 | 0.0398376 | 1.0398376 | ✓ |

### N5：selection → action 8

path=[2]；visits=194；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7382。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5737095 | 0.1594833 | 0.7331928 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.087667 | 0.0713204 | 0.1589874 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2419827 | 0.2419827 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6799755 | 0.133012 | 0.8129875 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5603992 | 0.1940572 | 0.7544564 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7767963 | 0.1310654 | 0.9078617 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8149068 | 0.1374143 | 0.9523212 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6118719 | 0.0462687 | 0.6581406 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.8865341 | 0.0641692 | 0.9507033 |  |
| 8 | 51 | 0.0824973 | 64.8560776 | 1.0 | 0.0309361 | 1.0309361 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.8981297 | 0.0875071 | 0.9856368 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7406336 | 0.0942279 | 0.8348616 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.8856467 | 0.0568954 | 0.9425421 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7660583 | 0.0234983 | 0.7895566 |  |

### N166：selection → action 0

path=[2, 8]；visits=51；children=8；K=8。最低访问优先：child.visits < 5；最少 2 次；候选 [0]，并列按 prior 抽样。trace 行 7384。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 0.4230735 | 0.1209253 | 0.5439989 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.3332204 | 0.0196658 | 0.3528862 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.2982291 | 0.1388527 | 0.4370818 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.3319929 | 0.2093567 | 0.5413496 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.3072117 | 0.0776252 | 0.3848369 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2655654 | 0.2655654 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.0802476 | 0.0668968 | 0.1471445 |  |
| 0 | 2 | 0.1446565 | 68.9043209 | 1.0 | 0.482092 | 1.482092 | ✓ |

### N266：expansion → action 17

path=[2, 8, 0]；visits=2；children=1；K=2。已有 1 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7386。

bucket=1，uniform_random；到达 N268（新建）；closure=[]。

## iteration 291

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[5, 21, 9, 25, 32, 12, 26, 18, 11, 10, 19, 1]

### N0：selection → action 2

path=[]；visits=290；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7406。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9079994 | 0.9079994 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8701575 | 0.0896294 | 0.9597868 |  |
| 2 | 195 | 0.3264006 | 64.2541464 | 1.0 | 0.0397029 | 1.0397029 | ✓ |

### N5：selection → action 8

path=[2]；visits=195；children=14；K=14。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7408。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5708856 | 0.1598938 | 0.7307794 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0872355 | 0.071504 | 0.1587395 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2426056 | 0.2426056 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6766286 | 0.1333543 | 0.8099829 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5576408 | 0.1945567 | 0.7521975 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7729728 | 0.1314027 | 0.9043755 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8108957 | 0.137768 | 0.9486637 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6088602 | 0.0463878 | 0.655248 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.8821704 | 0.0643344 | 0.9465048 |  |
| 8 | 52 | 0.0824973 | 64.8871522 | 1.0 | 0.0304305 | 1.0304305 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.8937089 | 0.0877323 | 0.9814413 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7369881 | 0.0944705 | 0.8314585 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.8812874 | 0.0570419 | 0.9383292 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7622876 | 0.0235587 | 0.7858464 |  |

### N166：selection → action 0

path=[2, 8]；visits=52；children=8；K=8。最低访问优先：child.visits < 5；最少 3 次；候选 [0]，并列按 prior 抽样。trace 行 7410。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 0.4883858 | 0.1221051 | 0.6104909 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.3846615 | 0.0198577 | 0.4045192 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.3442684 | 0.1402074 | 0.4844758 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.3832445 | 0.2113992 | 0.5946437 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.3546377 | 0.0783825 | 0.4330202 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2681563 | 0.2681563 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.0926359 | 0.0675495 | 0.1601854 |  |
| 0 | 3 | 0.1446565 | 68.0935326 | 1.0 | 0.3650966 | 1.3650966 | ✓ |

### N266：selection → action 36

path=[2, 8, 0]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [36, 17]，并列按 prior 抽样。trace 行 7412。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 1 | 0.0363233 | 59.3581453 | 0.0 | 0.0440396 | 0.0440396 | ✓ |
| 17 | 1 | 0.0393489 | 66.4719559 | 1.0 | 0.0477081 | 1.0477081 |  |

### N267：expansion → action 35

path=[2, 8, 0, 36]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7414。

bucket=0，compatibility_richness_prior；到达 N269（新建）；closure=[]。

## iteration 292

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[26, 30, 1, 25, 39, 33, 32, 35, 28, 37, 34, 11, 6, 17, 18, 13]

### N0：selection → action 2

path=[]；visits=291；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7432。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9095635 | 0.9095635 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8967412 | 0.0897838 | 0.986525 |  |
| 2 | 196 | 0.3264006 | 64.1884739 | 1.0 | 0.0395694 | 1.0395694 | ✓ |

### N5：selection → action 8

path=[2]；visits=196；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7434。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5863463 | 0.1603032 | 0.7466495 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.089598 | 0.0716871 | 0.1612851 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2432269 | 0.2432269 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6949529 | 0.1336958 | 0.8286488 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5727428 | 0.1950549 | 0.7677977 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7939063 | 0.1317392 | 0.9256456 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8328563 | 0.1381208 | 0.9709771 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6253493 | 0.0465066 | 0.6718558 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9060612 | 0.0644991 | 0.9705604 |  |
| 8 | 53 | 0.0824973 | 64.7206884 | 1.0 | 0.0299435 | 1.0299435 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9179123 | 0.087957 | 1.0058693 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7569471 | 0.0947124 | 0.8516595 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9051543 | 0.0571879 | 0.9623422 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7829318 | 0.0236191 | 0.8065509 |  |

### N166：selection → action 0

path=[2, 8]；visits=53；children=8；K=8。最低访问优先：child.visits < 5；最少 4 次；候选 [0]，并列按 prior 抽样。trace 行 7436。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 1.0 | 0.1232736 | 1.1232736 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.7876181 | 0.0200477 | 0.8076658 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.7049109 | 0.1415491 | 0.8464599 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.7847167 | 0.2134223 | 0.998139 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.7261425 | 0.0791326 | 0.8052751 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2707225 | 0.2707225 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.1896778 | 0.0681959 | 0.2578737 |  |
| 0 | 4 | 0.1446565 | 65.0862926 | 0.8751603 | 0.2948723 | 1.1700326 | ✓ |

### N266：expansion → action 39

path=[2, 8, 0]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 35 个代表动作。trace 行 7438。

bucket=0，compatibility_richness_prior；到达 N199（复用）；closure=[]。

## iteration 293

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[35, 30, 38, 37, 20, 33, 13, 3, 29, 24, 10, 11, 18]

### N0：selection → action 2

path=[]；visits=292；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7461。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.911125 | 0.911125 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.8812084 | 0.0899379 | 0.9711463 |  |
| 2 | 197 | 0.3264006 | 64.2263649 | 1.0 | 0.0394371 | 1.0394371 | ✓ |

### N5：selection → action 8

path=[2]；visits=197；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7463。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5789665 | 0.1607117 | 0.7396782 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0884704 | 0.0718697 | 0.1603401 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2438465 | 0.2438465 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.6862063 | 0.1340365 | 0.8202427 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5655343 | 0.1955519 | 0.7610862 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.7839142 | 0.1320749 | 0.9159891 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.822374 | 0.1384727 | 0.9608467 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6174786 | 0.0466251 | 0.6641037 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.8946576 | 0.0646635 | 0.959321 |  |
| 8 | 54 | 0.0824973 | 64.7990366 | 1.0 | 0.0294739 | 1.0294739 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9063594 | 0.0881811 | 0.9945405 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7474202 | 0.0949537 | 0.8423739 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.893762 | 0.0573336 | 0.9510957 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7730778 | 0.0236792 | 0.7967571 |  |

### N166：selection → action 0

path=[2, 8]；visits=54；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7465。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 0.8499519 | 0.1244311 | 0.9743831 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.6694375 | 0.020236 | 0.6896735 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.5991403 | 0.1428782 | 0.7420185 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.6669715 | 0.2154263 | 0.8823977 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.6171862 | 0.0798757 | 0.6970619 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2732645 | 0.2732645 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.161217 | 0.0688363 | 0.2300532 |  |
| 0 | 5 | 0.1446565 | 65.8593322 | 1.0 | 0.2480343 | 1.2480343 | ✓ |

### N266：selection → action 17

path=[2, 8, 0]；visits=5；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [17]，并列按 prior 抽样。trace 行 7467。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0363233 | 57.7113591 | 0.0 | 0.0379033 | 0.0379033 |  |
| 17 | 1 | 0.0393489 | 66.4719559 | 0.7794034 | 0.0615908 | 0.8409943 | ✓ |
| 39 | 2 | 0.0470856 | 68.9514903 | 1.0 | 0.0491337 | 1.0491337 |  |

### N268：expansion → action 15

path=[2, 8, 0, 17]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7469。

bucket=0，compatibility_richness_prior；到达 N270（新建）；closure=[]。

## iteration 294

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[32, 15, 17, 10, 29, 5, 36, 13, 7, 38, 30, 11, 0]

### N0：selection → action 2

path=[]；visits=293；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7488。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9126838 | 0.9126838 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9025164 | 0.0900918 | 0.9926082 |  |
| 2 | 198 | 0.3264006 | 64.1747183 | 1.0 | 0.0393061 | 1.0393061 | ✓ |

### N5：selection → action 8

path=[2]；visits=198；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7490。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.591256 | 0.161119 | 0.7523751 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0903483 | 0.0720519 | 0.1624002 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2444647 | 0.2444647 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7007721 | 0.1343762 | 0.8351483 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5775387 | 0.1960476 | 0.7735862 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8005541 | 0.1324097 | 0.9329638 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8398302 | 0.1388237 | 0.978654 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6305856 | 0.0467433 | 0.6773289 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9136481 | 0.0648274 | 0.9784755 |  |
| 8 | 55 | 0.0824973 | 64.6696466 | 1.0 | 0.029021 | 1.029021 | ✓ |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9255984 | 0.0884046 | 1.014003 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7632854 | 0.0951944 | 0.8584798 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9127336 | 0.057479 | 0.9702126 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7894877 | 0.0237393 | 0.813227 |  |

### N166：selection → action 12

path=[2, 8]；visits=55；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7492。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 10 | 0.1330445 | 65.4065098 | 1.0 | 0.125578 | 1.125578 | ✓ |
| 14 | 10 | 0.0216367 | 64.8617446 | 0.7876181 | 0.0204225 | 0.8080405 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.7049109 | 0.1441951 | 0.8491059 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.7847167 | 0.2174118 | 1.0021285 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.7261425 | 0.0806119 | 0.8067544 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2757831 | 0.2757831 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.1896778 | 0.0694707 | 0.2591485 |  |
| 0 | 6 | 0.1446565 | 64.4965416 | 0.6452402 | 0.2145603 | 0.8598005 |  |

### N167：selection → action 39

path=[2, 8, 12]；visits=10；children=4；K=4。最低访问优先：child.visits < 5；最少 1 次；候选 [39]，并列按 prior 抽样。trace 行 7494。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 3 | 0.0206411 | 67.2158548 | 1.0 | 0.0228455 | 1.0228455 |  |
| 17 | 2 | 0.0423435 | 67.0126978 | 0.9629608 | 0.0624876 | 1.0254484 |  |
| 3 | 3 | 0.0185985 | 61.7309322 | 0.0 | 0.0205847 | 0.0205847 |  |
| 39 | 1 | 0.0432944 | 65.8895326 | 0.7581876 | 0.0958362 | 0.8540238 | ✓ |

### N265：expansion → action 35

path=[2, 8, 12, 39]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7496。

bucket=0，compatibility_richness_prior；到达 N271（新建）；closure=[]。

## iteration 295

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[12, 10, 6, 34, 22, 14, 25, 36, 17, 28, 24, 5, 39, 19]

### N0：selection → action 2

path=[]；visits=294；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7515。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.91424 | 0.91424 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9306034 | 0.0902454 | 1.0208488 |  |
| 2 | 199 | 0.3264006 | 64.110254 | 1.0 | 0.0391762 | 1.0391762 | ✓ |

### N5：selection → action 21

path=[2]；visits=199；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7517。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.6067689 | 0.1615254 | 0.7682943 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0927188 | 0.0722336 | 0.1649524 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2450812 | 0.2450812 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7191584 | 0.1347151 | 0.8538735 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5926916 | 0.196542 | 0.7892337 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8215584 | 0.1327436 | 0.954302 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.861865 | 0.1391739 | 1.0010388 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6471304 | 0.0468612 | 0.6939915 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9376197 | 0.0649909 | 1.0026106 |  |
| 8 | 56 | 0.0824973 | 64.5138033 | 1.0 | 0.0285838 | 1.0285838 |  |
| 21 | 15 | 0.0718016 | 64.2161212 | 0.9498835 | 0.0886276 | 1.038511 | ✓ |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7833118 | 0.0954345 | 0.8787463 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9366811 | 0.0576239 | 0.9943051 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.8102016 | 0.0237991 | 0.8340007 |  |

### N183：selection → action 0

path=[2, 21]；visits=15；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [0, 3]，并列按 prior 抽样。trace 行 7519。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 25 | 5 | 0.2828124 | 62.520623 | 0.1141167 | 0.2555765 | 0.3696932 |  |
| 8 | 7 | 0.1610609 | 74.4381899 | 1.0 | 0.1091626 | 1.1091626 |  |
| 0 | 4 | 0.1722104 | 60.9854395 | 0.0 | 0.186751 | 0.186751 | ✓ |
| 3 | 4 | 0.0151285 | 65.1000737 | 0.3058582 | 0.0164059 | 0.3222641 |  |

### N186：expansion → action 29

path=[2, 21, 0]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 25 个代表动作。trace 行 7521。

bucket=0，compatibility_richness_prior；到达 N272（新建）；closure=[]。

## iteration 296

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[38, 4, 23, 34, 20, 8, 11, 13, 16, 37, 25, 31, 7]

### N0：selection → action 2

path=[]；visits=295；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7541。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9157935 | 0.9157935 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.941695 | 0.0903987 | 1.0320938 |  |
| 2 | 200 | 0.3264006 | 64.0858561 | 1.0 | 0.0390476 | 1.0390476 | ✓ |

### N5：selection → action 8

path=[2]；visits=200；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7543。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.6067689 | 0.1619307 | 0.7686996 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0927188 | 0.0724149 | 0.1651336 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2456962 | 0.2456962 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7191584 | 0.1350532 | 0.8542116 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5926916 | 0.1970352 | 0.7897269 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8215584 | 0.1330767 | 0.9546351 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.861865 | 0.1395231 | 1.0013881 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6471304 | 0.0469787 | 0.6941091 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9376197 | 0.065154 | 1.0027736 |  |
| 8 | 56 | 0.0824973 | 64.5138033 | 1.0 | 0.0286555 | 1.0286555 | ✓ |
| 21 | 16 | 0.0718016 | 64.0132492 | 0.9157287 | 0.0836235 | 0.9993523 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7833118 | 0.095674 | 0.8789858 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9366811 | 0.0577686 | 0.9944497 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.8102016 | 0.0238589 | 0.8340605 |  |

### N166：selection → action 25

path=[2, 8]；visits=56；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7545。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 11 | 0.1330445 | 64.5461378 | 0.8437792 | 0.1161549 | 0.9599341 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 1.0 | 0.0206073 | 1.0206073 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.8949907 | 0.1455001 | 1.0404908 |  |
| 25 | 7 | 0.1675189 | 64.8543026 | 0.9963163 | 0.2193794 | 1.2156957 | ✓ |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.9219475 | 0.0813414 | 1.0032889 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2782789 | 0.2782789 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.2408245 | 0.0700994 | 0.310924 |  |
| 0 | 6 | 0.1446565 | 64.4965416 | 0.8192298 | 0.2165021 | 1.0357318 |  |

### N11：selection → action 27

path=[24, 3, 9]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [27]，并列按 prior 抽样。trace 行 7547。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 20 | 2 | 0.0527497 | 60.7350753 | 0.2120287 | 0.0651293 | 0.277158 |  |
| 37 | 2 | 0.0649189 | 59.5731495 | 0.0 | 0.0801544 | 0.0801544 |  |
| 27 | 1 | 0.060169 | 65.053189 | 1.0 | 0.1114345 | 1.1114345 | ✓ |

### N178：expansion → action 6

path=[24, 3, 9, 27]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 17 个代表动作。trace 行 7549。

bucket=0，compatibility_richness_prior；到达 N273（新建）；closure=[]。

## iteration 297

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[29, 22, 11, 32, 3, 14, 34, 19, 20, 26, 2, 10]

### N0：selection → action 0

path=[]；visits=296；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7568。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9173444 | 0.9173444 |  |
| 0 | 87 | 0.3308308 | 63.9665044 | 0.9547885 | 0.0905518 | 1.0453403 | ✓ |
| 2 | 201 | 0.3264006 | 64.0577843 | 1.0 | 0.0389201 | 1.0389201 |  |

### N2：selection → action 24

path=[0]；visits=87；children=10；K=10。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7570。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 28 | 5 | 0.0672856 | 62.9348103 | 0.497578 | 0.1464396 | 0.6440176 |  |
| 23 | 8 | 0.0329248 | 57.3365786 | 0.0 | 0.0477714 | 0.0477714 |  |
| 6 | 6 | 0.0683749 | 63.9944715 | 0.591762 | 0.1275518 | 0.7193138 |  |
| 19 | 6 | 0.0820736 | 61.6333545 | 0.3819029 | 0.1531064 | 0.5350093 |  |
| 38 | 22 | 0.0720943 | 64.9302919 | 0.6749389 | 0.0409318 | 0.7158707 |  |
| 26 | 6 | 0.0572402 | 63.4199909 | 0.5407015 | 0.1067802 | 0.6474817 |  |
| 20 | 15 | 0.0586424 | 64.8175027 | 0.6649141 | 0.0478607 | 0.7127748 |  |
| 24 | 13 | 0.0744855 | 68.587542 | 1.0 | 0.0694755 | 1.0694755 | ✓ |
| 8 | 11 | 0.0620846 | 64.653077 | 0.6502997 | 0.0675601 | 0.7178598 |  |
| 1 | 5 | 0.0197967 | 59.9538984 | 0.2326307 | 0.0430852 | 0.2757159 |  |

### N113：selection → action 27

path=[0, 24]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 3 次；候选 [27, 28, 36, 5]，并列按 prior 抽样。trace 行 7572。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 27 | 3 | 0.0443344 | 66.3339083 | 0.2218201 | 0.0559475 | 0.2777676 | ✓ |
| 28 | 3 | 0.0441969 | 66.7189927 | 0.2331831 | 0.055774 | 0.288957 |  |
| 36 | 3 | 0.0083742 | 58.8165668 | 0.0 | 0.0105678 | 0.0105678 |  |
| 5 | 3 | 0.0451795 | 92.7059276 | 1.0 | 0.0570139 | 1.0570139 |  |

### N114：selection → action 30

path=[0, 24, 27]；visits=3；children=2；K=2。最低访问优先：child.visits < 5；最少 1 次；候选 [30, 29]，并列按 prior 抽样。trace 行 7574。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 30 | 1 | 0.0454229 | 64.657808 | 0.0 | 0.0550723 | 0.0550723 | ✓ |
| 29 | 1 | 0.0499874 | 68.043873 | 1.0 | 0.0606065 | 1.0606065 |  |

### N118：expansion → action 13

path=[0, 24, 27, 30]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 36 个代表动作。trace 行 7576。

bucket=0，compatibility_richness_prior；到达 N274（新建）；closure=[]。

## iteration 298

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:non_coplanar；rollout：[39, 30, 36, 6, 3, 21, 5, 18, 29, 1, 12, 25, 19]

### N0：selection → action 2

path=[]；visits=297；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7594。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9188926 | 0.9188926 |  |
| 0 | 88 | 0.3308308 | 62.7922527 | 0.3731742 | 0.0896855 | 0.4628597 |  |
| 2 | 201 | 0.3264006 | 64.0577843 | 1.0 | 0.0389857 | 1.0389857 | ✓ |

### N5：selection → action 8

path=[2]；visits=201；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7596。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.6141498 | 0.162335 | 0.7764848 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0938466 | 0.0725957 | 0.1664423 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2463097 | 0.2463097 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7279064 | 0.1353904 | 0.8632968 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5999013 | 0.1975272 | 0.7974285 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.831552 | 0.133409 | 0.964961 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8723489 | 0.1398715 | 1.0122204 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6550022 | 0.047096 | 0.7020983 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9490251 | 0.0653167 | 1.0143418 |  |
| 8 | 57 | 0.0824973 | 64.4424183 | 1.0 | 0.0282317 | 1.0282317 | ✓ |
| 21 | 16 | 0.0718016 | 64.0132492 | 0.9268679 | 0.0838323 | 1.0107002 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7928403 | 0.0959128 | 0.8887531 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9480752 | 0.0579128 | 1.005988 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.8200571 | 0.0239184 | 0.8439755 |  |

### N166：selection → action 15

path=[2, 8]；visits=57；children=8；K=8。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7598。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 12 | 11 | 0.1330445 | 64.5461378 | 0.8437792 | 0.1171874 | 0.9609666 |  |
| 14 | 10 | 0.0216367 | 64.8617446 | 1.0 | 0.0207905 | 1.0207905 |  |
| 15 | 7 | 0.1111044 | 64.6495985 | 0.8949907 | 0.1467934 | 1.0417841 | ✓ |
| 25 | 8 | 0.1675189 | 63.9724145 | 0.5597944 | 0.1967373 | 0.7565317 |  |
| 33 | 14 | 0.1164611 | 64.7040582 | 0.9219475 | 0.0820644 | 1.0040119 |  |
| 32 | 6 | 0.1859329 | 62.8414838 | 0.0 | 0.2807526 | 0.2807526 |  |
| 3 | 10 | 0.0736012 | 63.3280122 | 0.2408245 | 0.0707225 | 0.3115471 |  |
| 0 | 6 | 0.1446565 | 64.4965416 | 0.8192298 | 0.2184266 | 1.0376563 |  |

### N170：selection → action 28

path=[2, 8, 15]；visits=7；children=3；K=3。最低访问优先：child.visits < 5；最少 1 次；候选 [28]，并列按 prior 抽样。trace 行 7600。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 36 | 2 | 0.0950075 | 60.7924577 | 0.0 | 0.1173042 | 0.1173042 |  |
| 28 | 1 | 0.0146656 | 66.0372356 | 1.0 | 0.0271611 | 1.0271611 | ✓ |
| 6 | 2 | 0.0790746 | 62.1800128 | 0.2645594 | 0.0976321 | 0.3621915 |  |

### N174：expansion → action 22

path=[2, 8, 15, 28]；visits=1；children=0；K=2。已有 0 条动作边 < K=2，且尚余 24 个代表动作。trace 行 7602。

bucket=0，compatibility_richness_prior；到达 N275（新建）；closure=[]。

## iteration 299

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：exact:class44；rollout：[1, 26, 4, 35, 13, 34, 15, 23, 9, 31, 10, 36, 14]

### N0：selection → action 2

path=[]；visits=298；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7621。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9204383 | 0.9204383 |  |
| 0 | 88 | 0.3308308 | 62.7922527 | 0.3846243 | 0.0898364 | 0.4744607 |  |
| 2 | 202 | 0.3264006 | 63.9976808 | 1.0 | 0.0388589 | 1.0388589 | ✓ |

### N5：selection → action 3

path=[2]；visits=202；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7623。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.6293874 | 0.1627384 | 0.7921257 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.096175 | 0.072776 | 0.1689511 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2469217 | 0.2469217 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7459664 | 0.1357268 | 0.8816932 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.6147853 | 0.198018 | 0.8128033 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8521835 | 0.1337404 | 0.985924 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.8939926 | 0.140219 | 1.0342116 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.6712534 | 0.0472131 | 0.7184665 |  |
| 3 | 12 | 0.04278 | 64.1432768 | 0.9725712 | 0.0654789 | 1.0380502 | ✓ |
| 8 | 58 | 0.0824973 | 64.3003432 | 1.0 | 0.0278222 | 1.0278222 |  |
| 21 | 16 | 0.0718016 | 64.0132492 | 0.9498643 | 0.0840406 | 1.0339049 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.8125113 | 0.0961511 | 0.9086624 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9715977 | 0.0580567 | 1.0296544 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.8404034 | 0.0239779 | 0.8643813 |  |

### N122：selection → action 18

path=[2, 3]；visits=12；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [6, 18]，并列按 prior 抽样。trace 行 7625。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 4 | 0.143538 | 63.2601553 | 0.3682814 | 0.1392245 | 0.5075059 |  |
| 18 | 4 | 0.1242323 | 56.8318776 | 0.0 | 0.120499 | 0.120499 | ✓ |
| 8 | 10 | 0.1439711 | 74.2866761 | 1.0 | 0.0634748 | 1.0634748 |  |
| 36 | 5 | 0.1853786 | 72.1266796 | 0.876252 | 0.1498397 | 1.0260917 |  |

### N124：expansion → action 6

path=[2, 3, 18]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 14 个代表动作。trace 行 7627。

bucket=0，compatibility_richness_prior；到达 N276（新建）；closure=[]。

## iteration 300

已发现集合：[1, 7, 8, 10, 11, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
终点：invalid:boundary；rollout：[36, 4, 16, 37, 27, 28, 35, 31, 14, 33, 34, 10, 19, 0, 1]

### N0：selection → action 2

path=[]；visits=299；children=3；K=18。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7646。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 24 | 8 | 0.3427686 | 62.0388318 | 0.0 | 0.9219814 | 0.9219814 |  |
| 0 | 88 | 0.3308308 | 62.7922527 | 0.3791338 | 0.089987 | 0.4691207 |  |
| 2 | 203 | 0.3264006 | 64.0260484 | 1.0 | 0.0387333 | 1.0387333 | ✓ |

### N5：selection → action 3

path=[2]；visits=203；children=14；K=15。已有子节点均达到 5 次访问，选择 normalized Q + exploration 最大者。trace 行 7648。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 9 | 8 | 0.0736086 | 62.1780906 | 0.5925625 | 0.1631407 | 0.7557032 |  |
| 37 | 16 | 0.0621776 | 59.1247377 | 0.0905479 | 0.072956 | 0.1635039 |  |
| 0 | 6 | 0.0868667 | 58.5740071 | 0.0 | 0.2475321 | 0.2475321 |  |
| 6 | 11 | 0.0818546 | 62.8456614 | 0.7023206 | 0.1360623 | 0.838383 |  |
| 28 | 7 | 0.0796143 | 62.0944746 | 0.5788149 | 0.1985075 | 0.7773224 |  |
| 25 | 14 | 0.1008208 | 63.4538963 | 0.8023231 | 0.1340711 | 0.9363942 |  |
| 36 | 16 | 0.1197987 | 63.6933094 | 0.841686 | 0.1405656 | 0.9822516 |  |
| 24 | 10 | 0.0261006 | 62.4178296 | 0.631979 | 0.0473298 | 0.6793088 |  |
| 3 | 13 | 0.04278 | 64.6562067 | 1.0 | 0.0609522 | 1.0609522 | ✓ |
| 8 | 58 | 0.0824973 | 64.3003432 | 0.941491 | 0.027891 | 0.969382 |  |
| 21 | 16 | 0.0718016 | 64.0132492 | 0.8942887 | 0.0842484 | 0.978537 |  |
| 18 | 15 | 0.0773162 | 63.2267198 | 0.7649721 | 0.0963888 | 0.8613609 |  |
| 29 | 29 | 0.0875326 | 64.1377021 | 0.9147505 | 0.0582002 | 0.9729507 |  |
| 20 | 5 | 0.0072303 | 63.3864395 | 0.7912322 | 0.0240371 | 0.8152694 |  |

### N122：selection → action 6

path=[2, 3]；visits=13；children=4；K=4。最低访问优先：child.visits < 5；最少 4 次；候选 [6]，并列按 prior 抽样。trace 行 7650。

| action | child visits | prior | raw Q | normalized Q | exploration | UCB | chosen |
|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 4 | 0.143538 | 63.2601553 | 0.1826436 | 0.1449094 | 0.327553 | ✓ |
| 18 | 5 | 0.1242323 | 60.7962078 | 0.0 | 0.1045161 | 0.1045161 |  |
| 8 | 10 | 0.1439711 | 74.2866761 | 1.0 | 0.0660667 | 1.0660667 |  |
| 36 | 5 | 0.1853786 | 72.1266796 | 0.8398872 | 0.1559581 | 0.9958453 |  |

### N123：expansion → action 38

path=[2, 3, 6]；visits=4；children=2；K=3。已有 2 条动作边 < K=3，且尚余 20 个代表动作。trace 行 7652。

bucket=0，compatibility_richness_prior；到达 N277（新建）；closure=[]。
