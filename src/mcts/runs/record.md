# MCTS Runs

和 Class 7 在同一个 pattern 下 的 Class 11 和 Class 22 没有被搜索到，进一步扩大搜索范围有可能能找到这两个解。
以下记录了`interrupt_search.py` 在 $\text{iteration} = 1000$时的工作栈，栈顶元素即为当前搜索状态：

| Search Index | Start Class id | Find Class | Stack Trace | Notes | Results |
| --- | --- | --- | --- | --- | --- |
| 1 | 7 | 44 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 44 搜索 | |
| 2 | 44 | ~ | <44, 7> | Class 44 搜索完成，回溯到 Class 7 | "exact:class44": 94, "invalid:non_coplanar": 906 |
| 1 | 7 | 43 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 43 搜索 | |
| 3 | 43 | ~ | <43, 7> | Class 43 搜索完成，回溯到 Class 7 | "exact:class43": 401 |
| 1 | 7 | 29 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 29 搜索 | |
| 4 | 29 | 42 | <29, 7> | 保存 Class 29 搜索状态，下一步开始从 Class 42 搜索 | |
| 5 | 42 | 46 | <42, 29, 7> | 保存 Class 42 搜索状态，下一步开始从 Class 46 搜索 | |
| 6 | 46 | ~ | <46, 42, 29, 7> | Class 46 搜索完成，回溯到 Class 42 | "exact:class46": 401 |
| 5 | 42 | ~ | <42, 29, 7> | Class 42 搜索完成，回溯到 Class 29 | "exact:class42": 401, "exact:class46": 115 |
| 4 | 29 | 38 | <29, 7> | 保存 Class 29 搜索状态，下一步开始从 Class 38 搜索 | |
| 7 | 38 | ~ | <38, 29, 7> | Class 38 搜索完成，回溯到 Class 29 | "exact:class38": 160, "exact:class44": 105, "exact:class46": 347, "invalid:non_coplanar": 388 |
| 4 | 29 | ~ | <29, 7> | Class 29 搜索完成，回溯到 Class 7 | "exact:class28": 33, "exact:class29": 372, "exact:class30": 11, "exact:class31": 3, "exact:class38": 8, "exact:class42": 116, "exact:class44": 227, "exact:class46": 208, "invalid:non_coplanar": 22 |
| 1 | 7 | 35 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 35 搜索 | |
| 8 | 35 | ~ | <35, 7> | Class 35 搜索完成，回溯到 Class 7 | "exact:class35": 401, "exact:class45": 98, "exact:class46": 51 |
| 1 | 7 | 25 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 25 搜索 | |
| 9 | 25 | 40 | <25, 7> | 保存 Class 25 搜索状态，下一步开始从 Class 40 搜索 | |
| 10 | 40 | ~ | <40, 25, 7> | Class 40 搜索完成，回溯到 Class 25 | "exact:class40": 401 |
| 9 | 25 | ~ | <25, 7> | Class 25 搜索完成，回溯到 Class 7 | "exact:class25": 401, "exact:class40": 132, "invalid:boundary": 398, "invalid:non_coplanar": 45 |
| 1 | 7 | 39 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 39 搜索 | |
| 11 | 39 | ~ | <39, 7> | Class 39 搜索完成，回溯到 Class 7 | "exact:class39": 401 |
| 1 | 7 | 34 | <7> | 保存 Class 7 搜索状态，下一步开始从 Class 34 搜索 | |
| 12 | 34 | ~ | <34, 7> | Class 34 搜索完成，回溯到 Class 7 | "exact:class34": 401, "invalid:boundary": 189, "invalid:non_coplanar": 24 |
| 1 | 7 | ~ | <7> | Class 7 搜索完成 | "exact:class10": 1, "exact:class12": 1, "exact:class15": 7, "exact:class19": 4, "exact:class20": 19, "exact:class25": 4, "exact:class28": 12, "exact:class29": 155, "exact:class30": 1, "exact:class31": 1, "exact:class32": 5, "exact:class34": 1, "exact:class35": 103, "exact:class39": 2, "exact:class40": 1, "exact:class42": 42, "exact:class43": 255, "exact:class44": 342, "exact:class45": 9, "exact:class46": 17, "exact:class7": 13, "exact:class8": 2, "exact:class9": 1 |