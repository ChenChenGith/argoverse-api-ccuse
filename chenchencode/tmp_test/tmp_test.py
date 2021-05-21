import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from argoverse.map_representation.map_api import ArgoverseMap
am = ArgoverseMap()  # map 操作对象
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
# 设置路径
root_dir = '../forecasting_sample/data/'
afl = ArgoverseForecastingLoader(root_dir)  # loader对象
print('Total number of sequences:',len(afl))  # 文件夹中的seq，即数据文件的数量

# 以下是关于对labeldata进行处理的测试
# x = DataFrame([[0.2, 0.3],
#                [0.9, 0.8],
#                [0.4, 0.7],
#                [0.6, 0.7],
#                [0.7, 0.3]])
# y = DataFrame([[1.0, 0.2],
#                [0.6, 0.2],
#                [0.2, 0.4],
#                [0.7, 0.3],
#                [0.2, 0.6]])
# x_ts = torch.tensor(np.array(x))
# y_ts = torch.tensor(np.array(y))
# z = pd.merge(x, y, left_on=0, right_on=0, how='outer')
# z = z[['1_x', '1_y']]
# print(x, '\n', y)
# print(z)
# m = torch.tensor(np.array(z))
# n = torch.tensor(np.array([[0.1, 0.3],
#                            [0.2, 0.8],
#                            [0.9, 96],
#                            [0.5, 97],
#                            [0.6, 0.2],
#                            [0.8, 0.1],
#                            [99, 0.6]]))
# print(m)
# print(n)
# criteria = torch.nn.MSELoss()
# l = criteria(n, m)
# print(l)
#
# print(n[torch.where(torch.isnan(m))])
# p = torch.where(torch.isnan(m), n, m)
# print(p)
# q = torch.where(torch.isnan(m), torch.full_like(m, 0), m)
# print(q)
#
#
# l = criteria(n, p)
# print(l)
# l = criteria(n, q)
# print(l)


