import torch
import pandas as pd
import numpy as np
from pandas import DataFrame

x = DataFrame([[0.2, 0.3],
               [0.9, 0.8],
               [0.4, 0.7],
               [0.6, 0.7],
               [0.7, 0.3]])
y = DataFrame([[1.0, 0.2],
               [0.6, 0.2],
               [0.2, 0.4],
               [0.7, 0.3],
               [0.2, 0.6]])
x_ts = torch.tensor(np.array(x))
y_ts = torch.tensor(np.array(y))
z = pd.merge(x, y, left_on=0, right_on=0, how='outer')
z = z[['1_x', '1_y']]
print(x, '\n', y)
print(z)
m = torch.tensor(np.array(z))
n = torch.tensor(np.array([[0.1, 0.3],
                           [0.2, 0.8],
                           [0.9, 96],
                           [0.5, 97],
                           [0.6, 0.2],
                           [0.8, 0.1],
                           [99, 0.6]]))
print(m)
print(n)
criteria = torch.nn.MSELoss()
l = criteria(n, m)
print(l)

print(n[torch.where(torch.isnan(m))])
p = torch.where(torch.isnan(m), n, m)
print(p)
q = torch.where(torch.isnan(m), torch.full_like(m, 0), m)
print(q)


l = criteria(n, p)
print(l)
l = criteria(n, q)
print(l)



# a = torch.Tensor([[1, 2, np.nan], [3, np.nan, 4], [3, 4, 5]])
# print(a)
# b = torch.Tensor([[11, 22, 66], [33, 44, 44], [33, 44, 55]])
# print(b)
# c = torch.where(torch.isnan(a), b, a)
# print(c)

