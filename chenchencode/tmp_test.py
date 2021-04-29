from chenchencode.arg_customized import find_centerline_veh_coor
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

path = r'e:\argoverse-api-ccuse\forecasting_sample\data\15.csv'
data = pd.read_csv(path)
data = data.sort_values(by=['OBJECT_TYPE', 'TRACK_ID'])
data['TIMESTAMP'] -= data['TIMESTAMP'][0]
data['TIMESTAMP'] = data['TIMESTAMP'].round(1)
data.index = range(data.shape[0])
data.reset_index(drop=True)

x0, y0, x1, y1 = data['X'][0], data['Y'][0], data['X'][1], data['Y'][1]
print(x0, y0, x1, y1, np.arctan2(y1 - y0, x1 - x0) / np.pi * 180)
f = find_centerline_veh_coor(x0, y0, np.arctan2(y1 - y0, x1 - x0), 'MIA', 100, 100, 30)
re_cl, range_box = f.find()
# print(DataFrame(re_cl[0]))

plt.figure(figsize=(10, 10))
group = data.groupby('TRACK_ID')
for gname, gdata in group:
    c = 'r' if gdata['OBJECT_TYPE'].iloc[0] == 'AGENT' else 'b'
    #     if i == 67: print(gname)
    #     plt.plot(gdata['X'].rolling(5).mean(),gdata['Y'].rolling(5).mean(), c=c)
    plt.plot(gdata['X'], gdata['Y'], c=c)
    plt.scatter(gdata['X'].iloc[0:2], gdata['Y'].iloc[0:2], c=c)
    if gdata['OBJECT_TYPE'].iloc[0] == 'AGENT':
        print(gdata['X'].iloc[0:2], gdata['Y'].iloc[0:2])
        break
    plt.text(gdata['X'].iloc[0], gdata['Y'].iloc[0], gname[-5:])

for i in range(len(re_cl)):
    x = re_cl[i]
    # display(Polygon(x))
    re_cl_df = DataFrame(x)
    plt.plot(re_cl_df[0], re_cl_df[1], linestyle='--', c='gray')

plt.plot(range_box[0], range_box[1])
plt.axis('equal')

plt.show()
