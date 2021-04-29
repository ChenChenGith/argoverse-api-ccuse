# Argoverse API - ChenChen customed

Customed code for my study use
> Original repository see [Argoverse api](https://github.com/argoai/argoverse-api)

## Changes
### 1. Maps
#### 1.1 Find surrounding centerline according to vehicle heading angle
> 20210412

Find surrounding centerline in the box whose length is in parallel with the vehicle heading.

The range can be customized as follow:

![](images/find_centerline.png)

Codes can be found [Here](chenchencode/arg_customized.py)

**Use:**
``` python
from chenchencode.arg_customized import find_centerline_veh_coor

f = find_centerline_veh_coor(x0, y0, theta, city, range_dis_front, range_dis_back, range_dis_side)
surr_centerline = f.find()  # -> np.array(n,m,3)
```

### 2. Data loader customized
#### 2.1 Get other vehicle data from forecasting data

*A new module in '../chenchencode/arg_customized'*

> 20210423

The original API only provides function for agent trajectory finding, but not other vehicles.

Codes can be found [Here](chenchencode/arg_customized.py)

**Use:**
``` python
from chenchencode.arg_customized import data_loader_customized

fdlc = data_loader_customized(root_dir)
fdlc[i].get_ov_traj(track_id)  # i: squence (*.csv file) int ID    -> np.array(n,2)
```
#### 2.2 Get training data from a sequence (a *.csv file)
> 20210425

A function that can extract data for a training algorithm.

Codes can be found [Here](chenchencode/arg_customized.py)

**Use:**
``` python
from chenchencode.arg_customized import data_loader_customized
# Args:
#     know_num: int, how many data is know for prediction, unit: ms
#     agent_first: bool, True if the agent vehicle' trajectory is the prediction target, else using AV
# Returns:
#     train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y'])
#     pred_data: pd.DataFrame(columns = ['X', 'Y'])  # order is in scending time

fdlc = data_loader_customized(root_dir)
train_data, pred_truth = fdlc.get_all_traj_for_train(know_num=20, agent_first=True)
```
