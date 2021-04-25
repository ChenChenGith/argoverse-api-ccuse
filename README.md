# Argoverse API - ChenChen customed

Customed code for my study use
> Original reository see [Argoverse api](https://github.com/argoai/argoverse-api)

## Changes
### 1. Maps
#### 1.1 Find surrounding centerline according to vehicle heading angle
> 20210412

Find surrounding centerline in the box whose length is in parallel with the vehicle heading.

The range can be customized as follow:

![](images/find_centerline.png)

Codes can be found [Here](chenchencode/find_centerline_veh_coor.py)

**Use:**
``` python
from find_centerline_veh_coor import find_centerline_veh_coor

f = find_centerline_veh_coor(x0, y0, theta, city, range_dis_front, range_dis_back, range_dis_side)
surr_centerline = f.find()  # -> np.array(n,m,3)
```

### 2. Data loader
#### 2.1 Get other vehicle data from forecasting data
> 20210423

The original API only provides function for agent trajectory finding, but not other vehicles.

Add a function in [ArgoverseForecastingLoader.py](argoverse/data_loading/argoverse_forecasting_loader.py)

**Use:**
``` python
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

afl = ArgoverseForecastingLoader(root_dir)
afl[i].get_ov_traj(track_id)  # i: squence (*.csv file) int ID    -> np.array(n,2)
```
#### 2.2 Get training data from a sequence (a *.csv file)
> 20210425

A function that can extract data for a training algorithm.

Add a function in [ArgoverseForecastingLoader.py](argoverse/data_loading/argoverse_forecasting_loader.py)

**Use:**
``` python
'''
Args:
    know_num: how many data is know for prediction, unit: ms
    agent_first: True if the agent vehicle' trajectory is the prediction target, else using AV
Returns:
    train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y', 'CITY_NAME'])
    pred_data: pd.DataFrame(columns = ['TIMESTAMP','X', 'Y'])
'''
afl = ArgoverseForecastingLoader(root_dir)
train_data, pred_truth = afl[0].get_all_traj_for_train(know_num=20, agent_first=True)
```
