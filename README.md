# Argoverse API - ChenChen customed

Customed code for my study use
> Original repository see [Argoverse api](https://github.com/argoai/argoverse-api)

## Changes for trajectory prediction

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
# Args:
#     x0: float, vehicle coordinate
#     y0: float, vehicle coordinate
#     theta: float, vehicle heading angle, in radian
#     city: str, city name, 'PIT' or 'MIA'
#     range_front: float, search range in front of the vehicle
#     range_back: float, search range in the back of the vehicle
#     range_sta: if true, means that the vehicle is amost stationary, all the range value will be set to 20
# Returns:
#     surr_centerline: np.array(m,m,3), the surrounding centerline coordinates
#     range_box: np.array(4,3), the range box coordinates
find = find_centerline_veh_coor(x0, y0, theta, city, range_front=80, range_back=20, range_side=30, range_sta=False)
surr_centerline, range_box = find.find() -> np.array(n,m,3), np.array(4,2)
```

**output:**

``` python
[[[x00, y00],            <- coordinates of centerline_0
  [x01, y01],
  [..., ...],
  [x0i, y0i]],
  
 [[x10, y10],            <- coordinates of centerline_1
  [x11, y11],
  [..., ...],
  [x1j, y1j]]
  
  [...      ]]
```

### 2. Data loader customized

#### 2.1 Get other vehicle data from forecasting data

*A new module in '../chenchencode/arg_customized'* ([Codes](chenchencode/arg_customized.py))

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

**Use:**

``` python
from chenchencode.arg_customized import data_loader_customized
# Args:
#     know_num: int, traj data that are known for the prediction
#     agent_first: bool, chose agent as the prediction target, else AV
#     forGCN: if true then the outputted know_data will have standard format for each trackID
#     relative: if true, the coordinates in both train and label data will be mapped to relative values with the start point of agent/AV
#     normalization: if true, the raw coordinates will be normalized to nearly [0,1] using the norm_range
                     note: when normalization=True, relative will be assign to be True.
#     norm_range: used for the normalization. points whose distance between the first point of agent/AV is equal to norm_range, then it will map to 1  
#     range_const: if true, only the coordinates in the range_box are extracted
#     range_box: the four point of the range, can get from function (find_centerline_veh_coor)              
# Returns:
#     train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y']), n*2
#     label_data: pd.DataFrame(columns = ['TIMESTAMP', 'X', 'Y']), (50-know_num)*2 ,order is in scending time

fdlc = data_loader_customized(file_path)
train_data, label_data = fdlc.get_all_traj_for_train(know_num=20, 
                                                     agent_first=True, 
                                                     forGCN=False, 
                                                     relative=False, 
                                                     normalization=False, 
                                                     norm_range=100
                                                     range_const=False,
                                                     range_box=None)
```

**Output:**

**1) <font color=blue>train_data:</font>**

<font color=gray>(*i=0,j=1 if agent_first=True else i=1,j=0*)</font>

If forGCN=**False**:

| index | TIMESTAMP | TRACK_ID | X    | Y    |    |  Description<br>(not in data)  |
|:----: |:----:     |:----:    |:----:|:----:|----|----|
|  1    | 0.1       |   i      |   x0 |  y0  | <- | Agent data start|
|  2    | 0.3       |   i      |   x1 |  y1  |    <-|**<font color=red>note TIMESTAMP=0.2 is missing</font>** |
|...|...|...|...|...| | |
|  n=know_num    | 1.9       |   i      |   xn |  yn  | <- | Agent data end|
|  n+1  | 0.1       |   j      | xn1  | yn1  | <- | AV data start|
|...|...|...|...|...| |
|  m=know_num*2| 1.9       |   j      |   xm |  ym  | <- | AV data end|
| m+1   | 0.1       |   2      |   xm1|  ym1 | <- | next track start |
|...|...|....|...|...| |
|...|...|p|...|...| |other track data|
|...|...|...|...|...| |
|end|...|track_num|...|...| <- |the last track end|

Example:

![](images/data_loader_customized_false.png)

* If forGCN=**True**:

| index | TIMESTAMP | TRACK_ID | X    | Y    |    |  Description<br>(not in data)  |
|:----: |:----:     |:----:    |:----:|:----:|----|----|
|  1    | 0.1       |   i      |   x0 |  y0  | <- | Agent data start|
|  2    | 0.2       |   i      |   NaN |  NaN  |  <-|**<font color=red>note data of TIMESTAMP=0.2 is added using NaN**
NaN</font>** |
|  2    | 0.3       |   i      |   x1 |  y1  |    | |
|...|...|...|...|...| | |
|  n    | 1.9       |   i      |   xn |  yn  | <- | Agent data end|
|  n+1  | 0.1       |   j      | xn1  | yn1  | <- | AV data start|
|...|...|...|...|...| |
|  m    | 1.9       |   j      |   xm |  ym  | <- | AV data end|
| m+1   | 0.1       |   2      |   xm1|  ym1 | <- | next track start |
|...|...|....|...|...| |
|...|...|p|...|...| |other track data|
|...|...|...|...|...| |
|end|...|track_num|...|...| <- |the last track end|

Example:

![](images/data_loader_customized_True.png)

**2) <font color=blue>label_data:</font>**

| index | TIMESTAMP |  X    | Y    |    |  Description<br>(not in data)  |
|:----: |:----:     |:----:|:----:|----|----|
|  1    | 2.0       |    x0 |  y0  | <- | data start|
|  2    | 2.1       |    NaN |  NaN  |  <-|<font color=red>standard format</font> |
|  2    | 2.2       |   x1 |  y1  |    | |
|...|...|...|...| | |
|  n    | 5.0       |    xn |  yn  | <- | data end, n=50-know_num|.

Example:

![](images/data_loader_customized_label.png)

#### 2.3 Get the trajectory direction (for the surrounding centerline finding)

> 20210507

Due to the data noise, or the initial fluctuation, vehicle angle using the first two point may not match the whole
trajectory.

**Use:**

``` python
from chenchencode.arg_customized import data_loader_customized
# Args:
#     use_point: the i-th point that used for the angle calculation
#     agent_first: if true, data of agent vehicle will be used: should be the same as that in get_all_traj_for_train
# Returns:
#     x0, y0: the initial coordinate, for the surrounding centerline extraction
#     angle: the over all angle, for the surrounding centerline extraction
#     city: city name
#     square_range: When vehicle is amost stationary, it's True, so suggest range_sta=True in finding surrounding centerline
fdlc = data_loader_customized(file_path)
x0, y0, angle, city, range_advice = get_main_dirction(self, use_point=10, agent_first=True) ->angle (rad in [-pi, pi])
```

#### 2.4 Making the most of data when a *.csv file contains more than 5 seconds data (TODO...)

### 3. For Tensor training

#### 3.1 Treat on label data

> 20200430

There are some missing values due to the sample frequencies are not stable. this function fill the nan data use the
predicted data, so that can be used to calculate the loss using torch.

In *'../chenchencode/arg_customized'* ([Codes](chenchencode/arg_customized.py))

**Use:**

``` python
from chenchencode.arg_customized import torch_treat
# Args:
#     pred_data: tensor(n,2), predicted trajectory coordinates from you algorithm
#     label: tensor(m,2), label trajectory coordinates that may contains NaN
# Returns:
#     treated_label: tensor(m,2), label trajectory without NaN
treated_label = torch_treat().label_tensor_treat(pred_data,label_data) -> label data (tensor)
```

**Output:**

example:

<img src="images/tensor_treat.png" width="500">

