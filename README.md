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
# Object Args:
#     x0: float, vehicle coordinate
#     y0: float, vehicle coordinate
#     theta: float, vehicle heading angle, in radian
#     city: str, city name, 'PIT' or 'MIA'
#     range_front: float, search range in front of the vehicle
#     range_back: float, search range in the back of the vehicle
#     range_sta: if true, means that the vehicle is amost stationary, all the range value will be set to 20
# Method find() Args:
#     output_type: which type of output will be: ['list', 'df', 'tensor']
# Returns:
#     surr_centerline: np.array(m,m,3), the surrounding centerline coordinates
#     range_box: np.array(4,3), the range box coordinates
find = find_centerline_veh_coor(x0, y0, theta, city, range_front=80, range_back=20, range_side=30, range_sta=False)
surr_centerline, range_box = find.find(output_type='list') -> np.array(n,m,3), np.array(4,2)
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

> creat  20210423   
> update 20210726  

**Use:**

``` python
from chenchencode.modules.arg_customized import data_loader_customized
fdlc = data_loader_customized(dir_data_path,
                              know_num=20,
                              agent_first=True,
                              normalization=False,
                              norm_range_time=5,
                              norm_range=100,
                              range_const=False,
                              range_dis_list='default',
                              return_type='df',
                              include_centerline=False,
                              rotation_to_standard=False,
                              save_preprocessed_data=False,
                              fast_read_check=True)

# Args:
#    dir_data_path: str, dir path that contains the data files (.csv), example:'../forecasting_sample/data'
#    know_num: int (default=20), traj data that are known for the prediction
#    agent_first: bool (default=True), chose agent as the prediction target, else AV
#    normalization: bool (default=False), if true, the raw coordinates will be normalized to nearly [0,1] using the norm_range
#    norm_range_time: float (default=5), used for the normalization on TIMESTAMP
#    norm_range: float (default=100), used for the normalization on [TRACKID,X,Y]. points whose distance between the first point of agent/AV is equal to norm_range, then it will map to 1
#    range_const: bool (default=False), if true, only the coordinates in the range_dis_list are extracted, and range_dis_list is needed
#    range_dis_list: list (default='default', means [range_front, range_back, range_side]=[80, 20, 30], as in 'find_centerline_veh_coor'), the range used in finding surrounding centerline
#    return_type: str, to chose the outputs' format, should be one of [dataframe, array, tensor, list[tensor]]
#    include_centerline: bool (default=False), if true, the center line will be found.
#    rotation_to_standard: bool (default=False), if true, all the data (traj and centerline) will be rotated to make the ego vehicle drive from south to north
#    save_preprocessed_data: bool (default=False), if true, the pre-processed data will be saved to '../forecasting_sample/preprocess_data' folder
#    fast_read_check: bool (default=True), if true, the function will check if there are preprocessed data
```

#### 2.1 Get other vehicle data from forecasting data

*A new module in 'chenchencode/modules/arg_customized.py'* ([Codes](chenchencode/modules/arg_customized.py))

> creat  20210423 

The original API only provides function for agent trajectory finding, but not other vehicles.

**Use:**

``` python
c.get_ov_traj(file_name, track_id)

# Args:
#   file_name: *.csv file name
#   track_id: other vehicle's id
# Returns:
#   numpy array of shape (seq_len x 2) for the vehicle trajectory
```

#### 2.2 Get training data from a sequence (a *.csv file)

> creat 20210425  
> update 20210726

A function that can extract data for a training algorithm.

**Use:**

``` python
(train_data, centerline_data, label_data) = fdlc.get_all_traj_for_train(r'4791.csv')

# Get the first (know_num, 2) coordinates of all track_ID in the current sequence for the use of trajectory prediction
# Data of the target track are placed at the first
# Args:
#    file_name: .csv file name
# Returns:
#    train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y']), n*2
#    centerline_data: pd.DataFrame(columns = ['TRACK_ID', 'X', 'Y']),
#    label_data: pd.DataFrame(columns = ['TIMESTAMP', 'X', 'Y']), (50-know_num)*2 ,order is in scending time
#    all of the output are interpolated
```

#### 2.3 Get the trajectory direction (for the surrounding centerline finding)

> 20210507

Due to the data noise, or the initial fluctuation, vehicle angle using the first two point may not match the whole
trajectory.

**Use:**

``` python
x0, y0, angle, city, vehicle_stabale = self.get_main_dirction(use_point=10)

# Args:
#     use_point: the i-th point that used for the angle calculation
#     agent_first: if true, data of agent vehicle will be used: should be the same as that in get_all_traj_for_train
# Returns:
#     x0, y0: the initial coordinate, for the surrounding centerline extraction
#     angle: the over all angle, for the surrounding centerline extraction
#     city: city name
#     square_range: When vehicle is amost stationary, it's True, so suggest range_sta=True in finding surrounding centerline
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
#     pred_data: tensor(n,2), predicted trajectory coordinates from your algorithm
#     label: tensor(m,2), label trajectory coordinates that may contains NaN
# Returns:
#     treated_label: tensor(m,2), label trajectory without NaN
treated_label = torch_treat().label_tensor_treat(pred_data,label_data) -> label data (tensor)
```

**Output:**

example:

<img src="images/tensor_treat.png" width="500">
