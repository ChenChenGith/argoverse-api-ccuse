# Argoverse API - ChenChen customed

Customed code for my study use
> Original reository see [Argoverse api](https://github.com/argoai/argoverse-api)

## Changes
### 1) Find surrounding centerline according to vehicle heading angle
> 20210412

Find surrounding centerline in the box whose length is in parallel with the vehicle heading.

The range can be customized as follow:

![](images/find_centerline.png)

Codes can be finded [Here](chenchencode/find_centerline_veh_coor.py)

**Use:**
``` python
from find_centerline_veh_coor import find_centerline_veh_coor

f = find_centerline_veh_coor(x0, y0, theta, city, range_dis_front, range_dis_back, range_dis_side)
surr_centerline = f.find()  # np.array(n,m,3)
```