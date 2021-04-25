# loader for the prediction
import sys
print(sys.path)
# sys.path.append('..')

from chenchencode.find_centerline_veh_coor import find_centerline_veh_coor
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

# 设置路径
root_dir = '../../forecasting_sample/data/'
afl = ArgoverseForecastingLoader(root_dir)  # loader对象

pivot_ID = afl[0].track_id_list[0]
print(afl[0].get_ov_traj(pivot_ID))
print(afl[0].get_all_traj_for_train())


