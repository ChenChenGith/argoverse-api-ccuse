# loader for the prediction

from chenchencode.find_centerline_veh_coor import find_centerline_veh_coor
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

# 设置路径
root_dir = '../forecasting_sample/data/'
afl = ArgoverseForecastingLoader(root_dir)  # loader对象

