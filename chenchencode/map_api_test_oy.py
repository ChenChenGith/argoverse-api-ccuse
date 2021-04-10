from shapely.geometry.polygon import Polygon
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from chenchencode.plt_Animation import plt_Anima


# from argoverse.map_representation.map_api import ArgoverseMap
# am = ArgoverseMap()  # map 操作对象
# traj = np.array([[3181,1671],[3178,1675],[3176,1680],[3170,1675]])
# altraj = am.get_candidate_centerlines_for_traj(traj,'PIT')
# print(len(altraj))
# print(altraj[0])
# plt_Anima(altraj[1])


x = plt_Anima([[1,2],[3,4],[7,12]])
plt.show()