from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry import Polygon
from shapely.geometry import LineString
from argoverse.utils.se2 import SE2
import numpy as np
import copy
from pandas import DataFrame
import pickle

# map 对象
am = ArgoverseMap()  # map 操作对象


class find_centerline_veh_coor(object):
    def __init__(self, x0, y0, theta, city, range_front, range_back, range_side, save_path=''):
        '''

        Args:
            x0: float, vehicle coordinate
            y0: float, vehicle coordinate
            theta: float, vehicle heading angle, in radian
            city: str, city name, 'PIT' or 'MIA'
            range_front: float, search range in front of the vehicle
            range_back: float, search range in the back of the vehicle
            save_path: str, path to save the rebuilt centerline in Shaple format
        '''
        self.x0, self.y0, self.theta = x0, y0, theta
        self.range_front, self.range_back, self.range_side = range_front, range_back, range_side
        self.city = city
        self.save_path = save_path

        self.rebuild_centerline()
        self.rotation_target_box()

    def rotation_target_box(self):
        '''determine the box according tot the state of vehicle'''
        x1, y1 = self.range_front, self.range_side
        x2, y2 = self.range_front, -self.range_side
        x3, y3 = -self.range_back, self.range_side
        x4, y4 = -self.range_back, -self.range_side
        rotation_matrix = self.rotation_mat()
        dss = SE2(rotation=rotation_matrix, translation=np.array([0, 0]))
        transformed_pts = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])
        pts = dss.transform_point_cloud(transformed_pts)
        tmpp = DataFrame(pts)
        tmpp[0] = tmpp[0] + self.x0
        tmpp[1] = tmpp[1] + self.y0
        polygon = Polygon(pts)

        self.polygon = polygon

    def rotation_mat(self):
        '''calculate the rotation matrix'''
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        r_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return r_mat

    def rebuild_centerline(self):
        '''rebuild centerline and save data'''
        save_path = self.save_path + self.city
        try:
            f = open(save_path, 'rb')
            line_set = pickle.load(f)
            f.close()
        except:
            line_set = {}
            centerline_ind = am.build_centerline_index()[self.city]  # 生成全部centerline
            for ctlID, ctlinfo in centerline_ind.items():
                line_set[ctlID] = LineString(ctlinfo.centerline)
            # 保存
            f = open(save_path, 'wb')
            pickle.dump(line_set, f)
            f.close()

        self.line_set = line_set

    def find(self):
        est_id = []
        for id, info in self.line_set.items():
            if self.polygon.intersects(info):
                est_id.append(id)

        re_cl = [am.get_lane_segment_centerline(lane_id, self.city) for lane_id in est_id]

        return re_cl
