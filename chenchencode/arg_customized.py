from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry import Polygon
from shapely.geometry import LineString
from argoverse.utils.se2 import SE2
import numpy as np
import copy
from pandas import DataFrame
import pickle
import pandas as pd


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
        polygon = Polygon(np.array(tmpp))

        self.polygon = polygon
        self.range_box = tmpp

    def rotation_mat(self):
        '''calculate the rotation matrix'''
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        r_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return r_mat

    def rebuild_centerline(self):
        '''rebuild centerline and save data'''
        save_path_shapely = self.save_path + self.city + '_shapely'
        save_path_array = self.save_path + self.city + '_array'
        try:
            f = open(save_path_shapely, 'rb')
            line_set_shapely = pickle.load(f)
            f.close()

            f = open(save_path_array, 'rb')
            line_set_array = pickle.load(f)
            f.close()
        except:
            am = ArgoverseMap()
            line_set_shapely = {}
            line_set_array = {}
            centerline_ind = am.build_centerline_index()[self.city]  # 生成全部centerline
            for ctlID, ctlinfo in centerline_ind.items():
                line_set_shapely[ctlID] = LineString(ctlinfo.centerline)
                line_set_array[ctlID] = ctlinfo.centerline
            # 保存
            f = open(save_path_shapely, 'wb')
            pickle.dump(line_set_shapely, f)
            f.close()

            f = open(save_path_array, 'wb')
            pickle.dump(line_set_array, f)
            f.close()

        self.line_set_shapely = line_set_shapely
        self.line_set_array = line_set_array

    def find(self):
        est_id = []
        for id, info in self.line_set_shapely.items():
            if self.polygon.intersects(info):
                est_id.append(id)

        re_cl = [self.line_set_array[lane_id] for lane_id in est_id]

        return re_cl, self.range_box

class data_loader_customized(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.seq_df = pd.read_csv(file_path)


    def get_ov_traj(self, track_ID) -> np.array:
        """Get the trajectory for the track_ID in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the vehicle trajectory
        """
        agent_x = self.seq_df[self.seq_df["TRACK_ID"] == track_ID]["X"]
        agent_y = self.seq_df[self.seq_df["TRACK_ID"] == track_ID]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        return agent_traj

    def get_all_traj_for_train(self, know_num=20, agent_first=True) -> (pd.DataFrame, pd.DataFrame):
        """Get the first (know_num, 2) coordinates of all track_ID in the current sequence for the use of trajectory prediction
        Data of the target track are placed at the first
        Args:
            know_num: traj data that are known for the prediction
            agent_first: chose agent as the prediction target, else AV
        Returns:
            numpy array of shape (track_num, know_num, 2) for the vehicle trajectory
        """
        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        seq_df['TIMESTAMP'] -= seq_df['TIMESTAMP'][0]  # time normalization
        seq_df['TIMESTAMP'] = seq_df['TIMESTAMP'].round(1)
        seq_df.sort_values('TIMESTAMP')
        know_data = seq_df[seq_df['TIMESTAMP'] < know_num / 10]  # the known data for all tracks
        know_data = know_data.sort_values(['OBJECT_TYPE', 'TRACK_ID'])  # sort by type and id, for the factorize
        know_data['TRACK_ID'] = pd.factorize(know_data['TRACK_ID'])[0]
        know_data = know_data[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]  # reserve useful data
        if not agent_first:  # exchange index of agent and AV
            know_data['TRACK_ID'][:know_num] = 1
            know_data['TRACK_ID'][know_num:know_num * 2] = 0

            pre_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AGENT')] \
                [['TIMESTAMP','X', 'Y']]
        else:
            pre_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AV')] \
                [['X', 'Y']]

        return (know_data, pre_data)


if __name__ == '__main__':
    theta = np.pi * 0.75
    city = 'MIA'
    x0, y0 = 165, 1647
    # city = 'PIT'
    # x0, y0 = 2870, 1530
    range_dis_front = 50
    range_dis_back = 3
    range_dis_side = 5

    find = find_centerline_veh_coor(x0, y0, theta, city, range_dis_front, range_dis_back, range_dis_side)
    re_cl = find.find()
    print(re_cl[0])