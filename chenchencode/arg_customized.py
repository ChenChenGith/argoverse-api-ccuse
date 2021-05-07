from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry import Polygon
from shapely.geometry import LineString
from argoverse.utils.se2 import SE2
import numpy as np
import copy
from pandas import DataFrame
import pickle
import pandas as pd
import time
import torch


class find_centerline_veh_coor(object):
    def __init__(self, x0, y0, theta, city, range_front=80, range_back=20, range_side=30, save_path=''):
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

    def get_ov_traj(self, track_ID=None) -> np.array:
        """Get the trajectory for the track_ID in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the vehicle trajectory
        """
        if track_ID == None: track_ID = self.seq_df['TRACK_ID'][0]
        agent_x = self.seq_df[self.seq_df["TRACK_ID"] == track_ID]["X"]
        agent_y = self.seq_df[self.seq_df["TRACK_ID"] == track_ID]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        return agent_traj

    def get_all_traj_for_train(self,
                               know_num=20,
                               agent_first=True,
                               forGCN=False,
                               relative=False,
                               normalization=False,
                               norm_range=100) -> (pd.DataFrame, pd.DataFrame):
        """Get the first (know_num, 2) coordinates of all track_ID in the current sequence for the use of trajectory prediction
        Data of the target track are placed at the first
        Args:
            know_num: int, traj data that are known for the prediction
            agent_first: bool, chose agent as the prediction target, else AV
            forGCN: if true then the outputted know_data will have standard format for each trackID
            relative: if true, the coordinates in both train and label data will be mapped to relative values with the start point of agent/AV
            normalization: if true, the raw coordinates will be normalized to nearly [0,1] using the norm_range
                           note: when normalization=True, relative will be assign to be True.
            norm_range: used for the normalization. points whose distance between the first point of agent/AV is equal to norm_range, then it will map to 1
        Returns:
            train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y']), n*2
            label_data: pd.DataFrame(columns = ['TIMESTAMP', 'X', 'Y']), (50-know_num)*2 ,order is in scending time
        """
        if normalization: relative=True

        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        seq_df['TIMESTAMP'] -= seq_df['TIMESTAMP'][0]  # time normalization
        seq_df['TIMESTAMP'] = seq_df['TIMESTAMP'].round(1)
        # sometimes, the sample frequency > 0.1s, thus need delete the extra data
        seq_df.drop(seq_df[seq_df['TIMESTAMP'] > 5].index, inplace=True)
        seq_df.sort_values('TIMESTAMP')

        know_data = seq_df[seq_df['TIMESTAMP'] < know_num / 10]  # the known data for all tracks
        know_data = know_data.sort_values(['OBJECT_TYPE', 'TRACK_ID'])  # sort by type and id, for the factorize
        know_data['TRACK_ID'] = pd.factorize(know_data['TRACK_ID'])[0]
        know_data = know_data[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]  # reserve useful data

        if forGCN:
            num_track = len(know_data['TRACK_ID'].unique())
            know_data['tmp'] = know_data['TIMESTAMP'] + know_data['TRACK_ID'] * 10
            standard_df = DataFrame(np.tile(np.linspace(0, (know_num - 1) / 10, know_num), num_track),
                                    columns=['TIMESTAMP_s']).round(1)
            standard_df['track_tmp'] = np.arange(num_track).repeat(know_num)
            standard_df['TIMESTAMP_s'] = standard_df['track_tmp'] * 10 + standard_df['TIMESTAMP_s']
            standard_know_data = pd.merge(standard_df, know_data, left_on='TIMESTAMP_s', right_on='tmp',
                                          how='outer')
            standard_know_data[['TIMESTAMP', 'TRACK_ID']] = standard_know_data[['TIMESTAMP_s', 'track_tmp']]
            know_data = standard_know_data.groupby('TIMESTAMP_s').mean()
            know_data = know_data[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]
            know_data['TIMESTAMP'] = know_data['TIMESTAMP'] - know_data['TRACK_ID'] * 10

        if not agent_first:  # exchange index of agent and AV
            know_data['TRACK_ID'].iloc[:know_num] = 1
            know_data['TRACK_ID'].iloc[know_num:know_num * 2] = 0

            label_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AV')] \
                [['TIMESTAMP', 'X', 'Y']]

            if relative:  # map the original data to relative values
                x0, y0 = know_data['X'].iloc[know_num], know_data['Y'].iloc[know_num]
                know_data['X'] -= x0
                know_data['Y'] -= y0
                label_data['X'] -= x0
                label_data['Y'] -= y0

        else:
            label_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AGENT')] \
                [['TIMESTAMP', 'X', 'Y']]

            if relative:  # map the original data to relative values
                x0, y0 = know_data['X'].iloc[0], know_data['Y'].iloc[0]
                know_data['X'] -= x0
                know_data['Y'] -= y0
                label_data['X'] -= x0
                label_data['Y'] -= y0

        # there may be same timestamp between two rows, or missing of some timestep, so fill it
        standard_df = DataFrame(np.linspace(know_num / 10, 4.9, 50 - know_num), columns=['TIMESTAMP']).round(1)
        standard_label_data = pd.merge(standard_df, label_data, left_on='TIMESTAMP', right_on='TIMESTAMP', how='outer')
        standard_label_data['TIMESTAMP_1'] = standard_label_data['TIMESTAMP']
        standard_label_data = standard_label_data.groupby('TIMESTAMP_1').mean()

        if normalization:  # normalizing the raw coordinates
            know_data[['X', 'Y']] = know_data[['X', 'Y']] / norm_range
            standard_label_data[['X', 'Y']] = standard_label_data[['X', 'Y']] / norm_range

        return (know_data, standard_label_data)

    def get_main_dirction(self, use_point=4, agent_first=True):
        """
        using the first point ant the use_point'th point coordinates to calculate the angle
        """
        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        veh_type = 'AGENT' if agent_first else 'AV'
        seq_df = seq_df[seq_df['OBJECT_TYPE'] == veh_type].sort_values('TIMESTAMP').reset_index(drop=True)
        x0, y0, x1, y1 = seq_df['X'][0], seq_df['Y'][0], seq_df['X'][use_point], seq_df['Y'][use_point]

        return np.arctan2(y1 - y0, x1 - x0) / np.pi * 180


class torch_treat(object):
    def __init__(self):
        pass

    def label_tensor_treat(self, pred_data, label) -> torch.tensor:
        '''
        used for the adjustment of label data, when label data have nan
        Args:
            pred_data: tensor(n,2), predicted trajectory coordinates from you algorithm
            label: tensor(m,2), label trajectory coordinates that may contains NaN
        Returns:
            treated_label: tensor(m,2), label trajectory without NaN
        '''
        treated_label = torch.where(torch.isnan(label), pred_data, label)
        return treated_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # find local centerline test

    # theta = np.pi * 0.75
    # city = 'MIA'
    # x0, y0 = 165, 1647
    # # city = 'PIT'
    # # x0, y0 = 2870, 1530
    # range_dis_front = 50
    # range_dis_back = 3
    # range_dis_side = 5
    #
    # find = find_centerline_veh_coor(x0, y0, theta, city, range_dis_front, range_dis_back, range_dis_side)
    # re_cl = find.find()
    # print(re_cl[0])

    # pd.set_option('max_colwidth', 200)

    # data loader test
    pd.set_option('max_rows', 300)
    file_path = r'e:\argoverse-api-ccuse\forecasting_sample\data\12.csv'
    fdlc = data_loader_customized(file_path)
    kd, pda = fdlc.get_all_traj_for_train(forGCN=False, normalization=False)
    g = kd.groupby('TRACK_ID')
    for name, data in g:
        c = 'gray' if name!=0 else 'red'
        plt.plot(data['X'],data['Y'],c=c)
    plt.scatter(pda['X'],pda['Y'],c='blue',s=10)
    dataq = pd.read_csv(file_path)
    g = dataq.groupby('TRACK_ID')
    for name, data in g:
        if data['OBJECT_TYPE'].iloc[0]=='AGENT': continue
        plt.scatter(data['X'].iloc[0]+0.5, data['Y'].iloc[0]+0.5,marker='x')
        plt.plot(data['X']+0.5, data['Y']+0.5,linestyle=':')
    dataq = dataq[dataq['OBJECT_TYPE']=='AGENT']
    plt.scatter(dataq['X'], dataq['Y'], marker='o',c='',edgecolors='g',s=30)
    plt.show()
