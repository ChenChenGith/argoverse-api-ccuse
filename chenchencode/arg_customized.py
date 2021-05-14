from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry import Polygon, LineString, MultiPoint, Point
from argoverse.utils.se2 import SE2
import numpy as np
import copy
from pandas import DataFrame, Series
import pickle
import pandas as pd
import torch

import os

py_path = os.path.dirname(os.path.abspath(__file__))


class find_centerline_veh_coor(object):
    def __init__(self, x0, y0, theta, city, range_front=80, range_back=20, range_side=30, range_sta=False,
                 save_path=py_path):
        '''
        Args:
            x0: float, vehicle coordinate
            y0: float, vehicle coordinate
            theta: float, vehicle heading angle, in radian
            city: str, city name, 'PIT' or 'MIA'
            range_front: float, search range in front of the vehicle
            range_back: float, search range in the back of the vehicle
            range_sta: if true, means that the vehicle is amost stationary, all the range value will be set to 20
            save_path: str, path to save the rebuilt centerline in Shaple format
        Returns:  Object.find()
            surr_centerline: np.array(m,m,3), the surrounding centerline coordinates
            range_box: np.array(4,3), the range box coordinates
        '''
        self.x0, self.y0, self.theta = x0, y0, theta
        self.center_point = Point([x0, y0])
        if not range_sta:
            self.range_front, self.range_back, self.range_side = range_front, range_back, range_side
        else:
            self.range_front, self.range_back, self.range_side = 20, 20, 20
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
        save_path_shapely = self.save_path + '\\' + self.city + '_shapely'
        save_path_array = self.save_path + '\\' + self.city + '_array'
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

    def find(self, output_type='list'):
        '''
        Args:
            output_type: which type of output will be: ['list', 'df', 'tensor']
        '''
        est_id = []
        est_dis = []
        for id, info in self.line_set_shapely.items():
            if self.polygon.intersects(info):
                distance = self.center_point.distance(info)
                est_id.append(id)
                est_dis.append(distance)

        est_id_Se = Series(est_dis, index=est_id)
        est_id_Se = est_id_Se.sort_values()
        est_id_Se = Series(range(1, est_id_Se.shape[0] + 1), index=est_id_Se.index)

        i = 0
        if output_type == 'tensor':
            for lane_id in est_id_Se.index:
                cl = DataFrame(self.line_set_array[lane_id], columns=['X', 'Y'])
                cl['TIMESTAMP'] = 0
                cl['TRACK_ID'] = -1 * est_id_Se[lane_id]
                cl = cl[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]
                cl = torch.from_numpy(cl.values).float()
                if i == 0:
                    re_cl = cl
                    i += 1
                else:
                    re_cl = torch.cat((re_cl, cl))
            pass
        elif output_type == 'df':
            for lane_id in est_id_Se.index:
                cl = DataFrame(self.line_set_array[lane_id], columns=['X', 'Y'])
                cl['TIMESTAMP'] = 0
                cl['TRACK_ID'] = -1 * est_id_Se[lane_id]
                cl = cl[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]
                if i == 0:
                    re_cl = cl
                    i += 1
                else:
                    re_cl = pd.concat((re_cl, cl), axis=0)
            pass
        else:
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
                               norm_range_time=1,
                               norm_range_2=100,
                               range_const=False,
                               range_box=None,
                               return_type='df',
                               include_centerline=False) -> (pd.DataFrame, pd.DataFrame):
        """Get the first (know_num, 2) coordinates of all track_ID in the current sequence for the use of trajectory prediction
        Data of the target track are placed at the first
        Args:
            know_num: int, traj data that are known for the prediction
            agent_first: bool, chose agent as the prediction target, else AV
            forGCN: if true then the outputted know_data will have standard format for each trackID
            relative: if true, the coordinates in both train and label data will be mapped to relative values with the start point of agent/AV
            normalization: if true, the raw coordinates will be normalized to nearly [0,1] using the norm_range
                           note: when normalization=True, relative will be assign to be True.
            norm_range_time: used for the normalization on TIMESTAMP
            norm_range_2: used for the normalization on [TRACKID,X,Y]. points whose distance between the first point of agent/AV is equal to norm_range, then it will map to 1
            range_const: if true, only the coordinates in the range_box are extracted
            range_box: the four point of the range
            return_type: to chose the outputs' format, [dataframe, array, tensor]
            include_centerline: if true, the center line will be found and cat with trajectory data.
                                note: when include_centerline=True, return_type will be assign to be tensor.
        Returns:
            train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y']), n*2
            label_data: pd.DataFrame(columns = ['TIMESTAMP', 'X', 'Y']), (50-know_num)*2 ,order is in scending time
        """
        if normalization: relative = True
        if range_const == True and range_box is None:
            raise ValueError('need range_box parameters')
        assert return_type in ['df', 'array', 'tensor'], 'return type should be df or array'
        if include_centerline: return_type = 'tensor'

        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        seq_df['TIMESTAMP'] -= seq_df['TIMESTAMP'][0]  # time normalization
        seq_df['TIMESTAMP'] = seq_df['TIMESTAMP'].round(1)
        # sometimes, the sample frequency > 0.1s, thus need delete the extra data
        seq_df.drop(seq_df[seq_df['TIMESTAMP'] > 4.9].index, inplace=True)
        seq_df.sort_values('TIMESTAMP', inplace=True)

        if range_const == True:
            seq_df['index'] = seq_df.index  # reserve index for the merge after shapely operation
            tmp_inrange = np.array(
                Polygon(range_box.to_numpy()).intersection(MultiPoint(seq_df[['X', 'Y', 'index']].to_numpy())))
            tmp_q = DataFrame(index=tmp_inrange[:, 2])
            seq_df = pd.merge(seq_df, tmp_q, left_index=True, right_index=True, how='inner')
            seq_df.drop('index', inplace=True, axis=1)

        know_data = seq_df[seq_df['TIMESTAMP'] < know_num / 10]  # the known data for all tracks
        know_data = know_data.sort_values(['OBJECT_TYPE', 'TRACK_ID'])  # sort by type and id, for the factorize
        know_data['TRACK_ID'] = pd.factorize(know_data['TRACK_ID'])[0]
        know_data = know_data[['TIMESTAMP', 'TRACK_ID', 'X', 'Y', 'OBJECT_TYPE']]  # reserve useful data

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
            if 'AGENT' in know_data['OBJECT_TYPE'].unique():
                know_data['TRACK_ID'][know_data['TRACK_ID'] == 0] = None
                know_data['TRACK_ID'][know_data['TRACK_ID'] == 1] = 0
                know_data['TRACK_ID'][know_data['TRACK_ID'] == None] = 1

            label_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AV')] \
                [['TIMESTAMP', 'X', 'Y']]

            x0, y0 = know_data[know_data['TRACK_ID'] == 0]['X'].iloc[0], \
                     know_data[know_data['TRACK_ID'] == 0]['Y'].iloc[0]

        else:
            label_data = seq_df[(seq_df['TIMESTAMP'] >= know_num / 10) & (seq_df['OBJECT_TYPE'] == 'AGENT')] \
                [['TIMESTAMP', 'X', 'Y']]

            x0, y0 = know_data['X'].iloc[0], know_data['Y'].iloc[0]

        know_data.drop(columns=['OBJECT_TYPE'], inplace=True)

        if include_centerline:  # 添加周边centerline数据
            x0, y0, angle, city, vehicle_stabale = self.get_main_dirction(agent_first=True)
            re_cl, range_box = find_centerline_veh_coor(x0, y0, angle, city, range_sta=vehicle_stabale).find(
                output_type='df')
            know_data = pd.concat((know_data, re_cl))

        # there may be same timestamp between two rows, or missing of some timestep, so fill it
        standard_df = DataFrame(np.linspace(know_num / 10, 4.9, 50 - know_num), columns=['TIMESTAMP']).round(1)
        standard_label_data = pd.merge(standard_df, label_data, left_on='TIMESTAMP', right_on='TIMESTAMP', how='outer')
        standard_label_data['TIMESTAMP_1'] = standard_label_data['TIMESTAMP']
        standard_label_data = standard_label_data.groupby('TIMESTAMP_1').mean()

        if relative:  # map the original data to relative values
            know_data['X'] -= x0
            know_data['Y'] -= y0
            standard_label_data['X'] -= x0
            standard_label_data['Y'] -= y0

        if normalization:  # normalizing the raw coordinates
            # know_data['TIMESTAMP'] = know_data['TIMESTAMP'] / norm_range_time
            know_data[['TRACK_ID', 'X', 'Y']] = know_data[['TRACK_ID', 'X', 'Y']] / norm_range_2
            # standard_label_data['TIMESTAMP'] = standard_label_data['TIMESTAMP'] / norm_range_time
            standard_label_data[['X', 'Y']] = standard_label_data[['X', 'Y']] / norm_range_2

            know_data.drop(columns=['TIMESTAMP'], inplace=True)
            standard_label_data.drop(columns=['TIMESTAMP'], inplace=True)

        if return_type == 'array':
            know_data = np.array(know_data)
            standard_label_data = np.array(standard_label_data)
        elif return_type == 'tensor':
            know_data = torch.from_numpy(know_data.values).float()
            standard_label_data = torch.from_numpy(standard_label_data.values).float()

        return (know_data, standard_label_data)

    def get_main_dirction(self, use_point=10, agent_first=True):
        """
        using the first point ant the use_point'th point coordinates to calculate the angle
        Args:
            use_point: the i-th point that used for the angle calculation
            agent_first: if true, data of agent vehicle will be used: should be the same as that in get_all_traj_for_train
        Returns:
            x0, y0: the initial coordinate, for the surrounding centerline extraction
            angle: the over all angle, for the surrounding centerline extraction
            city: city name
            vehicle_stabale: When vehicle is amost stationary, it's True, so suggest range_sta=True in finding surrounding centerline
        """
        city = self.seq_df['CITY_NAME'].iloc[0]
        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        veh_type = 'AGENT' if agent_first else 'AV'
        seq_df = seq_df[seq_df['OBJECT_TYPE'] == veh_type].sort_values('TIMESTAMP').reset_index(drop=True)
        x0, y0, x1, y1 = seq_df['X'][0], seq_df['Y'][0], seq_df['X'][use_point], seq_df['Y'][use_point]

        if np.abs(x1 - x0 + y1 - y0) < 0.5:  # if the moving distance <0.5m
            vehicle_stabale = True
            angle = 0
        else:
            vehicle_stabale = False
            angle = np.arctan2(y1 - y0, x1 - x0)

        return (x0, y0, angle, city, vehicle_stabale)


class torch_treat(object):
    def __init__(self):
        pass

    def label_tensor_treat(self, pred_data, label) -> torch.tensor:
        '''
        used for the adjustment of label data, when label data have nan
        Args:
            pred_data: tensor(n,2), predicted trajectory coordinates from your algorithm
            label: tensor(m,2), label trajectory coordinates that may contains NaN
        Returns:
            treated_label: tensor(m,2), label trajectory without NaN
        '''
        treated_label = torch.where(torch.isnan(label), pred_data, label)
        return treated_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ===================================================
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
    # ===================================================
    # object establishment
    pd.set_option('max_rows', 300)
    file_path = r'e:\argoverse-api-ccuse\forecasting_sample\data\24.csv'
    fdlc = data_loader_customized(file_path)

    # ===================================================
    # get trajectory direction
    x0, y0, angle, city, vehicle_stabale = fdlc.get_main_dirction(agent_first=True)
    print(vehicle_stabale)
    re_cl, range_box = find_centerline_veh_coor(x0, y0, angle, city, range_sta=vehicle_stabale).find()
    for i in range(len(re_cl)):
        x = re_cl[i]
        # display(Polygon(x))
        re_cl_df = DataFrame(x)
        plt.plot(re_cl_df[0], re_cl_df[1], linestyle='-.', c='lightcoral', linewidth=0.4)
    plt.plot(range_box[0], range_box[1], c='crimson', linestyle='--')

    # ===================================================
    # get data
    kd, pda = fdlc.get_all_traj_for_train(agent_first=True, forGCN=False, normalization=True, range_const=True,
                                          range_box=range_box, include_centerline=True)
    g = kd.groupby('TRACK_ID')
    for name, data in g:
        c = 'black' if name != 0 else 'red'
        plt.plot(data['X'], data['Y'], c=c, linewidth=0.8)
        plt.scatter(data['X'].iloc[0], data['Y'].iloc[0], marker='o', s=10, c=c)
    pda = pda.fillna(method='ffill')
    pda = pda.fillna(method='backfill')
    plt.plot(pda['X'], pda['Y'], c='blue')

    ##  full data plot ------------
    # dataq = pd.read_csv(file_path)
    # g = dataq.groupby('TRACK_ID')
    # for name, data in g:
    #     if data['OBJECT_TYPE'].iloc[0] == 'AV': continue
    #     plt.scatter(data['X'].iloc[0] + 0.05, data['Y'].iloc[0] + 0.05, marker='x', s=10, c='gray')
    #     plt.plot(data['X'] + 0.5, data['Y'] + 0.5, linestyle='-', c='gray')
    # dataq = dataq[dataq['OBJECT_TYPE'] == 'AV']
    # plt.scatter(dataq['X'], dataq['Y'], marker='o', c='', edgecolors='g', s=30)

    plt.axis('equal')
    plt.show()
