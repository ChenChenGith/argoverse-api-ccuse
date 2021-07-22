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
print(py_path)


class find_centerline_veh_coor(object):
    def __init__(self, x0, y0, theta, city, range_dis_list=[80, 20, 30], range_sta=False,
                 save_path=py_path):
        '''
        Args:
            x0: float, vehicle coordinate
            y0: float, vehicle coordinate
            theta: float, vehicle heading angle, in radian
            city: str, city name, 'PIT' or 'MIA'
            range_dis_list: list, ranges for finding centerline, [range_front, range_back, range_side]
            range_sta: if true, means that the vehicle is amost stationary, all the range value will be set to 20
            save_path: str, path to save the rebuilt centerline in Shaple format
        Returns:  Object.find()
            surr_centerline: np.array(m,m,3), the surrounding centerline coordinates
            range_box: np.array(4,3), the range box coordinates
        '''
        self.x0, self.y0, self.theta = x0, y0, theta
        self.center_point = Point([x0, y0])
        if not range_sta:
            self.range_front, self.range_back, self.range_side = range_dis_list
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

        return self.range_box

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
        Returns:  Object.find()
            surr_centerline: np.array(m,m,3), the surrounding centerline coordinates
            range_box: np.array(4,3), the range box coordinates
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
                               normalization=False,
                               norm_range_time=5,
                               norm_range=100,
                               range_const=False,
                               range_dis_list='default',
                               return_type='df',
                               include_centerline=False) -> (pd.DataFrame, pd.DataFrame):
        """
        Get the first (know_num, 2) coordinates of all track_ID in the current sequence for the use of trajectory prediction
        Data of the target track are placed at the first
        Args:
            know_num: int, traj data that are known for the prediction
            agent_first: bool, chose agent as the prediction target, else AV
            normalization: if true, the raw coordinates will be normalized to nearly [0,1] using the norm_range
            norm_range_time: used for the normalization on TIMESTAMP
            norm_range: used for the normalization on [TRACKID,X,Y]. points whose distance between the first point of agent/AV is equal to norm_range, then it will map to 1
            range_const: if true, only the coordinates in the range_dis_list are extracted
            range_dis_list: [range_front, range_back, range_side] the range used in finding surrounding centerline
            return_type: to chose the outputs' format, [dataframe, array, tensor]
            include_centerline: if true, the center line will be found.
                                note: when include_centerline=True, return_type will be assign to be tensor.
        Returns:
            train_data: pd.DataFrame(columns = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y']), n*2
            label_data: pd.DataFrame(columns = ['TIMESTAMP', 'X', 'Y']), (50-know_num)*2 ,order is in scending time
            all of the output are interpolated
        """

        self.max_time = 4.9
        self.data_point_num = 50
        self.know_num, self.norm_range_time, self.norm_range = know_num, norm_range_time, norm_range

        if range_const == True:
            if isinstance(range_dis_list, str):
                assert range_dis_list == 'default', 'range_dis_list is needed'
                range_dis_list = [80, 20, 30]
            else:
                assert isinstance(range_dis_list, list), 'range_dis_list need to be DataFrame'
                assert len(range_dis_list) == 3, 'shape of range box should be (4,2)'
        assert return_type in ['df', 'array', 'tensor',
                               'list[tensor]'], 'return type should be df, array, tensor or list[tensor]'
        obj_type = 'AGENT' if agent_first else 'AV'

        seq_df = copy.deepcopy(self.seq_df[:])  # copy seq_df
        seq_df['TIMESTAMP'] -= seq_df['TIMESTAMP'][0]  # time normalization
        seq_df['TIMESTAMP'] = seq_df['TIMESTAMP'].round(1)
        # sometimes, the sample frequency > 0.1s, thus need delete the extra data
        seq_df.drop(seq_df[seq_df['TIMESTAMP'] > self.max_time].index, inplace=True)

        if include_centerline:  # add centerline information
            x0, y0, angle, city, vehicle_stabale = self.get_main_dirction(agent_first=agent_first)
            re_cl, range_box = find_centerline_veh_coor(x0, y0, angle, city, range_dis_list=range_dis_list,
                                                        range_sta=vehicle_stabale).find(output_type='df')
            re_cl = re_cl[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]

        if range_const == True:
            seq_df['index'] = seq_df.index  # reserve index for the merge after shapely operation
            try:
                range_box_np = range_box.to_numpy()
            except:
                x0, y0, angle, city, vehicle_stabale = self.get_main_dirction(agent_first=agent_first)
                range_box = find_centerline_veh_coor(x0, y0, angle, city, range_dis_list=range_dis_list,
                                                     range_sta=vehicle_stabale).range_box
                range_box_np = range_box.to_numpy()
            tmp_inrange = np.array(
                Polygon(range_box_np).intersection(MultiPoint(seq_df[['X', 'Y', 'index']].to_numpy())))
            tmp_q = DataFrame(tmp_inrange, columns=['X', 'Y', 'index'])
            seq_df = pd.merge(seq_df, tmp_q, left_on=['X', 'Y'], right_on=['X', 'Y'], how='inner')
            # tmp_q = DataFrame(index=tmp_inrange[:, 2])
            # seq_df = pd.merge(seq_df, tmp_q, left_index=True, right_index=True, how='inner')

        seq_df.sort_values(['OBJECT_TYPE', 'TRACK_ID', 'TIMESTAMP'], inplace=True)

        standard_df = DataFrame(np.linspace(0, self.max_time, self.data_point_num), columns=['TIMESTAMP']).round(1)

        target_data = seq_df[seq_df['OBJECT_TYPE'] == obj_type]  # get data of agent or av
        target_data = pd.merge(standard_df, target_data, left_on='TIMESTAMP', right_on='TIMESTAMP', how='outer')
        target_data = target_data.interpolate().fillna(method='bfill').fillna(method='ffill')  # 插值和bfill填充

        know_data = pd.concat((target_data[target_data['TIMESTAMP'] < know_num / 10],
                               seq_df[(seq_df['TIMESTAMP'] < know_num / 10) & (seq_df['OBJECT_TYPE'] != obj_type)]))
        know_data['TRACK_ID'] = pd.factorize(know_data['TRACK_ID'])[0]
        know_data = know_data.drop_duplicates(['TIMESTAMP', 'TRACK_ID'])
        know_data = know_data[['TIMESTAMP', 'TRACK_ID', 'X', 'Y']]

        label_data = target_data[target_data['TIMESTAMP'] >= know_num / 10]
        label_data = label_data.drop_duplicates('TIMESTAMP')
        label_data = label_data[['X', 'Y']]

        self.raw_label_data = label_data  # 保存原始真值

        self.x0, self.y0 = target_data['X'].iloc[0], target_data['Y'].iloc[0]

        if normalization:  # normalizing the raw coordinates
            know_data = self.standardization(know_data)
            label_data = self.standardization(label_data)
            if include_centerline: re_cl = self.standardization(re_cl)

            # know_data['X'] -= self.x0
            # know_data['Y'] -= self.y0
            # know_data['TIMESTAMP'] -= know_num / 10 / 2
            # label_data['X'] -= self.x0
            # label_data['Y'] -= self.y0
            # if include_centerline:
            #     re_cl['X'] -= self.x0
            #     re_cl['Y'] -= self.y0
            #
            # know_data['TIMESTAMP'] = know_data['TIMESTAMP'] / norm_range_time
            # know_data[['TRACK_ID', 'X', 'Y']] = know_data[['TRACK_ID', 'X', 'Y']] / norm_range
            # # label_data['TIMESTAMP'] = label_data['TIMESTAMP'] / norm_range_time
            # label_data[['X', 'Y']] = label_data[['X', 'Y']] / norm_range
            # if include_centerline:
            #     re_cl[['X', 'Y']] = re_cl[['X', 'Y']] / norm_range

        if return_type == 'array':
            if include_centerline: know_data = pd.concat((know_data, re_cl))
            know_data = np.array(know_data)
            label_data = np.array(label_data)
        elif return_type == 'tensor':
            if include_centerline: know_data = pd.concat((know_data, re_cl))
            know_data = torch.from_numpy(know_data.values).float()
            label_data = torch.from_numpy(label_data.values).float()
        elif return_type == 'list[tensor]':
            ite = know_data['TRACK_ID'].unique()
            know_data_out = []
            for i in ite:
                know_data_out.append(
                    torch.from_numpy(know_data[know_data['TRACK_ID'] == i][['TIMESTAMP', 'X', 'Y']].values).float())

            label_data = torch.from_numpy(label_data.values).float()

            if include_centerline:
                ite = re_cl['TRACK_ID'].unique()
                center_data_out = []
                for i in ite:
                    center_data_out.append(
                        torch.from_numpy(re_cl[re_cl['TRACK_ID'] == i][['TIMESTAMP', 'X', 'Y']].values).float())

                return (know_data_out, center_data_out, label_data)
            return (know_data_out, label_data)

        return (know_data, label_data)

    def standardization(self, raw_data: DataFrame):
        '''
        Used to standard the data
        '''
        raw_data['X'] -= self.x0
        raw_data['Y'] -= self.y0
        if 'TRACK_ID' in raw_data.columns:
            raw_data[['TRACK_ID', 'X', 'Y']] = raw_data[['TRACK_ID', 'X', 'Y']] / self.norm_range
        else:
            raw_data[['X', 'Y']] = raw_data[['X', 'Y']] / self.norm_range
        if 'TIMESTAMP' in raw_data.columns:
            raw_data['TIMESTAMP'] -= self.know_num / 10 / 2
            raw_data['TIMESTAMP'] = raw_data['TIMESTAMP'] / self.norm_range_time

        return raw_data

    def de_standardization(self, raw_data: torch.tensor):
        '''
        Used to de-standard the predicted coordinates
        '''
        return (raw_data * self.norm_range) + torch.tensor([self.x0, self.y0])

    def get_absolute_error(self, pred, y):
        pred, y = pred * self.norm_range, y * self.norm_range
        error_all = (pred - y).pow(2).sum(-1).sqrt()
        # each sample
        each_error_mean = error_all.mean(1)
        each_error_at_1sec = [x[9] for x in error_all]
        each_error_at_2sec = [x[19] for x in error_all]
        each_error_at_3sec = [x[29] for x in error_all]
        # all test sample
        error_mean = error_all.mean()
        error_at_1sec = error_all.mean(0)[9]
        error_at_2sec = error_all.mean(0)[19]
        error_at_3sec = error_all.mean(0)[29]

        print('For each sample: \n ->mean_DE=%s m \n -> DE@1=%s m \n -> DE@2=%s m \n -> DE@3=%s m' % (
            each_error_mean, each_error_at_1sec, each_error_at_2sec, each_error_at_3sec))
        print('For all sample: \n ->mean_DE=%s m \n -> DE@1=%s m \n -> DE@2=%s m \n -> DE@3=%s m' % (
            error_mean, error_at_1sec, error_at_2sec, error_at_3sec))

        return ([each_error_mean, each_error_at_1sec, each_error_at_2sec, each_error_at_3sec],
                [error_mean, error_at_1sec, error_at_2sec, error_at_3sec])

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
            pred_data: tensor(n,2), predicted trajectory coordinates from your modules
            label: tensor(m,2), label trajectory coordinates that may contains NaN
        Returns:
            treated_label: tensor(m,2), label trajectory without NaN
        '''
        treated_label = torch.where(torch.isnan(label), pred_data, label)
        return treated_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # =>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>
    # # data loader test - old

    # # object establishment
    # pd.set_option('max_rows', 300)
    # file_path = r'e:\argoverse-api-ccuse\forecasting_sample\data\24.csv'
    # fdlc = data_loader_customized(file_path)

    # -----------------------------
    # # get trajectory direction
    # x0, y0, angle, city, vehicle_stabale = fdlc.get_main_dirction(agent_first=True)
    # print(vehicle_stabale)
    # re_cl, range_box = find_centerline_veh_coor(x0, y0, angle, city, range_sta=vehicle_stabale).find()
    # for i in range(len(re_cl)):
    #     x = re_cl[i]
    #     # display(Polygon(x))
    #     re_cl_df = DataFrame(x)
    #     plt.plot(re_cl_df[0], re_cl_df[1], linestyle='-.', c='lightcoral', linewidth=0.4)
    # plt.plot(range_box[0], range_box[1], c='crimson', linestyle='--')
    #
    # # -----------------------------
    # # get data
    # kd, pda = fdlc.get_all_traj_for_train(agent_first=True, forGCN=False, normalization=True, range_const=True,
    #                                       range_box=range_box, include_centerline=True)
    # g = kd.groupby('TRACK_ID')
    # for name, data in g:
    #     c = 'black' if name != 0 else 'red'
    #     plt.plot(data['X'], data['Y'], c=c, linewidth=0.8)
    #     plt.scatter(data['X'].iloc[0], data['Y'].iloc[0], marker='o', s=10, c=c)
    # pda = pda.fillna(method='ffill')
    # pda = pda.fillna(method='backfill')
    # plt.plot(pda['X'], pda['Y'], c='blue')

    ##  full data plot ------------
    # dataq = pd.read_csv(file_path)
    # g = dataq.groupby('TRACK_ID')
    # for name, data in g:
    #     if data['OBJECT_TYPE'].iloc[0] == 'AV': continue
    #     plt.scatter(data['X'].iloc[0] + 0.05, data['Y'].iloc[0] + 0.05, marker='x', s=10, c='gray')
    #     plt.plot(data['X'] + 0.5, data['Y'] + 0.5, linestyle='-', c='gray')
    # dataq = dataq[dataq['OBJECT_TYPE'] == 'AV']
    # plt.scatter(dataq['X'], dataq['Y'], marker='o', c='', edgecolors='g', s=30)

    # plt.axis('equal')
    # plt.show()

    # =>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>
    # # data loader test - new

    # object establishment
    pd.set_option('max_rows', 300)
    file_path = r'e:\argoverse-api-ccuse\forecasting_sample\data\3828.csv'
    fdlc = data_loader_customized(file_path)

    x0_out, y0_out, angle_out, city_out, vehicle_stabale_out = fdlc.get_main_dirction(agent_first=True)
    print(vehicle_stabale_out)
    re_cl, range_box_out = find_centerline_veh_coor(x0_out, y0_out, angle_out, city_out,
                                                    range_sta=vehicle_stabale_out).find(output_type='list')
    for i in range(len(re_cl)):
        x = re_cl[i]
        # display(Polygon(x))
        re_cl_df = DataFrame(x)
        plt.plot(re_cl_df[0], re_cl_df[1], linestyle='-.', c='lightcoral', linewidth=0.4)
        plt.scatter(re_cl_df[0].iloc[0] + 0.5, re_cl_df[1].iloc[0] + 0.5, marker='o', s=20, c='forestgreen')
        plt.scatter(re_cl_df[0].iloc[1] + 0.5, re_cl_df[1].iloc[1] + 0.5, marker='.', s=20, c='darkgreen')
        plt.scatter(re_cl_df[0].iloc[-1], re_cl_df[1].iloc[-1], marker='x', s=20, c='darkorange')
    plt.plot(range_box_out[0], range_box_out[1], c='crimson', linestyle='--')

    kd, pda = fdlc.get_all_traj_for_train(agent_first=True, normalization=False, range_const=True,
                                          include_centerline=False)
    g = kd.groupby('TRACK_ID')
    for name, data in g:
        c = 'black' if name != 0 else 'red'
        plt.plot(data['X'], data['Y'], c=c, linewidth=0.8)
        plt.scatter(data['X'].iloc[0], data['Y'].iloc[0], marker='o', s=10, c=c)
        # if name==0:
        #     plt.scatter(data['X'], data['Y'], c=c, linewidth=0.8)
    pda = pda.fillna(method='ffill')
    pda = pda.fillna(method='backfill')
    plt.plot(pda['X'], pda['Y'], c='blue')
    plt.axis('equal')
    plt.show()
