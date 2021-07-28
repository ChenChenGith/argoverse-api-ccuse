# 新建：7月22日
# 辅助性函数库
# 7月22日: 增加保存模型的函数

import os

import numpy as np
import torch
import time
import pandas as pd


class Recorder(object):
    def __init__(self, method_version, path=r'../Saved_resultes/'):
        self.date = time.strftime('%Y%m%d', time.localtime())
        self.save_dir = path + self.date + '_' + method_version
        self.check_dir()

    def check_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def recode_state(self, ite_num, net_parameter, optimizer_parameter, loss, loss_all, loss_scheduler_parameter):
        state = {'net': net_parameter,
                 'optimizer': optimizer_parameter,
                 'loss': loss,
                 'loss_all': loss_all,
                 'scheduler': loss_scheduler_parameter
                 }
        save_file = r'i_' + str(ite_num) + '_full_net_state.pkl'
        torch.save(state, os.path.join(self.save_dir, save_file))

    def general_record(self, ite_num, name, info):
        save_file = r'i_' + str(ite_num) + str(name) + '.pkl'
        torch.save(info, os.path.join(self.save_dir, save_file))

    def save_test(self):
        x = pd.DataFrame()
        save_file = r'x.csv'
        x.to_csv(os.path.join(self.save_dir, save_file))


class Position_encoding(object):
    def __init__(self, output_size, max_len=100):
        assert output_size > 0 and output_size % 2 == 0, 'position encoding output size should > 0 and be a even number'
        self.pe = np.zeros((max_len, output_size))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, output_size, 2) * - (np.log(10000.0) / output_size))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def encoding(self, raw_data_length):
        return self.pe[:raw_data_length, :]


if __name__ == '__main__':
    position_ecder = Position_encoding(2)
    print(position_ecder.encoding(5))
