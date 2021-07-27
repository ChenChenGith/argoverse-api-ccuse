# 新建：7月22日
# 辅助性函数库
# 7月22日: 增加保存模型的函数

import os
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
        pass

    def save_test(self):
        x = pd.DataFrame()
        save_file = r'x.csv'
        x.to_csv(os.path.join(self.save_dir, save_file))

def position_encoding(output_size, x, max_len=50):
    pe = torch.zeros(max_len, output_size)
    position = torch.arange(0, max_len).unsqueeze(1)
    print(torch.arange(0, output_size, 2))
    div_term = torch.exp(torch.arange(0, output_size, 2) *- (math.log(10000.0) / output_size))
    pe[:,0::2] = torch.sin(position*div_term)
    pe[:,1::2] = torch.cos(position*div_term)
    pe = pe.unsqueeze(0)
    return pe[:,:x.size(1)]

if __name__ == '__main__':
    r = Recorder(method_version='Method_test')
    print(r.save_dir)
    r.save_test()

