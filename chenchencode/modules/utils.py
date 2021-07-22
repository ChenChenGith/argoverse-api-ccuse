# 新建：7月22日
# 辅助性函数库
# 7月22日: 增加保存模型的函数

import os
import torch
import time


class Recorder(object):
    def __init__(self, path):
        self.date = time.strftime('%Y%m%d', time.localtime())
        self.save_dir = path + '_' + self.date
        self.check_dir()
        pass

    def check_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def recode_state(self, ite_num, net_parameter, optimizer_parameter, loss, loss_scheduler_parameter):
        state = {'net': net_parameter,
                 'optimizer': optimizer_parameter,
                 'loss': loss,
                 'scheduler': loss_scheduler_parameter
                 }
        save_file = r'i_' + str(ite_num) + 'net_state.pkl'
        torch.save(state, os.path.join(self.save_dir, save_file))
        pass


if __name__ == '__main__':
    print(os.path.join('rs', 'd'))
