# loader for the prediction
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch.nn.utils.rnn as rnn_utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from chenchencode.arg_customized import find_centerline_veh_coor
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

import torch.utils.data as data_


class MyData(data_.Dataset):
    def __init__(self, data_dir_path):
        self.afl = ArgoverseForecastingLoader(data_dir_path)  # loader对象

    def __len__(self):
        return len(self.afl)

    def __getitem__(self, idx):
        train_data, pred_data = self.afl[idx].get_all_traj_for_train()
        tuple_ = (
        torch.FloatTensor(np.array(train_data)), torch.FloatTensor(np.array(pred_data.iloc[:30, :]).reshape(-1)))
        return tuple_


def collate_fn(data_tuple):  # data_tuple是一个列表，列表中包含batchsize个元组，每个元组中包含数据和标签
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)  # 这行代码只是为了把列表变为tensor
    re_data = data.unsqueeze(-1) if data.dim() < 3 else data
    return re_data, label, data_length


class RNN(nn.Module):
    def __init__(self, inputsize, hiddensize, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=inputsize,
            hidden_size=hiddensize,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(hiddensize, 60)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # out_pad, out_len = rnn_utils.pad_packed_sequence(r_out, batch_first=True)
        out = self.out(h_n[-1])
        return out

# ————————————————
# 版权声明：本文为CSDN博主「肥宅_Sean」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/a19990412/article/details/85139058


if __name__ == '__main__':

    EPOCH = 5
    inputsize = 4
    batchsize = 2
    hiddensize = 128
    num_layers = 2
    learning_rate = 0.001
    root_dir = '../../forecasting_sample/data/'

    data_ = MyData(root_dir)  # 注意这里是一个数据集对象，其中定义了__getitem__方法，调用时才是输出对应的数据
    data_loader = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
    net = RNN(inputsize, hiddensize, num_layers)
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 训练方法一
    for epoch in range(EPOCH):
        for batch_id, (batch_x, batch_y, batch_x_len) in enumerate(data_loader):
            batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_x_len, batch_first=True)
            out = net(batch_x_pack)  # out.data's shape (所有序列总长度, hiddensize)
            # out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
            loss = criteria(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))

    # # 训练方法二
    # for epoch in range(EPOCH):
    #     for batch_id, (batch_x, batch_y, batch_x_len) in enumerate(data_loader):
    #         batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_x_len, batch_first=True)
    #         batch_y_pack = rnn_utils.pack_padded_sequence(batch_y, batch_x_len, batch_first=True)
    #         out, _ = net(batch_x_pack)  # out.data's shape (所有序列总长度, hiddensize)
    #         loss = criteria(out.data, batch_y_pack.data)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))

    print('Training done!')
