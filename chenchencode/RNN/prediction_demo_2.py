# 按照新的网络结构进行了调整主要是：
# 1. 对于（每条，共p条）自车和周车轨迹：raw_data(n,2) -1x1卷积-> (n,8) -1x3卷积-> (n,10) -lstm-> (128,)
# 2. 对于（每条，共q条）车道中心线： raw_data(10,2) -1x1卷积-> (10,8) -1x3卷积-> (10,1) -fc-> (128,)
# concat(1,2) --> (m+n,128) -注意力权重-> (m+n,128) with c_0 --> LSTM -逐步预测->(x_n+1, y_n+1)
#                                                                ^
#                                           自车轨迹最后一点坐标 (x_n,y_n)


import torch
import torch.utils.data as data_
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utlils
import torch.nn.functional as F
from torch import nn
from chenchencode.arg_customized import data_loader_customized
from chenchencode.arg_customized import torch_treat
import os
import random
from torchviz import make_dot

import sys

print(sys.path)

teacher_forcing_ratio = 0.5
learning_rate = 0.0001


class Data_read(data_.Dataset):
    def __init__(self, file_path_list):
        self.file_path_list = file_path_list

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        data_reader = data_loader_customized(self.file_path_list[idx])
        raw_data = data_reader.get_all_traj_for_train(return_type='list[tensor]',
                                                      normalization=True,
                                                      range_const=True,
                                                      include_centerline=True)  # TODO:后期需要根据网络形式来更改该函数参数

        return raw_data


def co_fn(data_tuple):
    x1 = [sq[0] for sq in data_tuple]
    x2 = [sq[1] for sq in data_tuple]
    y = [sq[2] for sq in data_tuple]
    y_st = [sq[0][0][-1][1:] for sq in data_tuple]
    return x1, x2, torch.cat(y).reshape(len(y), 30, 2), torch.cat(y_st).reshape(len(y_st), 2).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch_x11=8, out_ch_x12=3, out_ch_x21=8, out_ch_x22=1, fc_in=10, out_ch_final=128):
        super(Encoder, self).__init__()
        # self.in_channel = in_ch
        # self.out_ch_x11 = out_ch_x11
        self.out_ch_x12 = out_ch_x12
        # self.out_ch_x21 = out_ch_x21
        # self.out_ch_x22 = out_ch_x22
        # self.fc_in = fc_in
        self.out_ch_final = out_ch_final

        # self.nn_x1_1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_ch_x11, kernel_size=1)
        # self.nn_x1_2 = nn.Tanh()
        # self.nn_x1_3 = nn.Conv1d(in_channels=self.out_ch_x11, out_channels=self.out_ch_x12, kernel_size=3, padding=1)
        # self.nn_x1_4 = nn.LeakyReLU()
        self.nn_x1_5 = nn.LSTM(input_size=self.out_ch_x12, hidden_size=self.out_ch_final, num_layers=1,
                               batch_first=True)

        # self.nn_x2_1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_ch_x21, kernel_size=1)
        # self.nn_x2_2 = nn.Tanh()
        # self.nn_x2_3 = nn.Conv1d(in_channels=self.out_ch_x21, out_channels=self.out_ch_x22, kernel_size=3, padding=1)
        # self.nn_x2_4 = nn.LeakyReLU()
        # self.nn_x2_5 = nn.Linear(in_features=self.fc_in, out_features=self.out_ch_final)

    def forward(self, x1, x2):
        input = x1[0][0].unsqueeze(0)
        out, (h_n, c_n) = self.nn_x1_5(input)

        return (h_n, c_n)

        # batch_num = len(x1)
        # final_out = torch.empty(batch_num, self.out_ch_final)
        # for batch_i in range(batch_num):
        #     traj_data = x1[batch_i]
        #     center_data = x2[batch_i]
        #     traj_num, center_num = len(traj_data), len(center_data)
        #     mid_out = torch.empty(traj_num + center_num, self.out_ch_final)
        #     for i in range(traj_num):
        #         input = traj_data[i].permute(1, 0).unsqueeze(0)
        #         y = self.nn_x1_1(input)
        #         y = self.nn_x1_2(y)
        #         y = self.nn_x1_3(y)
        #         y = self.nn_x1_4(y)
        #         y = y.permute(0, 2, 1)
        #         out, (h_n, c_n) = self.nn_x1_5(y)
        #         mid_out[i] = c_n.squeeze(-2)
        #         if i == 0:
        #             h_n_reserve = h_n
        #     for i in range(center_num):
        #         input = center_data[i].permute(1, 0).unsqueeze(0)
        #         y = self.nn_x2_1(input)
        #         y = self.nn_x2_2(y)
        #         y = self.nn_x2_3(y)
        #         y = self.nn_x2_4(y)
        #         y = y.squeeze(-1)
        #         out = self.nn_x2_5(y)
        #         mid_out[traj_num + i] = out.squeeze(-2)
        #
        #     attention_weight = F.softmax(mid_out, dim=0)
        #     mid_out = torch.mul(attention_weight, mid_out).sum(0)
        #
        #     final_out[batch_i] = mid_out
        #
        # final_out = final_out.unsqueeze(0)
        #
        # return final_out, h_n_reserve


class Decoder(nn.Module):
    def __init__(self, in_ch=2, fc_out=16, lstm_ch_hidden=128, out_ch_final=2):
        super(Decoder, self).__init__()
        self.in_ch = in_ch
        self.fc_out = fc_out
        self.lstm_ch_hidden = lstm_ch_hidden
        self.out_ch_final = out_ch_final

        self.line = nn.Linear(in_features=self.in_ch, out_features=self.fc_out)
        self.rnn = nn.LSTM(input_size=self.fc_out, hidden_size=self.lstm_ch_hidden, num_layers=1, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(in_features=self.lstm_ch_hidden, out_features=self.lstm_ch_hidden),
            nn.Tanh(),
            nn.Linear(in_features=self.lstm_ch_hidden, out_features=self.out_ch_final)
        )

    def forward(self, input, hidden):
        input = self.line(input)
        lstm_out, hidden_out = self.rnn(input, hidden)
        lstm_out = self.out(lstm_out)

        return lstm_out, hidden_out


class Seq2Seq(nn.Module):
    def __init__(self, batch_size, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size

    def forward(self, x1, x2, y, y_st):
        encoder_out, h_n = self.encoder(x1, x2)
        decoder_input = y_st
        decoder_rec = torch.empty(self.batch_size, 30, 2)
        decoder_hidden = (h_n, encoder_out)
        for i in range(30):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if True:  # random.random() < teacher_forcing_ratio:
                decoder_input = decoder_out
            else:
                decoder_input = y[:, i, :].unsqueeze(-2)
            for j in range(self.batch_size):
                decoder_rec[j][i] = decoder_out[j]

        return decoder_rec


def get_file_path_list(dir_path):
    result = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


if __name__ == '__main__':
    batch_size = 1

    file_list = get_file_path_list(r'e:\argoverse-api-ccuse\forecasting_sample\data')
    data = Data_read(file_list)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)
    encoder_net = Encoder()
    decoder_net = Decoder()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net)
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(1):
        for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
            pred = net(x1, x2, y, y_st)
            g = make_dot(pred, params=dict(net.named_parameters()))
            g.format = "png"
            g.view()
            loss = criteria(pred, y)
            optimizer.step()
            print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))
            pass
