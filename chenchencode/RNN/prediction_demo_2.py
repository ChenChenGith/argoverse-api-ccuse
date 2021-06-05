# 从demo1修改而来
# 用来从自车轨迹中，训练加速度出来

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
import netron

import sys

print(sys.path)

teacher_forcing_ratio = 0.5


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
    def __init__(self, in_ch=3, out_ch_x11=8, out_ch_x12=8, out_ch_x21=8, out_ch_x22=1, fc_in=8, out_ch_final=64):
        super(Encoder, self).__init__()
        self.in_channel = in_ch
        self.out_ch_x11 = out_ch_x11
        self.out_ch_x12 = out_ch_x12
        self.out_ch_x21 = out_ch_x21
        self.out_ch_x22 = out_ch_x22
        self.fc_in = fc_in
        self.out_ch_final = out_ch_final

        self.nn_x1_1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_ch_x11, kernel_size=1)
        self.nn_x1_2 = nn.Tanh()
        self.nn_x1_3 = nn.Conv1d(in_channels=self.out_ch_x11, out_channels=self.out_ch_x12, kernel_size=3, padding=1)
        self.nn_x1_4 = nn.LeakyReLU()
        self.nn_x1_5 = nn.LSTM(input_size=self.out_ch_x12, hidden_size=self.out_ch_final, num_layers=1,
                               batch_first=True)

        self.nn_x2_1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_ch_x21, kernel_size=1)
        self.nn_x2_2 = nn.Tanh()
        self.nn_x2_3 = nn.Conv1d(in_channels=self.out_ch_x21, out_channels=self.out_ch_x22, kernel_size=3)
        self.nn_x2_4 = nn.LeakyReLU()
        self.nn_x2_5 = nn.Linear(in_features=self.fc_in, out_features=self.out_ch_final)

    def forward(self, x1, x2):
        batch_num = len(x1)
        final_out = torch.empty(batch_num, self.out_ch_final)
        for batch_i in range(batch_num):
            traj_data = x1[batch_i]
            center_data = x2[batch_i]
            traj_num, center_num = len(traj_data), len(center_data)
            mid_out = torch.empty(traj_num + center_num, self.out_ch_final)
            for i in range(traj_num):
                input = traj_data[i].permute(1, 0).unsqueeze(0)
                y = self.nn_x1_1(input)
                y = self.nn_x1_2(y)
                y = self.nn_x1_3(y)
                y = self.nn_x1_4(y)
                y = y.permute(0, 2, 1)
                out, (h_n, c_n) = self.nn_x1_5(y)
                mid_out[i] = c_n.squeeze(-2)
                if i == 0:
                    h_n_reserve = h_n
            for i in range(center_num):
                input = center_data[i].permute(1, 0).unsqueeze(0)
                y = self.nn_x2_1(input)
                y = self.nn_x2_2(y)
                y = self.nn_x2_3(y)
                y = self.nn_x2_4(y)
                y = y.squeeze(-1)
                out = self.nn_x2_5(y)
                mid_out[traj_num + i] = out.squeeze(-2)

            attention_weight = F.softmax(mid_out, dim=0)
            mid_out = torch.mul(attention_weight, mid_out).sum(0)

            final_out[batch_i] = mid_out

        final_out = final_out.unsqueeze(0)

        return final_out, h_n_reserve


class Decoder(nn.Module):
    def __init__(self, in_ch=2, fc_out=16, lstm_ch_hidden=64, out_ch_final=2):
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
            if random.random() < teacher_forcing_ratio:
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


class Acce_trainer(nn.Module):
    def __init__(self, in_ch=3, med_ch=8, hidden_ch=32, out_c=2):
        super(Acce_trainer, self).__init__()
        self.input_net = nn.Linear(in_features=in_ch, out_features=med_ch)
        self.rnn =nn.LSTM(input_size=med_ch, hidden_size=hidden_ch, num_layers=1, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_ch, out_features=hidden_ch),
            nn.Tanh(),
            nn.Linear(in_features=hidden_ch, out_features=out_c)
        )

    def forward(self, x):
        y = self.input_net(x)
        y = y.unsqueeze(0)
        y, (h_n, c_n) = self.rnn(y)
        h_n = h_n.squeeze(0).squeeze(0)
        y = self.out(h_n)

        return y



if __name__ == '__main__':
    test_version = 0
    learning_rate = 0.0001

    batch_size = 1

    file_list = get_file_path_list(r'e:\argoverse-api-ccuse\forecasting_sample\data')
    data = Data_read(file_list)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)
    encoder_net = Encoder()
    decoder_net = Decoder()
    if test_version == 0:
        net = Acce_trainer()
    else:
        net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net)

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, )

    loss_all = 0
    stop_label = 0
    e = 1


    while stop_label == 0:
        for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
            if test_version == 0:
                # 测试学习加速度
                x = x1[0][0]
                y = torch.tensor([0.01296, -0.00015])  # excel计算的速度和加速度
                pred = net(x)
                l1 = torch.abs(pred.max() - y.max())  # 最大值、最小值限定
                l2 = torch.abs(pred.min() - y.min())
                loss = criteria(pred, y) + l1 + l2
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            e += 1
            loss_all += loss
            if e % 10 == 0:  # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f}, ave loss: {:.5f}, lr: {:.10f}'.format(e + 1, loss.item(),
                                                                                      loss_all / (e + 1),
                                                                                      optimizer.param_groups[0]['lr']))
            if loss < 0.000001:
                stop_label = 1

    # torch.onnx.export(net, (x1, x2, y, y_st), 'viz.pt', opset_version=11)
    # netron.start('viz.pt')
