# 按照新的网络结构进行了调整主要是：
# 1. 对于（每条，共p条）自车和周车轨迹：raw_data(n,2) -1x1卷积-> (n,8) -1x3卷积-> (n,10) -lstm-> (128,)
# 2. 对于（每条，共q条）车道中心线： raw_data(10,2) -1x1卷积-> (10,8) -1x3卷积-> (10,1) -fc-> (128,)
# concat(1,2) --> (m+n,128) -注意力权重-> (m+n,128) with c_0 --> LSTM -逐步预测->(x_n+1, y_n+1)
#                                                                ^
#                                           自车轨迹最后一点坐标 (x_n,y_n)

# 20210604 修改： 发现使用mseloss会导致所有的点都学习到均值：因为这样确实是loss最小的情况，但不是想要的结果，所以考虑引入方差
# 20210608 进展：可以成功对单个样本进行预测，精度可以到10-20cm以内


import torch
import torch.utils.data as data_
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utlils
import torch.nn.functional as F
from torch import nn

from chenchencode.arg_customized import data_loader_customized
import os
import random
import netron
import time

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
        raw_data = data_reader.get_all_traj_for_train(normalization=True, range_const=True, return_type='list[tensor]',
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
        final_out = []
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
            for i in range(center_num):
                input = center_data[i].permute(1, 0).unsqueeze(0)
                y = self.nn_x2_1(input)
                y = self.nn_x2_2(y)
                y = self.nn_x2_3(y)
                y = self.nn_x2_4(y)
                y = y.squeeze(-1)
                out = self.nn_x2_5(y)
                mid_out[traj_num + i] = out.squeeze(-2)

            final_out.append(mid_out)

        return final_out


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


class Attention_net(nn.Module):
    def __init__(self, input_ch=64, mid_out_ch=64):
        super(Attention_net, self).__init__()
        self.input_ch = input_ch
        self.mid_out_ch = mid_out_ch
        self.out_ch = 1

        self.ln1 = nn.Linear(in_features=self.input_ch, out_features=self.mid_out_ch, bias=False)
        self.ln2 = nn.Linear(in_features=self.input_ch, out_features=self.mid_out_ch, bias=False)
        self.ln3 = nn.Tanh()
        self.ln4 = nn.Linear(in_features=self.mid_out_ch, out_features=self.out_ch, bias=False)

    def forward(self, encoder_h, decoder_h):
        out_s = self.ln1(encoder_h)
        out_h = self.ln2(decoder_h)
        out = self.ln3(out_h + out_s)
        out = self.ln4(out)
        weight = F.softmax(out, dim=1)
        out = torch.mul(weight, encoder_h).sum(1)
        out = out.unsqueeze(0)

        return out


class Seq2Seq(nn.Module):
    def __init__(self, batch_size, encoder, decoder, attention):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.batch_size = batch_size

    def forward(self, x1, x2, y, y_st):
        encoder_out = self.encoder(x1, x2)  # list([[],[],..])
        batch_num = len(x1)
        init_hn = torch.zeros(batch_num, 1, self.decoder.lstm_ch_hidden)
        decoder_rec = torch.empty(batch_num, 30, 2)
        for j in range(batch_num):
            out_encoder = encoder_out[j].unsqueeze(0)
            decoder_hi = self.attention(out_encoder, init_hn[j].unsqueeze(0))
            decoder_input = y_st[j].unsqueeze(0)
            decoder_ci = torch.zeros(1, 1, self.decoder.lstm_ch_hidden)
            for i in range(30):
                decoder_out, (decoder_hi, decoder_ci) = self.decoder(decoder_input, (decoder_hi, decoder_ci))
                decoder_hi = self.attention(out_encoder, decoder_hi)
                if random.random() < teacher_forcing_ratio:
                    decoder_input = decoder_out
                else:
                    decoder_input = y[j, i, :].reshape(1, 1, 2)
                decoder_rec[j][i] = decoder_out[0]

        return decoder_rec


def logger(net, optimizer, loss, scheduler, save_path):
    torch.save(net, save_path + 'net.pkl')
    torch.save(net.state_dict(), save_path + 'netstate_dic.pkl')
    all_state = {'net': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'loss': loss,
                 'scheduler': scheduler.state_dict()
                 }
    torch.save(all_state, save_path + 'all_state.pkl')


def absolute_error(pred, y, norm_range=100):
    pred, y = pred * norm_range, y * norm_range
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


def get_file_path_list(dir_path):
    result = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


def loss_cal(pred, y, version=0):
    if loss_version == 0:  # 仅MSELoss，即坐标的曼哈顿距离
        loss = criteria(pred, y)
    elif loss_version == 1:  # sqrt(MSE)
        # loss_var = torch.abs(pred.var(1) - y.var(1)).sum()
        loss = torch.sqrt(criteria(pred, y))
    elif loss_version == 2:  # 欧氏距离
        loss_med = torch.pow(pred - y, 2)
        loss = loss_med.sum(-1)
        loss = torch.sqrt(loss).sum() / 30 / 2
    elif loss_version == 3:  # 欧氏距离+邻接点间距，鼓励离散
        loss_dis = torch.abs(pred.diff(dim=1) - y.diff(dim=1))
        loss_dis = loss_dis.sum()
        loss_med = torch.pow(pred - y, 2)
        loss_med = loss_med.sum(-1)
        loss_med = torch.sqrt(loss_med).sum() / 30 / 2  # TODO： 治理用的均值，但是上边用的和，上边是否需要取均值？
        loss = loss_dis + loss_med
    elif loss_version == 4:  # sqrt(MSE)+邻接点间距，鼓励离散
        loss_dis = torch.abs(pred.diff(dim=1) - y.diff(dim=1))
        loss_dis = loss_dis.sum()
        loss = loss_dis + criteria(pred, y)

    return loss


if __name__ == '__main__':

    net_ = torch.load(r'Saved_model/20210702_68sample/i11861/net.pkl')

    laod_exit_net = True
    learning_rate = 0.0001

    batch_size = 8

    file_list = get_file_path_list(r'e:\argoverse-api-ccuse\forecasting_sample\data')
    data = Data_read(file_list)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    encoder_net = Encoder()
    decoder_net = Decoder()
    attention_net = Attention_net()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net)

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000, )

    loss_all = 0
    stop_label = 0
    e = 1
    loss_version = 3
    while stop_label == 0:
        for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
            pred = net(x1, x2, y, y_st)
            loss = loss_cal(pred, y, loss_version)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            e += 1
            loss_all += float(loss)
            if e % 10 == 0:  # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f}, ave loss: {:.5f}, lr: {:.10f}'.format(e + 1, loss.item(),
                                                                                      loss_all / (e + 1),
                                                                                      optimizer.param_groups[0]['lr']))
            # del loss
            if loss_all < 0.3:
                teacher_forcing_ratio = 0.1
            if loss < 0.001:
                stop_label = 1

    # torch.onnx.export(net, (x1, x2, y, y_st), 'viz.pt', opset_version=11)
    # netron.start('viz.pt')

    save_path = 'Saved_model/20210702_68sample/i10001'
    logger(net, optimizer, loss, scheduler, save_path)
