# 按照新的网络结构进行了调整主要是：
# 1. 对于（每条，共p条）自车和周车轨迹：raw_data(n,2) -1x1卷积-> (n,8) -1x3卷积-> (n,10) -lstm-> (128,)
# 2. 对于（每条，共q条）车道中心线： raw_data(10,2) -1x1卷积-> (10,8) -1x3卷积-> (10,1) -fc-> (128,)
# concat(1,2) --> (m+n,128) -注意力权重-> (m+n,128) with c_0 --> LSTM -逐步预测->(x_n+1, y_n+1)
#                                                                ^
#                                           自车轨迹最后一点坐标 (x_n,y_n)

# 20210604 修改： 发现使用mseloss会导致所有的点都学习到均值：因为这样确实是loss最小的情况，但不是想要的结果，所以考虑引入方差
# 20210608 进展：可以成功对单个样本进行预测，精度可以到10-20cm以内
import numpy as np
import torch
import torch.utils.data as data_
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utlils
import torch.nn.functional as F
from torch import nn
from chenchencode.modules.arg_customized import data_loader_customized
import os
import random
from chenchencode.modules.utils import Recorder
import netron
import time

import sys

print(sys.path)

teacher_forcing_ratio = 0.5


class Data_read(data_.Dataset):
    def __init__(self, file_path_list, argo_data_reader):
        self.file_path_list = file_path_list
        self.argo_data_reader = argo_data_reader

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        raw_data = self.argo_data_reader.get_all_traj_for_train(self.file_path_list[idx])

        return raw_data


def co_fn(data_tuple):
    x1 = [sq[0] for sq in data_tuple]
    x2 = [sq[1] for sq in data_tuple]
    y = [sq[2] for sq in data_tuple]
    y_st = [sq[0][0][-1][1:] for sq in data_tuple]
    return x1, x2, torch.cat(y).reshape(len(y), 30, 2), torch.cat(y_st).reshape(len(y_st), 2).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch_x11=8, out_ch_x12=8, out_ch_x21=8, out_ch_x22=1, fc_in=8, out_ch_final=128):
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


class Attention_net(nn.Module):
    def __init__(self, input_ch=128, mid_out_ch=128):
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

        self.learning_rate = teacher_forcing_ratio

        return decoder_rec

    def check_learning_rate(self):
        return self.learning_rate


def get_file_path_list(dir_path):
    result = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            result.append(filename)
    return result


def loss_cal(pred, y, loss_version=0):
    criteria = nn.MSELoss()
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

def load_exist_net(load_path, net, optimizer, scheduler):
    file_name = os.path.basename(load_path)
    e = int(file_name.split('_')[1])
    info = torch.load(load_path)
    net.load_state_dict(info['net'])
    optimizer.load_state_dict(info['optimizer'])
    scheduler.load_state_dict(info['scheduler'])
    loss_all = info['loss_all']

    return e, net, optimizer, loss_all, scheduler

def training():
    laod_exit_net = False
    learning_rate = 0.0001
    recode_freq = 500
    method_version = 'version_1'
    loss_version = 3

    raw_data_dir = r'e:\数据集\03_Argoverse\forecasting_train_v1.1.tar\train\data'
    file_list = get_file_path_list(raw_data_dir)
    argo_data_reader = data_loader_customized(raw_data_dir,
                                              normalization=True,
                                              range_const=True,
                                              return_type='list[tensor]',
                                              include_centerline=True,
                                              rotation_to_standard=True,
                                              save_preprocessed_data=True,
                                              fast_read_check=True)

    batch_size = 128
    data = Data_read(file_list, argo_data_reader)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    encoder_net = Encoder()
    decoder_net = Decoder()
    attention_net = Attention_net()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000, )

    recorder = Recorder(method_version)

    loss_all = 0
    stop_label = 0
    e = 1
    ave_loss_rec = []
    tic = time.time()

    e, net, optimizer, loss_all, scheduler = load_exist_net(
        r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210730_version_1\i_1000_full_net_state.pkl',
        net, optimizer, scheduler)

    print('Trainning start..., current ite number=%d, ave_loss=%f' % (e, loss_all / e / batch_size))
    while stop_label == 0:
        for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
            pred = net(x1, x2, y, y_st)
            loss = loss_cal(pred, y, loss_version)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            loss_all += float(loss)
            ave_loss = loss_all / (e + 1)
            if e % 100 == 0:  # 每 100 次输出结果，记录ave_loss曲线
                print('Epoch: {}, Loss: {:.5f}, ave loss: {:.5f}, lr: {:.10f}, teaching rate: {:.3f}, (l+al)/2: {:.5f}'
                      .format(e + 1, loss.item(), ave_loss, optimizer.param_groups[0]['lr'], net.check_learning_rate(),
                              (ave_loss + float(loss)) / 2))
                ave_loss_rec.append(ave_loss)
                print('time {:.5f} s'.format(time.time() - tic))
                tic = time.time()
            if e % recode_freq == 0:
                recorder.recode_state(e, net.state_dict(), optimizer.state_dict(), loss, loss_all,
                                      scheduler.state_dict())
                abs_error = argo_data_reader.get_absolute_error(pred, y)
                recorder.general_record(e, 'abs_error', {'error': abs_error['Average_error'], 'ave_loss': ave_loss_rec})
            # del loss
            teacher_forcing_ratio = np.clip(np.round((ave_loss + float(loss)) / 2 / batch_size, 1), 0.0, 0.9)
            # if ave_loss < 0.3:
            #     teacher_forcing_ratio = 0.1
            # if ave_loss >= 0.3:
            #     teacher_forcing_ratio = 0.5
            if loss < 0.001:
                stop_label = 1
            e += 1
    # torch.onnx.export(net, (x1, x2, y, y_st), 'viz.pt', opset_version=11)
    # netron.start('viz.pt')

def verify():
    teacher_forcing_ratio = -1.0
    raw_data_dir = r'E:\数据集\03_Argoverse\forecasting_val_v1.1.tar\forecasting_val_v1.1\val\data'
    file_list = get_file_path_list(raw_data_dir)
    argo_data_reader = data_loader_customized(raw_data_dir,
                                              normalization=True,
                                              range_const=True,
                                              return_type='list[tensor]',
                                              include_centerline=True,
                                              rotation_to_standard=True,
                                              save_preprocessed_data=True,
                                              fast_read_check=True)
    batch_size = 128
    data = Data_read(file_list, argo_data_reader)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    encoder_net = Encoder()
    decoder_net = Decoder()
    attention_net = Attention_net()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net)

    info = torch.load(r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000_full_net_state.pkl')
    net.load_state_dict(info['net'])

    error_save_mean = []
    error_save_at_1sec = []
    error_save_at_2sec = []
    error_save_at_3sec = []

    e = 0
    recorder = Recorder(method_version='version_1_val')
    record_freq = 100
    print_error = True

    for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
        pred = net(x1, x2, y, y_st)
        abs_error = argo_data_reader.get_absolute_error(pred, y, print_error=print_error)
        if print_error: print_error = False

        error_save_mean += abs_error['Each_eror'][0].tolist()
        error_save_at_1sec += abs_error['Each_eror'][1]
        error_save_at_2sec += abs_error['Each_eror'][2]
        error_save_at_3sec += abs_error['Each_eror'][3]

        e += 1

        if e % 50 == 0:
            print('ite num: %d' % e)
            print_error = True
        if e % record_freq == 0 or e > 300:
            recorder.general_record(e, 'val_abs_error', {'error_save_mean': error_save_mean,
                                                         'error_save_at_1sec': error_save_at_1sec,
                                                         'error_save_at_2sec': error_save_at_2sec,
                                                         'error_save_at_3sec': error_save_at_3sec})
            print('ite num %d saved' % e)


def data_test():
    teacher_forcing_ratio = -1.0
    raw_data_dir = r'E:\数据集\03_Argoverse\forecasting_test_v1.1.tar\forecasting_test_v1.1\test_obs\data'
    file_list = get_file_path_list(raw_data_dir)
    argo_data_reader = data_loader_customized(raw_data_dir,
                                              normalization=True,
                                              range_const=True,
                                              return_type='list[tensor]',
                                              include_centerline=True,
                                              rotation_to_standard=True,
                                              save_preprocessed_data=True,
                                              fast_read_check=True)
    batch_size = 128
    data = Data_read(file_list, argo_data_reader)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    encoder_net = Encoder()
    decoder_net = Decoder()
    attention_net = Attention_net()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net)

    info = torch.load(r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000_full_net_state.pkl')
    net.load_state_dict(info['net'])

    error_save_mean = []
    error_save_at_1sec = []
    error_save_at_2sec = []
    error_save_at_3sec = []

    e = 0
    recorder = Recorder(method_version='version_1_val')
    record_freq = 100
    print_error = True

    for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
        pred = net(x1, x2, y, y_st)
        abs_error = argo_data_reader.get_absolute_error(pred, y, print_error=print_error)
        if print_error: print_error = False

        error_save_mean += abs_error['Each_eror'][0].tolist()
        error_save_at_1sec += abs_error['Each_eror'][1]
        error_save_at_2sec += abs_error['Each_eror'][2]
        error_save_at_3sec += abs_error['Each_eror'][3]

        e += 1

        if e % 50 == 0:
            print('ite num: %d' % e)
            print_error = True
        if e % record_freq == 0 or e > 600:
            recorder.general_record(e, 'val_abs_error', {'error_save_mean': error_save_mean,
                                                         'error_save_at_1sec': error_save_at_1sec,
                                                         'error_save_at_2sec': error_save_at_2sec,
                                                         'error_save_at_3sec': error_save_at_3sec})
            print('ite num %d saved' % e)

if __name__ == '__main__':
    # verify()

    data_test()

    # xx = torch.load(r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210806_version_1_val\i_309val_abs_error.pkl')
    # print(np.array(xx['error_save_mean']).mean())
    # print(np.array(xx['error_save_at_1sec']).mean())
    # print(np.array(xx['error_save_at_2sec']).mean())
    # print(np.array(xx['error_save_at_3sec']).mean())