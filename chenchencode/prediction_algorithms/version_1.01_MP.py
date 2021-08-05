# 按照新的网络结构进行了调整主要是：
# 1. 对于（每条，共p条）自车和周车轨迹：raw_data(n,2) -1x1卷积-> (n,8) -1x3卷积-> (n,10) -lstm-> (128,)
# 2. 对于（每条，共q条）车道中心线： raw_data(10,2) -1x1卷积-> (10,8) -1x3卷积-> (10,1) -fc-> (128,)
# concat(1,2) --> (m+n,128) -注意力权重-> (m+n,128) with c_0 --> LSTM -逐步预测->(x_n+1, y_n+1)
#                                                                ^
#                                           自车轨迹最后一点坐标 (x_n,y_n)

# 20210604 修改： 发现使用mseloss会导致所有的点都学习到均值：因为这样确实是loss最小的情况，但不是想要的结果，所以考虑引入方差
# 20210608 进展：可以成功对单个样本进行预测，精度可以到10-20cm以内
# 20210804 尝试多进程测试

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
from torch.multiprocessing import Process, Queue, Manager
import torch.multiprocessing as mp

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
    def __init__(self, batch_size, encoder, decoder, attention, teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

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
                if random.random() < self.teacher_forcing_ratio.value:
                    decoder_input = decoder_out
                else:
                    decoder_input = y[j, i, :].reshape(1, 1, 2)
                decoder_rec[j][i] = decoder_out[0]

        return decoder_rec

    def check_learning_rate(self):
        return self.teacher_forcing_ratio.value


def get_file_path_list(dir_path):
    result = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            result.append(filename)
    return result


def loss_cal(pred, y):
    loss_dis = torch.abs(pred.diff(dim=1) - y.diff(dim=1))
    loss_dis = loss_dis.sum()
    loss_med = torch.pow(pred - y, 2)
    loss_med = loss_med.sum(-1)
    loss_med = torch.sqrt(loss_med).sum() / 30 / 2  # TODO： 治理用的均值，但是上边用的和，上边是否需要取均值？
    loss = loss_dis + loss_med

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


def load_ave_loss(load_path):
    info = torch.load(load_path)

    return info['ave_loss']


class Learner(object):
    def __init__(self, data_loader, net, net_share, stop_sign, criteria, optimizer, scheduler, recorder, p_num,
                 ite_num, loss_all, teacher_forcing_ratio, ave_loss_rec, argo_data_reader):
        self.data_loader = data_loader
        self.net = net
        self.net_share = net_share
        self.stop_sign = stop_sign
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = recorder

        self.p_num = p_num
        self.ite_num = ite_num
        self.loss_all = loss_all
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.ave_loss_rec = ave_loss_rec

        self.recode_freq = 500

        self.argo_data_reader = argo_data_reader

    def run(self):
        print('learner %d start...' % self.p_num)
        tic = time.time()
        while self.stop_sign.value > 0.001:
            for batch_id, (x1, x2, y, y_st) in enumerate(self.data_loader):
                pred = self.net(x1, x2, y, y_st)
                loss = self.criteria(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                self.loss_all.value += float(loss)
                ave_loss = self.loss_all.value / (self.ite_num.value + 1)
                if self.ite_num.value % 100 == 0:  # 每 100 次输出结果，记录ave_loss曲线
                    print(
                        'Epoch: {}, Loss: {:.5f}, ave loss: {:.5f}, lr: {:.10f}, teaching rate: {:.3f}, (l+al)/2: {:.5f}'
                            .format(self.ite_num.value + 1, loss.item(), ave_loss, self.optimizer.param_groups[0]['lr'],
                                    self.net.check_learning_rate(),
                                    (ave_loss + float(loss)) / 2))
                    self.ave_loss_rec.append(ave_loss)
                    print('learner {}: hundred ite time {:.5f} s'.format(self.p_num, time.time() - tic))
                    tic = time.time()
                if self.ite_num.value % self.recode_freq == 0:
                    self.recorder.recode_state(self.ite_num.value, self.net.state_dict(), self.optimizer.state_dict(), loss,
                                               self.loss_all.value, self.scheduler.state_dict())
                    abs_error = self.argo_data_reader.get_absolute_error(pred, y)
                    self.recorder.general_record(self.ite_num.value, 'abs_error',
                                                 {'error': abs_error['Average_error'], 'ave_loss': self.ave_loss_rec})
                self.teacher_forcing_ratio.value = np.clip(np.round((ave_loss + float(loss)) / 2 / 128 - 0.1, 1), 0.0,
                                                     0.9)
                self.ite_num.value += 1


def learnning(data_loader, net, net_share, stop_sign, criteria, optimizer, scheduler, recorder, p_num,
              ite_num, loss_all, teacher_forcing_ratio, ave_loss_rec, argo_data_reader):
    learner = Learner(data_loader, net, net_share, stop_sign, criteria, optimizer, scheduler, recorder, p_num,
                      ite_num, loss_all, teacher_forcing_ratio, ave_loss_rec, argo_data_reader)
    learner.run()


def mp_training():
    mp.set_start_method('spawn')

    learning_rate = 0.0001
    recode_freq = 500
    method_version = 'version_1'
    batch_size = 128
    teacher_forcing_ratio_init = 0.1

    manager = Manager()
    teacher_forcing_ratio = mp.Value('f', teacher_forcing_ratio_init)

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

    data = Data_read(file_list, argo_data_reader)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=co_fn)

    encoder_net = Encoder()
    encoder_net.share_memory()
    decoder_net = Decoder()
    decoder_net.share_memory()
    attention_net = Attention_net()
    attention_net.share_memory()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net,
                  teacher_forcing_ratio=teacher_forcing_ratio)
    net.share_memory()

    encoder_net_share = Encoder()
    encoder_net_share.share_memory()
    decoder_net_share = Decoder()
    decoder_net_share.share_memory()
    attention_net_share = Attention_net()
    attention_net_share.share_memory()
    net_share = Seq2Seq(batch_size=batch_size, encoder=encoder_net_share, decoder=decoder_net_share,
                        attention=attention_net_share, teacher_forcing_ratio=teacher_forcing_ratio)
    net_share.share_memory()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000)

    e, net, optimizer, loss_all, scheduler = load_exist_net(
        r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000_full_net_state.pkl',
        net, optimizer, scheduler)

    ave_loss_rec = load_ave_loss(
        r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000abs_error.pkl')
    ave_loss_rec = manager.list(ave_loss_rec)

    net_share.load_state_dict(net.state_dict())

    recorder = Recorder(method_version)

    stop_sign = mp.Value('i', 20)
    ite_num = mp.Value('i', e)
    loss_all = mp.Value('f', loss_all)

    print('Trainning start..., current ite number=%d, ave_loss=%f' % (e, loss_all.value / e / batch_size))

    p_process = []
    for x in range(3):
        p_process.append(
            Process(target=learnning,
                    args=(data_loader, net, net_share, stop_sign, loss_cal, optimizer, scheduler, recorder, x,
                          ite_num, loss_all, teacher_forcing_ratio, ave_loss_rec, argo_data_reader)))

    for p in p_process:
        p.start()
    for p in p_process:
        p.join()

def mp_verify():

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
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=co_fn)

    encoder_net = Encoder()
    decoder_net = Decoder()
    attention_net = Attention_net()
    net = Seq2Seq(batch_size=batch_size, encoder=encoder_net, decoder=decoder_net, attention=attention_net)

    info = torch.load(r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000_full_net_state.pkl')
    net.load_state_dict(info['net'])

    for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):
        pred = net(x1, x2, y, y_st)
        abs_error = argo_data_reader.get_absolute_error(pred, y)


if __name__ == '__main__':
    # mp_training()
    mp.set_start_method("spawn")
    num_workers = mp.cpu_count() - 1
    print(f"num of workers {num_workers}")

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

    data_loader = DataLoader(data, batch_size=batch_size, num_workers=3, collate_fn=co_fn)
    for batch_id, (x1, x2, y, y_st) in enumerate(data_loader):

        pass

