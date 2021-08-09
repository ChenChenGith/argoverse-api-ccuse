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


def verify():
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

    info = torch.load(
        r'E:\argoverse-api-ccuse\chenchencode\Saved_resultes\20210802_version_1\i_9000_full_net_state.pkl')
    net.load_state_dict(info['net'])

    e = 0
    recorder = Recorder(method_version='version_1_val')
    record_freq = 100
    print_error = True

    output_all = {}
    for file_name in file_list:
        raw_data = argo_data_reader.get_all_traj_for_train(file_name)
        x1, x2, y, y_st = co_fn([raw_data])
        pred = net(x1, x2, y, y_st)
        seq_id = int(file_name[:-4])
        output_all[seq_id] = pred.detach().numpy()

        e += 1

        if e % 50 == 0:
            print('ite num: %d' % e)
        if e % record_freq == 0 or e > 300:
            recorder.general_record(1234, 'test_pred_coor', output_all)
            print('ite num %d saved' % e)

if __name__ == '__main__':
    verify()