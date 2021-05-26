import torch
import torch.utils.data as data_
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utlils
from torch import nn
from chenchencode.arg_customized import data_loader_customized
from chenchencode.arg_customized import torch_treat
import os
import random

import sys

print(sys.path)


class Data_read(data_.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raw_data = (self.x1[idx], self.x2[idx], self.y[idx])

        return raw_data


def co_fn(data_tuple):
    x1 = [sq[0] for sq in data_tuple]
    x2 = [sq[1] for sq in data_tuple]
    y = [sq[2] for sq in data_tuple]
    return x1, x2, y


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.nn_x1_1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1)
        self.nn_x1_2 = nn.Tanh()
        self.nn_x1_3 = nn.Conv1d(in_channels=8, out_channels=10, kernel_size=3, padding=1)
        self.nn_x1_4 = nn.Tanh()
        self.nn_x1_5 = nn.LSTM(input_size=10, hidden_size=128, num_layers=1, batch_first=True)

        self.nn_x2_1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1)
        self.nn_x2_2 = nn.Tanh()
        self.nn_x2_3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.nn_x2_4 = nn.Linear(in_features=10, out_features=128)

        self.nn_w = nn.
        # self.nn_x1 = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=8, kernel_size=(1,)),
        #     nn.Tanh(),
        #     nn.Conv1d(in_channels=1, out_channels=10, kernel_size=4),
        #     nn.Tanh(),
        #     nn.LSTM(input_size=4, hidden_size=128, num_layers=1, batch_first=True)
        # )
        # self.nn_x2 = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1),
        #     nn.Tanh(),
        #     nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3),
        #     nn.Linear(in_features=10, out_features=128)
        # )

    def forward(self, x1, x2):
        batch_num = len(x1)
        final_out = torch.empty(batch_num, 128)
        for batch_i in range(batch_num):
            traj_data = x1[batch_i]
            center_data = x2[batch_i]
            traj_num, center_num = len(traj_data), len(center_data)
            mid_out = torch.empty(traj_num + center_num, 128)
            for i in range(traj_num):
                input = traj_data[i].permute(1, 0).unsqueeze(0)
                y = self.nn_x1_1(input)
                y = self.nn_x1_2(y)
                y = self.nn_x1_3(y)
                y = self.nn_x1_4(y)
                y = y.permute(0, 2, 1)
                out, (h_n, h_c) = self.nn_x1_5(y)
                mid_out[i] = h_c.squeeze(-2)
            for i in range(center_num):
                input = center_data[i].permute(1, 0).unsqueeze(0)
                y = self.nn_x2_1(input)
                y = self.nn_x2_2(y)
                y = self.nn_x2_3(y)
                y = y.squeeze(-1)
                out = self.nn_x2_4(y)
                mid_out[traj_num + i] = out.squeeze(-2)

            attention_weight = mid_out.sum(1) / mid_out.sum()
            attention_weight = attention_weight.reshape(traj_num + center_num, 1)
            mid_out = mid_out * attention_weight
            mid_out = mid_out.sum(0)

            final_out[batch_i] = mid_out

        return final_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


if __name__ == '__main__':
    train_x1 = [[torch.randn(31, 2), torch.randn(21, 2)],
                [torch.randn(27, 2), torch.randn(23, 2), torch.randn(22, 2), torch.randn(26, 2)],
                [torch.randn(15, 2), torch.randn(43, 2), torch.randn(32, 2)]]
    train_x2 = [[torch.randn(10, 2), torch.randn(10, 2), torch.randn(10, 2), torch.randn(10, 2)],
                [torch.randn(10, 2), torch.randn(10, 2)],
                [torch.randn(10, 2)]]
    train_y = [torch.randn(30, 2),
               torch.randn(30, 2),
               torch.randn(30, 2)]

    data_ = Data_read(train_x1, train_x2, train_y)
    data_loader = DataLoader(data_, batch_size=2, shuffle=True, collate_fn=co_fn)
    net = Encoder()

    for epoch in range(3):
        for batch_id, (x1, x2, y) in enumerate(data_loader):
            out = net(x1, x2)
            pass
