# loader for the prediction
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.utils.data as data_
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utlils
from torch import nn
from chenchencode.arg_customized import data_loader_customized
from chenchencode.arg_customized import torch_treat
import os
import random

teacher_forcing_ratio = 0.5


class Data_read(data_.Dataset):
    '''
    用于构造原始数据集的一个类。
    被dataloader调用时，会自动调用__getitem__方法，根据batch的大小，共调用batch_size次，将返回的数据送入collate_fn进一步处理成为可以直接输入到网络中的格式

    在本轨迹预测中，因为读取csv数据的方法被封装到了chenchencode.arg_customized.data_loader_customized中，所以传入的是csv file list
    由__getitem__调用此方法来返回数据
    '''

    def __init__(self, file_path_list):
        self.file_path_list = file_path_list

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        data_reader = data_loader_customized(self.file_path_list[idx])
        raw_data = data_reader.get_all_traj_for_train(normalization=True, return_type='tensor',
                                                      include_centerline=False)  # TODO:后期需要根据网络形式来更改该函数参数
        return raw_data


def collate_fn(data_tuple):
    '''
    被dataloader调用，将原始数据处理成为可以输入到网络中的格式

    v0.0版本: 拟采用lstm方法，所以要有1）按数据size排序，2）pad齐整数据
    '''
    version = 0.0
    if version == 0.0:
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)  # 1) 根据batch中各个数据的长度进行排序，这是做pad_sqequence的要求
        data_x = [sq[0] for sq in data_tuple]  # 已知数据，即x
        data_y = [sq[1] for sq in data_tuple]  # 已知数据，即y
        data_length = [len(sq[0]) for sq in data_tuple]  # batch每条数据的长度
        data_x = rnn_utlils.pad_sequence(data_x, batch_first=True, padding_value=0.0)  # 规整不等长数据至等长
        data_y = rnn_utlils.pad_sequence(data_y, batch_first=True, padding_value=0.0)
        return data_x, data_y, data_length


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Encoder, self).__init__()
        self.input_s = input_size
        self.hidden_s = hidden_size
        self.num_lay = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        nn.init.normal_(self.rnn.bias_hh_l0)
        nn.init.normal_(self.rnn.bias_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, x):
        lstm_out, hidden_out = self.rnn(x)
        # mid_out, data_len = rnn_utlils.pad_packed_sequence(lstm_out, batch_first=True) # 由于输入是pack_padded_sequence对象，输出也是此类型，要使用反打包函数解包
        return lstm_out, hidden_out


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.input_s = input_size
        self.hidden_s = hidden_size
        self.num_lay = num_layers
        self.output_s = output_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        nn.init.normal_(self.rnn.bias_hh_l0)
        nn.init.normal_(self.rnn.bias_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, input, hidden_input):
        lstm_out, hidden_out = self.rnn(input, hidden_input)
        lstm_out = self.out(lstm_out)
        return lstm_out, hidden_out


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, know_data, decoder_init):
        encoder_out, encoder_hidden = self.encoder(know_data)
        decoder_hidden = encoder_hidden
        batch_num = encoder_hidden[0].shape[1]
        # decoder_input = torch.zeros(batch_num, 1, self.decoder.input_s)
        decoder_input = decoder_init[:, 0, :].unsqueeze(-2)
        decoder_rec = torch.empty(batch_num, 30, self.decoder.output_s)
        for i in range(30):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if random.random() < teacher_forcing_ratio:
                decoder_input = decoder_out
            else:
                decoder_input = decoder_init[:, i, :].unsqueeze(-2)
            for j in range(batch_num):
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
    EPOCH = 20

    encoder_input_size = 4
    encoder_hidden_size = 128
    encoder_num_layer = 1
    encoder_output_size = 3

    decoder_input_size = 2
    decoder_hidden_size = encoder_hidden_size
    decoder_num_layer = 1
    decoder_output_size = 2

    batch_size = 3
    learning_rate = 0.001

    file_list = get_file_path_list(r'e:\argoverse-api-ccuse\forecasting_sample\data')
    data = Data_read(file_list)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(encoder_input_size, encoder_hidden_size, encoder_num_layer, encoder_output_size)
    decoder = Decoder(decoder_input_size, decoder_hidden_size, decoder_num_layer, decoder_output_size)

    net = Seq2seq(encoder, decoder)
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(EPOCH):
        for batch_id, (batch_x, batch_y, batch_X_len) in enumerate(data_loader):
            batch_x_pack = rnn_utlils.pack_padded_sequence(batch_x, batch_X_len, batch_first=True)
            out = net(batch_x_pack, batch_y)
            loss = criteria(out, batch_y)
            loss.backward()
            optimizer.step()
            print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))

    pass
