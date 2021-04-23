import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data_


class MyData(data_.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx])
        return tuple_


def collate_fn(data_tuple):  # data_tuple是一个列表，列表中包含batchsize个元组，每个元组中包含数据和标签
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)  # 这行代码只是为了把列表变为tensor
    re_data = data.unsqueeze(-1) if data. dim() < 3 else data
    return re_data, label, data_length


if __name__ == '__main__':

    EPOCH = 2
    inputsize = 2
    batchsize = 3
    hiddensize = 4
    num_layers = 2
    learning_rate = 0.001

    # 训练数据
    # train_x = [torch.FloatTensor([1]*7),
    #            torch.FloatTensor([2]*6),
    #            torch.FloatTensor([3]*5),
    #            torch.FloatTensor([4]*4),
    #            torch.FloatTensor([5]*3),
    #            torch.FloatTensor([6]*2),
    #            torch.FloatTensor([7])]

    train_x = [torch.FloatTensor([[1, 1.2]] * 7),
               torch.FloatTensor([[2, 2.2]] * 6),
               torch.FloatTensor([[3, 3.3]] * 5),
               torch.FloatTensor([[4, 4.4]] * 4),
               torch.FloatTensor([[5, 5.5]] * 3),
               torch.FloatTensor([[6, 6.6]] * 2),
               torch.FloatTensor([[7, 7.7]])]
    # train_x = [torch.FloatTensor([[1] * 7, [1.1] * 7]),
    #            torch.FloatTensor([[2] * 6, [2.2] * 6]),
    #            torch.FloatTensor([[3] * 5, [3.2] * 5]),
    #            torch.FloatTensor([[4] * 4, [4.2] * 4]),
    #            torch.FloatTensor([[5] * 3, [5.2] * 3]),
    #            torch.FloatTensor([[6] * 2, [6.2] * 2]),
    #            torch.FloatTensor([[7, 7.7]])]

    # train_x = [torch.FloatTensor([[1, 1.1], [1, 1.1], [1, 1.1], [1, 1.1], [1, 1.1], [1, 1.1], [1, 1.1]]),
    #            torch.FloatTensor([[2, 2.2], [2, 2.2], [2, 2.2], [2, 2.2], [2, 2.2], [2, 2.2]]),
    #            torch.FloatTensor([[3, 3.3], [3, 3.3], [3, 3.3], [3, 3.3], [3, 3.3]]),
    #            torch.FloatTensor([[4, 4.4], [4, 4.4], [4, 4.4], [4, 4.4]]),
    #            torch.FloatTensor([[5, 5.5], [5, 5.5], [5, 5.5]]),
    #            torch.FloatTensor([[6, 6.6], [6, 6.6]]),
    #            torch.FloatTensor([[7, 7.7]])]
    # 标签
    train_y = [torch.rand(7, hiddensize),
               torch.rand(6, hiddensize),
               torch.rand(5, hiddensize),
               torch.rand(4, hiddensize),
               torch.rand(3, hiddensize),
               torch.rand(2, hiddensize),
               torch.rand(1, hiddensize)]

    data_ = MyData(train_x, train_y)  # 注意这里是一个数据集对象，其中定义了__getitem__方法，调用时才是输出对应的数据
    data_loader = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
    net = nn.LSTM(input_size=inputsize, hidden_size=hiddensize, num_layers=num_layers, batch_first=True)
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 训练方法一
    for epoch in range(EPOCH):
        for batch_id, (batch_x, batch_y, batch_x_len) in enumerate(data_loader):
            batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_x_len, batch_first=True)
            out, _ = net(batch_x_pack)  # out.data's shape (所有序列总长度, hiddensize)
            out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
            loss = criteria(out_pad, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))

    # 训练方法二
    for epoch in range(EPOCH):
        for batch_id, (batch_x, batch_y, batch_x_len) in enumerate(data_loader):
            batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_x_len, batch_first=True)
            batch_y_pack = rnn_utils.pack_padded_sequence(batch_y, batch_x_len, batch_first=True)
            out, _ = net(batch_x_pack)  # out.data's shape (所有序列总长度, hiddensize)
            loss = criteria(out.data, batch_y_pack.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch:{:2d}, batch_id:{:2d}, loss:{:6.4f}'.format(epoch, batch_id, loss))

    print('Training done!')
