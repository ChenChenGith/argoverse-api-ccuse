import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data_
from torchviz import make_dot
from torch.autograd import Variable
import netron


class RNN(nn.Module):
    def __init__(self, inputsize, hiddensize, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=inputsize,
            hidden_size=hiddensize,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)


if __name__ == '__main__':
    train_x = torch.FloatTensor([[1, 1.2, 3, 5, 1, 2]] * 7)
    train_x = train_x.unsqueeze(0)
    print(train_x)
    # net = RNN(2, 8, 1)
    net = nn.LSTM(6, 9, 1)
    print(net)
    out = net(train_x)
    # g = make_dot(out[0])
    # g.view()

    torch.onnx.export(net, train_x, 'viz.pt')
    netron.start('viz.pt')



    # model = nn.Sequential()
    # model.add_module('W0', nn.Linear(8, 16))
    # model.add_module('tanh', nn.Tanh())
    # model.add_module('W1', nn.Linear(16, 1))
    #
    # x = Variable(torch.randn(1, 8))
    # y = model(x)
    #
    # g = make_dot(y.mean(), params=dict(model.named_parameters()))
    # g.view()
