import torch
from torch import nn
from torch.autograd import Variable
import hiddenlayer as h
from torchviz import make_dot
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data_

input_size = 10  # 输入x的特征数量
hidden_size = 20 # 隐层的特征数量
num_layer = 2 # RNN层数量
batch_size = 3
rnn = nn.RNN(input_size, hidden_size, num_layer)
print('rnn:', rnn)
input = Variable(torch.randn(5, batch_size, input_size))
print('input', input)
h0 = Variable(torch.randn(num_layer, batch_size, hidden_size))
print('h0',h0)
output, hn = rnn(input, h0)
print('output, hn', output, hn)