from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp
import torch
import time
import random
import numpy as np


class Net_class(torch.nn.Module):
    def __init__(self):
        super(Net_class, self).__init__()
        self.net_1 = torch.nn.Linear(3, 8)
        self.net_2 = torch.nn.ReLU()
        self.net_3 = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = self.net_1(x)
        y = self.net_2(y)
        y = self.net_3(y)

        return y


def data_in(data, stop_sign, data_queue):
    print('data_in')
    while not data_queue.full():  # stop_sign.value <= 500:
        data_use = torch.zeros((4, 3))
        for i in range(4):
            idx = int(np.round(random.random() * 19, 0))
            data_use[i, :] = data[idx]
        if not data_queue.full():
            data_queue.put(data_use)
        else:
            print('queue is full')
            pass
    while stop_sign.value <= 500:
        pass


class Sub_trainner(object):
    def __init__(self, net, net_share, stop_sign, criteria, optimizer, data_queue, x, expect_out, data):
        self.net = net
        self.net_share = net_share
        self.stop_sign = stop_sign
        self.criteria = criteria
        self.optimizer = optimizer
        self.data_queue = data_queue
        self.expect_out = expect_out
        self.x = x
        self.data = data

    def run(self):
        print('trainner:', self.x)
        while self.stop_sign.value <= 5000:
            self.net.load_state_dict(self.net_share.state_dict())
            # while self.data_queue.empty():
            #     print('wait for data')
            #     time.sleep(0.05)
            # data_now = self.data_queue.get()

            data_now = torch.zeros((4, 3))
            for i in range(4):
                idx = int(np.round(random.random() * 19, 0))
                data_now[i, :] = self.data[idx]

            out = self.net(data_now)
            loss = self.criteria(out, self.expect_out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.net_share.load_state_dict(self.net.state_dict())

            self.stop_sign.value += 1

            if self.stop_sign.value % 10 == 0:
                print('trainner:', self.x, 'ite num:', self.stop_sign.value, float(loss))


def trainning(net, net_share, stop_sign, criteria, optimizer, data_queue, x, expect_out, data):
    trainner = Sub_trainner(net, net_share, stop_sign, criteria, optimizer, data_queue, x, expect_out, data)
    trainner.run()


def method_1():
    mp.set_start_method('spawn')
    data = torch.randn((20, 3)).float()
    data = (data - data.mean()) / data.std()
    expect_out = torch.randn((4)).float()
    expect_out = (expect_out - expect_out.mean()) / expect_out.std()
    # print(data, expect_out)

    data_queue = Queue(maxsize=500)

    stop_sign = mp.Value('i', 0)
    net = Net_class()
    net.share_memory()
    net_share = Net_class()
    net_share.share_memory()

    net_share.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    criteria = torch.nn.MSELoss()

    p_process = []
    for x in range(2):
        p_process.append(
            Process(target=trainning,
                    args=(net, net_share, stop_sign, criteria, optimizer, data_queue, x, expect_out, data)))
    # d_process = Process(target=data_in, args=(data, stop_sign, data_queue))

    # d_process.start()
    for p in p_process:
        p.start()

    # d_process.join()
    for p in p_process:
        p.join()

def method_2():
    data = torch.randn((20, 3)).float()
    data = (data - data.mean()) / data.std()
    expect_out = torch.randn((4)).float()
    expect_out = (expect_out - expect_out.mean()) / expect_out.std()
    net = Net_class()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criteria = torch.nn.MSELoss()
    loss_all = 0
    for n in range(5000):
        data_use = torch.zeros((4, 3))
        for i in range(4):
            idx = int(np.round(random.random() * 19, 0))
            data_use[i, :] = data[idx]
        #     data_use = data_1[n%6]
        out = net(data_use)
        # print(out, expect_out)
        loss = criteria(out, expect_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #     sch.step(loss)
        loss_all += float(loss)
        print(n, loss, loss_all / (n+1), n % 20)

if __name__ == '__main__':
    tic = time.time()
    method_2()
    print((time.time() - tic) * 1000)