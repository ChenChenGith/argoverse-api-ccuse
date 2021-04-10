import matplotlib;
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class plt_ani(object):
    def __init__(self, data):
        self.data = data
        self.use()

    def use(self):
        x = []
        y = []
        # plt.ion()
        fig, ax = plt.subplots()
        ln, = ax.plot(x, y)
        fig.show()
        for i in self.data:  #
            x.append(i[0])
            y.append(i[1])
            ln.set_data(x, y)
            ax.relim()  # recompute the data limits
            ax.autoscale_view()  # automatic axis scaling
            plt.pause(0.01)
        plt.ioff()
        plt.show()




def plt_Anima(data):
    fig, ax = plt.subplots()          #生成轴和fig,  可迭代的对象
    x, y= [], []    #用于接受后更新的数据
    line, = plt.plot([], [], '.-')   #绘制线对象，plot返回值类型，要加逗号

    #------说明--------#
    #核心函数包含两个：
    #一个是用于初始化画布的函数init()
    #另一个是用于更新数据做动态显示的update()


    def init():
        #初始化函数用于绘制一块干净的画布，为后续绘图做准备
        # ax.set_xlim(0, 12)    #初始函数，设置绘图范围
        # ax.set_ylim(0, 12)
        return line

    def update(step):           #通过帧数来不断更新新的数值
        ax.relim()  # recompute the data limits
        ax.autoscale_view()  # automatic axis scaling
        x.append(step[0])
        y.append(step[1])    #计算y
        line.set_data(x, y)
        return line

    #fig 是绘图的画布
    #update 为更新绘图的函数，step数值是从frames 传入
    #frames 数值是用于动画每一帧的数据  np.linspace(0, 13*np.pi, 128)
    ani = animation.FuncAnimation(fig, update, frames=data,
                        init_func=init,interval=200)

    plt.ioff()
    #plt.show()
    return ani