import os 
import torch 
import sys 
import time 
from torch import nn 
from d2l import torch as d2l 
from IPython import display 

def func1():
    """
    感知机
    """
    #max(0, -y<w,x>)，max0表示if语句那部分逻辑 
    #感知机不能拟合XOR函数，它只能产生线性分割面
    #Minsky & Papert, 1969 
    #感知机的求解算法，等价于batch_size=1的随机梯度下降算法

    return 


def func2():
    """
    多层感知机 
    """
    #隐藏层大小是超参数 
    #sigmoid函数 
    #多层感知机就是在softmax回归基础上，增加了隐藏层 
    
    #多层感知机使用隐藏层和激活函数来得到非线性模型 
    #常用激活函数是Sigmoid,Tanh,ReLU 
    #使用Softmax来处理多分类 
    #超参数为隐藏层数，和各个隐藏层大小 


    return 

def accuracy(y_hat, y):
    """
    计算预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y 
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = Accumulator(2) #正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """
    在n个变量上累加
    """
    def __init__(self, n):
        self.data = [0.0] * n 
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Animator:
    """
    在动画中绘制数据
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g--', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        #增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        #使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts 

    def add(self, x, y):
        """
        向图表中添加多个数据点
        """ 
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n 
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes() 
        display.display(self.fig)
        display.clear_output(wait=True)

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3) 
    for X, y in train_iter:
        y_hat = net(X) 
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            # loss 可能是 per-sample（reduction='none'），需变为标量再 backward
            # 必须用 mean 而非 sum：sum 会使梯度放大 batch_size 倍，导致有效 lr 过大、无法收敛
            l_scalar = l.mean() if l.numel() > 1 else l
            l_scalar.backward()
            updater.step()
            total_loss = l.sum().detach() if l.numel() > 1 else l.detach() * len(y)
            metric.add(float(total_loss), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics 
    print(f"train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

def func3():
    """
    多层感知机的从零开始实现 
    """
    batch_size = 256 
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 500 
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)) 
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True)) 
    params = [W1, b1, W2, b2] 

    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    
    def net(X):
        X = X.reshape((-1, num_inputs)) #-1表示自动维数推导
        H = relu(X @ W1 + b1) #@表示矩阵乘法，它是python中的标准定义 
        return (H @ W2 + b2) 

    loss = nn.CrossEntropyLoss() 
    num_epochs, lr = 10, 0.1 
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    return 

def func4():
    """
    多层感知机的简洁实现 
    """
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    def init_weights(m):
        """
        初始化权重
        """
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    batch_size, lr, num_epochs = 256, 0.1, 10 
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr) 
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  
    return 

def func5():
    """
    QA  
    """
    
    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    #func3() 
    #func4()
    func5()
    print("end...")