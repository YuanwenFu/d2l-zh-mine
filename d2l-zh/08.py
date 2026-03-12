import torch 
import os 
import time 
import sys 
import random 
from d2l import torch as d2l
import numpy as np
from torch.utils import data
from torch import nn

def func1():
    """
    线性回归+基础优化算法
    """
    #带权重的层数为1，单层神经网络
    #线性回归基础知识
    return 

def func2():
    """
    优化方法基础知识
    """
    #选择学习率
    #太大,loss震荡,不收敛
    #太小,收敛速度慢

    #小批量随机梯度下降
    #有两个超参数:批量大小和学习率

    #回顾自动求导知识点
    # torch.randn() 生成标准正态分布（均值0，标准差1）的随机数
    # dtype=torch.float32 指定浮点类型
    # requires_grad=True 启用自动求导
    x = torch.randn(size=(4,), dtype=torch.float32, requires_grad=True)
    #x = torch.arange(4, requires_grad=True, dtype=torch.float32)
    print(f"x = {x}")
    y = 2 * torch.dot(x, x)
    y.backward()
    print(x.grad)
    print(f"x.grad == 4 * x = {x.grad == 4 * x}")

    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(f"x.grad = {x.grad}")

    return 

def func3():
    """
    线性回归的从零开始实现
    """
    def synthetic_data(w, b, num_examples):
        """
        生成y = Xw + b + 噪声
        """
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b 
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))
    
    true_w = torch.tensor([2, -3.4]) #权重真值
    true_b = 4.2 #偏置真值
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.savefig('data.png')

    def data_iter(batch_size, features, labels):
        """
        小批量随机梯度下降
        """
        num_examples = len(features)
        indices = list(range(num_examples))
        #这些样本是随机读取的，没有特定的顺序
        random.shuffle(indices) 
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]
    
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, "\n", y)
        break 

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    def linreg(X, w, b):
        """
        线性回归模型
        """
        return torch.matmul(X, w) + b
    
    def squared_loss(y_hat, y):
        """
        均方损失
        """
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    
    def sgd(params, lr, batch_size):
        """
        小批量随机梯度下降
        """
        #参数更新
        #with表示python的上下文管理器语法
        with torch.no_grad(): #禁用梯度更新
            for param in params:
                param -= lr * param.grad / batch_size 
                param.grad.zero_()
        
    lr = 0.01 #学习率
    num_epochs = 3 
    net = linreg #函数别名
    loss = squared_loss #函数别名

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            #X和y是小批量
            true_batch_size = len(y)
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, true_batch_size) 
        with torch.no_grad(): #no_grad()表示禁用梯度更新
            #features和labels是整个数据集
            train_1 = loss(net(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_1.mean()):f}")
    print(f"w的估计误差: {true_w - w.reshape(true_w.shape)}")
    print(f"b的估计误差: {true_b - b}")

    return 

def func4():
    """
    线性回归的简洁实现
    """
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2 
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    def load_array(data_arrays, batch_size, is_train=True):
        """
        构造一个PyTorch数据迭代器
        """
        dataset = data.TensorDataset(*data_arrays) #*表示解包，说明data_arrays是一个元组
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    X1, y1 = next(iter(data_iter))
    print(f"X1 shape: {X1.shape}, X1 : {X1}")
    print(f"y1 shape: {y1.shape}, y1 : {y1}")

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    print("*"*30)
    print(f"{type(net[0].weight)}, {type(net[0].weight.data)}")
    print("*"*30)
    print(net[0].bias)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03) #stochastic gradiant descent

    num_epochs = 3 
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y) #计算小批量损失
            trainer.zero_grad() #清零梯度
            l.backward() #计算梯度
            trainer.step() #更新参数
        l = loss(net(features), labels) #计算整个数据集损失
        print(f"epoch {epoch + 1}, loss {float(l.mean()):.8f}")

    return 

def func5():
    """
    QA
    """
    #colab 
    
    return

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    #func3()
    #func4()
    func5()
    print("end...")