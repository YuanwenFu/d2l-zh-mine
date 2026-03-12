import sys 
import time 
import os 
import torch 
from torch import nn 
import my_d2l 

def func1():
    """
    从全连接到卷积 
    """ 
    #对全连接层使用平移不变性和局部性，就可以得到卷积层 

    return 

def func2():
    """
    卷积层 
    """
    #卷积核 kernel 
    #输入: a * b
    #核: c * d 
    #输出: (a - c + 1) * (b - d + 1) 
    #卷积核和bias是可学习的参数 
    #卷积不会随着输入的维度增加而急剧增加 
    return 

def func3():
    """
    代码 
    """
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]) 
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) 
    out = my_d2l.coor2d(X, K) 
    print(out) 

    X = torch.ones((6, 8)) 
    X[:, 2:6] = 0 
    print(f"X = {X}") 
    K = torch.tensor([[1.0, -1.0]]) 
    Y = my_d2l.coor2d(X, K)
    print(f"Y = {Y}") 
    Y1 = my_d2l.coor2d(X.t(), K)
    print(f"Y1 = {Y1}") 

    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False) #batch_size, channel 
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7)) 
    for i in range(10):
        Y_hat = conv2d(X) 
        l = (Y_hat - Y) ** 2 
        conv2d.zero_grad() 
        l.sum().backward() 
        conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad 
        if (i + 1) % 2 == 0:
            print(f"batch {i + 1}, loss {l.sum():.3f}") 
    print(conv2d.weight.data)
    print(conv2d.weight.data.reshape((1,2)))
    return 

def func4():
    """
    QA 
    """
    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2() 
    #func3() 
    func4()
    print("end...") 