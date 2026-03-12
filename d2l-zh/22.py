import os 
import sys 
import time 
import torch 
from torch import nn 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    池化层 
    """
    #二维最大池化：返回滑动窗口中的最大值 
    #二维平均池化：返回滑动窗口中的平均值 
    #池化层的作用：缓解卷积层对位置的敏感性 

    return 

def func2():
    """
    代码实现 
    """
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]) 
    Y = my_d2l.pool2d(X, (2, 2)) 
    print(f"Y = {Y}") 
    Y = my_d2l.pool2d(X, (2, 2), "avg") 
    print(f"Y = {Y}") 

    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4)) 
    print(X) 

    pool2d = nn.MaxPool2d(3) 
    Y = pool2d(X)
    print(Y) 

    pool2d = nn.MaxPool2d(3, padding=1, stride=2) 
    print(pool2d(X)) 

    pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3)) 
    print(pool2d(X)) 

    X = torch.cat((X, X + 1), 1) #0表示batch, 1表示通道
    print(X) 
    pool2d = nn.MaxPool2d(3, padding=1, stride=2) 
    print(pool2d(X)) 
    
    return 

def func3():
    """
    QA 
    """
    #数据增强淡化了池化层的作用。
    
    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2()  
    func3()
    print("end...") 

