import sys 
import time 
import os 
import torch 
from torch import nn 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    LeNet 
    """
    #总共6层：卷积层，池化层，卷积层，池化层，全连接层，全连接层
    return 

def func2():
    """
    代码实现
    """
    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(-1, 1, 28, 28) 
    
    net = nn.Sequential(
        Reshape(),
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), #1表示输入通道数，6表示输出通道数 
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), #6表示输入通道数，16表示输出通道数 
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), 
        nn.Linear(120, 84), nn.Sigmoid(), 
        nn.Linear(84, 10)
    )

    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32) 
    for layer in net:
        X = layer(X) 
        print(f"{layer.__class__.__name__}, output shape: \t{X.shape}") 
    
    batch_size = 256 
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size) 
    lr, num_epochs = 0.9, 10 
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 

    return 

def func3():
    """
    QA 
    """
    #时序是可以用卷积的，一维卷积！
    #中间层的大小与数据复杂度有关。

    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2()
    func3()
    print("end...") 