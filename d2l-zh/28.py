import os 
import sys 
import time 
import torch 
from torch import nn 
import torchvision 
import my_d2l 
import pdb 

def func1():
    """
    batch norm，批量归一化 
    """
    #批量归一化层：可学习的参数为gamma和beta 
    #作用在激活函数之前 

    #批量归一化固定小批量中的均值和方差，然后学习出适合的偏移和缩放 
    #可以加速收敛速度，但一般不改变模型精度 

    return 

def func2():
    """
    代码 
    """
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), my_d2l.BatchNorm(6, num_dims=4), nn.Sigmoid(), 
        nn.AvgPool2d(kernel_size=2, stride=2), 
        nn.Conv2d(6, 16, kernel_size=5), my_d2l.BatchNorm(16, num_dims=4), nn.Sigmoid(), 
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(), 
        nn.Linear(16 * 4 * 4, 120), my_d2l.BatchNorm(120, num_dims=2), nn.Sigmoid(), 
        nn.Linear(120, 84), my_d2l.BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10) 
    )

    X = torch.rand(size=(2, 1, 28, 28)) 
    for layer in net:
        X = layer(X) 
        print(layer.__class__.__name__, "output shape:\t", X.shape) 

    #pdb.set_trace()
    lr, num_epochs, batch_size = 1.0, 10, 256 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size) 
    t1 = time.time() 
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 
    t2 = time.time() 
    print(f"used time: {t2 - t1:.2f} seconds!") 

    print(net[1].gamma.data.reshape((-1,)), net[1].beta.data.reshape((-1,)))

    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(), 
        nn.AvgPool2d(kernel_size=2, stride=2), 
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(), 
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(), 
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(), 
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(), 
        nn.Linear(84, 10)
    )
    t1 = time.time() 
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 
    t2 = time.time() 
    print(f"used time: {t2 - t1:.2f} seconds!") 

    return 

def func3():
    """
    QA 
    """
    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2()  
    func3() 
    print("end...") 