import os 
import sys 
import time 
import torch 
from torch import nn 
import torchvision 
from torch.nn import functional as F 
import my_d2l 

def func1():
    """
    GoogLeNet 
    """
    #含并行连结的网络 
    #1*1卷积是对不同通道的结果做加权 
    #inception块融合了4路输出 
    
    #高宽减半就是一个stage
    #总共5个stage
    #总共9个inception块 
    #其中一个inception块中有6个卷积层。 

    return 

def func2():
    """
    代码 
    """
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(), 
        nn.Conv2d(64, 192, kernel_size=3, padding=1), 
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
    )
    b3 = nn.Sequential(
        my_d2l.Inception(192, 64, (96, 128), (16, 32), 32), 
        my_d2l.Inception(256, 128, (128, 192), (32, 96), 64), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b4 = nn.Sequential(
        my_d2l.Inception(480, 192, (96, 208), (16, 48), 64), 
        my_d2l.Inception(512, 160, (112, 224), (24, 64), 64), 
        my_d2l.Inception(512, 128, (128, 256), (24, 64), 64),
        my_d2l.Inception(512, 112, (144, 288), (32, 64), 64), 
        my_d2l.Inception(528, 256, (160, 320), (32, 128), 128), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ) 
    b5 = nn.Sequential(
        my_d2l.Inception(832, 256, (160, 320), (32, 128), 128), 
        my_d2l.Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten()
    )
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10)) 
    X = torch.randn(size=(1, 1, 96, 96)) 
    for layer in net:
        X = layer(X) 
        print(layer.__class__.__name__, "output shape:\t", X.shape) 
    
    lr, num_epochs, batch_size = 0.01, 20, 128 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size, resize=96) 
    t1 = time.time()
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds") 

    return 

def func3():
    """
    QA 
    """
    #1*1卷积层来降低通道数，且计算量可接受 
    return 

if __name__ == "__main__":
    print("start...")
    #func1() 
    #func2() 
    func3() 
    print("end...") 