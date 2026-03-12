import os 
import sys 
import time 
import torch 
from torch import nn 
import torchvision 
import my_d2l 

def func1():
    """
    ResNet
    """
    #卷积层，BatchNorm层，激活层
    #这三个层的位置可以调整
    #对应着不同的残差块 

    #残差块使得很深的网络更加容易训练。
    return

def func2():
    """
    代码 
    """
    blk = my_d2l.Residual(3, 3) 
    X = torch.rand(4, 3, 6, 6) 
    Y = blk(X) 
    print(Y.shape) 

    blk = my_d2l.Residual(3, 6, use_1x1conv=True, strides=2) #增加通道数的同时，高和宽减半 
    Y = blk(X)
    print(Y.shape)

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), 
        nn.BatchNorm2d(64), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
    ) 
    b2 = nn.Sequential(*my_d2l.resnet_block(64, 64, 2, first_block=True)) 
    b3 = nn.Sequential(*my_d2l.resnet_block(64, 128, 2)) 
    b4 = nn.Sequential(*my_d2l.resnet_block(128, 256, 2)) 
    b5 = nn.Sequential(*my_d2l.resnet_block(256, 512, 2)) 
    net = nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten(), 
        nn.Linear(512, 10) 
    ) 
    X = torch.rand(size=(1, 1, 224, 224)) 
    for layer in net:
        X = layer(X) 
        print(layer.__class__.__name__, "output shape:\t", X.shape) 
    
    lr, num_epochs, batch_size = 0.05, 10, 256 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size, resize=96) 
    t1 = time.time() 
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu())   
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
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