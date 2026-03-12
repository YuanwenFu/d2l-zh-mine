import os 
import sys 
import time 
import torch 
import torchvision 
from torch import nn 
import my_d2l 

def func1():
    """
    NIN
    """
    #NiN块：一个卷积层+2个1*1的卷积层 
    #它有比较少的参数，因此不那么容易过拟合。
    return 

def func2():
    """
    代码 
    """
    net = nn.Sequential(
        my_d2l.nin_block(1, 96, kernel_size=11, strides=4, padding=0), 
        nn.MaxPool2d(kernel_size=3, stride=2), 
        my_d2l.nin_block(96, 256, kernel_size=5, strides=1, padding=2), 
        nn.MaxPool2d(kernel_size=3, stride=2), 
        my_d2l.nin_block(256, 384, kernel_size=3, strides=1, padding=1), 
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.Dropout(0.5),
        my_d2l.nin_block(384, 10, kernel_size=3, strides=1, padding=1), 
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ) 
    X = torch.rand(size=(1, 1, 224, 224)) 
    for layer in net:
        X = layer(X) 
        print(layer.__class__.__name__, "output shape:\t", X.shape) 

    lr, num_epochs, batch_size = 0.01, 10, 128 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size, resize=224) 
    t1 = time.time()
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds") 
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