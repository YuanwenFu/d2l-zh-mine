import sys 
import time
import os 
import torch 
from torch import nn 
import d2l 
import my_d2l 

def func1():
    """
    AlexNet
    """
    #AlexNet本质是更大更深的LeNet
    #它主要的改进有3点：dropout, RelU而非sigmoid, 最大池化而非平均池化 

    #通过CNN来学习特征 

    return 

def func2():
    """
    代码
    """
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(), 
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5), 
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5), 
        nn.Linear(4096, 10)
    )

    X = torch.rand(1, 1, 224, 224) 
    for layer in net:
        X = layer(X) 
        print(layer.__class__.__name__, "output shape:\t", X.shape) 
    
    batch_size = 128 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size, resize=224) 
    lr, num_epochs = 0.01, 10 
    my_d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, my_d2l.try_gpu()) 

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


