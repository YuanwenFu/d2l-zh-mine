import os 
import time 
import sys 
import torch
from torch import nn  
import my_d2l 

def func1():
    """
    使用GPU 
    """
    print(torch.device("cpu")) 
    print(torch.cuda.device("cuda")) 
    print(torch.cuda.device("cuda:0"))
    print(torch.cuda.device_count()) 

    print("*"*60)
    print(my_d2l.try_gpu())
    print(my_d2l.try_gpu(10))
    print(my_d2l.try_all_gpus())

    #查询张量所在的设备
    x = torch.tensor([1, 2, 3])
    print(x.device) 

    #创建时指定GPU 
    X = torch.ones(2, 3, device=my_d2l.try_gpu()) 
    print(X) 

    Y = torch.rand(2, 3, device=my_d2l.try_gpu(0))
    print(Y) 

    Z = X.cuda(0) #把X复制到GPU(0)中
    A = torch.randn(2, 3) 
    B = Y + Z 
    print(f"B = {B}") 

    #神经网络和GPU 
    net = nn.Sequential(nn.Linear(3, 1)) 
    net = net.to(my_d2l.try_gpu()) 
    print(net(X)) 
    print(net[0].weight.data)
    print(net[0].weight.grad) 
    print(net[0].weight.data.device)
    return 

def func2():
    """
    购买GPU 
    """
    return 

def func3():
    """
    QA 
    """
    #显存越大越好嘛？ 
    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2() 
    func3() 
    print("end...")