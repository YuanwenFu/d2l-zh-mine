import sys 
import time 
import os 
import torch 
from torch import nn 
import my_d2l 

def func1():
    """
    填充和步骤 
    """
    #输入维数: d1 * d2
    #卷积核维数: p1 * p2 
    #输出维数: (d1 - p1 + 1) * (d2 - p2 + 1) 
    #由此可见，通过卷积之后，输出维数变小了 

    #填充p行和p列 
    #输出维数: (d1 - p1 + 1 + p) * (d2 - p2 + 1 + p) 

    #步幅s1和s2
    #输出维数：((d1 - p1 + s1 + p) / s1) * ((d2 - p2 + s2 + p) / s2) ,向下取整 
    return 

def func2():
    """
    代码实现 
    """
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) 
    X = torch.rand(size=(8, 8)) 
    Y = my_d2l.comp_conv2d(conv2d, X)
    print(Y) 

    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1)) 
    Y = my_d2l.comp_conv2d(conv2d, X)
    print(Y.shape)  

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2) 
    Y = my_d2l.comp_conv2d(conv2d, X)
    print(Y.shape) 

    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4)) 
    Y = my_d2l.comp_conv2d(conv2d, X)
    print(Y.shape) 
    return 

def func3():
    """
    QA 
    """
    #通常来说将padding设置为(kernel-1)/2，使得输出维度与输入维度相同 
    #通常我们将步幅设置为1。设置成大于1是因为计算量太大了、
    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2() 
    func3() 
    print("end...") 