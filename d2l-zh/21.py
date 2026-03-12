import os 
import sys 
import time 
import torch 
from torch import nn 
import my_d2l 

def func1():
    """
    多输入多输出通道 
    """
    #卷积神经网络中另一个比较重要的参数：通道数 
    #1*1卷积层，它不识别空间模式，只是融合通道 
    #输出通道数是卷积层的超参数 
    return 

def func2():
    """
    代码实现 
    """
    X = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    ) 
    K = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0]],
         [[1.0, 2.0], [3.0, 4.0]]]
    )
    Y = my_d2l.corr2d_multi_in(X, K) 
    print(f"Y = {Y}") 

    K = torch.stack((K, K + 1, K + 2), 0) 
    print(K.shape) 
    Y = my_d2l.corr2d_multi_in_out(X, K) 
    print(f"Y = {Y}") 

    X = torch.normal(0, 1, (3, 3, 3)) 
    K = torch.normal(0, 1, (2, 3, 1, 1)) 
    Y1 = my_d2l.corr2d_multi_in_out_1x1(X, K) 
    Y2 = my_d2l.corr2d_multi_in_out(X, K) 
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
    return 

def func3():
    """
    QA  
    """
    #卷积层的核就是我们需要学习的参数 
    
    return 

if __name__ == "__main__":
    print("start...") 
    #func1() 
    #func2() 
    func3() 
    print("end...") 

