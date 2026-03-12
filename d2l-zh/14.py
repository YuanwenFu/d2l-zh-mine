import os 
import torch 
import sys 
import time 

def func1():
    """
    数值稳定性 
    """
    #数值稳定性的常见问题:梯度爆炸，比如1.5^100；梯度消失，比如0.8^100 
    #梯度爆炸的问题：值超出值域；对学习率敏感。

    #当数值过大或者过小时，会导致数值问题。
    #这常发生在深度模型中，因为它会对n个数累乘 

    return 

def func2():
    """
    模型初始化和激活函数 
    """
    #Xavier初始化：权重初始化的方差通过输入维度和输出维度共同决定 

    return 

def func3():
    """
    QA
    """
    #sigmoid激活函数容易引起梯度消失,ReLU激活函数可以缓解这个问题 
    #权重是每个batch就会更新，epoch是完整地扫过一遍数据了，它进行了很多次的权重更新了 

    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    func3()
    print("end...") 