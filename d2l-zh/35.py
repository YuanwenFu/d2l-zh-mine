import os 
import time 
import sys 
import torch 
from torch import nn 
import torchvision 

def func1():
    """
    分布式训练 
    """
    #计算和通讯可以并行 
    #可能存在的问题：你的计算被收发挡住了 
    #使用一个大数据集，从数据多样性的角度考虑这个问题。
    #注意机器-机器带宽 
    #batch_size中的same sample会影响训练效果 
    
    return 

def func2():
    """
    QA 
    """
    return 

if __name__ == "__main__":
    print("start...") 
    func1()
    print("end...") 