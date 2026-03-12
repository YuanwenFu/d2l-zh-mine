import os 
import sys 
import time 
import torch 
import torchvision 
from torch import nn 
import my_d2l 

def func1():
    """
    VGG 
    """ 
    #更大更深的AlexNet (重复的VGG块) 
    #使用重复的VGG块来构成深度神经网络。

    return 

def func2():
    """
    代码
    """
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) 

    def vgg(conv_arch):
        conv_blks = [] 
        in_channels = 1
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(my_d2l.wgg_block(num_convs, in_channels, out_channels)) 
            in_channels = out_channels 
        
        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(4096, 10)
        )
    
    net = vgg(conv_arch) #VGG-11,其中8层卷积网络，3层全连接层网络。
    X = torch.randn(size=(1, 1, 224, 224)) 
    for blk in net:
        X = blk(X) 
        print(blk.__class__.__name__, "output shape:\t", X.shape) 

    ratio = 4 
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch] 
    net = vgg(small_conv_arch) 
    lr, num_epochs, batch_size = 0.05, 10, 128 
    train_iter, test_iter = my_d2l.load_data_fashion_mnist(batch_size, resize=224) 
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