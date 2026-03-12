import os 
import sys 
import time 
import torch 
from torch import nn 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    丢弃法
    """
    #dropout用来解决模型过拟合问题 
    #正则项只在训练中使用：他们影响模型参数的更新 
    #在推理过程中，丢弃法直接返回输入h=dropout(h),这样也能保证确定性的输出 
    #丢弃概率是控制模型复杂度的超参数 

    return 

def func2():
    """
    代码实现 
    """
    def dropout_layer(X, dropout):
        """
        dropout表示丢弃的概率 
        """
        assert 0 <= dropout <= 1 
        if dropout == 1:
            return torch.zeros_like(X) 
        if dropout == 0:
            return X 
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout) #*表示对应元素相乘
    
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8)) 
    print(X) 
    print(dropout_layer(X, 0.)) 
    print(dropout_layer(X, 0.5)) 
    print(dropout_layer(X, 1.)) 

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256 
    dropout1, dropout2 = 0.0, 0.0 
    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
            super(Net, self).__init__()  # 父类会设置 self.training，由 train()/eval() 控制
            self.num_inputs = num_inputs 
            self.lin1 = nn.Linear(num_inputs, num_hiddens1) 
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2) 
            self.lin3 = nn.Linear(num_hiddens2, num_outputs) 
            self.relu = nn.ReLU() 
        
        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs))) 
            #只有在训练模型时，才使用dropout 
            if self.training:
                #在第一个全连接层之后添加一个dropout层 
                H1 = dropout_layer(H1, dropout1) 
            H2 = self.relu(self.lin2(H1)) 
            if self.training:
                #在第二个全连接层之后添加一个dropout层
                H2 = dropout_layer(H2, dropout2) 
            out = self.lin3(H2) 
            return out 
    
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2) 
    num_epochs, lr, batch_size = 10, 0.5, 256 
    loss = nn.CrossEntropyLoss(reduction='none') 
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
    trainer = torch.optim.SGD(net.parameters(), lr=lr) 
    my_d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) 

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        #在第一个全连接层之后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        #在第二个全连接层之后添加一个dropout层 
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10)) 
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    my_d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    
    return 

def func3():
    """
    QA
    """
    #dropout可以简单理解成正则项 
    #dropout是作用在全连接层的上的，BN是作用在卷积层上的 
    
    #正则项唯一的作用是让你在更新你的模型权重时，让你的模型复杂度变低一些 
    #dropout部分为0，而剩余部分需要除以(1-dropout)，从而保证期望不变 
    #dropout唯一改变的是隐藏层的输出 

    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    func3()
    print("end...")

